"""Compare two fixed feature bundles using registered development only."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
from threadpoolctl import threadpool_limits

from quant_metric_research import (
    FEATURE_BUNDLES,
    BenchmarkConfig,
    BenchmarkRun,
    ExperimentRegistry,
    compare_feature_bundles,
    preflight_benchmark,
    run_stage3_benchmark,
    write_benchmark_run,
    write_feature_bundle_comparison,
)
from quant_metric_research.benchmark_data import validate_benchmark_panel
from quant_metric_research.intake import _json_object, _local_file, _local_path
from quant_metric_research.io import read_table

BUNDLES = ("legacy10_v1", "legacy10_plus_price3_v1")
FAMILIES = ("ridge", "hist_gradient_boosting")
RETURN = "forward_excess_return"
EXPLICIT_SETTINGS = {
    "feature_columns",
    "split",
    "target_column",
    "realized_return_column",
    "min_cross_section",
    "minimum_coverage",
    "redundancy_threshold",
    "quantiles",
    "hac_lags",
    "ridge_alphas",
    "hist_learning_rates",
    "hist_max_leaf_nodes",
    "hist_l2_regularization",
    "hist_max_iter",
    "hist_min_samples_leaf",
    "include_pca_model",
    "rank_features",
    "random_seed",
    "primary_baseline",
    "model_families",
}


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _configs(payload: bytes) -> dict[str, BenchmarkConfig]:
    settings = _json_object(payload)
    missing = sorted(EXPLICIT_SETTINGS - set(settings))
    if missing:
        raise ValueError(f"Declare explicit benchmark settings: {', '.join(missing)}")
    config = BenchmarkConfig.from_mapping(settings)
    if (
        config.feature_columns != FEATURE_BUNDLES[BUNDLES[0]].feature_columns
        or config.model_families != FAMILIES
        or config.include_pca_model
        or config.primary_baseline != "equal_weight_rank"
        or config.target_column != RETURN
        or config.realized_return_column != RETURN
    ):
        raise ValueError(
            "Declare legacy10_v1, Ridge and histogram boosting, no PCA, "
            "equal_weight_rank, and forward_excess_return for both return fields."
        )
    return {
        name: replace(config, feature_columns=FEATURE_BUNDLES[name].feature_columns)
        for name in BUNDLES
    }


def _without_feature_coverage(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_feature_coverage(item)
            for key, item in value.items()
            if key != "feature_coverage"
        }
    if isinstance(value, list):
        return [_without_feature_coverage(item) for item in value]
    return value


def _check_preflights(preflights: dict[str, Any]) -> None:
    if not all(report["feasible"] for report in preflights.values()):
        raise ValueError("Feature ablation preflight is infeasible.")
    # Feature coverage and its warnings legitimately differ. Compare temporal
    # boundaries, nested folds, label maxima and outcome counts before fitting.
    evidence = [
        _without_feature_coverage(
            {"summary": report["summary"], "folds": report["folds"]}
        )
        for report in preflights.values()
    ]
    if evidence[0] != evidence[1]:
        raise ValueError(
            "Feature ablation preflight schedules or label boundaries differ."
        )


def _snapshot(
    destination: Path, sources: tuple[Path, Path], panel: pd.DataFrame
) -> dict[str, str]:
    panel_file, config_file = sources
    paths = (
        destination / f"input{''.join(panel_file.suffixes)}",
        destination / "input_config.json",
        destination / "panel.parquet",
    )
    shutil.copyfile(panel_file, paths[0])
    shutil.copyfile(config_file, paths[1])
    panel.to_parquet(paths[2], index=False)
    return {path.name: _digest(path) for path in paths}


def _fit_arm(
    name: str,
    panel: pd.DataFrame,
    config: BenchmarkConfig,
    registry: ExperimentRegistry,
    destination: Path,
    study_id: str,
    hypothesis: str,
) -> tuple[BenchmarkRun, dict[str, Any]]:
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        result = run_stage3_benchmark(
            panel,
            config=config,
            registry=registry,
            evaluate_lockbox=False,
            study_id=study_id,
            hypothesis=hypothesis,
        )
    timing = {
        "arm": name,
        "run_id": result.manifest["experiment"]["run_id"],
        "elapsed_seconds": time.perf_counter() - started,
        "timing_scope": "both_model_families_including_nested_tuning",
    }
    # Keep completed fitting evidence if later artifact publication fails.
    _write_json(destination / f"{name}_timing.json", timing)
    write_benchmark_run(result, destination / name)
    return result, timing


def run_feature_ablation(
    *,
    panel_path: str | Path,
    config_path: str | Path,
    registry_path: str | Path,
    output_dir: str | Path,
    study_id: str,
    hypothesis: str,
) -> dict[str, Any]:
    """Run fixed ten/thirteen-feature comparisons on one supplied enriched panel.

    Requires explicit benchmark settings, an existing nonempty local registry
    outside a new output directory, and both bundles' columns. Invalid inputs
    fail before fitting; later failures retain partial files and research history.
    """
    sources = (_local_file(panel_path), _local_file(config_path))
    registry_file = _local_file(registry_path)
    requested = _local_path(output_dir)
    if requested.exists() or requested.is_symlink():
        raise FileExistsError(
            "Output directory already exists; choose a fresh directory."
        )
    destination = _local_path(requested.resolve())
    if registry_file == destination or destination in registry_file.parents:
        raise ValueError("Keep the existing registry outside the output directory.")
    source_hashes = tuple(_digest(path) for path in sources)
    configs = _configs(sources[1].read_bytes())
    # Normalize all candidate inputs once, including numeric dtypes, so the
    # full-panel identity remains identical across the two feature subsets.
    panel = validate_benchmark_panel(
        read_table(sources[0]),
        feature_columns=FEATURE_BUNDLES[BUNDLES[1]].feature_columns,
        target_column=RETURN,
    ).frame
    preflights = {
        name: preflight_benchmark(panel, config=config)
        for name, config in configs.items()
    }
    _check_preflights(preflights)
    registry = ExperimentRegistry(registry_file)
    if not registry.list_runs():
        raise ValueError("Use the existing registry containing prior research history.")
    destination.mkdir(parents=True, exist_ok=False)
    try:
        fingerprints = _snapshot(destination, sources, panel)
        copied_hashes = (
            fingerprints[f"input{''.join(sources[0].suffixes)}"],
            fingerprints["input_config.json"],
        )
        if copied_hashes != source_hashes:
            raise ValueError("Supplied panel or config changed while preparing inputs.")
        _write_json(
            destination / "declaration.json",
            {
                "study_id": study_id,
                "hypothesis": hypothesis,
                "evaluate_lockbox": False,
                "arms": {name: config.to_mapping() for name, config in configs.items()},
                "arm_order": list(BUNDLES),
                "comparison_families": list(FAMILIES),
                "training_threads": 1,
                "input_sha256": fingerprints,
                "example_sha256": _digest(Path(__file__)),
                "registry_path": str(registry_file),
                "evaluation_target": RETURN,
            },
        )
        _write_json(destination / "preflight.json", preflights)
        runs, timings = {}, []
        for name, config in configs.items():
            result, timing = _fit_arm(
                name, panel, config, registry, destination, study_id, hypothesis
            )
            runs[name] = result
            timings.append(timing)
        comparisons = {}
        for family in FAMILIES:
            comparison = compare_feature_bundles(
                runs[BUNDLES[0]],
                runs[BUNDLES[1]],
                model_family=family,
            )
            directory = f"comparison_{family}"
            write_feature_bundle_comparison(comparison, destination / directory)
            comparisons[family] = directory
        if any(
            _digest(destination / name) != digest
            for name, digest in fingerprints.items()
        ):
            raise ValueError("Input snapshot changed during the experiment.")
        report = {
            "status": "complete",
            "claim_scope": "descriptive_development_feature_ablation",
            "evaluate_lockbox": False,
            "runs": timings,
            "comparisons": comparisons,
            "raw_spread_units": "fractional_forward_excess_return",
        }
        _write_json(destination / "completion.json", report)
        return report
    except BaseException as error:
        try:
            _write_json(
                destination / "failure.json", {"error_type": type(error).__name__}
            )
        except OSError as recording_error:
            error.add_note(
                f"Failure record unavailable: {type(recording_error).__name__}."
            )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("panel", "config", "registry", "output-dir", "study-id", "hypothesis"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    try:
        report = run_feature_ablation(
            panel_path=args.panel,
            config_path=args.config,
            registry_path=args.registry,
            output_dir=args.output_dir,
            study_id=args.study_id,
            hypothesis=args.hypothesis,
        )
    except (ValueError, OSError) as error:
        parser.exit(2, f"{type(error).__name__}: {error}\n")
    print(json.dumps(report, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
