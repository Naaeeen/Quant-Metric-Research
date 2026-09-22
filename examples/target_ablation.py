"""Compare raw-return and within-date rank targets using registered development."""

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
    BenchmarkConfig,
    BenchmarkRun,
    ExperimentRegistry,
    preflight_benchmark,
    run_stage3_benchmark,
    write_benchmark_run,
)
from quant_metric_research.benchmark_data import validate_benchmark_panel
from quant_metric_research.benchmark_metrics import evaluate_prediction_frame
from quant_metric_research.config import DEFAULT_FEATURE_COLUMNS
from quant_metric_research.intake import _json_object, _local_file, _local_path
from quant_metric_research.io import read_table

RAW_TARGET = "forward_excess_return"
RANK_TARGET = "forward_excess_rank"
FAMILIES = ("ridge", "hist_gradient_boosting")
BASELINE = "equal_weight_rank"
KEYS = ["phase", "fold", "as_of_date", "symbol"]
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
SUMMARY_COLUMNS = [
    "phase",
    "model",
    "evaluation_scope",
    "date_count",
    "spread_date_count",
    "mean_rank_ic",
    "median_rank_ic",
    "rank_ic_std",
    "positive_date_rate",
    "mean_spread",
    "mean_score_coverage",
    "mean_rank_ic_coverage",
    "mean_spread_coverage",
]


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _config(payload: bytes) -> BenchmarkConfig:
    settings = _json_object(payload)
    missing = sorted(EXPLICIT_SETTINGS - set(settings))
    if missing:
        raise ValueError(f"Declare explicit benchmark settings: {', '.join(missing)}")
    config = BenchmarkConfig.from_mapping(settings)
    if (
        config.feature_columns != DEFAULT_FEATURE_COLUMNS
        or config.model_families != FAMILIES
        or config.include_pca_model
        or config.primary_baseline != BASELINE
        or config.target_column != RAW_TARGET
        or config.realized_return_column != RAW_TARGET
    ):
        raise ValueError(
            "Use the original ten features, Ridge and histogram boosting, "
            "equal_weight_rank, and forward_excess_return for both return fields."
        )
    return config


def prepare_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Regenerate labels without filtering the contemporaneous feature universe."""
    panel = validate_benchmark_panel(
        frame, feature_columns=DEFAULT_FEATURE_COLUMNS, target_column=RAW_TARGET
    ).frame
    maturity_counts = panel.groupby("as_of_date")["label_end_date"].nunique()
    if (maturity_counts > 1).any():
        raise ValueError(
            "Rank labels require one shared label maturity per decision date."
        )
    # A shared maturity prevents immature peers leaking into retained rank labels.
    # Missing outcomes remain missing; every feature row is still available to score.
    return panel.assign(
        forward_excess_rank=panel.groupby("as_of_date")[RAW_TARGET].rank(
            method="average", pct=True
        )
    )


def _aligned(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame.sort_values(KEYS, kind="stable").loc[:, columns].reset_index(drop=True)


def _require_equal(left: pd.DataFrame, right: pd.DataFrame, description: str) -> None:
    try:
        pd.testing.assert_frame_equal(left, right, check_exact=True)
    except AssertionError as error:
        raise ValueError(f"Target arms differ in {description}.") from error


def comparison_predictions(raw: BenchmarkRun, ranked: BenchmarkRun) -> pd.DataFrame:
    """Align five arms on already-masked raw outcomes, leaving saved runs intact."""
    for field in (
        "source_fingerprint",
        "panel_fingerprint",
        "evaluation_schedule",
        "python_version",
        "numpy_version",
        "pandas_version",
        "scipy_version",
        "scikit_learn_version",
    ):
        if raw.manifest[field] != ranked.manifest[field]:
            raise ValueError(f"Target arms differ in {field}.")
    if any(run.manifest["execution_mode"] != "development" for run in (raw, ranked)):
        raise ValueError("Target comparison requires development results.")
    _require_equal(raw.fold_assignments, ranked.fold_assignments, "fold assignments")
    raw_base = raw.predictions.loc[raw.predictions["model"] == BASELINE]
    rank_base = ranked.predictions.loc[ranked.predictions["model"] == BASELINE]
    evidence = KEYS + [
        "score",
        "realized_return",
        "selected_features",
        "feature_count",
        "selected_feature_count",
        "zero_observed_features",
    ]
    _require_equal(
        _aligned(raw_base, evidence), _aligned(rank_base, evidence), "baseline evidence"
    )
    arms = [raw_base.copy(deep=True)]
    for name, run in (("raw", raw), ("rank", ranked)):
        for family in FAMILIES:
            arm = run.predictions.loc[run.predictions["model"] == family].copy(
                deep=True
            )
            _require_equal(
                _aligned(raw_base, KEYS + ["realized_return"]),
                _aligned(arm, KEYS + ["realized_return"]),
                "scoring rows or masked raw outcomes",
            )
            other = raw.predictions.loc[raw.predictions["model"] == family]
            _require_equal(
                _aligned(other, KEYS + ["selected_features"]),
                _aligned(arm, KEYS + ["selected_features"]),
                "selected features",
            )
            arms.append(arm.assign(model=f"{name}_{family}"))
    combined = pd.concat(arms, ignore_index=True)
    # Never restore outcomes from the original panel near the final boundary.
    return combined.assign(target=combined["realized_return"])


def _save_comparison(
    destination: Path, raw: BenchmarkRun, ranked: BenchmarkRun, config: BenchmarkConfig
) -> None:
    predictions = comparison_predictions(raw, ranked)
    evaluation = evaluate_prediction_frame(
        predictions,
        min_cross_section=config.min_cross_section,
        quantiles=config.quantiles,
        hac_lags=config.hac_lags,
        primary_models=(
            BASELINE,
            *(f"{arm}_{family}" for arm in ("raw", "rank") for family in FAMILIES),
        ),
        expected_dates_by_phase=raw.manifest["evaluation_schedule"]["dates_by_phase"],
    )
    predictions.to_parquet(destination / "comparison_predictions.parquet", index=False)
    evaluation.daily_metrics.to_csv(destination / "daily_metrics.csv", index=False)
    evaluation.fold_metrics.to_csv(destination / "fold_metrics.csv", index=False)
    evaluation.summary.loc[:, SUMMARY_COLUMNS].to_csv(
        destination / "summary.csv", index=False
    )


def _inputs(
    panel_path: str | Path,
    config_path: str | Path,
    registry_path: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path, Path, Path]:
    panel_file, config_file = _local_file(panel_path), _local_file(config_path)
    registry_file = _local_file(registry_path)
    raw_destination = _local_path(output_dir)
    if raw_destination.exists() or raw_destination.is_symlink():
        raise FileExistsError(
            "Output directory already exists; choose a fresh directory."
        )
    destination = _local_path(raw_destination.resolve())
    if registry_file == destination or destination in registry_file.parents:
        raise ValueError("Keep the existing registry outside the output directory.")
    return panel_file, config_file, registry_file, destination


def _snapshot(
    destination: Path, panel_file: Path, config_file: Path, panel: pd.DataFrame
) -> dict[str, str]:
    shutil.copyfile(panel_file, destination / f"input{''.join(panel_file.suffixes)}")
    shutil.copyfile(config_file, destination / "input_config.json")
    panel.to_parquet(destination / "panel.parquet", index=False)
    return {
        path.name: _digest(path) for path in destination.iterdir() if path.is_file()
    }


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
    # Persist timing even if standard bundle publication fails afterward.
    _write_json(destination / f"{name}_timing.json", timing)
    write_benchmark_run(result, destination / name)
    return result, timing


def run_target_ablation(
    *,
    panel_path: str | Path,
    config_path: str | Path,
    registry_path: str | Path,
    output_dir: str | Path,
    study_id: str,
    hypothesis: str,
) -> dict[str, Any]:
    """Run two target-only development experiments in an existing local history."""
    panel_file, config_file, registry_file, destination = _inputs(
        panel_path, config_path, registry_path, output_dir
    )
    source_hashes = (_digest(panel_file), _digest(config_file))
    config = _config(config_file.read_bytes())
    panel = prepare_panel(read_table(panel_file))
    ranked_config = replace(config, target_column=RANK_TARGET)
    configs = {"raw": config, "rank": ranked_config}
    preflights = {
        name: preflight_benchmark(panel, config=value)
        for name, value in configs.items()
    }
    if not all(value["feasible"] for value in preflights.values()):
        raise ValueError(
            "Target ablation preflight is infeasible; revise the declared schedule."
        )
    registry = ExperimentRegistry(registry_file)
    if not registry.list_runs():
        raise ValueError("Use the existing registry containing prior research history.")
    destination.mkdir(parents=True, exist_ok=False)
    try:
        fingerprints = _snapshot(destination, panel_file, config_file, panel)
        copied_hashes = (
            fingerprints[f"input{''.join(panel_file.suffixes)}"],
            fingerprints["input_config.json"],
        )
        if copied_hashes != source_hashes:
            raise ValueError("Supplied panel or config changed while preparing inputs.")
        declaration = {
            "study_id": study_id,
            "hypothesis": hypothesis,
            "evaluate_lockbox": False,
            "feature_bundle": "legacy10_v1",
            "arms": {name: value.to_mapping() for name, value in configs.items()},
            "arm_order": ["raw", "rank"],
            "training_threads": 1,
            "input_sha256": fingerprints,
            "example_sha256": _digest(Path(__file__)),
            "registry_path": str(registry_file),
            "evaluation_target": RAW_TARGET,
            "rank_definition": "within_date_average_percentile_of_observed_raw_returns",
        }
        _write_json(destination / "declaration.json", declaration)
        _write_json(destination / "preflight.json", preflights)
        raw, raw_timing = _fit_arm(
            "raw", panel, config, registry, destination, study_id, hypothesis
        )
        ranked, rank_timing = _fit_arm(
            "rank", panel, ranked_config, registry, destination, study_id, hypothesis
        )
        _save_comparison(destination, raw, ranked, config)
        if any(
            _digest(destination / name) != digest
            for name, digest in fingerprints.items()
        ):
            raise ValueError("Input snapshot changed during the experiment.")
        report = {
            "status": "complete",
            "claim_scope": "descriptive_development_target_ablation",
            "evaluate_lockbox": False,
            "runs": [raw_timing, rank_timing],
            "raw_spread_units": "fractional_forward_excess_return",
            "files": [
                "declaration.json",
                "daily_metrics.csv",
                "fold_metrics.csv",
                "summary.csv",
                "comparison_predictions.parquet",
                "raw",
                "rank",
            ],
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
        report = run_target_ablation(
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
