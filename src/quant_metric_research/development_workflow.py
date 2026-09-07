"""Offline raw-input to registered development evidence; never evaluate final data."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ._version import __version__
from .benchmark import run_stage3_benchmark
from .benchmark_config import BenchmarkConfig
from .benchmark_io import BenchmarkArtifacts, write_benchmark_run
from .config import PanelConfig
from .contracts import validate_as_of_dates, validate_memberships, validate_prices
from .experiment_registry import ExperimentRegistry
from .input_audit import audit_inputs
from .intake import _local_path
from .panel import build_point_in_time_panel
from .preflight import preflight_benchmark

_RETURN_COLUMNS = {"forward_return", "forward_excess_return"}


def _write_json(value: Any, path: Path) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded + "\n")


def _fingerprints(paths: list[Path], destination: Path) -> dict[str, str]:
    return {
        path.relative_to(destination).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in paths
    }


def _verify_snapshots(destination: Path, fingerprints: dict[str, str]) -> None:
    current = _fingerprints([destination / name for name in fingerprints], destination)
    if current != fingerprints:
        raise ValueError("Input snapshot changed; workflow completion refused.")


def _validate_request(
    panel_config: PanelConfig,
    benchmark_config: BenchmarkConfig,
    source_evidence: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(panel_config, PanelConfig):
        raise ValueError("panel_config must be a PanelConfig.")
    if not isinstance(benchmark_config, BenchmarkConfig):
        raise ValueError("benchmark_config must be a BenchmarkConfig.")
    if not set(benchmark_config.feature_columns) <= set(panel_config.feature_columns):
        raise ValueError("Benchmark features must be declared panel features.")
    if benchmark_config.target_column not in _RETURN_COLUMNS | {"forward_excess_rank"}:
        raise ValueError("Benchmark target must be a generated forward target.")
    if benchmark_config.realized_return_column not in _RETURN_COLUMNS:
        raise ValueError(
            "Benchmark realized return must be a generated forward return."
        )
    if not isinstance(source_evidence, dict):
        raise ValueError("source_evidence must be a JSON object of supplied claims.")
    return json.loads(json.dumps(source_evidence, allow_nan=False, sort_keys=True))


def _snapshot_inputs(
    destination: Path,
    prices: pd.DataFrame,
    memberships: pd.DataFrame,
    dates: tuple[pd.Timestamp, ...],
    panel_config: PanelConfig,
    benchmark_config: BenchmarkConfig,
    evidence: dict[str, Any],
) -> dict[str, str]:
    prices.to_parquet(destination / "prices.parquet", index=False)
    memberships.to_parquet(destination / "memberships.parquet", index=False)
    pd.DataFrame({"as_of_date": dates}).to_csv(
        destination / "as_of_dates.csv", index=False
    )
    for name, value in (
        ("panel_config.json", asdict(panel_config)),
        ("benchmark_config.json", benchmark_config.to_mapping()),
        ("source_evidence.json", evidence),
    ):
        _write_json(value, destination / name)
    return _fingerprints(
        [
            destination / name
            for name in (
                "prices.parquet",
                "memberships.parquet",
                "as_of_dates.csv",
                "panel_config.json",
                "benchmark_config.json",
                "source_evidence.json",
            )
        ],
        destination,
    )


def _prepare_panel(
    destination: Path,
    prices: pd.DataFrame,
    memberships: pd.DataFrame,
    dates: tuple[pd.Timestamp, ...],
    panel_config: PanelConfig,
    benchmark_config: BenchmarkConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    audit = audit_inputs(prices, memberships, as_of_dates=dates, config=panel_config)
    _write_json(audit, destination / "input_audit.json")
    panel = build_point_in_time_panel(
        prices, memberships, as_of_dates=dates, config=panel_config
    )
    panel.to_parquet(destination / "metric_panel.parquet", index=False)
    panel_snapshot = _fingerprints([destination / "metric_panel.parquet"], destination)
    preflight = preflight_benchmark(panel, config=benchmark_config)
    _write_json(preflight, destination / "preflight.json")
    if not preflight["feasible"]:
        raise ValueError(
            "Development preflight is infeasible; training was not started."
        )
    _verify_snapshots(destination, panel_snapshot)
    return panel, panel_snapshot


def _verified_results(
    artifacts: BenchmarkArtifacts, registry: ExperimentRegistry, run_id: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], pd.DataFrame]:
    for path in artifacts.files.values():
        if not path.is_file():
            raise ValueError("Published development bundle is incomplete.")
    manifest = json.loads(artifacts.files["benchmark_manifest"].read_text("utf-8"))
    acceptance = json.loads(artifacts.files["acceptance"].read_text("utf-8"))
    data_gate = json.loads(artifacts.files["data_gate"].read_text("utf-8"))
    record = registry.get_run(run_id)
    if (
        manifest["execution_mode"] != "development"
        or manifest["experiment"]["run_id"] != run_id
        or manifest["experiment"]["registered"] is not True
        or acceptance["acceptance_status"] != "not_evaluated"
        or acceptance["stage4_eligible"] is not False
        or acceptance["empirical_data_provenance_verified"] is not False
        or acceptance["model_gate_passed"] is not False
        or acceptance["eligible_for_stage4_data_review"] is not False
        or acceptance["lockbox_evaluated_once_in_this_run"] is not False
        or data_gate["claim_scope"] != "development_only"
        or record["kind"] != "development"
        or record["status"] != "completed"
        or record["manifest"]["run_fingerprint"] != manifest["run_fingerprint"]
        or record["frozen_family"] != acceptance["frozen_model_family"]
    ):
        raise ValueError("Published evidence must match completed development only.")
    tables = {}
    for name in (
        "fold_assignments",
        "hyperparameter_trials",
        "screening_by_fold",
        "oos_predictions",
        "daily_metrics",
        "fold_summary",
        "benchmark_summary",
    ):
        path = artifacts.files[name]
        frame = (
            pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        )
        if "phase" not in frame or set(frame["phase"]) != {"development"}:
            raise ValueError("Published tables must contain development phases only.")
        tables[name] = frame
    return manifest, acceptance, record, tables["benchmark_summary"]


def _report(
    destination: Path,
    registry_path: Path,
    panel: pd.DataFrame,
    panel_config: PanelConfig,
    benchmark_config: BenchmarkConfig,
    evidence: dict[str, Any],
    fingerprints: dict[str, str],
    artifacts: BenchmarkArtifacts,
    registry: ExperimentRegistry,
    run_id: str,
) -> dict[str, Any]:
    manifest, acceptance, record, summary = _verified_results(
        artifacts, registry, run_id
    )
    _write_json(record, destination / "registry_record.json")
    selected = summary.loc[
        summary["model"].isin(
            (benchmark_config.primary_baseline, *benchmark_config.model_families)
        )
    ]
    records = selected.astype(object).where(selected.notna(), None).to_dict("records")
    output_paths = [
        destination / name
        for name in (
            "input_audit.json",
            "metric_panel.parquet",
            "preflight.json",
            "registry_record.json",
        )
    ] + list(artifacts.files.values())
    return {
        "schema_version": 1,
        "package_version": __version__,
        "status": "complete",
        "completed_at": datetime.now(UTC).isoformat(),
        "claim_scope": "development_only",
        "network_accessed": False,
        "lockbox_evaluated": False,
        "empirical_alpha_claim": False,
        "empirical_data_provenance_verified": False,
        "stage4_eligible": False,
        "source_evidence_is_supplied_claim": True,
        "source_evidence": evidence,
        "development_run_id": run_id,
        "registry_path": str(registry_path),
        "dataset_version": panel_config.dataset_version,
        "universe_id": panel_config.universe_id,
        "benchmark_symbol": panel_config.benchmark_symbol,
        "cohort_size": int(panel["symbol"].nunique()),
        "panel_row_count": len(panel),
        "decision_date_count": int(panel["as_of_date"].nunique()),
        "model_families": list(benchmark_config.model_families),
        "frozen_model_family": acceptance["frozen_model_family"],
        "primary_baseline": benchmark_config.primary_baseline,
        "acceptance_status": acceptance["acceptance_status"],
        "development_summary": records,
        "artifacts": {
            **{name: name for name in fingerprints},
            **{
                path.relative_to(destination).as_posix(): path.relative_to(
                    destination
                ).as_posix()
                for path in output_paths
            },
        },
        "input_fingerprints": fingerprints,
        "output_fingerprints": _fingerprints(output_paths, destination),
        "panel_fingerprint": manifest["panel_fingerprint"],
        "source_fingerprint": manifest["source_fingerprint"],
        "limitations": (
            "Source evidence is supplied, not independently certified. The panel "
            "contains forward outcomes and preflight inspects their availability; "
            "this is not a sealed holdout. Development statistics are descriptive "
            "research evidence, not final acceptance or tradable performance."
        ),
    }


def run_development_workflow(
    prices: pd.DataFrame,
    memberships: pd.DataFrame,
    *,
    as_of_dates: list[pd.Timestamp] | tuple[pd.Timestamp, ...],
    panel_config: PanelConfig,
    benchmark_config: BenchmarkConfig,
    output_dir: str | Path,
    registry_path: str | Path,
    study_id: str,
    hypothesis: str,
    source_evidence: dict[str, Any],
) -> dict[str, Any]:
    """Snapshot inputs, preflight, and register development without opening final.

    The registry must live outside this new run directory and be retained across
    related studies. Only development_report.json marks workflow completion.
    Partial evidence and registry exposure survive failure; retry in a new output
    directory with the same registry. Inputs and source claims are copied.
    """
    raw_registry = str(registry_path)
    if (
        not isinstance(registry_path, (str, Path))
        or not raw_registry.strip()
        or raw_registry == ":memory:"
        or raw_registry.startswith("file:")
    ):
        raise ValueError("The registry must name a durable local file.")
    output_path = _local_path(output_dir)
    registry_path = _local_path(registry_path)
    destination = _local_path(output_path.resolve())
    registry_file = _local_path(registry_path.resolve())
    if destination.exists() or output_path.is_symlink():
        raise FileExistsError(
            "Output directory already exists; choose a new directory."
        )
    if registry_file == destination or destination in registry_file.parents:
        raise ValueError(
            "The durable registry must be outside the run output directory."
        )
    evidence = _validate_request(panel_config, benchmark_config, source_evidence)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        normalized_prices = validate_prices(prices.copy(deep=True))
        normalized_members = validate_memberships(memberships.copy(deep=True))
        dates = validate_as_of_dates(as_of_dates)
        fingerprints = _snapshot_inputs(
            destination,
            normalized_prices,
            normalized_members,
            dates,
            panel_config,
            benchmark_config,
            evidence,
        )
        panel, panel_snapshot = _prepare_panel(
            destination,
            normalized_prices,
            normalized_members,
            dates,
            panel_config,
            benchmark_config,
        )
        fingerprints = {**fingerprints, **panel_snapshot}
        _verify_snapshots(destination, fingerprints)
        registry = ExperimentRegistry(registry_file)
        result = run_stage3_benchmark(
            panel,
            config=benchmark_config,
            evaluate_lockbox=False,
            registry=registry,
            study_id=study_id,
            hypothesis=hypothesis,
        )
        artifacts = write_benchmark_run(result, destination / "development")
        report = _report(
            destination,
            registry_file,
            panel,
            panel_config,
            benchmark_config,
            evidence,
            fingerprints,
            artifacts,
            registry,
            str(result.manifest["experiment"]["run_id"]),
        )
        _verify_snapshots(destination, fingerprints)
        pending = destination / "development_report.pending.json"
        _write_json(report, pending)
        pending.rename(destination / "development_report.json")
        return report
    except BaseException as error:
        try:
            _write_json(
                {"status": "failed", "error_type": type(error).__name__},
                destination / "development_failure.json",
            )
        except OSError as recording_error:
            error.add_note(
                f"Failure record unavailable ({type(recording_error).__name__})."
            )
        raise
