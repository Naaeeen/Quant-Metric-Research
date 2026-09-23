"""Declared development-only experiment for the pinned public research archive."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .archive_sample import normalize_mendeley_prices
from .benchmark_config import BenchmarkConfig, NestedSplitConfig
from .config import DEFAULT_FEATURE_COLUMNS, PanelConfig
from .development_workflow import run_development_workflow
from .intake import _local_path, _write_json
from .io import read_table
from .public_archive import verify_public_archive

STUDY_ID = "public-archive-ridge-v1"
HYPOTHESIS = (
    "Engineering demonstration: test whether a fixed Ridge combination of trailing "
    "metrics improves development stock ranking over equal-weight oriented ranks. "
    "The retrospective selected cohort cannot establish empirical or tradable alpha."
)


def public_demo_configs() -> tuple[PanelConfig, BenchmarkConfig]:
    """Return fresh configs frozen before the first archive development results."""
    panel = PanelConfig(
        dataset_version="mendeley-ndxfrshm74-v3-sp500-1216-alphabetical30-v1",
        universe_id="MENDELEY_2017_SELECTED_COHORT_DEMO",
        benchmark_symbol="FF_MARKET_PROXY",
    )
    benchmark = BenchmarkConfig(
        feature_columns=DEFAULT_FEATURE_COLUMNS,
        split=NestedSplitConfig(
            final_test_date_count=63,
            outer_n_splits=3,
            outer_test_date_count=63,
            outer_min_train_date_count=252,
            inner_n_splits=2,
            inner_validation_date_count=42,
            inner_min_train_date_count=126,
        ),
        min_cross_section=20,
        quantiles=3,
        hac_lags=19,
        ridge_alphas=(1.0,),
        model_families=("ridge",),
    )
    return panel, benchmark


def _destination_and_registry(
    output_dir: str | Path, registry_path: str | Path
) -> tuple[Path, Path]:
    destination = _local_path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Public demo output already exists; choose a new one.")
    destination = _local_path(destination.resolve())
    registry = _local_path(_local_path(registry_path).resolve())
    if registry == destination or destination in registry.parents:
        raise ValueError("Keep the durable registry outside the public demo directory.")
    return destination, registry


def _prepare_sample(archive_dir: str | Path):
    archive_dir = _local_path(archive_dir)
    archive_manifest = verify_public_archive(archive_dir)
    sample = normalize_mendeley_prices(
        read_table(archive_dir / "raw" / "sp500-1216.csv"),
        read_table(archive_dir / "raw" / "FF3-0317.csv"),
        cohort_size=30,
    )
    # A normalizer must not silently use files changed after acquisition checks.
    if verify_public_archive(archive_dir) != archive_manifest:
        raise ValueError("Archive evidence changed during normalization.")
    return sample, archive_manifest


def run_public_demo(
    *, archive_dir: str | Path, output_dir: str | Path, registry_path: str | Path
) -> dict:
    """Use local files without fetching data or evaluating final outcomes."""
    destination, registry = _destination_and_registry(output_dir, registry_path)
    sample, archive_manifest = _prepare_sample(archive_dir)
    panel_config, benchmark_config = public_demo_configs()
    benchmark_dates = sample.prices.loc[
        sample.prices["symbol"] == panel_config.benchmark_symbol, "date"
    ]
    dates = tuple(
        benchmark_dates.loc[
            benchmark_dates.between("2013-01-02", "2016-12-30")
        ].sort_values()
    )
    memberships = pd.DataFrame(
        {
            "universe_id": panel_config.universe_id,
            "symbol": sample.symbols,
            "effective_from": pd.Timestamp("2012-01-03"),
            "effective_to": pd.Timestamp("2016-12-31"),
            "source": "mendeley-v3-2017-constituents-alphabetical-demo",
        }
    )
    destination.mkdir(parents=True, exist_ok=False)
    try:
        _write_json(sample.profile, destination / "quality_profile.json")
        sample.missing_prices.to_csv(destination / "missing_prices.csv", index=False)
        evidence = {
            "source_kind": "real_public_historical_archive",
            "archive_manifest": archive_manifest,
            "normalization_profile": sample.profile,
            "cohort_rule": "first_30_normalized_stock_columns_alphabetically",
            "final_outcomes_evaluated": False,
            "empirical_data_provenance_verified": False,
            "stage4_eligible": False,
        }
        result = run_development_workflow(
            sample.prices,
            memberships,
            as_of_dates=dates,
            panel_config=panel_config,
            benchmark_config=benchmark_config,
            output_dir=destination / "research",
            registry_path=registry,
            study_id=STUDY_ID,
            hypothesis=HYPOTHESIS,
            source_evidence=evidence,
        )
        report = {
            "schema_version": 1,
            "status": "complete",
            "claim_scope": "public_archive_engineering_demo",
            "network_accessed": False,
            "final_outcomes_evaluated": False,
            "empirical_data_provenance_verified": False,
            "stage4_eligible": False,
            "development_run_id": result["development_run_id"],
            "development_summary": result["development_summary"],
            "development_report": "research/development_report.json",
            "cohort_symbols": list(sample.symbols),
            "file_fingerprints": {
                name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
                for name in (
                    "quality_profile.json",
                    "missing_prices.csv",
                    "research/development_report.json",
                )
            },
        }
        pending = destination / "public_demo_report.pending.json"
        _write_json(report, pending)
        pending.rename(destination / "public_demo_report.json")
        return report
    except Exception as error:
        _write_json(
            {"status": "failed", "error_type": type(error).__name__},
            destination / "public_demo_failure.json",
        )
        raise
