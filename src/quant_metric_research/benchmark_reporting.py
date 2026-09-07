from __future__ import annotations

import json
import platform
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn

from ._version import __version__
from .benchmark_config import BenchmarkConfig
from .benchmark_data import Stage3DataPlan
from .benchmark_schedules import evaluation_schedule_metadata

IMPLEMENTATION_VERSION = __version__
ARTIFACT_SCHEMA_VERSION = "6"


def _fingerprints(
    plan: Stage3DataPlan,
    *,
    config: BenchmarkConfig,
    evaluate_lockbox: bool = True,
) -> MappingProxyType[str, Any]:
    frame = plan.panel.frame
    relevant = [
        "row_id",
        "as_of_date",
        "symbol",
        "label_end_date",
        *config.feature_columns,
        config.target_column,
    ]
    if config.realized_return_column not in relevant:
        relevant.append(config.realized_return_column)
    model_frame = frame.loc[:, relevant]
    model_hashes = pd.util.hash_pandas_object(model_frame, index=False)
    model_input_fingerprint = sha256(model_hashes.to_numpy().tobytes()).hexdigest()
    contract_columns = sorted(str(column) for column in frame.columns)
    contract_frame = frame.loc[:, contract_columns].sort_values("row_id", kind="stable")
    panel_hasher = sha256()
    for column in contract_columns:
        panel_hasher.update(column.encode("utf-8"))
        panel_hasher.update(b"\0")
        panel_hasher.update(str(contract_frame[column].dtype).encode("utf-8"))
        panel_hasher.update(b"\0")
    contract_hashes = pd.util.hash_pandas_object(contract_frame, index=False)
    panel_hasher.update(contract_hashes.to_numpy().tobytes())
    panel_fingerprint = panel_hasher.hexdigest()
    source_hasher = sha256()
    package_directory = Path(__file__).resolve().parent
    for source_path in sorted(package_directory.glob("*.py")):
        source_hasher.update(source_path.name.encode("utf-8"))
        source_hasher.update(b"\0")
        source_hasher.update(source_path.read_bytes())
        source_hasher.update(b"\0")
    source_fingerprint = source_hasher.hexdigest()
    config_json = json.dumps(
        config.to_mapping(), sort_keys=True, separators=(",", ":"), default=str
    )
    run_fingerprint = sha256(
        (
            f"{IMPLEMENTATION_VERSION}|{source_fingerprint}|"
            f"{panel_fingerprint}|{config_json}|{evaluate_lockbox}"
        ).encode()
    ).hexdigest()
    dataset_versions = (
        sorted(str(value) for value in frame["dataset_version"].dropna().unique())
        if "dataset_version" in frame.columns
        else []
    )
    return MappingProxyType(
        {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "execution_mode": "full" if evaluate_lockbox else "development",
            "evaluation_schedule": evaluation_schedule_metadata(
                plan, evaluate_lockbox=evaluate_lockbox
            ),
            "implementation_version": IMPLEMENTATION_VERSION,
            "package_version": __version__,
            "source_fingerprint": source_fingerprint,
            "fingerprint_scope": (
                "source_code+configuration+validated_panel_contract+execution_mode"
            ),
            "run_fingerprint": run_fingerprint,
            "panel_fingerprint": panel_fingerprint,
            "model_input_fingerprint": model_input_fingerprint,
            "dataset_versions": dataset_versions,
            "configuration": config.to_mapping(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scipy_version": scipy.__version__,
            "scikit_learn_version": sklearn.__version__,
            "development_start": str(frame["as_of_date"].min().date()),
            "locked_test_start": str(plan.locked_test.test_start_date.date()),
            "locked_test_end": str(plan.locked_test.test_end_date.date()),
            "locked_label_end_max": str(
                frame.loc[
                    frame["as_of_date"].between(
                        plan.locked_test.test_start_date, plan.locked_test.test_end_date
                    ),
                    "label_end_date",
                ]
                .max()
                .date()
            ),
        }
    )


def _data_gate(
    plan: Stage3DataPlan, config: BenchmarkConfig
) -> MappingProxyType[str, Any]:
    panel = plan.panel.frame
    locked = panel.loc[
        (panel["as_of_date"] >= plan.locked_test.test_start_date)
        & (panel["as_of_date"] <= plan.locked_test.test_end_date)
    ].copy(deep=True)
    target_coverage = (
        locked.groupby("as_of_date", sort=True)[config.target_column]
        .apply(lambda values: float(values.notna().mean()))
        .to_dict()
    )
    realized_coverage = (
        locked.groupby("as_of_date", sort=True)[config.realized_return_column]
        .apply(lambda values: float(values.notna().mean()))
        .to_dict()
    )
    cross_section_counts = locked.groupby("as_of_date", sort=True).size().to_dict()
    evaluable_counts = (
        locked.assign(
            _evaluable=(
                locked[config.target_column].notna()
                & locked[config.realized_return_column].notna()
            )
        )
        .groupby("as_of_date", sort=True)["_evaluable"]
        .sum()
        .to_dict()
    )
    return MappingProxyType(
        {
            "structural_contract_passed": True,
            "locked_test_date_count": int(locked["as_of_date"].nunique()),
            "locked_target_coverage_by_date": {
                str(pd.Timestamp(date).date()): coverage
                for date, coverage in target_coverage.items()
            },
            "locked_realized_return_coverage_by_date": {
                str(pd.Timestamp(date).date()): coverage
                for date, coverage in realized_coverage.items()
            },
            "locked_cross_section_count_by_date": {
                str(pd.Timestamp(date).date()): int(count)
                for date, count in cross_section_counts.items()
            },
            "locked_evaluable_count_by_date": {
                str(pd.Timestamp(date).date()): int(count)
                for date, count in evaluable_counts.items()
            },
            "point_in_time_provider_verified": False,
            "stable_identifier_policy_verified": False,
            "corporate_action_policy_verified": False,
            "delisting_return_policy_verified": False,
            "claim_scope": "benchmark_engine_only",
        }
    )
