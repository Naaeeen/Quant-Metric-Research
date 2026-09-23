"""Conditional consistency checks for caller-owned development evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from math import isfinite
from types import MappingProxyType
from typing import Any

import pandas as pd

from ._comparison_predictions import (
    prepare_predictions,
    validate_assignments,
    validate_phase,
)
from .benchmark import BenchmarkRun
from .benchmark_config import BenchmarkConfig
from .contracts import _daily_date
from .feature_bundles import FEATURE_BUNDLES
from .scheduled_inference import normalize_expected_dates

_IDENTITY_KEYS = (
    "source_fingerprint",
    "package_version",
    "implementation_version",
    "python_version",
    "numpy_version",
    "pandas_version",
    "scipy_version",
    "scikit_learn_version",
    "panel_fingerprint",
    "dataset_versions",
    "fingerprint_scope",
    "development_start",
    "locked_test_start",
    "locked_test_end",
    "locked_label_end_max",
)
_BOUNDARY_KEYS = (
    "development_start",
    "locked_test_start",
    "locked_test_end",
    "locked_label_end_max",
)
_BUNDLES = ("legacy10_v1", "legacy10_plus_price3_v1")


@dataclass(frozen=True, slots=True)
class ComparisonInputs:
    config: BenchmarkConfig
    expected_dates: tuple[str, ...]
    predictions: pd.DataFrame
    producer_identity: Mapping[str, Any]
    input_identities: Mapping[str, Any]


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise ValueError("Evidence mappings require nonempty string keys.")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and isfinite(value):
        return value
    raise ValueError("Evidence must contain JSON-safe finite scalar values.")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _canonical(value: Any) -> str:
    return json.dumps(
        _plain(_freeze(value)), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _mapping(value: Any, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _scope(run: BenchmarkRun) -> None:
    if not isinstance(run, BenchmarkRun):
        raise ValueError("Comparison inputs must be BenchmarkRun objects.")
    manifest = _mapping(run.manifest, "manifest")
    acceptance = _mapping(run.acceptance, "acceptance")
    gate = _mapping(run.data_gate, "data_gate")
    if (
        manifest.get("artifact_schema_version") != "6"
        or manifest.get("execution_mode") != "development"
    ):
        raise ValueError("Comparison requires development-only schema 6 manifests.")
    if (
        acceptance.get("acceptance_status") != "not_evaluated"
        or acceptance.get("lockbox_evaluated_once_in_this_run") is not False
        or gate.get("claim_scope") != "development_only"
    ):
        raise ValueError("Comparison scope contradicts development-only evidence.")
    for evidence in (manifest, acceptance, gate):
        for flag in (
            "model_gate_passed",
            "eligible_for_stage4_data_review",
            "stage4_eligible",
            "final_outcomes_evaluated",
            "lockbox_evaluated_once_in_this_run",
            "lockbox_reuse_registry_enforced",
        ):
            if flag in evidence and evidence[flag] is not False:
                raise ValueError(
                    f"Development comparison rejects contradictory {flag}."
                )
    if "experiment" in manifest:
        experiment = _mapping(manifest["experiment"], "experiment")
        if (
            "kind" in experiment and experiment["kind"] != "development"
        ) or experiment.get("development_run_id") is not None:
            raise ValueError("Comparison rejects non-development experiment metadata.")


def _identity(manifest: Mapping) -> Mapping:
    if any(
        key not in manifest
        for key in (*_IDENTITY_KEYS, "run_fingerprint", "model_input_fingerprint")
    ):
        raise ValueError("Producer identity metadata is incomplete.")
    for key in (*_IDENTITY_KEYS, "run_fingerprint", "model_input_fingerprint"):
        if key == "dataset_versions":
            values = manifest[key]
            if not isinstance(values, (tuple, list)) or any(
                not isinstance(value, str) or not value for value in values
            ):
                raise ValueError(
                    "dataset_versions must be a sequence of nonempty strings."
                )
        elif (
            not isinstance(manifest[key], str)
            or not manifest[key]
            or manifest[key] != manifest[key].strip()
        ):
            raise ValueError(f"Invalid producer identity field {key}.")
    for key in (
        "source_fingerprint",
        "panel_fingerprint",
        "run_fingerprint",
        "model_input_fingerprint",
    ):
        value = manifest[key]
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"Invalid {key}.")
    if (
        manifest["fingerprint_scope"]
        != "source_code+configuration+validated_panel_contract+execution_mode"
    ):
        raise ValueError("Unsupported fingerprint scope.")
    boundaries = [
        _daily_date(manifest[key], field=key, nullable=False) for key in _BOUNDARY_KEYS
    ]
    if not (boundaries[0] < boundaries[1] <= boundaries[2] < boundaries[3]):
        raise ValueError("Producer split boundaries are inconsistent.")
    return _freeze({key: manifest[key] for key in _IDENTITY_KEYS})


def _config(manifest: Mapping, bundle: str) -> BenchmarkConfig:
    supplied = _mapping(manifest.get("configuration"), "configuration")
    if set(supplied) != {field.name for field in fields(BenchmarkConfig)}:
        raise ValueError("Stored configuration must explicitly contain every field.")
    raw_features = supplied["feature_columns"]
    if (
        not isinstance(raw_features, (list, tuple))
        or tuple(raw_features) != FEATURE_BUNDLES[bundle].feature_columns
    ):
        raise ValueError(f"Configuration must use literal {bundle} features.")
    try:
        return BenchmarkConfig.from_mapping(_plain(_freeze(supplied)))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Invalid stored benchmark configuration.") from error


def _schedule(manifest: Mapping) -> tuple[str, ...]:
    schedule = _mapping(manifest.get("evaluation_schedule"), "evaluation_schedule")
    if (
        set(schedule) != {"definition_version", "source", "lag_unit", "dates_by_phase"}
        or schedule.get("definition_version") != "1"
        or schedule.get("source") != "validated_panel_within_planned_phase_bounds"
        or schedule.get("lag_unit") != "scheduled_observations"
    ):
        raise ValueError("Unsupported evaluation schedule definition.")
    phases = _mapping(schedule["dates_by_phase"], "dates_by_phase")
    if set(phases) != {"development"}:
        raise ValueError("Schedule must contain only development dates.")
    dates = phases["development"]
    if not isinstance(dates, (tuple, list)) or any(
        not isinstance(value, str) for value in dates
    ):
        raise ValueError("Stored schedule must contain ISO daily strings.")
    normalized = normalize_expected_dates(dates)
    if tuple(dates) != tuple(str(value.date()) for value in normalized):
        raise ValueError("Stored schedule must contain canonical ISO daily strings.")
    return tuple(dates)


def validate_comparison_inputs(
    legacy: BenchmarkRun, candidate: BenchmarkRun, *, model_family: str
) -> ComparisonInputs:
    # Inspect all scope markers before touching either input's caller-owned frames.
    for run in (legacy, candidate):
        _scope(run)
    identities = [_identity(run.manifest) for run in (legacy, candidate)]
    if _canonical(identities[0]) != _canonical(identities[1]):
        raise ValueError("Producer identities must match between comparison inputs.")
    configs = [
        _config(run.manifest, bundle)
        for run, bundle in zip((legacy, candidate), _BUNDLES, strict=True)
    ]
    comparable = [
        {
            key: value
            for key, value in config.to_mapping().items()
            if key != "feature_columns"
        }
        for config in configs
    ]
    if _canonical(comparable[0]) != _canonical(comparable[1]):
        raise ValueError("Configurations may differ only in feature_columns.")
    if (
        not isinstance(model_family, str)
        or not model_family
        or model_family != model_family.strip()
        or any(model_family not in config.model_families for config in configs)
    ):
        raise ValueError("model_family must be explicitly configured in both inputs.")
    schedules = [_schedule(run.manifest) for run in (legacy, candidate)]
    if schedules[0] != schedules[1]:
        raise ValueError("Development schedules must match exactly.")
    for run in (legacy, candidate):
        validate_phase(run.fold_assignments, "assignments")
        validate_phase(run.predictions, "predictions")
    assignments = validate_assignments(
        legacy.fold_assignments,
        candidate.fold_assignments,
        config=configs[0],
        expected_dates=schedules[0],
        identity=identities[0],
    )
    predictions = prepare_predictions(
        legacy.predictions,
        candidate.predictions,
        assignments=assignments,
        configs=tuple(configs),
        expected_dates=schedules[0],
        model_family=model_family,
    )
    return ComparisonInputs(
        configs[0],
        schedules[0],
        predictions,
        identities[0],
        _freeze(
            {
                name: {
                    "bundle_id": bundle,
                    "run_fingerprint": run.manifest["run_fingerprint"],
                    "model_input_fingerprint": run.manifest["model_input_fingerprint"],
                }
                for name, bundle, run in zip(
                    ("legacy", "candidate"), _BUNDLES, (legacy, candidate), strict=True
                )
            }
        ),
    )
