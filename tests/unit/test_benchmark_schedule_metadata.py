from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pandas as pd
import pytest

from quant_metric_research.benchmark_config import BenchmarkConfig, NestedSplitConfig
from quant_metric_research.benchmark_data import (
    BenchmarkPanel,
    DevelopmentFold,
    LockedFinalTest,
    Stage3DataPlan,
)
from quant_metric_research.benchmark_io import _json_value
from quant_metric_research.benchmark_reporting import _fingerprints
from quant_metric_research.benchmark_schedules import (
    evaluation_schedule_metadata,
    phase_schedule,
)
from quant_metric_research.scheduled_inference import normalize_expected_dates


def _config():
    return BenchmarkConfig(
        feature_columns=("signal",),
        target_column="target",
        realized_return_column="target",
        min_cross_section=2,
        quantiles=2,
        split=NestedSplitConfig(1, 2, 1, 2, 1, 1, 1),
        hac_lags=1,
    )


def _fold(start, end, number):
    return DevelopmentFold(
        number,
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-01-01"),
        start,
        end,
        (),
        (),
        (),
        (),
        (),
    )


def _plan(interior_date="2025-01-07"):
    dates = pd.to_datetime(
        [
            "2025-01-02",
            "2025-01-03",
            "2025-01-06",
            interior_date,
            "2025-01-09",
            "2025-01-10",
            "2025-01-13",
        ]
    )
    frame = (
        pd.DataFrame(
            [
                {
                    "row_id": f"{i}-{symbol}",
                    "as_of_date": date,
                    "symbol": symbol,
                    "label_end_date": date + pd.Timedelta(days=1),
                    "signal": float(i),
                    "target": None if i == 3 else float(i) / 100,
                    "dataset_version": "synthetic-schedule-v1",
                }
                for i, date in enumerate(dates)
                for symbol in ("A", "B")
            ]
        )
        .iloc[::-1]
        .reset_index(drop=True)
    )
    panel = BenchmarkPanel(
        ("signal",), "target", "as_of_date", "symbol", "label_end_date", frame
    )
    locked = LockedFinalTest(dates[5], dates[5], (), (), (), (), (), ())
    return Stage3DataPlan(
        panel, locked, (_fold(dates[1], dates[2], 1), _fold(dates[4], dates[4], 2))
    )


def _old_phase_schedule(plan, phase):
    """Frozen version0.11 extraction, independent of the moved helper alias."""
    if phase == "development":
        start = min(fold.evaluation_start_date for fold in plan.development_folds)
        end = max(fold.evaluation_end_date for fold in plan.development_folds)
    elif phase == "locked_test":
        start, end = plan.locked_test.test_start_date, plan.locked_test.test_end_date
    else:
        raise ValueError("Unknown evaluation phase.")
    dates = plan.panel.frame["as_of_date"]
    return normalize_expected_dates(
        dates.loc[dates.between(start, end)].drop_duplicates().sort_values()
    )


@pytest.mark.parametrize("phase", ["development", "locked_test"])
def test_shared_derivation_exactly_preserves_old_phase_semantics_and_inputs(phase):
    plan = _plan()
    before = plan.panel.frame
    result = phase_schedule(plan, phase=phase)
    pd.testing.assert_index_equal(result, _old_phase_schedule(plan, phase))
    pd.testing.assert_frame_equal(plan.panel.frame, before)
    assert result.is_monotonic_increasing and result.is_unique


def test_metadata_records_full_irregular_panel_schedule_not_observed_outcomes():
    plan = _plan()
    metadata = evaluation_schedule_metadata(plan, evaluate_lockbox=False)
    assert metadata == {
        "definition_version": "1",
        "source": "validated_panel_within_planned_phase_bounds",
        "lag_unit": "scheduled_observations",
        "dates_by_phase": {
            "development": ("2025-01-03", "2025-01-06", "2025-01-07", "2025-01-09")
        },
    }
    # Jan7 has no outcomes and lies between folds; Jan8 is absent from the panel.
    assert "2025-01-07" in metadata["dates_by_phase"]["development"]
    assert "2025-01-08" not in metadata["dates_by_phase"]["development"]


def test_full_mode_adds_only_locked_schedule_to_unchanged_development_dates():
    plan = _plan()
    development = evaluation_schedule_metadata(plan, evaluate_lockbox=False)
    full = evaluation_schedule_metadata(plan, evaluate_lockbox=True)
    assert list(development["dates_by_phase"]) == ["development"]
    assert list(full["dates_by_phase"]) == ["development", "locked_test"]
    assert (
        full["dates_by_phase"]["development"]
        == (development["dates_by_phase"]["development"])
    )
    assert full["dates_by_phase"]["locked_test"] == ("2025-01-10",)


@pytest.mark.parametrize("bad", ["false", "true", 1, 0, None, (), []])
def test_metadata_requires_an_explicit_boolean_execution_mode(bad):
    with pytest.raises(ValueError, match="evaluate_lockbox.*bool"):
        evaluation_schedule_metadata(_plan(), evaluate_lockbox=bad)


def test_deep_immutable_metadata_contains_no_mutable_date_sequences():
    metadata = evaluation_schedule_metadata(_plan(), evaluate_lockbox=True)
    assert isinstance(metadata, MappingProxyType)
    assert isinstance(metadata["dates_by_phase"], MappingProxyType)
    assert isinstance(metadata["dates_by_phase"]["development"], tuple)
    with pytest.raises(TypeError):
        metadata["definition_version"] = "changed"
    with pytest.raises(TypeError):
        metadata["dates_by_phase"]["development"] = ()
    with pytest.raises(TypeError):
        metadata["dates_by_phase"]["development"][0] = "2020-01-01"


def test_same_counts_and_endpoints_do_not_hide_different_interior_dates():
    first = evaluation_schedule_metadata(_plan(), evaluate_lockbox=False)
    second = evaluation_schedule_metadata(_plan("2025-01-08"), evaluate_lockbox=False)
    left, right = [item["dates_by_phase"]["development"] for item in (first, second)]
    assert len(left) == len(right) and left[0] == right[0] and left[-1] == right[-1]
    assert first != second


@pytest.mark.parametrize("phase", ["unknown", "", None])
def test_unknown_phase_keeps_old_clear_failure(phase):
    with pytest.raises(ValueError, match="phase"):
        phase_schedule(_plan(), phase=phase)


def test_empty_development_folds_keep_old_invalid_plan_failure():
    plan = replace(_plan(), development_folds=())
    with pytest.raises(ValueError):
        phase_schedule(plan, phase="development")
    with pytest.raises(ValueError):
        evaluation_schedule_metadata(plan, evaluate_lockbox=False)


def test_empty_planned_interval_is_rejected_not_filled_with_generated_dates():
    plan = _plan()
    missing = pd.Timestamp("2025-01-08")
    altered = replace(
        plan,
        locked_test=replace(
            plan.locked_test, test_start_date=missing, test_end_date=missing
        ),
    )
    with pytest.raises(ValueError, match="empty"):
        phase_schedule(altered, phase="locked_test")
    # Development does not need to inspect the unavailable locked sequence.
    assert list(
        evaluation_schedule_metadata(altered, evaluate_lockbox=False)["dates_by_phase"]
    ) == ["development"]
    with pytest.raises(ValueError, match="empty"):
        evaluation_schedule_metadata(altered, evaluate_lockbox=True)


@pytest.mark.parametrize("full", [False, True])
def test_manifest_appends_schema6_metadata_with_unchanged_identity_shapes(full):
    plan, config = _plan(), _config()
    metadata = _fingerprints(plan, config=config, evaluate_lockbox=full)
    assert metadata["artifact_schema_version"] == "6"
    assert metadata["evaluation_schedule"] == evaluation_schedule_metadata(
        plan, evaluate_lockbox=full
    )
    assert isinstance(metadata["evaluation_schedule"], MappingProxyType)
    assert metadata["fingerprint_scope"] == (
        "source_code+configuration+validated_panel_contract+execution_mode"
    )
    assert metadata["configuration"] == config.to_mapping()
    assert all(
        len(metadata[key]) == 64
        for key in (
            "run_fingerprint",
            "source_fingerprint",
            "panel_fingerprint",
            "model_input_fingerprint",
        )
    )


def test_json_roundtrip_preserves_exact_iso_dates_and_deterministic_metadata():
    metadata = evaluation_schedule_metadata(_plan(), evaluate_lockbox=True)
    converted = _json_value(metadata, location="evaluation_schedule")
    encoded = json.dumps(converted, sort_keys=True, allow_nan=False)
    decoded = json.loads(encoded)
    assert decoded == converted
    assert decoded["dates_by_phase"]["development"] == list(
        metadata["dates_by_phase"]["development"]
    )
    assert encoded == json.dumps(
        _json_value(
            evaluation_schedule_metadata(_plan(), evaluate_lockbox=True),
            location="evaluation_schedule",
        ),
        sort_keys=True,
        allow_nan=False,
    )


def test_schedule_module_is_in_existing_whole_package_source_fingerprint(monkeypatch):
    plan, config = _plan(), _config()
    original = _fingerprints(plan, config=config, evaluate_lockbox=False)
    original_read = Path.read_bytes
    inspected = []

    def changed_read(path):
        data = original_read(path)
        if path.name == "benchmark_schedules.py":
            inspected.append(path.name)
            return data + b"\n# synthetic source-identity probe\n"
        return data

    monkeypatch.setattr(Path, "read_bytes", changed_read)
    changed = _fingerprints(plan, config=config, evaluate_lockbox=False)
    assert inspected == ["benchmark_schedules.py"]
    assert changed["source_fingerprint"] != original["source_fingerprint"]
    assert changed["run_fingerprint"] != original["run_fingerprint"]
    for key in ("panel_fingerprint", "model_input_fingerprint", "evaluation_schedule"):
        assert changed[key] == original[key]
