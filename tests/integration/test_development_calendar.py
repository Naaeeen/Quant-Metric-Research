from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest
from test_development_workflow import request_inputs as request_inputs

from quant_metric_research import ExperimentRegistry
from quant_metric_research import development_workflow as workflow
from quant_metric_research import panel as panel_module
from quant_metric_research.session_calendar import ExpectedSessionCalendar


@pytest.fixture()
def calendar(request_inputs):
    dates = tuple(sorted(request_inputs["prices"]["date"].unique()))
    return ExpectedSessionCalendar(
        sessions=dates,
        coverage_start=dates[0],
        coverage_end=dates[-1],
        source="invented-calendar",
        version="fixture-v1",
    )


def test_declared_calendar_registered_run_retains_history_and_snapshot(
    request_inputs,
    calendar,
):
    first = workflow.run_development_workflow(**request_inputs)
    assert "expected_calendar" not in first
    assert "expected_calendar_fingerprint" not in first
    assert not (request_inputs["output_dir"] / "expected_calendar.json").exists()
    registry = ExperimentRegistry(request_inputs["registry_path"])
    original_record = registry.get_run(first["development_run_id"])
    output = request_inputs["output_dir"].with_name("declared-calendar")
    report = workflow.run_development_workflow(
        **{**request_inputs, "output_dir": output},
        expected_calendar=calendar,
    )
    assert registry.get_run(first["development_run_id"]) == original_record
    records = registry.list_runs()
    assert len(records) == 2
    assert all(row["kind"] == "development" for row in records)
    assert all(row["status"] == "completed" for row in records)
    declaration = output / "expected_calendar.json"
    assert json.loads(declaration.read_text()) == calendar.to_mapping()
    assert report["expected_calendar"] == calendar.to_mapping()
    assert report["expected_calendar_fingerprint"] == calendar.fingerprint
    assert (
        report["input_fingerprints"][declaration.name]
        == hashlib.sha256(declaration.read_bytes()).hexdigest()
    )
    assert report["artifacts"][declaration.name] == declaration.name
    assert report["lockbox_evaluated"] is False
    assert report["empirical_data_provenance_verified"] is False
    audit = json.loads((output / "input_audit.json").read_text())
    assert audit["calendar_check"]["status"] == "matched"
    assert audit["calendar_check"]["fingerprint"] == calendar.fingerprint
    panel = pd.read_parquet(output / "metric_panel.parquet")
    assert set(panel["session_calendar_sha256"]) == {calendar.fingerprint}
    assert set(panel["session_calendar_source"]) == {calendar.source}
    assert set(panel["session_calendar_version"]) == {calendar.version}
    predictions = pd.read_parquet(output / "development/oos_predictions.parquet")
    assert set(predictions["phase"]) == {"development"}
    assert not (output / "final").exists()


def test_calendar_mismatch_retains_previous_registry_without_calculation(
    request_inputs,
    calendar,
    monkeypatch,
):
    workflow.run_development_workflow(**request_inputs)
    registry_path = request_inputs["registry_path"]
    registry = ExperimentRegistry(registry_path)
    previous_records = registry.list_runs()
    previous_bytes = registry_path.read_bytes()
    output = request_inputs["output_dir"].with_name("calendar-mismatch")
    prices = request_inputs["prices"]
    missing = prices.loc[prices["date"] != calendar.sessions[20]].copy()

    def forbidden(*args, **kwargs):
        pytest.fail("Calendar mismatch must fail before features, fitting or registry.")

    monkeypatch.setattr(panel_module, "compute_price_metrics", forbidden)
    monkeypatch.setattr(workflow, "run_stage3_benchmark", forbidden)
    monkeypatch.setattr(workflow, "ExperimentRegistry", forbidden)
    with pytest.raises(ValueError, match="calendar"):
        workflow.run_development_workflow(
            **{**request_inputs, "prices": missing, "output_dir": output},
            expected_calendar=calendar,
        )
    assert registry_path.read_bytes() == previous_bytes
    assert registry.list_runs() == previous_records
    assert (
        json.loads((output / "expected_calendar.json").read_text())
        == calendar.to_mapping()
    )
    audit = json.loads((output / "input_audit.json").read_text())
    assert audit["calendar_check"]["status"] == "mismatch"
    assert (output / "development_failure.json").is_file()
    assert not (output / "metric_panel.parquet").exists()
    assert not (output / "development_report.json").exists()


@pytest.mark.parametrize("phase", ["before_registration", "after_development"])
def test_calendar_snapshot_tamper_refuses_completion(
    request_inputs,
    calendar,
    monkeypatch,
    phase,
):
    output = request_inputs["output_dir"]
    attribute = (
        "preflight_benchmark"
        if phase == "before_registration"
        else "write_benchmark_run"
    )
    original = getattr(workflow, attribute)

    def tamper(*args, **kwargs):
        result = original(*args, **kwargs)
        (output / "expected_calendar.json").write_text('{"changed": true}')
        return result

    monkeypatch.setattr(workflow, attribute, tamper)
    with pytest.raises(ValueError, match="snapshot"):
        workflow.run_development_workflow(**request_inputs, expected_calendar=calendar)
    assert not (output / "development_report.json").exists()
    assert (output / "development_failure.json").is_file()
    if phase == "before_registration":
        assert not request_inputs["registry_path"].exists()
    else:
        records = ExperimentRegistry(request_inputs["registry_path"]).list_runs()
        assert len(records) == 1
        assert records[0]["kind"] == "development"
        assert records[0]["status"] == "completed"


def test_invalid_calendar_type_rejected_before_artifacts(request_inputs):
    with pytest.raises(ValueError, match="ExpectedSessionCalendar"):
        workflow.run_development_workflow(**request_inputs, expected_calendar={})
    assert not request_inputs["output_dir"].exists()
    assert not request_inputs["registry_path"].exists()
