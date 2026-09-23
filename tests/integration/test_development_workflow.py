from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_metric_research import BenchmarkConfig, ExperimentRegistry, PanelConfig
from quant_metric_research import development_workflow as workflow


@pytest.fixture()
def request_inputs(tmp_path):
    random = np.random.default_rng(20260907)
    dates = pd.bdate_range("2022-01-03", periods=75)
    symbols = [f"SAMPLE_{index}" for index in range(8)]
    prices = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": symbol,
                    "adjusted_close": 100
                    * np.exp(
                        np.cumsum(random.normal(index * 0.0003, 0.012, len(dates)))
                    ),
                }
            )
            for index, symbol in enumerate([*symbols, "BENCH"])
        ],
        ignore_index=True,
    )
    memberships = pd.DataFrame(
        {
            "universe_id": "SYNTHETIC_COHORT",
            "symbol": symbols,
            "effective_from": dates[0],
            "effective_to": None,
            "source": "invented-software-fixture",
        }
    )
    features = ("trailing_return", "annualized_volatility", "max_drawdown")
    panel_config = PanelConfig(
        dataset_version="synthetic-workflow-test-v1",
        universe_id="SYNTHETIC_COHORT",
        benchmark_symbol="BENCH",
        lookback_sessions=10,
        min_observations=8,
        target_horizon_sessions=2,
        entry_lag_sessions=1,
        feature_columns=features,
    )
    benchmark_config = BenchmarkConfig.from_mapping(
        {
            "feature_columns": features,
            "model_families": ["ridge"],
            "ridge_alphas": [1.0],
            "min_cross_section": 8,
            "quantiles": 2,
            "hac_lags": 1,
            "split": {
                "final_test_date_count": 6,
                "outer_n_splits": 2,
                "outer_test_date_count": 6,
                "outer_min_train_date_count": 18,
                "inner_n_splits": 1,
                "inner_validation_date_count": 4,
                "inner_min_train_date_count": 8,
            },
        }
    )
    return {
        "prices": prices,
        "memberships": memberships,
        "as_of_dates": tuple(dates[10:65]),
        "panel_config": panel_config,
        "benchmark_config": benchmark_config,
        "output_dir": tmp_path / "development-one",
        "registry_path": tmp_path / "durable.sqlite3",
        "study_id": "workflow-test",
        "hypothesis": "Invented prices exercise the development workflow.",
        "source_evidence": {"synthetic_data": True, "source": "invented fixture"},
    }


def test_real_pipeline_trains_ridge_and_preserves_durable_history(request_inputs):
    prices_before = request_inputs["prices"].copy(deep=True)
    memberships_before = request_inputs["memberships"].copy(deep=True)
    first = workflow.run_development_workflow(**request_inputs)
    output = request_inputs["output_dir"]
    assert first == json.loads((output / "development_report.json").read_text())
    assert first["status"] == "complete"
    assert first["claim_scope"] == "development_only"
    assert first["stage4_eligible"] is False
    assert first["empirical_data_provenance_verified"] is False
    assert first["lockbox_evaluated"] is False
    assert first["source_evidence"] == request_inputs["source_evidence"]
    assert first["source_evidence_is_supplied_claim"] is True
    assert first["cohort_size"] == 8
    assert first["panel_row_count"] == 440
    assert first["decision_date_count"] == 55
    summary = pd.DataFrame(first["development_summary"])
    assert set(summary["model"]) == {"ridge", "equal_weight_rank"}
    assert set(summary["evaluation_scope"]) == {"native", "common"}
    assert summary["mean_rank_ic"].notna().all()
    predictions = pd.read_parquet(output / "development/oos_predictions.parquet")
    assert set(predictions["phase"]) == {"development"}
    assert not (output / "final").exists()
    for name, digest in first["input_fingerprints"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    for name, digest in first["output_fingerprints"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    pd.testing.assert_frame_equal(request_inputs["prices"], prices_before)
    pd.testing.assert_frame_equal(request_inputs["memberships"], memberships_before)
    second = workflow.run_development_workflow(
        **{**request_inputs, "output_dir": output.with_name("development-two")}
    )
    assert first["development_run_id"] != second["development_run_id"]
    records = ExperimentRegistry(request_inputs["registry_path"]).list_runs()
    assert len(records) == 2
    assert {run["kind"] for run in records} == {"development"}
    assert {run["status"] for run in records} == {"completed"}


def test_existing_destination_refused_before_any_work(request_inputs, monkeypatch):
    output = request_inputs["output_dir"]
    output.mkdir()
    marker = output / "existing.txt"
    marker.write_text("preserve")

    def forbidden(*args, **kwargs):
        pytest.fail("Existing destination must fail before audit or registration.")

    monkeypatch.setattr(workflow, "audit_inputs", forbidden)
    monkeypatch.setattr(workflow, "ExperimentRegistry", forbidden)
    with pytest.raises(FileExistsError, match="already exists"):
        workflow.run_development_workflow(**request_inputs)
    assert marker.read_text() == "preserve"
    assert not request_inputs["registry_path"].exists()
    assert not (output / "development_failure.json").exists()


def test_registry_must_not_live_inside_run_directory(request_inputs):
    request = {
        **request_inputs,
        "registry_path": request_inputs["output_dir"] / "registry.sqlite3",
    }
    with pytest.raises(ValueError, match="outside"):
        workflow.run_development_workflow(**request)
    assert not request_inputs["output_dir"].exists()


@pytest.mark.parametrize("registry", ["", ":memory:", "file:temporary.sqlite3"])
def test_registry_requires_durable_file_name(request_inputs, registry, monkeypatch):
    monkeypatch.chdir(request_inputs["output_dir"].parent)
    with pytest.raises(ValueError, match="registry"):
        workflow.run_development_workflow(
            **{**request_inputs, "registry_path": registry}
        )
    assert not request_inputs["output_dir"].exists()


def test_infeasible_preflight_preserved_without_training(request_inputs, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Infeasible preflight must not train or create a registry.")

    monkeypatch.setattr(workflow, "run_stage3_benchmark", forbidden)
    monkeypatch.setattr(workflow, "ExperimentRegistry", forbidden)
    request = {**request_inputs, "as_of_dates": request_inputs["as_of_dates"][:10]}
    with pytest.raises(ValueError, match="preflight"):
        workflow.run_development_workflow(**request)
    output = request_inputs["output_dir"]
    assert json.loads((output / "preflight.json").read_text())["feasible"] is False
    assert (output / "metric_panel.parquet").is_file()
    assert not (output / "development_report.json").exists()
    assert json.loads((output / "development_failure.json").read_text()) == {
        "status": "failed",
        "error_type": "ValueError",
    }
    assert not request_inputs["registry_path"].exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("feature_columns", ("invented_feature",)),
        ("target_column", "external_target"),
        ("realized_return_column", "external_return"),
    ],
)
def test_incompatible_config_rejected_before_artifacts(request_inputs, field, value):
    request = {
        **request_inputs,
        "benchmark_config": replace(
            request_inputs["benchmark_config"], **{field: value}
        ),
    }
    with pytest.raises(ValueError, match="(feature|target|return)"):
        workflow.run_development_workflow(**request)
    assert not request_inputs["output_dir"].exists()


@pytest.mark.parametrize("evidence", [[], {"invalid": float("nan")}])
def test_non_json_evidence_rejected_before_artifacts(request_inputs, evidence):
    with pytest.raises((ValueError, TypeError)):
        workflow.run_development_workflow(
            **{**request_inputs, "source_evidence": evidence}
        )
    assert not request_inputs["output_dir"].exists()


def test_publication_failure_keeps_completed_registry_and_no_success_marker(
    request_inputs,
    monkeypatch,
):
    def fail_publication(*args, **kwargs):
        raise OSError("sensitive local provider path")

    monkeypatch.setattr(workflow, "write_benchmark_run", fail_publication)
    with pytest.raises(OSError):
        workflow.run_development_workflow(**request_inputs)
    output = request_inputs["output_dir"]
    assert not (output / "development_report.json").exists()
    failure = json.loads((output / "development_failure.json").read_text())
    assert failure == {"status": "failed", "error_type": "OSError"}
    records = ExperimentRegistry(request_inputs["registry_path"]).list_runs()
    assert len(records) == 1
    assert records[0]["kind"] == "development"
    assert records[0]["status"] == "completed"


@pytest.mark.parametrize("filename", ["source_evidence.json", "metric_panel.parquet"])
def test_snapshot_change_prevents_success_marker(request_inputs, monkeypatch, filename):
    original = workflow.preflight_benchmark

    def change_snapshot(*args, **kwargs):
        path = request_inputs["output_dir"] / filename
        path.write_text('{"changed": true}')
        return original(*args, **kwargs)

    monkeypatch.setattr(workflow, "preflight_benchmark", change_snapshot)
    with pytest.raises(ValueError, match="snapshot"):
        workflow.run_development_workflow(**request_inputs)
    assert not (request_inputs["output_dir"] / "development_report.json").exists()


def test_published_final_phase_is_rejected(request_inputs, monkeypatch):
    original = workflow.write_benchmark_run

    def contaminate_result(*args, **kwargs):
        result = original(*args, **kwargs)
        path = result.output_dir / "oos_predictions.parquet"
        frame = pd.read_parquet(path).assign(phase="locked_test")
        frame.to_parquet(path, index=False)
        return result

    monkeypatch.setattr(workflow, "write_benchmark_run", contaminate_result)
    with pytest.raises(ValueError, match="development"):
        workflow.run_development_workflow(**request_inputs)
    assert not (request_inputs["output_dir"] / "development_report.json").exists()
