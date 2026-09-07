from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from quant_metric_research import benchmark, cli, experiment_workflow
from quant_metric_research.benchmark_config import BenchmarkConfig, NestedSplitConfig
from quant_metric_research.experiment_registry import ExperimentRegistry


@pytest.fixture()
def experiment_input():
    dates = pd.bdate_range("2024-01-02", periods=32)
    panel = pd.DataFrame(
        [
            {
                "as_of_date": date,
                "symbol": f"S{stock}",
                "feature_available_at": date,
                "label_end_date": date + pd.offsets.BDay(2),
                "signal": float(stock),
                "other": float((stock * 3 + day) % 7),
                "forward_excess_return": stock * 0.02 + ((stock * 3 + day) % 7) * 0.003,
            }
            for day, date in enumerate(dates)
            for stock in range(8)
        ]
    )
    config = BenchmarkConfig(
        feature_columns=("signal", "other"),
        split=NestedSplitConfig(3, 1, 3, 8, 1, 2, 4),
        min_cross_section=4,
        quantiles=2,
        hac_lags=1,
        model_families=("ridge",),
        ridge_alphas=(1.0,),
    )
    return panel, config


def _develop(panel, config, registry):
    return benchmark.run_stage3_benchmark(
        panel,
        config=config,
        registry=registry,
        study_id="study-1",
        hypothesis="Test a combined ranking on synthetic inputs.",
    )


def test_unregistered_final_is_rejected_before_evaluation(
    experiment_input, monkeypatch
):
    panel, config = experiment_input
    monkeypatch.setattr(
        benchmark,
        "_locked_run",
        lambda *args, **kwargs: pytest.fail(
            "Unregistered final evaluation was called."
        ),
    )
    with pytest.raises(ValueError, match="registry"):
        benchmark.run_stage3_benchmark(panel, config=config, evaluate_lockbox=True)


def test_registered_development_final_and_reuse_refusal(experiment_input, tmp_path):
    panel, config = experiment_input
    registry = ExperimentRegistry(tmp_path / "experiments.sqlite3")
    development = _develop(panel, config, registry)
    development_id = development.manifest["experiment"]["run_id"]
    assert registry.get_run(development_id)["status"] == "completed"
    assert development.manifest["experiment"]["registered"] is True
    final = benchmark.run_stage3_benchmark(
        panel,
        config=config,
        evaluate_lockbox=True,
        registry=registry,
        development_run_id=development_id,
    )
    final_id = final.manifest["experiment"]["run_id"]
    assert registry.get_run(final_id)["status"] == "completed"
    assert final.acceptance["lockbox_reuse_registry_enforced"] is True
    assert final.acceptance["stage4_eligible"] is False
    assert final.manifest["locked_label_end_max"] > final.manifest["locked_test_end"]
    with pytest.raises(ValueError):
        benchmark.run_stage3_benchmark(
            panel,
            config=config,
            evaluate_lockbox=True,
            registry=registry,
            development_run_id=development_id,
        )
    assert len(registry.list_runs()) == 2


def test_bad_input_is_retained_as_a_failed_development(experiment_input, tmp_path):
    panel, config = experiment_input
    registry = ExperimentRegistry(tmp_path / "experiments.sqlite3")
    with pytest.raises(ValueError):
        _develop(panel.drop(columns="signal"), config, registry)
    records = registry.list_runs()
    assert len(records) == 1
    assert records[0]["status"] == "failed"
    assert records[0]["error_type"] == "ValueError"


def test_training_failure_keeps_declared_exposure(
    experiment_input, tmp_path, monkeypatch
):
    panel, config = experiment_input
    registry = ExperimentRegistry(tmp_path / "experiments.sqlite3")

    def fail(*args, **kwargs):
        assert registry.list_runs()[0]["manifest"]["locked_test_start"]
        raise RuntimeError("Synthetic training failure.")

    monkeypatch.setattr(benchmark, "_development_run", fail)
    with pytest.raises(RuntimeError, match="Synthetic training"):
        _develop(panel, config, registry)
    assert registry.list_runs()[0]["status"] == "failed"


def test_crash_after_reservation_cannot_reopen_lockbox(
    experiment_input,
    tmp_path,
    monkeypatch,
):
    panel, config = experiment_input
    path = tmp_path / "experiments.sqlite3"
    registry = ExperimentRegistry(path)
    reference = _develop(panel, config, registry).manifest["experiment"]["run_id"]

    def interrupt(*args, **kwargs):
        # A distinct connection must see the already committed reservation.
        final_rows = [
            row
            for row in ExperimentRegistry(path).list_runs()
            if row["kind"] == "final"
        ]
        assert len(final_rows) == 1 and final_rows[0]["status"] == "running"
        raise KeyboardInterrupt()

    monkeypatch.setattr(benchmark, "_locked_run", interrupt)
    with pytest.raises(KeyboardInterrupt):
        benchmark.run_stage3_benchmark(
            panel,
            config=config,
            evaluate_lockbox=True,
            registry=registry,
            development_run_id=reference,
        )
    assert any(row["status"] == "failed" for row in registry.list_runs())
    with pytest.raises(ValueError):
        benchmark.run_stage3_benchmark(
            panel,
            config=config,
            evaluate_lockbox=True,
            registry=registry,
            development_run_id=reference,
        )


def test_changed_configuration_cannot_open_frozen_test(
    experiment_input,
    tmp_path,
    monkeypatch,
):
    panel, config = experiment_input
    registry = ExperimentRegistry(tmp_path / "experiments.sqlite3")
    reference = _develop(panel, config, registry).manifest["experiment"]["run_id"]
    monkeypatch.setattr(
        benchmark,
        "_development_run",
        lambda *args, **kwargs: pytest.fail("Stale evidence reached development."),
    )
    monkeypatch.setattr(
        benchmark,
        "_locked_run",
        lambda *args, **kwargs: pytest.fail("Stale evidence reached final evaluation."),
    )
    with pytest.raises(ValueError):
        benchmark.run_stage3_benchmark(
            panel,
            config=replace(config, ridge_alphas=(2.0,)),
            evaluate_lockbox=True,
            registry=registry,
            development_run_id=reference,
        )
    assert len(registry.list_runs()) == 1


def test_untracked_development_remains_explicit(experiment_input):
    panel, config = experiment_input
    result = benchmark.run_stage3_benchmark(panel, config=config)
    assert result.manifest["experiment"]["registered"] is False
    assert set(result.predictions["phase"]) == {"development"}


@pytest.mark.parametrize("evaluate_lockbox", [False, True])
def test_identity_drift_during_computation_cannot_complete(
    experiment_input, tmp_path, monkeypatch, evaluate_lockbox
):
    panel, config = experiment_input
    registry = ExperimentRegistry(tmp_path / "experiments.sqlite3")
    reference = (
        _develop(panel, config, registry).manifest["experiment"]["run_id"]
        if evaluate_lockbox
        else None
    )
    original = experiment_workflow._fingerprints
    calls = 0

    def drifting_identity(*args, **kwargs):
        nonlocal calls
        calls += 1
        manifest = original(*args, **kwargs)
        return (
            manifest
            if calls == 1
            else {**manifest, "source_fingerprint": "changed-during-run"}
        )

    monkeypatch.setattr(experiment_workflow, "_fingerprints", drifting_identity)
    monkeypatch.setattr(benchmark, "_fingerprints", drifting_identity)
    with pytest.raises(ValueError, match="identity differs"):
        if evaluate_lockbox:
            benchmark.run_stage3_benchmark(
                panel,
                config=config,
                evaluate_lockbox=True,
                registry=registry,
                development_run_id=reference,
            )
        else:
            _develop(panel, config, registry)
    failed = [row for row in registry.list_runs() if row["status"] == "failed"]
    assert len(failed) == 1
    assert failed[0]["exposure_start"] and failed[0]["exposure_end"]
    assert failed[0]["kind"] == ("final" if evaluate_lockbox else "development")
    if evaluate_lockbox:
        monkeypatch.setattr(experiment_workflow, "_fingerprints", original)
        monkeypatch.setattr(benchmark, "_fingerprints", original)
        with pytest.raises(ValueError, match="previously exposed"):
            benchmark.run_stage3_benchmark(
                panel,
                config=config,
                evaluate_lockbox=True,
                registry=registry,
                development_run_id=reference,
            )


def test_artifact_failure_does_not_release_completed_final(
    experiment_input, tmp_path, monkeypatch
):
    panel, config = experiment_input
    registry_path = tmp_path / "experiments.sqlite3"
    registry = ExperimentRegistry(registry_path)
    reference = _develop(panel, config, registry).manifest["experiment"]["run_id"]
    monkeypatch.setattr(cli, "read_table", lambda path: panel.copy(deep=True))
    monkeypatch.setattr(cli, "read_json_object", lambda path: config.to_mapping())

    def fail_publication(*args, **kwargs):
        raise OSError("Synthetic artifact publication failure.")

    monkeypatch.setattr(cli, "write_benchmark_run", fail_publication)
    output = tmp_path / "not-published"
    arguments = [
        "benchmark",
        "--panel",
        "panel.csv",
        "--config",
        "config.json",
        "--output-dir",
        str(output),
        "--registry",
        str(registry_path),
        "--evaluate-lockbox",
        "--development-run-id",
        reference,
    ]
    with pytest.raises(OSError, match="publication failure"):
        cli.main(arguments)
    # Completed means computation finished, not that files were published.
    records = registry.list_runs()
    assert len(records) == 2 and all(row["status"] == "completed" for row in records)
    assert not output.exists()
    with pytest.raises(ValueError, match="previously exposed"):
        cli.main(arguments)
