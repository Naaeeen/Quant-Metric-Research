from __future__ import annotations

import json
from types import MappingProxyType

import pandas as pd
import pytest

from quant_metric_research import benchmark, experiment_workflow
from quant_metric_research.benchmark_config import BenchmarkConfig, NestedSplitConfig
from quant_metric_research.benchmark_data import build_stage3_data_plan
from quant_metric_research.benchmark_io import write_benchmark_run
from quant_metric_research.benchmark_metrics import evaluate_prediction_frame
from quant_metric_research.benchmark_reporting import _fingerprints
from quant_metric_research.experiment_registry import ExperimentRegistry


@pytest.fixture()
def evidence_input():
    dates = pd.bdate_range("2024-01-02", periods=40).delete([3, 10, 30])
    gap_date = dates[-9]
    panel = pd.DataFrame(
        [
            {
                "as_of_date": date,
                "symbol": f"S{stock}",
                "feature_available_at": date,
                "label_end_date": date + pd.offsets.BDay(2),
                "signal": float(stock),
                "other": float((stock * 3 + day) % 7),
                "forward_excess_return": (
                    float("nan")
                    if date == gap_date
                    else stock * 0.02 + ((stock * 3 + day) % 7) * 0.003
                ),
            }
            for day, date in enumerate(dates)
            for stock in range(6)
        ]
    )
    config = BenchmarkConfig(
        feature_columns=("signal", "other"),
        split=NestedSplitConfig(3, 2, 3, 8, 1, 2, 4),
        min_cross_section=4,
        quantiles=2,
        hac_lags=1,
        model_families=("ridge",),
        ridge_alphas=(1.0,),
    )
    return panel, config, gap_date


@pytest.fixture()
def evidence_plan(evidence_input):
    panel, config, _ = evidence_input
    return build_stage3_data_plan(
        panel,
        feature_columns=config.feature_columns,
        target_column=config.target_column,
        locked_test_date_count=config.split.final_test_date_count,
        locked_min_cross_section=config.min_cross_section,
        n_splits=config.split.outer_n_splits,
        evaluation_date_count=config.split.outer_test_date_count,
        min_train_date_count=config.split.outer_min_train_date_count,
    )


def _iso(dates):
    return tuple(str(pd.Timestamp(date).date()) for date in dates)


def _assert_frozen_metadata(metadata):
    assert isinstance(metadata, MappingProxyType)
    assert metadata["definition_version"] == "1"
    assert metadata["source"] == "validated_panel_within_planned_phase_bounds"
    assert metadata["lag_unit"] == "scheduled_observations"
    phases = metadata["dates_by_phase"]
    assert isinstance(phases, MappingProxyType)
    assert all(isinstance(dates, tuple) for dates in phases.values())
    assert all(type(date) is str for dates in phases.values() for date in dates)
    with pytest.raises(TypeError):
        metadata["source"] = "changed"
    with pytest.raises(TypeError):
        phases["development"] = ()
    with pytest.raises(TypeError):
        phases["development"][0] = "1900-01-01"


def test_manifest_keeps_exact_panel_schedule_across_evaluation_gaps(
    evidence_input, evidence_plan
):
    from quant_metric_research.benchmark_schedules import (
        evaluation_schedule_metadata,
        phase_schedule,
    )

    panel, config, gap_date = evidence_input
    original = panel.copy(deep=True)
    first, last = evidence_plan.development_folds
    expected_dates = panel.loc[
        panel["as_of_date"].between(
            first.evaluation_start_date, last.evaluation_end_date
        ),
        "as_of_date",
    ].drop_duplicates()
    assert first.evaluation_end_date < gap_date < last.evaluation_start_date
    assert len(pd.bdate_range(expected_dates.min(), expected_dates.max())) > len(
        expected_dates
    )
    assert _iso(phase_schedule(evidence_plan, phase="development")) == _iso(
        expected_dates
    )

    development = _fingerprints(evidence_plan, config=config, evaluate_lockbox=False)
    full = _fingerprints(evidence_plan, config=config, evaluate_lockbox=True)
    for manifest, full_mode in ((development, False), (full, True)):
        metadata = manifest["evaluation_schedule"]
        assert metadata == evaluation_schedule_metadata(
            evidence_plan, evaluate_lockbox=full_mode
        )
        _assert_frozen_metadata(metadata)
        assert metadata["dates_by_phase"]["development"] == _iso(expected_dates)
        assert str(gap_date.date()) in metadata["dates_by_phase"]["development"]
    assert set(development["evaluation_schedule"]["dates_by_phase"]) == {"development"}
    assert set(full["evaluation_schedule"]["dates_by_phase"]) == {
        "development",
        "locked_test",
    }
    assert full["evaluation_schedule"]["dates_by_phase"]["locked_test"] == _iso(
        phase_schedule(evidence_plan, phase="locked_test")
    )
    pd.testing.assert_frame_equal(panel, original)


def _gapped_predictions(plan):
    missing_date = plan.development_folds[0].evaluation_start_date
    patterns = ((0, 1, 2, 3), (0, 2, 1, 3), (2, 0, 3, 1), (2, 3, 1, 0), (1, 0, 3, 2))
    dates = plan.panel.frame["as_of_date"].drop_duplicates().sort_values()
    rows = []
    for fold_number, fold in enumerate(plan.development_folds, start=1):
        for day, date in enumerate(dates):
            if not fold.evaluation_start_date <= date <= fold.evaluation_end_date:
                continue
            if date == missing_date:
                continue
            for stock, score in enumerate(patterns[day % len(patterns)]):
                rows.append(
                    {
                        "phase": "development",
                        "fold": fold_number,
                        "as_of_date": date,
                        "symbol": f"S{stock}",
                        "model": "ridge",
                        "score": float(score),
                        "target": float(stock),
                        "realized_return": stock / 10,
                    }
                )
    return pd.DataFrame(rows), missing_date


def _replay(predictions, schedules, config):
    return evaluate_prediction_frame(
        predictions,
        min_cross_section=config.min_cross_section,
        quantiles=config.quantiles,
        hac_lags=config.hac_lags,
        primary_models=("ridge",),
        expected_dates_by_phase=schedules,
    )


def test_written_schedule_replays_missing_inference_without_training(
    evidence_input, evidence_plan, tmp_path
):
    _, config, gap_date = evidence_input
    manifest = _fingerprints(evidence_plan, config=config, evaluate_lockbox=False)
    schedule = manifest["evaluation_schedule"]["dates_by_phase"]
    predictions, missing_date = _gapped_predictions(evidence_plan)
    prediction_dates = _iso(predictions["as_of_date"].unique())
    assert str(missing_date.date()) not in prediction_dates
    assert str(gap_date.date()) not in prediction_dates
    assert str(missing_date.date()) in schedule["development"]
    assert str(gap_date.date()) in schedule["development"]
    evaluation = _replay(predictions, schedule, config)
    result = benchmark.BenchmarkRun(
        data_gate=MappingProxyType({"claim_scope": "synthetic_replay_only"}),
        fold_assignments=predictions.loc[:, ["phase", "fold", "as_of_date"]],
        predictions=predictions,
        daily_metrics=evaluation.daily_metrics,
        fold_metrics=evaluation.fold_metrics,
        tuning_trials=pd.DataFrame(columns=["phase", "parameters"]),
        screening_by_fold=pd.DataFrame(columns=["phase", "feature"]),
        summary=evaluation.summary,
        acceptance=MappingProxyType({"acceptance_status": "not_evaluated"}),
        manifest=manifest,
    )
    first = write_benchmark_run(result, tmp_path / "first")
    second = write_benchmark_run(result, tmp_path / "second")
    assert (
        first.files["benchmark_manifest"].read_bytes()
        == second.files["benchmark_manifest"].read_bytes()
    )
    _assert_frozen_metadata(first.manifest["evaluation_schedule"])
    assert first.manifest["evaluation_schedule"] == manifest["evaluation_schedule"]

    saved = json.loads(first.files["benchmark_manifest"].read_text(encoding="utf-8"))
    saved_schedule = saved["evaluation_schedule"]["dates_by_phase"]
    assert saved_schedule == {"development": list(schedule["development"])}
    assert json.loads(json.dumps(saved, allow_nan=False)) == saved
    saved_predictions = pd.read_parquet(first.files["oos_predictions"])
    replayed = _replay(saved_predictions, saved_schedule, config)
    for name in ("daily_metrics", "fold_metrics", "summary"):
        pd.testing.assert_frame_equal(
            getattr(replayed, name), getattr(evaluation, name)
        )
    assert set(replayed.summary["inference_status"]) == {"missing_scheduled_values"}
    assert (
        replayed.summary[["newey_west_t_stat", "p_value", "bh_q_value"]]
        .isna()
        .all()
        .all()
    )

    # Recovering dates from surviving rows erases both kinds of evidence gaps.
    wrong_schedule = {
        "development": saved_predictions["as_of_date"].drop_duplicates().sort_values()
    }
    compressed = _replay(saved_predictions, wrong_schedule, config)
    assert set(compressed.summary["inference_status"]) == {"ok"}
    assert compressed.summary["p_value"].notna().all()
    assert (
        replayed.summary["scheduled_date_count"] == len(schedule["development"])
    ).all()
    assert (compressed.summary["scheduled_date_count"] == len(prediction_dates)).all()
    assert replayed.summary["mean_rank_ic"].equals(compressed.summary["mean_rank_ic"])


def test_registered_lifecycle_persists_the_schedules_used_for_inference(
    evidence_input, evidence_plan, monkeypatch, tmp_path
):
    panel, config, _ = evidence_input
    registry_path = tmp_path / "synthetic.sqlite3"
    registry = ExperimentRegistry(registry_path)
    evaluation_calls, paired_calls, locked_calls = [], [], []
    original_evaluate = evaluate_prediction_frame
    original_paired = benchmark.scheduled_newey_west_mean
    original_locked = benchmark._locked_run

    def capture_evaluation(predictions, **kwargs):
        evaluation_calls.append(
            {
                phase: _iso(dates)
                for phase, dates in kwargs["expected_dates_by_phase"].items()
            }
        )
        return original_evaluate(predictions, **kwargs)

    def capture_paired(values, **kwargs):
        paired_calls.append(_iso(kwargs["expected_dates"]))
        return original_paired(values, **kwargs)

    def reserved_locked(*args, **kwargs):
        records = [
            row
            for row in ExperimentRegistry(registry_path).list_runs()
            if row["kind"] == "final"
        ]
        assert len(records) == 1 and records[0]["status"] == "running"
        assert set(records[0]["manifest"]["evaluation_schedule"]["dates_by_phase"]) == {
            "development",
            "locked_test",
        }
        locked_calls.append(records[0]["run_id"])
        return original_locked(*args, **kwargs)

    monkeypatch.setattr(benchmark, "evaluate_prediction_frame", capture_evaluation)
    monkeypatch.setattr(
        experiment_workflow, "evaluate_prediction_frame", capture_evaluation
    )
    monkeypatch.setattr(benchmark, "scheduled_newey_west_mean", capture_paired)
    monkeypatch.setattr(benchmark, "_locked_run", reserved_locked)
    development = benchmark.run_stage3_benchmark(
        panel,
        config=config,
        registry=registry,
        study_id="schedule-evidence",
        hypothesis="Synthetic evidence for preserving declared date schedules.",
    )
    development_id = development.manifest["experiment"]["run_id"]
    development_schedule = development.manifest["evaluation_schedule"]["dates_by_phase"]
    assert set(development_schedule) == {"development"}
    assert locked_calls == paired_calls == []
    assert evaluation_calls == [dict(development_schedule)]
    assert registry.get_run(development_id)["status"] == "completed"

    final = benchmark.run_stage3_benchmark(
        panel,
        config=config,
        registry=registry,
        development_run_id=development_id,
        evaluate_lockbox=True,
    )
    final_id = final.manifest["experiment"]["run_id"]
    final_schedule = final.manifest["evaluation_schedule"]["dates_by_phase"]
    assert set(final_schedule) == {"development", "locked_test"}
    assert final_schedule["development"] == development_schedule["development"]
    assert evaluation_calls == [
        dict(development_schedule),
        dict(development_schedule),
        {"locked_test": final_schedule["locked_test"]},
    ]
    assert paired_calls == [final_schedule["locked_test"]]
    assert locked_calls == [final_id]
    assert registry.get_run(final_id)["status"] == "completed"
    assert len(registry.list_runs()) == 2
    assert final.acceptance["lockbox_reuse_registry_enforced"] is True
    assert final.acceptance["stage4_eligible"] is False
    assert (
        development.manifest["source_fingerprint"]
        == final.manifest["source_fingerprint"]
    )
    _assert_frozen_metadata(development.manifest["evaluation_schedule"])
    _assert_frozen_metadata(final.manifest["evaluation_schedule"])
    for name in ("predictions", "daily_metrics", "fold_metrics", "summary"):
        repeated = getattr(final, name).loc[lambda rows: rows["phase"] == "development"]
        pd.testing.assert_frame_equal(
            getattr(development, name).reset_index(drop=True),
            repeated.reset_index(drop=True),
        )
