from __future__ import annotations

import json
from types import MappingProxyType

import pandas as pd
import pytest

from quant_metric_research.benchmark import (
    BenchmarkConfig,
    NestedSplitConfig,
    _model_predictions,
    run_stage3_benchmark,
)
from quant_metric_research.benchmark_models import ModelCandidate
from quant_metric_research.experiment_workflow import _execute_benchmark
from quant_metric_research.screening import MetricScreenResult


def _panel() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=36)
    rows: list[dict[str, object]] = []
    for date_index, as_of_date in enumerate(dates):
        for symbol_index in range(10):
            centered = float(symbol_index) - 4.5
            momentum = centered + 0.03 * date_index
            value = float((symbol_index * 3 + date_index) % 11) - 5.0
            quality = -centered + 0.02 * date_index
            noise = float((symbol_index * 7 + date_index * 5) % 13) - 6.0
            target = 0.03 * momentum + 0.008 * value - 0.002 * quality
            rows.append(
                {
                    "dataset_version": "synthetic-v1",
                    "as_of_date": as_of_date,
                    "symbol": f"S{symbol_index:02d}",
                    "feature_available_at": as_of_date,
                    "label_end_date": as_of_date + pd.offsets.BDay(2),
                    "momentum": momentum,
                    "value": value,
                    "quality": quality,
                    "noise": noise,
                    "forward_excess_return": target,
                    "realized_return": (
                        float("nan")
                        if date_index == len(dates) - 1 and symbol_index == 9
                        else target
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    no_observed_features = (frame["as_of_date"] == dates[-1]) & (
        frame["symbol"] == "S08"
    )
    frame.loc[
        no_observed_features,
        ["momentum", "value", "quality", "noise"],
    ] = float("nan")
    return frame


def _config() -> BenchmarkConfig:
    return BenchmarkConfig(
        feature_columns=("momentum", "value", "quality", "noise"),
        target_column="forward_excess_return",
        realized_return_column="realized_return",
        split=NestedSplitConfig(
            final_test_date_count=3,
            outer_n_splits=2,
            outer_test_date_count=3,
            outer_min_train_date_count=12,
            inner_n_splits=2,
            inner_validation_date_count=2,
            inner_min_train_date_count=6,
        ),
        min_cross_section=6,
        minimum_coverage=0.8,
        redundancy_threshold=0.95,
        quantiles=3,
        hac_lags=1,
        ridge_alphas=(0.1, 1.0),
        hist_learning_rates=(0.05,),
        hist_max_leaf_nodes=(7,),
        hist_l2_regularization=(1.0,),
        hist_max_iter=30,
        hist_min_samples_leaf=3,
        random_seed=17,
    )


def test_stage3_runs_nested_development_and_one_locked_test() -> None:
    # This test isolates the numerical kernel. Public lifecycle tests separately
    # enforce durable evidence and reservation; this callback is test-only.
    result = _execute_benchmark(
        _panel(),
        config=_config(),
        evaluate_lockbox=True,
        before_lockbox=lambda manifest, family: None,
    )

    development = result.predictions.loc[result.predictions["phase"] == "development"]
    locked = result.predictions.loc[result.predictions["phase"] == "locked_test"]
    models = set(result.summary["model"])

    assert {"ridge", "hist_gradient_boosting"} <= models
    assert {"best_metric", "equal_weight_rank"} <= models
    assert {"metric:momentum", "metric:value"} <= models
    assert set(development["fold"]) == {1, 2}
    assert locked["fold"].nunique() == 1
    assert locked["as_of_date"].nunique() == 3
    assert result.acceptance["lockbox_evaluated_once_in_this_run"] is True
    assert result.acceptance["lockbox_reuse_registry_enforced"] is False
    assert result.acceptance["frozen_model_family"] in {
        "ridge",
        "hist_gradient_boosting",
    }
    assert (
        pd.to_datetime(result.fold_assignments["train_label_end_max"])
        < pd.to_datetime(result.fold_assignments["evaluation_start"])
    ).all()
    assert result.data_gate["locked_cross_section_count_by_date"]
    assert result.data_gate["locked_evaluable_count_by_date"]
    assert (
        min(result.data_gate["locked_realized_return_coverage_by_date"].values()) == 0.9
    )
    assert result.manifest["artifact_schema_version"] == "4"
    assert result.manifest["package_version"] == "0.9.0"
    assert result.manifest["execution_mode"] == "full"
    assert len(result.manifest["source_fingerprint"]) == 64
    assert result.manifest["fingerprint_scope"] == (
        "source_code+configuration+validated_panel_contract+execution_mode"
    )
    assert all(
        isinstance(json.loads(value), list)
        for value in result.predictions["selected_features"].dropna().unique()
    )
    zero_input_models = locked.loc[
        (locked["symbol"] == "S08")
        & (locked["as_of_date"] == locked["as_of_date"].max())
        & locked["model"].isin({"ridge", "hist_gradient_boosting", "ridge_pca"})
    ]
    assert zero_input_models["score"].isna().all()
    assert (zero_input_models["feature_count"] == 0).all()
    assert zero_input_models["zero_observed_features"].all()
    assert (zero_input_models["selected_feature_count"] > 0).all()


def test_default_run_never_evaluates_or_exports_lockbox(monkeypatch, tmp_path) -> None:
    import quant_metric_research.benchmark as benchmark
    import quant_metric_research.experiment_workflow as workflow
    from quant_metric_research.benchmark_io import write_benchmark_run

    def forbidden(*args, **kwargs):
        raise AssertionError("Development must not evaluate locked outcomes.")

    monkeypatch.setattr(benchmark, "_locked_run", forbidden)
    monkeypatch.setattr(workflow, "_data_gate", forbidden)
    result = run_stage3_benchmark(_panel(), config=_config())
    for frame in (
        result.predictions,
        result.daily_metrics,
        result.fold_metrics,
        result.tuning_trials,
        result.screening_by_fold,
        result.summary,
        result.fold_assignments,
    ):
        assert set(frame["phase"]) == {"development"}
    assert result.acceptance["lockbox_evaluated_once_in_this_run"] is False
    assert result.acceptance["acceptance_status"] == "not_evaluated"
    assert result.acceptance["model_gate_passed"] is False
    assert result.acceptance["stage4_eligible"] is False
    assert result.manifest["execution_mode"] == "development"
    assert not any("coverage_by_date" in key for key in result.data_gate)
    artifacts = write_benchmark_run(result, tmp_path / "development")
    saved = pd.read_parquet(artifacts.files["oos_predictions"])
    assert set(saved["phase"]) == {"development"}


def test_development_results_do_not_depend_on_reserved_outcomes() -> None:
    panel = _panel()
    first = run_stage3_benchmark(panel, config=_config())
    locked_start = pd.Timestamp(first.manifest["locked_test_start"])
    changed = panel.assign(
        forward_excess_return=panel["forward_excess_return"].where(
            panel["as_of_date"] < locked_start, -999.0
        ),
        realized_return=panel["realized_return"].where(
            panel["as_of_date"] < locked_start, 999.0
        ),
    )
    second = run_stage3_benchmark(changed, config=_config())
    assert set(first.predictions["phase"]) == {"development"}
    for name in ("predictions", "summary", "tuning_trials", "screening_by_fold"):
        pd.testing.assert_frame_equal(getattr(first, name), getattr(second, name))
    assert first.acceptance == second.acceptance


@pytest.mark.parametrize("value", [1, "false", None])
def test_lockbox_opt_in_requires_a_boolean(value) -> None:
    with pytest.raises(ValueError, match="evaluate_lockbox must be a boolean"):
        run_stage3_benchmark(_panel(), config=_config(), evaluate_lockbox=value)


def test_model_feature_count_only_counts_selected_observed_inputs() -> None:
    training_dates = pd.to_datetime(
        ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]
    )
    training = pd.DataFrame(
        {
            "as_of_date": training_dates,
            "symbol": ["A", "B", "A", "B"],
            "label_end_date": training_dates + pd.offsets.BDay(1),
            "momentum": [0.0, 1.0, 0.5, 1.5],
            "value": [10.0, 11.0, 12.0, 13.0],
            "forward_excess_return": [0.0, 0.1, 0.05, 0.15],
        }
    )
    evaluation = pd.DataFrame(
        {
            "row_id": [99],
            "as_of_date": [pd.Timestamp("2024-01-08")],
            "symbol": ["A"],
            "momentum": [float("nan")],
            "value": [999.0],
            "forward_excess_return": [0.2],
            "realized_return": [0.2],
        }
    )
    screen = MetricScreenResult(
        quality=pd.DataFrame(),
        daily_rank_ic=pd.DataFrame(),
        ic_summary=pd.DataFrame(),
        quantile_spreads=pd.DataFrame(),
        redundancy=pd.DataFrame(),
        selected_features=("momentum",),
        dropped_features=MappingProxyType({"value": "test-only"}),
        fitted_through=pd.Timestamp("2024-01-03"),
        training_row_count=len(training),
    )
    candidate = ModelCandidate(
        family="ridge",
        candidate_id="ridge:test",
        parameters=MappingProxyType({"alpha": 1.0}),
    )

    prediction = _model_predictions(
        training,
        evaluation,
        screen=screen,
        candidate=candidate,
        config=_config(),
        phase="locked_test",
        fold=3,
        evaluation_start=pd.Timestamp("2024-01-08"),
    ).iloc[0]

    assert evaluation.loc[0, "value"] == 999.0
    assert prediction["feature_count"] == 0
    assert prediction["selected_feature_count"] == 1
    assert bool(prediction["zero_observed_features"]) is True
    assert pd.isna(prediction["score"])


def test_locked_targets_cannot_change_development_or_frozen_model_scores() -> None:
    panel = _panel()
    config = _config()
    first = _execute_benchmark(
        panel,
        config=config,
        evaluate_lockbox=True,
        before_lockbox=lambda manifest, family: None,
    )
    lock_dates = sorted(panel["as_of_date"].unique())[-3:]

    changed = panel.copy(deep=True)
    changed.loc[
        changed["as_of_date"].isin(lock_dates), "forward_excess_return"
    ] *= -1000
    second = _execute_benchmark(
        changed,
        config=config,
        evaluate_lockbox=True,
        before_lockbox=lambda manifest, family: None,
    )

    prediction_key = ["phase", "fold", "as_of_date", "symbol", "model"]
    first_scores = first.predictions.sort_values(prediction_key).reset_index(drop=True)
    second_scores = second.predictions.sort_values(prediction_key).reset_index(
        drop=True
    )

    pd.testing.assert_frame_equal(first.tuning_trials, second.tuning_trials)
    pd.testing.assert_series_equal(first_scores["score"], second_scores["score"])
    assert (
        first.acceptance["frozen_model_family"]
        == second.acceptance["frozen_model_family"]
    )


def test_stage3_run_is_deterministic() -> None:
    first = run_stage3_benchmark(_panel(), config=_config())
    second = run_stage3_benchmark(_panel(), config=_config())

    pd.testing.assert_frame_equal(first.fold_assignments, second.fold_assignments)
    pd.testing.assert_frame_equal(first.predictions, second.predictions)
    pd.testing.assert_frame_equal(first.daily_metrics, second.daily_metrics)
    assert first.manifest["run_fingerprint"] == second.manifest["run_fingerprint"]
