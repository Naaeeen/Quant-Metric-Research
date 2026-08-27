from __future__ import annotations

import pandas as pd

from quant_metric_research.benchmark import (
    BenchmarkConfig,
    NestedSplitConfig,
    run_stage3_benchmark,
)


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
                }
            )
    return pd.DataFrame(rows)


def _config() -> BenchmarkConfig:
    return BenchmarkConfig(
        feature_columns=("momentum", "value", "quality", "noise"),
        target_column="forward_excess_return",
        realized_return_column="forward_excess_return",
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
    result = run_stage3_benchmark(_panel(), config=_config())

    development = result.predictions.loc[result.predictions["phase"] == "development"]
    locked = result.predictions.loc[result.predictions["phase"] == "locked_test"]
    models = set(result.summary["model"])

    assert {"ridge", "hist_gradient_boosting"} <= models
    assert {"best_metric", "equal_weight_rank"} <= models
    assert {"metric:momentum", "metric:value"} <= models
    assert set(development["fold"]) == {1, 2}
    assert locked["fold"].nunique() == 1
    assert locked["as_of_date"].nunique() == 3
    assert result.acceptance["locked_test_used_once"] is True
    assert result.acceptance["frozen_model_family"] in {
        "ridge",
        "hist_gradient_boosting",
    }
    assert (
        pd.to_datetime(result.fold_assignments["train_label_end_max"])
        < pd.to_datetime(result.fold_assignments["evaluation_start"])
    ).all()


def test_locked_targets_cannot_change_development_or_frozen_model_scores() -> None:
    panel = _panel()
    config = _config()
    first = run_stage3_benchmark(panel, config=config)
    lock_dates = sorted(panel["as_of_date"].unique())[-3:]

    changed = panel.copy(deep=True)
    changed.loc[
        changed["as_of_date"].isin(lock_dates), "forward_excess_return"
    ] *= -1000
    second = run_stage3_benchmark(changed, config=config)

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
