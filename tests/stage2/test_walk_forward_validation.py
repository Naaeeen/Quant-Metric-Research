from __future__ import annotations

import pandas as pd

from quant_metric_research.validation import (
    WalkForwardMetricConfig,
    evaluate_metrics_walk_forward,
)


def _panel() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=14)
    rows: list[dict[str, object]] = []
    for date_index, date in enumerate(dates):
        for symbol_index in range(6):
            signal = float(symbol_index)
            rows.append(
                {
                    "as_of_date": date,
                    "symbol": f"S{symbol_index}",
                    "label_end_date": dates[min(date_index + 2, len(dates) - 1)],
                    "signal": signal,
                    "inverse_signal": -signal,
                    "signal_clone": signal * 10.0 + date_index,
                    "noise": float((symbol_index * 3 + date_index * 2) % 7),
                    "constant": 1.0,
                    "target": signal * 0.01 + date_index * 0.0001,
                }
            )
    return pd.DataFrame(rows)


def test_walk_forward_validation_reports_out_of_sample_metric_behavior() -> None:
    result = evaluate_metrics_walk_forward(
        _panel(),
        feature_columns=(
            "signal",
            "signal_clone",
            "noise",
            "constant",
        ),
        target_column="target",
        config=WalkForwardMetricConfig(
            n_splits=2,
            test_date_count=2,
            min_train_date_count=3,
        ),
        min_cross_section=5,
        minimum_coverage=0.8,
        redundancy_threshold=0.9,
        hac_lags=1,
        quantiles=3,
    )

    signal = result.summary.set_index("feature").loc["signal"]
    assert signal["mean_test_rank_ic"] > 0.99
    assert signal["selection_rate"] == 1.0
    assert result.fold_metrics.shape[0] == 8
    assert (
        result.fold_metrics["train_label_end_max"]
        < result.fold_metrics["test_start_date"]
    ).all()


def test_walk_forward_validation_tracks_redundant_feature_selection() -> None:
    result = evaluate_metrics_walk_forward(
        _panel(),
        feature_columns=("signal", "signal_clone"),
        target_column="target",
        config=WalkForwardMetricConfig(
            n_splits=2,
            test_date_count=2,
            min_train_date_count=3,
        ),
        min_cross_section=5,
        minimum_coverage=0.8,
        redundancy_threshold=0.9,
        hac_lags=1,
    )

    summary = result.summary.set_index("feature")
    assert summary.loc["signal", "selection_rate"] == 1.0
    assert summary.loc["signal_clone", "selection_rate"] == 0.0


def test_walk_forward_validation_preserves_predictive_direction() -> None:
    result = evaluate_metrics_walk_forward(
        _panel(),
        feature_columns=("inverse_signal",),
        target_column="target",
        config=WalkForwardMetricConfig(
            n_splits=2,
            test_date_count=2,
            min_train_date_count=3,
        ),
        min_cross_section=5,
        minimum_coverage=0.8,
        redundancy_threshold=0.9,
        hac_lags=1,
    )

    inverse = result.summary.set_index("feature").loc["inverse_signal"]
    assert inverse["mean_test_rank_ic"] < -0.99
    assert inverse["mean_directional_test_rank_ic"] > 0.99
    assert inverse["direction_consistency_rate"] == 1.0
