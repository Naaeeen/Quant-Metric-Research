from __future__ import annotations

import pandas as pd

from quant_metric_research.redundancy import compute_feature_redundancy
from quant_metric_research.screening import fit_metric_screen


def _panel() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=8)
    rows = []
    for date_index, date in enumerate(dates):
        for symbol_index in range(6):
            signal = float(symbol_index)
            rows.append(
                {
                    "as_of_date": date,
                    "symbol": f"S{symbol_index}",
                    "signal": signal,
                    "signal_clone": signal * 100.0 + date_index,
                    "noise": float((symbol_index * 3 + date_index * 2) % 7),
                    "constant": 1.0,
                    "target": signal * 0.01 + date_index * 0.0001,
                }
            )
    return pd.DataFrame(rows)


def test_redundancy_is_aggregated_from_cross_sections() -> None:
    redundancy = compute_feature_redundancy(
        _panel(),
        feature_columns=("signal", "signal_clone", "noise"),
        min_cross_section=5,
    )
    pair = redundancy.loc[
        (redundancy["left"] == "signal") & (redundancy["right"] == "signal_clone")
    ].iloc[0]

    assert pair["mean_spearman"] > 0.999
    assert pair["date_count"] == 8


def test_metric_screen_fits_only_through_explicit_training_date() -> None:
    panel = _panel()
    train_end = panel["as_of_date"].sort_values().unique()[4]

    original = fit_metric_screen(
        panel,
        feature_columns=("signal", "signal_clone", "noise", "constant"),
        target_column="target",
        train_end_date=train_end,
        min_cross_section=5,
        minimum_coverage=0.8,
        redundancy_threshold=0.9,
        hac_lags=1,
    )
    changed = panel.copy(deep=True)
    future = changed["as_of_date"] > train_end
    changed.loc[future, "signal"] *= -1000.0
    changed.loc[future, "target"] *= -1000.0
    rebuilt = fit_metric_screen(
        changed,
        feature_columns=("signal", "signal_clone", "noise", "constant"),
        target_column="target",
        train_end_date=train_end,
        min_cross_section=5,
        minimum_coverage=0.8,
        redundancy_threshold=0.9,
        hac_lags=1,
    )

    pd.testing.assert_frame_equal(original.quality, rebuilt.quality)
    pd.testing.assert_frame_equal(original.ic_summary, rebuilt.ic_summary)
    pd.testing.assert_frame_equal(original.redundancy, rebuilt.redundancy)
    assert original.selected_features == rebuilt.selected_features
    assert original.fitted_through == pd.Timestamp(train_end)


def test_metric_screen_drops_constant_and_redundant_columns() -> None:
    result = fit_metric_screen(
        _panel(),
        feature_columns=("signal", "signal_clone", "noise", "constant"),
        target_column="target",
        train_end_date="2025-01-13",
        min_cross_section=5,
        minimum_coverage=0.8,
        redundancy_threshold=0.9,
        hac_lags=1,
    )

    assert "constant" not in result.selected_features
    assert not ({"signal", "signal_clone"} <= set(result.selected_features))
    assert result.dropped_features["constant"] == "zero_variance"
