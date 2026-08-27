from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_metric_research.config import PanelConfig
from quant_metric_research.panel import build_point_in_time_panel


def _config() -> PanelConfig:
    return PanelConfig(
        dataset_version="test-v1",
        universe_id="TEST",
        benchmark_symbol="BENCH",
        lookback_sessions=5,
        min_observations=4,
        target_horizon_sessions=2,
        entry_lag_sessions=1,
        annualization_sessions=5,
        annual_risk_free_rate=0.0,
    )


def test_panel_has_one_row_per_active_symbol_and_date(market_fixture) -> None:
    prices, memberships, dates = market_fixture
    as_of_dates = [dates[6], dates[10]]

    panel = build_point_in_time_panel(
        prices,
        memberships,
        as_of_dates=as_of_dates,
        config=_config(),
    )

    keys = list(panel[["as_of_date", "symbol"]].itertuples(index=False, name=None))
    assert keys == [(dates[6], "AAA"), (dates[10], "AAA"), (dates[10], "BBB")]
    assert not panel.duplicated(["as_of_date", "symbol"]).any()
    assert set(_config().feature_columns).issubset(panel.columns)


def test_features_are_unchanged_when_only_future_prices_change(market_fixture) -> None:
    prices, memberships, dates = market_fixture
    as_of = dates[10]
    original = build_point_in_time_panel(
        prices,
        memberships,
        as_of_dates=[as_of],
        config=_config(),
    )

    changed = prices.copy(deep=True)
    changed.loc[
        (changed["symbol"] == "AAA") & (changed["date"] == dates[13]),
        "adjusted_close",
    ] *= 50.0
    rebuilt = build_point_in_time_panel(
        changed,
        memberships,
        as_of_dates=[as_of],
        config=_config(),
    )

    pd.testing.assert_frame_equal(
        original[["symbol", *_config().feature_columns]],
        rebuilt[["symbol", *_config().feature_columns]],
    )
    assert not np.allclose(
        original["forward_excess_return"],
        rebuilt["forward_excess_return"],
    )


def test_label_uses_benchmark_sessions_and_explicit_entry_lag(market_fixture) -> None:
    prices, memberships, dates = market_fixture
    as_of = dates[10]

    panel = build_point_in_time_panel(
        prices,
        memberships,
        as_of_dates=[as_of],
        config=_config(),
    )
    aaa = panel.loc[panel["symbol"] == "AAA"].iloc[0]

    assert aaa["label_start_date"] == dates[11]
    assert aaa["label_end_date"] == dates[13]
    aaa_prices = prices.loc[prices["symbol"] == "AAA"].set_index("date")
    benchmark = prices.loc[prices["symbol"] == "BENCH"].set_index("date")
    expected = (
        aaa_prices.loc[dates[13], "adjusted_close"]
        / aaa_prices.loc[dates[11], "adjusted_close"]
        - 1.0
    ) - (
        benchmark.loc[dates[13], "adjusted_close"]
        / benchmark.loc[dates[11], "adjusted_close"]
        - 1.0
    )
    assert aaa["forward_excess_return"] == pytest.approx(expected)


def test_missing_entry_price_keeps_row_and_marks_label_unavailable(
    market_fixture,
) -> None:
    prices, memberships, dates = market_fixture
    as_of = dates[10]
    missing = prices.loc[
        ~((prices["symbol"] == "AAA") & (prices["date"] == dates[11]))
    ].copy()

    panel = build_point_in_time_panel(
        missing,
        memberships,
        as_of_dates=[as_of],
        config=_config(),
    )
    aaa = panel.loc[panel["symbol"] == "AAA"].iloc[0]

    assert bool(aaa["target_available"]) is False
    assert np.isnan(aaa["forward_excess_return"])
    assert aaa["target_status"] == "missing_symbol_price"


def test_panel_retains_insufficient_history_with_quality_status(market_fixture) -> None:
    prices, memberships, dates = market_fixture

    panel = build_point_in_time_panel(
        prices,
        memberships,
        as_of_dates=[dates[2]],
        config=_config(),
    )
    row = panel.iloc[0]

    assert bool(row["feature_eligible"]) is False
    assert row["feature_status"] == "insufficient_history"
    assert all(np.isnan(row[name]) for name in _config().feature_columns)


def test_cross_sectional_target_rank_is_computed_within_date(market_fixture) -> None:
    prices, memberships, dates = market_fixture
    panel = build_point_in_time_panel(
        prices,
        memberships,
        as_of_dates=[dates[10]],
        config=_config(),
    ).sort_values("forward_excess_return")

    assert panel["forward_excess_rank"].tolist() == pytest.approx([0.5, 1.0])


def test_panel_rejects_as_of_date_outside_benchmark_calendar(market_fixture) -> None:
    prices, memberships, _ = market_fixture
    with pytest.raises(ValueError, match="benchmark calendar"):
        build_point_in_time_panel(
            prices,
            memberships,
            as_of_dates=[pd.Timestamp("2025-01-04")],
            config=_config(),
        )


def test_panel_persists_reproducibility_configuration(market_fixture) -> None:
    prices, memberships, dates = market_fixture

    panel = build_point_in_time_panel(
        prices,
        memberships,
        as_of_dates=[dates[10]],
        config=_config(),
    )
    row = panel.iloc[0]

    assert row["dataset_version"] == "test-v1"
    assert row["lookback_sessions"] == 5
    assert row["min_observations"] == 4
    assert row["target_horizon_sessions"] == 2
    assert row["entry_lag_sessions"] == 1
    assert row["annualization_sessions"] == 5
