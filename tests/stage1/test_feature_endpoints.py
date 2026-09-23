from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_metric_research.config import PanelConfig
from quant_metric_research.features import compute_price_metrics
from quant_metric_research.panel import build_point_in_time_panel


def _config() -> PanelConfig:
    return PanelConfig(
        dataset_version="endpoint-test",
        universe_id="TEST",
        benchmark_symbol="BENCH",
        lookback_sessions=7,
        min_observations=4,
        target_horizon_sessions=2,
        entry_lag_sessions=1,
        annualization_sessions=252,
    )


def _prices() -> tuple[pd.Series, pd.Series]:
    dates = pd.bdate_range("2025-01-02", periods=8)
    return (
        pd.Series(
            [100.0, 103.0, 101.0, 105.0, 104.0, 108.0, 106.0, 110.0], index=dates
        ),
        pd.Series(
            [100.0, 101.0, 102.0, 101.0, 103.0, 104.0, 103.0, 105.0], index=dates
        ),
    )


@pytest.mark.parametrize("missing", [(0,), (7,), (0, 7)])
def test_missing_endpoints_do_not_move_the_return_window(missing) -> None:
    stock, benchmark = _prices()
    stock = stock.mask(stock.index.isin(stock.index[list(missing)]))
    original_stock, original_benchmark = stock.copy(), benchmark.copy()

    metrics = compute_price_metrics(stock, benchmark, _config())

    assert np.isnan(metrics["trailing_return"])
    assert metrics["window_start_price_available"] is (0 not in missing)
    assert metrics["decision_price_available"] is (7 not in missing)
    assert metrics["eligible"] is True
    assert metrics["feature_status"] == "ok"
    assert metrics["observation_count"] == 7 - len(missing)
    # Removing only missing boundary positions preserves every adjacent pair
    # and the legacy drawdown path, but must not supply replacement endpoints.
    kept = stock.dropna().index
    trimmed = compute_price_metrics(stock.loc[kept], benchmark.loc[kept], _config())
    for name in _config().feature_columns:
        if name != "trailing_return":
            assert metrics[name] == pytest.approx(trimmed[name], nan_ok=True)
    pd.testing.assert_series_equal(stock, original_stock)
    pd.testing.assert_series_equal(benchmark, original_benchmark)


@pytest.mark.parametrize("missing", [(), (3,)])
def test_complete_endpoints_keep_exact_return_and_adjacent_pair_counts(missing) -> None:
    stock, benchmark = _prices()
    stock = stock.mask(stock.index.isin(stock.index[list(missing)]))
    metrics = compute_price_metrics(stock, benchmark, _config())

    assert metrics["trailing_return"] == pytest.approx(0.10)
    assert metrics["window_start_price_available"] is True
    assert metrics["decision_price_available"] is True
    assert metrics["observation_count"] == (5 if missing else 7)
    paired_returns = stock.pct_change(fill_method=None).dropna()
    assert metrics["annualized_volatility"] == pytest.approx(
        paired_returns.std(ddof=1) * np.sqrt(252)
    )


@pytest.mark.parametrize(
    ("values", "start_available", "end_available", "pair_count"),
    [
        ([], False, False, 0),
        ([np.nan] * 8, False, False, 0),
        ([100.0, 101.0], True, True, 1),
        ([np.nan, 101.0], False, True, 0),
        ([100.0, np.nan], True, False, 0),
    ],
)
def test_insufficient_history_reports_endpoint_availability(
    values,
    start_available,
    end_available,
    pair_count,
) -> None:
    stock = pd.Series(values, dtype=float)
    benchmark = pd.Series(np.arange(len(values)) + 100.0, dtype=float)
    metrics = compute_price_metrics(stock, benchmark, _config())

    assert metrics["window_start_price_available"] is start_available
    assert metrics["decision_price_available"] is end_available
    assert metrics["observation_count"] == pair_count
    assert metrics["eligible"] is False
    assert metrics["feature_status"] == "insufficient_history"
    assert all(np.isnan(metrics[name]) for name in _config().feature_columns)


@pytest.mark.parametrize("features", [("trailing_return",), ("beta", "max_drawdown")])
def test_endpoint_diagnostics_do_not_expand_configured_feature_subsets(
    features,
) -> None:
    stock, benchmark = _prices()
    stock = stock.mask(stock.index == stock.index[-1])
    metrics = compute_price_metrics(
        stock, benchmark, replace(_config(), feature_columns=features)
    )

    assert set(metrics) == set(features) | {
        "observation_count",
        "eligible",
        "feature_status",
        "window_start_price_available",
        "decision_price_available",
    }
    assert metrics["window_start_price_available"] is True
    assert metrics["decision_price_available"] is False


@pytest.mark.parametrize("missing_positions", [(3,), (10,), (3, 10), tuple(range(11))])
def test_panel_retains_rows_labels_and_future_invariant_diagnostics(
    market_fixture,
    missing_positions,
) -> None:
    prices, memberships, dates = market_fixture
    original_prices, original_memberships = prices.copy(), memberships.copy()
    missing = prices.loc[
        ~(
            (prices["symbol"] == "AAA")
            & prices["date"].isin(dates[list(missing_positions)])
        )
    ].copy()
    panel = build_point_in_time_panel(
        missing,
        memberships,
        as_of_dates=[dates[10]],
        config=_config(),
    )
    baseline = build_point_in_time_panel(
        prices,
        memberships,
        as_of_dates=[dates[10]],
        config=_config(),
    )
    assert panel["symbol"].tolist() == ["AAA", "BBB"]
    label_columns = [
        name for name in panel if name.startswith(("label_", "target_", "forward_"))
    ]
    pd.testing.assert_frame_equal(panel[label_columns], baseline[label_columns])
    aaa = panel.loc[panel["symbol"] == "AAA"].iloc[0]
    assert np.isnan(aaa["trailing_return"])
    assert bool(aaa["window_start_price_available"]) is (3 not in missing_positions)
    assert bool(aaa["decision_price_available"]) is (10 not in missing_positions)
    assert bool(aaa["feature_eligible"]) is (len(missing_positions) < 3)

    future_changed = missing.assign(
        adjusted_close=missing["adjusted_close"].where(
            ~((missing["symbol"] == "AAA") & (missing["date"] == dates[13])),
            missing["adjusted_close"] * 2.0,
        )
    )
    rebuilt = build_point_in_time_panel(
        future_changed,
        memberships,
        as_of_dates=[dates[10]],
        config=_config(),
    )
    historical_columns = [name for name in panel if name not in label_columns]
    pd.testing.assert_frame_equal(
        panel[historical_columns], rebuilt[historical_columns]
    )
    assert not np.allclose(panel["forward_return"], rebuilt["forward_return"])
    pd.testing.assert_frame_equal(prices, original_prices)
    pd.testing.assert_frame_equal(memberships, original_memberships)


def test_complete_data_matches_pre_correction_metrics() -> None:
    stock, benchmark = _prices()
    metrics = compute_price_metrics(stock, benchmark, _config())
    # Recorded on the synthetic fixture using baseline revision c6b90b0.
    expected = {
        "trailing_return": 0.10000000000000009,
        "annualized_volatility": 0.44903886718249003,
        "sharpe_ratio": 7.884162926445231,
        "sortino_ratio": 20.723561331889556,
        "max_drawdown": -0.01941747572815533,
        "benchmark_correlation": 0.058143881400468354,
        "beta": 0.13419587848193198,
        "capm_alpha": 3.3015939191768777,
        "information_ratio": 3.678432057922955,
        "historical_var_5pct": 0.01914778856526428,
    }
    assert {name: metrics[name] for name in expected} == pytest.approx(expected)


def test_panel_keeps_active_member_without_any_prices(market_fixture) -> None:
    prices, memberships, dates = market_fixture
    panel = build_point_in_time_panel(
        prices.loc[prices["symbol"] != "AAA"],
        memberships,
        as_of_dates=[dates[10]],
        config=_config(),
    )
    assert panel["symbol"].tolist() == ["AAA", "BBB"]
    aaa = panel.loc[panel["symbol"] == "AAA"].iloc[0]
    assert not bool(aaa["window_start_price_available"])
    assert not bool(aaa["decision_price_available"])
    assert aaa["observation_count"] == 0
    assert not bool(aaa["feature_eligible"])
    assert aaa["feature_status"] == "insufficient_history"
    assert not bool(aaa["target_available"])
    assert aaa["target_status"] == "missing_symbol_price"
    assert all(np.isnan(aaa[name]) for name in _config().feature_columns)
