from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_metric_research.config import PanelConfig
from quant_metric_research.features import compute_price_metrics


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


def test_compute_price_metrics_matches_known_return_and_drawdown() -> None:
    dates = pd.bdate_range("2025-01-02", periods=6)
    stock = pd.Series([100.0, 110.0, 99.0, 108.9, 119.79, 107.811], index=dates)
    benchmark = pd.Series(
        [100.0, 105.0, 99.75, 104.7375, 109.974375, 104.47565625], index=dates
    )

    metrics = compute_price_metrics(stock, benchmark, _config())

    assert metrics["trailing_return"] == pytest.approx(0.07811)
    assert metrics["max_drawdown"] == pytest.approx(-0.10)
    assert metrics["observation_count"] == 5
    assert metrics["benchmark_correlation"] == pytest.approx(1.0)
    assert metrics["historical_var_5pct"] >= 0.0


def test_identical_stock_and_benchmark_have_unit_beta_and_zero_alpha() -> None:
    dates = pd.bdate_range("2025-01-02", periods=6)
    prices = pd.Series([100.0, 101.0, 99.0, 102.0, 101.0, 103.0], index=dates)

    metrics = compute_price_metrics(prices, prices, _config())

    assert metrics["beta"] == pytest.approx(1.0)
    assert metrics["capm_alpha"] == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(metrics["information_ratio"])


def test_compute_price_metrics_does_not_bridge_missing_sessions() -> None:
    dates = pd.bdate_range("2025-01-02", periods=6)
    stock = pd.Series([100.0, 101.0, np.nan, 105.0, 106.0, 107.0], index=dates)
    benchmark = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0], index=dates)

    metrics = compute_price_metrics(stock, benchmark, _config())

    assert metrics["observation_count"] == 3
    assert metrics["eligible"] is False
    assert all(np.isnan(metrics[name]) for name in _config().feature_columns)


def test_risk_adjusted_metrics_keep_existing_annualized_semantics() -> None:
    dates = pd.bdate_range("2025-01-02", periods=6)
    stock_returns = pd.Series(
        [0.02, -0.01, 0.03, -0.02, 0.01],
        index=dates[1:],
    )
    benchmark_returns = pd.Series(
        [0.01, -0.005, 0.015, -0.01, 0.005],
        index=dates[1:],
    )
    stock = pd.Series(
        [100.0, *(100.0 * (1.0 + stock_returns).cumprod())],
        index=dates,
    )
    benchmark = pd.Series(
        [100.0, *(100.0 * (1.0 + benchmark_returns).cumprod())],
        index=dates,
    )
    config = PanelConfig(
        dataset_version="test-v1",
        universe_id="TEST",
        benchmark_symbol="BENCH",
        lookback_sessions=5,
        min_observations=4,
        target_horizon_sessions=2,
        entry_lag_sessions=1,
        annualization_sessions=5,
        annual_risk_free_rate=0.05,
    )

    metrics = compute_price_metrics(stock, benchmark, config)
    stock_std = stock_returns.std(ddof=1)
    expected_sharpe = (stock_returns.mean() * 5 - 0.05) / (stock_std * np.sqrt(5))
    daily_target = 0.05 / 5
    downside_shortfall = np.minimum(stock_returns - daily_target, 0.0)
    downside_deviation = np.sqrt(np.mean(np.square(downside_shortfall))) * np.sqrt(5)
    expected_sortino = (stock_returns.mean() * 5 - 0.05) / downside_deviation
    expected_beta = stock_returns.cov(benchmark_returns) / (
        benchmark_returns.var(ddof=1)
    )
    expected_alpha = stock_returns.mean() * 5 - (
        0.05 + expected_beta * (benchmark_returns.mean() * 5 - 0.05)
    )

    assert metrics["sharpe_ratio"] == pytest.approx(expected_sharpe)
    assert metrics["sortino_ratio"] == pytest.approx(expected_sortino)
    assert metrics["capm_alpha"] == pytest.approx(expected_alpha)
