from __future__ import annotations

from math import sqrt

import numpy as np
import pandas as pd

from .config import PanelConfig


def _empty_metrics(
    config: PanelConfig,
    observation_count: int,
    status: str,
) -> dict[str, float | bool | str]:
    metrics: dict[str, float | bool | str] = {
        feature: float("nan") for feature in config.feature_columns
    }
    metrics.update(
        {
            "observation_count": observation_count,
            "eligible": False,
            "feature_status": status,
        }
    )
    return metrics


def _safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    if not np.isfinite(denominator) or denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def compute_price_metrics(
    stock_prices: pd.Series,
    benchmark_prices: pd.Series,
    config: PanelConfig,
) -> dict[str, float | bool | str]:
    stock = pd.to_numeric(stock_prices, errors="coerce")
    benchmark = pd.to_numeric(benchmark_prices, errors="coerce")
    stock_returns = stock.pct_change(fill_method=None)
    benchmark_returns = benchmark.pct_change(fill_method=None)
    paired = pd.concat(
        [
            stock_returns.rename("stock"),
            benchmark_returns.rename("benchmark"),
        ],
        axis=1,
    ).dropna()
    observation_count = int(paired.shape[0])
    if observation_count < config.min_observations:
        return _empty_metrics(
            config,
            observation_count,
            "insufficient_history",
        )

    clean_stock = stock.dropna()
    stock_series = paired["stock"]
    benchmark_series = paired["benchmark"]
    active_returns = stock_series - benchmark_series
    annualized_stock_return = float(stock_series.mean()) * config.annualization_sessions
    annualized_benchmark_return = (
        float(benchmark_series.mean()) * config.annualization_sessions
    )
    annualized_excess_return = annualized_stock_return - config.annual_risk_free_rate

    numerical_floor = np.finfo(float).eps
    benchmark_variance = float(benchmark_series.var(ddof=1))
    beta = (
        float(stock_series.cov(benchmark_series) / benchmark_variance)
        if benchmark_variance > numerical_floor
        else float("nan")
    )
    capm_alpha = (
        float(
            annualized_stock_return
            - (
                config.annual_risk_free_rate
                + beta * (annualized_benchmark_return - config.annual_risk_free_rate)
            )
        )
        if np.isfinite(beta)
        else float("nan")
    )

    stock_std = float(stock_series.std(ddof=1))
    daily_target = config.annual_risk_free_rate / config.annualization_sessions
    downside_shortfall = np.minimum(
        stock_series.to_numpy(dtype=float) - daily_target,
        0.0,
    )
    annualized_downside_deviation = float(
        np.sqrt(np.mean(np.square(downside_shortfall)))
        * sqrt(config.annualization_sessions)
    )
    active_std = float(active_returns.std(ddof=1))
    rolling_peak = clean_stock.cummax()
    drawdowns = clean_stock / rolling_peak - 1.0

    all_metrics: dict[str, float] = {
        "trailing_return": float(clean_stock.iloc[-1] / clean_stock.iloc[0] - 1.0),
        "annualized_volatility": (
            float(stock_std * sqrt(config.annualization_sessions))
            if stock_std > numerical_floor
            else float("nan")
        ),
        "sharpe_ratio": _safe_ratio(
            annualized_excess_return,
            stock_std * sqrt(config.annualization_sessions),
        ),
        "sortino_ratio": _safe_ratio(
            annualized_excess_return,
            annualized_downside_deviation,
        ),
        "max_drawdown": float(drawdowns.min()),
        "benchmark_correlation": (
            float(stock_series.corr(benchmark_series))
            if stock_std > numerical_floor
            and sqrt(benchmark_variance) > numerical_floor
            else float("nan")
        ),
        "beta": beta,
        "capm_alpha": capm_alpha,
        "information_ratio": _safe_ratio(
            float(active_returns.mean()) * config.annualization_sessions,
            active_std * sqrt(config.annualization_sessions),
        ),
        "historical_var_5pct": float(max(0.0, -stock_series.quantile(0.05))),
    }
    selected: dict[str, float | bool | str] = {
        name: all_metrics[name] for name in config.feature_columns
    }
    selected.update(
        {
            "observation_count": observation_count,
            "eligible": True,
            "feature_status": "ok",
        }
    )
    return selected
