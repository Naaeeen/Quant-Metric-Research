from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PanelConfig
from .contracts import (
    DataContractError,
    validate_as_of_dates,
    validate_memberships,
    validate_prices,
)
from .features import compute_price_metrics


def _active_symbols(
    memberships: pd.DataFrame,
    *,
    universe_id: str,
    as_of_date: pd.Timestamp,
) -> list[str]:
    active = memberships.loc[
        (memberships["universe_id"] == universe_id)
        & (memberships["effective_from"] <= as_of_date)
        & (
            memberships["effective_to"].isna()
            | (as_of_date < memberships["effective_to"])
        ),
        "symbol",
    ]
    return sorted(active.unique().tolist())


def _price_lookup(
    prices: pd.DataFrame,
) -> dict[tuple[str, pd.Timestamp], float]:
    return {
        (str(row.symbol), pd.Timestamp(row.date)): float(row.adjusted_close)
        for row in prices.itertuples(index=False)
    }


def _label_dates(
    benchmark_calendar: pd.DatetimeIndex,
    current_index: int,
    config: PanelConfig,
) -> tuple[pd.Timestamp | pd.NaT, pd.Timestamp | pd.NaT]:
    start_index = current_index + config.entry_lag_sessions
    end_index = start_index + config.target_horizon_sessions
    if end_index >= len(benchmark_calendar):
        return pd.NaT, pd.NaT
    return (
        pd.Timestamp(benchmark_calendar[start_index]),
        pd.Timestamp(benchmark_calendar[end_index]),
    )


def build_point_in_time_panel(
    prices: pd.DataFrame,
    memberships: pd.DataFrame,
    *,
    as_of_dates: list[pd.Timestamp] | tuple[pd.Timestamp, ...],
    config: PanelConfig,
) -> pd.DataFrame:
    validated_prices = validate_prices(prices)
    validated_memberships = validate_memberships(memberships)
    normalized_dates = validate_as_of_dates(as_of_dates)

    benchmark_prices = validated_prices.loc[
        validated_prices["symbol"] == config.benchmark_symbol
    ].copy()
    if benchmark_prices.empty:
        raise DataContractError("Benchmark price history is required.")

    benchmark_calendar = pd.DatetimeIndex(
        benchmark_prices["date"].sort_values().unique()
    )
    if any(date not in benchmark_calendar for date in normalized_dates):
        raise ValueError("All as_of_dates must be present in the benchmark calendar.")

    benchmark_by_date = benchmark_prices.set_index("date")[
        "adjusted_close"
    ].sort_index()
    calendar_positions = {
        pd.Timestamp(date): position for position, date in enumerate(benchmark_calendar)
    }
    prices_by_symbol = {
        symbol: group.set_index("date")["adjusted_close"].sort_index()
        for symbol, group in validated_prices.groupby("symbol", sort=False)
    }
    price_lookup = _price_lookup(validated_prices)
    rows: list[dict[str, object]] = []

    for as_of_date in sorted(normalized_dates):
        current_index = calendar_positions[as_of_date]
        start = max(0, current_index - config.lookback_sessions)
        window_dates = benchmark_calendar[start : current_index + 1]
        benchmark_window = benchmark_by_date.reindex(window_dates)
        label_start_date, label_end_date = _label_dates(
            benchmark_calendar,
            current_index,
            config,
        )

        for symbol in _active_symbols(
            validated_memberships,
            universe_id=config.universe_id,
            as_of_date=as_of_date,
        ):
            stock_window = prices_by_symbol.get(
                symbol,
                pd.Series(dtype=float),
            ).reindex(window_dates)
            metrics = compute_price_metrics(
                stock_window,
                benchmark_window,
                config,
            )
            feature_metrics = {
                name: value for name, value in metrics.items() if name != "eligible"
            }

            target_available = False
            target_status = "insufficient_future_benchmark_history"
            forward_return = float("nan")
            benchmark_return = float("nan")
            excess_return = float("nan")
            if pd.notna(label_start_date) and pd.notna(label_end_date):
                entry_key = (symbol, pd.Timestamp(label_start_date))
                exit_key = (symbol, pd.Timestamp(label_end_date))
                benchmark_entry_key = (
                    config.benchmark_symbol,
                    pd.Timestamp(label_start_date),
                )
                benchmark_exit_key = (
                    config.benchmark_symbol,
                    pd.Timestamp(label_end_date),
                )
                symbol_entry = price_lookup.get(entry_key)
                symbol_exit = price_lookup.get(exit_key)
                benchmark_entry = price_lookup.get(benchmark_entry_key)
                benchmark_exit = price_lookup.get(benchmark_exit_key)
                if symbol_entry is None or symbol_exit is None:
                    target_status = "missing_symbol_price"
                elif benchmark_entry is None or benchmark_exit is None:
                    target_status = "missing_benchmark_price"
                else:
                    forward_return = symbol_exit / symbol_entry - 1.0
                    benchmark_return = benchmark_exit / benchmark_entry - 1.0
                    excess_return = forward_return - benchmark_return
                    target_available = True
                    target_status = "ok"

            rows.append(
                {
                    "dataset_version": config.dataset_version,
                    "universe_id": config.universe_id,
                    "benchmark_symbol": config.benchmark_symbol,
                    "lookback_sessions": config.lookback_sessions,
                    "min_observations": config.min_observations,
                    "target_horizon_sessions": (config.target_horizon_sessions),
                    "entry_lag_sessions": config.entry_lag_sessions,
                    "annualization_sessions": (config.annualization_sessions),
                    "annual_risk_free_rate": (config.annual_risk_free_rate),
                    "as_of_date": as_of_date,
                    "symbol": symbol,
                    "feature_available_at": as_of_date,
                    "feature_window_start": pd.Timestamp(window_dates[0]),
                    "feature_window_end": as_of_date,
                    **feature_metrics,
                    "feature_eligible": bool(metrics["eligible"]),
                    "label_start_date": label_start_date,
                    "label_end_date": label_end_date,
                    "target_available": target_available,
                    "target_status": target_status,
                    "forward_return": forward_return,
                    "benchmark_forward_return": benchmark_return,
                    "forward_excess_return": excess_return,
                }
            )

    panel = pd.DataFrame.from_records(rows)
    if panel.empty:
        raise DataContractError(
            "No active universe members were found for the requested dates."
        )
    panel = panel.sort_values(
        ["as_of_date", "symbol"],
        kind="stable",
    ).reset_index(drop=True)
    panel["forward_excess_rank"] = np.nan
    available = panel["target_available"].astype(bool)
    panel.loc[available, "forward_excess_rank"] = (
        panel.loc[available]
        .groupby("as_of_date")["forward_excess_return"]
        .rank(method="average", pct=True)
    )
    return panel
