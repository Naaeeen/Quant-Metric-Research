"""Explicit raw-factor enrichment that preserves the legacy Stage 1 panel."""

from __future__ import annotations

import pandas as pd

from .config import PanelConfig
from .contracts import DataContractError, _daily_dates, _normalized_text
from .panel import build_point_in_time_panel
from .price_factors import PriceFactorValue, _requested_specs, compute_price_factors


def _column_schema(factor_names: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (name + suffix, dtype)
        for name in factor_names
        for suffix, dtype in (
            ("", "float64"),
            ("_available_at", "datetime64[ns]"),
            ("_status", "string"),
            ("_window_start", "datetime64[ns]"),
            ("_window_end", "datetime64[ns]"),
            ("_required_price_count", "int64"),
            ("_observed_price_count", "int64"),
            ("_formula_version", "string"),
        )
    )


def _raw_price_history(
    prices: pd.DataFrame, benchmark_symbol: str
) -> tuple[dict[str, pd.Series], pd.DatetimeIndex]:
    # The legacy builder already validated the entire table. Normalize keys on a
    # separate copy, retaining original scalars so unrelated prices cannot change
    # factor-window numeric conversion through a whole-column dtype coercion.
    raw = prices.loc[:, ["date", "symbol", "adjusted_close"]].copy(deep=True)
    normalized = raw.assign(
        date=_daily_dates(raw["date"], field="date"),
        symbol=_normalized_text(raw["symbol"]).str.upper(),
    )
    benchmark_dates = normalized.loc[normalized["symbol"] == benchmark_symbol, "date"]
    calendar = pd.DatetimeIndex(benchmark_dates.sort_values().unique())
    histories = {
        str(symbol): group.set_index("date")["adjusted_close"]
        for symbol, group in normalized.groupby("symbol", sort=False)
    }
    return histories, calendar


def _value_columns(factor: PriceFactorValue) -> dict[str, object]:
    name = factor.name
    return {
        name: factor.value,
        name + "_available_at": factor.as_of_date if factor.status == "ok" else pd.NaT,
        name + "_status": factor.status,
        name + "_window_start": factor.window_start,
        name + "_window_end": factor.window_end,
        name + "_required_price_count": factor.required_price_count,
        name + "_observed_price_count": factor.observed_price_count,
        name + "_formula_version": factor.formula_version,
    }


def _factor_rows(
    legacy: pd.DataFrame,
    histories: dict[str, pd.Series],
    calendar: pd.DatetimeIndex,
    factor_names: tuple[str, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in legacy.itertuples(index=False):
        result = compute_price_factors(
            histories.get(row.symbol, pd.Series(dtype=float)),
            calendar=calendar,
            as_of_date=row.as_of_date,
            factor_names=factor_names,
        )
        rows.append(
            {
                column: value
                for factor in result.factors
                for column, value in _value_columns(factor).items()
            }
        )
    return rows


def build_factor_panel(
    prices: pd.DataFrame,
    memberships: pd.DataFrame,
    *,
    as_of_dates: list[pd.Timestamp] | tuple[pd.Timestamp, ...],
    config: PanelConfig,
    factor_names: tuple[str, ...],
) -> pd.DataFrame:
    """Append requested factors without changing legacy rows, labels or columns.

    This adapter inherits the legacy builder's strict whole-table input checks,
    including future prices. It does not inherit the pure calculator's tolerance
    for malformed future values. Missing observations must be absent price rows,
    not explicit invalid cells in the normalized price table. Factor calculation
    uses every active row regardless of legacy eligibility or target availability.
    Availability is an after-close convention, not verified publication evidence.
    No feature is filtered, ranked, imputed, oriented or fitted by this adapter.
    """
    _requested_specs(factor_names)
    if not isinstance(config, PanelConfig):
        raise DataContractError("config must be a PanelConfig.")
    legacy = build_point_in_time_panel(
        prices, memberships, as_of_dates=as_of_dates, config=config
    )
    schema = _column_schema(factor_names)
    collisions = sorted(set(legacy.columns) & {column for column, _ in schema})
    if collisions:
        raise DataContractError(
            "Factor output columns collide with existing columns: "
            f"{', '.join(collisions)}"
        )
    histories, calendar = _raw_price_history(prices, config.benchmark_symbol)
    rows = _factor_rows(legacy, histories, calendar, factor_names)
    additional = pd.DataFrame(
        {
            column: pd.Series(
                [row[column] for row in rows], index=legacy.index, dtype=dtype
            )
            for column, dtype in schema
        },
        index=legacy.index,
    )
    return pd.concat([legacy, additional], axis=1, verify_integrity=True)
