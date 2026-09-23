"""Pure, opt-in single-security factors with explicit daily-window coverage."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import fsum, isfinite
from typing import Literal

import pandas as pd

from .contracts import (
    DataContractError,
    _daily_date,
    _positive_price_values,
    validate_as_of_dates,
)
from .price_factor_catalog import PRICE_FACTOR_CATALOG, PriceFactorSpec


@dataclass(frozen=True, slots=True)
class PriceFactorValue:
    """One raw factor and its required inclusive source-window diagnostics."""

    name: str
    formula_version: str
    formula: str
    as_of_date: pd.Timestamp
    window_start: pd.Timestamp | None
    window_end: pd.Timestamp | None
    required_price_count: int
    observed_price_count: int
    value: float | None
    status: Literal["insufficient_history", "missing_required_prices", "ok"]


@dataclass(frozen=True, slots=True)
class PriceFactorResult:
    as_of_date: pd.Timestamp
    factors: tuple[PriceFactorValue, ...]


def _requested_specs(factor_names: tuple[str, ...]) -> tuple[PriceFactorSpec, ...]:
    if (
        not isinstance(factor_names, tuple)
        or not factor_names
        or any(not isinstance(name, str) or not name for name in factor_names)
        or len(set(factor_names)) != len(factor_names)
        or any(name not in PRICE_FACTOR_CATALOG for name in factor_names)
    ):
        raise DataContractError(
            "factor_names must be a nonempty tuple of unique catalog names."
        )
    return tuple(PRICE_FACTOR_CATALOG[name] for name in factor_names)


def _calendar_and_cutoff(
    calendar: Iterable[object], as_of_date: object
) -> tuple[pd.DatetimeIndex, pd.Timestamp]:
    supplied_calendar = pd.DatetimeIndex(validate_as_of_dates(calendar))
    if not supplied_calendar.is_monotonic_increasing:
        raise DataContractError("calendar must be strictly increasing and unique.")
    cutoff = _daily_date(as_of_date, field="as_of_date", nullable=False)
    if cutoff not in supplied_calendar:
        raise DataContractError("as_of_date must be an exact calendar member.")
    return supplied_calendar, pd.Timestamp(cutoff)


def _aligned_history(
    stock_prices: pd.Series, calendar: pd.DatetimeIndex, cutoff: pd.Timestamp
) -> pd.Series:
    if not isinstance(stock_prices, pd.Series):
        raise DataContractError(
            "stock_prices must be a pandas Series with a date index."
        )
    dates = pd.DatetimeIndex(
        validate_as_of_dates(stock_prices.index) if len(stock_prices) else ()
    )
    # Validate every date, but never inspect future price values. This assumes
    # appending future values has not already changed historical scalar types.
    historical = stock_prices.set_axis(dates).loc[dates <= cutoff].copy(deep=True)
    in_span = historical.index >= calendar[0]
    if (in_span & ~historical.index.isin(calendar)).any():
        raise DataContractError("Historical prices contain off-calendar dates.")
    observed = historical.loc[historical.notna()]
    # Validate globally, but convert only the required factor window later.
    # Otherwise an excluded decimal string can change the conversion precision
    # of required integer strings. An all-NA dtype remains an ordinary absence.
    if not observed.empty:
        _positive_price_values(observed)
    return historical.reindex(calendar)


def _factor_number(values: pd.Series, spec: PriceFactorSpec) -> float:
    prices = values.to_numpy(dtype=float)
    if spec.name == "ma_distance_63s":
        # Work in scaled units: an ordinary mean can overflow on valid prices,
        # while dividing by its infinite result would silently produce -1.
        scale = float(max(prices))
        scaled_mean = fsum(float(price) / scale for price in prices) / len(prices)
        result = (float(prices[-1]) / scale) / scaled_mean - 1.0
    else:
        result = float(prices[-1]) / float(prices[0]) - 1.0
    if not isfinite(result):
        raise DataContractError(f"{spec.name} produced a nonfinite factor value.")
    return result


def _calculate_factor(
    aligned: pd.Series,
    calendar: pd.DatetimeIndex,
    cutoff: pd.Timestamp,
    position: int,
    spec: PriceFactorSpec,
) -> PriceFactorValue:
    start = position - spec.start_lag_sessions
    end = position - spec.end_lag_sessions
    window = aligned.iloc[max(0, start) : end + 1] if end >= 0 else aligned.iloc[:0]
    observed_count = int(window.notna().sum())
    status: Literal["insufficient_history", "missing_required_prices", "ok"]
    if start < 0:
        status = "insufficient_history"
    elif observed_count < spec.required_price_count:
        status = "missing_required_prices"
    else:
        status = "ok"
    return PriceFactorValue(
        name=spec.name,
        formula_version=spec.formula_version,
        formula=spec.formula,
        as_of_date=cutoff,
        window_start=pd.Timestamp(calendar[start]) if start >= 0 else None,
        window_end=pd.Timestamp(calendar[end]) if end >= 0 else None,
        required_price_count=spec.required_price_count,
        observed_price_count=observed_count,
        value=_factor_number(window, spec) if status == "ok" else None,
        status=status,
    )


def compute_price_factors(
    stock_prices: pd.Series,
    *,
    calendar: Iterable[object],
    as_of_date: object,
    factor_names: tuple[str, ...],
) -> PriceFactorResult:
    """Calculate explicitly requested price factors without fitting or I/O.

    Calendar and price dates must be normalized, timezone-naive and unique;
    only the calendar must already be sorted. All historical nonmissing prices
    must be finite positive real values, even outside requested factor windows.
    Actual NA values remain gaps; numeric strings are supported but NA-like text
    is not. Every price in each inclusive factor interval is required. Future
    values are ignored after validating all date structure. Supplied calendar,
    historical availability and adjusted-price provenance are not certified.
    """
    specs = _requested_specs(factor_names)
    normalized_calendar, cutoff = _calendar_and_cutoff(calendar, as_of_date)
    aligned = _aligned_history(stock_prices, normalized_calendar, cutoff)
    position = int(normalized_calendar.get_loc(cutoff))
    return PriceFactorResult(
        as_of_date=cutoff,
        factors=tuple(
            _calculate_factor(aligned, normalized_calendar, cutoff, position, spec)
            for spec in specs
        ),
    )
