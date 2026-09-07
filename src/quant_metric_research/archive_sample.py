"""Offline normalization of the pinned Mendeley daily-price teaching sample."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from quant_metric_research.contracts import (
    DataContractError,
    validate_as_of_dates,
    validate_prices,
)

BENCHMARK_SYMBOL = "FF_MARKET_PROXY"
SOURCE_START = pd.Timestamp("2012-01-03")
SOURCE_END = pd.Timestamp("2016-12-30")
_MOVE_THRESHOLD = 0.5
_MAX_MOVE_EXAMPLES = 20
_SELECTION_RULE = (
    "First normalized symbols alphabetically across ALL stock columns; "
    "no coverage or performance filtering."
)
_BENCHMARK_NOTE = (
    "Constructed research index: first included date base 100; subsequent levels "
    "compound 1 + (Mkt-RF + RF)/100 from percentage daily returns."
)
_LIMITATIONS = (
    "Fixed S&P 500 constituent snapshot as of 2017-01-06; survivorship bias "
    "and no point-in-time membership.",
    "Publisher describes adjusted close; corporate-action completeness and "
    "adjustment factors are independently unverified.",
    "Ticker labels are not stable security identifiers; "
    "delisting outcomes are unverified.",
    "Original Yahoo source and archive CC BY 4.0 declaration do not "
    "independently certify third-party rights.",
    "Availability timestamps, revision history, exchange calendar and "
    "execution assumptions are unverified.",
    "Benchmark is a constructed research-market return proxy, "
    "not a traded ETF or official S&P 500 index.",
    "Coverage and large-move diagnostics are not evidence of alpha or tradability.",
)


@dataclass(frozen=True)
class ArchiveSample:
    prices: pd.DataFrame
    missing_prices: pd.DataFrame
    symbols: tuple[str, ...]
    profile: dict[str, object]


def _validate_frame(
    frame: pd.DataFrame, required: tuple[str, ...], *, name: str
) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise DataContractError(f"{name} must be a pandas DataFrame.")
    if isinstance(frame.columns, pd.MultiIndex) or not frame.columns.is_unique:
        raise DataContractError(
            f"Duplicate or non-flat {name} columns are not allowed."
        )
    if any(column not in frame.columns for column in required):
        raise DataContractError(f"{name} is missing required columns: {required}.")


def _source_dates(series: pd.Series, *, factor: bool = False) -> pd.Series:
    pattern = r"\d{1,2}/\d{1,2}/\d{4}" if factor else r"\d{4}-\d{2}-\d{2}"
    date_format = "%m/%d/%Y" if factor else "%Y-%m-%d"
    parsed = []
    for value in series:
        if isinstance(value, str):
            if not re.fullmatch(pattern, value):
                raise DataContractError(f"Date must use source format {date_format}.")
            try:
                value = datetime.strptime(value, date_format)
            except ValueError as error:
                raise DataContractError(
                    "Date contains invalid calendar dates."
                ) from error
        parsed.append(value)
    dates = validate_as_of_dates(parsed)
    return pd.Series(dates, index=series.index, name="Date", dtype="datetime64[ns]")


def _stock_window(
    stock_prices: pd.DataFrame, cohort_size: int
) -> tuple[pd.DataFrame, tuple[str, ...], int]:
    _validate_frame(stock_prices, ("Date",), name="stock prices")
    if isinstance(cohort_size, bool) or not isinstance(cohort_size, (int, np.integer)):
        raise DataContractError("cohort_size must be a positive integer.")
    headers = tuple(column for column in stock_prices.columns if column != "Date")
    if any(not isinstance(column, str) or not column.strip() for column in headers):
        raise DataContractError("Stock symbols must be nonempty text.")
    aliases = {column: column.strip().upper() for column in headers}
    normalized = tuple(aliases.values())
    if len(set(normalized)) != len(normalized):
        raise DataContractError("Stock symbol aliases normalize to the same symbol.")
    if BENCHMARK_SYMBOL in normalized:
        raise DataContractError("Stock symbol collision with the benchmark symbol.")
    if not 1 <= cohort_size <= len(normalized):
        raise DataContractError("cohort_size exceeds the available stock columns.")
    symbols = tuple(sorted(normalized)[:cohort_size])
    dates = _source_dates(stock_prices["Date"])
    selected = stock_prices.rename(columns=aliases).assign(Date=dates)
    window = selected.loc[dates.between(SOURCE_START, SOURCE_END), ["Date", *symbols]]
    if window.empty:
        raise DataContractError("Stock prices have no dates in the source window.")
    ordered = window.reset_index(drop=True).sort_values("Date").reset_index(drop=True)
    return ordered, symbols, len(normalized)


def _factor_values(series: pd.Series) -> np.ndarray:
    forbidden = (
        bool,
        np.bool_,
        date,
        timedelta,
        np.datetime64,
        np.timedelta64,
        complex,
        np.complexfloating,
    )
    if series.dtype.kind in "bcmM" or any(isinstance(v, forbidden) for v in series):
        raise DataContractError("Factor returns must be finite real numbers.")
    try:
        values = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise DataContractError(
            "Factor returns must be finite real numbers."
        ) from error
    if not np.isfinite(values).all():
        raise DataContractError("Factor returns must be finite and nonmissing.")
    # Compare numpy floats at their source precision before float64 widening.
    # For example float32(-99.99) becomes -99.98999786376953 after widening.
    source_sentinel = any(
        isinstance(value, np.floating)
        and any(value == type(value)(sentinel) for sentinel in (-99.99, -999.0))
        for value in series.to_numpy()
    )
    if source_sentinel or np.isin(values, [-99.99, -999.0]).any():
        raise DataContractError("Factor returns contain a missing-data sentinel.")
    return values


def _benchmark(factor_returns: pd.DataFrame, stock_dates: pd.Series) -> pd.DataFrame:
    _validate_frame(factor_returns, ("Date", "Mkt-RF", "RF"), name="factor returns")
    dates = _source_dates(factor_returns["Date"], factor=True)
    # Validate all raw component values, including the first base day and dates
    # outside the sample window, before restriction or compounding.
    market_excess = _factor_values(factor_returns["Mkt-RF"])
    risk_free = _factor_values(factor_returns["RF"])
    with np.errstate(over="ignore", invalid="ignore"):
        returns = (market_excess + risk_free) / 100.0
    if not np.isfinite(returns).all() or (returns <= -1.0).any():
        raise DataContractError("Market returns must be finite and greater than -100%.")
    factor_window = (
        pd.DataFrame({"date": dates.to_numpy(), "return": returns})
        .loc[dates.between(SOURCE_START, SOURCE_END).to_numpy()]
        .sort_values("date")
    )
    if factor_window.empty or not stock_dates.isin(factor_window["date"]).all():
        raise DataContractError("The benchmark calendar is missing stock dates.")
    subsequent_gross = 1.0 + factor_window["return"].to_numpy()[1:]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        levels = 100.0 * np.cumprod(np.concatenate(([1.0], subsequent_gross)))
    return validate_prices(
        pd.DataFrame(
            {
                "date": factor_window["date"].to_numpy(),
                "symbol": BENCHMARK_SYMBOL,
                "adjusted_close": levels,
            }
        )
    )


def _split_prices(window: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    melted = window.rename(columns={"Date": "date"}).melt(
        id_vars="date", var_name="symbol", value_name="adjusted_close"
    )
    null_prices = melted["adjusted_close"].isna()
    missing = (
        melted.loc[null_prices, ["date", "symbol"]]
        .assign(reason="missing_adjusted_close")
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )
    present = validate_prices(melted.loc[~null_prices])
    return present, missing


def _coverage_rows(
    panel: pd.DataFrame, *, year: int | None = None
) -> list[dict[str, object]]:
    return [
        {
            "symbol": symbol,
            **({"year": year} if year is not None else {}),
            "expected_count": len(panel),
            "observed_count": int(panel[symbol].notna().sum()),
            "missing_count": int(panel[symbol].isna().sum()),
            "missing_rate": float(panel[symbol].isna().mean()),
        }
        for symbol in panel.columns
    ]


def _large_moves(panel: pd.DataFrame) -> dict[str, object]:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        changes = panel.astype(float).pct_change(fill_method=None)
    finite = np.isfinite(changes)
    large = changes.where(finite & changes.abs().gt(_MOVE_THRESHOLD))
    examples = (
        large.rename_axis(index="date", columns="symbol")
        .stack()
        .dropna()
        .rename("simple_return")
        .reset_index()
    )
    return {
        "absolute_return_threshold": _MOVE_THRESHOLD,
        "count": len(examples),
        "nonfinite_derived_return_count": int((changes.notna() & ~finite).sum().sum()),
        "examples": [
            {
                "date": row.date.date().isoformat(),
                "symbol": row.symbol,
                "simple_return": float(row.simple_return),
            }
            for row in examples.head(_MAX_MOVE_EXAMPLES).itertuples(index=False)
        ],
        "examples_limit": _MAX_MOVE_EXAMPLES,
        "policy": "Diagnostics only; no clipping or bridging missing sessions.",
    }


def _profile(
    stock_prices: pd.DataFrame,
    factor_returns: pd.DataFrame,
    window: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark: pd.DataFrame,
    symbols: tuple[str, ...],
    source_symbol_count: int,
) -> dict[str, object]:
    panel = prices.pivot(
        index="date", columns="symbol", values="adjusted_close"
    ).reindex(index=benchmark["date"], columns=list(symbols))
    return {
        "claim_scope": "archived_real_data_engineering_demo",
        "source_window": {
            "start": SOURCE_START.date().isoformat(),
            "end": SOURCE_END.date().isoformat(),
        },
        "source_stock_row_count": len(stock_prices),
        "source_factor_row_count": len(factor_returns),
        "source_symbol_count": source_symbol_count,
        "stock_date_count": len(window),
        "benchmark_date_count": len(benchmark),
        "factor_dates_without_stock_rows": int(
            (~benchmark["date"].isin(window["Date"])).sum()
        ),
        "selected_symbols": list(symbols),
        "selection_rule": _SELECTION_RULE,
        "coverage_by_symbol": _coverage_rows(panel),
        "coverage_by_symbol_year": [
            row
            for year, group in panel.groupby(panel.index.year)
            for row in _coverage_rows(group, year=int(year))
        ],
        "large_daily_moves": _large_moves(panel),
        "benchmark": {
            "symbol": BENCHMARK_SYMBOL,
            "construction": _BENCHMARK_NOTE,
            "tradable": False,
            "official_sp500_index": False,
        },
        "limitations": list(_LIMITATIONS),
        "empirical_data_provenance_verified": False,
        "stage4_eligible": False,
    }


def normalize_mendeley_prices(
    stock_prices: pd.DataFrame,
    factor_returns: pd.DataFrame,
    *,
    cohort_size: int = 30,
) -> ArchiveSample:
    """Normalize source-format frames without downloading or certifying the data.

    Null selected prices remain in ``missing_prices``; selected members are never
    replaced. Dates and present prices are validated before missingness is split.
    CSV strings use the pinned files' explicit date formats. Already parsed daily
    date objects are accepted under the shared daily-date contract.
    """
    window, symbols, source_symbol_count = _stock_window(stock_prices, cohort_size)
    benchmark = _benchmark(factor_returns, window["Date"])
    prices, missing = _split_prices(window)
    profile = _profile(
        stock_prices,
        factor_returns,
        window,
        prices,
        benchmark,
        symbols,
        source_symbol_count,
    )
    return ArchiveSample(
        prices=validate_prices(pd.concat([prices, benchmark], ignore_index=True)),
        missing_prices=missing,
        symbols=symbols,
        profile=profile,
    )
