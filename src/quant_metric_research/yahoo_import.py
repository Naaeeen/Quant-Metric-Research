"""Offline normalization for a single flat Yahoo daily adjusted-price export."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .contracts import DataContractError, validate_as_of_dates, validate_prices


@dataclass(frozen=True)
class YahooPriceExport:
    """Validated prices and explicit gaps; neither table certifies provenance."""

    prices: pd.DataFrame
    missing_prices: pd.DataFrame


def _validate_export_shape(frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise DataContractError("Yahoo export must be a pandas DataFrame.")
    if isinstance(frame.columns, pd.MultiIndex):
        raise DataContractError(
            "Yahoo export must have flat columns; yfinance MultiIndex exports "
            "are not supported."
        )
    if not frame.columns.is_unique:
        raise DataContractError("Duplicate Yahoo export columns are not allowed.")
    if isinstance(frame.index, pd.DatetimeIndex):
        raise DataContractError(
            "Yahoo export requires a Date column, not a DatetimeIndex."
        )
    missing = [column for column in ("Date", "Adj Close") if column not in frame]
    if missing:
        raise DataContractError(
            f"Missing required Yahoo export columns: {', '.join(missing)}. "
            "Close is never substituted for Adj Close."
        )


def normalize_yahoo_export(frame: pd.DataFrame, *, symbol: str) -> YahooPriceExport:
    """Normalize one supplied daily export without downloading or repairing data.

    Only the explicit ``Adj Close`` column supplies prices. Missing observations
    remain in ``missing_prices``; malformed non-null values fail the contract.
    Empty exports retain both output schemas so callers can audit their absence.
    Yahoo symbols are labels here, not proof of stable security identifiers.
    """
    _validate_export_shape(frame)
    if not isinstance(symbol, str) or not symbol.strip():
        raise DataContractError("Yahoo export symbol must be a nonempty string.")
    normalized_symbol = symbol.strip().upper()
    # Validate every date before separating gaps so missing prices cannot hide
    # duplicate dates or invalid timestamps.
    dates = validate_as_of_dates(frame["Date"]) if len(frame) else ()
    normalized = pd.DataFrame(
        {
            "date": pd.Series(dates, dtype="datetime64[ns]"),
            "symbol": pd.Series([normalized_symbol] * len(frame), dtype="string"),
            "adjusted_close": frame["Adj Close"].copy(deep=True).reset_index(drop=True),
        }
    )
    missing = normalized["adjusted_close"].isna()
    prices = validate_prices(normalized.loc[~missing].copy(deep=True))
    missing_prices = (
        normalized.loc[missing, ["date", "symbol"]]
        .assign(reason="missing_adjusted_close")
        .sort_values(["date", "symbol"], kind="stable")
        .reset_index(drop=True)
    )
    return YahooPriceExport(prices=prices, missing_prices=missing_prices)
