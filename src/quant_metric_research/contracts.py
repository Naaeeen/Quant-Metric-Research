from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, timedelta

import numpy as np
import pandas as pd


class DataContractError(ValueError):
    """Raised when an input dataset violates the research data contract."""


def _require_columns(
    frame: pd.DataFrame,
    required: tuple[str, ...],
    *,
    dataset_name: str,
) -> None:
    if not frame.columns.is_unique:
        raise DataContractError(f"Duplicate {dataset_name} columns are not allowed.")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        joined = ", ".join(missing)
        raise DataContractError(f"Missing required {dataset_name} columns: {joined}")


def _normalized_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def _daily_date(value: object, *, field: str, nullable: bool) -> pd.Timestamp | None:
    message = (
        f"{field} contains invalid dates; expected normalized, "
        "timezone-naive calendar dates."
    )
    if not pd.api.types.is_scalar(value):
        raise DataContractError(message)
    if pd.isna(value):
        if nullable:
            return None
        raise DataContractError(message)
    # pandas interprets numbers as epoch offsets; that is not our daily contract.
    if not isinstance(value, (str, date, np.datetime64)):
        raise DataContractError(message)
    try:
        parsed = pd.Timestamp(value)
        if pd.isna(parsed) or parsed.tzinfo is not None or parsed != parsed.normalize():
            raise DataContractError(message)
    except (TypeError, ValueError, OverflowError) as error:
        raise DataContractError(message) from error
    return parsed


def _daily_dates(series: pd.Series, *, field: str, nullable: bool = False) -> pd.Series:
    parsed = [_daily_date(value, field=field, nullable=nullable) for value in series]
    try:
        return pd.Series(
            parsed, index=series.index, name=series.name, dtype="datetime64[ns]"
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise DataContractError(
            f"{field} contains invalid or out-of-range dates."
        ) from error


def validate_as_of_dates(values: Iterable[object]) -> tuple[pd.Timestamp, ...]:
    """Return ordered daily decision dates without repairing or mutating inputs."""
    if isinstance(values, (str, bytes, Mapping, pd.DataFrame)):
        raise DataContractError(
            "as_of_dates must be a nonempty sequence of daily dates."
        )
    try:
        supplied = tuple(values)
    except TypeError as error:
        raise DataContractError(
            "as_of_dates must be a nonempty sequence of daily dates."
        ) from error
    if not supplied:
        raise DataContractError("as_of_dates must not be empty.")
    parsed = _daily_dates(pd.Series(supplied, dtype=object), field="as_of_dates")
    dates = tuple(parsed)
    if len(set(dates)) != len(dates):
        raise DataContractError("as_of_dates must be unique.")
    return dates


def _positive_price_values(series: pd.Series) -> pd.Series:
    message = (
        "prices must contain finite, strictly positive real adjusted_close values."
    )
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
    # Numeric conversion can turn temporal dtypes into epoch/duration integers
    # and casting complex arrays to float silently discards their imaginary part.
    if series.dtype.kind in "bcmM" or any(
        isinstance(value, forbidden) for value in series
    ):
        raise DataContractError(message)
    try:
        parsed = pd.to_numeric(series, errors="coerce")
        if pd.api.types.is_complex_dtype(parsed.dtype):
            raise DataContractError(message)
        values = parsed.to_numpy(dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise DataContractError(message) from error
    if not np.isfinite(values).all() or (values <= 0).any():
        raise DataContractError(message)
    return parsed


def validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame):
        raise DataContractError("prices must be a pandas DataFrame.")

    required = ("date", "symbol", "adjusted_close")
    _require_columns(prices, required, dataset_name="price")
    validated = prices.copy(deep=True)
    validated["date"] = _daily_dates(validated["date"], field="date")
    validated["symbol"] = _normalized_text(validated["symbol"]).str.upper()
    validated["adjusted_close"] = _positive_price_values(validated["adjusted_close"])

    required_values = validated.loc[:, list(required)]
    if required_values.isna().any().any() or (validated["symbol"] == "").any():
        raise DataContractError("prices contains invalid required values.")
    if validated.duplicated(["date", "symbol"], keep=False).any():
        raise DataContractError("Duplicate symbol-date rows found in prices.")

    return validated.sort_values(
        ["date", "symbol"],
        kind="stable",
    ).reset_index(drop=True)


def validate_memberships(memberships: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(memberships, pd.DataFrame):
        raise DataContractError("memberships must be a pandas DataFrame.")

    required = (
        "universe_id",
        "symbol",
        "effective_from",
        "effective_to",
        "source",
    )
    _require_columns(memberships, required, dataset_name="membership")
    validated = memberships.copy(deep=True)
    validated["universe_id"] = _normalized_text(validated["universe_id"])
    validated["symbol"] = _normalized_text(validated["symbol"]).str.upper()
    validated["source"] = _normalized_text(validated["source"])
    validated["effective_from"] = _daily_dates(
        validated["effective_from"],
        field="effective_from",
    )
    validated["effective_to"] = _daily_dates(
        validated["effective_to"],
        field="effective_to",
        nullable=True,
    )

    required_non_null = ["universe_id", "symbol", "effective_from", "source"]
    if validated[required_non_null].isna().any().any():
        raise DataContractError("memberships contains invalid required values.")
    for column in ("universe_id", "symbol", "source"):
        if (validated[column] == "").any():
            raise DataContractError("memberships contains empty required text.")

    ordered = validated.sort_values(
        ["universe_id", "symbol", "effective_from"],
        kind="stable",
    ).reset_index(drop=True)
    grouped = ordered.groupby(["universe_id", "symbol"], sort=False)
    for _, group in grouped:
        previous_end: pd.Timestamp | None = None
        for position, row in enumerate(group.itertuples(index=False)):
            start = pd.Timestamp(row.effective_from)
            end = pd.Timestamp(row.effective_to) if pd.notna(row.effective_to) else None
            if end is not None and end <= start:
                raise DataContractError("effective_to must be after effective_from.")
            if position > 0 and (previous_end is None or start < previous_end):
                raise DataContractError(
                    "membership intervals overlap for the same symbol."
                )
            previous_end = end

    return ordered
