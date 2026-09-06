from __future__ import annotations

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
    missing = [column for column in required if column not in frame.columns]
    if missing:
        joined = ", ".join(missing)
        raise DataContractError(f"Missing required {dataset_name} columns: {joined}")


def _normalized_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame):
        raise DataContractError("prices must be a pandas DataFrame.")

    required = ("date", "symbol", "adjusted_close")
    _require_columns(prices, required, dataset_name="price")
    validated = prices.copy(deep=True)
    validated["date"] = pd.to_datetime(validated["date"], errors="coerce")
    validated["symbol"] = _normalized_text(validated["symbol"]).str.upper()
    validated["adjusted_close"] = pd.to_numeric(
        validated["adjusted_close"],
        errors="coerce",
    )

    required_values = validated.loc[:, list(required)]
    if required_values.isna().any().any() or (validated["symbol"] == "").any():
        raise DataContractError("prices contains invalid required values.")
    values = validated["adjusted_close"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise DataContractError(
            "prices must contain finite, strictly positive adjusted_close values."
        )
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
    validated["effective_from"] = pd.to_datetime(
        validated["effective_from"],
        errors="coerce",
    )
    originally_missing_end = validated["effective_to"].isna()
    validated["effective_to"] = pd.to_datetime(
        validated["effective_to"],
        errors="coerce",
    )
    if (validated["effective_to"].isna() & ~originally_missing_end).any():
        raise DataContractError("effective_to contains invalid non-null dates.")

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
