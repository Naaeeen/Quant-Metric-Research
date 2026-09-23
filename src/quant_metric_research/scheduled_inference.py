"""Fail-closed HAC inference against an explicit, uncompressed date schedule."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from .contracts import DataContractError, validate_as_of_dates
from .statistics import (
    _complete_newey_west_mean,
    _numeric_observations,
    _validate_hac_lags,
)


@dataclass(frozen=True, slots=True)
class HACMeanResult:
    """Nullable inference and explicit schedule/lag coverage diagnostics."""

    status: Literal[
        "ok",
        "schedule_unavailable",
        "missing_scheduled_values",
        "insufficient_observations",
        "insufficient_lag_support",
        "undefined_variance",
    ]
    t_stat: float | None
    p_value: float | None
    scheduled_count: int | None
    observed_count: int
    requested_lags: int
    effective_lags: int | None
    lag_unit: Literal["scheduled_observations"] = "scheduled_observations"


def normalize_expected_dates(expected_dates: Sequence[object]) -> pd.DatetimeIndex:
    """Validate a nonempty, unique, increasing sequence of naive daily dates.

    No dates are sorted, snapped, deduplicated or inferred. This validates the
    supplied schedule, not an exchange calendar or equally spaced observations.
    """
    if isinstance(expected_dates, (set, frozenset)):
        raise DataContractError("expected_dates must be an ordered date sequence.")
    try:
        dates = pd.DatetimeIndex(validate_as_of_dates(expected_dates))
    except DataContractError as error:
        raise DataContractError(f"Invalid expected_dates: {error}") from error
    if not dates.is_monotonic_increasing:
        raise DataContractError("expected_dates must be strictly increasing.")
    return dates


def inference_diagnostics(result: HACMeanResult) -> dict[str, str | int | None]:
    """Return fresh scalar summary metadata without changing point estimates."""
    return {
        "inference_status": result.status,
        "scheduled_date_count": result.scheduled_count,
        "observed_date_count": result.observed_count,
        "requested_hac_lags": result.requested_lags,
        "effective_hac_lags": result.effective_lags,
        "hac_lag_unit": result.lag_unit,
    }


def scheduled_newey_west_mean(
    values: pd.Series, *, expected_dates: Sequence[object], hac_lags: int
) -> HACMeanResult:
    """Compute only on complete schedules with supported lags and finite variance.

    Unsorted values are aligned to the supplied schedule without filling or
    compressing gaps. Every present scalar and every date is validated before
    reporting availability. Small-sample or stationarity validity is not proven.
    Effective lags are reported only when finite inference is available.
    """
    _validate_hac_lags(hac_lags)
    schedule = normalize_expected_dates(expected_dates)
    if not isinstance(values, pd.Series):
        raise DataContractError("values must be a pandas Series with a date index.")
    dates = pd.DatetimeIndex(validate_as_of_dates(values.index) if len(values) else ())
    if not dates.isin(schedule).all():
        raise DataContractError("values contains dates outside expected_dates.")
    aligned = _numeric_observations(values).set_axis(dates).reindex(schedule)
    scheduled_count, observed_count = len(schedule), int(aligned.notna().sum())
    status, t_stat, p_value, effective_lags = "ok", None, None, None
    if observed_count != scheduled_count:
        status = "missing_scheduled_values"
    elif scheduled_count < 2:
        status = "insufficient_observations"
    elif hac_lags > scheduled_count - 1:
        status = "insufficient_lag_support"
    else:
        t_stat, p_value = _complete_newey_west_mean(aligned, hac_lags)
        if t_stat is None:
            status = "undefined_variance"
        else:
            effective_lags = hac_lags
    return HACMeanResult(
        status,
        t_stat,
        p_value,
        scheduled_count,
        observed_count,
        hac_lags,
        effective_lags,
    )
