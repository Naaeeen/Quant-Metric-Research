from __future__ import annotations

from collections.abc import Mapping
from datetime import date, timedelta
from math import erf, isfinite, sqrt

import numpy as np
import pandas as pd

from .contracts import DataContractError


def _validate_hac_lags(hac_lags: int) -> None:
    if isinstance(hac_lags, bool) or not isinstance(hac_lags, int) or hac_lags < 0:
        raise ValueError("hac_lags must be a non-negative integer.")


def _numeric_observations(
    values: pd.Series | list[float], *, allow_nonfinite: bool = False
) -> pd.Series:
    """Validate original scalars before float64 conversion; preserve NA positions."""
    message = "values must contain real numeric observations or actual missing values."
    if isinstance(values, (str, bytes, Mapping, pd.DataFrame)) or np.isscalar(values):
        raise DataContractError(message)
    try:
        original = (
            values if isinstance(values, pd.Series) else pd.Series(values, dtype=object)
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise DataContractError(message) from error
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
    normalized = []
    for value in original:
        if not pd.api.types.is_scalar(value):
            raise DataContractError(message)
        if pd.isna(value):
            normalized.append(np.nan)
            continue
        if isinstance(value, forbidden):
            raise DataContractError(message)
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise DataContractError(message) from error
        if not isfinite(number) and (not allow_nonfinite or isinstance(value, str)):
            raise DataContractError(message)
        normalized.append(value)
    # Keep the legacy float64 constructor path, not pd.to_numeric: decimal and
    # large integer strings can otherwise acquire subtly different rounding.
    return pd.Series(
        normalized, index=original.index, name=original.name, dtype="float64"
    )


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def newey_west_mean_tstat(
    values: pd.Series | list[float],
    hac_lags: int,
) -> tuple[float | None, float | None]:
    """HAC mean inference for an already contiguous ordered sequence.

    Missing/nonfinite observations and unsupported lags withhold inference;
    they are never dropped or truncated. Use scheduled_newey_west_mean when
    expected dates are available. Neither API certifies sampling assumptions.
    """
    _validate_hac_lags(hac_lags)
    series = _numeric_observations(values, allow_nonfinite=True)
    sample_size = int(series.shape[0])
    if (
        sample_size < 2
        or hac_lags > sample_size - 1
        or not np.isfinite(series.to_numpy()).all()
    ):
        return None, None
    return _complete_newey_west_mean(series, hac_lags)


def _complete_newey_west_mean(
    series: pd.Series, hac_lags: int
) -> tuple[float | None, float | None]:
    """Legacy Bartlett arithmetic after complete-input and lag-support checks."""
    sample_size = len(series)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        mean = float(series.mean())
        demeaned = series.to_numpy(dtype=float) - mean
        long_run_variance = float(np.dot(demeaned, demeaned) / sample_size)
        for lag in range(1, hac_lags + 1):
            covariance = float(np.dot(demeaned[lag:], demeaned[:-lag]) / sample_size)
            bartlett_weight = 1.0 - lag / (hac_lags + 1)
            long_run_variance += 2.0 * bartlett_weight * covariance
        variance_of_mean = long_run_variance / sample_size
    if not isfinite(variance_of_mean) or variance_of_mean <= 0.0:
        return None, None
    t_stat = float(mean / sqrt(variance_of_mean))
    if not isfinite(t_stat):
        return None, None
    p_value = float(2.0 * (1.0 - _normal_cdf(abs(t_stat))))
    return t_stat, min(max(p_value, 0.0), 1.0)


def benjamini_hochberg(
    p_values: list[float] | np.ndarray | pd.Series,
) -> list[float]:
    series = pd.Series(p_values, dtype="float64")
    finite = series.dropna()
    if ((finite < 0.0) | (finite > 1.0)).any():
        raise ValueError("p-values must be between zero and one.")
    if finite.empty:
        return [float("nan") for _ in range(len(series))]

    ordered = finite.sort_values()
    count = int(ordered.shape[0])
    adjusted = pd.Series(index=ordered.index, dtype="float64")
    running_minimum = 1.0
    for reverse_position, (index, p_value) in enumerate(
        ordered.iloc[::-1].items(),
        start=1,
    ):
        rank = count - reverse_position + 1
        running_minimum = min(
            running_minimum,
            float(p_value) * count / rank,
        )
        adjusted.at[index] = min(running_minimum, 1.0)

    result = pd.Series(np.nan, index=series.index, dtype="float64")
    result.loc[adjusted.index] = adjusted
    return result.tolist()
