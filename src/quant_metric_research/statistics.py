from __future__ import annotations

from math import erf, sqrt

import numpy as np
import pandas as pd


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def newey_west_mean_tstat(
    values: pd.Series | list[float],
    hac_lags: int,
) -> tuple[float | None, float | None]:
    if isinstance(hac_lags, bool) or not isinstance(hac_lags, int) or hac_lags < 0:
        raise ValueError("hac_lags must be a non-negative integer.")

    series = (
        pd.Series(values, dtype="float64")
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )
    sample_size = int(series.shape[0])
    if sample_size < 2:
        return None, None

    max_lag = min(hac_lags, sample_size - 1)
    demeaned = series.to_numpy(dtype=float) - float(series.mean())
    gamma_zero = float(np.dot(demeaned, demeaned) / sample_size)
    long_run_variance = gamma_zero
    for lag in range(1, max_lag + 1):
        covariance = float(np.dot(demeaned[lag:], demeaned[:-lag]) / sample_size)
        bartlett_weight = 1.0 - lag / (max_lag + 1)
        long_run_variance += 2.0 * bartlett_weight * covariance

    standard_error = sqrt(max(long_run_variance, 0.0) / sample_size)
    if standard_error == 0.0:
        return None, None
    t_stat = float(series.mean() / standard_error)
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
