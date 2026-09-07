from __future__ import annotations

from datetime import date, timedelta
from math import erf, sqrt

import numpy as np
import pandas as pd
import pytest

from quant_metric_research.statistics import (
    benjamini_hochberg,
    newey_west_mean_tstat,
)


def old_complete_hac(values, hac_lags):
    """Frozen pre-hardening arithmetic, used only on finite supported samples."""
    series = (
        pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    )
    count = len(series)
    max_lag = min(hac_lags, count - 1)
    demeaned = series.to_numpy(dtype=float) - float(series.mean())
    variance = float(np.dot(demeaned, demeaned) / count)
    for lag in range(1, max_lag + 1):
        covariance = float(np.dot(demeaned[lag:], demeaned[:-lag]) / count)
        variance += 2.0 * (1.0 - lag / (max_lag + 1)) * covariance
    error = sqrt(max(variance, 0.0) / count)
    if error == 0:
        return None, None
    t_stat = float(series.mean() / error)
    cdf = 0.5 * (1.0 + erf(abs(t_stat) / sqrt(2.0)))
    return t_stat, min(max(float(2.0 * (1.0 - cdf)), 0.0), 1.0)


@pytest.mark.parametrize("lags", [0, 1, 3, 7])
@pytest.mark.parametrize("kind", ["float64", "float32", "Int64", "string"])
def test_complete_supported_hac_has_exact_old_numerical_parity(lags, kind):
    values = pd.Series([4, -3, 2, 8, -1, 2, 3, 0], dtype=kind)
    before = values.copy(deep=True)
    assert newey_west_mean_tstat(values, lags) == old_complete_hac(values, lags)
    pd.testing.assert_series_equal(values, before)


@pytest.mark.parametrize("missing", [np.nan, None, pd.NA, np.inf, -np.inf])
def test_tuple_does_not_compress_missing_or_nonfinite_values(missing):
    assert newey_west_mean_tstat([0.1, missing, 0.2, -0.1], 1) == (None, None)


@pytest.mark.parametrize("lags", [3, 4, 100])
def test_tuple_does_not_truncate_unsupported_lags(lags):
    assert newey_west_mean_tstat([0.1, 0.4, -0.2], lags) == (None, None)


@pytest.mark.parametrize(
    "values",
    [
        [],
        [0.2],
        [0, 0, 0],
        [2, 2, 2],
        [1e308, -1e308, 1e308],
        [1e308, 1e308],
        [1e-300, -1e-300],
    ],
)
def test_tuple_fails_closed_for_insufficient_or_undefined_variance(values):
    with np.errstate(all="raise"):
        assert newey_west_mean_tstat(values, 0) == (None, None)


@pytest.mark.parametrize(
    "bad",
    [
        True,
        np.bool_(False),
        "NaN",
        "",
        "NA",
        "bad",
        1 + 0j,
        np.complex128(2),
        date(2025, 1, 1),
        timedelta(days=1),
        np.datetime64("2025-01-01"),
        np.timedelta64(1, "D"),
        [1],
        {"x": 1},
    ],
)
def test_tuple_rejects_malformed_present_values(bad):
    values = pd.Series([0.1, bad, 0.3], dtype=object)
    with pytest.raises(ValueError, match="real|numeric"):
        newey_west_mean_tstat(values, 0)


@pytest.mark.parametrize("bad", [True, np.int64(1), -1, 1.0, None, "1"])
def test_tuple_rejects_invalid_lag_parameter(bad):
    with pytest.raises(ValueError, match="hac_lags"):
        newey_west_mean_tstat([1, 2, 3], bad)


def test_bh_preserves_unavailable_inference():
    adjusted = benjamini_hochberg([None, 0.01, np.nan, 0.04])
    assert np.isnan(adjusted[0]) and np.isnan(adjusted[2])
    assert adjusted[1] == 0.02 and adjusted[3] == 0.04


@pytest.mark.parametrize(
    "values",
    [
        [
            "9007199254740993",
            "9007199254740999",
            "-9007199254740901",
            "0.12345678901234567",
        ],
        ["17.237272018387371", "-23.322677674561464", "33.24147174324341", "-1"],
        [1e100, 1e100 + 1e90, 1e100 - 1e90, 1e100],
        [-1.0, 1.0, -1.0, 1.0],
    ],
)
@pytest.mark.parametrize("lags", [0, 1, 3])
def test_precision_and_tail_probability_boundaries_retain_exact_old_result(
    values, lags
):
    assert newey_west_mean_tstat(values, lags) == old_complete_hac(values, lags)


@pytest.mark.parametrize(
    "dtype",
    ["Float64", "Int64", "boolean", "string", "datetime64[ns]", "timedelta64[ns]"],
)
def test_tuple_accepts_all_na_dtypes_as_unavailable(dtype):
    values = pd.Series([None, None, None], dtype=dtype)
    assert newey_west_mean_tstat(values, 0) == (None, None)


@pytest.mark.parametrize("values", ["1.0", 2.0, {"x": 1}, pd.DataFrame(), [[1, 2]]])
def test_tuple_rejects_nonsequence_or_nonscalar_structures(values):
    with pytest.raises(ValueError, match="real numeric"):
        newey_west_mean_tstat(values, 0)


def test_bh_validates_bounds_and_all_missing():
    assert all(np.isnan(value) for value in benjamini_hochberg([None, np.nan]))
    for p_value in [-0.1, 1.1, np.inf]:
        with pytest.raises(ValueError, match="between zero and one"):
            benjamini_hochberg([p_value])
