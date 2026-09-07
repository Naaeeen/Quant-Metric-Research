from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, asdict
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from quant_metric_research.contracts import DataContractError
from quant_metric_research.scheduled_inference import (
    HACMeanResult,
    inference_diagnostics,
    normalize_expected_dates,
    scheduled_newey_west_mean,
)
from quant_metric_research.statistics import newey_west_mean_tstat

DATES = pd.DatetimeIndex(["2025-01-03", "2025-01-06", "2025-01-15", "2025-02-01"])


@pytest.mark.parametrize("lags", [0, 1, 3])
def test_complete_irregular_supplied_schedule_matches_tuple_without_mutation(lags):
    values = pd.Series([0.1, 0.3, -0.2, 0.2], index=DATES, name="ic")
    shuffled = values.iloc[[2, 0, 3, 1]].copy(deep=True)
    before, schedule = shuffled.copy(deep=True), DATES.copy(deep=True)
    result = scheduled_newey_west_mean(shuffled, expected_dates=DATES, hac_lags=lags)
    assert result == HACMeanResult(
        "ok", *newey_west_mean_tstat(values, lags), 4, 4, lags, lags
    )
    assert result.lag_unit == "scheduled_observations"
    pd.testing.assert_series_equal(shuffled, before)
    pd.testing.assert_index_equal(DATES, schedule)


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA, pd.NaT])
@pytest.mark.parametrize("position", [0, 1, 3])
def test_explicit_missing_and_absent_dates_have_identical_diagnostics(
    missing, position
):
    values = pd.Series([0.1, 0.3, -0.2, 0.2], index=DATES, dtype=object)
    values.iloc[position] = missing
    result = scheduled_newey_west_mean(values, expected_dates=DATES, hac_lags=99)
    absent = scheduled_newey_west_mean(
        values.drop(DATES[position]), expected_dates=DATES, hac_lags=99
    )
    assert (
        result
        == absent
        == HACMeanResult("missing_scheduled_values", None, None, 4, 3, 99, None)
    )


@pytest.mark.parametrize(
    "values",
    [
        pd.Series(dtype=float),
        pd.Series([pd.NA] * 4, index=DATES, dtype="Float64"),
        pd.Series([pd.NaT] * 4, index=DATES, dtype="datetime64[ns]"),
    ],
)
def test_no_observations_preserves_nonempty_schedule(values):
    result = scheduled_newey_west_mean(values, expected_dates=DATES, hac_lags=0)
    assert result == HACMeanResult(
        "missing_scheduled_values", None, None, 4, 0, 0, None
    )


def test_status_precedence_single_observation_before_lag_support():
    result = scheduled_newey_west_mean(
        pd.Series([1.0], index=DATES[:1]), expected_dates=DATES[:1], hac_lags=99
    )
    assert result == HACMeanResult(
        "insufficient_observations", None, None, 1, 1, 99, None
    )


def test_lag_support_before_undefined_variance():
    result = scheduled_newey_west_mean(
        pd.Series([1.0] * 4, index=DATES), expected_dates=DATES, hac_lags=4
    )
    assert result == HACMeanResult(
        "insufficient_lag_support", None, None, 4, 4, 4, None
    )


@pytest.mark.parametrize(
    "values",
    [
        [0.0] * 4,
        [0.3] * 4,
        [1e308, -1e308, 1e308, -1e308],
        [1e308] * 4,
        [1e-300, -1e-300, 1e-300, -1e-300],
    ],
)
def test_undefined_or_nonfinite_variance_withholds_inference(values):
    with np.errstate(all="raise"):
        result = scheduled_newey_west_mean(
            pd.Series(values, index=DATES), expected_dates=DATES, hac_lags=0
        )
    assert result == HACMeanResult("undefined_variance", None, None, 4, 4, 0, None)


@pytest.mark.parametrize(
    "bad",
    [
        [],
        (),
        None,
        "2025-01-01",
        {"2025-01-01": 1},
        ["2025-01-03", "2025-01-03"],
        ["2025-01-03", "2025-01-01"],
        ["bad"],
        [pd.NaT],
        [20250101],
        [True],
        ["2025-01-01 01:00"],
        [pd.Timestamp("2025-01-01", tz="UTC")],
        [["2025-01-01"]],
        ["2500-01-01"],
    ],
)
def test_schedule_validation_rejects_malformed_empty_or_nonincreasing_inputs(bad):
    with pytest.raises(DataContractError):
        normalize_expected_dates(bad)
    with pytest.raises(DataContractError):
        scheduled_newey_west_mean(
            pd.Series(dtype=float), expected_dates=bad, hac_lags=0
        )


def test_schedule_supports_daily_representations_but_not_repairs():
    raw = [date(2025, 1, 3), "2025-01-06", np.datetime64("2025-01-15")]
    actual = normalize_expected_dates(raw)
    pd.testing.assert_index_equal(actual, DATES[:3].as_unit("ns"))
    assert isinstance(raw[0], date) and isinstance(raw[1], str)


@pytest.mark.parametrize("container", [set, frozenset])
def test_unordered_schedule_containers_are_not_a_declared_order(container):
    with pytest.raises(DataContractError, match="expected_dates"):
        normalize_expected_dates(container(["2025-01-03"]))


@pytest.mark.parametrize(
    "index",
    [
        [DATES[0], DATES[0]],
        ["bad", DATES[0]],
        [DATES[0], pd.NaT],
        [0, 1],
        [DATES[0], pd.Timestamp("2025-01-04")],
        [DATES[0], pd.Timestamp("2025-01-06 00:01")],
        [DATES[0], pd.Timestamp("2025-01-06", tz="UTC")],
    ],
)
def test_invalid_duplicate_or_unexpected_value_dates_fail_even_when_missing(index):
    with pytest.raises(DataContractError):
        scheduled_newey_west_mean(
            pd.Series([None, None], index=index), expected_dates=DATES, hac_lags=0
        )


@pytest.mark.parametrize(
    "bad",
    [
        True,
        np.bool_(False),
        "NaN",
        "",
        "NA",
        "nan",
        "null",
        "bad",
        "inf",
        np.inf,
        -np.inf,
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
def test_malformed_present_values_fail_before_gap_status(bad):
    values = pd.Series([0.1, bad, None, 0.3], index=DATES, dtype=object)
    with pytest.raises(DataContractError, match="real|numeric"):
        scheduled_newey_west_mean(values, expected_dates=DATES, hac_lags=0)


@pytest.mark.parametrize("bad", [True, np.int64(1), -1, 1.0, None, "1"])
def test_lags_are_validated_even_with_no_observed_values(bad):
    with pytest.raises(ValueError, match="hac_lags"):
        scheduled_newey_west_mean(
            pd.Series(dtype=float), expected_dates=DATES, hac_lags=bad
        )


@pytest.mark.parametrize("bad", [[], [1, 2], {"date": 1}, pd.DataFrame()])
def test_values_must_be_a_series(bad):
    with pytest.raises(DataContractError, match="Series"):
        scheduled_newey_west_mean(bad, expected_dates=DATES, hac_lags=0)


def test_result_is_frozen_slotted_and_diagnostics_are_json_safe():
    result = HACMeanResult("schedule_unavailable", None, None, None, 3, 2, None)
    assert inference_diagnostics(result) == {
        "inference_status": "schedule_unavailable",
        "scheduled_date_count": None,
        "observed_date_count": 3,
        "requested_hac_lags": 2,
        "effective_hac_lags": None,
        "hac_lag_unit": "scheduled_observations",
    }
    assert json.loads(json.dumps(asdict(result), allow_nan=False)) == asdict(result)
    assert not hasattr(result, "__dict__")
    with pytest.raises(FrozenInstanceError):
        result.status = "ok"
    diagnostics = inference_diagnostics(result)
    diagnostics["observed_date_count"] = 99
    assert result.observed_count == 3
