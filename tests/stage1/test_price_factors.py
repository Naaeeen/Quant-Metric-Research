from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_index_equal, assert_series_equal

from quant_metric_research.contracts import DataContractError
from quant_metric_research.price_factor_catalog import PRICE_FACTOR_CATALOG
from quant_metric_research.price_factors import compute_price_factors

NAMES = ("return_21s", "momentum_252s_skip_21s", "ma_distance_63s")


def _prices(count=253):
    calendar = pd.bdate_range("2020-01-02", periods=count)
    return calendar, pd.Series(100.0 + np.arange(count), index=calendar)


def _compute(prices, calendar, names=NAMES, as_of_date=None):
    return compute_price_factors(
        prices,
        calendar=calendar,
        as_of_date=calendar[-1] if as_of_date is None else as_of_date,
        factor_names=names,
    )


def test_catalog_is_immutable_explicit_and_versioned():
    assert tuple(PRICE_FACTOR_CATALOG) == NAMES
    for name, spec in PRICE_FACTOR_CATALOG.items():
        assert spec.name == name
        assert spec.formula_version == "1"
        assert spec.required_input == "adjusted_close"
        assert spec.required_price_count == (
            spec.start_lag_sessions - spec.end_lag_sessions + 1
        )
        assert spec.source_references and isinstance(spec.source_references, tuple)
        assert spec.availability_assumption and spec.adjustment_assumption
        with pytest.raises(FrozenInstanceError):
            spec.name = "changed"
    with pytest.raises(TypeError):
        PRICE_FACTOR_CATALOG["new"] = PRICE_FACTOR_CATALOG[NAMES[0]]


def test_exact_windows_formulas_metadata_and_requested_order():
    calendar, prices = _prices()
    before = prices.copy(deep=True)
    calendar_before = calendar.copy()
    result = _compute(prices.iloc[::-1], calendar, names=NAMES[::-1])
    expected = {
        "return_21s": 352.0 / 331.0 - 1,
        "momentum_252s_skip_21s": 331.0 / 100.0 - 1,
        "ma_distance_63s": 352.0 / 321.0 - 1,
    }
    assert result.as_of_date == calendar[-1]
    assert tuple(value.name for value in result.factors) == NAMES[::-1]
    for factor in result.factors:
        spec = PRICE_FACTOR_CATALOG[factor.name]
        assert factor.value == pytest.approx(expected[factor.name])
        assert factor.status == "ok"
        assert factor.formula == spec.formula
        assert factor.formula_version == spec.formula_version
        assert factor.as_of_date == result.as_of_date
        assert factor.window_start == calendar[-1 - spec.start_lag_sessions]
        assert factor.window_end == calendar[-1 - spec.end_lag_sessions]
        assert factor.observed_price_count == spec.required_price_count
        assert factor.required_price_count == spec.required_price_count
        with pytest.raises(FrozenInstanceError):
            factor.value = 1.0
    with pytest.raises(FrozenInstanceError):
        result.as_of_date = calendar[0]
    assert_series_equal(prices, before)
    assert_index_equal(calendar, calendar_before)


@pytest.mark.parametrize("name", NAMES)
def test_exact_calendar_boundary_and_one_session_short(name):
    spec = PRICE_FACTOR_CATALOG[name]
    calendar, prices = _prices(spec.start_lag_sessions + 1)
    complete = _compute(prices, calendar, names=(name,)).factors[0]
    incomplete = _compute(prices.iloc[1:], calendar[1:], names=(name,)).factors[0]

    assert complete.status == "ok"
    assert incomplete.status == "insufficient_history"
    assert incomplete.value is None
    assert incomplete.window_start is None
    assert incomplete.window_end == complete.window_end
    assert incomplete.required_price_count == spec.required_price_count
    assert incomplete.observed_price_count == spec.required_price_count - 1


@pytest.mark.parametrize(
    "count, observed, end_resolvable", [(10, 0, False), (30, 9, True)]
)
def test_short_momentum_window_reports_only_resolvable_intersection(
    count, observed, end_resolvable
):
    calendar, prices = _prices(count)
    factor = _compute(prices, calendar, names=(NAMES[1],)).factors[0]
    assert factor.status == "insufficient_history"
    assert factor.observed_price_count == observed
    assert factor.required_price_count == 232
    assert factor.window_start is None
    assert (factor.window_end is not None) is end_resolvable


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("missing_position", ["start", "middle", "end"])
@pytest.mark.parametrize("drop_row", [False, True])
def test_required_missing_prices_never_compress_the_calendar(
    name, missing_position, drop_row
):
    calendar, prices = _prices()
    spec = PRICE_FACTOR_CATALOG[name]
    start, end = 252 - spec.start_lag_sessions, 252 - spec.end_lag_sessions
    position = {"start": start, "middle": (start + end) // 2, "end": end}[
        missing_position
    ]
    changed = prices.drop(calendar[position]) if drop_row else prices.copy()
    if not drop_row:
        changed.loc[calendar[position]] = np.nan
    factor = _compute(changed, calendar, names=(name,)).factors[0]
    assert factor.status == "missing_required_prices"
    assert factor.value is None
    assert factor.observed_price_count == spec.required_price_count - 1


def test_excluded_recent_prices_do_not_change_momentum_value_or_completeness():
    calendar, prices = _prices()
    expected = _compute(prices, calendar, names=(NAMES[1],))
    missing = prices.iloc[:-21]
    changed = prices.copy()
    changed.iloc[-21:] = 1e100

    assert _compute(missing, calendar, names=(NAMES[1],)) == expected
    assert _compute(changed, calendar, names=(NAMES[1],)) == expected


def test_status_priority_and_per_factor_observed_counts():
    calendar, prices = _prices(63)
    changed = prices.copy()
    changed.iloc[0] = np.nan
    values = {factor.name: factor for factor in _compute(changed, calendar).factors}

    assert values[NAMES[0]].status == "ok"
    assert values[NAMES[0]].observed_price_count == 22
    assert values[NAMES[1]].status == "insufficient_history"
    assert values[NAMES[1]].observed_price_count == 41
    assert values[NAMES[2]].status == "missing_required_prices"
    assert values[NAMES[2]].observed_price_count == 62


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA, pd.NaT])
def test_all_actual_missing_prices_and_empty_series_are_valid_gaps(missing):
    calendar, _ = _prices()
    prices = pd.Series([missing] * len(calendar), index=calendar)
    for supplied in (prices, pd.Series(dtype=float)):
        factors = _compute(supplied, calendar).factors
        assert all(factor.status == "missing_required_prices" for factor in factors)
        assert all(factor.observed_price_count == 0 for factor in factors)


@pytest.mark.parametrize(
    "bad",
    [
        0,
        -1,
        np.inf,
        -np.inf,
        True,
        complex(1, 0),
        "bad",
        "NaN",
        "",
        date(2020, 1, 1),
        timedelta(days=1),
    ],
)
def test_every_nonmissing_historical_price_is_validated_even_outside_factor_window(bad):
    calendar, prices = _prices()
    changed = prices.astype(object)
    changed.iloc[0] = bad
    with pytest.raises(DataContractError, match="positive real adjusted_close"):
        _compute(changed, calendar, names=(NAMES[0],))


def test_numeric_strings_are_supported_without_mutating_inputs():
    calendar, prices = _prices()
    strings = prices.astype(str)
    before = strings.copy(deep=True)
    assert _compute(strings, calendar) == _compute(prices, calendar)
    assert_series_equal(strings, before)


@pytest.mark.parametrize(
    "name, changed_position", [(NAMES[0], 0), (NAMES[1], -1), (NAMES[2], 0)]
)
def test_valid_excluded_numeric_strings_cannot_change_window_conversion(
    name, changed_position
):
    calendar, _ = _prices()
    prices = pd.Series(
        [str(2**60 + index * 13) for index in range(253)],
        index=calendar,
        dtype=object,
    )
    expected = _compute(prices, calendar, names=(name,))
    changed = prices.copy(deep=True)
    changed.iloc[changed_position] = "1.2345"

    assert _compute(changed, calendar, names=(name,)) == expected


def test_future_bad_values_are_ignored_after_structural_date_validation():
    calendar, prices = _prices(260)
    cutoff = calendar[252]
    changed = prices.astype(object)
    changed.iloc[253:] = ["bad", 0, np.inf, False, complex(1, 2), pd.NA, "NaN"]
    assert _compute(changed, calendar, as_of_date=cutoff) == _compute(
        prices.iloc[:253], calendar, as_of_date=cutoff
    )


@pytest.mark.parametrize(
    "future_index", ["bad-date", "2020-01-02", "2025-01-01T12:00:00"]
)
def test_future_structural_date_errors_are_still_rejected(future_index):
    calendar, prices = _prices()
    changed = pd.concat([prices, pd.Series(["bad"], index=[future_index])])
    with pytest.raises(DataContractError):
        _compute(changed, calendar)


def test_historical_off_calendar_rows_fail_but_valid_older_prices_are_ignored():
    calendar, prices = _prices()
    off_calendar = pd.concat(
        [prices, pd.Series([100.0], index=[pd.Timestamp("2020-01-04")])]
    )
    with pytest.raises(DataContractError, match="off-calendar"):
        _compute(off_calendar, calendar)
    older = pd.concat(
        [prices, pd.Series([100.0], index=[calendar[0] - pd.Timedelta(days=1)])]
    )
    assert _compute(older, calendar) == _compute(prices, calendar)
    older.iloc[-1] = -1.0
    with pytest.raises(DataContractError):
        _compute(older, calendar)


@pytest.mark.parametrize(
    "names", [(), [], "return_21s", ("unknown",), ("return_21s", "return_21s"), (None,)]
)
def test_factor_names_must_be_a_nonempty_unique_known_tuple(names):
    calendar, prices = _prices()
    with pytest.raises(DataContractError, match="factor_names"):
        _compute(prices, calendar, names=names)


@pytest.mark.parametrize(
    "calendar",
    [
        [],
        ["2020-01-02", "2020-01-02"],
        ["2020-01-03", "2020-01-02"],
        [pd.NaT],
        [1],
        ["2020-01-02T00:00:00Z"],
        ["2020-01-02T12:00:00"],
    ],
)
def test_calendar_must_be_nonempty_sorted_unique_normalized_daily_dates(calendar):
    with pytest.raises(DataContractError):
        compute_price_factors(
            pd.Series(dtype=float),
            calendar=calendar,
            as_of_date="2020-01-02",
            factor_names=NAMES,
        )


@pytest.mark.parametrize(
    "cutoff",
    ["2020-01-04", "bad-date", 1, "2020-01-02T12:00:00", "2020-01-02T00:00:00Z"],
)
def test_cutoff_must_be_a_normalized_exact_calendar_member(cutoff):
    calendar, prices = _prices()
    with pytest.raises(DataContractError):
        _compute(prices, calendar, as_of_date=cutoff)


@pytest.mark.parametrize("name", NAMES)
def test_constant_prices_produce_zero_without_cross_factor_observation_gate(name):
    spec = PRICE_FACTOR_CATALOG[name]
    calendar, prices = _prices(spec.start_lag_sessions + 1)
    factor = _compute(prices * 0 + 55.0, calendar, names=(name,)).factors[0]
    assert factor.status == "ok"
    assert factor.value == 0.0


def test_ma_uses_scaled_mean_to_avoid_intermediate_overflow():
    calendar, _ = _prices(63)
    factor = _compute(
        pd.Series(1e308, index=calendar), calendar, names=(NAMES[2],)
    ).factors[0]
    assert factor.status == "ok"
    assert factor.value == 0.0


@pytest.mark.parametrize("name", NAMES[:2])
def test_nonfinite_ratio_results_raise_with_the_factor_name(name):
    calendar, prices = _prices()
    spec = PRICE_FACTOR_CATALOG[name]
    prices.iloc[252 - spec.start_lag_sessions] = 1e-308
    prices.iloc[252 - spec.end_lag_sessions] = 1e308
    with pytest.raises(DataContractError, match=name):
        _compute(prices, calendar, names=(name,))


def test_prices_require_a_series_with_normalized_unique_date_index():
    calendar, prices = _prices()
    for invalid in (
        prices.to_frame(),
        prices.reset_index(drop=True),
        pd.concat([prices, prices.iloc[:1]]),
    ):
        with pytest.raises(DataContractError):
            _compute(invalid, calendar)
