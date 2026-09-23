"""Declared calendar boundaries must not shrink to observed price coverage."""

import json
from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from quant_metric_research.config import PanelConfig
from quant_metric_research.contracts import DataContractError
from quant_metric_research.session_calendar import (
    ExpectedSessionCalendar,
    compare_session_calendar,
)

SESSIONS = ("2026-04-02", "2026-04-06", "2026-04-07", "2026-04-08")


def declaration(**changes):
    return {
        "sessions": list(SESSIONS),
        "coverage_start": "2026-04-01",
        "coverage_end": "2026-04-09",
        "source": "test exchange schedule",
        "version": "fixture-v1",
        **changes,
    }


def compare(rows, dates=SESSIONS, **changes):
    prices = pd.DataFrame(rows, columns=["date", "symbol", "adjusted_close"])
    prices["date"] = pd.to_datetime(prices["date"])
    return compare_session_calendar(
        prices,
        as_of_dates=tuple(pd.Timestamp(date) for date in dates),
        config=PanelConfig(
            "fixture",
            "universe",
            "BENCH",
            lookback_sessions=1,
            min_observations=1,
            target_horizon_sessions=1,
        ),
        expected_calendar=ExpectedSessionCalendar.from_mapping(declaration(**changes)),
    )


def test_matching_calendar_keeps_holiday_and_edges_explicit():
    report = compare([(day, "BENCH", 100.0) for day in SESSIONS])
    assert report["status"] == "matched"
    assert report["independently_verified"] is False
    assert report["insufficient_lookback_dates"] == [SESSIONS[0]]
    assert report["insufficient_label_horizon_dates"] == list(SESSIONS[-2:])
    assert report["declaration"] == declaration()
    json.dumps(report, allow_nan=False)


def test_missing_first_and_last_sessions_do_not_narrow_scope():
    report = compare([(day, "BENCH", 100.0) for day in SESSIONS[1:-1]])
    assert report["status"] == "mismatch"
    assert report["missing_benchmark_sessions"] == [SESSIONS[0], SESSIONS[-1]]
    assert report["missing_all_price_sessions"] == [SESSIONS[0], SESSIONS[-1]]
    assert report["decision_dates_outside_calendar"] == []


def test_benchmark_gap_is_distinct_from_whole_market_gap():
    rows = [(SESSIONS[0], "BENCH", 100.0), (SESSIONS[1], "STOCK", 100.0)]
    report = compare(rows)
    assert report["missing_benchmark_sessions"] == list(SESSIONS[1:])
    assert report["missing_all_price_sessions"] == list(SESSIONS[2:])


def test_closed_days_bounds_and_unknown_decisions_are_not_discarded():
    rows = [
        (day, "BENCH", 100.0)
        for day in (
            *SESSIONS,
            "2026-04-01",
            "2026-04-03",
            "2026-04-09",
            "2026-03-31",
            "2026-04-10",
        )
    ]
    report = compare(rows, dates=("2026-04-03", "2026-04-10"))
    assert report["status"] == "mismatch"
    assert report["unexpected_benchmark_sessions"] == [
        "2026-04-01",
        "2026-04-03",
        "2026-04-09",
    ]
    assert report["benchmark_sessions_outside_coverage"] == ["2026-03-31", "2026-04-10"]
    assert report["decision_dates_outside_calendar"] == ["2026-04-03", "2026-04-10"]
    assert report["insufficient_lookback_dates"] == []
    assert report["insufficient_label_horizon_dates"] == []


def test_empty_benchmark_reports_every_expected_gap():
    report = compare([])
    assert report["missing_benchmark_sessions"] == list(SESSIONS)
    assert report["missing_all_price_sessions"] == list(SESSIONS)


def test_unordered_session_container_is_rejected():
    with pytest.raises(DataContractError, match="ordered"):
        ExpectedSessionCalendar.from_mapping(declaration(sessions=set(SESSIONS)))


def test_comparison_preserves_caller_inputs_and_checks_calendar_type():
    prices = pd.DataFrame(
        {"date": pd.to_datetime(SESSIONS), "symbol": "BENCH", "adjusted_close": 100.0}
    )
    original = prices.copy(deep=True)
    dates = tuple(pd.Timestamp(day) for day in SESSIONS)
    config = PanelConfig(
        "fixture",
        "universe",
        "BENCH",
        lookback_sessions=1,
        min_observations=1,
        target_horizon_sessions=1,
    )
    calendar = ExpectedSessionCalendar.from_mapping(declaration())
    report = compare_session_calendar(
        prices, as_of_dates=dates, config=config, expected_calendar=calendar
    )
    pd.testing.assert_frame_equal(prices, original)
    report["declaration"]["sessions"].clear()
    assert calendar.to_mapping() == declaration()
    with pytest.raises(DataContractError, match="ExpectedSessionCalendar"):
        compare_session_calendar(
            prices, as_of_dates=dates, config=config, expected_calendar=None
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"sessions": []},
        {"sessions": list(reversed(SESSIONS))},
        {"sessions": [SESSIONS[0], SESSIONS[0]]},
        {"sessions": "2026-04-02"},
        {"sessions": None},
        {"sessions": [1]},
        {"sessions": [None]},
        {"sessions": ["2026-04-02T12:00:00"]},
        {"sessions": ["2026-04-02T00:00:00Z"]},
        {"coverage_start": "2026-04-03"},
        {"coverage_end": "2026-04-07"},
        {"coverage_start": "2026-04-10"},
        {"coverage_end": None},
        {"coverage_start": 123},
        {"source": " "},
        {"source": None},
        {"version": ""},
        {"version": 1},
    ],
)
def test_invalid_declarations_fail(changes):
    with pytest.raises(DataContractError):
        ExpectedSessionCalendar.from_mapping(declaration(**changes))


@pytest.mark.parametrize(
    "value", [None, [], {"sessions": list(SESSIONS)}, declaration(extra=True)]
)
def test_mapping_requires_exact_fields(value):
    with pytest.raises(DataContractError):
        ExpectedSessionCalendar.from_mapping(value)


def test_declaration_copies_inputs_and_returns_independent_mapping():
    supplied = declaration()
    calendar = ExpectedSessionCalendar.from_mapping(supplied)
    supplied["sessions"].clear()
    exported = calendar.to_mapping()
    exported["sessions"].clear()
    assert calendar.to_mapping() == declaration()
    assert isinstance(calendar.sessions, tuple)
    with pytest.raises(FrozenInstanceError):
        calendar.source = "changed"


@pytest.mark.parametrize(
    "changes",
    [
        {"source": "other source"},
        {"version": "v2"},
        {"coverage_start": "2026-03-31"},
        {"coverage_end": "2026-04-10"},
        {"sessions": list(SESSIONS[:-1])},
    ],
)
def test_fingerprint_binds_full_declaration(changes):
    original = ExpectedSessionCalendar.from_mapping(declaration())
    altered = ExpectedSessionCalendar.from_mapping(declaration(**changes))
    assert original.fingerprint != altered.fingerprint
    assert (
        original.fingerprint
        == ExpectedSessionCalendar.from_mapping(original.to_mapping()).fingerprint
    )
