"""Caller-declared session coverage, independent of observed price availability."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Set
from dataclasses import dataclass

import pandas as pd

from .config import PanelConfig
from .contracts import DataContractError, _daily_date, validate_as_of_dates

_FIELDS = frozenset(("sessions", "coverage_start", "coverage_end", "source", "version"))


@dataclass(frozen=True)
class ExpectedSessionCalendar:
    """Immutable daily sessions and inclusive declared coverage bounds.

    Bounds may be closed days. Sessions must already be unique and increasing;
    no holiday inference, date snapping, or observed-price narrowing is applied.
    Source/version identify a supplied declaration, not verified provenance.
    """

    sessions: tuple[pd.Timestamp, ...]
    coverage_start: pd.Timestamp
    coverage_end: pd.Timestamp
    source: str
    version: str

    def __post_init__(self) -> None:
        if isinstance(self.sessions, Set):
            raise DataContractError("Expected sessions must be an ordered sequence.")
        sessions = validate_as_of_dates(self.sessions)
        start = _daily_date(self.coverage_start, field="coverage_start", nullable=False)
        end = _daily_date(self.coverage_end, field="coverage_end", nullable=False)
        if start > end:
            raise DataContractError("coverage_start must not exceed coverage_end.")
        if any(
            left >= right for left, right in zip(sessions, sessions[1:], strict=False)
        ):
            raise DataContractError("Expected sessions must be strictly increasing.")
        if sessions[0] < start or sessions[-1] > end:
            raise DataContractError(
                "Expected sessions must lie within coverage bounds."
            )
        for name in ("source", "version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise DataContractError(f"Calendar {name} must be a nonempty string.")
        object.__setattr__(self, "sessions", sessions)
        object.__setattr__(self, "coverage_start", start)
        object.__setattr__(self, "coverage_end", end)

    def to_mapping(self) -> dict:
        """Return an independent, strict-JSON-compatible full declaration."""
        return {
            "sessions": [session.date().isoformat() for session in self.sessions],
            "coverage_start": self.coverage_start.date().isoformat(),
            "coverage_end": self.coverage_end.date().isoformat(),
            "source": self.source,
            "version": self.version,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping) -> ExpectedSessionCalendar:
        """Parse exact fields, raising DataContractError for invalid declarations."""
        if not isinstance(mapping, Mapping) or set(mapping) != _FIELDS:
            raise DataContractError(
                "Expected calendar requires exactly sessions, coverage_start, "
                "coverage_end, source, and version."
            )
        return cls(**dict(mapping))

    @property
    def fingerprint(self) -> str:
        """SHA-256 of canonical JSON, including full bounds and source metadata."""
        encoded = json.dumps(
            self.to_mapping(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _iso_dates(dates: Iterable[pd.Timestamp]) -> list[str]:
    return [date.date().isoformat() for date in sorted(dates)]


def compare_session_calendar(
    prices: pd.DataFrame,
    *,
    as_of_dates: tuple[pd.Timestamp, ...],
    config: PanelConfig,
    expected_calendar: ExpectedSessionCalendar,
) -> dict:
    """Compare validated prices and decision dates against the full declaration.

    Inputs must already satisfy the normalized price, decision-date and config
    contracts. An absent benchmark yields missing-session diagnostics. Insufficient
    declared lookback/future length is diagnostic, not itself a calendar mismatch.
    No input is changed and no prices, returns or expected sessions are inferred.
    """
    if not isinstance(expected_calendar, ExpectedSessionCalendar):
        raise DataContractError("expected_calendar must be an ExpectedSessionCalendar.")
    expected = set(expected_calendar.sessions)
    benchmark = set(prices.loc[prices["symbol"] == config.benchmark_symbol, "date"])
    observed = set(prices["date"])
    inside = {
        date
        for date in benchmark
        if expected_calendar.coverage_start <= date <= expected_calendar.coverage_end
    }
    mismatches = {
        "missing_benchmark_sessions": _iso_dates(expected - benchmark),
        "missing_all_price_sessions": _iso_dates(expected - observed),
        "unexpected_benchmark_sessions": _iso_dates(inside - expected),
        "benchmark_sessions_outside_coverage": _iso_dates(benchmark - inside),
        "decision_dates_outside_calendar": _iso_dates(set(as_of_dates) - expected),
    }
    positions = {
        date: position for position, date in enumerate(expected_calendar.sessions)
    }
    requested_positions = {
        date: positions[date] for date in as_of_dates if date in positions
    }
    return {
        "status": "mismatch" if any(mismatches.values()) else "matched",
        "independently_verified": False,
        "declaration": expected_calendar.to_mapping(),
        "fingerprint": expected_calendar.fingerprint,
        **mismatches,
        "insufficient_lookback_dates": _iso_dates(
            date
            for date, position in requested_positions.items()
            if position < config.lookback_sessions
        ),
        "insufficient_label_horizon_dates": _iso_dates(
            date
            for date, position in requested_positions.items()
            if position + config.entry_lag_sessions + config.target_horizon_sessions
            >= len(expected_calendar.sessions)
        ),
    }
