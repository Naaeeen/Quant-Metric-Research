"""Offline coverage diagnostics, not certification of historical data provenance."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ._version import __version__
from .config import PanelConfig
from .contracts import (
    DataContractError,
    validate_as_of_dates,
    validate_memberships,
    validate_prices,
)
from .panel import _active_symbols, _label_dates

_COUNTS = (
    "active_members",
    "missing_decision_price",
    "sufficient_return_history",
    "label_endpoints_available",
    "history_and_label_endpoints_available",
    "missing_label_entry",
    "missing_label_exit",
    "label_calendar_unavailable",
)
_EVIDENCE = (
    "historical_universe_and_reconstruction_policy",
    "stable_security_identifiers_and_symbol_mapping",
    "corporate_action_adjustments",
    "delisting_and_terminal_outcome_treatment",
    "historical_availability_and_revision_policy",
    "independent_exchange_calendar",
    "snapshot_lineage_and_acquisition_time",
    "license_and_redistribution_rights",
)


@dataclass(frozen=True)
class _Presence:
    sessions: frozenset[int]
    adjacent_return_ends: np.ndarray

    def history_count(self, start: int, end: int) -> int:
        # A paired return needs both prices inside [start, end].
        positions = self.adjacent_return_ends
        return int(
            np.searchsorted(positions, end, side="right")
            - np.searchsorted(positions, start, side="right")
        )


def _presence_by_symbol(
    prices: pd.DataFrame, calendar: pd.DatetimeIndex
) -> dict[str, _Presence]:
    result = {}
    for symbol, group in prices.groupby("symbol", sort=True):
        located = calendar.get_indexer(group["date"])
        sessions = frozenset(int(value) for value in located if value >= 0)
        result[str(symbol)] = _Presence(
            sessions=sessions,
            adjacent_return_ends=np.array(
                sorted(value for value in sessions if value - 1 in sessions),
                dtype=np.int64,
            ),
        )
    return result


def _cell(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (float, np.floating)):
        return float(value).hex()
    return str(value)


def _frame_fingerprint(frame: pd.DataFrame, columns: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(columns).encode("utf-8"))
    for row in frame.loc[:, list(columns)].itertuples(index=False, name=None):
        encoded = json.dumps([_cell(value) for value in row], ensure_ascii=True)
        digest.update(b"\n" + encoded.encode("utf-8"))
    return digest.hexdigest()


def _fingerprints(
    prices: pd.DataFrame,
    memberships: pd.DataFrame,
    dates: tuple[pd.Timestamp, ...],
    config: PanelConfig,
) -> dict[str, str]:
    normalized_prices = prices.assign(
        adjusted_close=prices["adjusted_close"].astype(float)
    )
    request = {
        "config": asdict(config),
        "as_of_dates": [date.isoformat() for date in dates],
    }
    return {
        "algorithm": "sha256-normalized-required-columns-v1",
        "scope": (
            "required normalized columns and request; "
            "excludes extra columns and raw file bytes"
        ),
        "prices": _frame_fingerprint(
            normalized_prices, ("date", "symbol", "adjusted_close")
        ),
        "memberships": _frame_fingerprint(
            memberships,
            ("universe_id", "symbol", "effective_from", "effective_to", "source"),
        ),
        "request": hashlib.sha256(
            json.dumps(request, allow_nan=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }


def _member_counts(
    presence: _Presence,
    *,
    current: int,
    label_start: int | None,
    label_end: int | None,
    config: PanelConfig,
) -> dict[str, int]:
    sufficient = (
        presence.history_count(max(0, current - config.lookback_sessions), current)
        >= config.min_observations
    )
    calendar_available = label_start is not None and label_end is not None
    entry_missing = calendar_available and label_start not in presence.sessions
    exit_missing = calendar_available and label_end not in presence.sessions
    endpoints = calendar_available and not entry_missing and not exit_missing
    return {
        "active_members": 1,
        "missing_decision_price": int(current not in presence.sessions),
        "sufficient_return_history": int(sufficient),
        "label_endpoints_available": int(endpoints),
        "history_and_label_endpoints_available": int(sufficient and endpoints),
        "missing_label_entry": int(entry_missing),
        "missing_label_exit": int(exit_missing),
        "label_calendar_unavailable": int(not calendar_available),
    }


def _coverage(
    prices: pd.DataFrame,
    memberships: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    dates: tuple[pd.Timestamp, ...],
    config: PanelConfig,
) -> tuple[list[dict], list[dict]]:
    presence = _presence_by_symbol(prices, calendar)
    absent = _Presence(frozenset(), np.array([], dtype=np.int64))
    by_date, by_security = [], {}
    for date in dates:
        current = int(calendar.get_loc(date))
        start, end = _label_dates(calendar, current, config)
        label_start = int(calendar.get_loc(start)) if pd.notna(start) else None
        label_end = int(calendar.get_loc(end)) if pd.notna(end) else None
        counts = {name: 0 for name in _COUNTS}
        for symbol in _active_symbols(
            memberships, universe_id=config.universe_id, as_of_date=date
        ):
            row = _member_counts(
                presence.get(symbol, absent),
                current=current,
                label_start=label_start,
                label_end=label_end,
                config=config,
            )
            counts = {name: counts[name] + row[name] for name in _COUNTS}
            previous = by_security.get(symbol, {name: 0 for name in _COUNTS})
            by_security[symbol] = {name: previous[name] + row[name] for name in _COUNTS}
        by_date.append(
            {
                "as_of_date": date.date().isoformat(),
                "label_start_date": start.date().isoformat()
                if pd.notna(start)
                else None,
                "label_end_date": end.date().isoformat() if pd.notna(end) else None,
                **counts,
            }
        )
    return by_date, [
        {
            "symbol": symbol,
            "active_dates": counts["active_members"],
            **{key: value for key, value in counts.items() if key != "active_members"},
        }
        for symbol, counts in sorted(by_security.items())
    ]


def _warnings(by_date: list[dict], off_calendar: int) -> list[dict[str, str]]:
    checks = (
        (
            any(row["active_members"] == 0 for row in by_date),
            "no_active_members",
            "Some requested dates have no active members in the supplied universe.",
        ),
        (
            off_calendar > 0,
            "off_calendar_prices",
            "Off-calendar prices are excluded from coverage.",
        ),
        (
            any(row["missing_decision_price"] for row in by_date),
            "missing_decision_prices",
            "Some active members have no decision-date price.",
        ),
        (
            any(
                row["sufficient_return_history"] < row["active_members"]
                for row in by_date
            ),
            "insufficient_return_history",
            "Some members lack enough adjacent historical price pairs.",
        ),
        (
            any(
                row["missing_label_entry"] or row["missing_label_exit"]
                for row in by_date
            ),
            "missing_label_endpoints",
            "Some scheduled stock label endpoints are missing; no repair is applied.",
        ),
        (
            any(row["label_calendar_unavailable"] for row in by_date),
            "label_calendar_unavailable",
            "Supplied benchmark history does not cover some complete label intervals.",
        ),
    )
    return [
        {"code": code, "message": message}
        for applies, code, message in checks
        if applies
    ]


def audit_inputs(
    prices: pd.DataFrame,
    memberships: pd.DataFrame,
    *,
    as_of_dates: Iterable[object],
    config: PanelConfig,
) -> dict:
    """Inspect required inputs without computing returns, fitting, or approving data.

    Future price *presence* is inspected. This is not a physically sealed holdout
    or a substitute for provider evidence, raw snapshots, or benchmark preflight.
    """
    if not isinstance(config, PanelConfig):
        raise ValueError("config must be a PanelConfig.")
    checked_prices = validate_prices(prices)
    checked_memberships = validate_memberships(memberships)
    dates = tuple(sorted(validate_as_of_dates(as_of_dates)))
    calendar = pd.DatetimeIndex(
        checked_prices.loc[checked_prices["symbol"] == config.benchmark_symbol, "date"]
    )
    if calendar.empty:
        raise DataContractError("Benchmark price history is required.")
    if any(date not in calendar for date in dates):
        raise DataContractError(
            "All as_of_dates must be present in the benchmark calendar."
        )
    by_date, by_security = _coverage(
        checked_prices, checked_memberships, calendar, dates, config
    )
    off_calendar = int((~checked_prices["date"].isin(calendar)).sum())
    available_symbols = set(checked_prices["symbol"])
    return {
        "schema_version": 1,
        "package_version": __version__,
        "config": {**asdict(config), "feature_columns": list(config.feature_columns)},
        "claim_scope": "raw_input_diagnostics_only",
        "no_outcomes_computed": True,
        "empirical_data_provenance_verified": False,
        "stage4_eligible": False,
        "pandas_version": pd.__version__,
        "input_fingerprints": _fingerprints(
            checked_prices, checked_memberships, dates, config
        ),
        "calendar": {
            "source": "supplied_benchmark_prices",
            "independently_verified": False,
            "first_session": calendar[0].date().isoformat(),
            "last_session": calendar[-1].date().isoformat(),
            "session_count": len(calendar),
        },
        "summary": {
            "requested_dates": len(dates),
            "active_member_dates": sum(row["active_members"] for row in by_date),
            "members_without_any_prices": [
                row["symbol"]
                for row in by_security
                if row["symbol"] not in available_symbols
            ],
            "off_calendar_price_rows": off_calendar,
        },
        "coverage_by_date": by_date,
        "coverage_by_security": by_security,
        "warnings": _warnings(by_date, off_calendar),
        "external_evidence": [
            {"check": check, "status": "unverified"} for check in _EVIDENCE
        ],
    }
