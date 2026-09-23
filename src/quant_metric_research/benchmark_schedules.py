"""Shared planned evaluation schedules and immutable manifest evidence."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import pandas as pd

from .benchmark_data import Stage3DataPlan
from .scheduled_inference import normalize_expected_dates


def phase_schedule(plan: Stage3DataPlan, *, phase: str) -> pd.DatetimeIndex:
    """Use planned boundaries and panel dates, never surviving prediction rows.

    Dates between development folds remain in the supplied observation clock.
    This cannot identify dates that were absent from the panel itself.
    """
    if phase == "development":
        start = min(fold.evaluation_start_date for fold in plan.development_folds)
        end = max(fold.evaluation_end_date for fold in plan.development_folds)
    elif phase == "locked_test":
        start, end = plan.locked_test.test_start_date, plan.locked_test.test_end_date
    else:
        raise ValueError("Unknown evaluation phase.")
    dates = plan.panel.frame["as_of_date"]
    return normalize_expected_dates(
        dates.loc[dates.between(start, end)].drop_duplicates().sort_values()
    )


def evaluation_schedule_metadata(
    plan: Stage3DataPlan, *, evaluate_lockbox: bool
) -> MappingProxyType[str, Any]:
    """Record exact planned phase dates, not evidence of successful evaluation.

    Development mode omits the locked sequence. ISO strings and immutable
    containers support the existing JSON writer without a new identity scheme.
    """
    if not isinstance(evaluate_lockbox, bool):
        raise ValueError("evaluate_lockbox must be a boolean.")
    phases = ("development", "locked_test") if evaluate_lockbox else ("development",)
    dates_by_phase = MappingProxyType(
        {
            phase: tuple(str(date.date()) for date in phase_schedule(plan, phase=phase))
            for phase in phases
        }
    )
    return MappingProxyType(
        {
            "definition_version": "1",
            "source": "validated_panel_within_planned_phase_bounds",
            "lag_unit": "scheduled_observations",
            "dates_by_phase": dates_by_phase,
        }
    )
