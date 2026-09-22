"""Outcome-free, conditional consistency checks for development trading inputs.

Caller-owned manifests are evidence, not authenticated training provenance. These
checks cannot prove that permitted scoring rows were not omitted from a panel.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from hashlib import sha256
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ._comparison_contract import (
    _canonical,
    _freeze,
    _identity,
    _plain,
    _schedule,
    _scope,
)
from ._comparison_predictions import (
    _BOUNDS,
    _assignment_frame,
    _fit_and_schedule,
    _integers,
    _require,
    _text,
    validate_phase,
)
from .benchmark import BenchmarkRun
from .benchmark_config import BenchmarkConfig
from .contracts import _daily_dates, validate_prices
from .session_calendar import ExpectedSessionCalendar
from .statistics import _numeric_observations

if TYPE_CHECKING:
    from .portfolio import LongOnlyConfig

_SCORE_COLUMNS = (
    "phase",
    "fold",
    "as_of_date",
    "symbol",
    "row_id",
    "model",
    "score",
    "fit_end_date",
    "train_label_end_max",
    "evaluation_start",
    "feature_count",
    "selected_feature_count",
    "zero_observed_features",
)


@dataclass(frozen=True, slots=True)
class PortfolioInputs:
    scores: pd.DataFrame
    prices: pd.DataFrame
    sessions: tuple[pd.Timestamp, ...]
    execution_dates: tuple[tuple[pd.Timestamp, pd.Timestamp], ...]
    identity: Mapping[str, Any]


def _assignments(run, benchmark, dates, identity):
    validate_phase(run.fold_assignments, "assignments")
    frame = _assignment_frame(run.fold_assignments)
    if set(frame.fold) != set(range(1, benchmark.split.outer_n_splits + 1)):
        raise ValueError("Assignments do not contain configured folds.")
    previous_end = None
    schedule = pd.DatetimeIndex(dates)
    row_ids = set(frame.loc[frame.fold.eq(1), "row_id"])
    for fold, group in frame.groupby("fold", sort=True):
        if (
            not group.split_id.eq(f"development_fold_{fold - 1}").all()
            or set(group.row_id) != row_ids
            or any(group[name].nunique() != 1 for name in _BOUNDS)
        ):
            raise ValueError("Inconsistent assignment split IDs, rows or boundaries.")
        bound = group.iloc[0]
        start, end = bound.evaluation_start, bound.evaluation_end
        if (
            not pd.Timestamp(identity["development_start"])
            <= bound.train_end_date
            < bound.train_label_end_max
            < start
            <= end
            < pd.Timestamp(identity["locked_test_start"])
            or (previous_end is not None and start <= previous_end)
            or start not in schedule
            or end not in schedule
        ):
            raise ValueError("Invalid assignment temporal boundaries.")
        scheduled_count = int(((schedule >= start) & (schedule <= end)).sum())
        if scheduled_count != benchmark.split.outer_test_date_count:
            raise ValueError(
                "Assignment scheduled date count differs from configuration."
            )
        previous_end = end
    if (
        schedule[0] != frame.evaluation_start.min()
        or schedule[-1] != frame.evaluation_end.max()
    ):
        raise ValueError("Schedule must span planned evaluation bounds.")
    return frame


def _scores(run, model, benchmark, dates, identity):
    validate_phase(run.predictions, "predictions")
    _require(run.predictions, _SCORE_COLUMNS, "predictions")
    frame = run.predictions.loc[:, list(_SCORE_COLUMNS)]
    frame = frame.loc[frame.model.eq(model)].copy(deep=True).reset_index(drop=True)
    if frame.empty:
        raise ValueError("Selected model has no predictions.")
    for name in ("phase", "symbol", "row_id", "model"):
        _text(frame[name], name)
    if not frame.symbol.eq(frame.symbol.str.upper()).all():
        raise ValueError("Prediction symbols must be canonical uppercase.")
    _integers(frame.fold, "fold", minimum=1)
    for name in (
        "as_of_date",
        "fit_end_date",
        "train_label_end_max",
        "evaluation_start",
    ):
        frame[name] = _daily_dates(frame[name], field=name)
    if frame.duplicated(["as_of_date", "symbol"]).any() or not frame.row_id.is_unique:
        raise ValueError("Prediction date/symbol keys and row IDs must be unique.")
    assignments = _assignments(run, benchmark, dates, identity)
    _fit_and_schedule(frame, assignments, dates)
    _integers(frame.feature_count, "feature_count")
    _integers(frame.selected_feature_count, "selected_feature_count", minimum=1)
    if (
        any(
            not isinstance(value, (bool, np.bool_))
            for value in frame.zero_observed_features
        )
        or not frame.zero_observed_features.eq(frame.feature_count.eq(0)).all()
        or (frame.feature_count > frame.selected_feature_count).any()
        or (frame.selected_feature_count > len(benchmark.feature_columns)).any()
        or (frame.groupby("fold").selected_feature_count.nunique() != 1).any()
    ):
        raise ValueError("Inconsistent feature availability metadata.")
    if (model == "best_metric" or model.startswith("metric:")) and (
        not frame.selected_feature_count.eq(1).all()
        or not frame.feature_count.isin((0, 1)).all()
    ):
        raise ValueError("Individual baselines must select exactly one feature.")
    frame["score"] = _numeric_observations(frame.score)
    if (frame.zero_observed_features & frame.score.notna()).any():
        raise ValueError("Zero-observed-feature predictions require missing scores.")
    return frame.sort_values(["as_of_date", "symbol"], kind="stable").reset_index(
        drop=True
    )


def _window(calendar, config, identity):
    if not isinstance(calendar, ExpectedSessionCalendar):
        raise ValueError("An explicit ExpectedSessionCalendar is required.")
    all_sessions = calendar.sessions
    if (
        config.valuation_start not in all_sessions
        or config.valuation_end not in all_sessions
        or config.valuation_end >= pd.Timestamp(identity["locked_test_start"])
    ):
        raise ValueError(
            "Valuation endpoints must be calendar sessions before final scope."
        )
    executions = []
    for decision in config.decision_dates:
        if decision not in all_sessions or decision < config.valuation_start:
            raise ValueError(
                "Decision dates must be calendar sessions within valuation scope."
            )
        index = all_sessions.index(decision) + config.execution_lag_sessions
        if index >= len(all_sessions) or all_sessions[index] >= config.valuation_end:
            raise ValueError(
                "Execution must precede the mandatory liquidation session."
            )
        executions.append((decision, all_sessions[index]))
    return tuple(
        x for x in all_sessions if config.valuation_start <= x <= config.valuation_end
    ), tuple(executions)


def _prices(prices, sessions, config):
    _require(prices, ("date", "symbol", "adjusted_close"), "prices")
    # Only date structure is inspected outside the authorized valuation window.
    dates = _daily_dates(prices["date"], field="date")
    window = dates.between(config.valuation_start, config.valuation_end)
    frame = prices.loc[window, ["date", "symbol", "adjusted_close"]].copy(deep=True)
    frame["date"] = dates.loc[window]
    frame = validate_prices(frame)
    if not frame.date.isin(sessions).all():
        raise ValueError("Price dates outside the declared session calendar.")
    benchmark_dates = set(frame.loc[frame.symbol.eq(config.benchmark_symbol), "date"])
    if benchmark_dates != set(sessions):
        raise ValueError("Missing benchmark price on an expected valuation session.")
    return frame


def _frame_hash(frame):
    # JSON encodes null scores explicitly and preserves float64 round-trip digits.
    payload = frame.to_json(
        orient="split", date_format="iso", double_precision=15, index=False
    )
    # pandas JSON caps precision at 15; exact binary floats are bound separately.
    numeric = frame.select_dtypes(include="number")
    exact = tuple(
        tuple(None if pd.isna(x) else float(x).hex() for x in row)
        for row in numeric.itertuples(index=False, name=None)
    )
    return sha256((payload + _canonical(exact)).encode()).hexdigest()


def prepare_portfolio_inputs(
    run: BenchmarkRun,
    prices: pd.DataFrame,
    *,
    model: str,
    calendar: ExpectedSessionCalendar,
    config: LongOnlyConfig,
) -> PortfolioInputs:
    """Validate supplied development evidence without reading target/return columns."""
    _scope(run)
    identity = _identity(run.manifest)
    supplied = run.manifest.get("configuration")
    if not isinstance(supplied, Mapping) or set(supplied) != {
        f.name for f in fields(BenchmarkConfig)
    }:
        raise ValueError(
            "Stored benchmark configuration must explicitly contain every field."
        )
    benchmark = BenchmarkConfig.from_mapping(_plain(_freeze(supplied)))
    if not isinstance(model, str) or model not in (
        *benchmark.model_families,
        "equal_weight_rank",
        "best_metric",
        *(f"metric:{feature}" for feature in benchmark.feature_columns),
    ):
        raise ValueError(
            "Selected model must be a configured model or supported baseline."
        )
    dates = _schedule(run.manifest)
    scores = _scores(run, model, benchmark, dates, identity)
    sessions, executions = _window(calendar, config, identity)
    if not set(config.decision_dates).issubset(set(scores.as_of_date)):
        raise ValueError(
            "Every declared decision needs scheduled score rows, including null scores."
        )
    scores = scores.loc[scores.as_of_date.isin(config.decision_dates)].reset_index(
        drop=True
    )
    normalized_prices = _prices(prices, sessions, config)
    evidence = dict(
        identity,
        run_fingerprint=run.manifest["run_fingerprint"],
        model_input_fingerprint=run.manifest["model_input_fingerprint"],
        model=model,
        score_fingerprint=_frame_hash(scores),
        price_fingerprint=_frame_hash(normalized_prices),
        calendar_fingerprint=calendar.fingerprint,
        calendar=calendar.to_mapping(),
        evidence_scope="conditional_consistency_not_authenticated_training",
        outcome_values_read=False,
    )
    if "experiment" in run.manifest:
        experiment = run.manifest["experiment"]
        evidence["experiment"] = {
            key: experiment[key]
            for key in ("kind", "run_id", "study_id")
            if key in experiment
        }
        for key in ("run_id", "study_id"):
            if key in evidence["experiment"]:
                _text(pd.Series([evidence["experiment"][key]]), key)
    return PortfolioInputs(
        scores, normalized_prices, sessions, executions, _freeze(evidence)
    )
