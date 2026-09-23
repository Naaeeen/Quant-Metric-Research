"""Non-fitting structural checks for a proposed nested benchmark experiment."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .benchmark_config import BenchmarkConfig
from .benchmark_data import (
    Stage3DataPlan,
    _numeric_values,
    build_stage3_data_plan,
    evaluation_cross_section,
    expand_training_cross_sections,
    validate_benchmark_panel,
)
from .splits import build_purged_walk_forward_splits


def _period(frame: pd.DataFrame) -> dict[str, Any]:
    dates = frame["as_of_date"]
    return {
        "start": None if frame.empty else dates.min().date().isoformat(),
        "end": None if frame.empty else dates.max().date().isoformat(),
        "date_count": int(dates.nunique()),
        "row_count": len(frame),
    }


def _feature_coverage(frame: pd.DataFrame, config: BenchmarkConfig) -> dict[str, Any]:
    return {
        feature: float(frame[feature].notna().mean()) if len(frame) else None
        for feature in config.feature_columns
    }


def _hac_bounds(date_count: int, requested_lags: int) -> dict[str, Any]:
    supported = date_count >= 2 and requested_lags <= date_count - 1
    return {
        "requested_lags": requested_lags,
        "eligible_date_count_upper_bound": date_count,
        "maximum_supported_lags": date_count - 1 if date_count >= 2 else None,
        "requested_lags_supported_by_count": supported,
        "maximum_effective_lags": requested_lags if supported else None,
        "interpretation": (
            "Rank-IC date-count bound from label availability, not an inference "
            "result; gaps or undefined scores/correlations withhold inference. "
            "Unsupported requested lags are never automatically reduced."
        ),
    }


def _outcome_availability(
    frame: pd.DataFrame, config: BenchmarkConfig
) -> dict[str, Any]:
    dates = frame["as_of_date"]
    targets = frame[config.target_column].notna()
    returns = frame[config.realized_return_column].notna()
    target_counts = targets.groupby(dates).sum()
    return_counts = returns.groupby(dates).sum()
    target_dates = int(target_counts.ge(config.min_cross_section).sum())
    return {
        **_period(frame),
        "expected_label_row_count": len(frame),
        "target_row_count": int(targets.sum()),
        "realized_return_row_count": int(returns.sum()),
        "target_evaluable_date_count": target_dates,
        "return_evaluable_date_count": int(
            return_counts.ge(config.min_cross_section).sum()
        ),
        "hac": _hac_bounds(target_dates, config.hac_lags),
    }


def _training_summary(
    candidates: pd.DataFrame,
    training: pd.DataFrame,
    evaluation_start: pd.Timestamp,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    potential = candidates.loc[candidates["as_of_date"] < evaluation_start]
    label_end = potential["label_end_date"]
    label_max = training["label_end_date"].max()
    supervised = training[config.target_column].notna()
    return {
        **_period(training),
        "candidate_row_count": len(potential),
        "supervised_row_count": int(supervised.sum()),
        "supervised_date_count": int(training.loc[supervised, "as_of_date"].nunique()),
        "feature_only_row_count": int((~supervised).sum()),
        "purged_label_overlap_count": int(label_end.ge(evaluation_start).sum()),
        "missing_label_count": int(label_end.isna().sum()),
        "missing_target_count": int(potential[config.target_column].isna().sum()),
        "label_end_max": None if pd.isna(label_max) else label_max.date().isoformat(),
        "feature_coverage": _feature_coverage(training, config),
    }


def _warning(scope: str, code: str, message: str) -> dict[str, str]:
    return {"scope": scope, "code": code, "message": message}


def _hac_warnings(
    scope: str, count: int, config: BenchmarkConfig
) -> list[dict[str, str]]:
    warnings = []
    if count < 2:
        warnings.append(
            _warning(
                scope,
                "insufficient_hac_dates",
                (
                    "Fewer than two potentially evaluable Rank-IC dates; HAC inference "
                    "cannot be computed even before accounting for undefined scores."
                ),
            )
        )
    if count >= 2 and config.hac_lags >= count:
        warnings.append(
            _warning(
                scope,
                "unsupported_hac_lags",
                (
                    f"Requested {config.hac_lags} HAC lags exceed the date-count "
                    "bound; inference is withheld rather than shortening the lag."
                ),
            )
        )
    if count <= 2 * (config.hac_lags + 1):
        warnings.append(
            _warning(
                scope,
                "short_hac_period",
                (
                    "Potentially evaluable dates are at most twice (requested HAC lags "
                    "+ 1). This is a transparent planning warning, not a universal "
                    "sample-size rule or a guarantee of reliable inference."
                ),
            )
        )
    return warnings


def _period_warnings(
    scope: str, availability: dict[str, Any], config: BenchmarkConfig
) -> list[dict[str, str]]:
    warnings = _hac_warnings(
        scope, availability["hac"]["eligible_date_count_upper_bound"], config
    )
    if availability["return_evaluable_date_count"] == 0:
        warnings.append(
            _warning(
                scope,
                "no_return_evaluable_dates",
                (
                    "No dates meet the configured realized-return cross-section; "
                    "a spread cannot be evaluated on this period."
                ),
            )
        )
    return warnings


def _inner_preflight(
    training: pd.DataFrame, config: BenchmarkConfig, scope: str
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, str]]]:
    normalized = training.reset_index(drop=True).copy(deep=True)
    split = config.split
    try:
        folds = build_purged_walk_forward_splits(
            normalized,
            n_splits=split.inner_n_splits,
            test_date_count=split.inner_validation_date_count,
            min_train_date_count=split.inner_min_train_date_count,
        )
    except ValueError as error:
        return [], [{"scope": f"{scope}.inner", "message": str(error)}], []
    reports = []
    warnings = []
    for number, fold in enumerate(folds, start=1):
        fitted_universe = expand_training_cross_sections(
            normalized,
            training_indices=fold.train_indices,
            outcome_columns=(config.target_column, config.realized_return_column),
        )
        validation = normalized.loc[list(fold.test_indices)].copy(deep=True)
        evaluation = _outcome_availability(validation, config)
        reports.append(
            {
                "fold": number,
                "training": _training_summary(
                    normalized, fitted_universe, fold.test_start_date, config
                ),
                "evaluation": evaluation,
            }
        )
        warnings.extend(_period_warnings(f"{scope}.inner_{number}", evaluation, config))
    return reports, [], warnings


def _training_universe(
    frame: pd.DataFrame, row_ids: tuple[str, ...], config: BenchmarkConfig
) -> pd.DataFrame:
    return expand_training_cross_sections(
        frame,
        training_indices=tuple(frame.index[frame["row_id"].isin(row_ids)]),
        outcome_columns=(config.target_column, config.realized_return_column),
    )


def _fold_reports(
    plan: Stage3DataPlan, config: BenchmarkConfig
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, str]]]:
    frame = plan.panel.frame
    reports, errors, warnings = [], [], []
    for number, fold in enumerate(plan.development_folds, start=1):
        training = _training_universe(frame, fold.train_row_ids, config)
        inner, inner_errors, inner_warnings = _inner_preflight(
            training, config, f"development_fold_{number}"
        )
        evaluation = _outcome_availability(
            evaluation_cross_section(
                plan.panel,
                start_date=fold.evaluation_start_date,
                end_date=fold.evaluation_end_date,
                outcome_columns=(config.target_column, config.realized_return_column),
                label_before=plan.locked_test.test_start_date,
            ),
            config,
        )
        reports.append(
            {
                "scope": "development",
                "fold": number,
                "evaluation_start": fold.evaluation_start_date.date().isoformat(),
                "training": _training_summary(
                    frame, training, fold.evaluation_start_date, config
                ),
                "evaluation": evaluation,
                "inner_folds": inner,
            }
        )
        errors.extend(inner_errors)
        warnings.extend(inner_warnings)
        warnings.extend(
            _period_warnings(f"development_fold_{number}", evaluation, config)
        )
    final_report, final_errors, final_warnings = _final_training_report(plan, config)
    return (
        [*reports, final_report],
        [*errors, *final_errors],
        [*warnings, *final_warnings],
    )


def _final_training_report(
    plan: Stage3DataPlan, config: BenchmarkConfig
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    frame = plan.panel.frame
    training = _training_universe(frame, plan.locked_test.training_row_ids, config)
    inner, inner_errors, inner_warnings = _inner_preflight(
        training, config, "final_training"
    )
    return (
        {
            "scope": "final_training",
            "evaluation_start": plan.locked_test.test_start_date.date().isoformat(),
            "training": _training_summary(
                frame, training, plan.locked_test.test_start_date, config
            ),
            "inner_folds": inner,
        },
        inner_errors,
        inner_warnings,
    )


def _base_report() -> dict[str, Any]:
    return {
        "schema_version": "1",
        "feasible": False,
        "errors": [],
        "warnings": [],
        "summary": {
            "claim_scope": "structural_preflight_only",
            "no_training_performed": True,
            "no_predictive_outcomes_computed": True,
            "provider_provenance_verified": False,
            "stable_identifiers_verified": False,
            "corporate_actions_verified": False,
            "delisting_policy_verified": False,
            "stage4_eligible": False,
            "limitations": (
                "Feasible means the requested outer and inner split schedules exist, "
                "not that screening, fitting or empirical acceptance will succeed. "
                "The entire input is structurally validated and final dates are "
                "reserved using label availability; this is not a secrecy boundary. "
                "Expected label rows are the scoring universe, not filled labels. "
                "Missing-label, missing-target and overlap counts may overlap."
            ),
        },
        "folds": [],
    }


def _validated_frame(panel: pd.DataFrame, config: BenchmarkConfig) -> pd.DataFrame:
    if not isinstance(panel, pd.DataFrame):
        raise ValueError("panel must be a pandas DataFrame.")
    if not panel.columns.is_unique:
        raise ValueError("Panel column names must be unique.")
    validated = validate_benchmark_panel(
        panel,
        feature_columns=config.feature_columns,
        target_column=config.target_column,
    ).frame
    column = config.realized_return_column
    if column not in validated.columns:
        raise ValueError(f"Missing realized return column: {column}")
    validated[column] = _numeric_values(validated[column], name=column)
    return validated


def preflight_benchmark(
    panel: pd.DataFrame, *, config: BenchmarkConfig
) -> dict[str, Any]:
    """Return a strict-JSON structural report without training or outcome scoring.

    Finite reserved outcome values do not affect this report. Changing their
    availability or dates can change reservation and structural feasibility.
    The returned report contains no per-security rows or final-test performance.
    """
    if not isinstance(config, BenchmarkConfig):
        raise ValueError("config must be a BenchmarkConfig.")
    report = _base_report()
    try:
        frame = _validated_frame(panel, config)
    except (ValueError, TypeError, OverflowError) as error:
        return {**report, "errors": [{"scope": "panel", "message": str(error)}]}
    try:
        plan = build_stage3_data_plan(
            frame,
            feature_columns=config.feature_columns,
            target_column=config.target_column,
            locked_test_date_count=config.split.final_test_date_count,
            locked_min_cross_section=config.min_cross_section,
            n_splits=config.split.outer_n_splits,
            evaluation_date_count=config.split.outer_test_date_count,
            min_train_date_count=config.split.outer_min_train_date_count,
        )
    except ValueError as error:
        return {**report, "errors": [{"scope": "split_plan", "message": str(error)}]}
    return _planned_report(report, plan, config)


def _planning_warnings(
    coverage: dict[str, Any], config: BenchmarkConfig
) -> list[dict[str, str]]:
    warnings = _hac_warnings(
        "locked_schedule", config.split.final_test_date_count, config
    )
    for feature, value in coverage.items():
        if value is not None and value < config.minimum_coverage:
            warnings.append(
                _warning(
                    "development",
                    "low_feature_coverage",
                    (
                        f"{feature} coverage {value:.3f} is below the configured "
                        "screening threshold; this diagnostic does not replace "
                        "fold-local screening."
                    ),
                )
            )
    if config.split.final_test_date_count < max(
        config.minimum_locked_test_date_count, config.minimum_locked_spread_date_count
    ):
        warnings.append(
            _warning(
                "locked_interval",
                "acceptance_date_count_unreachable",
                (
                    "The scheduled final-test dates are fewer than a configured "
                    "acceptance minimum. A full run cannot pass that date-count gate."
                ),
            )
        )
    return warnings


def _planned_report(
    report: dict[str, Any], plan: Stage3DataPlan, config: BenchmarkConfig
) -> dict[str, Any]:
    frame = plan.panel.frame
    locked = plan.locked_test
    development = frame.loc[frame["as_of_date"] < locked.test_start_date].copy(
        deep=True
    )
    authorized = development["label_end_date"] < locked.test_start_date
    outcomes = list(
        dict.fromkeys((config.target_column, config.realized_return_column))
    )
    development.loc[~authorized, outcomes] = float("nan")
    reserved = frame.loc[
        frame["as_of_date"].between(locked.test_start_date, locked.test_end_date)
    ]
    folds, errors, warnings = _fold_reports(plan, config)
    coverage = _feature_coverage(development, config)
    return {
        **report,
        "feasible": not errors,
        "errors": errors,
        "warnings": [*warnings, *_planning_warnings(coverage, config)],
        "summary": {
            **report["summary"],
            "development": {
                **_outcome_availability(development, config),
                "feature_coverage": coverage,
            },
            "locked_interval": _period(reserved),
            "hac": {
                "requested_lags": config.hac_lags,
                "locked_scheduled_date_count": config.split.final_test_date_count,
                "locked_maximum_effective_lags": _hac_bounds(
                    config.split.final_test_date_count, config.hac_lags
                )["maximum_effective_lags"],
                "interpretation": (
                    "Locked bounds use the schedule only, not evaluated outcomes. "
                    "Missing metric dates withhold inference; effective lags "
                    "equal the request or are unavailable."
                ),
            },
        },
        "folds": folds,
    }
