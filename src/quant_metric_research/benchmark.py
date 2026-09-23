from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from . import benchmark_schedules
from .baselines import fit_non_ml_baselines, predict_non_ml_baselines
from .benchmark_config import BenchmarkConfig, NestedSplitConfig
from .benchmark_data import (
    Stage3DataPlan,
    evaluation_cross_section,
    expand_training_cross_sections,
)
from .benchmark_metrics import (
    PredictionEvaluation,
    evaluate_prediction_frame,
)
from .benchmark_models import ModelCandidate, fit_candidate
from .benchmark_reporting import (
    IMPLEMENTATION_VERSION as IMPLEMENTATION_VERSION,
)
from .benchmark_reporting import _fingerprints
from .benchmark_tuning import (
    _tune_model_families,
    fit_fold_screen,
    screen_records,
)
from .contracts import _daily_dates
from .scheduled_inference import (
    HACMeanResult,
    inference_diagnostics,
    scheduled_newey_west_mean,
)
from .screening import MetricScreenResult
from .statistics import _numeric_observations, _validate_hac_lags

if TYPE_CHECKING:
    from .experiment_registry import ExperimentRegistry


# Keep the existing private entry point while sharing one pure implementation.
_phase_schedule = benchmark_schedules.phase_schedule


@dataclass(frozen=True)
class BenchmarkRun:
    data_gate: MappingProxyType[str, Any]
    fold_assignments: pd.DataFrame
    predictions: pd.DataFrame
    daily_metrics: pd.DataFrame
    fold_metrics: pd.DataFrame
    tuning_trials: pd.DataFrame
    screening_by_fold: pd.DataFrame
    summary: pd.DataFrame
    acceptance: MappingProxyType[str, Any]
    manifest: MappingProxyType[str, Any]


def _validate_realized_return(panel: pd.DataFrame, config: BenchmarkConfig) -> None:
    column = config.realized_return_column
    if column not in panel.columns:
        raise ValueError(f"Missing realized return column: {column}")
    original = panel[column]
    numeric = pd.to_numeric(original, errors="coerce")
    if (original.notna() & numeric.isna()).any():
        raise ValueError("realized_return_column contains non-numeric values.")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("realized_return_column contains infinite values.")


def _usable_baseline_features(screen: MetricScreenResult) -> tuple[str, ...]:
    ic = screen.ic_summary.set_index("feature")
    usable = tuple(
        feature
        for feature in screen.selected_features
        if feature in ic.index
        and np.isfinite(float(ic.at[feature, "mean_rank_ic"]))
        and not np.isclose(float(ic.at[feature, "mean_rank_ic"]), 0.0, atol=1e-12)
    )
    if not usable:
        raise ValueError("No selected feature has a learnable baseline direction.")
    return usable


def _prediction_targets(
    evaluation: pd.DataFrame,
    *,
    config: BenchmarkConfig,
) -> pd.DataFrame:
    values = evaluation.loc[:, ["row_id", "as_of_date", "symbol"]].copy(deep=True)
    values["target"] = pd.to_numeric(evaluation[config.target_column], errors="coerce")
    values["realized_return"] = pd.to_numeric(
        evaluation[config.realized_return_column], errors="coerce"
    )
    return values


def _baseline_predictions(
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    screen: MetricScreenResult,
    config: BenchmarkConfig,
    phase: str,
    fold: int,
    evaluation_start: pd.Timestamp,
) -> pd.DataFrame:
    features = _usable_baseline_features(screen)
    fitted = fit_non_ml_baselines(
        training,
        feature_columns=features,
        target_column=config.target_column,
        min_cross_section=config.min_cross_section,
    )
    predictions = predict_non_ml_baselines(fitted, evaluation)
    targets = _prediction_targets(evaluation, config=config)
    enriched = predictions.merge(targets, on=["as_of_date", "symbol"], how="left")
    enriched["phase"] = phase
    enriched["fold"] = fold
    enriched["candidate_id"] = pd.NA
    enriched["fit_end_date"] = fitted.fitted_through
    enriched["train_label_end_max"] = pd.Timestamp(training["label_end_date"].max())
    enriched["evaluation_start"] = evaluation_start
    enriched["selected_features"] = json.dumps(
        list(features), ensure_ascii=False, separators=(",", ":")
    )
    enriched["selected_feature_count"] = enriched["model"].map(
        lambda model: len(features) if model == "equal_weight_rank" else 1
    )
    enriched["zero_observed_features"] = enriched["feature_count"].eq(0)
    enriched["best_metric_feature"] = fitted.best_feature
    return enriched


def _model_predictions(
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    screen: MetricScreenResult,
    candidate: ModelCandidate,
    config: BenchmarkConfig,
    phase: str,
    fold: int,
    evaluation_start: pd.Timestamp,
) -> pd.DataFrame:
    fitted = fit_candidate(
        training,
        feature_columns=screen.selected_features,
        target_column=config.target_column,
        candidate=candidate,
        rank_features=config.rank_features,
        random_seed=config.random_seed,
        hist_max_iter=config.hist_max_iter,
        hist_min_samples_leaf=config.hist_min_samples_leaf,
    )
    targets = _prediction_targets(evaluation, config=config)
    targets["model"] = candidate.family
    observed_feature_count = (
        evaluation.loc[:, list(screen.selected_features)].notna().sum(axis=1)
    )
    scores = fitted.predict(evaluation)
    scores.loc[observed_feature_count == 0] = float("nan")
    targets["score"] = scores.to_numpy(dtype=float)
    targets["baseline_feature"] = pd.NA
    targets["feature_count"] = observed_feature_count.to_numpy(dtype=int)
    targets["selected_feature_count"] = len(screen.selected_features)
    targets["zero_observed_features"] = observed_feature_count.eq(0).to_numpy(
        dtype=bool
    )
    targets["phase"] = phase
    targets["fold"] = fold
    targets["candidate_id"] = candidate.candidate_id
    targets["fit_end_date"] = fitted.fitted_through
    targets["train_label_end_max"] = fitted.train_label_end_max
    targets["evaluation_start"] = evaluation_start
    targets["selected_features"] = json.dumps(
        list(screen.selected_features), ensure_ascii=False, separators=(",", ":")
    )
    targets["best_metric_feature"] = pd.NA
    return targets


def _fit_evaluation_block(
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    config: BenchmarkConfig,
    phase: str,
    fold: int,
    families: tuple[str, ...],
    evaluation_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    screen = fit_fold_screen(training, config=config)
    if not screen.selected_features:
        raise ValueError("No features survive screening for the final fold fit.")
    predictions = [
        _baseline_predictions(
            training,
            evaluation,
            screen=screen,
            config=config,
            phase=phase,
            fold=fold,
            evaluation_start=evaluation_start,
        )
    ]
    trial_frames: list[pd.DataFrame] = []
    screen_frames: list[pd.DataFrame] = [
        screen_records(
            screen,
            phase=phase,
            outer_fold=fold,
            inner_fold="final_fit",
        ).assign(family="all", fit_kind="final_fit")
    ]
    for family, tuned in _tune_model_families(
        training,
        config=config,
        families=families,
        phase=phase,
        outer_fold=fold,
    ):
        predictions.append(
            _model_predictions(
                training,
                evaluation,
                screen=screen,
                candidate=tuned.selected_candidate,
                config=config,
                phase=phase,
                fold=fold,
                evaluation_start=evaluation_start,
            )
        )
        trial_frames.append(tuned.trials)
        screen_frames.append(
            tuned.screening.assign(family=family, fit_kind="inner_tuning")
        )
    return (
        pd.concat(predictions, ignore_index=True),
        pd.concat(trial_frames, ignore_index=True),
        pd.concat(screen_frames, ignore_index=True),
    )


def _training_universe(
    plan: Stage3DataPlan,
    row_ids: tuple[str, ...],
    config: BenchmarkConfig,
) -> pd.DataFrame:
    frame = plan.panel.frame
    return expand_training_cross_sections(
        frame,
        training_indices=tuple(frame.index[frame["row_id"].isin(row_ids)]),
        outcome_columns=(config.target_column, config.realized_return_column),
    )


def _development_run(
    plan: Stage3DataPlan,
    *,
    config: BenchmarkConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_frames: list[pd.DataFrame] = []
    trial_frames: list[pd.DataFrame] = []
    screen_frames: list[pd.DataFrame] = []
    for exposed_fold, fold in enumerate(plan.development_folds, start=1):
        training = _training_universe(plan, fold.train_row_ids, config)
        evaluation = evaluation_cross_section(
            plan.panel,
            start_date=fold.evaluation_start_date,
            end_date=fold.evaluation_end_date,
            outcome_columns=(config.target_column, config.realized_return_column),
            label_before=plan.locked_test.test_start_date,
        )
        predictions, trials, screens = _fit_evaluation_block(
            training,
            evaluation,
            config=config,
            phase="development",
            fold=exposed_fold,
            families=config.model_families,
            evaluation_start=fold.evaluation_start_date,
        )
        prediction_frames.append(predictions)
        trial_frames.append(trials)
        screen_frames.append(screens)
    return (
        pd.concat(prediction_frames, ignore_index=True),
        pd.concat(trial_frames, ignore_index=True),
        pd.concat(screen_frames, ignore_index=True),
    )


def _locked_run(
    plan: Stage3DataPlan,
    *,
    config: BenchmarkConfig,
    frozen_family: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    training = _training_universe(plan, plan.locked_test.training_row_ids, config)
    evaluation = evaluation_cross_section(
        plan.panel,
        start_date=plan.locked_test.test_start_date,
        end_date=plan.locked_test.test_end_date,
        outcome_columns=(config.target_column, config.realized_return_column),
    )
    locked_fold = config.split.outer_n_splits + 1
    return _fit_evaluation_block(
        training,
        evaluation,
        config=config,
        phase="locked_test",
        fold=locked_fold,
        families=(frozen_family,),
        evaluation_start=plan.locked_test.test_start_date,
    )


def _assignment_frame(
    plan: Stage3DataPlan, config: BenchmarkConfig, *, include_lockbox: bool = True
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for exposed_fold, fold in enumerate(plan.development_folds, start=1):
        for assignment in fold.assignments:
            rows.append(
                {
                    "phase": "development",
                    "fold": exposed_fold,
                    "split_id": assignment.split_id,
                    "row_id": assignment.row_id,
                    "role": assignment.role,
                    "exclusion_reason": assignment.exclusion_reason,
                    "train_end_date": fold.train_end_date,
                    "train_label_end_max": fold.training_label_end_max,
                    "evaluation_start": fold.evaluation_start_date,
                    "evaluation_end": fold.evaluation_end_date,
                }
            )
    if not include_lockbox:
        frame = plan.panel.frame
        development_ids = frame.loc[
            frame["as_of_date"] < plan.locked_test.test_start_date, "row_id"
        ]
        return (
            pd.DataFrame(rows)
            .loc[lambda values: values["row_id"].isin(development_ids)]
            .reset_index(drop=True)
        )
    locked = plan.locked_test
    locked_training = plan.panel.rows(locked.training_row_ids)
    train_end = pd.Timestamp(locked_training["as_of_date"].max())
    label_end_max = pd.Timestamp(locked_training["label_end_date"].max())
    locked_fold = config.split.outer_n_splits + 1
    for assignment in locked.assignments:
        rows.append(
            {
                "phase": "locked_test",
                "fold": locked_fold,
                "split_id": assignment.split_id,
                "row_id": assignment.row_id,
                "role": assignment.role,
                "exclusion_reason": assignment.exclusion_reason,
                "train_end_date": train_end,
                "train_label_end_max": label_end_max,
                "evaluation_start": locked.test_start_date,
                "evaluation_end": locked.test_end_date,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["phase", "fold", "split_id", "row_id"], kind="stable")
        .reset_index(drop=True)
    )


def _evaluate_development(
    predictions: pd.DataFrame,
    *,
    config: BenchmarkConfig,
    expected_dates: Sequence[object] | None = None,
) -> PredictionEvaluation:
    return evaluate_prediction_frame(
        predictions,
        min_cross_section=config.min_cross_section,
        quantiles=config.quantiles,
        hac_lags=config.hac_lags,
        primary_models=(*config.model_families, config.primary_baseline),
        expected_dates_by_phase=(
            {"development": expected_dates} if expected_dates is not None else None
        ),
    )


def _summary_row(
    summary: pd.DataFrame,
    *,
    phase: str,
    model: str,
    evaluation_scope: str = "common",
) -> pd.Series:
    rows = summary.loc[
        (summary["phase"] == phase)
        & (summary["model"] == model)
        & (summary["evaluation_scope"] == evaluation_scope)
    ]
    if rows.shape[0] != 1:
        raise ValueError(
            f"Missing unique {evaluation_scope} summary for {phase}/{model}."
        )
    return rows.iloc[0]


def _paired_locked_rank_ic_improvement(
    daily_metrics: pd.DataFrame,
    *,
    model: str,
    baseline: str,
    hac_lags: int,
    expected_locked_dates: Sequence[object] | None = None,
) -> tuple[float, HACMeanResult]:
    _validate_hac_lags(hac_lags)
    locked_common = daily_metrics.loc[
        (daily_metrics["phase"] == "locked_test")
        & (daily_metrics["evaluation_scope"] == "common")
        & daily_metrics["model"].isin([model, baseline]),
        ["fold", "as_of_date", "model", "rank_ic"],
    ]
    locked_common = locked_common.assign(
        as_of_date=_daily_dates(locked_common["as_of_date"], field="as_of_date"),
        rank_ic=_numeric_observations(locked_common["rank_ic"]),
    )
    if locked_common.duplicated(["as_of_date", "model"]).any():
        raise ValueError("Paired IC values must be unique by date/model across folds.")
    pivot = locked_common.pivot(
        index="as_of_date",
        columns="model",
        values="rank_ic",
    )
    paired = pivot.reindex(columns=[model, baseline])
    improvements = paired[model] - paired[baseline]
    valid = improvements.dropna()
    mean_improvement = float(valid.mean()) if not valid.empty else float("nan")
    inference = (
        scheduled_newey_west_mean(
            improvements, expected_dates=expected_locked_dates, hac_lags=hac_lags
        )
        if expected_locked_dates is not None
        else HACMeanResult(
            status="schedule_unavailable",
            t_stat=None,
            p_value=None,
            scheduled_count=None,
            observed_count=int(valid.shape[0]),
            requested_lags=hac_lags,
            effective_lags=None,
        )
    )
    return mean_improvement, inference


def _acceptance(
    evaluation: PredictionEvaluation,
    *,
    config: BenchmarkConfig,
    frozen_family: str,
    expected_locked_dates: Sequence[object] | None = None,
) -> MappingProxyType[str, Any]:
    model_dev = _summary_row(
        evaluation.summary, phase="development", model=frozen_family
    )
    model_locked = _summary_row(
        evaluation.summary, phase="locked_test", model=frozen_family
    )
    model_locked_native = _summary_row(
        evaluation.summary,
        phase="locked_test",
        model=frozen_family,
        evaluation_scope="native",
    )
    baseline_locked_native = _summary_row(
        evaluation.summary,
        phase="locked_test",
        model=config.primary_baseline,
        evaluation_scope="native",
    )
    development = evaluation.fold_metrics.loc[
        (evaluation.fold_metrics["phase"] == "development")
        & (evaluation.fold_metrics["evaluation_scope"] == "common")
        & evaluation.fold_metrics["model"].isin(
            [frozen_family, config.primary_baseline]
        )
    ]
    fold_pivot = development.pivot(
        index="fold", columns="model", values="mean_rank_ic"
    ).dropna()
    development_win_rate = (
        float(
            (
                fold_pivot[frozen_family] - fold_pivot[config.primary_baseline]
                > config.minimum_rank_ic_improvement
            ).mean()
        )
        if not fold_pivot.empty
        else 0.0
    )
    locked_native_coverage = float(model_locked_native["mean_score_coverage"])
    baseline_native_coverage = float(baseline_locked_native["mean_score_coverage"])
    coverage_ratio = (
        locked_native_coverage / baseline_native_coverage
        if baseline_native_coverage > 0
        else 0.0
    )
    locked_native_spread_coverage = float(model_locked_native["mean_spread_coverage"])
    locked_spread_date_count = int(model_locked["spread_date_count"])
    locked_rank_ic_improvement, inference = _paired_locked_rank_ic_improvement(
        evaluation.daily_metrics,
        model=frozen_family,
        baseline=config.primary_baseline,
        hac_lags=config.hac_lags,
        expected_locked_dates=expected_locked_dates,
    )
    locked_valid_date_count = inference.observed_count
    locked_rank_ic_improvement_t_stat = inference.t_stat
    locked_rank_ic_improvement_p_value = inference.p_value
    checks = {
        "development_mean_rank_ic_positive": float(model_dev["mean_rank_ic"]) > 0,
        "development_median_rank_ic_positive": float(model_dev["median_rank_ic"]) > 0,
        "locked_mean_rank_ic_positive": float(model_locked["mean_rank_ic"]) > 0,
        "locked_median_rank_ic_positive": float(model_locked["median_rank_ic"]) > 0,
        "locked_spread_positive": float(model_locked["mean_spread"]) > 0,
        "locked_rank_ic_beats_baseline": bool(
            np.isfinite(locked_rank_ic_improvement)
            and locked_rank_ic_improvement > config.minimum_rank_ic_improvement
        ),
        "development_win_rate_passed": (
            development_win_rate >= config.minimum_development_win_rate
        ),
        "locked_native_coverage_ratio_passed": (
            coverage_ratio >= config.minimum_coverage_ratio
        ),
        "locked_native_score_coverage_passed": (
            locked_native_coverage >= config.minimum_locked_score_coverage
        ),
        "locked_valid_date_count_passed": (
            locked_valid_date_count >= config.minimum_locked_test_date_count
        ),
        "locked_spread_date_count_passed": (
            locked_spread_date_count >= config.minimum_locked_spread_date_count
        ),
        "locked_native_spread_coverage_passed": (
            locked_native_spread_coverage >= config.minimum_locked_spread_coverage
        ),
        "locked_rank_ic_improvement_p_value_passed": bool(
            locked_rank_ic_improvement_p_value is not None
            and np.isfinite(locked_rank_ic_improvement_p_value)
            and locked_rank_ic_improvement_p_value
            <= config.maximum_locked_rank_ic_improvement_p_value
        ),
    }
    model_gate_passed = all(bool(value) for value in checks.values())
    return MappingProxyType(
        {
            "lockbox_evaluated_once_in_this_run": True,
            "lockbox_reuse_registry_enforced": False,
            "frozen_model_family": frozen_family,
            "primary_baseline": config.primary_baseline,
            "development_win_rate": development_win_rate,
            "locked_native_coverage_ratio": coverage_ratio,
            "locked_native_score_coverage": locked_native_coverage,
            "locked_native_spread_coverage": locked_native_spread_coverage,
            "locked_valid_date_count": locked_valid_date_count,
            "locked_spread_date_count": locked_spread_date_count,
            "locked_rank_ic_improvement": locked_rank_ic_improvement,
            "locked_rank_ic_improvement_t_stat": (locked_rank_ic_improvement_t_stat),
            "locked_rank_ic_improvement_p_value": (locked_rank_ic_improvement_p_value),
            **{
                f"locked_rank_ic_improvement_{key}": value
                for key, value in inference_diagnostics(inference).items()
            },
            "checks": checks,
            "model_gate_passed": model_gate_passed,
            "eligible_for_stage4_data_review": model_gate_passed,
            "empirical_data_provenance_verified": False,
            "stage4_eligible": False,
            "stage4_blocker": (
                "Research-grade provider, identifier, corporate-action, and "
                "delisting policies require independent verification."
            ),
        }
    )


def _sort_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "phase",
        "fold",
        "as_of_date",
        "symbol",
        "row_id",
        "model",
        "score",
        "target",
        "realized_return",
        "candidate_id",
        "fit_end_date",
        "train_label_end_max",
        "evaluation_start",
        "selected_features",
        "best_metric_feature",
        "baseline_feature",
        "feature_count",
        "selected_feature_count",
        "zero_observed_features",
    ]
    return (
        predictions.loc[:, columns]
        .sort_values(["phase", "fold", "as_of_date", "symbol", "model"], kind="stable")
        .reset_index(drop=True)
    )


def _development_result(
    plan: Stage3DataPlan,
    *,
    config: BenchmarkConfig,
    predictions: pd.DataFrame,
    trials: pd.DataFrame,
    screening: pd.DataFrame,
    evaluation: PredictionEvaluation,
    frozen_family: str,
) -> BenchmarkRun:
    return BenchmarkRun(
        data_gate=MappingProxyType(
            {
                "structural_contract_passed": True,
                "locked_test_date_count": config.split.final_test_date_count,
                "point_in_time_provider_verified": False,
                "stable_identifier_policy_verified": False,
                "corporate_action_policy_verified": False,
                "delisting_return_policy_verified": False,
                "claim_scope": "development_only",
            }
        ),
        fold_assignments=_assignment_frame(plan, config, include_lockbox=False),
        predictions=_sort_predictions(predictions),
        daily_metrics=evaluation.daily_metrics,
        fold_metrics=evaluation.fold_metrics,
        tuning_trials=trials,
        screening_by_fold=screening,
        summary=evaluation.summary,
        acceptance=MappingProxyType(
            {
                "acceptance_status": "not_evaluated",
                "lockbox_evaluated_once_in_this_run": False,
                "lockbox_reuse_registry_enforced": False,
                "frozen_model_family": frozen_family,
                "primary_baseline": config.primary_baseline,
                "model_gate_passed": False,
                "eligible_for_stage4_data_review": False,
                "empirical_data_provenance_verified": False,
                "stage4_eligible": False,
                "stage4_blocker": (
                    "Final-test evaluation and data provenance review pending."
                ),
            }
        ),
        manifest=_fingerprints(plan, config=config, evaluate_lockbox=False),
    )


def run_stage3_benchmark(
    panel: pd.DataFrame,
    *,
    config: BenchmarkConfig,
    evaluate_lockbox: bool = False,
    registry: ExperimentRegistry | None = None,
    study_id: str | None = None,
    hypothesis: str | None = None,
    development_run_id: str | None = None,
) -> BenchmarkRun:
    """Develop by default; final evaluation requires persisted matching evidence.

    The local registry guards accidental reuse, not raw-data access or deliberate
    bypass. Register development with a study and hypothesis to enable a final run.
    """
    from .experiment_workflow import run_experiment

    return run_experiment(
        panel,
        config=config,
        evaluate_lockbox=evaluate_lockbox,
        registry=registry,
        study_id=study_id,
        hypothesis=hypothesis,
        development_run_id=development_run_id,
    )


__all__ = [
    "BenchmarkConfig",
    "BenchmarkRun",
    "NestedSplitConfig",
    "run_stage3_benchmark",
]
