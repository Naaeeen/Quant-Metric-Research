from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from ._version import __version__
from .baselines import fit_non_ml_baselines, predict_non_ml_baselines
from .benchmark_config import BenchmarkConfig, NestedSplitConfig
from .benchmark_data import Stage3DataPlan, build_stage3_data_plan
from .benchmark_metrics import (
    PredictionEvaluation,
    choose_frozen_model,
    evaluate_prediction_frame,
)
from .benchmark_models import ModelCandidate, fit_candidate
from .benchmark_tuning import (
    fit_fold_screen,
    screen_records,
    tune_model_family,
)
from .screening import MetricScreenResult

IMPLEMENTATION_VERSION = __version__
ARTIFACT_SCHEMA_VERSION = "1"


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
    targets["score"] = fitted.predict(evaluation).to_numpy(dtype=float)
    targets["baseline_feature"] = pd.NA
    targets["feature_count"] = len(screen.selected_features)
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
    for family in families:
        tuned = tune_model_family(
            training,
            config=config,
            family=family,
            phase=phase,
            outer_fold=fold,
        )
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


def _development_run(
    plan: Stage3DataPlan,
    *,
    config: BenchmarkConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_frames: list[pd.DataFrame] = []
    trial_frames: list[pd.DataFrame] = []
    screen_frames: list[pd.DataFrame] = []
    for exposed_fold, fold in enumerate(plan.development_folds, start=1):
        training = plan.panel.rows(fold.train_row_ids)
        evaluation = plan.panel.rows(fold.evaluation_row_ids)
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
    training = plan.panel.rows(plan.locked_test.training_row_ids)
    evaluation = plan.panel.rows(plan.locked_test.test_row_ids)
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


def _assignment_frame(plan: Stage3DataPlan, config: BenchmarkConfig) -> pd.DataFrame:
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
) -> PredictionEvaluation:
    return evaluate_prediction_frame(
        predictions,
        min_cross_section=config.min_cross_section,
        quantiles=config.quantiles,
        hac_lags=config.hac_lags,
        primary_models=(*config.model_families, config.primary_baseline),
    )


def _summary_row(
    summary: pd.DataFrame,
    *,
    phase: str,
    model: str,
) -> pd.Series:
    rows = summary.loc[
        (summary["phase"] == phase)
        & (summary["model"] == model)
        & (summary["evaluation_scope"] == "common")
    ]
    if rows.shape[0] != 1:
        raise ValueError(f"Missing unique common summary for {phase}/{model}.")
    return rows.iloc[0]


def _acceptance(
    evaluation: PredictionEvaluation,
    *,
    config: BenchmarkConfig,
    frozen_family: str,
) -> MappingProxyType[str, Any]:
    model_dev = _summary_row(
        evaluation.summary, phase="development", model=frozen_family
    )
    model_locked = _summary_row(
        evaluation.summary, phase="locked_test", model=frozen_family
    )
    baseline_locked = _summary_row(
        evaluation.summary,
        phase="locked_test",
        model=config.primary_baseline,
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
                >= config.minimum_rank_ic_improvement
            ).mean()
        )
        if not fold_pivot.empty
        else 0.0
    )
    coverage_ratio = (
        float(model_locked["mean_score_coverage"])
        / float(baseline_locked["mean_score_coverage"])
        if float(baseline_locked["mean_score_coverage"]) > 0
        else 0.0
    )
    checks = {
        "development_mean_rank_ic_positive": float(model_dev["mean_rank_ic"]) > 0,
        "development_median_rank_ic_positive": float(model_dev["median_rank_ic"]) > 0,
        "locked_mean_rank_ic_positive": float(model_locked["mean_rank_ic"]) > 0,
        "locked_median_rank_ic_positive": float(model_locked["median_rank_ic"]) > 0,
        "locked_spread_positive": float(model_locked["mean_spread"]) > 0,
        "locked_rank_ic_beats_baseline": (
            float(model_locked["mean_rank_ic"]) - float(baseline_locked["mean_rank_ic"])
            >= config.minimum_rank_ic_improvement
        ),
        "development_win_rate_passed": (
            development_win_rate >= config.minimum_development_win_rate
        ),
        "coverage_ratio_passed": coverage_ratio >= config.minimum_coverage_ratio,
    }
    model_gate_passed = all(bool(value) for value in checks.values())
    return MappingProxyType(
        {
            "locked_test_used_once": True,
            "frozen_model_family": frozen_family,
            "primary_baseline": config.primary_baseline,
            "development_win_rate": development_win_rate,
            "locked_coverage_ratio": coverage_ratio,
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


def _fingerprints(
    plan: Stage3DataPlan,
    *,
    config: BenchmarkConfig,
) -> MappingProxyType[str, Any]:
    frame = plan.panel.frame
    relevant = [
        "row_id",
        "as_of_date",
        "symbol",
        "label_end_date",
        *config.feature_columns,
        config.target_column,
    ]
    if config.realized_return_column not in relevant:
        relevant.append(config.realized_return_column)
    model_frame = frame.loc[:, relevant]
    model_hashes = pd.util.hash_pandas_object(model_frame, index=False)
    model_input_fingerprint = sha256(model_hashes.to_numpy().tobytes()).hexdigest()
    contract_columns = sorted(str(column) for column in frame.columns)
    contract_frame = frame.loc[:, contract_columns].sort_values("row_id", kind="stable")
    panel_hasher = sha256()
    for column in contract_columns:
        panel_hasher.update(column.encode("utf-8"))
        panel_hasher.update(b"\0")
        panel_hasher.update(str(contract_frame[column].dtype).encode("utf-8"))
        panel_hasher.update(b"\0")
    contract_hashes = pd.util.hash_pandas_object(contract_frame, index=False)
    panel_hasher.update(contract_hashes.to_numpy().tobytes())
    panel_fingerprint = panel_hasher.hexdigest()
    source_hasher = sha256()
    package_directory = Path(__file__).resolve().parent
    for source_path in sorted(package_directory.glob("*.py")):
        source_hasher.update(source_path.name.encode("utf-8"))
        source_hasher.update(b"\0")
        source_hasher.update(source_path.read_bytes())
        source_hasher.update(b"\0")
    source_fingerprint = source_hasher.hexdigest()
    config_json = json.dumps(
        config.to_mapping(), sort_keys=True, separators=(",", ":"), default=str
    )
    run_fingerprint = sha256(
        (
            f"{IMPLEMENTATION_VERSION}|{source_fingerprint}|"
            f"{panel_fingerprint}|{config_json}"
        ).encode()
    ).hexdigest()
    dataset_versions = (
        sorted(str(value) for value in frame["dataset_version"].dropna().unique())
        if "dataset_version" in frame.columns
        else []
    )
    return MappingProxyType(
        {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "implementation_version": IMPLEMENTATION_VERSION,
            "package_version": __version__,
            "source_fingerprint": source_fingerprint,
            "fingerprint_scope": ("source_code+configuration+validated_panel_contract"),
            "run_fingerprint": run_fingerprint,
            "panel_fingerprint": panel_fingerprint,
            "model_input_fingerprint": model_input_fingerprint,
            "dataset_versions": dataset_versions,
            "configuration": config.to_mapping(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scikit_learn_version": sklearn.__version__,
            "development_start": str(frame["as_of_date"].min().date()),
            "locked_test_start": str(plan.locked_test.test_start_date.date()),
            "locked_test_end": str(plan.locked_test.test_end_date.date()),
        }
    )


def _data_gate(
    plan: Stage3DataPlan, config: BenchmarkConfig
) -> MappingProxyType[str, Any]:
    panel = plan.panel.frame
    locked = panel.loc[
        (panel["as_of_date"] >= plan.locked_test.test_start_date)
        & (panel["as_of_date"] <= plan.locked_test.test_end_date)
    ].copy(deep=True)
    target_coverage = (
        locked.groupby("as_of_date", sort=True)[config.target_column]
        .apply(lambda values: float(values.notna().mean()))
        .to_dict()
    )
    realized_coverage = (
        locked.groupby("as_of_date", sort=True)[config.realized_return_column]
        .apply(lambda values: float(values.notna().mean()))
        .to_dict()
    )
    cross_section_counts = locked.groupby("as_of_date", sort=True).size().to_dict()
    evaluable_counts = (
        locked.assign(
            _evaluable=(
                locked[config.target_column].notna()
                & locked[config.realized_return_column].notna()
            )
        )
        .groupby("as_of_date", sort=True)["_evaluable"]
        .sum()
        .to_dict()
    )
    return MappingProxyType(
        {
            "structural_contract_passed": True,
            "locked_test_date_count": int(locked["as_of_date"].nunique()),
            "locked_target_coverage_by_date": {
                str(pd.Timestamp(date).date()): coverage
                for date, coverage in target_coverage.items()
            },
            "locked_realized_return_coverage_by_date": {
                str(pd.Timestamp(date).date()): coverage
                for date, coverage in realized_coverage.items()
            },
            "locked_cross_section_count_by_date": {
                str(pd.Timestamp(date).date()): int(count)
                for date, count in cross_section_counts.items()
            },
            "locked_evaluable_count_by_date": {
                str(pd.Timestamp(date).date()): int(count)
                for date, count in evaluable_counts.items()
            },
            "point_in_time_provider_verified": False,
            "stable_identifier_policy_verified": False,
            "corporate_action_policy_verified": False,
            "delisting_return_policy_verified": False,
            "claim_scope": "benchmark_engine_only",
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
    ]
    return (
        predictions.loc[:, columns]
        .sort_values(["phase", "fold", "as_of_date", "symbol", "model"], kind="stable")
        .reset_index(drop=True)
    )


def run_stage3_benchmark(
    panel: pd.DataFrame,
    *,
    config: BenchmarkConfig,
) -> BenchmarkRun:
    if not isinstance(config, BenchmarkConfig):
        raise ValueError("config must be a BenchmarkConfig.")
    _validate_realized_return(panel, config)
    plan = build_stage3_data_plan(
        panel,
        feature_columns=config.feature_columns,
        target_column=config.target_column,
        locked_test_date_count=config.split.final_test_date_count,
        n_splits=config.split.outer_n_splits,
        evaluation_date_count=config.split.outer_test_date_count,
        min_train_date_count=config.split.outer_min_train_date_count,
    )
    development_predictions, development_trials, development_screens = _development_run(
        plan, config=config
    )
    development_evaluation = _evaluate_development(
        development_predictions,
        config=config,
    )
    frozen_family = choose_frozen_model(
        development_evaluation.summary,
        model_families=config.model_families,
    )
    locked_predictions, locked_trials, locked_screens = _locked_run(
        plan,
        config=config,
        frozen_family=frozen_family,
    )
    locked_evaluation = evaluate_prediction_frame(
        locked_predictions,
        min_cross_section=config.min_cross_section,
        quantiles=config.quantiles,
        hac_lags=config.hac_lags,
        primary_models=(frozen_family, config.primary_baseline),
    )
    predictions = _sort_predictions(
        pd.concat([development_predictions, locked_predictions], ignore_index=True)
    )
    evaluation = PredictionEvaluation(
        daily_metrics=pd.concat(
            [
                development_evaluation.daily_metrics,
                locked_evaluation.daily_metrics,
            ],
            ignore_index=True,
        ),
        fold_metrics=pd.concat(
            [
                development_evaluation.fold_metrics,
                locked_evaluation.fold_metrics,
            ],
            ignore_index=True,
        ),
        summary=pd.concat(
            [development_evaluation.summary, locked_evaluation.summary],
            ignore_index=True,
        ),
    )
    tuning_trials = (
        pd.concat([development_trials, locked_trials], ignore_index=True)
        .sort_values(
            ["phase", "outer_fold", "family", "candidate_order"], kind="stable"
        )
        .reset_index(drop=True)
    )
    screening = (
        pd.concat([development_screens, locked_screens], ignore_index=True)
        .sort_values(
            ["phase", "outer_fold", "fit_kind", "family", "inner_fold", "feature"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    return BenchmarkRun(
        data_gate=_data_gate(plan, config),
        fold_assignments=_assignment_frame(plan, config),
        predictions=predictions,
        daily_metrics=evaluation.daily_metrics,
        fold_metrics=evaluation.fold_metrics,
        tuning_trials=tuning_trials,
        screening_by_fold=screening,
        summary=evaluation.summary,
        acceptance=_acceptance(
            evaluation,
            config=config,
            frozen_family=frozen_family,
        ),
        manifest=_fingerprints(plan, config=config),
    )


__all__ = [
    "BenchmarkConfig",
    "BenchmarkRun",
    "NestedSplitConfig",
    "run_stage3_benchmark",
]
