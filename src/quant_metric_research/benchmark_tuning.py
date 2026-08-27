from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .benchmark_config import BenchmarkConfig
from .benchmark_models import ModelCandidate, candidate_specs, fit_candidate
from .screening import MetricScreenResult, fit_metric_screen
from .signals import compute_daily_rank_ic, compute_quantile_spreads
from .splits import build_purged_walk_forward_splits


@dataclass(frozen=True)
class TuningResult:
    selected_candidate: ModelCandidate
    trials: pd.DataFrame
    screening: pd.DataFrame


def fit_fold_screen(
    training: pd.DataFrame,
    *,
    config: BenchmarkConfig,
) -> MetricScreenResult:
    if training.empty:
        raise ValueError("Fold training data must not be empty.")
    return fit_metric_screen(
        training,
        feature_columns=config.feature_columns,
        target_column=config.target_column,
        train_end_date=pd.to_datetime(training["as_of_date"]).max(),
        min_cross_section=config.min_cross_section,
        minimum_coverage=config.minimum_coverage,
        redundancy_threshold=config.redundancy_threshold,
        hac_lags=config.hac_lags,
        quantiles=config.quantiles,
    )


def screen_records(
    screen: MetricScreenResult,
    *,
    phase: str,
    outer_fold: int | str,
    inner_fold: int | str,
) -> pd.DataFrame:
    ic = screen.ic_summary.set_index("feature")
    rows: list[dict[str, object]] = []
    for feature in screen.quality["feature"]:
        rows.append(
            {
                "phase": phase,
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "feature": feature,
                "selected": feature in screen.selected_features,
                "drop_reason": screen.dropped_features.get(feature, pd.NA),
                "train_mean_rank_ic": (
                    float(ic.at[feature, "mean_rank_ic"])
                    if feature in ic.index
                    else float("nan")
                ),
                "fitted_through": screen.fitted_through,
                "training_row_count": screen.training_row_count,
            }
        )
    return pd.DataFrame(rows)


def _inner_folds(
    training: pd.DataFrame,
    *,
    config: BenchmarkConfig,
) -> list[tuple[pd.DataFrame, pd.DataFrame, MetricScreenResult]]:
    normalized = training.reset_index(drop=True).copy(deep=True)
    split = config.split
    folds = build_purged_walk_forward_splits(
        normalized,
        n_splits=split.inner_n_splits,
        test_date_count=split.inner_validation_date_count,
        min_train_date_count=split.inner_min_train_date_count,
    )
    prepared: list[tuple[pd.DataFrame, pd.DataFrame, MetricScreenResult]] = []
    for fold in folds:
        inner_training = normalized.loc[list(fold.train_indices)].copy(deep=True)
        validation = normalized.loc[list(fold.test_indices)].copy(deep=True)
        screen = fit_fold_screen(inner_training, config=config)
        if not screen.selected_features:
            raise ValueError("No features survive screening in an inner fold.")
        prepared.append((inner_training, validation, screen))
    return prepared


def _candidate_validation_scores(
    candidate: ModelCandidate,
    prepared_folds: list[tuple[pd.DataFrame, pd.DataFrame, MetricScreenResult]],
    *,
    config: BenchmarkConfig,
) -> tuple[float, float, int]:
    daily_frames: list[pd.DataFrame] = []
    spread_frames: list[pd.DataFrame] = []
    for inner_training, validation, screen in prepared_folds:
        fitted = fit_candidate(
            inner_training,
            feature_columns=screen.selected_features,
            target_column=config.target_column,
            candidate=candidate,
            rank_features=config.rank_features,
            random_seed=config.random_seed,
            hist_max_iter=config.hist_max_iter,
            hist_min_samples_leaf=config.hist_min_samples_leaf,
        )
        scored = validation.loc[:, ["as_of_date", config.target_column]].copy(deep=True)
        scored["model_score"] = fitted.predict(validation).to_numpy(dtype=float)
        daily_frames.append(
            compute_daily_rank_ic(
                scored,
                feature_columns=("model_score",),
                target_column=config.target_column,
                min_cross_section=config.min_cross_section,
            )
        )
        spread_frames.append(
            compute_quantile_spreads(
                scored,
                feature_columns=("model_score",),
                target_column=config.target_column,
                quantiles=config.quantiles,
                min_cross_section=config.min_cross_section,
            )
        )
    daily = pd.concat(daily_frames, ignore_index=True)
    spreads = pd.concat(spread_frames, ignore_index=True)
    rank_values = pd.to_numeric(daily["rank_ic"], errors="coerce").dropna()
    spread_values = pd.to_numeric(spreads["spread"], errors="coerce").dropna()
    mean_rank_ic = float(rank_values.mean()) if not rank_values.empty else float("nan")
    mean_spread = (
        float(spread_values.mean()) if not spread_values.empty else float("nan")
    )
    return mean_rank_ic, mean_spread, int(rank_values.shape[0])


def tune_model_family(
    training: pd.DataFrame,
    *,
    config: BenchmarkConfig,
    family: str,
    phase: str,
    outer_fold: int | str,
) -> TuningResult:
    prepared = _inner_folds(training, config=config)
    screening = pd.concat(
        [
            screen_records(
                screen,
                phase=phase,
                outer_fold=outer_fold,
                inner_fold=inner_number,
            )
            for inner_number, (_, _, screen) in enumerate(prepared, start=1)
        ],
        ignore_index=True,
    )
    rows: list[dict[str, object]] = []
    candidates = candidate_specs(config, family=family)
    for candidate_order, candidate in enumerate(candidates):
        mean_rank_ic, mean_spread, date_count = _candidate_validation_scores(
            candidate,
            prepared,
            config=config,
        )
        rows.append(
            {
                "phase": phase,
                "outer_fold": outer_fold,
                "family": family,
                "candidate_id": candidate.candidate_id,
                "parameters": dict(candidate.parameters),
                "candidate_order": candidate_order,
                "validation_mean_rank_ic": mean_rank_ic,
                "validation_mean_spread": mean_spread,
                "validation_date_count": date_count,
            }
        )
    trials = pd.DataFrame(rows)
    finite = trials.loc[
        np.isfinite(pd.to_numeric(trials["validation_mean_rank_ic"], errors="coerce"))
    ].copy()
    if finite.empty:
        raise ValueError(f"No valid inner validation Rank IC for {family}.")
    ordered = finite.sort_values(
        ["validation_mean_rank_ic", "candidate_order"],
        ascending=[False, True],
        kind="stable",
    )
    selected_id = str(ordered.iloc[0]["candidate_id"])
    trials["selected"] = trials["candidate_id"] == selected_id
    selected = next(
        candidate for candidate in candidates if candidate.candidate_id == selected_id
    )
    return TuningResult(
        selected_candidate=selected,
        trials=trials,
        screening=screening,
    )
