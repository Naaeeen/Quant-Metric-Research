from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._spreads import fractional_quantile_spread
from .contracts import _daily_dates
from .scheduled_inference import (
    HACMeanResult,
    inference_diagnostics,
    normalize_expected_dates,
    scheduled_newey_west_mean,
)
from .statistics import benjamini_hochberg

PREDICTION_KEYS = ("phase", "fold", "as_of_date", "symbol", "model")


@dataclass(frozen=True)
class PredictionEvaluation:
    daily_metrics: pd.DataFrame
    fold_metrics: pd.DataFrame
    summary: pd.DataFrame


def _validate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(predictions, pd.DataFrame):
        raise ValueError("predictions must be a pandas DataFrame.")
    required = {*PREDICTION_KEYS, "score", "target", "realized_return"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Missing prediction columns: {', '.join(missing)}")
    normalized = predictions.copy(deep=True)
    normalized["as_of_date"] = _daily_dates(
        normalized["as_of_date"], field="Prediction dates"
    )
    for column in ("phase", "symbol", "model"):
        normalized[column] = normalized[column].astype("string").str.strip()
        if normalized[column].isna().any() or (normalized[column] == "").any():
            raise ValueError(f"Prediction {column} values must be non-empty.")
    if normalized.duplicated(list(PREDICTION_KEYS), keep=False).any():
        raise ValueError("Predictions must be unique by phase/fold/date/symbol/model.")
    for column in ("score", "target", "realized_return"):
        original = normalized[column]
        numeric = pd.to_numeric(original, errors="coerce")
        if (original.notna() & numeric.isna()).any():
            raise ValueError(f"Prediction {column} contains non-numeric values.")
        if np.isinf(numeric.to_numpy(dtype=float)).any():
            raise ValueError(f"Prediction {column} contains infinite values.")
        normalized[column] = numeric
    return normalized.sort_values(list(PREDICTION_KEYS), kind="stable").reset_index(
        drop=True
    )


def _daily_row(
    frame: pd.DataFrame,
    *,
    phase: str,
    fold: int,
    as_of_date: pd.Timestamp,
    model: str,
    scope: str,
    min_cross_section: int,
    quantiles: int,
    scoring_universe_count: int,
    eligible_target_count: int,
    eligible_realized_return_count: int,
) -> dict[str, object]:
    scored_count = int(frame["score"].notna().sum())
    rank_pairs = frame.loc[:, ["score", "target"]].dropna()
    spread_pairs = frame.loc[:, ["score", "realized_return"]].dropna()
    rank_count = int(rank_pairs.shape[0])
    spread_count = int(spread_pairs.shape[0])
    rank_ic = float("nan")
    spread = float("nan")
    if (
        rank_count >= min_cross_section
        and rank_pairs["score"].nunique() > 1
        and rank_pairs["target"].nunique() > 1
    ):
        rank_ic = float(
            rank_pairs["score"].corr(rank_pairs["target"], method="spearman")
        )
    if spread_count >= min_cross_section and spread_pairs["score"].nunique() > 1:
        spread = fractional_quantile_spread(
            spread_pairs["score"], spread_pairs["realized_return"], quantiles=quantiles
        )
    tied_fraction = (
        float(1.0 - rank_pairs["score"].nunique() / rank_count)
        if rank_count
        else float("nan")
    )
    score_coverage = (
        scored_count / scoring_universe_count if scoring_universe_count else 0.0
    )
    rank_coverage = rank_count / eligible_target_count if eligible_target_count else 0.0
    spread_coverage = (
        spread_count / eligible_realized_return_count
        if eligible_realized_return_count
        else 0.0
    )
    return {
        "phase": phase,
        "fold": fold,
        "as_of_date": as_of_date,
        "model": model,
        "evaluation_scope": scope,
        "rank_ic": rank_ic,
        "spread": spread,
        "scoring_universe_count": scoring_universe_count,
        "scored_count": scored_count,
        "evaluation_count": rank_count,
        "rank_ic_count": rank_count,
        "spread_count": spread_count,
        "eligible_target_count": eligible_target_count,
        "eligible_realized_return_count": eligible_realized_return_count,
        "score_coverage": float(score_coverage),
        "rank_ic_coverage": float(rank_coverage),
        "spread_coverage": float(spread_coverage),
        "tied_score_fraction": tied_fraction,
    }


def _native_daily_rows(
    predictions: pd.DataFrame,
    *,
    min_cross_section: int,
    quantiles: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    keys = ["phase", "fold", "as_of_date", "model"]
    for (phase, fold, as_of_date, model), group in predictions.groupby(
        keys, sort=True, observed=True
    ):
        eligible_target = int(group["target"].notna().sum())
        eligible_realized = int(group["realized_return"].notna().sum())
        rows.append(
            _daily_row(
                group,
                phase=str(phase),
                fold=int(fold),
                as_of_date=pd.Timestamp(as_of_date),
                model=str(model),
                scope="native",
                min_cross_section=min_cross_section,
                quantiles=quantiles,
                scoring_universe_count=int(group.shape[0]),
                eligible_target_count=eligible_target,
                eligible_realized_return_count=eligible_realized,
            )
        )
    return rows


def _common_daily_rows(
    predictions: pd.DataFrame,
    *,
    primary_models: tuple[str, ...],
    min_cross_section: int,
    quantiles: int,
) -> list[dict[str, object]]:
    if not primary_models:
        return []
    rows: list[dict[str, object]] = []
    date_keys = ["phase", "fold", "as_of_date"]
    for (phase, fold, as_of_date), group in predictions.groupby(
        date_keys, sort=True, observed=True
    ):
        primary = group.loc[group["model"].isin(primary_models)].copy(deep=True)
        if set(primary["model"].unique()) != set(primary_models):
            raise ValueError(
                "Every primary model must have rows on each evaluation date."
            )
        target_counts = primary.groupby("symbol", sort=False)["target"].nunique(
            dropna=False
        )
        realized_counts = primary.groupby("symbol", sort=False)[
            "realized_return"
        ].nunique(dropna=False)
        if (target_counts > 1).any() or (realized_counts > 1).any():
            raise ValueError("Targets must agree across models for each security-date.")

        base = (
            primary.drop_duplicates("symbol")
            .set_index("symbol")[["target", "realized_return"]]
            .sort_index()
        )
        scores = primary.pivot(index="symbol", columns="model", values="score")
        combined = base.join(scores, how="inner")
        eligible_target = int(combined["target"].notna().sum())
        eligible_realized = int(combined["realized_return"].notna().sum())
        common = combined.dropna(subset=[*primary_models])
        for model in primary_models:
            model_frame = common.loc[:, [model, "target", "realized_return"]].rename(
                columns={model: "score"}
            )
            rows.append(
                _daily_row(
                    model_frame,
                    phase=str(phase),
                    fold=int(fold),
                    as_of_date=pd.Timestamp(as_of_date),
                    model=model,
                    scope="common",
                    min_cross_section=min_cross_section,
                    quantiles=quantiles,
                    scoring_universe_count=int(combined.shape[0]),
                    eligible_target_count=eligible_target,
                    eligible_realized_return_count=eligible_realized,
                )
            )
    return rows


def _summaries(
    daily_metrics: pd.DataFrame,
    *,
    hac_lags: int,
    expected_dates_by_phase: Mapping[str, pd.DatetimeIndex] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    inference_keys = ["phase", "model", "evaluation_scope", "as_of_date"]
    if daily_metrics.duplicated(inference_keys, keep=False).any():
        raise ValueError(
            "Daily metrics must be unique by phase/model/scope/date across folds."
        )
    fold_rows: list[dict[str, object]] = []
    fold_keys = ["phase", "fold", "model", "evaluation_scope"]
    for keys, group in daily_metrics.groupby(fold_keys, sort=True, observed=True):
        valid = pd.to_numeric(group["rank_ic"], errors="coerce").dropna()
        spreads = pd.to_numeric(group["spread"], errors="coerce").dropna()
        fold_rows.append(
            {
                **dict(zip(fold_keys, keys, strict=True)),
                "date_count": int(valid.shape[0]),
                "spread_date_count": int(spreads.shape[0]),
                "mean_rank_ic": float(valid.mean())
                if not valid.empty
                else float("nan"),
                "median_rank_ic": (
                    float(valid.median()) if not valid.empty else float("nan")
                ),
                "positive_date_rate": (
                    float((valid > 0).mean()) if not valid.empty else float("nan")
                ),
                "mean_spread": (
                    float(spreads.mean()) if not spreads.empty else float("nan")
                ),
                "mean_score_coverage": float(group["score_coverage"].mean()),
                "mean_rank_ic_coverage": float(group["rank_ic_coverage"].mean()),
                "mean_spread_coverage": float(group["spread_coverage"].mean()),
            }
        )

    summary_rows: list[dict[str, object]] = []
    summary_keys = ["phase", "model", "evaluation_scope"]
    for keys, group in daily_metrics.groupby(summary_keys, sort=True, observed=True):
        valid = pd.to_numeric(group["rank_ic"], errors="coerce").dropna()
        spreads = pd.to_numeric(group["spread"], errors="coerce").dropna()
        expected_dates = (
            expected_dates_by_phase.get(str(keys[0]))
            if expected_dates_by_phase is not None
            else None
        )
        inference = (
            scheduled_newey_west_mean(
                group.set_index("as_of_date")["rank_ic"],
                expected_dates=expected_dates,
                hac_lags=hac_lags,
            )
            if expected_dates is not None
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
        rank_std = float(valid.std(ddof=1)) if valid.shape[0] > 1 else float("nan")
        summary_rows.append(
            {
                **dict(zip(summary_keys, keys, strict=True)),
                "date_count": int(valid.shape[0]),
                "spread_date_count": int(spreads.shape[0]),
                "mean_rank_ic": float(valid.mean())
                if not valid.empty
                else float("nan"),
                "median_rank_ic": (
                    float(valid.median()) if not valid.empty else float("nan")
                ),
                "rank_ic_std": rank_std,
                "rank_ic_ir": (
                    float(valid.mean() / rank_std)
                    if np.isfinite(rank_std) and rank_std > 0
                    else float("nan")
                ),
                "positive_date_rate": (
                    float((valid > 0).mean()) if not valid.empty else float("nan")
                ),
                "mean_spread": (
                    float(spreads.mean()) if not spreads.empty else float("nan")
                ),
                "mean_score_coverage": float(group["score_coverage"].mean()),
                "mean_rank_ic_coverage": float(group["rank_ic_coverage"].mean()),
                "mean_spread_coverage": float(group["spread_coverage"].mean()),
                "newey_west_t_stat": inference.t_stat,
                "p_value": inference.p_value,
                **inference_diagnostics(inference),
            }
        )
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary.insert(
            summary.columns.get_loc("p_value") + 1,
            "bh_q_value",
            benjamini_hochberg(summary["p_value"]),
        )
    return pd.DataFrame(fold_rows), summary


def _normalize_phase_schedules(
    schedules: Mapping[str, Sequence] | None,
) -> dict[str, pd.DatetimeIndex]:
    if schedules is None:
        return {}
    if not isinstance(schedules, Mapping):
        raise ValueError("expected_dates_by_phase must be a mapping.")
    if any(
        not isinstance(phase, str) or not phase or phase != phase.strip()
        for phase in schedules
    ):
        raise ValueError(
            "expected_dates_by_phase keys must be non-empty normalized phase names."
        )
    # Validate absent phases too: an invalid supplied schedule is never omission.
    return {
        phase: normalize_expected_dates(expected_dates)
        for phase, expected_dates in schedules.items()
    }


def evaluate_prediction_frame(
    predictions: pd.DataFrame,
    *,
    min_cross_section: int,
    quantiles: int,
    hac_lags: int,
    primary_models: tuple[str, ...],
    expected_dates_by_phase: Mapping[str, Sequence] | None = None,
) -> PredictionEvaluation:
    """Evaluate scores; summary inference requires an explicit phase schedule."""
    if (
        isinstance(min_cross_section, bool)
        or not isinstance(min_cross_section, int)
        or min_cross_section < 2
    ):
        raise ValueError("min_cross_section must be an integer of at least 2.")
    if isinstance(quantiles, bool) or not isinstance(quantiles, int) or quantiles < 2:
        raise ValueError("quantiles must be an integer of at least 2.")
    if quantiles > min_cross_section:
        raise ValueError("quantiles cannot exceed min_cross_section.")
    if isinstance(hac_lags, bool) or not isinstance(hac_lags, int) or hac_lags < 0:
        raise ValueError("hac_lags must be a non-negative integer.")
    schedules = _normalize_phase_schedules(expected_dates_by_phase)
    normalized = _validate_predictions(predictions)
    primary = tuple(str(model).strip() for model in primary_models)
    if len(set(primary)) != len(primary) or any(not model for model in primary):
        raise ValueError("primary_models must contain unique non-empty names.")
    daily_rows = _native_daily_rows(
        normalized,
        min_cross_section=min_cross_section,
        quantiles=quantiles,
    )
    daily_rows.extend(
        _common_daily_rows(
            normalized,
            primary_models=primary,
            min_cross_section=min_cross_section,
            quantiles=quantiles,
        )
    )
    daily = (
        pd.DataFrame(daily_rows)
        .sort_values(
            ["phase", "fold", "as_of_date", "evaluation_scope", "model"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    fold_metrics, summary = _summaries(
        daily, hac_lags=hac_lags, expected_dates_by_phase=schedules
    )
    return PredictionEvaluation(
        daily_metrics=daily,
        fold_metrics=fold_metrics,
        summary=summary,
    )


def choose_frozen_model(
    summary: pd.DataFrame,
    *,
    model_families: tuple[str, ...],
) -> str:
    required = {"phase", "model", "evaluation_scope", "mean_rank_ic"}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"Missing summary columns: {', '.join(missing)}")
    families = tuple(model_families)
    candidates = summary.loc[
        (summary["phase"] == "development")
        & (summary["evaluation_scope"] == "common")
        & summary["model"].isin(families),
        ["model", "mean_rank_ic"],
    ].copy()
    candidates["mean_rank_ic"] = pd.to_numeric(
        candidates["mean_rank_ic"], errors="coerce"
    )
    candidates = candidates.dropna(subset=["mean_rank_ic"])
    if set(candidates["model"]) != set(families):
        raise ValueError("Every model family needs a valid development common Rank IC.")
    preference = {model: position for position, model in enumerate(families)}
    candidates["preference"] = candidates["model"].map(preference)
    ordered = candidates.sort_values(
        ["mean_rank_ic", "preference"],
        ascending=[False, True],
        kind="stable",
    )
    return str(ordered.iloc[0]["model"])
