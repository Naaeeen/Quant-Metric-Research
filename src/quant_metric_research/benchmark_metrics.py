from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .statistics import benjamini_hochberg, newey_west_mean_tstat

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
    normalized["as_of_date"] = pd.to_datetime(normalized["as_of_date"], errors="coerce")
    if normalized["as_of_date"].isna().any():
        raise ValueError("Prediction dates contain invalid values.")
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
    eligible_target_count: int,
) -> dict[str, object]:
    paired = frame.loc[:, ["score", "target", "realized_return"]].dropna()
    count = int(paired.shape[0])
    rank_ic = float("nan")
    spread = float("nan")
    if (
        count >= min_cross_section
        and paired["score"].nunique() > 1
        and paired["target"].nunique() > 1
    ):
        rank_ic = float(paired["score"].corr(paired["target"], method="spearman"))
        ordered = paired.sort_values("score", kind="stable")
        bucket_size = max(1, count // quantiles)
        spread = float(
            ordered.iloc[-bucket_size:]["realized_return"].mean()
            - ordered.iloc[:bucket_size]["realized_return"].mean()
        )
    tied_fraction = (
        float(1.0 - paired["score"].nunique() / count) if count else float("nan")
    )
    coverage = count / eligible_target_count if eligible_target_count else 0.0
    return {
        "phase": phase,
        "fold": fold,
        "as_of_date": as_of_date,
        "model": model,
        "evaluation_scope": scope,
        "rank_ic": rank_ic,
        "spread": spread,
        "evaluation_count": count,
        "eligible_target_count": eligible_target_count,
        "score_coverage": float(coverage),
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
        eligible = int(group[["target", "realized_return"]].dropna().shape[0])
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
                eligible_target_count=eligible,
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
        eligible = int(combined[["target", "realized_return"]].dropna().shape[0])
        common = combined.dropna(subset=[*primary_models, "target", "realized_return"])
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
                    eligible_target_count=eligible,
                )
            )
    return rows


def _summaries(
    daily_metrics: pd.DataFrame,
    *,
    hac_lags: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, object]] = []
    fold_keys = ["phase", "fold", "model", "evaluation_scope"]
    for keys, group in daily_metrics.groupby(fold_keys, sort=True, observed=True):
        valid = pd.to_numeric(group["rank_ic"], errors="coerce").dropna()
        spreads = pd.to_numeric(group["spread"], errors="coerce").dropna()
        fold_rows.append(
            {
                **dict(zip(fold_keys, keys, strict=True)),
                "date_count": int(valid.shape[0]),
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
            }
        )

    summary_rows: list[dict[str, object]] = []
    summary_keys = ["phase", "model", "evaluation_scope"]
    for keys, group in daily_metrics.groupby(summary_keys, sort=True, observed=True):
        valid = pd.to_numeric(group["rank_ic"], errors="coerce").dropna()
        spreads = pd.to_numeric(group["spread"], errors="coerce").dropna()
        t_stat, p_value = newey_west_mean_tstat(valid, hac_lags)
        rank_std = float(valid.std(ddof=1)) if valid.shape[0] > 1 else float("nan")
        summary_rows.append(
            {
                **dict(zip(summary_keys, keys, strict=True)),
                "date_count": int(valid.shape[0]),
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
                "newey_west_t_stat": t_stat,
                "p_value": p_value,
            }
        )
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary["bh_q_value"] = benjamini_hochberg(summary["p_value"])
    return pd.DataFrame(fold_rows), summary


def evaluate_prediction_frame(
    predictions: pd.DataFrame,
    *,
    min_cross_section: int,
    quantiles: int,
    hac_lags: int,
    primary_models: tuple[str, ...],
) -> PredictionEvaluation:
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
    fold_metrics, summary = _summaries(daily, hac_lags=hac_lags)
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
