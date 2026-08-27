from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd

from .signals import compute_daily_rank_ic

PREDICTION_COLUMNS = (
    "as_of_date",
    "symbol",
    "model",
    "score",
    "baseline_feature",
    "feature_count",
)


@dataclass(frozen=True, slots=True)
class FittedNonMLBaselines:
    """Train-only state for signed cross-sectional metric baselines."""

    selected_features: tuple[str, ...]
    feature_signs: Mapping[str, int]
    mean_rank_ic: Mapping[str, float]
    best_feature: str
    target_column: str
    as_of_date_column: str
    min_cross_section: int
    fitted_through: pd.Timestamp
    training_row_count: int


def _validate_feature_columns(feature_columns: tuple[str, ...]) -> None:
    if not feature_columns:
        raise ValueError("feature_columns must contain at least one feature.")
    if any(not isinstance(feature, str) or not feature for feature in feature_columns):
        raise ValueError("feature_columns must contain non-empty strings.")
    if len(set(feature_columns)) != len(feature_columns):
        raise ValueError("feature_columns must be unique.")


def _validate_min_cross_section(min_cross_section: int) -> None:
    if (
        isinstance(min_cross_section, bool)
        or not isinstance(min_cross_section, int)
        or min_cross_section < 2
    ):
        raise ValueError("min_cross_section must be an integer of at least 2.")


def _normalize_training_panel(
    panel: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    target_column: str,
    as_of_date_column: str,
    symbol_column: str,
) -> pd.DataFrame:
    if not isinstance(panel, pd.DataFrame):
        raise ValueError("training_panel must be a pandas DataFrame.")
    required = {as_of_date_column, symbol_column, target_column, *feature_columns}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"Missing baseline columns: {', '.join(missing)}")
    if panel.empty:
        raise ValueError("training_panel must not be empty.")

    normalized = panel.copy(deep=True)
    normalized[as_of_date_column] = pd.to_datetime(
        normalized[as_of_date_column],
        errors="coerce",
    )
    if normalized[as_of_date_column].isna().any():
        raise ValueError("as_of_date contains invalid values.")
    normalized[symbol_column] = normalized[symbol_column].astype("string").str.strip()
    if (
        normalized[symbol_column].isna().any()
        or (normalized[symbol_column] == "").any()
    ):
        raise ValueError("symbol contains invalid values.")
    if normalized.duplicated(
        [as_of_date_column, symbol_column],
        keep=False,
    ).any():
        raise ValueError("Duplicate as_of_date-symbol rows found in training_panel.")
    numeric_columns = [*feature_columns, target_column]
    normalized[numeric_columns] = (
        normalized[numeric_columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    return normalized


def fit_non_ml_baselines(
    training_panel: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    target_column: str,
    min_cross_section: int,
    as_of_date_column: str = "as_of_date",
    symbol_column: str = "symbol",
) -> FittedNonMLBaselines:
    """Fit metric directions and selection using only a fold's training rows.

    The caller is responsible for passing an already purged training slice. Each
    date contributes one Spearman rank-IC observation, so dates with more symbols
    do not receive extra weight when the feature direction is learned.
    """

    _validate_feature_columns(feature_columns)
    _validate_min_cross_section(min_cross_section)
    if target_column in feature_columns:
        raise ValueError("target_column must not be a feature.")
    if {as_of_date_column, symbol_column} & set(feature_columns):
        raise ValueError("row-key columns must not be features.")
    normalized = _normalize_training_panel(
        training_panel,
        feature_columns=feature_columns,
        target_column=target_column,
        as_of_date_column=as_of_date_column,
        symbol_column=symbol_column,
    )
    daily_rank_ic = compute_daily_rank_ic(
        normalized,
        feature_columns=feature_columns,
        target_column=target_column,
        min_cross_section=min_cross_section,
        as_of_date_column=as_of_date_column,
    )
    rank_ic_by_feature = daily_rank_ic.groupby("feature", sort=False)["rank_ic"].mean()

    learned_rank_ic: dict[str, float] = {}
    unlearnable: list[str] = []
    for feature in feature_columns:
        mean_rank_ic = float(rank_ic_by_feature.get(feature, float("nan")))
        if not np.isfinite(mean_rank_ic) or np.isclose(mean_rank_ic, 0.0, atol=1e-12):
            unlearnable.append(feature)
            continue
        learned_rank_ic[feature] = mean_rank_ic
    if unlearnable:
        joined = ", ".join(unlearnable)
        raise ValueError(f"cannot learn a non-zero direction for feature(s): {joined}")

    feature_signs = {
        feature: 1 if learned_rank_ic[feature] > 0.0 else -1
        for feature in feature_columns
    }
    best_feature = min(
        feature_columns,
        key=lambda feature: (-abs(learned_rank_ic[feature]), feature),
    )
    return FittedNonMLBaselines(
        selected_features=feature_columns,
        feature_signs=MappingProxyType(feature_signs),
        mean_rank_ic=MappingProxyType(dict(learned_rank_ic)),
        best_feature=best_feature,
        target_column=target_column,
        as_of_date_column=as_of_date_column,
        min_cross_section=min_cross_section,
        fitted_through=pd.Timestamp(normalized[as_of_date_column].max()),
        training_row_count=int(normalized.shape[0]),
    )


def _normalize_prediction_panel(
    frame: pd.DataFrame,
    *,
    fitted: FittedNonMLBaselines,
    symbol_column: str,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("frame must be a pandas DataFrame.")
    required = {
        fitted.as_of_date_column,
        symbol_column,
        *fitted.selected_features,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing prediction columns: {', '.join(missing)}")

    normalized = frame.copy(deep=True)
    normalized[fitted.as_of_date_column] = pd.to_datetime(
        normalized[fitted.as_of_date_column],
        errors="coerce",
    )
    if normalized[fitted.as_of_date_column].isna().any():
        raise ValueError("as_of_date contains invalid values.")
    normalized[symbol_column] = normalized[symbol_column].astype("string").str.strip()
    if (
        normalized[symbol_column].isna().any()
        or (normalized[symbol_column] == "").any()
    ):
        raise ValueError("symbol contains invalid values.")
    if normalized.duplicated(
        [fitted.as_of_date_column, symbol_column],
        keep=False,
    ).any():
        raise ValueError("Duplicate as_of_date-symbol rows found in prediction frame.")

    feature_columns = list(fitted.selected_features)
    normalized[feature_columns] = (
        normalized[feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    return normalized


def _prediction_frame(
    row_keys: pd.DataFrame,
    *,
    model: str,
    score: pd.Series,
    baseline_feature: str | None,
    feature_count: int | pd.Series,
) -> pd.DataFrame:
    count_values = (
        feature_count.to_numpy(dtype=int)
        if isinstance(feature_count, pd.Series)
        else feature_count
    )
    return row_keys.assign(
        model=model,
        score=score.to_numpy(dtype=float),
        baseline_feature=(baseline_feature if baseline_feature is not None else pd.NA),
        feature_count=count_values,
    )


def predict_non_ml_baselines(
    fitted: FittedNonMLBaselines,
    frame: pd.DataFrame,
    *,
    symbol_column: str = "symbol",
) -> pd.DataFrame:
    """Return individual and composite signed-rank scores in long form."""

    if not isinstance(fitted, FittedNonMLBaselines):
        raise ValueError("fitted must be a FittedNonMLBaselines instance.")
    normalized = _normalize_prediction_panel(
        frame,
        fitted=fitted,
        symbol_column=symbol_column,
    )
    feature_columns = list(fitted.selected_features)
    percentile_ranks = normalized.groupby(
        fitted.as_of_date_column,
        sort=False,
    )[feature_columns].rank(method="average", pct=True)
    signed_ranks = percentile_ranks.assign(
        **{
            feature: (
                percentile_ranks[feature]
                if fitted.feature_signs[feature] > 0
                else 1.0 - percentile_ranks[feature]
            )
            for feature in fitted.selected_features
        }
    )
    row_keys = normalized.loc[
        :,
        [fitted.as_of_date_column, symbol_column],
    ].rename(
        columns={
            fitted.as_of_date_column: "as_of_date",
            symbol_column: "symbol",
        }
    )

    predictions = [
        _prediction_frame(
            row_keys,
            model=f"metric:{feature}",
            score=signed_ranks[feature],
            baseline_feature=feature,
            feature_count=signed_ranks[feature].notna().astype(int),
        )
        for feature in fitted.selected_features
    ]
    predictions.append(
        _prediction_frame(
            row_keys,
            model="best_metric",
            score=signed_ranks[fitted.best_feature],
            baseline_feature=fitted.best_feature,
            feature_count=signed_ranks[fitted.best_feature].notna().astype(int),
        )
    )
    predictions.append(
        _prediction_frame(
            row_keys,
            model="equal_weight_rank",
            score=signed_ranks.mean(axis="columns", skipna=True),
            baseline_feature=None,
            feature_count=signed_ranks.notna().sum(axis="columns"),
        )
    )
    return pd.concat(predictions, ignore_index=True).loc[:, list(PREDICTION_COLUMNS)]
