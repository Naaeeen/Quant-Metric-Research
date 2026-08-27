from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .screening import fit_metric_screen
from .signals import compute_daily_rank_ic, compute_quantile_spreads
from .splits import build_purged_walk_forward_splits


@dataclass(frozen=True)
class WalkForwardMetricConfig:
    n_splits: int
    test_date_count: int
    min_train_date_count: int

    def __post_init__(self) -> None:
        for field_name in (
            "n_splits",
            "test_date_count",
            "min_train_date_count",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer.")


@dataclass(frozen=True)
class WalkForwardMetricResult:
    fold_metrics: pd.DataFrame
    summary: pd.DataFrame


def _feature_fold_values(
    frame: pd.DataFrame,
    *,
    feature: str,
    value_column: str,
) -> tuple[float, int]:
    values = pd.to_numeric(
        frame.loc[frame["feature"] == feature, value_column],
        errors="coerce",
    ).dropna()
    if values.empty:
        return float("nan"), 0
    return float(values.mean()), int(values.shape[0])


def evaluate_metrics_walk_forward(
    panel: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    target_column: str,
    config: WalkForwardMetricConfig,
    min_cross_section: int,
    minimum_coverage: float,
    redundancy_threshold: float,
    hac_lags: int,
    quantiles: int = 5,
    as_of_date_column: str = "as_of_date",
    label_end_date_column: str = "label_end_date",
) -> WalkForwardMetricResult:
    folds = build_purged_walk_forward_splits(
        panel,
        n_splits=config.n_splits,
        test_date_count=config.test_date_count,
        min_train_date_count=config.min_train_date_count,
        as_of_date_column=as_of_date_column,
        label_end_date_column=label_end_date_column,
    )
    rows: list[dict[str, object]] = []
    for fold_number, fold in enumerate(folds, start=1):
        training = panel.loc[list(fold.train_indices)].copy(deep=True)
        testing = panel.loc[list(fold.test_indices)].copy(deep=True)
        screen = fit_metric_screen(
            training,
            feature_columns=feature_columns,
            target_column=target_column,
            train_end_date=fold.train_end_date,
            min_cross_section=min_cross_section,
            minimum_coverage=minimum_coverage,
            redundancy_threshold=redundancy_threshold,
            hac_lags=hac_lags,
            quantiles=quantiles,
            as_of_date_column=as_of_date_column,
        )
        test_rank_ic = compute_daily_rank_ic(
            testing,
            feature_columns=feature_columns,
            target_column=target_column,
            min_cross_section=min_cross_section,
            as_of_date_column=as_of_date_column,
        )
        test_spreads = compute_quantile_spreads(
            testing,
            feature_columns=feature_columns,
            target_column=target_column,
            quantiles=quantiles,
            min_cross_section=min_cross_section,
            as_of_date_column=as_of_date_column,
        )
        train_label_end_max = pd.Timestamp(
            pd.to_datetime(training[label_end_date_column]).max()
        )
        train_ic_by_feature = screen.ic_summary.set_index("feature")
        for feature in feature_columns:
            mean_rank_ic, rank_ic_date_count = _feature_fold_values(
                test_rank_ic,
                feature=feature,
                value_column="rank_ic",
            )
            mean_spread, spread_date_count = _feature_fold_values(
                test_spreads,
                feature=feature,
                value_column="spread",
            )
            train_mean_rank_ic = (
                float(train_ic_by_feature.at[feature, "mean_rank_ic"])
                if feature in train_ic_by_feature.index
                else float("nan")
            )
            directional_test_rank_ic = (
                float(np.sign(train_mean_rank_ic) * mean_rank_ic)
                if np.isfinite(train_mean_rank_ic)
                and train_mean_rank_ic != 0.0
                and np.isfinite(mean_rank_ic)
                else float("nan")
            )
            rows.append(
                {
                    "fold": fold_number,
                    "feature": feature,
                    "selected_on_train": (feature in screen.selected_features),
                    "train_end_date": fold.train_end_date,
                    "train_label_end_max": train_label_end_max,
                    "test_start_date": fold.test_start_date,
                    "test_end_date": fold.test_end_date,
                    "train_mean_rank_ic": train_mean_rank_ic,
                    "test_mean_rank_ic": mean_rank_ic,
                    "directional_test_rank_ic": (directional_test_rank_ic),
                    "test_rank_ic_date_count": rank_ic_date_count,
                    "test_mean_spread": mean_spread,
                    "test_spread_date_count": spread_date_count,
                }
            )

    fold_metrics = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    for feature in feature_columns:
        feature_folds = fold_metrics.loc[fold_metrics["feature"] == feature]
        valid_ic = pd.to_numeric(
            feature_folds["test_mean_rank_ic"],
            errors="coerce",
        ).dropna()
        valid_spread = pd.to_numeric(
            feature_folds["test_mean_spread"],
            errors="coerce",
        ).dropna()
        valid_directional_ic = pd.to_numeric(
            feature_folds["directional_test_rank_ic"],
            errors="coerce",
        ).dropna()
        summary_rows.append(
            {
                "feature": feature,
                "fold_count": int(feature_folds.shape[0]),
                "selection_count": int(feature_folds["selected_on_train"].sum()),
                "selection_rate": float(feature_folds["selected_on_train"].mean()),
                "test_fold_count": int(valid_ic.shape[0]),
                "mean_test_rank_ic": (
                    float(valid_ic.mean()) if not valid_ic.empty else float("nan")
                ),
                "median_test_rank_ic": (
                    float(valid_ic.median()) if not valid_ic.empty else float("nan")
                ),
                "positive_test_fold_rate": (
                    float((valid_ic > 0).mean()) if not valid_ic.empty else float("nan")
                ),
                "mean_directional_test_rank_ic": (
                    float(valid_directional_ic.mean())
                    if not valid_directional_ic.empty
                    else float("nan")
                ),
                "direction_consistency_rate": (
                    float((valid_directional_ic > 0).mean())
                    if not valid_directional_ic.empty
                    else float("nan")
                ),
                "mean_test_spread": (
                    float(valid_spread.mean())
                    if not valid_spread.empty
                    else float("nan")
                ),
            }
        )
    summary = pd.DataFrame(summary_rows).replace(
        [np.inf, -np.inf],
        np.nan,
    )
    return WalkForwardMetricResult(
        fold_metrics=fold_metrics,
        summary=summary,
    )
