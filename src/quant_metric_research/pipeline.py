from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import PanelConfig
from .contracts import DataContractError, validate_as_of_dates
from .panel import build_point_in_time_panel
from .reduction import PCABaseline, fit_pca_baseline
from .screening import MetricScreenResult, fit_metric_screen
from .validation import (
    WalkForwardMetricConfig,
    WalkForwardMetricResult,
    evaluate_metrics_walk_forward,
)


@dataclass(frozen=True)
class ResearchRun:
    panel: pd.DataFrame
    screen: MetricScreenResult
    pca: PCABaseline | None
    pca_scores: pd.DataFrame | None
    walk_forward: WalkForwardMetricResult | None


def run_research(
    prices: pd.DataFrame,
    memberships: pd.DataFrame,
    *,
    as_of_dates: tuple[pd.Timestamp, ...] | list[pd.Timestamp],
    config: PanelConfig,
    train_end_date: str | pd.Timestamp,
    min_cross_section: int,
    minimum_coverage: float,
    redundancy_threshold: float,
    hac_lags: int,
    quantiles: int = 5,
    run_pca: bool = False,
    pca_variance_to_keep: float = 0.95,
    walk_forward_config: WalkForwardMetricConfig | None = None,
) -> ResearchRun:
    try:
        cutoff = validate_as_of_dates((train_end_date,))[0]
    except DataContractError as error:
        raise DataContractError(
            "train_end_date must be a normalized, timezone-naive daily date."
        ) from error

    panel = build_point_in_time_panel(
        prices,
        memberships,
        as_of_dates=as_of_dates,
        config=config,
    )
    # Keep contemporaneous features for quality, redundancy, and PCA while
    # screening only outcomes known by this end-of-day training cutoff.
    screening_panel = panel.copy(deep=True)
    mature = panel["label_end_date"].notna() & panel["label_end_date"].le(cutoff)
    screening_panel.loc[~mature, "forward_excess_return"] = float("nan")
    screen = fit_metric_screen(
        screening_panel,
        feature_columns=config.feature_columns,
        target_column="forward_excess_return",
        train_end_date=cutoff,
        min_cross_section=min_cross_section,
        minimum_coverage=minimum_coverage,
        redundancy_threshold=redundancy_threshold,
        hac_lags=hac_lags,
        quantiles=quantiles,
    )

    pca: PCABaseline | None = None
    pca_scores: pd.DataFrame | None = None
    if run_pca and screen.selected_features:
        pca = fit_pca_baseline(
            panel,
            feature_columns=screen.selected_features,
            train_end_date=cutoff,
            variance_to_keep=pca_variance_to_keep,
        )
        pca_scores = pca.transform(panel)

    walk_forward: WalkForwardMetricResult | None = None
    if walk_forward_config is not None:
        walk_forward = evaluate_metrics_walk_forward(
            panel,
            feature_columns=config.feature_columns,
            target_column="forward_excess_return",
            config=walk_forward_config,
            min_cross_section=min_cross_section,
            minimum_coverage=minimum_coverage,
            redundancy_threshold=redundancy_threshold,
            hac_lags=hac_lags,
            quantiles=quantiles,
        )

    return ResearchRun(
        panel=panel,
        screen=screen,
        pca=pca,
        pca_scores=pca_scores,
        walk_forward=walk_forward,
    )
