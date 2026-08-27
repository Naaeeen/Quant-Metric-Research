from __future__ import annotations

import pandas as pd

from quant_metric_research.config import PanelConfig
from quant_metric_research.pipeline import run_research
from quant_metric_research.validation import WalkForwardMetricConfig


def _config() -> PanelConfig:
    return PanelConfig(
        dataset_version="integration-v1",
        universe_id="TEST",
        benchmark_symbol="BENCH",
        lookback_sessions=5,
        min_observations=4,
        target_horizon_sessions=2,
        entry_lag_sessions=1,
        annualization_sessions=5,
    )


def test_run_research_connects_point_in_time_panel_to_train_only_screening(
    market_fixture,
) -> None:
    prices, memberships, dates = market_fixture

    result = run_research(
        prices,
        memberships,
        as_of_dates=tuple(dates[4:11]),
        config=_config(),
        train_end_date=dates[10],
        min_cross_section=2,
        minimum_coverage=0.8,
        redundancy_threshold=0.9,
        hac_lags=1,
        run_pca=True,
        pca_variance_to_keep=0.95,
        walk_forward_config=WalkForwardMetricConfig(
            n_splits=1,
            test_date_count=2,
            min_train_date_count=2,
        ),
    )

    assert not result.panel.empty
    assert result.screen.fitted_through == pd.Timestamp(dates[10])
    assert result.screen.selected_features
    assert result.pca is not None
    assert result.pca_scores is not None
    assert result.walk_forward is not None
    assert not result.walk_forward.summary.empty
    assert result.pca_scores[["as_of_date", "symbol"]].equals(
        result.panel[["as_of_date", "symbol"]].reset_index(drop=True)
    )


def test_run_research_can_leave_dimensionality_reduction_disabled(
    market_fixture,
) -> None:
    prices, memberships, dates = market_fixture

    result = run_research(
        prices,
        memberships,
        as_of_dates=tuple(dates[8:11]),
        config=_config(),
        train_end_date=dates[10],
        min_cross_section=2,
        minimum_coverage=0.8,
        redundancy_threshold=0.9,
        hac_lags=1,
        run_pca=False,
    )

    assert result.pca is None
    assert result.pca_scores is None
    assert result.walk_forward is None
