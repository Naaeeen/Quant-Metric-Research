from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_metric_research.config import PanelConfig
from quant_metric_research.panel import build_point_in_time_panel
from quant_metric_research.pipeline import run_research
from quant_metric_research.reduction import fit_pca_baseline


def _config() -> PanelConfig:
    return PanelConfig(
        dataset_version="cutoff-regression",
        universe_id="TEST",
        benchmark_symbol="BENCH",
        lookback_sessions=4,
        min_observations=3,
        target_horizon_sessions=2,
        entry_lag_sessions=1,
        feature_columns=("trailing_return",),
    )


def _run(prices, memberships, dates, cutoff, *, run_pca=False):
    return run_research(
        prices,
        memberships,
        as_of_dates=tuple(dates),
        config=_config(),
        train_end_date=cutoff,
        min_cross_section=2,
        minimum_coverage=0.0,
        redundancy_threshold=0.9,
        hac_lags=0,
        quantiles=2,
        run_pca=run_pca,
    )


def test_future_prices_cannot_change_single_cutoff_screen_or_pca(
    market_fixture,
) -> None:
    prices, memberships, dates = market_fixture
    memberships = memberships.assign(effective_from=dates[0])
    decisions = dates[4:11]
    cutoff = dates[10]
    altered_prices = prices.copy(deep=True)
    future = altered_prices["date"] > cutoff
    future_aaa = future & altered_prices["symbol"].eq("AAA")
    future_bbb = future & altered_prices["symbol"].eq("BBB")
    altered_prices.loc[future_aaa, "adjusted_close"] = 5000.0 / np.arange(
        1, int(future_aaa.sum()) + 1
    )
    altered_prices.loc[future_bbb, "adjusted_close"] = 1.0 * np.arange(
        1, int(future_bbb.sum()) + 1
    )

    original = _run(prices, memberships, decisions, cutoff, run_pca=True)
    changed = _run(altered_prices, memberships, decisions, cutoff, run_pca=True)

    assert not original.panel["forward_excess_return"].equals(
        changed.panel["forward_excess_return"]
    )
    for attribute in (
        "daily_rank_ic",
        "ic_summary",
        "quantile_spreads",
        "quality",
        "redundancy",
    ):
        pd.testing.assert_frame_equal(
            getattr(original.screen, attribute), getattr(changed.screen, attribute)
        )
    assert original.screen.selected_features == changed.screen.selected_features
    assert original.screen.dropped_features == changed.screen.dropped_features
    assert original.pca is not None and changed.pca is not None
    np.testing.assert_array_equal(original.pca.components, changed.pca.components)
    pd.testing.assert_frame_equal(original.pca_scores, changed.pca_scores)


def test_maturity_at_cutoff_is_inclusive_and_returned_panel_is_unmasked(
    market_fixture,
) -> None:
    prices, memberships, dates = market_fixture
    memberships = memberships.assign(effective_from=dates[0])
    decisions = dates[4:11]
    cutoff = dates[10]
    expected_panel = build_point_in_time_panel(
        prices, memberships, as_of_dates=tuple(decisions), config=_config()
    )

    result = _run(prices, memberships, decisions, cutoff)

    mature_dates = expected_panel.loc[
        expected_panel["label_end_date"] <= cutoff, "as_of_date"
    ].unique()
    assert set(result.screen.daily_rank_ic["as_of_date"]) == set(mature_dates)
    assert dates[7] in set(result.screen.daily_rank_ic["as_of_date"])
    pd.testing.assert_frame_equal(result.panel, expected_panel)
    assert (
        result.panel.loc[
            result.panel["label_end_date"] > cutoff, "forward_excess_return"
        ]
        .notna()
        .all()
    )
    assert result.screen.training_row_count == len(expected_panel)
    assert result.screen.quality.iloc[0]["row_count"] == len(expected_panel)


def test_unknown_and_future_label_ends_mask_targets_not_feature_quality(
    monkeypatch,
) -> None:
    dates = pd.bdate_range("2025-01-02", periods=4)
    panel = pd.DataFrame(
        {
            "as_of_date": np.repeat(dates[:3], 3),
            "symbol": ["A", "B", "C"] * 3,
            "label_end_date": np.repeat([dates[2], dates[3], pd.NaT], 3),
            "trailing_return": [1.0, 2.0, 3.0] * 2 + [11.0, np.nan, 13.0],
            "forward_excess_return": [1.0, 2.0, 3.0] * 3,
        }
    )
    expected_panel = panel.copy(deep=True)
    monkeypatch.setattr(
        "quant_metric_research.pipeline.build_point_in_time_panel",
        lambda *args, **kwargs: panel,
    )

    result = _run(pd.DataFrame(), pd.DataFrame(), dates[:3], dates[2], run_pca=True)

    assert result.screen.daily_rank_ic["as_of_date"].tolist() == [dates[0]]
    assert result.screen.quantile_spreads["as_of_date"].tolist() == [dates[0]]
    assert result.screen.training_row_count == 9
    assert result.screen.quality.iloc[0]["row_count"] == 9
    assert result.screen.quality.iloc[0]["non_missing_count"] == 8
    assert result.screen.quality.iloc[0]["overall_coverage"] == pytest.approx(8 / 9)
    pd.testing.assert_frame_equal(result.panel, expected_panel)
    pd.testing.assert_frame_equal(panel, expected_panel)
    expected_pca = fit_pca_baseline(
        expected_panel,
        feature_columns=_config().feature_columns,
        train_end_date=dates[2],
        variance_to_keep=0.95,
    )
    pd.testing.assert_frame_equal(
        result.pca_scores, expected_pca.transform(expected_panel)
    )


@pytest.mark.parametrize(
    "cutoff",
    [
        None,
        pd.NaT,
        "not-a-date",
        20250116,
        True,
        "2025-01-16 12:00:00",
        pd.Timestamp("2025-01-16", tz="UTC"),
        ["2025-01-16"],
    ],
)
def test_train_end_rejects_non_daily_scalar_before_building_panel(
    cutoff, monkeypatch
) -> None:
    def unexpected_build(*args, **kwargs):
        pytest.fail("Invalid train_end_date must fail before panel construction.")

    monkeypatch.setattr(
        "quant_metric_research.pipeline.build_point_in_time_panel",
        unexpected_build,
    )

    with pytest.raises(ValueError, match="train_end_date"):
        _run(pd.DataFrame(), pd.DataFrame(), [], cutoff)
