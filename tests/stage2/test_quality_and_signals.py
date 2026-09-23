from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd
import pytest

from quant_metric_research.quality import compute_feature_quality
from quant_metric_research.signals import (
    compute_daily_rank_ic,
    compute_quantile_spreads,
    summarize_rank_ic,
)
from quant_metric_research.statistics import benjamini_hochberg


def _signal_panel() -> pd.DataFrame:
    dates = pd.to_datetime(["2025-01-03", "2025-01-10", "2025-01-17"])
    rows = []
    targets = (
        [0.01, 0.02, 0.03, 0.04],
        [0.04, 0.03, 0.02, 0.01],
        [0.01, 0.02, 0.03, 0.04],
    )
    for date, date_targets in zip(dates, targets, strict=True):
        for index, target in enumerate(date_targets):
            rows.append(
                {
                    "as_of_date": date,
                    "symbol": f"S{index}",
                    "good": float(index),
                    "duplicate": float(index) * 10.0,
                    "partial": float(index) if index < 3 else np.nan,
                    "constant": 1.0,
                    "target": target,
                }
            )
    return pd.DataFrame(rows)


def test_quality_report_measures_overall_and_per_date_coverage() -> None:
    report = compute_feature_quality(
        _signal_panel(),
        feature_columns=("good", "partial", "constant"),
    ).set_index("feature")

    assert report.loc["good", "overall_coverage"] == pytest.approx(1.0)
    assert report.loc["partial", "overall_coverage"] == pytest.approx(0.75)
    assert report.loc["partial", "minimum_date_coverage"] == pytest.approx(0.75)
    assert bool(report.loc["constant", "zero_variance"]) is True


def test_rank_ic_is_computed_per_date_not_on_pooled_rows() -> None:
    daily = compute_daily_rank_ic(
        _signal_panel(),
        feature_columns=("good",),
        target_column="target",
        min_cross_section=4,
    )

    assert daily["rank_ic"].tolist() == pytest.approx([1.0, -1.0, 1.0])
    summary = summarize_rank_ic(daily, hac_lags=1).set_index("feature")
    assert summary.loc["good", "mean_rank_ic"] == pytest.approx(1.0 / 3.0)
    assert summary.loc["good", "positive_rate"] == pytest.approx(2.0 / 3.0)


def test_quantile_spread_is_top_minus_bottom_within_each_date() -> None:
    spreads = compute_quantile_spreads(
        _signal_panel(),
        feature_columns=("good",),
        target_column="target",
        quantiles=2,
        min_cross_section=4,
    )

    assert spreads["spread"].tolist() == pytest.approx([0.02, -0.02, 0.02])


@pytest.mark.parametrize("tied_returns", list(permutations([0.0, 10.0, 20.0])))
@pytest.mark.parametrize("quantiles, expected", [(2, -4.5), (4, -9.0)])
def test_quantile_spread_shares_boundary_ties_independent_of_row_order(
    tied_returns: tuple[float, ...], quantiles: int, expected: float
) -> None:
    panel = pd.DataFrame(
        {
            "as_of_date": pd.Timestamp("2025-01-03"),
            "signal": [0.0, 0.0, 0.0, 1.0],
            "target": [*tied_returns, 1.0],
        }
    )

    spreads = compute_quantile_spreads(
        panel,
        feature_columns=("signal",),
        target_column="target",
        quantiles=quantiles,
        min_cross_section=4,
    )

    assert spreads["spread"].tolist() == pytest.approx([expected])
    assert spreads["cross_section_size"].tolist() == [4]


def test_quantile_spread_does_not_invent_a_spread_for_constant_scores() -> None:
    spreads = compute_quantile_spreads(
        _signal_panel(),
        feature_columns=("constant",),
        target_column="target",
        quantiles=2,
        min_cross_section=4,
    )

    assert spreads.empty


def test_benjamini_hochberg_returns_monotone_bounded_q_values() -> None:
    q_values = benjamini_hochberg([0.01, 0.04, 0.03, 0.8])

    assert all(0.0 <= value <= 1.0 for value in q_values)
    ordered = [q_values[index] for index in np.argsort([0.01, 0.04, 0.03, 0.8])]
    assert ordered == sorted(ordered)
    assert q_values[0] == pytest.approx(0.04)


def test_rank_ic_skips_dates_below_minimum_cross_section() -> None:
    panel = _signal_panel()
    panel.loc[
        (panel["as_of_date"] == panel["as_of_date"].min()) & (panel["symbol"] != "S0"),
        "good",
    ] = np.nan

    daily = compute_daily_rank_ic(
        panel,
        feature_columns=("good",),
        target_column="target",
        min_cross_section=4,
    )

    assert daily["as_of_date"].nunique() == 2
