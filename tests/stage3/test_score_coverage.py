from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_metric_research.benchmark_metrics import (
    PredictionEvaluation,
    evaluate_prediction_frame,
)


def _predictions(
    *,
    model: str,
    symbols: list[str],
    scores: list[float],
    targets: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "phase": "development",
            "fold": 1,
            "as_of_date": pd.Timestamp("2025-01-03"),
            "symbol": symbols,
            "model": model,
            "score": scores,
            "target": targets,
            "realized_return": targets,
        }
    )


def _evaluate(
    frame: pd.DataFrame, primary_models: tuple[str, ...] = ("ridge",)
) -> PredictionEvaluation:
    return evaluate_prediction_frame(
        frame,
        min_cross_section=2,
        quantiles=2,
        hac_lags=0,
        primary_models=primary_models,
    )


@pytest.mark.parametrize("last_score, expected", [(np.nan, 0.75), (3.0, 1.0)])
def test_masking_outcomes_does_not_change_prediction_coverage(
    last_score: float, expected: float
) -> None:
    frame = _predictions(
        model="ridge",
        symbols=["S0", "S1", "S2", "S3"],
        scores=[0.0, 1.0, 2.0, last_score],
        targets=[0.0, 1.0, 2.0, 3.0],
    )
    hidden = frame.assign(
        target=[0.0, 1.0, 2.0, np.nan],
        realized_return=[0.0, 1.0, 2.0, np.nan],
    )

    before = _evaluate(frame)
    after = _evaluate(hidden)

    assert before.daily_metrics["score_coverage"].tolist() == pytest.approx(
        [expected, expected]
    )
    assert after.daily_metrics["score_coverage"].tolist() == pytest.approx(
        [expected, expected]
    )
    assert after.daily_metrics["scoring_universe_count"].tolist() == [4, 4]
    assert after.daily_metrics["scored_count"].tolist() == [int(4 * expected)] * 2
    assert before.daily_metrics["rank_ic_coverage"].tolist() == pytest.approx(
        [expected, expected]
    )
    assert after.daily_metrics["rank_ic_coverage"].tolist() == [1.0, 1.0]
    assert after.daily_metrics["spread_coverage"].tolist() == [1.0, 1.0]
    assert after.summary["mean_score_coverage"].tolist() == pytest.approx(
        [expected, expected]
    )
    assert after.fold_metrics["mean_score_coverage"].tolist() == pytest.approx(
        [expected, expected]
    )


def test_scores_remain_visible_when_every_outcome_is_missing() -> None:
    frame = _predictions(
        model="ridge",
        symbols=["S0", "S1", "S2", "S3"],
        scores=[0.0, 1.0, 2.0, np.nan],
        targets=[np.nan] * 4,
    )

    daily = _evaluate(frame).daily_metrics

    assert daily["score_coverage"].tolist() == pytest.approx([0.75, 0.75])
    assert daily["scoring_universe_count"].tolist() == [4, 4]
    assert daily["scored_count"].tolist() == [3, 3]
    assert daily["rank_ic_coverage"].tolist() == [0.0, 0.0]
    assert daily["spread_coverage"].tolist() == [0.0, 0.0]


def test_common_score_coverage_uses_union_before_score_intersection() -> None:
    ridge = _predictions(
        model="ridge",
        symbols=["S0", "S1", "S2", "S3"],
        scores=[0.0, 1.0, 2.0, 3.0],
        targets=[0.0, 1.0, 2.0, np.nan],
    )
    baseline = _predictions(
        model="baseline",
        symbols=["S1", "S2", "S3", "S4"],
        scores=[1.0, 2.0, 3.0, 4.0],
        targets=[1.0, 2.0, np.nan, np.nan],
    )

    daily = _evaluate(
        pd.concat([ridge, baseline], ignore_index=True), ("ridge", "baseline")
    ).daily_metrics
    common = daily.loc[daily["evaluation_scope"] == "common"]
    native = daily.loc[daily["evaluation_scope"] == "native"]

    assert common["score_coverage"].tolist() == pytest.approx([0.6, 0.6])
    assert common["scoring_universe_count"].tolist() == [5, 5]
    assert common["scored_count"].tolist() == [3, 3]
    assert common["rank_ic_coverage"].tolist() == pytest.approx([2 / 3, 2 / 3])
    assert native["scoring_universe_count"].tolist() == [4, 4]
    assert native["score_coverage"].tolist() == [1.0, 1.0]


def test_no_scores_keeps_the_full_scoring_universe_visible() -> None:
    frame = _predictions(
        model="ridge",
        symbols=["S0", "S1", "S2", "S3"],
        scores=[np.nan] * 4,
        targets=[np.nan] * 4,
    )

    daily = _evaluate(frame).daily_metrics

    assert daily["scoring_universe_count"].tolist() == [4, 4]
    assert daily["scored_count"].tolist() == [0, 0]
    assert daily["score_coverage"].tolist() == [0.0, 0.0]
