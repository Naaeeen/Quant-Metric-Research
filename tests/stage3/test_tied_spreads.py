from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd
import pytest

from quant_metric_research.benchmark_metrics import evaluate_prediction_frame
from quant_metric_research.signals import compute_quantile_spreads


def _predictions(scores: list[float], returns: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "phase": "development",
            "fold": 1,
            "as_of_date": pd.Timestamp("2025-01-03"),
            "symbol": [f"S{index}" for index in range(len(scores))],
            "model": "ridge",
            "score": scores,
            "target": returns,
            "realized_return": returns,
        }
    )


def _daily(
    predictions: pd.DataFrame, *, quantiles: int = 4, min_cross_section: int = 4
) -> pd.DataFrame:
    return evaluate_prediction_frame(
        predictions,
        min_cross_section=min_cross_section,
        quantiles=quantiles,
        hac_lags=0,
        primary_models=("ridge",),
    ).daily_metrics


@pytest.mark.parametrize("tied_returns", list(permutations([0.0, 10.0, 20.0])))
@pytest.mark.parametrize("quantiles, expected", [(2, -4.5), (4, -9.0)])
def test_spread_is_independent_of_security_names_with_tied_scores(
    tied_returns: tuple[float, ...], quantiles: int, expected: float
) -> None:
    frame = _predictions([0.0, 0.0, 0.0, 1.0], [*tied_returns, 1.0])
    original = frame.copy(deep=True)

    daily = _daily(frame, quantiles=quantiles)

    assert daily["spread"].tolist() == pytest.approx([expected, expected])
    assert set(daily["evaluation_scope"]) == {"native", "common"}
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize(
    "scores, returns, quantiles, expected",
    [
        (list(range(7)), [0, 1, 4, 9, 16, 25, 36], 3, 30.0),
        ([0, 0, 0, 1, 2, 2, 2], [0, 3, 6, 100, 12, 15, 18], 3, 12.0),
        ([0, 0, 1, 2, 3, 4, 5, 5], [10, 20, 0, 0, 0, 0, 70, 90], 4, 65.0),
    ],
)
def test_stages_agree_on_fixed_bucket_mass_with_unique_and_tied_scores(
    scores: list[float], returns: list[float], quantiles: int, expected: float
) -> None:
    frame = _predictions(scores, returns)

    stage3 = _daily(frame, quantiles=quantiles)
    stage2 = compute_quantile_spreads(
        frame,
        feature_columns=("score",),
        target_column="realized_return",
        quantiles=quantiles,
        min_cross_section=4,
    )

    assert stage3["spread"].tolist() == pytest.approx([expected, expected])
    assert stage2["spread"].tolist() == pytest.approx([expected])


def test_constant_predictions_have_no_spread() -> None:
    daily = _daily(_predictions([1.0] * 4, [0.0, 10.0, 20.0, 1.0]))

    assert daily["spread"].isna().all()


def test_spread_uses_realized_return_pairs_and_preserves_coverage() -> None:
    frame = _predictions([0.0, 0.0, 0.0, 1.0, 2.0], [0.0, 10.0, 20.0, 1.0, np.nan])
    frame = frame.assign(target=[np.nan, 10.0, 20.0, 1.0, np.nan])

    daily = _daily(frame)

    assert daily["spread"].tolist() == pytest.approx([-9.0, -9.0])
    assert daily["spread_count"].tolist() == [4, 4]
    assert daily["spread_coverage"].tolist() == pytest.approx([1.0, 1.0])
    assert daily["rank_ic"].isna().all()


def test_tied_scores_do_not_bypass_minimum_cross_section() -> None:
    daily = _daily(_predictions([0.0, 0.0, 1.0], [0.0, 10.0, 1.0]))

    assert daily["spread"].isna().all()
    assert daily["spread_count"].tolist() == [3, 3]
