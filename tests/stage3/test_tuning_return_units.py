from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_metric_research import benchmark_tuning
from quant_metric_research.benchmark import run_stage3_benchmark
from quant_metric_research.benchmark_config import BenchmarkConfig, NestedSplitConfig
from quant_metric_research.benchmark_models import candidate_specs


def _config() -> BenchmarkConfig:
    return BenchmarkConfig(
        feature_columns=("signal",),
        target_column="rank_target",
        realized_return_column="return",
        split=NestedSplitConfig(3, 2, 2, 10, 2, 2, 4),
        min_cross_section=3,
        quantiles=2,
        hac_lags=1,
        ridge_alphas=(0.1, 1.0),
        model_families=("ridge",),
    )


def _validation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "as_of_date": pd.to_datetime(["2025-01-02"] * 4),
            "rank_target": [0.25, 0.5, 0.75, 1.0],
            "return": [0.0, 0.02, 0.03, 0.2],
            "signal": [1.0, 2.0, 3.0, 4.0],
        }
    )


def _validation_scores(monkeypatch, validation, config):
    original = validation.copy(deep=True)
    predictions = []

    def predict(frame):
        predictions.append(frame["signal"].copy(deep=True))
        return frame["signal"]

    monkeypatch.setattr(
        benchmark_tuning,
        "fit_candidate",
        lambda *args, **kwargs: SimpleNamespace(predict=predict),
    )
    screen = SimpleNamespace(selected_features=("signal",))
    candidate = candidate_specs(config, family="ridge")[0]
    result = benchmark_tuning._candidate_validation_scores(
        candidate, ((validation, validation, screen),), config=config
    )

    pd.testing.assert_frame_equal(validation, original)
    pd.testing.assert_series_equal(predictions[0], original["signal"])
    return result


@pytest.mark.parametrize(
    ("missing_targets", "missing_returns", "expected_ic", "expected_spread"),
    [
        ((), (), 1.0, 0.105),
        ((0,), (), 1.0, 0.105),
        ((), (0,), 1.0, 0.18),
        ((0,), (3,), 1.0, 0.03),
        ((0, 1, 2, 3), (), np.nan, 0.105),
        ((), (0, 1, 2, 3), 1.0, np.nan),
    ],
)
def test_validation_uses_independent_target_and_return_pairs(
    monkeypatch, missing_targets, missing_returns, expected_ic, expected_spread
):
    frame = _validation_frame()
    frame.loc[list(missing_targets), "rank_target"] = np.nan
    frame.loc[list(missing_returns), "return"] = np.nan

    rank_ic, spread, date_count = _validation_scores(monkeypatch, frame, _config())

    assert rank_ic == pytest.approx(expected_ic, nan_ok=True)
    assert spread == pytest.approx(expected_spread, nan_ok=True)
    assert date_count == (0 if np.isnan(expected_ic) else 1)


def test_validation_supports_one_column_for_target_and_return(monkeypatch):
    config = replace(_config(), target_column="return")

    rank_ic, spread, date_count = _validation_scores(
        monkeypatch, _validation_frame().drop(columns="rank_target"), config
    )

    assert rank_ic == pytest.approx(1.0)
    assert spread == pytest.approx(0.105)
    assert date_count == 1


@pytest.mark.parametrize("renamed_column", ["rank_target", "return"])
def test_validation_preserves_outcomes_named_model_score(monkeypatch, renamed_column):
    frame = _validation_frame().assign(rank_target=[1.0, 0.25, 0.75, 0.5])
    frame = frame.rename(columns={renamed_column: "model_score"})
    config = replace(
        _config(),
        target_column="model_score"
        if renamed_column == "rank_target"
        else "rank_target",
        realized_return_column="model_score"
        if renamed_column == "return"
        else "return",
    )

    rank_ic, spread, date_count = _validation_scores(monkeypatch, frame, config)

    assert rank_ic == pytest.approx(-0.4)
    assert spread == pytest.approx(0.105)
    assert date_count == 1


def _rank_panel() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "as_of_date": date,
                "symbol": f"S{stock}",
                "label_end_date": date + pd.offsets.BDay(2),
                "signal": float(stock),
                "rank_target": (stock + 1) / 6,
                "return": 0.01 * stock,
            }
            for date in pd.bdate_range("2025-01-02", periods=30)
            for stock in range(6)
        ]
    )


def test_rank_target_development_reports_return_units_and_keeps_model_choice():
    panel = _rank_panel()
    original = panel.copy(deep=True)
    scaled_panel = panel.assign(return_scaled=panel["return"] * 10)
    config = _config()

    result = run_stage3_benchmark(panel, config=config)
    scaled = run_stage3_benchmark(
        scaled_panel, config=replace(config, realized_return_column="return_scaled")
    )

    pd.testing.assert_frame_equal(panel, original)
    assert result.tuning_trials["validation_mean_spread"].tolist() == (
        pytest.approx([0.03] * len(result.tuning_trials))
    )
    assert result.daily_metrics["spread"].tolist() == pytest.approx(
        [0.03] * len(result.daily_metrics)
    )
    pd.testing.assert_series_equal(
        result.predictions["score"], scaled.predictions["score"]
    )
    trial_columns = ["candidate_id", "validation_mean_rank_ic", "selected"]
    pd.testing.assert_frame_equal(
        result.tuning_trials[trial_columns],
        scaled.tuning_trials[trial_columns],
    )
    assert scaled.tuning_trials["validation_mean_spread"].tolist() == (
        pytest.approx([0.3] * len(scaled.tuning_trials))
    )
    assert scaled.daily_metrics["spread"].tolist() == pytest.approx(
        [0.3] * len(scaled.daily_metrics)
    )
