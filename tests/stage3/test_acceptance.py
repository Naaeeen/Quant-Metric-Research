from __future__ import annotations

import json

import pandas as pd
import pytest

from quant_metric_research.benchmark import _acceptance
from quant_metric_research.benchmark_config import (
    BenchmarkConfig,
    NestedSplitConfig,
)
from quant_metric_research.benchmark_metrics import PredictionEvaluation


def _config() -> BenchmarkConfig:
    return BenchmarkConfig(
        feature_columns=("metric",),
        split=NestedSplitConfig(
            final_test_date_count=20,
            outer_n_splits=2,
            outer_test_date_count=3,
            outer_min_train_date_count=12,
            inner_n_splits=2,
            inner_validation_date_count=2,
            inner_min_train_date_count=6,
        ),
        min_cross_section=10,
        quantiles=5,
        hac_lags=1,
        model_families=("ridge",),
        minimum_rank_ic_improvement=0.0,
        minimum_locked_test_date_count=20,
        minimum_locked_score_coverage=0.75,
        minimum_locked_spread_date_count=20,
        minimum_locked_spread_coverage=0.75,
        minimum_coverage_ratio=0.8,
        maximum_locked_rank_ic_improvement_p_value=0.05,
    )


def _evaluation(
    *,
    model_locked_rank_ic: float = 0.03,
    baseline_locked_rank_ic: float = 0.01,
    locked_date_count: int = 20,
    locked_spread_date_count: int = 20,
    model_native_coverage: float = 0.4,
    baseline_native_coverage: float = 0.8,
    model_native_spread_coverage: float = 0.9,
    development_tie: bool = False,
) -> PredictionEvaluation:
    rows = [
        {
            "phase": "development",
            "model": "ridge",
            "evaluation_scope": "common",
            "date_count": 20,
            "spread_date_count": 20,
            "mean_rank_ic": 0.03,
            "median_rank_ic": 0.02,
            "mean_spread": 0.01,
            "mean_score_coverage": 0.5,
            "mean_spread_coverage": 0.9,
            "p_value": 0.01,
        },
        {
            "phase": "locked_test",
            "model": "ridge",
            "evaluation_scope": "common",
            "date_count": locked_date_count,
            "spread_date_count": locked_spread_date_count,
            "mean_rank_ic": model_locked_rank_ic,
            "median_rank_ic": 0.02,
            "mean_spread": 0.01,
            "mean_score_coverage": 0.5,
            "mean_spread_coverage": 0.9,
            "p_value": 0.01,
        },
        {
            "phase": "locked_test",
            "model": "equal_weight_rank",
            "evaluation_scope": "common",
            "date_count": locked_date_count,
            "spread_date_count": locked_spread_date_count,
            "mean_rank_ic": baseline_locked_rank_ic,
            "median_rank_ic": 0.01,
            "mean_spread": 0.005,
            "mean_score_coverage": 0.5,
            "mean_spread_coverage": 0.9,
            "p_value": 0.2,
        },
        {
            "phase": "locked_test",
            "model": "ridge",
            "evaluation_scope": "native",
            "date_count": locked_date_count,
            "spread_date_count": locked_spread_date_count,
            "mean_rank_ic": model_locked_rank_ic,
            "median_rank_ic": 0.02,
            "mean_spread": 0.01,
            "mean_score_coverage": model_native_coverage,
            "mean_spread_coverage": model_native_spread_coverage,
            "p_value": 0.01,
        },
        {
            "phase": "locked_test",
            "model": "equal_weight_rank",
            "evaluation_scope": "native",
            "date_count": locked_date_count,
            "spread_date_count": locked_spread_date_count,
            "mean_rank_ic": baseline_locked_rank_ic,
            "median_rank_ic": 0.01,
            "mean_spread": 0.005,
            "mean_score_coverage": baseline_native_coverage,
            "mean_spread_coverage": 0.9,
            "p_value": 0.2,
        },
    ]
    fold_rows = []
    for fold in (1, 2):
        fold_rows.extend(
            [
                {
                    "phase": "development",
                    "fold": fold,
                    "model": "ridge",
                    "evaluation_scope": "common",
                    "mean_rank_ic": 0.01 if development_tie else 0.03,
                },
                {
                    "phase": "development",
                    "fold": fold,
                    "model": "equal_weight_rank",
                    "evaluation_scope": "common",
                    "mean_rank_ic": 0.01,
                },
            ]
        )
    daily_rows = []
    for index, as_of_date in enumerate(
        pd.bdate_range("2025-01-02", periods=locked_date_count)
    ):
        baseline_rank_ic = baseline_locked_rank_ic + 0.001 * (index % 3 - 1)
        rank_ic_gap = model_locked_rank_ic - baseline_locked_rank_ic
        varying_gap = rank_ic_gap + 0.001 * ((index * 7) % 5 - 2)
        daily_rows.extend(
            [
                {
                    "phase": "locked_test",
                    "fold": 3,
                    "as_of_date": as_of_date,
                    "model": "ridge",
                    "evaluation_scope": "common",
                    "rank_ic": baseline_rank_ic + varying_gap,
                },
                {
                    "phase": "locked_test",
                    "fold": 3,
                    "as_of_date": as_of_date,
                    "model": "equal_weight_rank",
                    "evaluation_scope": "common",
                    "rank_ic": baseline_rank_ic,
                },
            ]
        )
    return PredictionEvaluation(
        daily_metrics=pd.DataFrame(daily_rows),
        fold_metrics=pd.DataFrame(fold_rows),
        summary=pd.DataFrame(rows),
    )


def test_acceptance_uses_native_and_absolute_coverage() -> None:
    acceptance = _acceptance(
        _evaluation(),
        config=_config(),
        frozen_family="ridge",
        expected_locked_dates=pd.bdate_range("2025-01-02", periods=20),
    )

    assert acceptance["locked_native_coverage_ratio"] == 0.5
    assert acceptance["locked_native_score_coverage"] == 0.4
    assert acceptance["checks"]["locked_native_coverage_ratio_passed"] is False
    assert acceptance["checks"]["locked_native_score_coverage_passed"] is False
    assert acceptance["checks"]["locked_rank_ic_improvement_p_value_passed"] is True
    assert acceptance["model_gate_passed"] is False


def test_acceptance_without_schedule_does_not_infer_from_surviving_rows() -> None:
    acceptance = _acceptance(
        _evaluation(model_native_coverage=0.9), config=_config(), frozen_family="ridge"
    )
    assert acceptance["locked_valid_date_count"] == 20
    assert acceptance["locked_rank_ic_improvement"] > 0
    assert (
        acceptance["locked_rank_ic_improvement_inference_status"]
        == "schedule_unavailable"
    )
    assert acceptance["locked_rank_ic_improvement_scheduled_date_count"] is None
    assert pd.isna(acceptance["locked_rank_ic_improvement_p_value"])
    assert not acceptance["checks"]["locked_rank_ic_improvement_p_value_passed"]
    strict = json.loads(json.dumps(dict(acceptance), allow_nan=False))
    assert strict["locked_rank_ic_improvement_p_value"] is None
    assert strict["locked_rank_ic_improvement_scheduled_date_count"] is None


@pytest.mark.parametrize("mode", ["missing_date", "missing_model", "null_model"])
def test_acceptance_keeps_gap_in_paired_schedule(mode) -> None:
    evaluation = _evaluation(model_native_coverage=0.9)
    original = evaluation.daily_metrics.copy(deep=True)
    missing_date = pd.Timestamp("2025-01-10")
    selected = original["as_of_date"].eq(missing_date)
    if mode != "missing_date":
        selected = selected & original["model"].eq("ridge")
    changed = (
        original.assign(rank_ic=original["rank_ic"].mask(selected))
        if mode == "null_model"
        else original.loc[~selected].copy()
    )
    acceptance = _acceptance(
        PredictionEvaluation(changed, evaluation.fold_metrics, evaluation.summary),
        config=_config(),
        frozen_family="ridge",
        expected_locked_dates=pd.bdate_range("2025-01-02", periods=20),
    )
    assert acceptance["locked_valid_date_count"] == 19
    assert acceptance["locked_rank_ic_improvement"] > 0
    assert (
        acceptance["locked_rank_ic_improvement_inference_status"]
        == "missing_scheduled_values"
    )
    assert acceptance["locked_rank_ic_improvement_scheduled_date_count"] == 20
    assert acceptance["locked_rank_ic_improvement_effective_hac_lags"] is None
    assert not acceptance["checks"]["locked_rank_ic_improvement_p_value_passed"]
    pd.testing.assert_frame_equal(evaluation.daily_metrics, original)


def test_acceptance_rejects_tie_short_lockbox_and_weak_inference() -> None:
    acceptance = _acceptance(
        _evaluation(
            model_locked_rank_ic=0.01,
            baseline_locked_rank_ic=0.01,
            locked_date_count=1,
            model_native_coverage=0.9,
            baseline_native_coverage=0.9,
            development_tie=True,
        ),
        config=_config(),
        frozen_family="ridge",
    )

    checks = acceptance["checks"]
    assert checks["locked_rank_ic_beats_baseline"] is False
    assert checks["development_win_rate_passed"] is False
    assert checks["locked_valid_date_count_passed"] is False
    assert checks["locked_rank_ic_improvement_p_value_passed"] is False
    assert pd.isna(acceptance["locked_rank_ic_improvement_p_value"])
    assert acceptance["model_gate_passed"] is False


def test_acceptance_requires_spread_sample_and_native_coverage() -> None:
    acceptance = _acceptance(
        _evaluation(
            locked_spread_date_count=1,
            model_native_coverage=0.9,
            baseline_native_coverage=0.9,
            model_native_spread_coverage=0.2,
        ),
        config=_config(),
        frozen_family="ridge",
    )

    checks = acceptance["checks"]
    assert checks["locked_spread_date_count_passed"] is False
    assert checks["locked_native_spread_coverage_passed"] is False
    assert acceptance["model_gate_passed"] is False
