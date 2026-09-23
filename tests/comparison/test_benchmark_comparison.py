"""Descriptive reports reuse saved scores and retain scheduled comparison gaps."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from importlib import import_module

import numpy as np
import pandas as pd
import pytest

SCHEDULE = ("2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09", "2025-01-10")
ARMS = ("legacy_equal_rank", "legacy_model", "candidate_model")
CONTRASTS = (
    "candidate_model_minus_legacy_model",
    "candidate_model_minus_legacy_equal_rank",
)
DAILY_DELTA_COLUMNS = (
    "evaluation_scope",
    "contrast",
    "as_of_date",
    "rank_ic_delta",
    "spread_delta",
)
DELTA_SUMMARY_COLUMNS = (
    "evaluation_scope",
    "contrast",
    "scheduled_date_count",
    "paired_rank_ic_date_count",
    "paired_rank_ic_date_coverage",
    "mean_paired_rank_ic_delta",
    "paired_spread_date_count",
    "paired_spread_date_coverage",
    "mean_paired_spread_delta",
)


def _module():
    return import_module("quant_metric_research.benchmark_comparison")


def _metric_fixture():
    dates = pd.to_datetime([SCHEDULE[i] for i in (0, 1, 3, 4)])
    values = {
        "candidate_model": ([1.0, np.nan, 0.4, 0.8], [0.2, 0.3, 0.4, 0.5]),
        "legacy_model": ([0.2, 0.9, np.nan, 0.3], [0.1, np.nan, 0.3, 0.2]),
        "legacy_equal_rank": ([0.0, 0.5, 0.2, 0.1], [0.1] * 4),
    }
    return pd.DataFrame(
        [
            {
                "evaluation_scope": scope,
                "model": model,
                "as_of_date": date,
                "rank_ic": ic[index],
                "spread": spreads[index],
            }
            for scope in ("native", "common")
            for model, (ic, spreads) in values.items()
            for index, date in enumerate(dates)
        ]
    )


def test_differences_are_paired_daily_before_aggregation():
    daily, summary = _module()._paired_contrasts(
        _metric_fixture(), expected_dates=SCHEDULE
    )
    assert tuple(daily.columns) == DAILY_DELTA_COLUMNS
    assert tuple(summary.columns) == DELTA_SUMMARY_COLUMNS
    assert len(daily) == 20 and len(summary) == 4
    for scope in ("native", "common"):
        row = summary.loc[
            summary["evaluation_scope"].eq(scope) & summary["contrast"].eq(CONTRASTS[0])
        ].iloc[0]
        assert row["mean_paired_rank_ic_delta"] == pytest.approx(0.65)
        assert row["mean_paired_rank_ic_delta"] != pytest.approx(
            np.mean([1.0, 0.4, 0.8]) - np.mean([0.2, 0.9, 0.3])
        )
        assert row["paired_rank_ic_date_count"] == 2
        assert row["paired_rank_ic_date_coverage"] == pytest.approx(2 / 5)
        assert row["paired_spread_date_count"] == 3
        assert row["paired_spread_date_coverage"] == pytest.approx(3 / 5)
        assert row["mean_paired_spread_delta"] == pytest.approx(1 / 6)
        assert row["scheduled_date_count"] == 5


def test_complete_schedule_keeps_between_fold_and_metric_gaps():
    daily, _ = _module()._paired_contrasts(_metric_fixture(), expected_dates=SCHEDULE)
    gap = daily.loc[daily["as_of_date"].eq(pd.Timestamp(SCHEDULE[2]))]
    assert len(gap) == 4
    assert gap[["rank_ic_delta", "spread_delta"]].isna().all().all()
    for _, group in daily.groupby(["evaluation_scope", "contrast"]):
        assert group["as_of_date"].tolist() == list(pd.to_datetime(SCHEDULE))


def test_no_pairs_remain_unavailable_instead_of_becoming_zero():
    metrics = _metric_fixture().assign(rank_ic=np.nan, spread=np.nan)
    daily, summary = _module()._paired_contrasts(metrics, expected_dates=SCHEDULE)
    assert daily[["rank_ic_delta", "spread_delta"]].isna().all().all()
    assert (
        summary[["mean_paired_rank_ic_delta", "mean_paired_spread_delta"]]
        .isna()
        .all()
        .all()
    )
    for metric in ("rank_ic", "spread"):
        assert summary[f"paired_{metric}_date_count"].eq(0).all()
        assert summary[f"paired_{metric}_date_coverage"].eq(0.0).all()


def test_contrasts_use_finite_pairs_without_mutating_or_reordering_inputs():
    metrics = _metric_fixture()
    metrics.loc[metrics["model"].eq("candidate_model"), "rank_ic"] = np.inf
    before = metrics.copy(deep=True)
    daily, summary = _module()._paired_contrasts(metrics, expected_dates=SCHEDULE)
    assert daily["rank_ic_delta"].isna().all()
    assert summary["paired_rank_ic_date_count"].eq(0).all()
    pd.testing.assert_frame_equal(metrics, before)
    shuffled = metrics.sample(frac=1, random_state=11)
    other_daily, other_summary = _module()._paired_contrasts(
        shuffled, expected_dates=SCHEDULE
    )
    pd.testing.assert_frame_equal(daily, other_daily)
    pd.testing.assert_frame_equal(summary, other_summary)


def test_large_finite_paired_mean_does_not_overflow_its_intermediate_sum():
    metrics = _metric_fixture().assign(spread=0.0)
    metrics.loc[metrics["model"].eq("candidate_model"), "spread"] = 1e308
    _, summary = _module()._paired_contrasts(metrics, expected_dates=SCHEDULE)
    assert summary["mean_paired_spread_delta"].eq(1e308).all()


def test_unrepresentable_daily_difference_fails_clearly():
    metrics = _metric_fixture().assign(spread=-1e308)
    metrics.loc[metrics["model"].eq("candidate_model"), "spread"] = 1e308
    with pytest.raises(ValueError, match="finite|represent"):
        _module()._paired_contrasts(metrics, expected_dates=SCHEDULE)


def test_paired_mean_retains_small_terms_between_opposite_large_differences():
    metrics = _metric_fixture().assign(spread=0.0)
    candidate = metrics["model"].eq("candidate_model")
    metrics.loc[candidate, "spread"] = [1e308, 3.0, -1e308, 0.0] * 2
    _, summary = _module()._paired_contrasts(metrics, expected_dates=SCHEDULE)
    assert summary["mean_paired_spread_delta"].to_numpy() == pytest.approx([0.75] * 4)


def test_paired_mean_does_not_underflow_tiny_terms_when_extremes_cancel():
    metrics = _metric_fixture().assign(spread=0.0)
    candidate = metrics["model"].eq("candidate_model")
    metrics.loc[candidate, "spread"] = [1e308, 1e-16, -1e308, 0.0] * 2
    _, summary = _module()._paired_contrasts(metrics, expected_dates=SCHEDULE)
    assert summary["mean_paired_spread_delta"].eq(2.5e-17).all()


def test_public_report_keeps_three_arms_and_two_explicit_scopes(comparison_runs):
    result = _module().compare_feature_bundles(*comparison_runs, model_family="ridge")
    assert isinstance(result, _module().FeatureBundleComparison)
    assert set(result.daily_metrics["model"]) == set(ARMS)
    assert set(result.daily_metrics["evaluation_scope"]) == {"native", "common"}
    assert len(result.daily_metrics) == 24
    assert len(result.fold_metrics) == 12
    assert len(result.summary) == 6
    assert len(result.daily_deltas) == 20
    assert len(result.delta_summary) == 4
    assert set(result.daily_deltas["contrast"]) == set(CONTRASTS)


def test_summary_exports_explicit_descriptive_fields_only(comparison_runs):
    result = _module().compare_feature_bundles(*comparison_runs, model_family="ridge")
    assert tuple(result.summary.columns) == _module().DESCRIPTIVE_SUMMARY_COLUMNS
    assert {"rank_ic_ir", "date_count", "mean_score_coverage"}.issubset(result.summary)
    for table in (result.summary, result.daily_deltas, result.delta_summary):
        assert not any(
            name in table
            for name in (
                "p_value",
                "bh_q_value",
                "newey_west_t_stat",
                "inference_status",
                "winner",
                "model_gate_passed",
                "acceptance_status",
            )
        )


def test_report_records_comparator_identity_separately_from_producer(comparison_runs):
    result = _module().compare_feature_bundles(*comparison_runs, model_family="ridge")
    metadata = result.metadata
    assert metadata["claim_scope"] == "development_descriptive_comparison"
    assert metadata["comparison_definition_version"] == "1"
    assert metadata["caller_owned_inputs"] is True
    for field in (
        "authenticity_verified",
        "history_verified",
        "final_outcomes_evaluated",
        "inference_reported",
    ):
        assert metadata[field] is False
    assert metadata["model_family"] == "ridge"
    assert metadata["expected_dates"] == SCHEDULE
    assert metadata["producer_identity"]["package_version"] == "fixture-producer-v1"
    assert metadata["comparison_identity"]["package_version"] != "fixture-producer-v1"
    fingerprint = metadata["comparison_identity"]["source_fingerprint"]
    assert len(fingerprint) == 64 and set(fingerprint) <= set("0123456789abcdef")
    assert metadata["comparison_identity"]["pandas_version"] == pd.__version__


@pytest.mark.parametrize(
    "name",
    ["daily_metrics", "fold_metrics", "summary", "daily_deltas", "delta_summary"],
)
def test_report_table_access_is_a_defensive_copy(comparison_runs, name):
    result = _module().compare_feature_bundles(*comparison_runs, model_family="ridge")
    original = getattr(result, name)
    changed = getattr(result, name)
    changed.iloc[0, 0] = "changed"
    pd.testing.assert_frame_equal(getattr(result, name), original)
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        setattr(result, name, changed)


def test_report_metadata_is_deeply_immutable(comparison_runs):
    result = _module().compare_feature_bundles(*comparison_runs, model_family="ridge")
    with pytest.raises(TypeError):
        result.metadata["claim_scope"] = "changed"
    with pytest.raises(TypeError):
        result.metadata["comparison_identity"]["package_version"] = "changed"
    with pytest.raises(TypeError):
        result.metadata["producer_identity"]["package_version"] = "changed"


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        ["not-a-mapping"],
        {1: "not-a-string-key"},
        {"bad": np.inf},
        {"bad": {1, 2}},
    ],
)
def test_result_constructor_rejects_non_json_metadata(comparison_runs, metadata):
    result = _module().compare_feature_bundles(*comparison_runs, model_family="ridge")
    with pytest.raises(TypeError, match="metadata|Metadata"):
        replace(result, metadata=metadata)


def test_constructor_copies_supplied_tables_and_mutable_nested_metadata(
    comparison_runs,
):
    result = _module().compare_feature_bundles(*comparison_runs, model_family="ridge")
    table = result.daily_metrics
    saved = table.copy(deep=True)
    metadata = {"nested": [{"value": "original"}]}
    copied = replace(result, metadata=metadata, _daily_metrics=table)
    metadata["nested"][0]["value"] = "changed"
    table.iloc[0, 0] = "changed"
    assert copied.metadata["nested"][0]["value"] == "original"
    pd.testing.assert_frame_equal(copied.daily_metrics, saved)
    with pytest.raises(TypeError, match="pandas DataFrames"):
        replace(result, _daily_metrics=None)


@pytest.mark.parametrize("cell", [[1], {"value": 1}, np.array([1])])
def test_constructor_rejects_mutable_containers_inside_table_cells(
    comparison_runs, cell
):
    result = _module().compare_feature_bundles(*comparison_runs, model_family="ridge")
    table = pd.DataFrame({"value": pd.Series([cell], dtype=object)})
    with pytest.raises(TypeError, match="scalar|container"):
        replace(result, _daily_metrics=table)


def test_one_three_arm_common_intersection_retains_native_coverage(comparison_runs):
    legacy, candidate = comparison_runs
    legacy_scores = legacy.predictions.copy(deep=True)
    candidate_scores = candidate.predictions.copy(deep=True)
    legacy_scores.loc[
        legacy_scores["model"].eq("ridge") & legacy_scores["symbol"].eq("B"), "score"
    ] = np.nan
    candidate_scores.loc[
        candidate_scores["model"].eq("ridge") & candidate_scores["symbol"].eq("A"),
        "score",
    ] = np.nan
    result = _module().compare_feature_bundles(
        replace(legacy, predictions=legacy_scores),
        replace(candidate, predictions=candidate_scores),
        model_family="ridge",
    )
    native = result.daily_metrics.loc[
        lambda frame: frame["evaluation_scope"].eq("native")
    ]
    common = result.daily_metrics.loc[
        lambda frame: frame["evaluation_scope"].eq("common")
    ]
    assert common["scoring_universe_count"].eq(4).all()
    assert common["scored_count"].eq(2).all()
    assert common["score_coverage"].eq(0.5).all()
    assert (
        native.loc[native["model"].eq("legacy_equal_rank"), "scored_count"].eq(4).all()
    )
    assert (
        native.loc[~native["model"].eq("legacy_equal_rank"), "scored_count"].eq(3).all()
    )


def test_outcome_missingness_changes_pairs_not_scores_or_score_coverage(
    comparison_runs, monkeypatch
):
    module = _module()
    captured = []
    original_evaluator = module.evaluate_prediction_frame

    def capture(predictions, **kwargs):
        captured.append(predictions.copy(deep=True))
        return original_evaluator(predictions, **kwargs)

    monkeypatch.setattr(module, "evaluate_prediction_frame", capture)
    original = module.compare_feature_bundles(*comparison_runs, model_family="ridge")
    changed = []
    for run in comparison_runs:
        predictions = run.predictions.copy(deep=True)
        predictions.loc[predictions["symbol"].eq("A"), "target"] = np.nan
        predictions.loc[predictions["symbol"].eq("B"), "realized_return"] = np.nan
        assignments = run.fold_assignments.copy(deep=True)
        missing_target_keys = pd.MultiIndex.from_frame(
            predictions.loc[predictions["symbol"].eq("A"), ["fold", "row_id"]]
        )
        affected = pd.MultiIndex.from_frame(assignments[["fold", "row_id"]]).isin(
            missing_target_keys
        )
        assignments.loc[affected, "role"] = "excluded"
        assignments.loc[affected, "exclusion_reason"] = "missing_target"
        changed.append(
            replace(run, predictions=predictions, fold_assignments=assignments)
        )
    result = module.compare_feature_bundles(*changed, model_family="ridge")
    keys = ["phase", "fold", "as_of_date", "symbol", "model"]
    pd.testing.assert_frame_equal(
        captured[0].set_index(keys)[["score"]], captured[1].set_index(keys)[["score"]]
    )
    pd.testing.assert_series_equal(
        original.daily_metrics["score_coverage"], result.daily_metrics["score_coverage"]
    )
    assert result.daily_metrics["rank_ic_count"].eq(3).all()
    assert result.daily_metrics["spread_count"].eq(3).all()


@pytest.mark.parametrize("score", [np.nan, 1.0])
def test_unusable_candidate_metrics_report_zero_pairs_without_a_winner(
    comparison_runs, score
):
    legacy, candidate = comparison_runs
    predictions = candidate.predictions.copy(deep=True)
    predictions.loc[predictions["model"].eq("ridge"), "score"] = score
    result = _module().compare_feature_bundles(
        legacy, replace(candidate, predictions=predictions), model_family="ridge"
    )
    assert result.delta_summary["paired_rank_ic_date_count"].eq(0).all()
    assert result.delta_summary["mean_paired_rank_ic_delta"].isna().all()
    assert result.delta_summary["paired_spread_date_count"].eq(0).all()


def test_inputs_and_results_are_unchanged_by_ties_and_row_order(comparison_runs):
    runs = []
    for run in comparison_runs:
        predictions = run.predictions.copy(deep=True)
        predictions.loc[predictions["model"].eq("ridge"), "score"] = predictions[
            "symbol"
        ].map({"A": 0.0, "B": 0.0, "C": 1.0, "D": 1.0})
        runs.append(replace(run, predictions=predictions))
    before = [run.predictions.copy(deep=True) for run in runs]
    original = _module().compare_feature_bundles(*runs, model_family="ridge")
    reordered = [
        replace(run, predictions=run.predictions.sample(frac=1, random_state=23))
        for run in runs
    ]
    result = _module().compare_feature_bundles(*reordered, model_family="ridge")
    for run, saved in zip(runs, before, strict=True):
        pd.testing.assert_frame_equal(run.predictions, saved)
    for name in (
        "daily_metrics",
        "fold_metrics",
        "summary",
        "daily_deltas",
        "delta_summary",
    ):
        pd.testing.assert_frame_equal(getattr(original, name), getattr(result, name))
