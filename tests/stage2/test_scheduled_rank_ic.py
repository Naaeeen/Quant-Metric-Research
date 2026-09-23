"""Stage 2 inference uses the supplied schedule without changing sparse ICs."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quant_metric_research.screening import fit_metric_screen
from quant_metric_research.signals import IC_SUMMARY_COLUMNS, summarize_rank_ic
from quant_metric_research.statistics import newey_west_mean_tstat

DATES = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-07", "2025-01-10"])
DIAGNOSTICS = (
    "inference_status",
    "scheduled_date_count",
    "observed_date_count",
    "requested_hac_lags",
    "effective_hac_lags",
    "hac_lag_unit",
)


def _daily() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "as_of_date": DATES,
            "feature": "signal",
            "rank_ic": [0.2, -0.1, 0.5, 0.3],
            "cross_section_size": 4,
        }
    )


def test_omitted_schedule_keeps_descriptives_but_withholds_inference():
    daily = _daily()
    before = daily.copy(deep=True)
    summary = summarize_rank_ic(daily, hac_lags=1)
    row = summary.iloc[0]

    assert row["mean_rank_ic"] == pytest.approx(0.225)
    assert row["date_count"] == row["observed_date_count"] == 4
    assert row["positive_rate"] == pytest.approx(0.75)
    assert row["inference_status"] == "schedule_unavailable"
    assert pd.isna(row["scheduled_date_count"])
    assert pd.isna(row["effective_hac_lags"])
    assert row["requested_hac_lags"] == 1
    assert row["hac_lag_unit"] == "scheduled_observations"
    assert summary[["newey_west_t_stat", "p_value", "bh_q_value"]].isna().all().all()
    assert tuple(summary.columns) == IC_SUMMARY_COLUMNS
    assert tuple(summary.columns[-len(DIAGNOSTICS) :]) == DIAGNOSTICS
    encoded = json.loads(summary.to_json(orient="records"))[0]
    assert encoded["p_value"] is None and encoded["bh_q_value"] is None
    pd.testing.assert_frame_equal(daily, before)


def test_complete_declared_irregular_schedule_retains_supported_hac_values():
    daily = _daily().iloc[::-1].copy()
    before = daily.copy(deep=True)
    expected_t, expected_p = newey_west_mean_tstat(_daily()["rank_ic"], 1)

    row = summarize_rank_ic(daily, hac_lags=1, expected_dates=DATES).iloc[0]

    assert row["inference_status"] == "ok"
    assert row["newey_west_t_stat"] == pytest.approx(expected_t)
    assert row["p_value"] == pytest.approx(expected_p)
    assert row["scheduled_date_count"] == row["observed_date_count"] == 4
    assert row["effective_hac_lags"] == 1
    assert row["mean_rank_ic"] == pytest.approx(0.225)
    pd.testing.assert_frame_equal(daily, before)


@pytest.mark.parametrize("absent_row", [False, True])
def test_missing_scheduled_ic_is_not_compressed(absent_row):
    daily = _daily()
    if absent_row:
        daily = daily.drop(index=1)
    else:
        daily.loc[1, "rank_ic"] = np.nan

    row = summarize_rank_ic(daily, hac_lags=1, expected_dates=DATES).iloc[0]

    assert row["inference_status"] == "missing_scheduled_values"
    assert row["scheduled_date_count"] == 4
    assert row["observed_date_count"] == row["date_count"] == 3
    assert row["mean_rank_ic"] == pytest.approx(1.0 / 3.0)
    assert pd.isna(row["newey_west_t_stat"])
    assert pd.isna(row["p_value"])
    assert pd.isna(row["bh_q_value"])


def test_schedule_gap_is_not_replaced_with_union_of_other_features():
    daily = _daily().drop(index=1)
    row = summarize_rank_ic(daily, hac_lags=1, expected_dates=DATES).iloc[0]
    assert row["inference_status"] == "missing_scheduled_values"
    assert row["scheduled_date_count"] == 4


@pytest.mark.parametrize("empty_frame", [False, True])
@pytest.mark.parametrize(
    "schedule",
    [
        (),
        ("2025-01-03", "2025-01-02"),
        ("2025-01-02", "2025-01-02"),
        ("2025-01-02T12:00:00",),
        ("2025-01-02T00:00:00Z",),
        (123,),
    ],
)
def test_invalid_supplied_schedule_fails_even_without_feature_rows(
    schedule, empty_frame
):
    daily = _daily().iloc[:0] if empty_frame else _daily()
    with pytest.raises(ValueError):
        summarize_rank_ic(daily, hac_lags=1, expected_dates=schedule)


@pytest.mark.parametrize("with_schedule", [False, True])
def test_duplicate_normalized_feature_date_values_are_rejected(with_schedule):
    daily = pd.concat([_daily(), _daily().iloc[:1]], ignore_index=True)
    daily["as_of_date"] = daily["as_of_date"].astype(object)
    daily.loc[len(daily) - 1, "as_of_date"] = "2025-01-02"
    options = {"expected_dates": DATES} if with_schedule else {}
    with pytest.raises(ValueError, match="unique|[Dd]uplicate"):
        summarize_rank_ic(daily, hac_lags=1, **options)


def test_observation_outside_schedule_is_rejected():
    with pytest.raises(ValueError):
        summarize_rank_ic(_daily(), hac_lags=1, expected_dates=DATES[:-1])


@pytest.mark.parametrize("invalid", ["bad", "NaN", True, complex(1, 2), np.inf])
def test_present_invalid_ic_is_not_coerced_into_a_schedule_gap(invalid):
    daily = _daily()
    daily["rank_ic"] = daily["rank_ic"].astype(object)
    daily.loc[1, "rank_ic"] = invalid
    with pytest.raises(ValueError):
        summarize_rank_ic(daily, hac_lags=1, expected_dates=DATES)


@pytest.mark.parametrize("hac_lags", [-1, True, 0.5])
def test_invalid_lags_fail_even_without_schedule_or_rows(hac_lags):
    with pytest.raises(ValueError, match="hac_lags"):
        summarize_rank_ic(_daily().iloc[:0], hac_lags=hac_lags)


def test_empty_sparse_ic_keeps_absent_features_and_complete_summary_schema():
    result = summarize_rank_ic(_daily().iloc[:0], hac_lags=1, expected_dates=DATES)
    assert result.empty
    assert tuple(result.columns) == IC_SUMMARY_COLUMNS
    assert set(DIAGNOSTICS).issubset(result.columns)


@pytest.mark.parametrize(
    "values, lags, status, observed",
    [
        ([0.2], 0, "insufficient_observations", 1),
        ([0.2, -0.1], 2, "insufficient_lag_support", 2),
        ([0.2, 0.2], 1, "undefined_variance", 2),
        ([np.nan, np.nan], 1, "missing_scheduled_values", 0),
    ],
)
def test_unavailable_inference_statuses_retain_sparse_descriptive_counts(
    values, lags, status, observed
):
    daily = _daily().iloc[: len(values)].assign(rank_ic=values)
    summary = summarize_rank_ic(
        daily, hac_lags=lags, expected_dates=DATES[: len(values)]
    )
    row = summary.iloc[0]
    assert row["inference_status"] == status
    assert row["scheduled_date_count"] == len(values)
    assert row["date_count"] == row["observed_date_count"] == observed
    assert row["requested_hac_lags"] == lags
    assert pd.isna(row["effective_hac_lags"])
    assert summary[["newey_west_t_stat", "p_value", "bh_q_value"]].isna().all().all()


def test_partial_feature_does_not_suppress_complete_feature_inference():
    daily = pd.concat(
        [_daily(), _daily().drop(index=1).assign(feature="partial")],
        ignore_index=True,
    )
    summary = summarize_rank_ic(daily, hac_lags=1, expected_dates=DATES).set_index(
        "feature"
    )
    assert summary.at["signal", "inference_status"] == "ok"
    assert summary.at["partial", "inference_status"] == "missing_scheduled_values"
    assert summary.at["signal", "bh_q_value"] == pytest.approx(
        summary.at["signal", "p_value"]
    )
    assert pd.isna(summary.at["partial", "bh_q_value"])


@pytest.mark.parametrize("daily", [None, [1.0], {"rank_ic": [1.0]}])
def test_summary_requires_a_dataframe(daily):
    with pytest.raises(ValueError, match="pandas DataFrame"):
        summarize_rank_ic(daily, hac_lags=1)


def test_legacy_two_column_input_remains_descriptive_without_a_schedule():
    daily = _daily().loc[:, ["feature", "rank_ic"]]
    row = summarize_rank_ic(daily, hac_lags=1).iloc[0]
    assert row["inference_status"] == "schedule_unavailable"
    assert row["date_count"] == 4
    assert row["mean_rank_ic"] == pytest.approx(0.225)
    with pytest.raises(ValueError, match="as_of_date"):
        summarize_rank_ic(daily, hac_lags=1, expected_dates=DATES)


@pytest.mark.parametrize("date", ["invalid", "2025-01-02T12:00:00", 123])
def test_provided_observation_dates_are_validated_without_a_schedule(date):
    daily = _daily()
    daily["as_of_date"] = daily["as_of_date"].astype(object)
    daily.loc[1, "as_of_date"] = date
    with pytest.raises(ValueError, match="as_of_date"):
        summarize_rank_ic(daily, hac_lags=1)


def test_duplicate_columns_fail_clearly():
    daily = pd.concat([_daily(), _daily()[["rank_ic"]]], axis=1)
    with pytest.raises(ValueError, match="Duplicate daily IC columns"):
        summarize_rank_ic(daily, hac_lags=1)


@pytest.mark.parametrize("feature", [None, np.nan, pd.NA])
@pytest.mark.parametrize("with_schedule", [False, True])
def test_null_feature_labels_cannot_bypass_row_validation(feature, with_schedule):
    extra = _daily().iloc[:1].assign(feature=feature, rank_ic="bad")
    extra["as_of_date"] = pd.Timestamp("2025-01-11")
    daily = pd.concat([_daily(), extra], ignore_index=True)
    options = {"expected_dates": DATES} if with_schedule else {}
    with pytest.raises(ValueError, match="feature"):
        summarize_rank_ic(daily, hac_lags=1, **options)


def _screen_panel() -> pd.DataFrame:
    rows = []
    targets = ([0.0, 1.0, 2.0, 3.0], [0.0] * 4, [3.0, 2.0, 1.0, 0.0], [0, 1, 3, 2])
    for date, target in zip(DATES, targets, strict=True):
        for symbol, value in enumerate(target):
            rows.append(
                {
                    "as_of_date": date,
                    "symbol": f"S{symbol}",
                    "signal": float(symbol),
                    "constant": 1.0,
                    "target": value,
                }
            )
    return pd.DataFrame(rows)


def test_screen_captures_all_training_dates_before_every_ic_on_one_date_is_invalid():
    panel = _screen_panel()
    before = panel.copy(deep=True)
    result = fit_metric_screen(
        panel,
        feature_columns=("signal", "constant"),
        target_column="target",
        train_end_date=DATES[-1],
        min_cross_section=4,
        minimum_coverage=0.8,
        redundancy_threshold=0.9,
        hac_lags=1,
        quantiles=2,
    )
    row = result.ic_summary.set_index("feature").loc["signal"]

    assert result.daily_rank_ic["as_of_date"].tolist() == list(DATES.delete(1))
    assert result.ic_summary["feature"].tolist() == ["signal"]
    assert row["inference_status"] == "missing_scheduled_values"
    assert row["scheduled_date_count"] == 4
    assert row["observed_date_count"] == row["date_count"] == 3
    assert row["mean_rank_ic"] == pytest.approx((1.0 - 1.0 + 0.8) / 3.0)
    assert result.selected_features == ("signal",)
    assert result.dropped_features["constant"] == "zero_variance"
    pd.testing.assert_frame_equal(panel, before)


def test_screen_schedule_stops_at_training_cutoff():
    panel = _screen_panel()
    result = fit_metric_screen(
        panel,
        feature_columns=("signal",),
        target_column="target",
        train_end_date=DATES[2],
        min_cross_section=4,
        minimum_coverage=0.8,
        redundancy_threshold=0.9,
        hac_lags=1,
        quantiles=2,
    )
    row = result.ic_summary.iloc[0]
    assert row["scheduled_date_count"] == 3
    assert row["observed_date_count"] == 2
    assert row["inference_status"] == "missing_scheduled_values"


def test_screen_schedule_uses_configured_date_column_without_filling_ic_gaps():
    panel = _screen_panel().rename(columns={"as_of_date": "decision_date"})
    result = fit_metric_screen(
        panel,
        feature_columns=("signal",),
        target_column="target",
        train_end_date=DATES[-1],
        min_cross_section=4,
        minimum_coverage=0.8,
        redundancy_threshold=0.9,
        hac_lags=1,
        quantiles=2,
        as_of_date_column="decision_date",
    )
    row = result.ic_summary.iloc[0]
    assert result.daily_rank_ic["as_of_date"].tolist() == list(DATES.delete(1))
    assert row["scheduled_date_count"] == 4
    assert row["observed_date_count"] == 3
    assert row["inference_status"] == "missing_scheduled_values"
