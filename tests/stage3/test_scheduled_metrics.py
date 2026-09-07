from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_metric_research.benchmark_metrics import evaluate_prediction_frame
from quant_metric_research.statistics import newey_west_mean_tstat

INFERENCE_COLUMNS = (
    "inference_status",
    "scheduled_date_count",
    "observed_date_count",
    "requested_hac_lags",
    "effective_hac_lags",
    "hac_lag_unit",
)


@pytest.fixture()
def scheduled_dates() -> pd.DatetimeIndex:
    # The declared observations, rather than calendar or business-day distance,
    # define a lag. Deliberately leave unequal gaps between decisions.
    return pd.to_datetime(
        ["2025-01-02", "2025-01-06", "2025-01-07", "2025-01-15", "2025-01-20"]
    )


@pytest.fixture()
def scheduled_predictions(scheduled_dates) -> pd.DataFrame:
    scores = ((0, 1, 2, 3), (0, 2, 1, 3), (2, 0, 3, 1), (2, 3, 1, 0), (1, 0, 3, 2))
    return pd.DataFrame(
        [
            {
                "phase": "development",
                "fold": 1 if day < 2 else 2,
                "as_of_date": date,
                "symbol": f"S{stock}",
                "model": model,
                "score": float(scores[day][stock] + offset),
                "target": float(stock),
                "realized_return": float(stock) / 10,
            }
            for day, date in enumerate(scheduled_dates)
            for stock in range(4)
            for model, offset in (("ridge", 0), ("equal_weight_rank", 0.5))
        ]
    )


def _evaluate(predictions, **kwargs):
    return evaluate_prediction_frame(
        predictions,
        min_cross_section=3,
        quantiles=3,
        hac_lags=1,
        primary_models=("ridge", "equal_weight_rank"),
        **kwargs,
    )


def test_complete_irregular_schedule_preserves_numerical_inference(
    scheduled_predictions, scheduled_dates
) -> None:
    predictions = scheduled_predictions.iloc[::-1].copy(deep=True)
    original = predictions.copy(deep=True)
    dates_before = scheduled_dates.copy(deep=True)
    result = _evaluate(
        predictions, expected_dates_by_phase={"development": scheduled_dates}
    )
    for row in result.summary.itertuples(index=False):
        daily = result.daily_metrics.loc[
            (result.daily_metrics["model"] == row.model)
            & (result.daily_metrics["evaluation_scope"] == row.evaluation_scope)
        ].sort_values("as_of_date")
        expected_t, expected_p = newey_west_mean_tstat(daily["rank_ic"], 1)
        assert row.inference_status == "ok"
        assert row.newey_west_t_stat == pytest.approx(expected_t, rel=1e-14)
        assert row.p_value == pytest.approx(expected_p, rel=1e-14)
        assert row.scheduled_date_count == row.observed_date_count == 5
        assert row.requested_hac_lags == row.effective_hac_lags == 1
        assert row.hac_lag_unit == "scheduled_observations"
    assert result.summary["bh_q_value"].notna().all()
    pd.testing.assert_frame_equal(predictions, original)
    pd.testing.assert_index_equal(scheduled_dates, dates_before)


@pytest.mark.parametrize("gap", ["absent_date", "undefined_ic"])
def test_missing_scheduled_ic_suppresses_inference_without_compressing_time(
    scheduled_predictions, scheduled_dates, gap
) -> None:
    missing_date = scheduled_dates[2]
    if gap == "absent_date":
        predictions = scheduled_predictions.loc[
            scheduled_predictions["as_of_date"] != missing_date
        ].copy(deep=True)
    else:
        predictions = scheduled_predictions.assign(
            score=scheduled_predictions["score"].where(
                scheduled_predictions["as_of_date"] != missing_date, 1.0
            )
        )
    original = predictions.copy(deep=True)
    scheduled = _evaluate(
        predictions, expected_dates_by_phase={"development": scheduled_dates}
    )
    unscheduled = _evaluate(predictions)

    assert set(scheduled.summary["inference_status"]) == {"missing_scheduled_values"}
    assert (scheduled.summary["scheduled_date_count"] == 5).all()
    assert (scheduled.summary["observed_date_count"] == 4).all()
    assert (scheduled.summary["date_count"] == 4).all()
    assert (
        scheduled.summary[
            ["newey_west_t_stat", "p_value", "bh_q_value", "effective_hac_lags"]
        ]
        .isna()
        .all()
        .all()
    )
    descriptive = scheduled.summary.columns.difference(
        ["newey_west_t_stat", "p_value", "bh_q_value", *INFERENCE_COLUMNS], sort=False
    )
    pd.testing.assert_frame_equal(
        scheduled.summary.loc[:, descriptive], unscheduled.summary.loc[:, descriptive]
    )
    pd.testing.assert_frame_equal(scheduled.daily_metrics, unscheduled.daily_metrics)
    pd.testing.assert_frame_equal(scheduled.fold_metrics, unscheduled.fold_metrics)
    pd.testing.assert_frame_equal(predictions, original)


@pytest.mark.parametrize(
    "schedules", ["omitted", None, {}, {"locked_test": ["2025-02-03"]}]
)
def test_missing_phase_schedule_has_unknown_count_and_no_inference(
    scheduled_predictions, schedules
) -> None:
    kwargs = {} if schedules == "omitted" else {"expected_dates_by_phase": schedules}
    result = _evaluate(scheduled_predictions, **kwargs)
    assert set(result.summary["inference_status"]) == {"schedule_unavailable"}
    assert (result.summary["observed_date_count"] == 5).all()
    assert (result.summary["requested_hac_lags"] == 1).all()
    assert (
        result.summary[
            [
                "scheduled_date_count",
                "effective_hac_lags",
                "newey_west_t_stat",
                "p_value",
            ]
        ]
        .isna()
        .all()
        .all()
    )


@pytest.mark.parametrize(
    "dates",
    [
        [],
        ["2025-01-02", "2025-01-02"],
        ["2025-01-03", "2025-01-02"],
        ["not-a-date"],
        [None],
        [0],
        ["2025-01-02T12:00:00"],
        ["2025-01-02T00:00:00Z"],
    ],
)
@pytest.mark.parametrize("phase", ["development", "absent_phase"])
def test_supplied_malformed_schedules_fail_even_without_corresponding_rows(
    scheduled_predictions, dates, phase
) -> None:
    with pytest.raises(ValueError):
        _evaluate(scheduled_predictions, expected_dates_by_phase={phase: dates})


@pytest.mark.parametrize("empty_predictions", [False, True])
def test_malformed_schedule_is_validated_before_daily_metrics(
    scheduled_predictions, empty_predictions
) -> None:
    predictions = (
        scheduled_predictions.iloc[:0] if empty_predictions else scheduled_predictions
    )
    with pytest.raises(ValueError, match="expected_dates|schedule"):
        _evaluate(predictions, expected_dates_by_phase={"absent_phase": []})


def test_prediction_date_outside_supplied_phase_schedule_is_rejected(
    scheduled_predictions, scheduled_dates
) -> None:
    with pytest.raises(ValueError, match="outside|unexpected"):
        _evaluate(
            scheduled_predictions,
            expected_dates_by_phase={"development": scheduled_dates[:-1]},
        )


@pytest.mark.parametrize("mapping", [[], 5, {"": ["2025-01-02"]}, {1: ["2025-01-02"]}])
def test_phase_schedules_require_a_mapping_with_nonempty_string_keys(
    scheduled_predictions, mapping
) -> None:
    with pytest.raises(ValueError, match="expected_dates_by_phase"):
        _evaluate(scheduled_predictions, expected_dates_by_phase=mapping)


@pytest.mark.parametrize("with_schedule", [False, True])
def test_duplicate_phase_model_scope_date_across_folds_is_rejected(
    scheduled_predictions, scheduled_dates, with_schedule
) -> None:
    repeated = scheduled_predictions.loc[
        scheduled_predictions["as_of_date"] == scheduled_dates[0]
    ].assign(fold=3, symbol=lambda rows: rows["symbol"] + "_other")
    predictions = pd.concat([scheduled_predictions, repeated], ignore_index=True)
    schedules = {"development": scheduled_dates} if with_schedule else None
    with pytest.raises(ValueError, match="unique.*phase/model/scope/date"):
        _evaluate(predictions, expected_dates_by_phase=schedules)


def test_missing_native_model_date_uses_full_phase_schedule(
    scheduled_predictions, scheduled_dates
) -> None:
    predictions = scheduled_predictions.loc[
        ~(
            (scheduled_predictions["as_of_date"] == scheduled_dates[2])
            & (scheduled_predictions["model"] == "ridge")
        )
    ]
    result = evaluate_prediction_frame(
        predictions,
        min_cross_section=3,
        quantiles=3,
        hac_lags=1,
        primary_models=(),
        expected_dates_by_phase={"development": scheduled_dates},
    )
    summary = result.summary.set_index("model")
    assert summary.at["ridge", "inference_status"] == "missing_scheduled_values"
    assert summary.at["ridge", "scheduled_date_count"] == 5
    assert summary.at["ridge", "observed_date_count"] == 4
    assert summary.at["equal_weight_rank", "inference_status"] == "ok"
    assert np.isnan(summary.at["ridge", "p_value"])


@pytest.mark.parametrize("date_kind", ["numeric_epoch", "intraday", "timezone_aware"])
@pytest.mark.parametrize("with_schedule", [False, True])
def test_prediction_date_input_is_validated_before_timestamp_coercion(
    scheduled_predictions, scheduled_dates, date_kind, with_schedule
) -> None:
    original_dates = scheduled_predictions["as_of_date"]
    invalid_dates = {
        "numeric_epoch": original_dates.map(lambda value: value.value),
        "intraday": original_dates + pd.Timedelta(hours=12),
        "timezone_aware": original_dates.dt.tz_localize("UTC"),
    }[date_kind]
    predictions = scheduled_predictions.assign(as_of_date=invalid_dates)
    original = predictions.copy(deep=True)
    schedules = {"development": scheduled_dates} if with_schedule else None

    with pytest.raises(ValueError, match="Prediction dates"):
        _evaluate(predictions, expected_dates_by_phase=schedules)
    pd.testing.assert_frame_equal(predictions, original)
