from types import SimpleNamespace

import pandas as pd
import pytest

from quant_metric_research.benchmark import (
    _paired_locked_rank_ic_improvement,
    _phase_schedule,
)


def test_engine_uses_the_shared_schedule_derivation():
    from quant_metric_research.benchmark_schedules import phase_schedule

    assert _phase_schedule is phase_schedule


def test_phase_schedule_uses_panel_bounds_and_keeps_between_fold_dates():
    dates = pd.to_datetime(
        ["2025-01-02", "2025-01-03", "2025-01-07", "2025-01-09", "2025-01-10"]
    )
    plan = SimpleNamespace(
        panel=SimpleNamespace(frame=pd.DataFrame({"as_of_date": [*dates, *dates]})),
        development_folds=(
            SimpleNamespace(
                evaluation_start_date=dates[0], evaluation_end_date=dates[1]
            ),
            SimpleNamespace(
                evaluation_start_date=dates[3], evaluation_end_date=dates[3]
            ),
        ),
        locked_test=SimpleNamespace(test_start_date=dates[4], test_end_date=dates[4]),
    )
    assert list(_phase_schedule(plan, phase="development")) == list(dates[:4])
    assert list(_phase_schedule(plan, phase="locked_test")) == [dates[4]]
    with pytest.raises(ValueError, match="phase"):
        _phase_schedule(plan, phase="unknown")


def _daily():
    dates = pd.date_range("2025-01-02", periods=4)
    return pd.DataFrame(
        [
            {
                "phase": "locked_test",
                "evaluation_scope": "common",
                "fold": 3,
                "as_of_date": date,
                "model": model,
                "rank_ic": value,
            }
            for date, gap in zip(dates, [0.02, 0.01, 0.03, 0.04], strict=True)
            for model, value in [("ridge", gap + 0.01), ("baseline", 0.01)]
        ]
    )


def test_paired_complete_parity_and_insufficient_lag_support():
    from quant_metric_research.statistics import newey_west_mean_tstat

    dates = pd.date_range("2025-01-02", periods=4)
    mean, inference = _paired_locked_rank_ic_improvement(
        _daily(),
        model="ridge",
        baseline="baseline",
        hac_lags=1,
        expected_locked_dates=dates,
    )
    assert mean == pytest.approx(0.025)
    assert inference.status == "ok"
    assert (inference.t_stat, inference.p_value) == pytest.approx(
        newey_west_mean_tstat([0.02, 0.01, 0.03, 0.04], 1)
    )
    _, unsupported = _paired_locked_rank_ic_improvement(
        _daily(),
        model="ridge",
        baseline="baseline",
        hac_lags=4,
        expected_locked_dates=dates,
    )
    assert unsupported.status == "insufficient_lag_support"
    assert unsupported.p_value is None


@pytest.mark.parametrize("mode", ["empty", "missing_baseline"])
def test_paired_empty_values_keep_declared_schedule(mode):
    daily = _daily()
    selected = (
        daily.iloc[:0] if mode == "empty" else daily.loc[daily["model"].eq("ridge")]
    )
    mean, inference = _paired_locked_rank_ic_improvement(
        selected,
        model="ridge",
        baseline="baseline",
        hac_lags=1,
        expected_locked_dates=pd.date_range("2025-01-02", periods=4),
    )
    assert pd.isna(mean)
    assert inference.status == "missing_scheduled_values"
    assert inference.observed_count == 0
    assert inference.scheduled_count == 4


def test_paired_duplicate_date_across_folds_is_not_an_independent_observation():
    daily = _daily()
    duplicate = pd.concat([daily, daily.iloc[:2].assign(fold=4)], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        _paired_locked_rank_ic_improvement(
            duplicate,
            model="ridge",
            baseline="baseline",
            hac_lags=1,
            expected_locked_dates=pd.date_range("2025-01-02", periods=4),
        )


@pytest.mark.parametrize(
    "bad", [True, pd.Timestamp("2025-01-01"), complex(1, 2), "nan", float("inf")]
)
@pytest.mark.parametrize("with_schedule", [False, True])
def test_paired_validates_original_scalar_before_subtracting(bad, with_schedule):
    daily = _daily()
    values = daily["rank_ic"].astype(object)
    values.iloc[0] = bad
    with pytest.raises(ValueError, match="numeric"):
        _paired_locked_rank_ic_improvement(
            daily.assign(rank_ic=values),
            model="ridge",
            baseline="baseline",
            hac_lags=1,
            expected_locked_dates=pd.date_range("2025-01-02", periods=4)
            if with_schedule
            else None,
        )


@pytest.mark.parametrize("bad", [0, "2025-01-02 00:00:01", "2025-01-02T00:00:00Z"])
def test_paired_rejects_invalid_daily_dates_before_pivot(bad):
    daily = _daily()
    dates = daily["as_of_date"].astype(object)
    dates.iloc[0] = bad
    with pytest.raises(ValueError, match="dates"):
        _paired_locked_rank_ic_improvement(
            daily.assign(as_of_date=dates),
            model="ridge",
            baseline="baseline",
            hac_lags=1,
            expected_locked_dates=pd.date_range("2025-01-02", periods=4),
        )
