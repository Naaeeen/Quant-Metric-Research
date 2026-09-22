from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_metric_research._portfolio_inputs import prepare_portfolio_inputs
from quant_metric_research.benchmark import BenchmarkRun
from quant_metric_research.benchmark_config import BenchmarkConfig, NestedSplitConfig
from quant_metric_research.session_calendar import ExpectedSessionCalendar


@pytest.fixture
def inputs():
    dates = pd.date_range("2020-01-01", periods=10)
    benchmark = BenchmarkConfig(
        ("x",), NestedSplitConfig(1, 1, 2, 1, 1, 1, 1), hac_lags=0
    )
    manifest = dict(
        artifact_schema_version="6",
        execution_mode="development",
        configuration=benchmark.to_mapping(),
        development_start="2020-01-01",
        locked_test_start="2020-01-09",
        locked_test_end="2020-01-09",
        locked_label_end_max="2020-01-10",
        dataset_versions=["synthetic"],
        fingerprint_scope="source_code+configuration+validated_panel_contract+execution_mode",
        evaluation_schedule=dict(
            definition_version="1",
            source="validated_panel_within_planned_phase_bounds",
            lag_unit="scheduled_observations",
            dates_by_phase={"development": ["2020-01-04", "2020-01-05"]},
        ),
    )
    for key in (
        "source_fingerprint",
        "panel_fingerprint",
        "run_fingerprint",
        "model_input_fingerprint",
    ):
        manifest[key] = "a" * 64
    for key in (
        "package_version",
        "implementation_version",
        "python_version",
        "numpy_version",
        "pandas_version",
        "scipy_version",
        "scikit_learn_version",
    ):
        manifest[key] = "synthetic"
    predictions = pd.DataFrame(
        [
            dict(
                phase="development",
                fold=1,
                as_of_date=day,
                symbol=symbol,
                row_id=f"{day}-{symbol}",
                model="ridge",
                score=1.0,
                fit_end_date="2020-01-02",
                train_label_end_max="2020-01-03",
                evaluation_start="2020-01-04",
                feature_count=1,
                selected_feature_count=1,
                zero_observed_features=False,
                target=9.0,
                realized_return=8.0,
            )
            for day in ("2020-01-04", "2020-01-05")
            for symbol in ("AAA", "BBB")
        ]
    )
    assignments = pd.DataFrame(
        [
            dict(
                phase="development",
                fold=1,
                split_id="development_fold_0",
                row_id=row_id,
                role="evaluation",
                exclusion_reason=None,
                train_end_date="2020-01-02",
                train_label_end_max="2020-01-03",
                evaluation_start="2020-01-04",
                evaluation_end="2020-01-05",
            )
            for row_id in predictions.row_id
        ]
    )
    run = BenchmarkRun(
        {"claim_scope": "development_only"},
        assignments,
        predictions,
        *[pd.DataFrame() for _ in range(5)],
        {
            "acceptance_status": "not_evaluated",
            "lockbox_evaluated_once_in_this_run": False,
        },
        manifest,
    )
    prices = pd.DataFrame(
        [
            dict(date=date, symbol=symbol, adjusted_close=100.0)
            for date in dates
            for symbol in ("AAA", "SPY")
        ]
    )
    calendar = ExpectedSessionCalendar(
        tuple(dates), dates[0], dates[-1], "synthetic", "1"
    )
    config = SimpleNamespace(
        valuation_start=dates[3],
        valuation_end=dates[6],
        decision_dates=tuple(dates[3:5]),
        execution_lag_sessions=1,
        benchmark_symbol="SPY",
        top_k=1,
        initial_cash=100.0,
    )
    return run, prices, calendar, config


def prepare(inputs):
    run, prices, calendar, config = inputs
    return prepare_portfolio_inputs(
        run, prices, model="ridge", calendar=calendar, config=config
    )


def test_score_only_projection_immutability_and_unpriced_universe(inputs):
    run, prices, _, _ = inputs
    original = run.predictions.copy(deep=True)
    result = prepare(inputs)
    assert set(result.scores.symbol) == {"AAA", "BBB"}
    assert "target" not in result.scores and "realized_return" not in result.scores
    assert len(result.sessions) == 4 and len(result.execution_dates) == 2
    assert result.prices.date.min() == pd.Timestamp("2020-01-04")
    result.scores.loc[0, "score"] = -5
    result.prices.loc[0, "adjusted_close"] = 2
    pd.testing.assert_frame_equal(run.predictions, original)
    assert prices.adjusted_close.eq(100).all()
    with pytest.raises(TypeError):
        result.identity["x"] = 1


def test_outcomes_and_future_prices_do_not_affect_identity(inputs):
    first = prepare(inputs)
    run, prices, calendar, config = inputs
    changed = replace(
        run, predictions=run.predictions.drop(columns=["target", "realized_return"])
    )
    later = prices.copy().astype({"adjusted_close": object})
    later.loc[later.date.ge("2020-01-09"), "adjusted_close"] = "DO NOT PARSE"
    second = prepare((changed, later, calendar, config))
    assert first.identity == second.identity


@pytest.mark.parametrize(
    "field,value",
    [
        ("fit_end_date", "2020-01-03"),
        ("phase", "locked_test"),
        ("score", np.inf),
        ("score", True),
        ("score", "bad"),
        ("feature_count", 2),
        ("selected_feature_count", 0),
        ("zero_observed_features", True),
        ("symbol", "aaa"),
    ],
)
def test_invalid_prediction_evidence(inputs, field, value):
    inputs[0].predictions[field] = inputs[0].predictions[field].astype(object)
    inputs[0].predictions.loc[0, field] = value
    with pytest.raises(ValueError):
        prepare(inputs)


def test_duplicate_scores(inputs):
    run, prices, calendar, config = inputs
    run = replace(
        run, predictions=pd.concat([run.predictions, run.predictions.iloc[[0]]])
    )
    with pytest.raises(ValueError):
        prepare((run, prices, calendar, config))


def test_reject_unselected_final_phase(inputs):
    run, prices, calendar, config = inputs
    extra = run.predictions.iloc[[0]].assign(model="unused", phase="locked_test")
    run = replace(run, predictions=pd.concat([run.predictions, extra]))
    with pytest.raises(ValueError):
        prepare((run, prices, calendar, config))


def test_missing_scores_or_benchmark_session(inputs):
    run, prices, calendar, config = inputs
    with pytest.raises(ValueError):
        prepare(
            (
                replace(run, predictions=run.predictions.iloc[:2]),
                prices,
                calendar,
                config,
            )
        )
    with pytest.raises(ValueError):
        prepare(
            (
                run,
                prices.drop(
                    prices[
                        (prices.symbol == "SPY") & (prices.date == config.valuation_end)
                    ].index
                ),
                calendar,
                config,
            )
        )


@pytest.mark.parametrize(
    "change", ["final_end", "last_execution", "missing_decision", "off_calendar"]
)
def test_invalid_declared_window(inputs, change):
    run, prices, calendar, config = inputs
    if change == "final_end":
        config.valuation_end = pd.Timestamp("2020-01-09")
    if change == "last_execution":
        config.valuation_end = pd.Timestamp("2020-01-06")
    if change == "missing_decision":
        config.decision_dates = (pd.Timestamp("2020-01-06"),)
    if change == "off_calendar":
        calendar = ExpectedSessionCalendar(
            tuple(x for x in calendar.sessions if x != pd.Timestamp("2020-01-06")),
            calendar.coverage_start,
            calendar.coverage_end,
            "synthetic",
            "2",
        )
    with pytest.raises(ValueError):
        prepare((run, prices, calendar, config))


def test_null_scores_and_outcome_exclusions_survive(inputs):
    run = inputs[0]
    run.predictions["score"] = np.nan
    run.fold_assignments.loc[0, ["role", "exclusion_reason"]] = [
        "excluded",
        "missing_label",
    ]
    assert len(prepare(inputs).scores) == 4


@pytest.mark.parametrize(
    "column,value",
    [("train_label_end_max", "2020-01-04"), ("split_id", "bad"), ("fold", 2)],
)
def test_invalid_assignments(inputs, column, value):
    inputs[0].fold_assignments[column] = value
    with pytest.raises(ValueError):
        prepare(inputs)


@pytest.mark.parametrize("model", ["best_metric", "metric:x", "equal_weight_rank"])
def test_supports_saved_baseline_models(inputs, model):
    run, prices, calendar, config = inputs
    run.predictions["model"] = model
    result = prepare_portfolio_inputs(
        run, prices, model=model, calendar=calendar, config=config
    )
    assert result.scores.model.eq(model).all()


def test_binds_experiment_identifiers(inputs):
    inputs[0].manifest["experiment"] = {
        "kind": "development",
        "run_id": "saved-run",
        "study_id": "saved-study",
    }
    assert prepare(inputs).identity["experiment"] == inputs[0].manifest["experiment"]


@pytest.mark.parametrize("model", ["metric:unknown", "unknown", " ridge"])
def test_unknown_model_rejected(inputs, model):
    run, prices, calendar, config = inputs
    with pytest.raises(ValueError):
        prepare_portfolio_inputs(
            run, prices, model=model, calendar=calendar, config=config
        )


def test_score_and_window_price_changes_change_hashes(inputs):
    first = prepare(inputs)
    inputs[0].predictions.loc[0, "score"] = np.nextafter(1.0, 2.0)
    second = prepare(inputs)
    assert first.identity["score_fingerprint"] != second.identity["score_fingerprint"]
    inputs[1].loc[inputs[1].date.eq("2020-01-04"), "adjusted_close"] = 101.0
    assert (
        second.identity["price_fingerprint"]
        != prepare(inputs).identity["price_fingerprint"]
    )


def test_fold_schedule_count_must_match_configuration(inputs):
    inputs[0].manifest["configuration"]["split"]["outer_test_date_count"] = 99
    with pytest.raises(ValueError, match="count"):
        prepare(inputs)


@pytest.mark.parametrize("model", ["metric:x", "best_metric"])
@pytest.mark.parametrize("observed", [0, 1, 2])
def test_individual_baseline_cannot_claim_multiple_selected_features(
    inputs, model, observed
):
    run, prices, calendar, config = inputs
    run.manifest["configuration"]["feature_columns"] = ("x", "y")
    run.predictions["model"] = model
    run.predictions["selected_feature_count"] = 2
    run.predictions["feature_count"] = observed
    run.predictions["zero_observed_features"] = observed == 0
    if observed == 0:
        run.predictions["score"] = np.nan
    with pytest.raises(ValueError, match="feature"):
        prepare_portfolio_inputs(
            run, prices, model=model, calendar=calendar, config=config
        )


@pytest.mark.parametrize("model", ["metric:x", "best_metric"])
def test_individual_baseline_all_missing_feature_is_allowed(inputs, model):
    run, prices, calendar, config = inputs
    run.predictions["model"] = model
    run.predictions["feature_count"] = 0
    run.predictions["zero_observed_features"] = True
    run.predictions["score"] = np.nan
    result = prepare_portfolio_inputs(
        run, prices, model=model, calendar=calendar, config=config
    )
    assert result.scores.score.isna().all()
