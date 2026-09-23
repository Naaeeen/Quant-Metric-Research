from dataclasses import replace

import pandas as pd
import pytest

from quant_metric_research.portfolio import LongOnlyConfig, evaluate_long_only
from quant_metric_research.portfolio_accounting import SideCosts, TradingCosts


def configuration(dates, **kwargs):
    return LongOnlyConfig(
        valuation_start=dates[2],
        valuation_end=dates[5],
        decision_dates=(dates[2],),
        benchmark_symbol="MARKET",
        top_k=1,
        **kwargs,
    )


def evaluate(case, *, config=None, **kwargs):
    dates, calendar, run, prices = case
    return evaluate_long_only(
        run,
        prices,
        model="ridge",
        calendar=calendar,
        config=config or configuration(dates),
        **kwargs,
    )


def test_entry_uses_next_session_not_decision_close(ledger_case):
    dates, calendar, run, prices = ledger_case
    for date, price in zip(dates[2:6], (100.0, 110.0, 121.0, 121.0), strict=True):
        prices.loc[prices.date.eq(date) & prices.symbol.eq("A"), "adjusted_close"] = (
            price
        )
    result = evaluate(ledger_case, config=configuration(dates, initial_cash=110.0))
    assert result.status == "completed"
    assert result.trades.execution_date.tolist() == [dates[3], dates[5]]
    assert result.trades.units.tolist() == pytest.approx([1.0, 1.0])
    assert result.summary["total_net_return"] == pytest.approx(0.1)
    assert result.cash == pytest.approx(121.0)
    assert result.unresolved_holdings == ()


def test_entry_and_exit_costs_reconcile_from_cash(ledger_case):
    dates = ledger_case[0]
    costs = TradingCosts(buy=SideCosts(commission=0.01), sell=SideCosts(spread=0.02))
    result = evaluate(
        ledger_case, config=configuration(dates, initial_cash=1000.0, costs=costs)
    )
    assert result.cash == pytest.approx(1000 / 1.01 * 0.98)
    assert result.trades.side.tolist() == ["buy", "sell"]
    for row in result.ledger.itertuples():
        assert row.nav == pytest.approx(row.pretrade_nav - row.total_cost)
        assert row.net_return == pytest.approx(
            row.gross_return - row.total_cost / row.previous_nav
        )
        assert row.cash >= 0


def test_missing_purchase_keeps_slot_cash_without_replacement(ledger_case):
    dates, calendar, run, prices = ledger_case
    prices = prices.loc[~(prices.date.eq(dates[3]) & prices.symbol.eq("A"))]
    result = evaluate((dates, calendar, run, prices))
    assert result.status == "completed"
    assert result.trades.empty
    assert result.rejections.symbol.tolist() == ["A"]
    assert result.cash == 100_000.0
    assert (
        result.targets.loc[result.targets.symbol.eq("A"), "target_weight"].iloc[0]
        == 1.0
    )


@pytest.mark.parametrize("blocked_index", [4, 5])
def test_missing_held_mark_preserves_incomplete_account(ledger_case, blocked_index):
    dates, calendar, run, prices = ledger_case
    prices = prices.loc[~(prices.date.eq(dates[blocked_index]) & prices.symbol.eq("A"))]
    result = evaluate((dates, calendar, run, prices))
    assert result.status == "blocked_missing_valuation"
    assert result.blocked_date == dates[blocked_index]
    assert result.last_valuation_date == dates[blocked_index - 1]
    assert result.unresolved_holdings == (("A", 1000.0),)
    assert result.blocked_symbols == ("A",)
    assert result.summary["total_net_return"] is None
    assert result.summary["terminal_nav"] is None
    assert result.ledger.date.max() == dates[blocked_index - 1]


def test_multiple_pending_decisions_keep_their_calendar_lag(ledger_case):
    dates, calendar, run, prices = ledger_case
    predictions = run.predictions.copy()
    predictions.loc[
        predictions.as_of_date.eq(dates[3]) & predictions.symbol.eq("B"), "score"
    ] = 5.0
    run = replace(run, predictions=predictions)
    config = replace(
        configuration(dates),
        decision_dates=(dates[2], dates[3]),
        execution_lag_sessions=2,
        valuation_end=dates[6],
    )
    result = evaluate((dates, calendar, run, prices), config=config)
    trades = result.trades
    assert set(trades.loc[trades.execution_date.eq(dates[4]), "symbol"]) == {"A"}
    assert trades.loc[trades.execution_date.eq(dates[5]), "side"].tolist() == [
        "sell",
        "buy",
    ]
    assert trades.loc[trades.execution_date.eq(dates[5]), "symbol"].tolist() == [
        "A",
        "B",
    ]


def test_missing_held_mark_blocks_all_pending_trades(ledger_case):
    dates, calendar, run, prices = ledger_case
    predictions = run.predictions.copy()
    predictions.loc[
        predictions.as_of_date.eq(dates[3]) & predictions.symbol.eq("B"), "score"
    ] = 5.0
    prices = prices.loc[~(prices.date.eq(dates[4]) & prices.symbol.eq("A"))]
    config = replace(configuration(dates), decision_dates=(dates[2], dates[3]))
    result = evaluate(
        (dates, calendar, replace(run, predictions=predictions), prices), config=config
    )
    assert result.status == "blocked_missing_valuation"
    assert result.trades.execution_date.tolist() == [dates[3]]
    assert result.unresolved_holdings == (("A", 1000.0),)


def test_future_prices_and_decisions_leave_earlier_account_unchanged(ledger_case):
    dates, calendar, run, prices = ledger_case
    config = replace(configuration(dates), decision_dates=(dates[2], dates[3]))
    expected = evaluate(ledger_case, config=config)
    predictions = run.predictions.copy()
    predictions.loc[
        predictions.as_of_date.eq(dates[3]) & predictions.symbol.eq("B"), "score"
    ] = 5.0
    prices = prices.copy()
    prices.loc[prices.date.ge(dates[4]), "adjusted_close"] *= 2.0
    actual = evaluate(
        (dates, calendar, replace(run, predictions=predictions), prices), config=config
    )
    pd.testing.assert_frame_equal(actual.ledger.iloc[:2], expected.ledger.iloc[:2])
    pd.testing.assert_frame_equal(actual.targets.iloc[:3], expected.targets.iloc[:3])


def test_labels_and_later_prices_do_not_change_consumed_identity(ledger_case):
    dates, calendar, run, prices = ledger_case
    expected = evaluate(ledger_case)
    run = replace(
        run, predictions=run.predictions.drop(columns=["target", "realized_return"])
    )
    prices = prices.copy()
    prices.loc[prices.date.gt(dates[5]), "adjusted_close"] = float("nan")
    actual = evaluate((dates, calendar, run, prices))
    pd.testing.assert_frame_equal(actual.ledger, expected.ledger)
    pd.testing.assert_frame_equal(actual.trades, expected.trades)
    assert actual.manifest == expected.manifest


def test_one_series_rescaling_changes_units_not_wealth(ledger_case):
    dates, calendar, run, prices = ledger_case
    expected = evaluate(ledger_case)
    prices = prices.copy()
    prices.loc[prices.symbol.eq("A"), "adjusted_close"] *= 7.0
    actual = evaluate((dates, calendar, run, prices))
    assert actual.ledger.nav.tolist() == pytest.approx(expected.ledger.nav.tolist())
    assert actual.trades.notional.tolist() == pytest.approx(
        expected.trades.notional.tolist()
    )


def test_all_missing_scores_hold_cash_and_report_coverage(ledger_case):
    dates, calendar, run, prices = ledger_case
    predictions = run.predictions.copy()
    predictions["score"] = float("nan")
    predictions["feature_count"] = 0
    predictions["zero_observed_features"] = True
    result = evaluate((dates, calendar, replace(run, predictions=predictions), prices))
    assert result.status == "completed"
    assert result.trades.empty
    assert result.coverage.scored_count.tolist() == [0]
    assert result.coverage.planned_cash_weight.tolist() == [1.0]


def test_caller_frames_unchanged(ledger_case):
    before_scores = ledger_case[2].predictions.copy(deep=True)
    before_prices = ledger_case[3].copy(deep=True)
    evaluate(ledger_case)
    pd.testing.assert_frame_equal(ledger_case[2].predictions, before_scores)
    pd.testing.assert_frame_equal(ledger_case[3], before_prices)


def test_unrepresentable_cumulative_return_is_not_reported_as_infinity(ledger_case):
    dates, calendar, run, prices = ledger_case
    prices = prices.copy()
    for date, price in zip(dates[3:6], (1e-200, 1.0, 1e200), strict=True):
        prices.loc[prices.date.eq(date) & prices.symbol.eq("A"), "adjusted_close"] = (
            price
        )
    with pytest.raises(ValueError, match="summary|return"):
        evaluate(
            (dates, calendar, run, prices),
            config=configuration(dates, initial_cash=1e-200),
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"top_k": True},
        {"top_k": 0},
        {"execution_lag_sessions": 0},
        {"initial_cash": True},
        {"initial_cash": float("inf")},
        {"initial_cash": -1.0},
        {"benchmark_symbol": ""},
        {"costs": None},
    ],
)
def test_invalid_configuration(ledger_case, changes):
    with pytest.raises(ValueError):
        replace(configuration(ledger_case[0]), **changes)
