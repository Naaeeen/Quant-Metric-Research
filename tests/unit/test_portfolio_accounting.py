"""Hand-worked self-financing accounting, independent of dates and labels."""

from dataclasses import FrozenInstanceError
from math import fsum

import numpy as np
import pytest

from quant_metric_research.contracts import DataContractError
from quant_metric_research.portfolio_accounting import (
    SideCosts,
    TradingCosts,
    rebalance,
    top_k_weights,
)


def execute(**changes):
    return rebalance(
        **{
            "holdings": {},
            "prices": {"A": 1.0},
            "cash": 100.0,
            "target_weights": {"A": 1.0},
            "costs": TradingCosts(),
            **changes,
        }
    )


def test_entry_cost_is_funded_from_cash():
    result = execute(costs=TradingCosts(buy=SideCosts(commission=0.01)))
    assert dict(result.holdings)["A"] == pytest.approx(100 / 1.01)
    assert result.cash == 0
    assert result.nav == pytest.approx(100 / 1.01)
    assert result.trades[0].commission == pytest.approx(100 - result.nav)


def test_replacement_sells_before_buying_and_funds_both_sides():
    result = execute(
        holdings={"Z": 100},
        cash=0,
        prices={"Z": 1, "A": 1},
        costs=TradingCosts(SideCosts(0.01), SideCosts(0.01)),
    )
    assert [trade.side for trade in result.trades] == ["sell", "buy"]
    assert result.nav == pytest.approx(99 / 1.01)
    assert dict(result.holdings) == pytest.approx({"A": 99 / 1.01})
    assert result.cash == 0


def test_exit_charges_cost_and_removes_holding():
    result = execute(
        holdings={"A": 100},
        cash=0,
        target_weights={},
        costs=TradingCosts(sell=SideCosts(0.01)),
    )
    assert result.holdings == ()
    assert result.cash == pytest.approx(99)
    assert result.nav == pytest.approx(99)


def test_drift_changes_trades_even_when_target_weights_unchanged():
    result = execute(
        holdings={"A": 50, "B": 50},
        cash=0,
        prices={"A": 1.2, "B": 1},
        target_weights={"A": 0.5, "B": 0.5},
    )
    assert result.pretrade_nav == 110
    assert result.nav == 110
    assert [(t.symbol, t.side) for t in result.trades] == [("A", "sell"), ("B", "buy")]
    assert [t.notional for t in result.trades] == pytest.approx([5, 5])


def test_boundary_ties_and_missing_scores_keep_fixed_k_denominator():
    assert top_k_weights({"A": 2, "B": 1, "C": 1}, top_k=2) == {
        "A": 0.5,
        "B": 0.25,
        "C": 0.25,
    }
    assert top_k_weights({"A": 2, "B": None, "C": float("nan")}, top_k=2) == {"A": 0.5}
    assert top_k_weights({}, top_k=2) == {}
    assert top_k_weights({"A": -2, "B": -1}, top_k=1) == {"B": 1}


def test_missing_new_purchase_leaves_cash_without_renormalizing():
    result = execute(target_weights={"A": 0.5, "B": 0.5})
    assert dict(result.holdings) == {"A": 50}
    assert result.cash == 50
    assert [(r.symbol, r.target_weight) for r in result.rejected] == [("B", 0.5)]


def test_missing_held_mark_is_error_even_when_exiting():
    with pytest.raises(DataContractError, match="held"):
        execute(holdings={"B": 1}, target_weights={})


def test_component_costs_cash_and_nav_reconcile():
    result = execute(
        holdings={"B": 2},
        cash=20,
        prices={"A": 7, "B": 40},
        target_weights={"A": 0.6, "B": 0.2},
        costs=TradingCosts(
            SideCosts(0.001, 0.002, 0.003), SideCosts(0.004, 0.005, 0.006)
        ),
    )
    costs = fsum(t.commission + t.spread + t.slippage for t in result.trades)
    assert result.pretrade_nav - result.nav == pytest.approx(costs)
    assert result.cash == pytest.approx(0.2 * result.nav)
    assert result.cash + fsum(
        q * {"A": 7, "B": 40}[s] for s, q in result.holdings
    ) == pytest.approx(result.nav)
    for trade in result.trades:
        assert trade.units > 0
        assert trade.notional == pytest.approx(
            trade.units * {"A": 7, "B": 40}[trade.symbol]
        )
        rate = 0.001 if trade.side == "buy" else 0.004
        assert trade.commission == pytest.approx(trade.notional * rate)


def test_permutation_and_price_unit_rescaling_preserve_economics():
    original = execute(
        holdings={"B": 3, "A": 2},
        prices={"A": 10, "B": 20},
        cash=20,
        target_weights={"B": 0.6, "A": 0.4},
        costs=TradingCosts(SideCosts(0.01)),
    )
    scaled = execute(
        holdings={"A": 0.2, "B": 3},
        prices={"B": 20, "A": 100},
        cash=20,
        target_weights={"A": 0.4, "B": 0.6},
        costs=TradingCosts(SideCosts(0.01)),
    )
    assert original.nav == pytest.approx(scaled.nav)
    assert original.cash == pytest.approx(scaled.cash)
    assert [t.notional for t in original.trades] == pytest.approx(
        [t.notional for t in scaled.trades]
    )
    scores = {"Z": 2, "A": 1, "Q": 1}
    assert top_k_weights(scores, top_k=2) == top_k_weights(
        dict(reversed(list(scores.items()))), top_k=2
    )


@pytest.mark.parametrize("scale", [1e-200, 1e-12, 1, 1e200])
def test_cash_scale_does_not_change_proportional_result(scale):
    result = execute(cash=scale, costs=TradingCosts(SideCosts(0.01)))
    assert result.nav / scale == pytest.approx(1 / 1.01, rel=1e-13)
    assert result.cash == 0


def test_zero_capital_noop_and_already_balanced_holdings():
    assert execute(cash=0).trades == ()
    assert execute(cash=0).nav == 0
    result = execute(
        holdings={"A": 100, "B": 0},
        cash=0,
        costs=TradingCosts(SideCosts(0.01), SideCosts(0.01)),
    )
    assert result.trades == ()
    assert result.holdings == (("A", 100),)


def test_near_total_sell_cost_and_tiny_cash_allocation():
    result = execute(
        holdings={"A": 100},
        cash=0,
        target_weights={},
        costs=TradingCosts(sell=SideCosts(1 - 1e-12)),
    )
    assert result.nav / 100 == pytest.approx(1 - (1 - 1e-12), rel=1e-4)
    result = execute(target_weights={"A": 1 - 1e-12})
    assert result.cash > 0
    assert result.cash == pytest.approx(100 * (1 - (1 - 1e-12)))


def test_inputs_are_preserved_and_results_are_frozen():
    holdings, prices, weights = {"A": 1}, {"A": 2}, {"A": 0.5}
    result = execute(holdings=holdings, prices=prices, target_weights=weights)
    assert (holdings, prices, weights) == ({"A": 1}, {"A": 2}, {"A": 0.5})
    with pytest.raises(FrozenInstanceError):
        result.cash = 0
    with pytest.raises(FrozenInstanceError):
        SideCosts().commission = 0.1


@pytest.mark.parametrize(
    "bad",
    [
        True,
        np.bool_(True),
        "0.1",
        None,
        complex(1),
        float("nan"),
        float("inf"),
        -0.1,
        1,
    ],
)
def test_invalid_cost_rate(bad):
    with pytest.raises(DataContractError):
        SideCosts(commission=bad)


def test_cost_sum_and_cost_container_validation():
    with pytest.raises(DataContractError):
        SideCosts(0.5, 0.25, 0.25)
    with pytest.raises(DataContractError):
        TradingCosts(buy={})


@pytest.mark.parametrize(
    "bad", [True, np.bool_(False), "1", complex(1), float("inf"), -float("inf")]
)
def test_invalid_scores(bad):
    with pytest.raises(DataContractError):
        top_k_weights({"A": bad}, top_k=1)


@pytest.mark.parametrize("bad", [True, 0, -1, 1.2, "2"])
def test_invalid_top_k(bad):
    with pytest.raises(DataContractError):
        top_k_weights({"A": 1}, top_k=bad)


@pytest.mark.parametrize(
    "changes",
    [
        {"cash": -1},
        {"cash": float("inf")},
        {"cash": True},
        {"holdings": {"A": -1}},
        {"holdings": {"A": "1"}},
        {"prices": {"A": 0}},
        {"prices": {"A": None}},
        {"prices": {"A": float("nan")}},
        {"target_weights": {"A": -0.1}},
        {"target_weights": {"A": 1.01}},
        {"target_weights": {"A": 0.6, "B": 0.5}},
        {"target_weights": {"A": True}},
        {"costs": None},
        {"holdings": []},
        {"prices": {"": 1}},
        {"target_weights": {1: 0.5}},
        {"prices": {" A": 1}},
        {"holdings": {"A": 1e300}, "prices": {"A": 1e300}},
    ],
)
def test_invalid_accounting_inputs(changes):
    with pytest.raises(DataContractError):
        execute(**changes)


def test_invalid_score_map_and_symbol():
    for scores in ([], {"": 1}, {1: 1}):
        with pytest.raises(DataContractError):
            top_k_weights(scores, top_k=1)


def test_oversized_score_raises_contract_error():
    with pytest.raises(DataContractError):
        top_k_weights({"A": 10**400}, top_k=1)


def test_unrepresentable_slot_weight_is_rejected():
    with pytest.raises(DataContractError):
        top_k_weights({"A": 1}, top_k=10**400)


def test_subnormal_capital_cannot_silently_erase_costs():
    with pytest.raises(DataContractError, match="representable"):
        execute(cash=5e-324, costs=TradingCosts(buy=SideCosts(0.5)))


@pytest.mark.parametrize(
    "changes",
    [
        {"cash": 1e308, "holdings": {"A": 1e308}},
        {"prices": {"A": 5e-324}},
        {"holdings": {"A": 5e-324}, "prices": {"A": 0.5}, "target_weights": {}},
    ],
)
def test_unrepresentable_accounting_ranges_are_rejected(changes):
    with pytest.raises(DataContractError):
        execute(**changes)


def test_many_equal_weight_slots_allow_only_float_roundoff():
    scores = {str(index): 1.0 for index in range(49)}
    weights = top_k_weights(scores, top_k=49)
    result = execute(prices={symbol: 1 for symbol in scores}, target_weights=weights)
    assert result.nav == 100
    assert result.cash >= 0
    assert fsum(units for _, units in result.holdings) == pytest.approx(100)
