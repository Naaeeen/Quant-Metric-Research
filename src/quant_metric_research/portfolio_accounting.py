"""Pure, self-financing accounting for fractional adjusted-price units.

Costs are one-way mark-notional debits, not displaced execution prices. A caller
using a symmetric quoted spread supplies its half-spread on each trading side.
No dates, dividends, actual-share rounding, borrowing or external cash flows are
inferred here. Caller-owned mappings are never changed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import groupby
from math import fsum, isclose, isfinite
from numbers import Integral, Real

from .contracts import DataContractError


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise DataContractError(f"{name} must be a finite real number.")
    try:
        converted = float(value)
    except OverflowError as error:
        raise DataContractError(f"{name} exceeds finite numeric range.") from error
    if not isfinite(converted):
        raise DataContractError(f"{name} must be a finite real number.")
    return converted


def _sum(values, *, name: str) -> float:
    try:
        value = fsum(values)
    except OverflowError as error:
        raise DataContractError(f"{name} exceeds finite numeric range.") from error
    if not isfinite(value):
        raise DataContractError(f"{name} exceeds finite numeric range.")
    return value


def _symbols(values: Mapping, *, name: str) -> None:
    if not isinstance(values, Mapping):
        raise DataContractError(f"{name} must be a symbol mapping.")
    if any(not isinstance(s, str) or not s or s.strip() != s for s in values):
        raise DataContractError(f"{name} requires nonempty, normalized string symbols.")


def _nonnegative_map(values: Mapping, *, name: str) -> dict[str, float]:
    _symbols(values, name=name)
    result = {s: _number(v, name=name) for s, v in values.items()}
    if any(value < 0 for value in result.values()):
        raise DataContractError(f"{name} must be nonnegative.")
    return result


@dataclass(frozen=True)
class SideCosts:
    """Finite nonnegative fractional rates whose total is strictly below one."""

    commission: float = 0.0
    spread: float = 0.0
    slippage: float = 0.0

    def __post_init__(self) -> None:
        for name in ("commission", "spread", "slippage"):
            value = _number(getattr(self, name), name=name)
            if value < 0:
                raise DataContractError(f"{name} must be nonnegative.")
            object.__setattr__(self, name, value)
        if self.total >= 1:
            raise DataContractError("Side cost rates must sum to less than one.")

    @property
    def total(self) -> float:
        """Combined one-way debit as a fraction of traded mark notional."""
        return _sum((self.commission, self.spread, self.slippage), name="Side costs")


@dataclass(frozen=True)
class TradingCosts:
    """Independent purchase and sale cost assumptions."""

    buy: SideCosts = SideCosts()
    sell: SideCosts = SideCosts()

    def __post_init__(self) -> None:
        if not isinstance(self.buy, SideCosts) or not isinstance(self.sell, SideCosts):
            raise DataContractError("Trading costs require SideCosts for buy and sell.")


@dataclass(frozen=True)
class Trade:
    """A positive unit change and its separate cash debits."""

    symbol: str
    side: str
    units: float
    notional: float
    commission: float
    spread: float
    slippage: float


@dataclass(frozen=True)
class RejectedPurchase:
    """An unavailable new purchase whose target allocation stays in cash."""

    symbol: str
    target_weight: float


@dataclass(frozen=True)
class RebalanceResult:
    """Immutable post-trade account; trades are ordered sells before buys."""

    holdings: tuple[tuple[str, float], ...]
    cash: float
    pretrade_nav: float
    nav: float
    trades: tuple[Trade, ...]
    rejected: tuple[RejectedPurchase, ...]


def top_k_weights(
    scores: Mapping[str, float | None], *, top_k: int
) -> dict[str, float]:
    """Allocate fixed k slots, dividing boundary ties without identifier bias.

    None/NaN scores are absent; other malformed/nonfinite scores are errors.
    Fewer than k finite scores leave unused slots as cash. Only positive target
    weights are returned. Exact score ties can produce more than k holdings.
    """
    _symbols(scores, name="scores")
    if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k < 1:
        raise DataContractError("top_k must be a positive integer.")
    if 1 / int(top_k) == 0:
        raise DataContractError("top_k slot weight is not representable.")
    finite = {}
    for symbol, value in scores.items():
        if value is None or (isinstance(value, Real) and value != value):
            continue
        finite[symbol] = _number(value, name="score")
    ranked = sorted(finite, key=lambda s: (-finite[s], s))
    groups = groupby(ranked, key=finite.__getitem__)
    remaining = int(top_k)
    weights = {}
    for _, group in groups:
        symbols = tuple(group)
        slots = min(remaining, len(symbols))
        weights.update({symbol: (slots / top_k) / len(symbols) for symbol in symbols})
        remaining -= slots
        if remaining == 0:
            break
    return weights


def _nav_fraction(
    marked: Mapping[str, float], weights: Mapping[str, float], costs: TradingCosts
) -> float:
    """Solve x + normalized trading costs = 1 on [0, 1]."""
    symbols = sorted(marked.keys() | weights.keys())

    def residual(fraction: float) -> float:
        changes = [weights.get(s, 0) * fraction - marked.get(s, 0) for s in symbols]
        return fsum(
            (
                fraction,
                -1.0,
                *(
                    change * costs.buy.total
                    if change > 0
                    else -change * costs.sell.total
                    for change in changes
                ),
            )
        )

    if residual(1.0) == 0:
        return 1.0
    lower, upper = 0.0, 1.0
    # 128 steps resolve even a near-total sale fee without dollar-scale cutoffs.
    for _ in range(128):
        middle = (lower + upper) / 2
        value = residual(middle)
        if value == 0:
            return middle
        if value > 0:
            upper = middle
        else:
            lower = middle
    return (lower + upper) / 2


def _positions_and_trades(holdings, prices, weights, nav, costs):
    positions = {
        symbol: weight * nav / prices[symbol]
        for symbol, weight in weights.items()
        if weight > 0 and nav > 0
    }
    if any(not isfinite(units) or units <= 0 for units in positions.values()):
        raise DataContractError("Target units exceed representable positive range.")
    trades = []
    for symbol in sorted(holdings.keys() | positions.keys()):
        change = positions.get(symbol, 0) - holdings.get(symbol, 0)
        if change == 0:
            continue
        side = "buy" if change > 0 else "sell"
        rates = getattr(costs, side)
        notional = abs(change) * prices[symbol]
        if not isfinite(notional) or notional <= 0:
            raise DataContractError(
                "Trade notional exceeds representable positive range."
            )
        trades.append(
            Trade(
                symbol,
                side,
                abs(change),
                notional,
                notional * rates.commission,
                notional * rates.spread,
                notional * rates.slippage,
            )
        )
    return tuple(sorted(positions.items())), tuple(
        sorted(trades, key=lambda t: (t.side == "buy", t.symbol))
    )


def _reconcile(result: RebalanceResult, prices, initial_cash) -> None:
    if result.pretrade_nav == 0:
        return
    scale = result.pretrade_nav
    fees = fsum((t.commission + t.spread + t.slippage) / scale for t in result.trades)
    marked = fsum(q * prices[s] / scale for s, q in result.holdings)
    cash_flow = fsum(
        (
            initial_cash / scale,
            *(
                (t.notional if t.side == "sell" else -t.notional) / scale
                for t in result.trades
            ),
            -fees,
        )
    )
    checks = (
        (result.nav / scale, 1 - fees),
        (result.nav / scale, result.cash / scale + marked),
        (result.cash / scale, cash_flow),
    )
    if any(not isclose(a, b, rel_tol=1e-12, abs_tol=1e-14) for a, b in checks):
        raise DataContractError(
            "Self-financing account failed numerical reconciliation."
        )


def rebalance(
    *,
    holdings: Mapping[str, float],
    prices: Mapping[str, float],
    cash: float,
    target_weights: Mapping[str, float],
    costs: TradingCosts,
) -> RebalanceResult:
    """Rebalance nonnegative units to post-cost NAV weights, without borrowing.

    Missing marks for positive holdings raise DataContractError. Missing marks
    for new purchases reject only those allocations; they remain cash. Invalid
    supplied marks, units, cash, weights or costs also raise DataContractError.
    Weight sums within 1e-14 above one are accepted as floating-point roundoff;
    no economically material target weight is renormalized or cash injected.
    """
    if not isinstance(costs, TradingCosts):
        raise DataContractError("costs must be TradingCosts.")
    held = {
        s: q for s, q in _nonnegative_map(holdings, name="holdings").items() if q > 0
    }
    marks = _nonnegative_map(prices, name="prices")
    if any(p <= 0 for p in marks.values()):
        raise DataContractError("prices must be strictly positive.")
    weights = _nonnegative_map(target_weights, name="target_weights")
    weight_sum = _sum(weights.values(), name="target weights")
    if weight_sum > 1 + 1e-14:
        raise DataContractError("Target weights must sum to at most one.")
    cash = _number(cash, name="cash")
    if cash < 0:
        raise DataContractError("cash must be nonnegative.")
    if held.keys() - marks.keys():
        raise DataContractError("Missing price for held symbol(s).")
    values = {s: q * marks[s] for s, q in held.items()}
    pretrade_nav = _sum((cash, *values.values()), name="Pretrade NAV")
    rejected = tuple(
        RejectedPurchase(s, w)
        for s, w in sorted(weights.items())
        if w > 0 and s not in marks
    )
    executable = {s: w for s, w in weights.items() if s in marks and w > 0}
    fraction = (
        _nav_fraction(
            {s: v / pretrade_nav for s, v in values.items()}, executable, costs
        )
        if pretrade_nav
        else 1.0
    )
    nav = pretrade_nav * fraction
    if pretrade_nav and not isclose(nav / pretrade_nav, fraction, rel_tol=1e-12):
        raise DataContractError("Post-cost NAV is not accurately representable.")
    positions, trades = _positions_and_trades(held, marks, executable, nav, costs)
    cash_weight = max(0.0, 1 - fsum(executable.values()))
    result = RebalanceResult(
        positions, cash_weight * nav, pretrade_nav, nav, trades, rejected
    )
    _reconcile(result, marks, cash)
    return result
