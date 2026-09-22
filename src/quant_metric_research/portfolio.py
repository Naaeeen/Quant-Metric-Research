"""Development-only, cash-funded holdings accounting from saved ranking scores."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any

import pandas as pd

from ._comparison_contract import _freeze
from ._portfolio_inputs import prepare_portfolio_inputs
from ._version import __version__
from .benchmark import BenchmarkRun
from .benchmark_reporting import _source_fingerprint
from .contracts import DataContractError, _daily_date, validate_as_of_dates
from .portfolio_accounting import TradingCosts, rebalance, top_k_weights
from .session_calendar import ExpectedSessionCalendar


@dataclass(frozen=True)
class LongOnlyConfig:
    """An explicit after-close decision schedule and mandatory terminal exit.

    Costs are one-way fractions of mark-price traded notional. Positions use
    fractional adjusted-price units, not broker shares. Unused top-k slots stay
    in cash. The final valuation session liquidates every remaining holding.
    """

    valuation_start: pd.Timestamp
    valuation_end: pd.Timestamp
    decision_dates: tuple[pd.Timestamp, ...]
    benchmark_symbol: str
    top_k: int
    execution_lag_sessions: int = 1
    initial_cash: float = 100_000.0
    costs: TradingCosts = field(default_factory=TradingCosts)

    def __post_init__(self) -> None:
        for name in ("valuation_start", "valuation_end"):
            object.__setattr__(
                self, name, _daily_date(getattr(self, name), field=name, nullable=False)
            )
        dates = validate_as_of_dates(self.decision_dates)
        if tuple(sorted(dates)) != dates:
            raise DataContractError("decision_dates must be increasing.")
        object.__setattr__(self, "decision_dates", dates)
        if self.valuation_start >= self.valuation_end:
            raise DataContractError("valuation_start must precede valuation_end.")
        for name in ("top_k", "execution_lag_sessions"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise DataContractError(f"{name} must be a positive integer.")
            object.__setattr__(self, name, int(value))
        if (
            isinstance(self.initial_cash, bool)
            or not isinstance(self.initial_cash, Real)
            or not math.isfinite(self.initial_cash)
            or self.initial_cash <= 0
        ):
            raise DataContractError(
                "initial_cash must be a positive finite real value."
            )
        object.__setattr__(self, "initial_cash", float(self.initial_cash))
        if (
            not isinstance(self.benchmark_symbol, str)
            or not self.benchmark_symbol.strip()
        ):
            raise DataContractError("benchmark_symbol must be nonempty text.")
        object.__setattr__(
            self, "benchmark_symbol", self.benchmark_symbol.strip().upper()
        )
        if not isinstance(self.costs, TradingCosts):
            raise DataContractError("costs must be TradingCosts.")

    def to_mapping(self) -> dict:
        result = asdict(self)
        for name in ("valuation_start", "valuation_end"):
            result[name] = str(getattr(self, name).date())
        result["decision_dates"] = [str(day.date()) for day in self.decision_dates]
        return result


@dataclass(frozen=True)
class LongOnlyEvaluation:
    """Daily evidence; blocked accounts have no terminal NAV or total return."""

    status: str
    ledger: pd.DataFrame
    positions: pd.DataFrame
    trades: pd.DataFrame
    targets: pd.DataFrame
    rejections: pd.DataFrame
    coverage: pd.DataFrame
    summary: MappingProxyType[str, Any]
    manifest: MappingProxyType[str, Any]
    cash: float
    unresolved_holdings: tuple[tuple[str, float], ...]
    blocked_date: pd.Timestamp | None
    blocked_symbols: tuple[str, ...]
    last_valuation_date: pd.Timestamp | None


_COLUMNS = {
    "ledger": (
        "date",
        "previous_nav",
        "pretrade_nav",
        "nav",
        "cash",
        "holding_value",
        "gross_return",
        "net_return",
        "commission",
        "spread",
        "slippage",
        "total_cost",
        "buy_notional",
        "sell_notional",
        "buy_turnover",
        "sell_turnover",
        "two_way_turnover",
    ),
    "positions": ("date", "symbol", "units", "mark", "value", "weight"),
    "trades": (
        "execution_date",
        "decision_date",
        "reason",
        "symbol",
        "side",
        "units",
        "notional",
        "commission",
        "spread",
        "slippage",
    ),
    "targets": (
        "decision_date",
        "execution_date",
        "fold",
        "symbol",
        "score",
        "target_weight",
    ),
    "rejections": (
        "execution_date",
        "decision_date",
        "symbol",
        "target_weight",
        "reason",
    ),
    "coverage": (
        "decision_date",
        "execution_date",
        "scoring_universe_count",
        "scored_count",
        "score_coverage",
        "selected_count",
        "planned_cash_weight",
    ),
}


def _account_value(cash: float, holdings: dict, marks: dict) -> float:
    try:
        value = math.fsum(
            (cash, *(units * marks[symbol] for symbol, units in holdings.items()))
        )
    except OverflowError as error:
        raise DataContractError("Account value exceeds finite arithmetic.") from error
    if not math.isfinite(value) or value <= 0:
        raise DataContractError("Account value must remain positive and finite.")
    return value


def _decision(day, execution, scores, *, top_k):
    values = dict(zip(scores["symbol"], scores["score"], strict=True))
    weights = top_k_weights(values, top_k=top_k)
    targets = [
        {
            "decision_date": day,
            "execution_date": execution,
            "fold": row.fold,
            "symbol": row.symbol,
            "score": row.score,
            "target_weight": weights.get(row.symbol, 0.0),
        }
        for row in scores.itertuples()
    ]
    count = int(scores["score"].notna().sum())
    coverage = {
        "decision_date": day,
        "execution_date": execution,
        "scoring_universe_count": len(scores),
        "scored_count": count,
        "score_coverage": count / len(scores),
        "selected_count": len(weights),
        "planned_cash_weight": max(0.0, 1.0 - math.fsum(weights.values())),
    }
    return weights, targets, coverage


def _daily_row(day, previous, pretrade, nav, cash, trades):
    components = {
        name: math.fsum(getattr(trade, name) for trade in trades)
        for name in ("commission", "spread", "slippage")
    }
    costs = math.fsum(components.values())
    buys = math.fsum(trade.notional for trade in trades if trade.side == "buy")
    sells = math.fsum(trade.notional for trade in trades if trade.side == "sell")
    gross, net = (pretrade - previous) / previous, (nav - previous) / previous
    if not all(
        math.isfinite(value)
        for value in (gross, net, buys / pretrade, sells / pretrade)
    ):
        raise DataContractError("Account returns or turnover exceed finite arithmetic.")
    return {
        "date": day,
        "previous_nav": previous,
        "pretrade_nav": pretrade,
        "nav": nav,
        "cash": cash,
        "holding_value": nav - cash,
        "gross_return": gross,
        "net_return": net,
        **components,
        "total_cost": costs,
        "buy_notional": buys,
        "sell_notional": sells,
        "buy_turnover": buys / pretrade,
        "sell_turnover": sells / pretrade,
        "two_way_turnover": math.fsum((buys / pretrade, sells / pretrade)),
    }


def _trade_rows(day, decision_date, result):
    reason = "terminal_liquidation" if decision_date is None else "scheduled_rebalance"
    trades = [
        {
            "execution_date": day,
            "decision_date": decision_date,
            "reason": reason,
            **asdict(trade),
        }
        for trade in result.trades
    ]
    rejections = [
        {
            "execution_date": day,
            "decision_date": decision_date,
            "symbol": item.symbol,
            "target_weight": item.target_weight,
            "reason": "missing_execution_price",
        }
        for item in result.rejected
    ]
    return trades, rejections


def _summary(ledger: pd.DataFrame, *, completed: bool, initial: float) -> dict:
    peak, worst = initial, 0.0
    for nav in ledger["nav"]:
        peak = max(peak, nav)
        worst = min(worst, nav / peak - 1.0)
    last_nav = float(ledger.iloc[-1]["nav"]) if not ledger.empty else None
    summary = {
        "valuation_count": len(ledger),
        "last_valued_nav": last_nav,
        "terminal_nav": last_nav if completed else None,
        "total_net_return": last_nav / initial - 1.0 if completed else None,
        "observed_max_drawdown": worst if not ledger.empty else None,
    }
    try:
        summary.update(
            observed_cost=math.fsum(ledger["total_cost"]),
            observed_two_way_turnover=math.fsum(ledger["two_way_turnover"]),
        )
    except OverflowError as error:
        raise DataContractError("Account summary exceeds finite arithmetic.") from error
    if any(
        value is not None and not math.isfinite(value) for value in summary.values()
    ):
        raise DataContractError("Account summary exceeds finite arithmetic.")
    return summary


def _execute(day, event, *, holdings, cash, marks, config, records):
    decision_date, weights = event
    result = rebalance(
        holdings=holdings,
        prices=marks,
        cash=cash,
        target_weights=weights,
        costs=config.costs,
    )
    trades, rejections = _trade_rows(day, decision_date, result)
    records["trades"].extend(trades)
    records["rejections"].extend(rejections)
    return dict(result.holdings), result.cash, result.trades


def _position_rows(day, holdings, marks, nav):
    return [
        {
            "date": day,
            "symbol": symbol,
            "units": units,
            "mark": marks[symbol],
            "value": units * marks[symbol],
            "weight": units * marks[symbol] / nav,
        }
        for symbol, units in sorted(holdings.items())
    ]


def _simulate(inputs, config):
    records = {name: [] for name in _COLUMNS}
    price_days = {
        day: dict(zip(group.symbol, group.adjusted_close, strict=True))
        for day, group in inputs.prices.groupby("date", sort=True)
    }
    score_days = dict(tuple(inputs.scores.groupby("as_of_date", sort=True)))
    execution_dates = dict(inputs.execution_dates)
    pending, holdings = {}, {}
    cash = previous_nav = config.initial_cash
    blocked_date, blocked_symbols = None, ()
    for day in inputs.sessions:
        marks = price_days[day]
        missing = tuple(sorted(set(holdings) - set(marks)))
        if missing:
            blocked_date, blocked_symbols = day, missing
            break
        pretrade_nav = _account_value(cash, holdings, marks)
        day_trades = ()
        if day in pending or day == config.valuation_end:
            holdings, cash, day_trades = _execute(
                day,
                pending.get(day, (None, {})),
                holdings=holdings,
                cash=cash,
                marks=marks,
                config=config,
                records=records,
            )
            pending = {date: event for date, event in pending.items() if date != day}
        nav = _account_value(cash, holdings, marks)
        records["ledger"].append(
            _daily_row(day, previous_nav, pretrade_nav, nav, cash, day_trades)
        )
        records["positions"].extend(_position_rows(day, holdings, marks, nav))
        if day in execution_dates:
            execution = execution_dates[day]
            weights, targets, coverage = _decision(
                day, execution, score_days[day], top_k=config.top_k
            )
            pending = {**pending, execution: (day, weights)}
            records["targets"].extend(targets)
            records["coverage"].append(coverage)
        previous_nav = nav
    return records, cash, holdings, blocked_date, blocked_symbols


def evaluate_long_only(
    run: BenchmarkRun,
    prices: pd.DataFrame,
    *,
    model: str,
    calendar: ExpectedSessionCalendar,
    config: LongOnlyConfig,
) -> LongOnlyEvaluation:
    """Value one continuous development account from saved native scores.

    No fitting, labels, registry writes or final evaluation occur. The adapter
    checks supplied development metadata and projects decision-time score fields.
    Missing new-buy marks leave cash; a missing held mark returns an incomplete
    account before any trade that session. Invalid input contracts raise ValueError.
    """
    if not isinstance(config, LongOnlyConfig):
        raise DataContractError("config must be LongOnlyConfig.")
    inputs = prepare_portfolio_inputs(
        run, prices, model=model, calendar=calendar, config=config
    )
    state = _simulate(inputs, config)
    return _evaluation(state, inputs=inputs, model=model, config=config)


def _evaluation(state, *, inputs, model, config):
    records, cash, holdings, blocked_date, blocked_symbols = state
    frames = {
        name: pd.DataFrame(rows, columns=_COLUMNS[name])
        for name, rows in records.items()
    }
    completed = blocked_date is None
    return LongOnlyEvaluation(
        status="completed" if completed else "blocked_missing_valuation",
        **frames,
        summary=MappingProxyType(
            _summary(frames["ledger"], completed=completed, initial=config.initial_cash)
        ),
        manifest=_freeze(
            {
                "schema_version": "1",
                "scope": "development_cost_sensitivity",
                "model": model,
                "configuration": config.to_mapping(),
                "inputs": inputs.identity,
                "package_version": __version__,
                "consumer_source_fingerprint": _source_fingerprint(),
                "decision_event": "after_close",
                "execution_event": "close",
                "position_unit": "fractional_adjusted_price_unit",
                "final_outcomes_evaluated": False,
            }
        ),
        cash=cash,
        unresolved_holdings=tuple(sorted(holdings.items())),
        blocked_date=blocked_date,
        blocked_symbols=blocked_symbols,
        last_valuation_date=frames["ledger"].iloc[-1]["date"]
        if not frames["ledger"].empty
        else None,
    )
