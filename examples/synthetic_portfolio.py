"""Train invented development scores and account for holdings, cash and costs.

No network or market data. Fractional adjusted-price units demonstrate plumbing,
not executable broker shares, realistic returns or evidence of tradable alpha.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from quant_metric_research import (
    BenchmarkConfig,
    ExperimentRegistry,
    NestedSplitConfig,
    PanelConfig,
    build_point_in_time_panel,
    preflight_benchmark,
    run_stage3_benchmark,
    write_benchmark_run,
)
from quant_metric_research.portfolio import LongOnlyConfig, evaluate_long_only
from quant_metric_research.portfolio_accounting import SideCosts, TradingCosts
from quant_metric_research.session_calendar import ExpectedSessionCalendar

FEATURES = ("trailing_return", "annualized_volatility", "beta")


def invented_inputs():
    """Declare a toy weekday calendar before constructing complete price histories."""
    dates = pd.bdate_range("2024-01-02", periods=64)
    calendar = ExpectedSessionCalendar(
        tuple(dates), dates[0], dates[-1], "invented weekdays, not an exchange", "1"
    )
    random = np.random.default_rng(20260922)
    market = random.normal(0.0003, 0.005, len(dates))
    prices = []
    symbols = tuple(f"SYNTH_{index}" for index in range(8))
    for index, symbol in enumerate(("BENCH", *symbols)):
        returns = (
            market
            if symbol == "BENCH"
            else (
                market
                + 0.006 * np.sin(np.arange(len(dates)) / 4 + index)
                + random.normal(0, 0.004, len(dates))
            )
        )
        prices.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": symbol,
                    "adjusted_close": 100 * np.exp(np.cumsum(returns)),
                }
            )
        )
    memberships = pd.DataFrame(
        [
            dict(
                universe_id="INVENTED",
                symbol=symbol,
                effective_from=dates[0],
                effective_to=None,
                source="invented example",
            )
            for symbol in symbols
        ]
    )
    panel_config = PanelConfig(
        dataset_version="synthetic-portfolio-v1",
        universe_id="INVENTED",
        benchmark_symbol="BENCH",
        lookback_sessions=8,
        min_observations=6,
        target_horizon_sessions=2,
        entry_lag_sessions=1,
        feature_columns=FEATURES,
    )
    return pd.concat(prices, ignore_index=True), memberships, calendar, panel_config


def _write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                allow_nan=False,
                default=lambda mapping: dict(mapping),
            )
            + "\n"
        )


def _save_account(result, directory):
    directory.mkdir(exist_ok=False)
    for name in ("ledger", "positions", "trades", "targets", "rejections", "coverage"):
        getattr(result, name).to_csv(directory / f"{name}.csv", index=False, mode="x")
    _write_json(
        directory / "summary.json",
        dict(
            result.summary,
            status=result.status,
            cash=result.cash,
            unresolved_holdings=result.unresolved_holdings,
            blocked_symbols=result.blocked_symbols,
            blocked_date=result.blocked_date.date().isoformat()
            if result.blocked_date is not None
            else None,
            last_valuation_date=result.last_valuation_date.date().isoformat()
            if result.last_valuation_date is not None
            else None,
        ),
    )
    _write_json(directory / "manifest.json", result.manifest)


def run_example(output: str | Path) -> dict:
    """Run one fresh synthetic benchmark and four fixed development ledgers."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    prices, memberships, calendar, panel_config = invented_inputs()
    # The final three sessions supply labels, not decision rows.
    panel = build_point_in_time_panel(
        prices,
        memberships,
        as_of_dates=calendar.sessions[8:-3],
        config=panel_config,
        expected_calendar=calendar,
    )
    config = BenchmarkConfig(
        feature_columns=FEATURES,
        split=NestedSplitConfig(6, 2, 6, 16, 1, 4, 8),
        model_families=("ridge",),
        ridge_alphas=(1.0,),
        min_cross_section=8,
        quantiles=2,
        hac_lags=1,
    )
    prices.to_csv(output / "invented_prices.csv", index=False)
    memberships.to_csv(output / "invented_memberships.csv", index=False)
    panel.to_parquet(output / "invented_panel.parquet", index=False)
    _write_json(output / "calendar.json", calendar.to_mapping())
    _write_json(output / "panel_config.json", asdict(panel_config))
    _write_json(output / "benchmark_config.json", config.to_mapping())
    preflight = preflight_benchmark(panel, config=config)
    _write_json(output / "preflight.json", preflight)
    if not preflight["feasible"]:
        raise ValueError("Synthetic benchmark preflight is infeasible.")
    with threadpool_limits(limits=1):
        run = run_stage3_benchmark(
            panel,
            config=config,
            registry=ExperimentRegistry(output / "experiments.sqlite3"),
            evaluate_lockbox=False,
            study_id="synthetic-portfolio-demo",
            hypothesis="Invented end-to-end accounting check; no alpha claim.",
        )
    write_benchmark_run(run, output / "benchmark")
    schedule = tuple(
        pd.Timestamp(day)
        for day in run.manifest["evaluation_schedule"]["dates_by_phase"]["development"]
    )
    accounts = []
    for model in ("ridge", "equal_weight_rank"):
        for cost_name, rate in (("zero", 0.0), ("5bps", 0.0005)):
            portfolio_config = LongOnlyConfig(
                valuation_start=schedule[0],
                valuation_end=schedule[-1],
                decision_dates=schedule[::3],
                benchmark_symbol="BENCH",
                top_k=2,
                execution_lag_sessions=1,
                initial_cash=100_000.0,
                costs=TradingCosts(
                    buy=SideCosts(commission=rate), sell=SideCosts(commission=rate)
                ),
            )
            result = evaluate_long_only(
                run, prices, model=model, calendar=calendar, config=portfolio_config
            )
            directory = f"{model}_{cost_name}"
            _save_account(result, output / directory)
            if result.status != "completed":
                raise ValueError(
                    "Synthetic account blocked; inspect retained evidence."
                )
            accounts.append(
                dict(
                    model=model,
                    cost_scenario=cost_name,
                    directory=directory,
                    status=result.status,
                    **result.summary,
                )
            )
    report = dict(
        status="complete",
        synthetic_data=True,
        empirical_alpha_claim=False,
        final_outcomes_evaluated=False,
        training_threads=1,
        outer_folds=2,
        elapsed_seconds=time.perf_counter() - started,
        development_run_id=run.manifest["experiment"]["run_id"],
        accounts=accounts,
        warning="Invented adjusted-price-unit accounting, "
        "not executable shares or alpha evidence.",
    )
    _write_json(output / "completion.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = run_example(args.output_dir)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Synthetic portfolio failed: {error}\n")
    print(json.dumps(report, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
