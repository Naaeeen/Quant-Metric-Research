"""Actual invented prices -> trained scores -> continuous development ledger."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_metric_research import ExperimentRegistry

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "synthetic_portfolio.py"


@pytest.fixture
def example():
    spec = importlib.util.spec_from_file_location("synthetic_portfolio", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trained_two_fold_accounts_and_outcome_independence(
    example, tmp_path, monkeypatch
):
    captured = []
    evaluate = example.evaluate_long_only

    def inspect(run, prices, **kwargs):
        assert run.manifest["execution_mode"] == "development"
        assert set(run.predictions.fold) == {1, 2}
        assert (
            run.predictions.train_label_end_max < run.predictions.evaluation_start
        ).all()
        result = evaluate(run, prices, **kwargs)
        stripped = replace(
            run, predictions=run.predictions.drop(columns=["target", "realized_return"])
        )
        without_outcomes = evaluate(stripped, prices, **kwargs)
        pd.testing.assert_frame_equal(
            result.ledger, without_outcomes.ledger, check_exact=True
        )
        captured.append((run, kwargs, result))
        return result

    monkeypatch.setattr(example, "evaluate_long_only", inspect)
    output = tmp_path / "demo"
    report = example.run_example(output)
    assert report["status"] == "complete"
    assert (
        report["synthetic_data"] is True and report["final_outcomes_evaluated"] is False
    )
    assert json.loads((output / "completion.json").read_text()) == report
    history = ExperimentRegistry(output / "experiments.sqlite3").list_runs()
    assert (
        len(history) == 1
        and history[0]["kind"] == "development"
        and history[0]["status"] == "completed"
    )
    assert (output / "benchmark" / "benchmark_manifest.json").is_file()
    assert len(captured) == 4
    for run, kwargs, result in captured:
        config = kwargs["config"]
        assert result.status == "completed" and result.unresolved_holdings == ()
        assert config.valuation_end < pd.Timestamp(run.manifest["locked_test_start"])
        assert set(result.targets.fold) == {1, 2}
        assert result.coverage.execution_date.max() < config.valuation_end
        assert result.ledger.date.is_unique
        assert result.ledger.iloc[0].previous_nav == config.initial_cash
        np.testing.assert_allclose(
            result.ledger.previous_nav.iloc[1:],
            result.ledger.nav.iloc[:-1],
            rtol=0,
            atol=0,
        )
        assert set(result.trades.side) == {"buy", "sell"}
        terminal = result.trades.loc[result.trades.reason.eq("terminal_liquidation")]
        assert not terminal.empty and terminal.side.eq("sell").all()
        assert terminal.execution_date.eq(config.valuation_end).all()
        assert result.ledger.iloc[-1].holding_value == pytest.approx(0.0)
        np.testing.assert_allclose(
            result.ledger.pretrade_nav - result.ledger.nav,
            result.ledger.total_cost,
            atol=1e-8,
        )
        costs = result.trades[["commission", "spread", "slippage"]].sum().sum()
        assert result.summary["observed_cost"] == pytest.approx(costs)
        rate = config.costs.buy.total
        assert config.costs.sell.total == rate
        assert costs == pytest.approx(result.trades.notional.sum() * rate)
        directory = output / f"{kwargs['model']}_{'zero' if rate == 0 else '5bps'}"
        for name in (
            "ledger",
            "positions",
            "trades",
            "targets",
            "rejections",
            "coverage",
        ):
            assert (directory / f"{name}.csv").is_file()
        assert (
            json.loads((directory / "manifest.json").read_text())[
                "final_outcomes_evaluated"
            ]
            is False
        )
    schedules = [item[1]["config"].decision_dates for item in captured]
    assert all(schedule == schedules[0] for schedule in schedules)
    for model in ("ridge", "equal_weight_rank"):
        zero, charged = [item[2] for item in captured if item[1]["model"] == model]
        assert zero.summary["observed_cost"] == 0
        assert charged.summary["observed_cost"] > 0
        assert charged.summary["terminal_nav"] < zero.summary["terminal_nav"]


def test_existing_output_refused_before_training(example, tmp_path, monkeypatch):
    monkeypatch.setattr(
        example, "run_stage3_benchmark", lambda *a, **k: pytest.fail("Must not train.")
    )
    with pytest.raises(FileExistsError):
        example.run_example(tmp_path)


def test_infeasible_preflight_retains_evidence_without_completion(
    example, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        example, "preflight_benchmark", lambda *a, **k: {"feasible": False}
    )
    monkeypatch.setattr(
        example, "run_stage3_benchmark", lambda *a, **k: pytest.fail("Must not train.")
    )
    output = tmp_path / "demo"
    with pytest.raises(ValueError, match="preflight"):
        example.run_example(output)
    assert (output / "preflight.json").is_file()
    assert not (output / "completion.json").exists()


def test_direct_cli_and_overwrite_refusal(tmp_path):
    command = [sys.executable, str(EXAMPLE), "--output-dir", str(tmp_path / "cli")]
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "complete"
    repeated = subprocess.run(
        command, capture_output=True, text=True, timeout=60, check=False
    )
    assert repeated.returncode == 2
    assert "exists" in repeated.stderr


def test_blocked_account_keeps_benchmark_and_no_completion(
    example, tmp_path, monkeypatch
):
    evaluate = example.evaluate_long_only
    observed = []

    def blocked(run, prices, **kwargs):
        completed = evaluate(run, prices, **kwargs)
        terminal = kwargs["config"].valuation_end
        held = completed.trades.loc[
            completed.trades.reason.eq("terminal_liquidation"), "symbol"
        ].iloc[0]
        missing_mark = prices.loc[
            ~(prices.date.eq(terminal) & prices.symbol.eq(held))
        ].copy()
        result = evaluate(run, missing_mark, **kwargs)
        assert result.status == "blocked_missing_valuation"
        assert result.blocked_date == terminal and result.blocked_symbols == (held,)
        observed.append(result)
        return result

    monkeypatch.setattr(example, "evaluate_long_only", blocked)
    output = tmp_path / "blocked"
    with pytest.raises(ValueError, match="blocked"):
        example.run_example(output)
    assert (output / "benchmark" / "benchmark_manifest.json").is_file()
    summary = json.loads((output / "ridge_zero" / "summary.json").read_text())
    assert summary["status"] == "blocked_missing_valuation"
    result = observed[0]
    assert summary["terminal_nav"] is None and summary["total_net_return"] is None
    assert summary["blocked_date"] == result.blocked_date.date().isoformat()
    assert (
        summary["last_valuation_date"] == result.last_valuation_date.date().isoformat()
    )
    assert summary["blocked_symbols"] == list(result.blocked_symbols)
    assert summary["cash"] == result.cash
    assert summary["unresolved_holdings"] == [
        list(item) for item in result.unresolved_holdings
    ]
    assert summary["unresolved_holdings"]
    assert not (output / "completion.json").exists()
