"""Independent capability and compatibility checks for opt-in price factors."""

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest

import quant_metric_research as qmr
from quant_metric_research.config import DEFAULT_FEATURE_COLUMNS
from quant_metric_research.public_demo import public_demo_configs

FACTORS = ("return_21s", "momentum_252s_skip_21s", "ma_distance_63s")
LEGACY = (
    "trailing_return",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "benchmark_correlation",
    "beta",
    "capm_alpha",
    "information_ratio",
    "historical_var_5pct",
)


def calculate(prices, dates, factors=FACTORS):
    return qmr.compute_price_factors(
        prices, calendar=dates, as_of_date=dates[252], factor_names=factors
    )


def test_public_factor_api_and_immutable_nested_values():
    dates = pd.bdate_range("2020-01-01", periods=253)
    prices = pd.Series(np.linspace(100, 150, len(dates)), index=dates)
    result = calculate(prices, dates)
    assert isinstance(result, qmr.PriceFactorResult)
    assert tuple(row.name for row in result.factors) == FACTORS
    assert all(isinstance(row, qmr.PriceFactorValue) for row in result.factors)
    assert all(
        isinstance(spec, qmr.PriceFactorSpec)
        for spec in qmr.PRICE_FACTOR_CATALOG.values()
    )
    with pytest.raises(FrozenInstanceError):
        result.factors[0].value = 0
    with pytest.raises(TypeError):
        qmr.PRICE_FACTOR_CATALOG["invented"] = result.factors[0]
    with pytest.raises(FrozenInstanceError):
        qmr.PRICE_FACTOR_CATALOG["return_21s"].required_price_count = 1


def test_nullable_factor_result_serializes_without_nan_and_keeps_input():
    dates = pd.bdate_range("2020-01-01", periods=253)
    prices = pd.Series(np.linspace(100, 150, len(dates)), index=dates, name="TEST")
    prices.iloc[240] = np.nan
    before = prices.copy(deep=True)
    result = calculate(prices, dates)
    encoded = json.dumps(
        asdict(result), default=lambda value: value.isoformat(), allow_nan=False
    )
    decoded = json.loads(encoded)
    assert decoded["factors"][0]["value"] is None
    assert decoded["factors"][0]["status"] == "missing_required_prices"
    assert decoded["factors"][1]["status"] == "ok"
    assert decoded["factors"][2]["value"] is None
    pd.testing.assert_series_equal(prices, before)


@pytest.mark.parametrize("magnitude", [1e308, 1e-300, np.nextafter(0.0, 1.0)])
def test_constant_extreme_finite_prices_do_not_overflow_the_mean(magnitude):
    dates = pd.bdate_range("2020-01-01", periods=253)
    prices = pd.Series(magnitude, index=dates)
    result = calculate(prices, dates)
    assert all(row.status == "ok" for row in result.factors)
    assert all(row.value == pytest.approx(0.0, abs=1e-15) for row in result.factors)


@pytest.mark.parametrize(
    "future_value", ["bad", True, complex(3, 2), np.inf, pd.Timestamp("2030-01-01")]
)
def test_future_values_do_not_change_historical_factors(future_value):
    dates = pd.bdate_range("2020-01-01", periods=258)
    prices = pd.Series(np.linspace(100, 150, len(dates)), index=dates, dtype=object)
    expected = calculate(prices, dates)
    changed = prices.copy(deep=True)
    changed.iloc[253:] = future_value
    assert calculate(changed, dates) == expected


def test_overflowing_factor_ratio_is_a_clear_contract_error():
    dates = pd.bdate_range("2020-01-01", periods=253)
    prices = pd.Series(1.0, index=dates)
    prices.iloc[231] = 1e-300
    prices.iloc[252] = 1e308
    with pytest.raises(qmr.DataContractError, match="return_21s"):
        calculate(prices, dates, ("return_21s",))


def test_catalog_does_not_expand_legacy_or_fixed_public_experiment():
    assert DEFAULT_FEATURE_COLUMNS == LEGACY
    panel, benchmark = public_demo_configs()
    assert panel.feature_columns == benchmark.feature_columns == LEGACY
    assert (
        panel.lookback_sessions,
        panel.min_observations,
        panel.target_horizon_sessions,
        panel.entry_lag_sessions,
    ) == (252, 126, 20, 1)
    assert benchmark.model_families == ("ridge",)
    assert benchmark.ridge_alphas == (1.0,)
    assert benchmark.include_pca_model is False
    assert benchmark.split.final_test_date_count == 63
    assert benchmark.split.outer_test_date_count == 63
    assert benchmark.split.inner_validation_date_count == 42


def test_colab_keeps_pre_factor_source_pin_and_no_automatic_new_experiment():
    root = Path(__file__).resolve().parents[2]
    notebook = nbformat.read(root / "examples/colab_public_demo.ipynb", as_version=4)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert 'SOURCE_REVISION = "7740dee1b4c6aeed463d6440511d7b606c6fe406"' in code
    assert "compute_price_factors" not in code
    assert "notebook-demo" in code
    assert all(
        cell.execution_count is None and cell.outputs == []
        for cell in notebook.cells
        if cell.cell_type == "code"
    )


def test_offline_factor_example_runs_without_training_or_market_data():
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, str(root / "examples/synthetic_price_factors.py")],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["claim_scope"] == "synthetic_factor_example"
    assert report["training_performed"] is False
    assert report["final_outcomes_evaluated"] is False
    assert report["network_accessed"] is False
    assert [row["name"] for row in report["result"]["factors"]] == list(FACTORS)
    assert all(row["status"] == "ok" for row in report["result"]["factors"])
