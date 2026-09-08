"""Small opt-in input fixtures for the existing notebook workflow test."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import nbformat
import numpy as np
import pandas as pd
import pytest

from quant_metric_research import BenchmarkConfig, PanelConfig, public_demo


@pytest.fixture
def synthetic_notebook_demo(monkeypatch):
    """Invented inputs only; the test still runs the real development workflow."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2013-01-02", periods=75)
    symbols = tuple(f"SAMPLE_{index}" for index in range(8))
    prices = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": symbol,
                    "adjusted_close": 100
                    * np.exp(np.cumsum(rng.normal(0.001, 0.012, len(dates)))),
                }
            )
            for symbol in (*symbols, "BENCH")
        ],
        ignore_index=True,
    )
    features = ("trailing_return", "annualized_volatility", "max_drawdown")
    panel = PanelConfig(
        dataset_version="synthetic-history-test-v1",
        universe_id="SYNTHETIC_COHORT",
        benchmark_symbol="BENCH",
        lookback_sessions=10,
        min_observations=8,
        target_horizon_sessions=2,
        entry_lag_sessions=1,
        feature_columns=features,
    )
    benchmark = BenchmarkConfig.from_mapping(
        {
            "feature_columns": features,
            "model_families": ["ridge"],
            "ridge_alphas": [1.0],
            "min_cross_section": 8,
            "quantiles": 2,
            "hac_lags": 1,
            "split": {
                "final_test_date_count": 6,
                "outer_n_splits": 2,
                "outer_test_date_count": 6,
                "outer_min_train_date_count": 18,
                "inner_n_splits": 1,
                "inner_validation_date_count": 4,
                "inner_min_train_date_count": 8,
            },
        }
    )
    sample = SimpleNamespace(
        prices=prices,
        symbols=symbols,
        profile={"synthetic_software_fixture": True},
        missing_prices=pd.DataFrame(columns=["date", "symbol", "reason"]),
    )
    monkeypatch.setattr(
        public_demo, "_prepare_sample", lambda path: (sample, {"synthetic": True})
    )
    monkeypatch.setattr(public_demo, "public_demo_configs", lambda: (panel, benchmark))
    return sample


@pytest.fixture
def notebook_summary_sources():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "compatibility_colab_builder", root / "examples/build_colab_notebook.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Revision is irrelevant to the summary cell; exact released-pin parity has
    # a separate checked-in notebook test. Never execute the bootstrap here.
    notebooks = (
        module.build_notebook("a" * 40),
        nbformat.read(root / "examples/colab_public_demo.ipynb", as_version=4),
    )
    return tuple(
        next(cell.source for cell in notebook.cells if cell.id == "qmr-colab-09")
        for notebook in notebooks
    )


@pytest.fixture
def snapshot_notebook_paths():
    """Independent byte/entry oracle for bounded synthetic history paths."""

    def snapshot(*roots):
        paths = [path for root in roots for path in (root, *root.rglob("*"))]
        return {path: None if path.is_dir() else path.read_bytes() for path in paths}

    return snapshot
