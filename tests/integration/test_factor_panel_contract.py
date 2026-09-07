"""Synthetic integration checks; no real-data model search or final scoring."""

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import quant_metric_research as qmr
from quant_metric_research.benchmark_data import (
    build_stage3_data_plan,
    validate_benchmark_panel,
)
from quant_metric_research.benchmark_reporting import _fingerprints
from quant_metric_research.config import DEFAULT_FEATURE_COLUMNS

FACTOR_NAMES = ("return_21s", "momentum_252s_skip_21s", "ma_distance_63s")


def inputs():
    dates = pd.bdate_range("2020-01-01", periods=320)
    position = np.arange(len(dates))
    prices = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": symbol,
                    "adjusted_close": 100
                    * np.exp(
                        0.0005 * position
                        + (number + 1) * 0.002 * np.sin(position / (number + 3))
                    ),
                }
            )
            for number, symbol in enumerate(("BENCH", "A", "B", "C", "D", "E", "F"))
        ],
        ignore_index=True,
    )
    memberships = pd.DataFrame(
        [
            {
                "universe_id": "INVENTED",
                "symbol": symbol,
                "effective_from": dates[0],
                "effective_to": None,
                "source": "invented fixture",
            }
            for symbol in ("A", "B", "C", "D", "E", "F", "MISSING")
        ]
    )
    config = qmr.PanelConfig(
        dataset_version="synthetic-factor-panel-v1",
        universe_id="INVENTED",
        benchmark_symbol="BENCH",
        target_horizon_sessions=2,
        entry_lag_sessions=1,
    )
    return prices, memberships, dates, config


def build(prices, memberships, dates, config):
    return qmr.build_factor_panel(
        prices,
        memberships,
        as_of_dates=dates,
        config=config,
        factor_names=FACTOR_NAMES,
    )


def test_public_bundle_schemas_are_fixed_and_immutable():
    assert tuple(qmr.FEATURE_BUNDLES) == ("legacy10_v1", "legacy10_plus_price3_v1")
    baseline = qmr.FEATURE_BUNDLES["legacy10_v1"]
    expanded = qmr.FEATURE_BUNDLES["legacy10_plus_price3_v1"]
    assert isinstance(expanded, qmr.FeatureBundleSpec)
    assert baseline.feature_columns == DEFAULT_FEATURE_COLUMNS
    assert expanded.feature_columns == DEFAULT_FEATURE_COLUMNS + FACTOR_NAMES
    assert baseline.price_factor_names == baseline.factor_formula_versions == ()
    assert expanded.price_factor_names == FACTOR_NAMES
    assert expanded.factor_formula_versions == tuple(
        (name, "1") for name in FACTOR_NAMES
    )
    with pytest.raises(FrozenInstanceError):
        expanded.bundle_id = "changed"
    with pytest.raises(TypeError):
        qmr.FEATURE_BUNDLES["extra"] = expanded


def test_enrichment_preserves_legacy_frame_and_missing_rows_exactly():
    prices, memberships, dates, config = inputs()
    requested = [dates[30], dates[63], dates[252], dates[-1]]
    before_prices, before_memberships = (
        prices.copy(deep=True),
        memberships.copy(deep=True),
    )
    expected = qmr.build_point_in_time_panel(
        prices, memberships, as_of_dates=requested, config=config
    )
    result = build(prices, memberships, requested, config)
    pd.testing.assert_frame_equal(result.loc[:, expected.columns], expected)
    pd.testing.assert_frame_equal(prices, before_prices)
    pd.testing.assert_frame_equal(memberships, before_memberships)
    assert len(result) == 28
    assert (result["symbol"] == "MISSING").sum() == 4
    assert (
        result.loc[result["symbol"] == "MISSING", list(FACTOR_NAMES)].isna().all().all()
    )
    early = result.loc[
        (result["symbol"] == "A") & (result["as_of_date"] == dates[30])
    ].iloc[0]
    assert not bool(early["feature_eligible"])
    assert early["return_21s_status"] == "ok"
    assert (
        result.loc[result["as_of_date"] == dates[-1], "target_available"]
        .eq(False)
        .all()
    )


def test_bundles_keep_identical_stage3_rows_and_purged_split_assignments():
    prices, memberships, dates, config = inputs()
    panel = build(prices, memberships, list(dates[252:302]) + list(dates[-2:]), config)
    plans = [
        build_stage3_data_plan(
            panel,
            feature_columns=bundle.feature_columns,
            target_column="forward_excess_return",
            locked_test_date_count=4,
            locked_min_cross_section=4,
            n_splits=2,
            evaluation_date_count=4,
            min_train_date_count=12,
        )
        for bundle in qmr.FEATURE_BUNDLES.values()
    ]
    assert (
        plans[0].panel.frame["row_id"].tolist()
        == plans[1].panel.frame["row_id"].tolist()
    )
    assert plans[0].locked_test == plans[1].locked_test
    assert plans[0].development_folds == plans[1].development_folds
    assert len(plans[1].panel.frame) == len(panel)
    benchmark = qmr.BenchmarkConfig(
        feature_columns=qmr.FEATURE_BUNDLES["legacy10_v1"].feature_columns,
        target_column="forward_excess_return",
        realized_return_column="forward_excess_return",
        min_cross_section=4,
        quantiles=2,
        hac_lags=1,
        split=qmr.NestedSplitConfig(
            final_test_date_count=4,
            outer_n_splits=2,
            outer_test_date_count=4,
            outer_min_train_date_count=12,
            inner_n_splits=2,
            inner_validation_date_count=3,
            inner_min_train_date_count=8,
        ),
    )
    fingerprints = []
    for plan, bundle in zip(plans, qmr.FEATURE_BUNDLES.values(), strict=True):
        bundle_config = replace(benchmark, feature_columns=bundle.feature_columns)
        report = qmr.preflight_benchmark(panel, config=bundle_config)
        assert report["feasible"] is True, report["errors"]
        assert report["summary"]["no_training_performed"] is True
        assert report["summary"]["no_predictive_outcomes_computed"] is True
        json.dumps(report, allow_nan=False)
        fingerprints.append(
            _fingerprints(plan, config=bundle_config, evaluate_lockbox=False)
        )
    assert fingerprints[0]["panel_fingerprint"] == fingerprints[1]["panel_fingerprint"]
    assert (
        fingerprints[0]["source_fingerprint"] == fingerprints[1]["source_fingerprint"]
    )
    assert (
        fingerprints[0]["model_input_fingerprint"]
        != fingerprints[1]["model_input_fingerprint"]
    )
    assert fingerprints[0]["run_fingerprint"] != fingerprints[1]["run_fingerprint"]


def test_stage3_rejects_future_availability_for_an_observed_new_factor():
    prices, memberships, dates, config = inputs()
    panel = build(prices, memberships, [dates[252]], config)
    features = qmr.FEATURE_BUNDLES["legacy10_plus_price3_v1"].feature_columns
    validate_benchmark_panel(
        panel, feature_columns=features, target_column="forward_excess_return"
    )
    changed = panel.copy(deep=True)
    changed.loc[0, "return_21s_available_at"] = dates[253]
    with pytest.raises(ValueError, match="available after"):
        validate_benchmark_panel(
            changed, feature_columns=features, target_column="forward_excess_return"
        )


def test_panel_example_is_offline_and_non_fitting():
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, str(root / "examples/synthetic_factor_panel.py")],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["claim_scope"] == "synthetic_factor_panel_example"
    assert report["legacy_panel_preserved"] is True
    assert report["training_performed"] is False
    assert report["network_accessed"] is False
    assert report["final_outcomes_evaluated"] is False
    assert report["row_count"] == 16
    assert report["candidate_feature_counts"] == {
        "legacy10_v1": 10,
        "legacy10_plus_price3_v1": 13,
    }
