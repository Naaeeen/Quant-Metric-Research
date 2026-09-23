from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from quant_metric_research.config import DEFAULT_FEATURE_COLUMNS
from quant_metric_research.feature_bundles import FEATURE_BUNDLES, FeatureBundleSpec
from quant_metric_research.price_factor_catalog import PRICE_FACTOR_CATALOG

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
PRICE = ("return_21s", "momentum_252s_skip_21s", "ma_distance_63s")


def test_bundles_have_exact_versioned_names_order_and_formula_versions():
    assert tuple(FEATURE_BUNDLES) == ("legacy10_v1", "legacy10_plus_price3_v1")
    legacy = FEATURE_BUNDLES["legacy10_v1"]
    extended = FEATURE_BUNDLES["legacy10_plus_price3_v1"]
    assert legacy.bundle_id == "legacy10_v1"
    assert legacy.feature_columns == LEGACY == DEFAULT_FEATURE_COLUMNS
    assert legacy.price_factor_names == ()
    assert legacy.factor_formula_versions == ()
    assert extended.bundle_id == "legacy10_plus_price3_v1"
    assert extended.feature_columns == LEGACY + PRICE
    assert extended.price_factor_names == PRICE
    assert extended.factor_formula_versions == tuple((name, "1") for name in PRICE)


def test_bundle_mapping_and_nested_catalog_records_are_immutable():
    with pytest.raises(TypeError):
        FEATURE_BUNDLES["new"] = FEATURE_BUNDLES["legacy10_v1"]
    for bundle in FEATURE_BUNDLES.values():
        assert isinstance(bundle, FeatureBundleSpec)
        assert not hasattr(bundle, "__dict__")
        assert isinstance(bundle.feature_columns, tuple)
        assert isinstance(bundle.price_factor_names, tuple)
        assert isinstance(bundle.factor_formula_versions, tuple)
        assert all(isinstance(pair, tuple) for pair in bundle.factor_formula_versions)
        with pytest.raises(FrozenInstanceError):
            bundle.bundle_id = "changed"


def test_declared_versions_match_the_current_calculator_catalog():
    for bundle in FEATURE_BUNDLES.values():
        assert tuple(name for name, _ in bundle.factor_formula_versions) == (
            bundle.price_factor_names
        )
        for name, version in bundle.factor_formula_versions:
            assert PRICE_FACTOR_CATALOG[name].formula_version == version


def test_bundles_do_not_expand_when_price_catalog_is_extended(monkeypatch):
    import quant_metric_research.price_factor_catalog as catalog

    monkeypatch.setattr(catalog, "PRICE_FACTOR_CATALOG", {"future_factor": object()})
    assert FEATURE_BUNDLES["legacy10_plus_price3_v1"].price_factor_names == PRICE
    assert FEATURE_BUNDLES["legacy10_plus_price3_v1"].feature_columns == LEGACY + PRICE
