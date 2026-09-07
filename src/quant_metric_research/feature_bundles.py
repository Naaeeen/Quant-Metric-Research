"""Fixed candidate feature schemas, not complete experiment preregistrations."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class FeatureBundleSpec:
    bundle_id: str
    feature_columns: tuple[str, ...]
    price_factor_names: tuple[str, ...]
    factor_formula_versions: tuple[tuple[str, str], ...]


# Versioned bundles must not expand when defaults or the factor catalog change.
_LEGACY_COLUMNS = (
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
_PRICE_NAMES = ("return_21s", "momentum_252s_skip_21s", "ma_distance_63s")
_PRICE_VERSIONS = (
    ("return_21s", "1"),
    ("momentum_252s_skip_21s", "1"),
    ("ma_distance_63s", "1"),
)

FEATURE_BUNDLES = MappingProxyType(
    {
        "legacy10_v1": FeatureBundleSpec(
            bundle_id="legacy10_v1",
            feature_columns=_LEGACY_COLUMNS,
            price_factor_names=(),
            factor_formula_versions=(),
        ),
        "legacy10_plus_price3_v1": FeatureBundleSpec(
            bundle_id="legacy10_plus_price3_v1",
            feature_columns=(*_LEGACY_COLUMNS, *_PRICE_NAMES),
            price_factor_names=_PRICE_NAMES,
            factor_formula_versions=_PRICE_VERSIONS,
        ),
    }
)
