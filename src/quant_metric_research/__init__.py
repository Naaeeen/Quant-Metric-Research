"""Leakage-aware cross-sectional metric research."""

from ._version import __version__
from .benchmark import (
    BenchmarkConfig,
    BenchmarkRun,
    NestedSplitConfig,
    run_stage3_benchmark,
)
from .benchmark_comparison import FeatureBundleComparison, compare_feature_bundles
from .benchmark_io import BenchmarkArtifacts, write_benchmark_run
from .config import PanelConfig
from .contracts import DataContractError, validate_memberships, validate_prices
from .experiment_registry import ExperimentRegistry
from .factor_panel import build_factor_panel
from .feature_bundles import FEATURE_BUNDLES, FeatureBundleSpec
from .input_audit import audit_inputs
from .intake import import_yahoo_files
from .panel import build_point_in_time_panel
from .pipeline import ResearchRun, run_research
from .preflight import preflight_benchmark
from .price_factor_catalog import PRICE_FACTOR_CATALOG, PriceFactorSpec
from .price_factors import PriceFactorResult, PriceFactorValue, compute_price_factors
from .reduction import PCABaseline, fit_pca_baseline
from .screening import MetricScreenResult, fit_metric_screen
from .splits import PurgedWalkForwardSplit, build_purged_walk_forward_splits
from .validation import (
    WalkForwardMetricConfig,
    WalkForwardMetricResult,
    evaluate_metrics_walk_forward,
)

__all__ = [
    "__version__",
    "BenchmarkArtifacts",
    "BenchmarkConfig",
    "BenchmarkRun",
    "DataContractError",
    "ExperimentRegistry",
    "FEATURE_BUNDLES",
    "FeatureBundleComparison",
    "FeatureBundleSpec",
    "MetricScreenResult",
    "NestedSplitConfig",
    "PCABaseline",
    "PanelConfig",
    "PRICE_FACTOR_CATALOG",
    "PriceFactorResult",
    "PriceFactorSpec",
    "PriceFactorValue",
    "PurgedWalkForwardSplit",
    "ResearchRun",
    "WalkForwardMetricConfig",
    "WalkForwardMetricResult",
    "audit_inputs",
    "import_yahoo_files",
    "build_point_in_time_panel",
    "build_factor_panel",
    "build_purged_walk_forward_splits",
    "compute_price_factors",
    "compare_feature_bundles",
    "fit_metric_screen",
    "fit_pca_baseline",
    "evaluate_metrics_walk_forward",
    "run_stage3_benchmark",
    "preflight_benchmark",
    "run_research",
    "validate_memberships",
    "validate_prices",
    "write_benchmark_run",
]
