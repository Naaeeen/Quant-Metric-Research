"""Leakage-aware cross-sectional metric research."""

from .benchmark import (
    BenchmarkConfig,
    BenchmarkRun,
    NestedSplitConfig,
    run_stage3_benchmark,
)
from .benchmark_io import BenchmarkArtifacts, write_benchmark_run
from .config import PanelConfig
from .contracts import DataContractError, validate_memberships, validate_prices
from .panel import build_point_in_time_panel
from .pipeline import ResearchRun, run_research
from .reduction import PCABaseline, fit_pca_baseline
from .screening import MetricScreenResult, fit_metric_screen
from .splits import PurgedWalkForwardSplit, build_purged_walk_forward_splits
from .validation import (
    WalkForwardMetricConfig,
    WalkForwardMetricResult,
    evaluate_metrics_walk_forward,
)

__all__ = [
    "BenchmarkArtifacts",
    "BenchmarkConfig",
    "BenchmarkRun",
    "DataContractError",
    "MetricScreenResult",
    "NestedSplitConfig",
    "PCABaseline",
    "PanelConfig",
    "PurgedWalkForwardSplit",
    "ResearchRun",
    "WalkForwardMetricConfig",
    "WalkForwardMetricResult",
    "build_point_in_time_panel",
    "build_purged_walk_forward_splits",
    "fit_metric_screen",
    "fit_pca_baseline",
    "evaluate_metrics_walk_forward",
    "run_stage3_benchmark",
    "run_research",
    "validate_memberships",
    "validate_prices",
    "write_benchmark_run",
]
