from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any


def _positive_integer(value: int, name: str, *, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}.")


def _finite_grid(
    values: tuple[float, ...],
    name: str,
    *,
    minimum: float,
    include_minimum: bool,
) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in values)
    if not normalized:
        raise ValueError(f"{name} must not be empty.")
    valid = all(
        isfinite(value) and (value >= minimum if include_minimum else value > minimum)
        for value in normalized
    )
    if not valid or len(set(normalized)) != len(normalized):
        comparator = ">=" if include_minimum else ">"
        raise ValueError(
            f"{name} must contain unique finite values {comparator} {minimum}."
        )
    return normalized


@dataclass(frozen=True)
class NestedSplitConfig:
    final_test_date_count: int
    outer_n_splits: int
    outer_test_date_count: int
    outer_min_train_date_count: int
    inner_n_splits: int
    inner_validation_date_count: int
    inner_min_train_date_count: int

    def __post_init__(self) -> None:
        for field_name in (
            "final_test_date_count",
            "outer_n_splits",
            "outer_test_date_count",
            "outer_min_train_date_count",
            "inner_n_splits",
            "inner_validation_date_count",
            "inner_min_train_date_count",
        ):
            _positive_integer(getattr(self, field_name), field_name)


@dataclass(frozen=True)
class BenchmarkConfig:
    feature_columns: tuple[str, ...]
    split: NestedSplitConfig
    target_column: str = "forward_excess_return"
    realized_return_column: str = "forward_excess_return"
    min_cross_section: int = 50
    minimum_coverage: float = 0.8
    redundancy_threshold: float = 0.9
    quantiles: int = 5
    hac_lags: int | None = None
    ridge_alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    hist_learning_rates: tuple[float, ...] = (0.05,)
    hist_max_leaf_nodes: tuple[int, ...] = (7,)
    hist_l2_regularization: tuple[float, ...] = (1.0,)
    hist_max_iter: int = 200
    hist_min_samples_leaf: int = 20
    include_pca_model: bool = False
    pca_variance_to_keep: tuple[float, ...] = (0.95,)
    rank_features: bool = True
    random_seed: int = 42
    minimum_rank_ic_improvement: float = 0.0
    minimum_development_win_rate: float = 0.5
    minimum_coverage_ratio: float = 0.95
    minimum_locked_test_date_count: int = 20
    minimum_locked_score_coverage: float = 0.8
    minimum_locked_spread_date_count: int = 20
    minimum_locked_spread_coverage: float = 0.8
    maximum_locked_rank_ic_improvement_p_value: float = 0.05
    primary_baseline: str = "equal_weight_rank"
    model_families: tuple[str, ...] = field(
        default_factory=lambda: ("ridge", "hist_gradient_boosting")
    )

    def __post_init__(self) -> None:
        if not isinstance(self.split, NestedSplitConfig):
            raise ValueError("split must be a NestedSplitConfig.")
        features = tuple(str(name).strip() for name in self.feature_columns)
        if not features or any(not name for name in features):
            raise ValueError("feature_columns must contain non-empty names.")
        if len(set(features)) != len(features):
            raise ValueError("feature_columns must be unique.")

        target = str(self.target_column).strip()
        realized = str(self.realized_return_column).strip()
        if not target or not realized:
            raise ValueError("target_column and realized_return_column are required.")
        if target in features or realized in features:
            raise ValueError("Target columns cannot also be feature columns.")

        _positive_integer(self.min_cross_section, "min_cross_section", minimum=2)
        _positive_integer(self.quantiles, "quantiles", minimum=2)
        if self.quantiles > self.min_cross_section:
            raise ValueError("quantiles cannot exceed min_cross_section.")
        if isinstance(self.hac_lags, bool) or not isinstance(self.hac_lags, int):
            raise ValueError("hac_lags must be a non-negative integer.")
        if self.hac_lags < 0:
            raise ValueError("hac_lags must be a non-negative integer.")
        _positive_integer(self.hist_max_iter, "hist_max_iter")
        _positive_integer(self.hist_min_samples_leaf, "hist_min_samples_leaf")
        _positive_integer(
            self.minimum_locked_test_date_count,
            "minimum_locked_test_date_count",
        )
        _positive_integer(
            self.minimum_locked_spread_date_count,
            "minimum_locked_spread_date_count",
        )
        if (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
            or self.random_seed < 0
        ):
            raise ValueError("random_seed must be a non-negative integer.")
        for name in ("include_pca_model", "rank_features"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean.")

        unit_values = {
            "minimum_coverage": self.minimum_coverage,
            "redundancy_threshold": self.redundancy_threshold,
            "minimum_development_win_rate": self.minimum_development_win_rate,
            "minimum_coverage_ratio": self.minimum_coverage_ratio,
            "minimum_locked_score_coverage": self.minimum_locked_score_coverage,
        }
        for name, value in unit_values.items():
            if not isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be between zero and one.")
        if (
            not isfinite(float(self.minimum_locked_spread_coverage))
            or not 0.0 < float(self.minimum_locked_spread_coverage) <= 1.0
        ):
            raise ValueError("minimum_locked_spread_coverage must be in (0, 1].")
        if (
            not isfinite(float(self.maximum_locked_rank_ic_improvement_p_value))
            or not 0.0 < float(self.maximum_locked_rank_ic_improvement_p_value) <= 1.0
        ):
            raise ValueError(
                "maximum_locked_rank_ic_improvement_p_value must be in (0, 1]."
            )
        if (
            not isfinite(float(self.minimum_rank_ic_improvement))
            or float(self.minimum_rank_ic_improvement) < 0.0
        ):
            raise ValueError(
                "minimum_rank_ic_improvement must be finite and non-negative."
            )

        alphas = _finite_grid(
            self.ridge_alphas,
            "ridge_alphas",
            minimum=0.0,
            include_minimum=False,
        )
        learning_rates = _finite_grid(
            self.hist_learning_rates,
            "hist_learning_rates",
            minimum=0.0,
            include_minimum=False,
        )
        l2_values = _finite_grid(
            self.hist_l2_regularization,
            "hist_l2_regularization",
            minimum=0.0,
            include_minimum=True,
        )
        pca_values = _finite_grid(
            self.pca_variance_to_keep,
            "pca_variance_to_keep",
            minimum=0.0,
            include_minimum=False,
        )
        if any(value > 1.0 for value in pca_values):
            raise ValueError("pca_variance_to_keep values cannot exceed one.")

        leaf_nodes = tuple(self.hist_max_leaf_nodes)
        if not leaf_nodes or len(set(leaf_nodes)) != len(leaf_nodes):
            raise ValueError("hist_max_leaf_nodes must be non-empty and unique.")
        for value in leaf_nodes:
            _positive_integer(value, "hist_max_leaf_nodes", minimum=2)

        families = tuple(str(name).strip() for name in self.model_families)
        supported = {"ridge", "hist_gradient_boosting", "ridge_pca"}
        if not families or len(set(families)) != len(families):
            raise ValueError("model_families must be non-empty and unique.")
        unknown = sorted(set(families) - supported)
        if unknown:
            raise ValueError(f"Unsupported model families: {', '.join(unknown)}")
        if self.include_pca_model and "ridge_pca" not in families:
            families = (*families, "ridge_pca")
        primary_baseline = str(self.primary_baseline).strip()
        if primary_baseline not in {"best_metric", "equal_weight_rank"}:
            raise ValueError(
                "primary_baseline must be best_metric or equal_weight_rank."
            )

        object.__setattr__(self, "feature_columns", features)
        object.__setattr__(self, "target_column", target)
        object.__setattr__(self, "realized_return_column", realized)
        object.__setattr__(self, "ridge_alphas", alphas)
        object.__setattr__(self, "hist_learning_rates", learning_rates)
        object.__setattr__(self, "hist_l2_regularization", l2_values)
        object.__setattr__(self, "hist_max_leaf_nodes", leaf_nodes)
        object.__setattr__(self, "pca_variance_to_keep", pca_values)
        object.__setattr__(self, "model_families", families)
        object.__setattr__(self, "primary_baseline", primary_baseline)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> BenchmarkConfig:
        copied = dict(value)
        split_value = copied.get("split")
        if not isinstance(split_value, Mapping):
            raise ValueError("Benchmark configuration requires a split object.")
        copied["split"] = NestedSplitConfig(**dict(split_value))
        return cls(**copied)

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)
