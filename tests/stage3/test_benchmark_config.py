from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from quant_metric_research.benchmark_config import (
    BenchmarkConfig,
    NestedSplitConfig,
)


def _split() -> NestedSplitConfig:
    return NestedSplitConfig(
        final_test_date_count=3,
        outer_n_splits=2,
        outer_test_date_count=3,
        outer_min_train_date_count=12,
        inner_n_splits=2,
        inner_validation_date_count=2,
        inner_min_train_date_count=6,
    )


def test_benchmark_config_is_frozen_and_normalizes_feature_names() -> None:
    config = BenchmarkConfig(
        feature_columns=(" momentum ", "value"),
        split=_split(),
        min_cross_section=10,
        hac_lags=1,
    )

    assert config.feature_columns == ("momentum", "value")
    with pytest.raises(FrozenInstanceError):
        config.random_seed = 99  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("final_test_date_count", 0),
        ("outer_n_splits", True),
        ("outer_test_date_count", -1),
        ("inner_n_splits", 0),
    ],
)
def test_nested_split_rejects_invalid_counts(
    field_name: str, bad_value: object
) -> None:
    values = {
        "final_test_date_count": 3,
        "outer_n_splits": 2,
        "outer_test_date_count": 3,
        "outer_min_train_date_count": 12,
        "inner_n_splits": 2,
        "inner_validation_date_count": 2,
        "inner_min_train_date_count": 6,
    }
    values[field_name] = bad_value

    with pytest.raises(ValueError, match=field_name):
        NestedSplitConfig(**values)  # type: ignore[arg-type]


def test_benchmark_config_rejects_duplicate_features_and_bad_grids() -> None:
    with pytest.raises(ValueError, match="unique"):
        BenchmarkConfig(
            feature_columns=("momentum", "momentum"),
            split=_split(),
            min_cross_section=10,
            hac_lags=1,
        )

    with pytest.raises(ValueError, match="ridge_alphas"):
        BenchmarkConfig(
            feature_columns=("momentum",),
            split=_split(),
            min_cross_section=10,
            hac_lags=1,
            ridge_alphas=(0.0,),
        )


def test_benchmark_config_requires_explicit_hac_lags() -> None:
    with pytest.raises(ValueError, match="hac_lags"):
        BenchmarkConfig(
            feature_columns=("momentum",),
            split=_split(),
            min_cross_section=10,
        )


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("minimum_locked_test_date_count", 0),
        ("minimum_locked_score_coverage", 1.1),
        ("minimum_locked_spread_date_count", 0),
        ("minimum_locked_spread_coverage", 0.0),
        ("maximum_locked_rank_ic_improvement_p_value", 0.0),
        ("minimum_rank_ic_improvement", -0.001),
    ],
)
def test_benchmark_config_rejects_invalid_minimum_evidence_thresholds(
    field_name: str,
    bad_value: object,
) -> None:
    values = {
        "feature_columns": ("momentum",),
        "split": _split(),
        "min_cross_section": 10,
        "hac_lags": 1,
        field_name: bad_value,
    }
    with pytest.raises(ValueError, match=field_name):
        BenchmarkConfig(**values)  # type: ignore[arg-type]
