from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import pandas as pd

from .quality import compute_feature_quality
from .redundancy import compute_feature_redundancy
from .signals import (
    SPREAD_COLUMNS,
    compute_daily_rank_ic,
    compute_quantile_spreads,
    summarize_rank_ic,
)


@dataclass(frozen=True)
class MetricScreenResult:
    quality: pd.DataFrame
    daily_rank_ic: pd.DataFrame
    ic_summary: pd.DataFrame
    quantile_spreads: pd.DataFrame
    redundancy: pd.DataFrame
    selected_features: tuple[str, ...]
    dropped_features: Mapping[str, str]
    fitted_through: pd.Timestamp
    training_row_count: int


def fit_metric_screen(
    panel: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    target_column: str,
    train_end_date: str | pd.Timestamp,
    min_cross_section: int,
    minimum_coverage: float,
    redundancy_threshold: float,
    hac_lags: int,
    quantiles: int = 5,
    as_of_date_column: str = "as_of_date",
    include_quantile_spreads: bool = True,
) -> MetricScreenResult:
    """Fit selection and diagnostics; omitted spreads retain their empty schema."""
    if not isinstance(include_quantile_spreads, bool):
        raise ValueError("include_quantile_spreads must be a boolean.")
    if not 0.0 <= minimum_coverage <= 1.0:
        raise ValueError("minimum_coverage must be between zero and one.")
    if not 0.0 <= redundancy_threshold <= 1.0:
        raise ValueError("redundancy_threshold must be between zero and one.")
    required = {as_of_date_column, target_column, *feature_columns}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"Missing screening columns: {', '.join(missing)}")

    normalized = panel.copy(deep=True)
    normalized[as_of_date_column] = pd.to_datetime(
        normalized[as_of_date_column],
        errors="coerce",
    )
    if normalized[as_of_date_column].isna().any():
        raise ValueError("as_of_date contains invalid values.")
    cutoff = pd.Timestamp(train_end_date)
    training = normalized.loc[normalized[as_of_date_column] <= cutoff].copy(deep=True)
    if training.empty:
        raise ValueError("training panel is empty for the requested train_end_date.")

    quality = compute_feature_quality(
        training,
        feature_columns=feature_columns,
        as_of_date_column=as_of_date_column,
    )
    daily_rank_ic = compute_daily_rank_ic(
        training,
        feature_columns=feature_columns,
        target_column=target_column,
        min_cross_section=min_cross_section,
        as_of_date_column=as_of_date_column,
    )
    ic_summary = summarize_rank_ic(daily_rank_ic, hac_lags=hac_lags)
    if isinstance(quantiles, bool) or not isinstance(quantiles, int) or quantiles < 2:
        raise ValueError("quantiles must be an integer of at least 2.")
    quantile_spreads = (
        compute_quantile_spreads(
            training,
            feature_columns=feature_columns,
            target_column=target_column,
            quantiles=quantiles,
            min_cross_section=min_cross_section,
            as_of_date_column=as_of_date_column,
        )
        if include_quantile_spreads
        else pd.DataFrame(columns=SPREAD_COLUMNS)
    )
    redundancy = compute_feature_redundancy(
        training,
        feature_columns=feature_columns,
        min_cross_section=min_cross_section,
        as_of_date_column=as_of_date_column,
    )

    quality_by_feature = quality.set_index("feature")
    ic_by_feature = ic_summary.set_index("feature")
    pair_redundancy: dict[frozenset[str], float] = {}
    for row in redundancy.itertuples(index=False):
        if pd.notna(row.mean_abs_spearman):
            pair_redundancy[frozenset((row.left, row.right))] = float(
                row.mean_abs_spearman
            )

    def ordering_key(feature: str) -> tuple[float, float, str]:
        mean_ic = (
            float(ic_by_feature.at[feature, "mean_rank_ic"])
            if feature in ic_by_feature.index
            else float("nan")
        )
        signal_strength = abs(mean_ic) if pd.notna(mean_ic) else -1.0
        coverage = float(quality_by_feature.at[feature, "overall_coverage"])
        return (-signal_strength, -coverage, feature)

    selected: list[str] = []
    dropped: dict[str, str] = {}
    for feature in sorted(feature_columns, key=ordering_key):
        quality_row = quality_by_feature.loc[feature]
        if bool(quality_row["zero_variance"]):
            dropped[feature] = "zero_variance"
            continue
        if float(quality_row["overall_coverage"]) < minimum_coverage:
            dropped[feature] = "low_coverage"
            continue
        if feature not in ic_by_feature.index:
            dropped[feature] = "insufficient_rank_ic"
            continue

        blocker = next(
            (
                kept
                for kept in selected
                if pair_redundancy.get(
                    frozenset((feature, kept)),
                    -1.0,
                )
                >= redundancy_threshold
            ),
            None,
        )
        if blocker is not None:
            dropped[feature] = f"redundant_with:{blocker}"
            continue
        selected.append(feature)

    return MetricScreenResult(
        quality=quality,
        daily_rank_ic=daily_rank_ic,
        ic_summary=ic_summary,
        quantile_spreads=quantile_spreads,
        redundancy=redundancy,
        selected_features=tuple(selected),
        dropped_features=MappingProxyType(dict(dropped)),
        fitted_through=pd.Timestamp(training[as_of_date_column].max()),
        training_row_count=int(training.shape[0]),
    )
