from __future__ import annotations

import numpy as np
import pandas as pd

from .statistics import benjamini_hochberg, newey_west_mean_tstat

DAILY_IC_COLUMNS = (
    "as_of_date",
    "feature",
    "rank_ic",
    "cross_section_size",
)
SPREAD_COLUMNS = (
    "as_of_date",
    "feature",
    "spread",
    "cross_section_size",
)
IC_SUMMARY_COLUMNS = (
    "feature",
    "mean_rank_ic",
    "rank_ic_std",
    "positive_rate",
    "date_count",
    "newey_west_t_stat",
    "p_value",
    "bh_q_value",
)


def _validate_panel(
    panel: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    target_column: str,
    as_of_date_column: str,
    min_cross_section: int,
) -> pd.DataFrame:
    if (
        isinstance(min_cross_section, bool)
        or not isinstance(min_cross_section, int)
        or min_cross_section < 2
    ):
        raise ValueError("min_cross_section must be an integer of at least 2.")
    required = {as_of_date_column, target_column, *feature_columns}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"Missing signal columns: {', '.join(missing)}")
    normalized = panel.copy(deep=True)
    normalized[as_of_date_column] = pd.to_datetime(
        normalized[as_of_date_column],
        errors="coerce",
    )
    if normalized[as_of_date_column].isna().any():
        raise ValueError("as_of_date contains invalid values.")
    return normalized


def _paired_numeric(
    group: pd.DataFrame,
    left: str,
    right: str,
) -> pd.DataFrame:
    return (
        group.loc[:, [left, right]]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )


def compute_daily_rank_ic(
    panel: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    target_column: str,
    min_cross_section: int,
    as_of_date_column: str = "as_of_date",
) -> pd.DataFrame:
    normalized = _validate_panel(
        panel,
        feature_columns=feature_columns,
        target_column=target_column,
        as_of_date_column=as_of_date_column,
        min_cross_section=min_cross_section,
    )
    rows: list[dict[str, object]] = []
    for as_of_date, group in normalized.groupby(
        as_of_date_column,
        sort=True,
    ):
        for feature in feature_columns:
            paired = _paired_numeric(group, feature, target_column)
            if paired.shape[0] < min_cross_section:
                continue
            if paired[feature].nunique() <= 1 or paired[target_column].nunique() <= 1:
                continue
            rank_ic = paired[feature].corr(
                paired[target_column],
                method="spearman",
            )
            if pd.notna(rank_ic):
                rows.append(
                    {
                        "as_of_date": pd.Timestamp(as_of_date),
                        "feature": feature,
                        "rank_ic": float(rank_ic),
                        "cross_section_size": int(paired.shape[0]),
                    }
                )
    return pd.DataFrame(rows, columns=DAILY_IC_COLUMNS)


def compute_quantile_spreads(
    panel: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    target_column: str,
    quantiles: int,
    min_cross_section: int,
    as_of_date_column: str = "as_of_date",
) -> pd.DataFrame:
    if isinstance(quantiles, bool) or not isinstance(quantiles, int) or quantiles < 2:
        raise ValueError("quantiles must be an integer of at least 2.")
    normalized = _validate_panel(
        panel,
        feature_columns=feature_columns,
        target_column=target_column,
        as_of_date_column=as_of_date_column,
        min_cross_section=min_cross_section,
    )
    rows: list[dict[str, object]] = []
    for as_of_date, group in normalized.groupby(
        as_of_date_column,
        sort=True,
    ):
        for feature in feature_columns:
            paired = _paired_numeric(group, feature, target_column)
            if (
                paired.shape[0] < min_cross_section
                or paired.shape[0] < quantiles
                or paired[feature].nunique() <= 1
            ):
                continue
            ordered = paired.sort_values(
                feature,
                kind="stable",
            ).reset_index(drop=True)
            bucket_size = max(1, ordered.shape[0] // quantiles)
            bottom = float(ordered.iloc[:bucket_size][target_column].mean())
            top = float(ordered.iloc[-bucket_size:][target_column].mean())
            rows.append(
                {
                    "as_of_date": pd.Timestamp(as_of_date),
                    "feature": feature,
                    "spread": top - bottom,
                    "cross_section_size": int(ordered.shape[0]),
                }
            )
    return pd.DataFrame(rows, columns=SPREAD_COLUMNS)


def summarize_rank_ic(
    daily_rank_ic: pd.DataFrame,
    *,
    hac_lags: int,
) -> pd.DataFrame:
    required = {"feature", "rank_ic"}
    missing = sorted(required - set(daily_rank_ic.columns))
    if missing:
        raise ValueError(f"Missing daily IC columns: {', '.join(missing)}")
    rows: list[dict[str, object]] = []
    for feature, group in daily_rank_ic.groupby("feature", sort=True):
        series = pd.to_numeric(
            group["rank_ic"],
            errors="coerce",
        ).dropna()
        t_stat, p_value = newey_west_mean_tstat(series, hac_lags)
        rows.append(
            {
                "feature": feature,
                "mean_rank_ic": float(series.mean()),
                "rank_ic_std": (
                    float(series.std(ddof=1)) if series.shape[0] > 1 else float("nan")
                ),
                "positive_rate": float((series > 0).mean()),
                "date_count": int(series.shape[0]),
                "newey_west_t_stat": t_stat,
                "p_value": p_value,
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return pd.DataFrame(columns=IC_SUMMARY_COLUMNS)
    summary["bh_q_value"] = benjamini_hochberg(summary["p_value"])
    return summary.loc[:, list(IC_SUMMARY_COLUMNS)]
