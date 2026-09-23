from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ._spreads import fractional_quantile_spread
from .contracts import _daily_dates
from .scheduled_inference import (
    HACMeanResult,
    inference_diagnostics,
    normalize_expected_dates,
    scheduled_newey_west_mean,
)
from .statistics import benjamini_hochberg

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
    "inference_status",
    "scheduled_date_count",
    "observed_date_count",
    "requested_hac_lags",
    "effective_hac_lags",
    "hac_lag_unit",
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
            spread = fractional_quantile_spread(
                paired[feature], paired[target_column], quantiles=quantiles
            )
            rows.append(
                {
                    "as_of_date": pd.Timestamp(as_of_date),
                    "feature": feature,
                    "spread": spread,
                    "cross_section_size": int(paired.shape[0]),
                }
            )
    return pd.DataFrame(rows, columns=SPREAD_COLUMNS)


def _summary_inputs(
    daily_rank_ic: pd.DataFrame,
    *,
    hac_lags: int,
    expected_dates: Sequence[object] | None,
) -> tuple[pd.DataFrame, pd.DatetimeIndex | None]:
    if not isinstance(daily_rank_ic, pd.DataFrame):
        raise ValueError("daily_rank_ic must be a pandas DataFrame.")
    if not daily_rank_ic.columns.is_unique:
        raise ValueError("Duplicate daily IC columns are not allowed.")
    if isinstance(hac_lags, bool) or not isinstance(hac_lags, int) or hac_lags < 0:
        raise ValueError("hac_lags must be a non-negative integer.")
    schedule = (
        normalize_expected_dates(expected_dates) if expected_dates is not None else None
    )
    required = {"feature", "rank_ic"}
    if schedule is not None:
        required.add("as_of_date")
    missing = sorted(required - set(daily_rank_ic.columns))
    if missing:
        raise ValueError(f"Missing daily IC columns: {', '.join(missing)}")
    if daily_rank_ic["feature"].isna().any():
        raise ValueError("feature contains null values.")
    normalized = daily_rank_ic.copy(deep=True)
    if "as_of_date" in normalized.columns:
        normalized["as_of_date"] = _daily_dates(
            normalized["as_of_date"], field="as_of_date"
        )
        if normalized.duplicated(["feature", "as_of_date"], keep=False).any():
            raise ValueError("Daily IC values must be unique by feature/as_of_date.")
    return normalized, schedule


def _summary_inference(
    group: pd.DataFrame,
    *,
    schedule: pd.DatetimeIndex | None,
    hac_lags: int,
    observed_count: int,
) -> HACMeanResult:
    if schedule is None:
        return HACMeanResult(
            status="schedule_unavailable",
            t_stat=None,
            p_value=None,
            scheduled_count=None,
            observed_count=observed_count,
            requested_lags=hac_lags,
            effective_lags=None,
        )
    # Keep original scalar inputs so malformed ICs cannot become missing values
    # through the descriptive summary's permissive numeric conversion.
    return scheduled_newey_west_mean(
        group["rank_ic"].set_axis(group["as_of_date"]),
        expected_dates=schedule,
        hac_lags=hac_lags,
    )


def summarize_rank_ic(
    daily_rank_ic: pd.DataFrame,
    *,
    hac_lags: int,
    expected_dates: Sequence[object] | None = None,
) -> pd.DataFrame:
    """Keep sparse IC descriptives; infer only on a complete supplied schedule.

    The schedule is declared observation dates, not a verified exchange calendar.
    Omitting it preserves descriptive support, including two-column inputs, but
    does not infer a schedule from surviving feature/date rows.
    """
    normalized, schedule = _summary_inputs(
        daily_rank_ic, hac_lags=hac_lags, expected_dates=expected_dates
    )
    rows: list[dict[str, object]] = []
    for feature, group in normalized.groupby("feature", sort=True):
        series = pd.to_numeric(
            group["rank_ic"],
            errors="coerce",
        ).dropna()
        inference = _summary_inference(
            group,
            schedule=schedule,
            hac_lags=hac_lags,
            observed_count=int(series.replace([np.inf, -np.inf], np.nan).count()),
        )
        rows.append(
            {
                "feature": feature,
                "mean_rank_ic": float(series.mean()),
                "rank_ic_std": (
                    float(series.std(ddof=1)) if series.shape[0] > 1 else float("nan")
                ),
                "positive_rate": float((series > 0).mean()),
                "date_count": int(series.shape[0]),
                "newey_west_t_stat": inference.t_stat,
                "p_value": inference.p_value,
                **inference_diagnostics(inference),
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return pd.DataFrame(columns=IC_SUMMARY_COLUMNS)
    summary["bh_q_value"] = benjamini_hochberg(summary["p_value"])
    return summary.loc[:, list(IC_SUMMARY_COLUMNS)]
