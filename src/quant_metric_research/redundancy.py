from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

REDUNDANCY_COLUMNS = (
    "left",
    "right",
    "mean_spearman",
    "mean_abs_spearman",
    "date_count",
)


def compute_feature_redundancy(
    panel: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    min_cross_section: int,
    as_of_date_column: str = "as_of_date",
) -> pd.DataFrame:
    if (
        isinstance(min_cross_section, bool)
        or not isinstance(min_cross_section, int)
        or min_cross_section < 2
    ):
        raise ValueError("min_cross_section must be an integer of at least 2.")
    required = {as_of_date_column, *feature_columns}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"Missing redundancy columns: {', '.join(missing)}")

    normalized = panel.copy(deep=True)
    normalized[as_of_date_column] = pd.to_datetime(
        normalized[as_of_date_column],
        errors="coerce",
    )
    if normalized[as_of_date_column].isna().any():
        raise ValueError("as_of_date contains invalid values.")

    pairs = tuple(combinations(feature_columns, 2))
    # Convert each date once, not the whole panel: numeric precision can depend on
    # the date-local dtype. Keep pair-major accumulation to avoid storing every
    # pair's correlations at once as the feature count grows.
    numeric_groups = tuple(
        (
            group.loc[:, list(dict.fromkeys(feature_columns))]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        for _, group in (
            normalized.groupby(as_of_date_column, sort=True) if pairs else ()
        )
    )
    rows: list[dict[str, object]] = []
    for left, right in pairs:
        correlations: list[float] = []
        for numeric in numeric_groups:
            paired = numeric.loc[:, [left, right]].dropna()
            if paired.shape[0] < min_cross_section:
                continue
            if paired[left].nunique() <= 1 or paired[right].nunique() <= 1:
                continue
            correlation = paired[left].corr(
                paired[right],
                method="spearman",
            )
            if pd.notna(correlation):
                correlations.append(float(correlation))

        if correlations:
            clipped = np.clip(
                np.asarray(correlations, dtype=float),
                -1.0 + 1e-12,
                1.0 - 1e-12,
            )
            mean_spearman = float(np.tanh(np.arctanh(clipped).mean()))
            mean_abs_spearman = float(np.abs(clipped).mean())
        else:
            mean_spearman = float("nan")
            mean_abs_spearman = float("nan")
        rows.append(
            {
                "left": left,
                "right": right,
                "mean_spearman": mean_spearman,
                "mean_abs_spearman": mean_abs_spearman,
                "date_count": len(correlations),
            }
        )
    return pd.DataFrame(rows, columns=REDUNDANCY_COLUMNS)
