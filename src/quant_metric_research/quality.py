from __future__ import annotations

import numpy as np
import pandas as pd


def compute_feature_quality(
    panel: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    as_of_date_column: str = "as_of_date",
) -> pd.DataFrame:
    if not isinstance(panel, pd.DataFrame):
        raise ValueError("panel must be a pandas DataFrame.")
    if not feature_columns:
        raise ValueError("feature_columns must not be empty.")
    missing = sorted({as_of_date_column, *feature_columns} - set(panel.columns))
    if missing:
        raise ValueError(f"Missing quality columns: {', '.join(missing)}")

    normalized = panel.copy(deep=True)
    normalized[as_of_date_column] = pd.to_datetime(
        normalized[as_of_date_column],
        errors="coerce",
    )
    if normalized[as_of_date_column].isna().any():
        raise ValueError("as_of_date contains invalid values.")

    rows: list[dict[str, object]] = []
    for feature in feature_columns:
        numeric = pd.to_numeric(normalized[feature], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        per_date = (
            pd.DataFrame(
                {
                    "date": normalized[as_of_date_column],
                    "present": numeric.notna(),
                }
            )
            .groupby("date", sort=True)["present"]
            .mean()
        )
        rows.append(
            {
                "feature": feature,
                "row_count": int(numeric.shape[0]),
                "non_missing_count": int(numeric.notna().sum()),
                "overall_coverage": float(numeric.notna().mean()),
                "mean_date_coverage": (
                    float(per_date.mean()) if not per_date.empty else 0.0
                ),
                "minimum_date_coverage": (
                    float(per_date.min()) if not per_date.empty else 0.0
                ),
                "unique_count": int(numeric.dropna().nunique()),
                "zero_variance": bool(numeric.dropna().nunique() <= 1),
            }
        )
    return pd.DataFrame(rows)
