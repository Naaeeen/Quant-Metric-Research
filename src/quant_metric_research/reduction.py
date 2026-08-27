from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _numeric_features(
    panel: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    missing = sorted(set(feature_columns) - set(panel.columns))
    if missing:
        raise ValueError(f"Missing PCA feature columns: {', '.join(missing)}")
    original = panel.loc[:, list(feature_columns)]
    numeric = original.apply(pd.to_numeric, errors="coerce")
    invalid_text = original.notna() & numeric.isna()
    if invalid_text.any().any():
        raise ValueError("PCA features contain non-numeric values.")
    values = numeric.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("PCA features contain infinite values.")
    return numeric


@dataclass(frozen=True)
class PCABaseline:
    feature_columns: tuple[str, ...]
    fitted_through: pd.Timestamp
    components: np.ndarray
    explained_variance_ratio: np.ndarray
    _pipeline: Pipeline = field(repr=False, compare=False)

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        required_keys = {"as_of_date", "symbol"}
        missing_keys = sorted(required_keys - set(panel.columns))
        if missing_keys:
            raise ValueError(f"Missing PCA key columns: {', '.join(missing_keys)}")
        numeric = _numeric_features(panel, self.feature_columns)
        projected = self._pipeline.transform(numeric)
        component_frame = pd.DataFrame(
            projected,
            columns=[f"pc_{position + 1}" for position in range(projected.shape[1])],
        )
        return pd.concat(
            [
                panel.loc[:, ["as_of_date", "symbol"]].reset_index(drop=True),
                component_frame,
            ],
            axis=1,
        )


def fit_pca_baseline(
    panel: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    train_end_date: str | pd.Timestamp,
    variance_to_keep: float,
    as_of_date_column: str = "as_of_date",
) -> PCABaseline:
    if not feature_columns:
        raise ValueError("feature_columns must not be empty.")
    if not 0.0 < variance_to_keep <= 1.0:
        raise ValueError("variance_to_keep must be in (0, 1].")
    if as_of_date_column not in panel.columns:
        raise ValueError(f"Missing PCA date column: {as_of_date_column}")

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
    numeric = _numeric_features(training, feature_columns)
    if numeric.shape[0] < 2:
        raise ValueError("PCA requires at least two training rows.")

    n_components: float | None = variance_to_keep if variance_to_keep < 1.0 else None
    pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            ),
            ("scaler", StandardScaler()),
            (
                "pca",
                PCA(
                    n_components=n_components,
                    svd_solver="full",
                ),
            ),
        ]
    )
    pipeline.fit(numeric)
    pca = pipeline.named_steps["pca"]
    components = np.array(pca.components_, copy=True)
    explained = np.array(pca.explained_variance_ratio_, copy=True)
    components.setflags(write=False)
    explained.setflags(write=False)
    return PCABaseline(
        feature_columns=tuple(feature_columns),
        fitted_through=pd.Timestamp(training[as_of_date_column].max()),
        components=components,
        explained_variance_ratio=explained,
        _pipeline=pipeline,
    )
