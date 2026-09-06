from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import product
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .benchmark_config import BenchmarkConfig
from .benchmark_data import equal_date_training_weights


@dataclass(frozen=True)
class ModelCandidate:
    family: str
    candidate_id: str
    parameters: MappingProxyType[str, Any]


@dataclass(frozen=True)
class FittedCandidate:
    family: str
    candidate_id: str
    feature_columns: tuple[str, ...]
    fitted_through: pd.Timestamp
    train_label_end_max: pd.Timestamp
    rank_features: bool
    model_parameters: MappingProxyType[str, Any]
    _pipeline: Pipeline = field(repr=False, compare=False)

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        features = _feature_frame(
            frame,
            self.feature_columns,
            rank_features=self.rank_features,
        )
        scores = self._pipeline.predict(features)
        result = pd.Series(scores, index=frame.index, name="score", dtype="float64")
        return result.where(features.notna().any(axis=1))


def _candidate(family: str, parameters: dict[str, Any]) -> ModelCandidate:
    encoded = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
    return ModelCandidate(
        family=family,
        candidate_id=f"{family}:{encoded}",
        parameters=MappingProxyType(dict(parameters)),
    )


def candidate_specs(
    config: BenchmarkConfig,
    *,
    family: str,
) -> tuple[ModelCandidate, ...]:
    if family == "ridge":
        return tuple(
            _candidate(family, {"alpha": alpha}) for alpha in config.ridge_alphas
        )
    if family == "hist_gradient_boosting":
        return tuple(
            _candidate(
                family,
                {
                    "l2_regularization": l2_value,
                    "learning_rate": learning_rate,
                    "max_leaf_nodes": max_leaf_nodes,
                },
            )
            for learning_rate, max_leaf_nodes, l2_value in product(
                config.hist_learning_rates,
                config.hist_max_leaf_nodes,
                config.hist_l2_regularization,
            )
        )
    if family == "ridge_pca":
        return tuple(
            _candidate(
                family,
                {"alpha": alpha, "variance_to_keep": variance},
            )
            for alpha, variance in product(
                config.ridge_alphas,
                config.pca_variance_to_keep,
            )
        )
    raise ValueError(f"Unsupported model family: {family}")


def _feature_frame(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    *,
    rank_features: bool,
    as_of_date_column: str = "as_of_date",
) -> pd.DataFrame:
    if not feature_columns:
        raise ValueError("feature_columns must not be empty.")
    required = {as_of_date_column, *feature_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing model columns: {', '.join(missing)}")
    dates = pd.to_datetime(frame[as_of_date_column], errors="coerce")
    if dates.isna().any():
        raise ValueError("Model as_of_date contains invalid values.")
    original = frame.loc[:, list(feature_columns)]
    numeric = original.apply(pd.to_numeric, errors="coerce")
    if (original.notna() & numeric.isna()).any().any():
        raise ValueError("Model features contain non-numeric values.")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("Model features contain infinite values.")
    if not rank_features:
        return numeric
    ranked = numeric.groupby(dates, sort=False).rank(method="average", pct=True)
    return ranked.loc[:, list(feature_columns)]


def _build_pipeline(
    candidate: ModelCandidate,
    *,
    random_seed: int,
    hist_max_iter: int,
    hist_min_samples_leaf: int,
) -> Pipeline:
    parameters = dict(candidate.parameters)
    if candidate.family in {"ridge", "ridge_pca"}:
        steps: list[tuple[str, Any]] = [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
        if candidate.family == "ridge_pca":
            variance = float(parameters["variance_to_keep"])
            steps.append(
                (
                    "pca",
                    PCA(
                        n_components=variance if variance < 1.0 else None,
                        svd_solver="full",
                    ),
                )
            )
        steps.append(("model", Ridge(alpha=float(parameters["alpha"]))))
        return Pipeline(steps)
    if candidate.family == "hist_gradient_boosting":
        model = HistGradientBoostingRegressor(
            learning_rate=float(parameters["learning_rate"]),
            max_leaf_nodes=int(parameters["max_leaf_nodes"]),
            l2_regularization=float(parameters["l2_regularization"]),
            max_iter=hist_max_iter,
            min_samples_leaf=hist_min_samples_leaf,
            early_stopping=False,
            random_state=random_seed,
        )
        return Pipeline([("model", model)])
    raise ValueError(f"Unsupported model family: {candidate.family}")


def fit_candidate(
    training: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    target_column: str,
    candidate: ModelCandidate,
    rank_features: bool,
    random_seed: int,
    hist_max_iter: int = 200,
    hist_min_samples_leaf: int = 20,
    as_of_date_column: str = "as_of_date",
    label_end_date_column: str = "label_end_date",
) -> FittedCandidate:
    required = {as_of_date_column, label_end_date_column, target_column}
    missing = sorted(required - set(training.columns))
    if missing:
        raise ValueError(f"Missing training columns: {', '.join(missing)}")
    normalized = training.copy(deep=True)
    normalized[as_of_date_column] = pd.to_datetime(
        normalized[as_of_date_column], errors="coerce"
    )
    normalized[label_end_date_column] = pd.to_datetime(
        normalized[label_end_date_column], errors="coerce"
    )
    original_target = normalized[target_column]
    normalized[target_column] = pd.to_numeric(original_target, errors="coerce")
    if (original_target.notna() & normalized[target_column].isna()).any():
        raise ValueError("Training target contains non-numeric values.")
    # Rank the complete contemporaneous universe before selecting the labeled
    # rows that contribute to the supervised loss.
    features = _feature_frame(
        normalized,
        feature_columns,
        rank_features=rank_features,
        as_of_date_column=as_of_date_column,
    )
    labeled = (
        normalized[as_of_date_column].notna()
        & normalized[label_end_date_column].notna()
        & normalized[target_column].notna()
    )
    features = features.loc[labeled]
    normalized = normalized.loc[labeled].copy(deep=True)
    if normalized.empty:
        raise ValueError("No valid labeled training rows remain.")
    target_values = normalized[target_column].to_numpy(dtype=float)
    if not np.isfinite(target_values).all():
        raise ValueError("Training target must contain finite values.")

    raw_weights = np.asarray(
        equal_date_training_weights(normalized, date_column=as_of_date_column),
        dtype=float,
    )
    weights = raw_weights * (raw_weights.shape[0] / raw_weights.sum())
    pipeline = _build_pipeline(
        candidate,
        random_seed=random_seed,
        hist_max_iter=hist_max_iter,
        hist_min_samples_leaf=hist_min_samples_leaf,
    )
    pipeline.fit(features, target_values, model__sample_weight=weights)
    model = pipeline.named_steps["model"]
    return FittedCandidate(
        family=candidate.family,
        candidate_id=candidate.candidate_id,
        feature_columns=tuple(feature_columns),
        fitted_through=pd.Timestamp(normalized[as_of_date_column].max()),
        train_label_end_max=pd.Timestamp(normalized[label_end_date_column].max()),
        rank_features=rank_features,
        model_parameters=MappingProxyType(dict(model.get_params(deep=False))),
        _pipeline=pipeline,
    )
