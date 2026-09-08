"""Conditional, descriptive comparisons of caller-owned development predictions."""

from __future__ import annotations

import math
import platform
import statistics
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
from pandas.api.types import is_scalar

from ._comparison_contract import ComparisonInputs, validate_comparison_inputs
from ._version import __version__
from .benchmark import BenchmarkRun
from .benchmark_metrics import evaluate_prediction_frame
from .benchmark_reporting import _source_fingerprint

_ARMS = ("legacy_equal_rank", "legacy_model", "candidate_model")
_SCOPES = ("native", "common")
_REFERENCES = ("legacy_model", "legacy_equal_rank")
DESCRIPTIVE_SUMMARY_COLUMNS = (
    "phase",
    "model",
    "evaluation_scope",
    "date_count",
    "spread_date_count",
    "mean_rank_ic",
    "median_rank_ic",
    "rank_ic_std",
    "rank_ic_ir",
    "positive_date_rate",
    "mean_spread",
    "mean_score_coverage",
    "mean_rank_ic_coverage",
    "mean_spread_coverage",
)
_DAILY_DELTA_COLUMNS = (
    "evaluation_scope",
    "contrast",
    "as_of_date",
    "rank_ic_delta",
    "spread_delta",
)
_DELTA_SUMMARY_COLUMNS = (
    "evaluation_scope",
    "contrast",
    "scheduled_date_count",
    "paired_rank_ic_date_count",
    "paired_rank_ic_date_coverage",
    "mean_paired_rank_ic_delta",
    "paired_spread_date_count",
    "paired_spread_date_coverage",
    "mean_paired_spread_delta",
)


def _freeze_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Comparison metadata requires string mapping keys.")
        return MappingProxyType(
            {key: _freeze_metadata(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_metadata(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError("Comparison metadata must contain finite JSON-compatible values.")


@dataclass(frozen=True, slots=True)
class FeatureBundleComparison:
    """Immutable metadata and defensive-copy access to descriptive result tables."""

    metadata: Mapping[str, Any]
    _daily_metrics: pd.DataFrame = field(repr=False, compare=False)
    _fold_metrics: pd.DataFrame = field(repr=False, compare=False)
    _summary: pd.DataFrame = field(repr=False, compare=False)
    _daily_deltas: pd.DataFrame = field(repr=False, compare=False)
    _delta_summary: pd.DataFrame = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, Mapping):
            raise TypeError("Comparison metadata must be a mapping.")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))
        for name in (
            "_daily_metrics",
            "_fold_metrics",
            "_summary",
            "_daily_deltas",
            "_delta_summary",
        ):
            frame = getattr(self, name)
            if not isinstance(frame, pd.DataFrame):
                raise TypeError("Comparison result tables must be pandas DataFrames.")
            if not frame.map(
                lambda value: is_scalar(value) and not isinstance(value, np.void)
            ).all(axis=None):
                raise TypeError("Comparison result tables require scalar cells.")
            object.__setattr__(self, name, frame.copy(deep=True))

    @property
    def daily_metrics(self) -> pd.DataFrame:
        return self._daily_metrics.copy(deep=True)

    @property
    def fold_metrics(self) -> pd.DataFrame:
        return self._fold_metrics.copy(deep=True)

    @property
    def summary(self) -> pd.DataFrame:
        return self._summary.copy(deep=True)

    @property
    def daily_deltas(self) -> pd.DataFrame:
        return self._daily_deltas.copy(deep=True)

    @property
    def delta_summary(self) -> pd.DataFrame:
        return self._delta_summary.copy(deep=True)


def _paired_difference(candidate: pd.Series, reference: pd.Series) -> pd.Series:
    paired = np.isfinite(candidate) & np.isfinite(reference)
    with np.errstate(over="ignore", invalid="ignore"):
        difference = (candidate - reference).where(paired)
    if (paired & ~np.isfinite(difference)).any():
        raise ValueError("A paired daily difference cannot be represented as finite.")
    return difference


def _paired_statistics(values: pd.Series, *, metric: str) -> dict[str, float | int]:
    observed = values.dropna().to_numpy(dtype=float)
    # mean preserves finite-float cancellation without a scaled underflow step.
    mean = statistics.mean(map(float, observed)) if observed.size else float("nan")
    return {
        f"paired_{metric}_date_count": int(observed.size),
        f"paired_{metric}_date_coverage": float(observed.size / len(values)),
        f"mean_paired_{metric}_delta": mean,
    }


def _paired_contrasts(
    daily_metrics: pd.DataFrame, *, expected_dates: tuple[str, ...]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair already-evaluated daily statistics on the complete validated schedule."""
    schedule = pd.DatetimeIndex(expected_dates)
    frames: list[pd.DataFrame] = []
    rows: list[dict[str, object]] = []
    for scope in _SCOPES:
        scoped = daily_metrics.loc[daily_metrics["evaluation_scope"].eq(scope)]
        wide = scoped.pivot(
            index="as_of_date", columns="model", values=["rank_ic", "spread"]
        ).reindex(schedule)
        for reference in _REFERENCES:
            contrast = f"candidate_model_minus_{reference}"
            differences = {
                metric: _paired_difference(
                    wide[(metric, "candidate_model")], wide[(metric, reference)]
                )
                for metric in ("rank_ic", "spread")
            }
            frames.append(
                pd.DataFrame(
                    {
                        "evaluation_scope": scope,
                        "contrast": contrast,
                        "as_of_date": schedule,
                        **{
                            f"{metric}_delta": values.to_numpy()
                            for metric, values in differences.items()
                        },
                    },
                    columns=_DAILY_DELTA_COLUMNS,
                )
            )
            rows.append(
                {
                    "evaluation_scope": scope,
                    "contrast": contrast,
                    "scheduled_date_count": len(schedule),
                    **_paired_statistics(differences["rank_ic"], metric="rank_ic"),
                    **_paired_statistics(differences["spread"], metric="spread"),
                }
            )
    return (
        pd.concat(frames, ignore_index=True),
        pd.DataFrame(rows, columns=_DELTA_SUMMARY_COLUMNS),
    )


def _comparison_metadata(
    inputs: ComparisonInputs, *, model_family: str
) -> Mapping[str, Any]:
    return {
        "claim_scope": "development_descriptive_comparison",
        "comparison_definition_version": "1",
        "caller_owned_inputs": True,
        "authenticity_verified": False,
        "history_verified": False,
        "final_outcomes_evaluated": False,
        "inference_reported": False,
        "model_family": model_family,
        "roles": _ARMS,
        "expected_dates": inputs.expected_dates,
        "metric_settings": {
            "min_cross_section": inputs.config.min_cross_section,
            "quantiles": inputs.config.quantiles,
        },
        "comparison_identity": {
            "package_version": __version__,
            "source_fingerprint": _source_fingerprint(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scipy_version": scipy.__version__,
            "scikit_learn_version": sklearn.__version__,
        },
        "producer_identity": inputs.producer_identity,
        "input_identities": inputs.input_identities,
        "limitations": (
            "Validation covers selected prediction arms, not unrelated saved outputs.",
            "Assignment records lack dates/symbols for omitted excluded rows; their "
            "scoring-universe completeness cannot be certified without the panel.",
            "Caller-owned objects and declared identities are not authenticated.",
            "Native contrasts use model-specific available-score populations.",
            "Common populations intersect finite scores across all three arms and "
            "therefore depend on candidate coverage too.",
            "Legacy equal ranks use the fold-screened, training-oriented legacy-ten "
            "candidate set, not necessarily every original metric.",
            "Development selection is not independent confirmatory evaluation; "
            "these descriptive differences do not establish tradable alpha.",
        ),
    }


def compare_feature_bundles(
    legacy: BenchmarkRun, candidate: BenchmarkRun, *, model_family: str
) -> FeatureBundleComparison:
    """Audit compatible development runs and describe paired saved-score results.

    This neither selects a family nor fits, opens a registry, or loads artifacts.
    Scope/identity/key/outcome checks are conditional on caller-owned evidence.
    Statistical inference is deliberately excluded from the returned tables.
    """
    inputs = validate_comparison_inputs(legacy, candidate, model_family=model_family)
    evaluation = evaluate_prediction_frame(
        inputs.predictions,
        min_cross_section=inputs.config.min_cross_section,
        quantiles=inputs.config.quantiles,
        hac_lags=inputs.config.hac_lags,
        primary_models=_ARMS,
        expected_dates_by_phase={"development": inputs.expected_dates},
    )
    daily_deltas, delta_summary = _paired_contrasts(
        evaluation.daily_metrics, expected_dates=inputs.expected_dates
    )
    return FeatureBundleComparison(
        metadata=_comparison_metadata(inputs, model_family=model_family),
        _daily_metrics=evaluation.daily_metrics,
        _fold_metrics=evaluation.fold_metrics,
        _summary=evaluation.summary.loc[:, list(DESCRIPTIVE_SUMMARY_COLUMNS)],
        _daily_deltas=daily_deltas,
        _delta_summary=delta_summary,
    )


__all__ = ["FeatureBundleComparison", "compare_feature_bundles"]
