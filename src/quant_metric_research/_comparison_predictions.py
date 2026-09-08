"""Assignment and selected-score validation without hidden row intersections."""

from __future__ import annotations

import json
from collections.abc import Mapping
from decimal import Decimal
from numbers import Integral

import numpy as np
import pandas as pd

from .benchmark_config import BenchmarkConfig
from .benchmark_models import candidate_specs
from .contracts import _daily_dates
from .statistics import _numeric_observations

_BOUNDS = (
    "train_end_date",
    "train_label_end_max",
    "evaluation_start",
    "evaluation_end",
)
_ASSIGNMENTS = (
    "phase",
    "fold",
    "split_id",
    "row_id",
    "role",
    "exclusion_reason",
    *_BOUNDS,
)
_KEYS = ("fold", "as_of_date", "symbol")
_STRUCTURE = (
    "phase",
    "fold",
    "as_of_date",
    "symbol",
    "row_id",
    "model",
    "candidate_id",
    "fit_end_date",
    "train_label_end_max",
    "evaluation_start",
    "selected_features",
    "feature_count",
    "selected_feature_count",
    "zero_observed_features",
)
_OUTCOMES = ("score", "target", "realized_return")


def _require(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{name} must be a nonempty DataFrame.")
    if not frame.columns.is_unique or not set(columns).issubset(frame.columns):
        raise ValueError(f"{name} has missing or duplicate columns.")


def _text(values: pd.Series, name: str, *, nullable: bool = False) -> None:
    for value in values:
        if nullable and pd.api.types.is_scalar(value) and pd.isna(value):
            continue
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"{name} requires nonempty normalized strings.")


def _integers(values: pd.Series, name: str, *, minimum: int = 0) -> None:
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value < minimum
        for value in values
    ):
        raise ValueError(f"{name} requires integer values of at least {minimum}.")


def validate_phase(frame: pd.DataFrame, name: str) -> None:
    _require(frame, ("phase",), name)
    _text(frame["phase"], f"{name} phase")
    if not frame["phase"].eq("development").all():
        raise ValueError(f"{name} must contain only development phase rows.")


def _assignment_frame(frame: pd.DataFrame) -> pd.DataFrame:
    _require(frame, _ASSIGNMENTS, "assignments")
    if set(frame.columns) != set(_ASSIGNMENTS):
        raise ValueError("Unsupported assignment columns.")
    normalized = frame.loc[:, list(_ASSIGNMENTS)].copy(deep=True)
    _integers(normalized["fold"], "assignment fold", minimum=1)
    for name in ("split_id", "row_id", "role"):
        _text(normalized[name], f"assignment {name}")
    _text(normalized["exclusion_reason"], "exclusion_reason", nullable=True)
    if normalized.duplicated(["fold", "row_id"]).any():
        raise ValueError("Assignment fold/row_id keys must be unique.")
    if not normalized["role"].isin(("training", "evaluation", "excluded")).all():
        raise ValueError("Unknown assignment role.")
    excluded = normalized["role"].eq("excluded")
    if (excluded != normalized["exclusion_reason"].notna()).any():
        raise ValueError("Only excluded assignments must carry exclusion reasons.")
    for name in _BOUNDS:
        normalized[name] = _daily_dates(normalized[name], field=name)
    return normalized.sort_values(["fold", "row_id"], kind="stable").reset_index(
        drop=True
    )


def validate_assignments(
    legacy: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    config: BenchmarkConfig,
    expected_dates: tuple[str, ...],
    identity: Mapping,
) -> pd.DataFrame:
    left, right = _assignment_frame(legacy), _assignment_frame(candidate)
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=True)
    except AssertionError as error:
        raise ValueError("Complete development assignments must match.") from error
    expected_folds = set(range(1, config.split.outer_n_splits + 1))
    if set(left["fold"]) != expected_folds:
        raise ValueError("Assignments do not contain the configured fold count.")
    previous_end = None
    schedule = pd.DatetimeIndex(expected_dates)
    row_ids = set(left.loc[left["fold"].eq(1), "row_id"])
    for fold, group in left.groupby("fold", sort=True):
        if (
            not group["split_id"].eq(f"development_fold_{fold - 1}").all()
            or set(group["row_id"]) != row_ids
        ):
            raise ValueError(
                "Assignment split IDs or complete per-fold row sets disagree."
            )
        if any(group[name].nunique() != 1 for name in _BOUNDS):
            raise ValueError(
                "Assignment fold boundaries must be internally consistent."
            )
        bound = group.iloc[0]
        start, end = bound["evaluation_start"], bound["evaluation_end"]
        if (
            start > end
            or bound["train_end_date"] < pd.Timestamp(identity["development_start"])
            or bound["train_end_date"] >= bound["train_label_end_max"]
            or bound["train_label_end_max"] >= start
            or start < pd.Timestamp(identity["development_start"])
            or end >= pd.Timestamp(identity["locked_test_start"])
            or (previous_end is not None and start <= previous_end)
        ):
            raise ValueError("Assignment temporal boundaries are invalid or overlap.")
        if start not in schedule or end not in schedule:
            raise ValueError("Assignment endpoints must occur in the stored schedule.")
        previous_end = end
    if (
        schedule[0] != left["evaluation_start"].min()
        or schedule[-1] != left["evaluation_end"].max()
    ):
        raise ValueError("Schedule must span exactly the planned evaluation bounds.")
    return left


def _selected_structure(
    source: pd.DataFrame, model: str
) -> tuple[pd.DataFrame, np.ndarray]:
    _require(source, (*_STRUCTURE, *_OUTCOMES), "predictions")
    positions = np.flatnonzero(source["model"].eq(model).to_numpy())
    if not len(positions):
        raise ValueError(f"Missing selected prediction model {model}.")
    # Outcome columns are deliberately not accessed until every arm's structure
    # and temporal scope has passed validation.
    frame = (
        source.loc[:, list(_STRUCTURE)]
        .iloc[positions]
        .copy(deep=True)
        .reset_index(drop=True)
    )
    for name in ("phase", "symbol", "row_id", "model"):
        _text(frame[name], f"prediction {name}")
    _integers(frame["fold"], "prediction fold", minimum=1)
    for name in (
        "as_of_date",
        "fit_end_date",
        "train_label_end_max",
        "evaluation_start",
    ):
        frame[name] = _daily_dates(frame[name], field=name)
    if frame.duplicated(list(_KEYS)).any() or not frame["row_id"].is_unique:
        raise ValueError("Selected prediction keys and row IDs must be unique.")
    return frame, positions


def _fit_and_schedule(
    frame: pd.DataFrame, assignments: pd.DataFrame, dates: tuple[str, ...]
) -> None:
    schedule = pd.DatetimeIndex(dates)
    if set(frame["fold"]) != set(assignments["fold"]):
        raise ValueError("Every selected arm must contain all assigned folds.")
    for fold, group in frame.groupby("fold", sort=True):
        evidence = assignments.loc[assignments["fold"].eq(fold)]
        bounds = evidence.iloc[0]
        expected = schedule[
            (schedule >= bounds["evaluation_start"])
            & (schedule <= bounds["evaluation_end"])
        ]
        if set(group["as_of_date"]) != set(expected):
            raise ValueError(
                "Prediction dates must exactly match their fold's scheduled window."
            )
        for actual, reference in (
            ("fit_end_date", "train_end_date"),
            ("train_label_end_max", "train_label_end_max"),
            ("evaluation_start", "evaluation_start"),
        ):
            if not group[actual].eq(bounds[reference]).all():
                raise ValueError(f"Prediction {actual} must match assignment evidence.")
        ids = set(group["row_id"])
        required_ids = set(evidence.loc[evidence["role"].eq("evaluation"), "row_id"])
        scored_exclusions = evidence["role"].eq("excluded") & evidence[
            "exclusion_reason"
        ].isin(("missing_label", "missing_target", "locked_test_overlap"))
        allowed_ids = set(
            evidence.loc[
                evidence["role"].eq("evaluation") | scored_exclusions, "row_id"
            ]
        )
        if not ids.issubset(allowed_ids) or not required_ids.issubset(ids):
            raise ValueError("Prediction row IDs disagree with assignment evidence.")


def _feature_metadata(frame: pd.DataFrame, config: BenchmarkConfig, model: str) -> None:
    for name in ("feature_count", "selected_feature_count"):
        _integers(
            frame[name], name, minimum=1 if name == "selected_feature_count" else 0
        )
    if any(
        not isinstance(value, (bool, np.bool_))
        for value in frame["zero_observed_features"]
    ):
        raise ValueError("zero_observed_features must contain booleans.")
    if (frame["feature_count"] > frame["selected_feature_count"]).any():
        raise ValueError("Observed feature_count exceeds selected_feature_count.")
    if not frame["zero_observed_features"].eq(frame["feature_count"].eq(0)).all():
        raise ValueError("zero_observed_features disagrees with feature_count.")
    parsed = []
    for encoded, count in zip(
        frame["selected_features"], frame["selected_feature_count"], strict=True
    ):
        try:
            features = json.loads(encoded) if isinstance(encoded, str) else None
        except (ValueError, TypeError) as error:
            raise ValueError(
                "selected_features must contain JSON feature lists."
            ) from error
        if (
            not isinstance(features, list)
            or not features
            or any(
                not isinstance(value, str) or value not in config.feature_columns
                for value in features
            )
            or len(set(features)) != len(features)
            or len(features) != count
        ):
            raise ValueError(
                "selected_features must be unique bundle subsets "
                "matching the selected count."
            )
        parsed.append(tuple(features))
    feature_sets = pd.Series(parsed, index=frame.index)
    for _, group in frame.groupby("fold", sort=True):
        if len(set(feature_sets.loc[group.index])) != 1:
            raise ValueError("Selected features must be fixed within each arm/fold.")
    if model == "equal_weight_rank":
        if frame["candidate_id"].notna().any():
            raise ValueError("Equal-rank baseline candidate_id must be missing.")
        return
    _text(frame["candidate_id"], "candidate_id")
    allowed = {
        candidate.candidate_id for candidate in candidate_specs(config, family=model)
    }
    if not frame["candidate_id"].isin(allowed).all():
        raise ValueError("candidate_id must belong to the configured family grid.")
    if (frame.groupby("fold")["candidate_id"].nunique() != 1).any():
        raise ValueError("candidate_id must be fixed within each arm/fold.")


def _outcome_authorization(frame: pd.DataFrame, assignments: pd.DataFrame) -> None:
    """Check existing producer masks, never repair or select by future outcomes."""
    keys = pd.MultiIndex.from_frame(frame[["fold", "row_id"]])
    evidence = assignments.set_index(["fold", "row_id"]).reindex(keys)
    evaluation = evidence["role"].eq("evaluation").to_numpy()
    missing_target = (
        evidence["exclusion_reason"].eq("missing_target").fillna(False).to_numpy()
    )
    unauthorized_label = (
        evidence["exclusion_reason"]
        .isin(("missing_label", "locked_test_overlap"))
        .to_numpy()
    )
    target_present = frame["target"].notna().to_numpy()
    return_present = frame["realized_return"].notna().to_numpy()
    if (
        (evaluation & ~target_present).any()
        or (missing_target & target_present).any()
        or (unauthorized_label & (target_present | return_present)).any()
    ):
        raise ValueError("Prediction outcomes contradict assignment authorization.")


def _exact_outcome(value: object) -> Decimal | None:
    """Compare original validated outcomes without float64 rounding collisions.

    Construction/comparison only: no ambient-context arithmetic or expansion of
    exponent strings into huge integer ratios. Decimal strings and integers keep
    their value; floats keep their exact binary value (so float 0.1 differs from
    decimal-string 0.1). This evidence does not change evaluation arithmetic.
    """
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    try:
        if isinstance(value, float):
            return Decimal.from_float(value)
        if isinstance(value, Integral):
            return Decimal(int(value))
        if isinstance(value, (str, Decimal)):
            return Decimal(value)
    except (TypeError, ValueError, ArithmeticError) as error:
        raise ValueError("Unsupported exact numeric outcome representation.") from error
    raise ValueError("Unsupported exact numeric outcome representation.")


def prepare_predictions(
    legacy: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    assignments: pd.DataFrame,
    configs: tuple[BenchmarkConfig, BenchmarkConfig],
    expected_dates: tuple[str, ...],
    model_family: str,
) -> pd.DataFrame:
    requests = (
        (legacy, "equal_weight_rank", "legacy_equal_rank", configs[0]),
        (legacy, model_family, "legacy_model", configs[0]),
        (candidate, model_family, "candidate_model", configs[1]),
    )
    prepared = []
    for source, model, role, config in requests:
        frame, positions = _selected_structure(source, model)
        _fit_and_schedule(frame, assignments, expected_dates)
        _feature_metadata(frame, config, model)
        prepared.append((source, frame, positions, role))
    frames = []
    for source, frame, positions, role in prepared:
        for name in _OUTCOMES:
            original = source[name].iloc[positions]
            frame[name] = _numeric_observations(original).to_numpy()
            if name != "score":
                frame[f"_exact_{name}"] = [_exact_outcome(value) for value in original]
        _outcome_authorization(frame, assignments)
        if (frame["feature_count"].eq(0) & frame["score"].notna()).any():
            raise ValueError(
                "Zero-observed-feature predictions must have a missing score."
            )
        frame["model"] = role
        frames.append(
            frame.sort_values(list(_KEYS), kind="stable").reset_index(drop=True)
        )
    aligned_columns = [*_KEYS, "row_id", "_exact_target", "_exact_realized_return"]
    for frame in frames[1:]:
        try:
            pd.testing.assert_frame_equal(
                frames[0][aligned_columns],
                frame[aligned_columns],
                check_dtype=False,
                check_exact=True,
            )
        except AssertionError as error:
            raise ValueError(
                "All three arms must have identical keys, row IDs, "
                "outcomes and missingness."
            ) from error
    return pd.concat(frames, ignore_index=True).drop(
        columns=["_exact_target", "_exact_realized_return"]
    )
