from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype


@dataclass(frozen=True, slots=True)
class BenchmarkPanel:
    """Validated panel whose public frame access always returns a defensive copy."""

    feature_columns: tuple[str, ...]
    target_column: str
    as_of_date_column: str
    symbol_column: str
    label_end_date_column: str
    _frame: pd.DataFrame = field(repr=False, compare=False)

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame.copy(deep=True)

    def rows(self, row_ids: Sequence[str]) -> pd.DataFrame:
        """Return rows in ``row_ids`` order without exposing internal state."""

        indexed = self._frame.set_index("row_id", drop=False)
        requested = tuple(row_ids)
        missing = sorted(set(requested) - set(indexed.index))
        if missing:
            raise ValueError(f"Unknown row_id values: {', '.join(missing)}")
        return indexed.loc[list(requested)].reset_index(drop=True).copy(deep=True)


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    row_id: str
    split_id: str
    role: str
    exclusion_reason: str | None


@dataclass(frozen=True, slots=True)
class LockedFinalTest:
    test_start_date: pd.Timestamp
    test_end_date: pd.Timestamp
    development_row_ids: tuple[str, ...]
    training_row_ids: tuple[str, ...]
    training_weights: tuple[float, ...]
    test_row_ids: tuple[str, ...]
    excluded_row_ids: tuple[str, ...]
    assignments: tuple[SplitAssignment, ...]


@dataclass(frozen=True, slots=True)
class DevelopmentFold:
    fold_number: int
    train_end_date: pd.Timestamp
    training_label_end_max: pd.Timestamp
    evaluation_start_date: pd.Timestamp
    evaluation_end_date: pd.Timestamp
    train_row_ids: tuple[str, ...]
    evaluation_row_ids: tuple[str, ...]
    excluded_row_ids: tuple[str, ...]
    training_weights: tuple[float, ...]
    assignments: tuple[SplitAssignment, ...]


@dataclass(frozen=True, slots=True)
class Stage3DataPlan:
    panel: BenchmarkPanel
    locked_test: LockedFinalTest
    development_folds: tuple[DevelopmentFold, ...]


def _column_tuple(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of column names.")
    columns = tuple(values)
    if not columns or any(
        not isinstance(column, str) or not column for column in columns
    ):
        raise ValueError(f"{name} must contain non-empty strings.")
    if len(columns) != len(set(columns)):
        raise ValueError(f"{name} must contain unique column names.")
    return columns


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required benchmark columns: {', '.join(missing)}")


def _datetime_values(
    series: pd.Series,
    *,
    name: str,
    allow_missing: bool,
    reject_timezone: bool = False,
    require_calendar_date: bool = False,
) -> pd.Series:
    originally_missing = series.isna()
    if reject_timezone:

        def is_timezone_aware(value: object) -> bool:
            try:
                return pd.Timestamp(value).tzinfo is not None
            except (TypeError, ValueError):
                return False

        if any(is_timezone_aware(value) for value in series.loc[~originally_missing]):
            raise ValueError(f"{name} must contain timezone-naive calendar dates.")
    try:
        converted = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} contains invalid date values.") from error
    invalid = converted.isna() & ~originally_missing
    if invalid.any() or (not allow_missing and converted.isna().any()):
        raise ValueError(f"{name} contains invalid date values.")
    if require_calendar_date:
        intraday = converted.notna() & (converted != converted.dt.normalize())
        if intraday.any():
            raise ValueError(f"{name} must contain normalized calendar dates.")
    return converted.dt.tz_convert(None)


def _numeric_values(series: pd.Series, *, name: str) -> pd.Series:
    contains_bool = (
        is_bool_dtype(series.dtype)
        or series.map(lambda value: isinstance(value, (bool, np.bool_))).any()
    )
    if contains_bool or is_datetime64_any_dtype(series.dtype):
        raise ValueError(f"{name} must contain numeric finite-or-NaN values.")

    converted = pd.to_numeric(series, errors="coerce")
    invalid = converted.isna() & ~series.isna()
    values = converted.to_numpy()
    if invalid.any() or np.iscomplexobj(values):
        raise ValueError(f"{name} must contain numeric finite-or-NaN values.")
    numeric = converted.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(numeric).any():
        raise ValueError(f"{name} must contain numeric finite-or-NaN values.")
    return pd.Series(numeric, index=series.index, dtype=float)


def _availability_columns(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    requested: Sequence[str] | None,
) -> tuple[str, ...]:
    if requested is not None:
        columns = _column_tuple(requested, "feature_availability_columns")
        _require_columns(frame, columns)
        return columns

    candidates = ("feature_available_at",) + tuple(
        f"{feature}_available_at" for feature in features
    )
    return tuple(column for column in candidates if column in frame.columns)


def _validate_availability(
    frame: pd.DataFrame,
    *,
    features: tuple[str, ...],
    availability_columns: tuple[str, ...],
    decision_time_column: str | None,
    as_of_date_column: str,
) -> None:
    if decision_time_column is not None and decision_time_column in frame.columns:
        frame[decision_time_column] = _datetime_values(
            frame[decision_time_column],
            name=decision_time_column,
            allow_missing=False,
        )
        decision_time = frame[decision_time_column]
        decision_name = decision_time_column
        if (decision_time < frame[as_of_date_column]).any():
            raise ValueError(
                f"{decision_time_column} must not be before {as_of_date_column}."
            )
        if (decision_time.dt.normalize() != frame[as_of_date_column]).any():
            raise ValueError(
                f"{decision_time_column} must be on the same calendar day as "
                f"{as_of_date_column}."
            )
    else:
        decision_time = frame[as_of_date_column]
        decision_name = as_of_date_column

    any_feature_present = frame.loc[:, list(features)].notna().any(axis=1)
    for column in availability_columns:
        frame[column] = _datetime_values(
            frame[column],
            name=column,
            allow_missing=True,
        )
        feature_name = column.removesuffix("_available_at")
        required = (
            frame[feature_name].notna()
            if feature_name in features and column != "feature_available_at"
            else any_feature_present
        )
        if (required & frame[column].isna()).any():
            raise ValueError(f"{column} is missing for an observed feature value.")
        if (required & (frame[column] > decision_time)).any():
            raise ValueError(f"{column} is available after {decision_name}.")


def _stable_row_ids(frame: pd.DataFrame, *, date: str, symbol: str) -> pd.Series:
    def digest(row: tuple[pd.Timestamp, str]) -> str:
        timestamp, security = row
        key = f"{pd.Timestamp(timestamp).value}|{security}".encode()
        return f"qmr_{sha256(key).hexdigest()}"

    keys = zip(frame[date], frame[symbol], strict=True)
    return pd.Series((digest(key) for key in keys), index=frame.index, dtype="string")


def validate_benchmark_panel(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_column: str,
    as_of_date_column: str = "as_of_date",
    symbol_column: str = "symbol",
    label_end_date_column: str = "label_end_date",
    decision_time_column: str | None = "decision_time",
    feature_availability_columns: Sequence[str] | None = None,
) -> BenchmarkPanel:
    """Validate and canonically order a Stage 3 feature/target panel."""

    if not isinstance(frame, pd.DataFrame):
        raise ValueError("frame must be a pandas DataFrame.")
    if frame.empty:
        raise ValueError("frame must not be empty.")
    features = _column_tuple(feature_columns, "feature_columns")
    if not isinstance(target_column, str) or not target_column:
        raise ValueError("target_column must be a non-empty string.")
    if target_column in features:
        raise ValueError("target_column cannot also be a feature column.")

    required = (
        as_of_date_column,
        symbol_column,
        label_end_date_column,
        target_column,
        *features,
    )
    _require_columns(frame, required)
    validated = frame.copy(deep=True)
    validated[as_of_date_column] = _datetime_values(
        validated[as_of_date_column],
        name=as_of_date_column,
        allow_missing=False,
        reject_timezone=True,
        require_calendar_date=True,
    )
    validated[label_end_date_column] = _datetime_values(
        validated[label_end_date_column],
        name=label_end_date_column,
        allow_missing=True,
        reject_timezone=True,
        require_calendar_date=True,
    )
    validated[symbol_column] = (
        validated[symbol_column].astype("string").str.strip().str.upper()
    )
    if validated[symbol_column].isna().any() or (validated[symbol_column] == "").any():
        raise ValueError(f"{symbol_column} contains invalid values.")
    if validated.duplicated([as_of_date_column, symbol_column], keep=False).any():
        raise ValueError("Rows must be unique by (as_of_date, symbol).")

    for column in (*features, target_column):
        validated[column] = _numeric_values(validated[column], name=column)
    known_labels = validated[label_end_date_column].notna()
    if (
        known_labels
        & (validated[label_end_date_column] <= validated[as_of_date_column])
    ).any():
        raise ValueError("label_end_date must be after as_of_date when present.")
    if (
        validated[target_column].notna() & validated[label_end_date_column].isna()
    ).any():
        raise ValueError("finite targets require label_end_date.")

    availability = _availability_columns(
        validated, features, feature_availability_columns
    )
    _validate_availability(
        validated,
        features=features,
        availability_columns=availability,
        decision_time_column=decision_time_column,
        as_of_date_column=as_of_date_column,
    )
    validated = validated.sort_values(
        [as_of_date_column, symbol_column], kind="stable"
    ).reset_index(drop=True)
    if "row_id" in validated.columns:
        validated = validated.drop(columns="row_id")
    validated.insert(
        0,
        "row_id",
        _stable_row_ids(validated, date=as_of_date_column, symbol=symbol_column),
    )
    if not validated["row_id"].is_unique:
        raise ValueError("Generated row_id values are not unique.")
    return BenchmarkPanel(
        feature_columns=features,
        target_column=target_column,
        as_of_date_column=as_of_date_column,
        symbol_column=symbol_column,
        label_end_date_column=label_end_date_column,
        _frame=validated,
    )


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def equal_date_training_weights(
    frame: pd.DataFrame,
    *,
    date_column: str = "as_of_date",
) -> tuple[float, ...]:
    """Give each date equal total influence and keep mean row weight at one."""

    if not isinstance(frame, pd.DataFrame):
        raise ValueError("frame must be a pandas DataFrame.")
    _require_columns(frame, (date_column,))
    if frame.empty:
        return ()
    dates = _datetime_values(
        frame[date_column],
        name=date_column,
        allow_missing=False,
        reject_timezone=True,
    )
    counts = dates.groupby(dates, sort=False).transform("size")
    per_date_total = len(dates) / dates.nunique()
    return tuple(float(value) for value in (per_date_total / counts).to_numpy())


def expand_training_cross_sections(
    frame: pd.DataFrame,
    *,
    training_indices: Sequence[int],
    outcome_columns: Sequence[str],
    as_of_date_column: str = "as_of_date",
    label_end_date_column: str = "label_end_date",
) -> pd.DataFrame:
    """Keep each fit date's feature universe, exposing only authorized labels.

    Purged/missing-label rows can supply contemporaneously available features
    for cross-sectional transforms, but cannot enter the supervised objective.
    """

    selected = frame.loc[list(training_indices)]
    training = frame.loc[
        frame[as_of_date_column].isin(selected[as_of_date_column])
    ].copy(deep=True)
    unauthorized = ~training.index.isin(training_indices)
    columns = list(dict.fromkeys(outcome_columns))
    training.loc[unauthorized, columns] = float("nan")
    training.loc[unauthorized, label_end_date_column] = pd.NaT
    return training


def evaluation_cross_section(
    panel: BenchmarkPanel,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    outcome_columns: Sequence[str],
    label_before: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Score a date block without selecting stocks by future outcome availability."""

    frame = panel.frame
    evaluation = frame.loc[
        frame[panel.as_of_date_column].between(start_date, end_date)
    ].copy(deep=True)
    label_end = evaluation[panel.label_end_date_column]
    authorized = label_end.notna()
    if label_before is not None:
        authorized = authorized & (label_end < label_before)
    # Missing targets and missing realized returns remain independent; only the
    # label-time authorization masks both outcomes.
    evaluation.loc[~authorized, list(dict.fromkeys(outcome_columns))] = float("nan")
    return evaluation.reset_index(drop=True)


def _training_exclusion_reason(
    row: Mapping[str, object],
    *,
    label_column: str,
    target_column: str,
    evaluation_start: pd.Timestamp,
) -> str | None:
    label_end = row[label_column]
    target = row[target_column]
    if pd.isna(label_end):
        return "missing_label"
    if pd.isna(target):
        return "missing_target"
    if pd.Timestamp(label_end) >= evaluation_start:
        return "label_overlap"
    return None


def reserve_locked_final_test(
    panel: BenchmarkPanel,
    *,
    locked_test_date_count: int,
    minimum_evaluable_count: int = 1,
) -> LockedFinalTest:
    """Reserve the chronologically final date block before model selection."""

    _positive_integer(locked_test_date_count, "locked_test_date_count")
    _positive_integer(minimum_evaluable_count, "minimum_evaluable_count")
    frame = panel.frame
    date_column = panel.as_of_date_column
    unique_dates = pd.DatetimeIndex(frame[date_column].drop_duplicates())
    evaluable = (
        frame[panel.label_end_date_column].notna() & frame[panel.target_column].notna()
    )
    evaluable_by_date = evaluable.groupby(frame[date_column], sort=False).sum()
    evaluable_dates = evaluable_by_date.index[evaluable_by_date > 0]
    if len(evaluable_dates) == 0:
        raise ValueError("No labeled dates are available for the locked final test.")
    last_evaluable_date = pd.Timestamp(evaluable_dates[-1])
    mature_dates = unique_dates[unique_dates <= last_evaluable_date]
    if len(mature_dates) <= locked_test_date_count:
        raise ValueError("Not enough dates remain before the locked final test block.")
    test_dates = mature_dates[-locked_test_date_count:]
    if (
        evaluable_by_date.reindex(test_dates, fill_value=0) < minimum_evaluable_count
    ).any():
        raise ValueError(
            "Every locked final-test date must meet the minimum labeled cross-section."
        )
    test_start = pd.Timestamp(test_dates[0])
    test_end = pd.Timestamp(test_dates[-1])
    split_id = "locked_final_test"
    assignments: list[SplitAssignment] = []
    for row in frame.to_dict(orient="records"):
        row_id = str(row["row_id"])
        as_of_date = pd.Timestamp(row[date_column])
        if as_of_date > test_end:
            assignments.append(
                SplitAssignment(
                    row_id,
                    split_id,
                    "excluded",
                    "unavailable_future_label",
                )
            )
            continue
        if as_of_date >= test_start:
            label_end = row[panel.label_end_date_column]
            target = row[panel.target_column]
            if pd.isna(label_end):
                assignments.append(
                    SplitAssignment(row_id, split_id, "excluded", "missing_label")
                )
            elif pd.isna(target):
                assignments.append(
                    SplitAssignment(row_id, split_id, "excluded", "missing_target")
                )
            else:
                assignments.append(
                    SplitAssignment(row_id, split_id, "evaluation", None)
                )
            continue
        reason = _training_exclusion_reason(
            row,
            label_column=panel.label_end_date_column,
            target_column=panel.target_column,
            evaluation_start=test_start,
        )
        role = "training" if reason is None else "excluded"
        assignments.append(SplitAssignment(row_id, split_id, role, reason))

    test_ids = tuple(row.row_id for row in assignments if row.role == "evaluation")
    training_ids = tuple(row.row_id for row in assignments if row.role == "training")
    excluded_ids = tuple(row.row_id for row in assignments if row.role == "excluded")
    development_ids = tuple(
        str(row["row_id"])
        for row in frame.to_dict(orient="records")
        if pd.Timestamp(row[date_column]) < test_start
    )
    return LockedFinalTest(
        test_start_date=test_start,
        test_end_date=test_end,
        development_row_ids=development_ids,
        training_row_ids=training_ids,
        training_weights=equal_date_training_weights(
            panel.rows(training_ids), date_column=date_column
        ),
        test_row_ids=test_ids,
        excluded_row_ids=excluded_ids,
        assignments=tuple(assignments),
    )


def _fold_assignment(
    row: Mapping[str, object],
    *,
    panel: BenchmarkPanel,
    split_id: str,
    evaluation_dates: frozenset[pd.Timestamp],
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
    reserved_reasons: Mapping[str, str],
    locked_test_start: pd.Timestamp,
) -> SplitAssignment:
    row_id = str(row["row_id"])
    as_of_date = pd.Timestamp(row[panel.as_of_date_column])
    if row_id in reserved_reasons:
        return SplitAssignment(row_id, split_id, "excluded", reserved_reasons[row_id])
    label_end = row[panel.label_end_date_column]
    if pd.notna(label_end) and pd.Timestamp(label_end) >= locked_test_start:
        return SplitAssignment(row_id, split_id, "excluded", "locked_test_overlap")
    if as_of_date in evaluation_dates:
        target = row[panel.target_column]
        if pd.isna(label_end):
            return SplitAssignment(row_id, split_id, "excluded", "missing_label")
        if pd.isna(target):
            return SplitAssignment(row_id, split_id, "excluded", "missing_target")
        return SplitAssignment(row_id, split_id, "evaluation", None)
    if evaluation_start <= as_of_date <= evaluation_end:
        target = row[panel.target_column]
        reason = "missing_label" if pd.isna(label_end) else "missing_target"
        if pd.notna(target):
            raise RuntimeError(
                "A labeled development date was omitted from evaluation."
            )
        return SplitAssignment(row_id, split_id, "excluded", reason)
    if as_of_date < evaluation_start:
        reason = _training_exclusion_reason(
            row,
            label_column=panel.label_end_date_column,
            target_column=panel.target_column,
            evaluation_start=evaluation_start,
        )
        role = "training" if reason is None else "excluded"
        return SplitAssignment(row_id, split_id, role, reason)
    if as_of_date > evaluation_end:
        return SplitAssignment(row_id, split_id, "excluded", "future_of_evaluation")
    raise RuntimeError("A development row was not assigned to a fold role.")


def build_development_outer_folds(
    panel: BenchmarkPanel,
    locked_test: LockedFinalTest,
    *,
    n_splits: int,
    evaluation_date_count: int,
    min_train_date_count: int,
) -> tuple[DevelopmentFold, ...]:
    """Build expanding-window outer folds using only development dates."""

    _positive_integer(n_splits, "n_splits")
    _positive_integer(evaluation_date_count, "evaluation_date_count")
    _positive_integer(min_train_date_count, "min_train_date_count")
    frame = panel.frame
    development = panel.rows(locked_test.development_row_ids)
    date_column = panel.as_of_date_column
    safe_for_selection = (
        development[panel.label_end_date_column].notna()
        & development[panel.target_column].notna()
        & (development[panel.label_end_date_column] < locked_test.test_start_date)
    )
    unique_dates = pd.DatetimeIndex(
        development.loc[safe_for_selection, date_column].drop_duplicates()
    )
    required_dates = min_train_date_count + n_splits * evaluation_date_count
    if len(unique_dates) < required_dates:
        raise ValueError("Not enough development dates for the requested outer folds.")

    initial_train_count = len(unique_dates) - n_splits * evaluation_date_count
    development_ids = frozenset(locked_test.development_row_ids)
    reserved_reasons = {
        assignment.row_id: (
            "locked_final_test"
            if assignment.role == "evaluation"
            else assignment.exclusion_reason or "locked_final_test"
        )
        for assignment in locked_test.assignments
        if assignment.row_id not in development_ids
    }
    folds: list[DevelopmentFold] = []
    for fold_number in range(n_splits):
        start = initial_train_count + fold_number * evaluation_date_count
        evaluation = unique_dates[start : start + evaluation_date_count]
        evaluation_start = pd.Timestamp(evaluation[0])
        evaluation_end = pd.Timestamp(evaluation[-1])
        split_id = f"development_fold_{fold_number}"
        assignments = tuple(
            _fold_assignment(
                row,
                panel=panel,
                split_id=split_id,
                evaluation_dates=frozenset(pd.Timestamp(date) for date in evaluation),
                evaluation_start=evaluation_start,
                evaluation_end=evaluation_end,
                reserved_reasons=reserved_reasons,
                locked_test_start=locked_test.test_start_date,
            )
            for row in frame.to_dict(orient="records")
        )
        train_ids = tuple(row.row_id for row in assignments if row.role == "training")
        evaluation_ids = tuple(
            row.row_id for row in assignments if row.role == "evaluation"
        )
        excluded_ids = tuple(
            row.row_id for row in assignments if row.role == "excluded"
        )
        training = panel.rows(train_ids)
        if training[date_column].nunique() < min_train_date_count:
            raise ValueError("Not enough training dates remain after label purging.")
        if not evaluation_ids:
            raise ValueError("An outer evaluation fold has no labeled rows.")
        label_max = pd.Timestamp(training[panel.label_end_date_column].max())
        if label_max >= evaluation_start:
            raise RuntimeError("Training labels overlap an evaluation fold.")
        folds.append(
            DevelopmentFold(
                fold_number=fold_number,
                train_end_date=pd.Timestamp(training[date_column].max()),
                training_label_end_max=label_max,
                evaluation_start_date=evaluation_start,
                evaluation_end_date=evaluation_end,
                train_row_ids=train_ids,
                evaluation_row_ids=evaluation_ids,
                excluded_row_ids=excluded_ids,
                training_weights=equal_date_training_weights(
                    training, date_column=date_column
                ),
                assignments=assignments,
            )
        )
    return tuple(folds)


def build_stage3_data_plan(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_column: str,
    locked_test_date_count: int,
    locked_min_cross_section: int = 1,
    n_splits: int,
    evaluation_date_count: int,
    min_train_date_count: int,
    as_of_date_column: str = "as_of_date",
    symbol_column: str = "symbol",
    label_end_date_column: str = "label_end_date",
    decision_time_column: str | None = "decision_time",
    feature_availability_columns: Sequence[str] | None = None,
) -> Stage3DataPlan:
    """Validate once, lock the final test, then build development outer folds."""

    panel = validate_benchmark_panel(
        frame,
        feature_columns=feature_columns,
        target_column=target_column,
        as_of_date_column=as_of_date_column,
        symbol_column=symbol_column,
        label_end_date_column=label_end_date_column,
        decision_time_column=decision_time_column,
        feature_availability_columns=feature_availability_columns,
    )
    locked_test = reserve_locked_final_test(
        panel,
        locked_test_date_count=locked_test_date_count,
        minimum_evaluable_count=locked_min_cross_section,
    )
    folds = build_development_outer_folds(
        panel,
        locked_test,
        n_splits=n_splits,
        evaluation_date_count=evaluation_date_count,
        min_train_date_count=min_train_date_count,
    )
    return Stage3DataPlan(
        panel=panel,
        locked_test=locked_test,
        development_folds=folds,
    )
