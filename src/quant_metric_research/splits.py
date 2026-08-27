from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PurgedWalkForwardSplit:
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    train_end_date: pd.Timestamp
    test_start_date: pd.Timestamp
    test_end_date: pd.Timestamp


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def build_purged_walk_forward_splits(
    frame: pd.DataFrame,
    *,
    n_splits: int,
    test_date_count: int,
    min_train_date_count: int,
    as_of_date_column: str = "as_of_date",
    label_end_date_column: str = "label_end_date",
) -> list[PurgedWalkForwardSplit]:
    _positive_integer(n_splits, "n_splits")
    _positive_integer(test_date_count, "test_date_count")
    _positive_integer(min_train_date_count, "min_train_date_count")
    required = {as_of_date_column, label_end_date_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required split columns: {', '.join(missing)}")

    normalized = frame.copy(deep=True)
    normalized[as_of_date_column] = pd.to_datetime(
        normalized[as_of_date_column],
        errors="coerce",
    )
    normalized[label_end_date_column] = pd.to_datetime(
        normalized[label_end_date_column],
        errors="coerce",
    )
    if normalized[as_of_date_column].isna().any():
        raise ValueError("as_of_date contains invalid values.")

    unique_dates = pd.DatetimeIndex(
        normalized[as_of_date_column].drop_duplicates().sort_values()
    )
    required_date_count = min_train_date_count + n_splits * test_date_count
    if len(unique_dates) < required_date_count:
        raise ValueError("Not enough unique dates for the requested split.")

    initial_train_count = len(unique_dates) - n_splits * test_date_count
    folds: list[PurgedWalkForwardSplit] = []
    for split_number in range(n_splits):
        test_start_position = initial_train_count + split_number * test_date_count
        test_dates = unique_dates[
            test_start_position : test_start_position + test_date_count
        ]
        test_start_date = pd.Timestamp(test_dates[0])
        test_end_date = pd.Timestamp(test_dates[-1])
        candidate_train_dates = unique_dates[:test_start_position]

        train_mask = normalized[as_of_date_column].isin(candidate_train_dates) & (
            normalized[label_end_date_column] < test_start_date
        )
        test_mask = normalized[as_of_date_column].isin(test_dates)
        train_indices = tuple(int(index) for index in normalized.index[train_mask])
        test_indices = tuple(int(index) for index in normalized.index[test_mask])
        retained_train_dates = normalized.loc[
            list(train_indices),
            as_of_date_column,
        ].nunique()
        if retained_train_dates < min_train_date_count:
            raise ValueError("Not enough training dates remain after label purging.")
        if not test_indices:
            raise ValueError("A requested test fold contains no rows.")

        folds.append(
            PurgedWalkForwardSplit(
                train_indices=train_indices,
                test_indices=test_indices,
                train_end_date=pd.Timestamp(
                    normalized.loc[
                        list(train_indices),
                        as_of_date_column,
                    ].max()
                ),
                test_start_date=test_start_date,
                test_end_date=test_end_date,
            )
        )
    return folds
