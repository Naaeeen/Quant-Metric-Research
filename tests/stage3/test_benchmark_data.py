from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from quant_metric_research.benchmark_data import (
    build_stage3_data_plan,
    equal_date_training_weights,
    validate_benchmark_panel,
)

FEATURES = ("value", "quality")


def _panel(*, date_count: int = 16) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=date_count)
    rows: list[dict[str, object]] = []
    for date_index, as_of_date in enumerate(dates):
        for symbol_index, symbol in enumerate(("AAA", "BBB", "CCC")):
            label_position = date_index + 2
            label_end_date = (
                dates[label_position] if label_position < len(dates) else pd.NaT
            )
            target = (
                float(symbol_index - 1) * 0.01 + date_index * 0.0001
                if pd.notna(label_end_date)
                else np.nan
            )
            rows.append(
                {
                    "as_of_date": as_of_date,
                    "decision_time": as_of_date + pd.Timedelta(hours=21),
                    "symbol": symbol,
                    "label_end_date": label_end_date,
                    "feature_available_at": as_of_date + pd.Timedelta(hours=20),
                    "value": float(symbol_index),
                    "quality": float(date_index - symbol_index),
                    "target": target,
                }
            )
    return pd.DataFrame(rows)


def _row_id_map(frame: pd.DataFrame) -> dict[tuple[pd.Timestamp, str], str]:
    return {
        (pd.Timestamp(row.as_of_date), str(row.symbol)): str(row.row_id)
        for row in frame.itertuples(index=False)
    }


def test_validation_is_deterministic_defensive_and_does_not_mutate_input() -> None:
    source = _panel().sample(frac=1.0, random_state=17)
    source.index = pd.Index(range(1000, 1000 + len(source)))
    original = source.copy(deep=True)

    first = validate_benchmark_panel(
        source,
        feature_columns=FEATURES,
        target_column="target",
    )
    second = validate_benchmark_panel(
        source.sample(frac=1.0, random_state=31).reset_index(drop=True),
        feature_columns=FEATURES,
        target_column="target",
    )

    assert_frame_equal(source, original)
    assert _row_id_map(first.frame) == _row_id_map(second.frame)
    assert first.frame.index.equals(pd.RangeIndex(len(first.frame)))
    assert first.frame["row_id"].is_unique

    changed_copy = first.frame
    changed_copy.loc[0, "value"] = 999.0
    assert first.frame.loc[0, "value"] != 999.0
    with pytest.raises(FrozenInstanceError):
        first.target_column = "other"  # type: ignore[misc]


def test_validation_rejects_duplicate_normalized_row_keys() -> None:
    frame = _panel().iloc[:2].copy()
    frame.loc[frame.index[1], ["as_of_date", "symbol"]] = [
        frame.iloc[0]["as_of_date"],
        " aaa ",
    ]

    with pytest.raises(ValueError, match=r"unique.*as_of_date.*symbol"):
        validate_benchmark_panel(
            frame,
            feature_columns=FEATURES,
            target_column="target",
        )


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [
        ("value", float("inf")),
        ("quality", "not-a-number"),
        ("target", float("-inf")),
        ("target", "not-a-number"),
    ],
)
def test_validation_allows_nan_but_rejects_non_numeric_or_infinite_values(
    column: str,
    bad_value: object,
) -> None:
    frame = _panel()
    frame.loc[0, "value"] = np.nan
    frame.loc[1, "target"] = np.nan
    if isinstance(bad_value, str):
        frame[column] = frame[column].astype(object)
    frame.loc[2, column] = bad_value

    with pytest.raises(ValueError, match=column):
        validate_benchmark_panel(
            frame,
            feature_columns=FEATURES,
            target_column="target",
        )


def test_validation_rejects_malformed_dates_and_non_forward_labels() -> None:
    malformed = _panel()
    malformed["label_end_date"] = malformed["label_end_date"].astype(object)
    malformed.loc[0, "label_end_date"] = "definitely-not-a-date"
    with pytest.raises(ValueError, match="label_end_date.*invalid"):
        validate_benchmark_panel(
            malformed,
            feature_columns=FEATURES,
            target_column="target",
        )

    timezone_aware = _panel()
    timezone_aware["as_of_date"] = timezone_aware["as_of_date"].dt.tz_localize(
        "Australia/Sydney"
    )
    with pytest.raises(ValueError, match="timezone-naive calendar dates"):
        validate_benchmark_panel(
            timezone_aware,
            feature_columns=FEATURES,
            target_column="target",
        )

    non_forward = _panel()
    non_forward.loc[0, "label_end_date"] = non_forward.loc[0, "as_of_date"]
    with pytest.raises(ValueError, match="after as_of_date"):
        validate_benchmark_panel(
            non_forward,
            feature_columns=FEATURES,
            target_column="target",
        )

    missing_label_for_known_target = _panel()
    missing_label_for_known_target.loc[0, "label_end_date"] = pd.NaT
    with pytest.raises(ValueError, match="finite targets require label_end_date"):
        validate_benchmark_panel(
            missing_label_for_known_target,
            feature_columns=FEATURES,
            target_column="target",
        )


def test_validation_enforces_feature_availability_at_decision_time() -> None:
    future_feature = _panel()
    future_feature.loc[0, "feature_available_at"] = future_feature.loc[
        0, "decision_time"
    ] + pd.Timedelta(seconds=1)

    with pytest.raises(ValueError, match="available after decision_time"):
        validate_benchmark_panel(
            future_feature,
            feature_columns=FEATURES,
            target_column="target",
        )

    missing_timestamp = _panel()
    missing_timestamp.loc[0, "feature_available_at"] = pd.NaT
    with pytest.raises(ValueError, match="feature_available_at.*missing"):
        validate_benchmark_panel(
            missing_timestamp,
            feature_columns=FEATURES,
            target_column="target",
        )

    unavailable_and_missing = _panel()
    unavailable_and_missing.loc[0, list(FEATURES)] = np.nan
    unavailable_and_missing.loc[0, "feature_available_at"] = pd.NaT
    validated = validate_benchmark_panel(
        unavailable_and_missing,
        feature_columns=FEATURES,
        target_column="target",
    )
    assert pd.isna(validated.frame.loc[0, "feature_available_at"])

    decision_before_observation = _panel()
    decision_before_observation.loc[0, "decision_time"] = (
        decision_before_observation.loc[0, "as_of_date"] - pd.Timedelta(seconds=1)
    )
    with pytest.raises(ValueError, match="decision_time must not be before"):
        validate_benchmark_panel(
            decision_before_observation,
            feature_columns=FEATURES,
            target_column="target",
        )


def test_validation_checks_per_feature_availability_columns_when_present() -> None:
    frame = _panel().drop(columns="feature_available_at")
    frame["value_available_at"] = frame["decision_time"]
    frame["quality_available_at"] = frame["decision_time"]
    frame.loc[0, "quality_available_at"] += pd.Timedelta(seconds=1)

    with pytest.raises(ValueError, match="quality_available_at.*decision_time"):
        validate_benchmark_panel(
            frame,
            feature_columns=FEATURES,
            target_column="target",
        )


def test_stage3_plan_supports_explicit_non_identifier_column_names() -> None:
    frame = _panel().rename(
        columns={
            "as_of_date": "as of",
            "symbol": "asset id",
            "label_end_date": "label end",
            "decision_time": "decision at",
            "target": "target return",
        }
    )

    plan = build_stage3_data_plan(
        frame,
        feature_columns=FEATURES,
        target_column="target return",
        locked_test_date_count=2,
        n_splits=2,
        evaluation_date_count=2,
        min_train_date_count=3,
        as_of_date_column="as of",
        symbol_column="asset id",
        label_end_date_column="label end",
        decision_time_column="decision at",
    )

    assert len(plan.development_folds) == 2
    assert plan.panel.as_of_date_column == "as of"


def test_stage3_plan_locks_final_dates_and_never_uses_them_in_development() -> None:
    plan = build_stage3_data_plan(
        _panel(),
        feature_columns=FEATURES,
        target_column="target",
        locked_test_date_count=2,
        n_splits=2,
        evaluation_date_count=2,
        min_train_date_count=3,
    )
    panel = plan.panel.frame.set_index("row_id")
    locked = plan.locked_test

    locked_dates = panel.loc[list(locked.test_row_ids), "as_of_date"].unique()
    assert list(locked_dates) == list(pd.bdate_range("2025-01-20", periods=2))
    assert locked.test_start_date == pd.Timestamp("2025-01-20")
    assert set(locked.development_row_ids).isdisjoint(locked.test_row_ids)
    assert panel.loc[list(locked.test_row_ids), "target"].notna().all()
    assert panel.loc[list(locked.test_row_ids), "label_end_date"].notna().all()
    assert {assignment.row_id for assignment in locked.assignments} == set(panel.index)
    assert (
        panel.loc[list(locked.training_row_ids), "label_end_date"]
        < locked.test_start_date
    ).all()
    locked_training = panel.loc[list(locked.training_row_ids), ["as_of_date"]].copy()
    locked_training["weight"] = locked.training_weights
    assert locked_training["weight"].mean() == pytest.approx(1.0)
    assert locked_training.groupby("as_of_date")["weight"].sum().nunique() == 1

    locked_ids = set(locked.test_row_ids)
    immature_tail_ids = set(panel.index[panel["as_of_date"] > locked.test_end_date])
    assert immature_tail_ids
    for fold in plan.development_folds:
        assert locked_ids.isdisjoint(fold.train_row_ids)
        assert locked_ids.isdisjoint(fold.evaluation_row_ids)
        assert (
            panel.loc[list(fold.evaluation_row_ids), "label_end_date"]
            < locked.test_start_date
        ).all()
        locked_assignments = [
            assignment
            for assignment in fold.assignments
            if assignment.row_id in locked_ids
        ]
        assert {row.exclusion_reason for row in locked_assignments} == {
            "locked_final_test"
        }
        assert {
            assignment.exclusion_reason
            for assignment in fold.assignments
            if assignment.row_id in immature_tail_ids
        } == {"unavailable_future_label"}


def test_development_folds_strictly_purge_labels_and_explain_every_row() -> None:
    plan = build_stage3_data_plan(
        _panel(),
        feature_columns=FEATURES,
        target_column="target",
        locked_test_date_count=2,
        n_splits=2,
        evaluation_date_count=2,
        min_train_date_count=3,
    )
    panel = plan.panel.frame.set_index("row_id")

    for fold in plan.development_folds:
        assignments = {assignment.row_id: assignment for assignment in fold.assignments}
        assert set(assignments) == set(panel.index)
        assert (
            panel.loc[list(fold.train_row_ids), "label_end_date"]
            < fold.evaluation_start_date
        ).all()
        assert (
            panel.loc[list(fold.train_row_ids), "as_of_date"]
            < fold.evaluation_start_date
        ).all()
        assert set(fold.train_row_ids).isdisjoint(fold.evaluation_row_ids)

        boundary_overlap = panel.index[
            (panel["as_of_date"] < fold.evaluation_start_date)
            & (panel["label_end_date"] == fold.evaluation_start_date)
        ]
        assert boundary_overlap.size > 0
        assert {
            assignments[row_id].exclusion_reason for row_id in boundary_overlap
        } == {"label_overlap"}

        future_rows = panel.index[
            (panel["as_of_date"] > fold.evaluation_end_date)
            & (panel["as_of_date"] < plan.locked_test.test_start_date)
            & (panel["label_end_date"] < plan.locked_test.test_start_date)
        ]
        if future_rows.size:
            assert {assignments[row_id].exclusion_reason for row_id in future_rows} == {
                "future_of_evaluation"
            }

        locked_overlap = panel.index[
            (panel["as_of_date"] < plan.locked_test.test_start_date)
            & (panel["label_end_date"] >= plan.locked_test.test_start_date)
        ]
        assert locked_overlap.size > 0
        assert {assignments[row_id].exclusion_reason for row_id in locked_overlap} == {
            "locked_test_overlap"
        }


def test_fold_training_weights_give_every_date_equal_total_weight() -> None:
    plan = build_stage3_data_plan(
        _panel(),
        feature_columns=FEATURES,
        target_column="target",
        locked_test_date_count=2,
        n_splits=1,
        evaluation_date_count=2,
        min_train_date_count=3,
    )
    fold = plan.development_folds[0]
    panel = plan.panel.frame.set_index("row_id")
    weighted = panel.loc[list(fold.train_row_ids), ["as_of_date"]].copy()
    weighted["weight"] = fold.training_weights

    date_weight_totals = weighted.groupby("as_of_date")["weight"].sum()
    assert date_weight_totals.nunique() == 1
    assert weighted["weight"].mean() == pytest.approx(1.0)
    assert sum(fold.training_weights) == pytest.approx(float(len(weighted)))

    uneven = pd.DataFrame(
        {
            "as_of_date": ["2025-01-02", "2025-01-02", "2025-01-03"],
        }
    )
    assert equal_date_training_weights(uneven) == (0.75, 0.75, 1.5)


@pytest.mark.parametrize(
    "overrides",
    [
        {"locked_test_date_count": 0},
        {"n_splits": 0},
        {"evaluation_date_count": 0},
        {"min_train_date_count": 0},
        {"locked_test_date_count": 5, "n_splits": 3},
    ],
)
def test_stage3_plan_rejects_invalid_or_impossible_split_requests(
    overrides: dict[str, int],
) -> None:
    arguments = {
        "locked_test_date_count": 2,
        "n_splits": 2,
        "evaluation_date_count": 2,
        "min_train_date_count": 3,
        **overrides,
    }

    with pytest.raises(ValueError):
        build_stage3_data_plan(
            _panel(),
            feature_columns=FEATURES,
            target_column="target",
            **arguments,
        )
