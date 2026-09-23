from __future__ import annotations

import pandas as pd
import pytest

from quant_metric_research.benchmark_data import validate_benchmark_panel


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "as_of_date": [pd.Timestamp("2025-01-02")],
            "symbol": ["AAA"],
            "decision_time": [pd.Timestamp("2025-01-02 21:00:00")],
            "feature_available_at": [pd.Timestamp("2025-01-02 20:00:00")],
            "label_end_date": [pd.Timestamp("2025-01-22")],
            "value": [1.0],
            "target": [0.01],
        }
    )


@pytest.mark.parametrize("column", ["as_of_date", "label_end_date"])
def test_benchmark_calendar_dates_reject_intraday_values(column: str) -> None:
    frame = _panel()
    frame.loc[0, column] += pd.Timedelta(hours=1)

    with pytest.raises(ValueError, match=f"{column}.*normalized calendar dates"):
        validate_benchmark_panel(
            frame, feature_columns=("value",), target_column="target"
        )


@pytest.mark.parametrize("decision", ["2025-01-03", "2025-02-01"])
def test_decision_time_cannot_move_past_its_as_of_day(decision: str) -> None:
    frame = _panel()
    frame.loc[0, "decision_time"] = pd.Timestamp(decision)
    frame.loc[0, "feature_available_at"] = pd.Timestamp(decision)

    with pytest.raises(ValueError, match="decision_time.*same calendar day"):
        validate_benchmark_panel(
            frame, feature_columns=("value",), target_column="target"
        )


def test_same_day_intraday_decision_and_availability_remain_valid() -> None:
    frame = _panel()
    original = frame.copy(deep=True)

    validated = validate_benchmark_panel(
        frame, feature_columns=("value",), target_column="target"
    )

    assert validated.frame.loc[0, "decision_time"] == pd.Timestamp(
        "2025-01-02 21:00:00"
    )
    pd.testing.assert_frame_equal(frame, original)


def test_missing_label_end_remains_valid_for_an_unlabeled_row() -> None:
    frame = _panel()
    frame.loc[0, "label_end_date"] = pd.NaT
    frame.loc[0, "target"] = float("nan")

    validated = validate_benchmark_panel(
        frame, feature_columns=("value",), target_column="target"
    )

    assert pd.isna(validated.frame.loc[0, "label_end_date"])
