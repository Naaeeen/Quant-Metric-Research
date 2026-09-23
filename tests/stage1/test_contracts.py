from __future__ import annotations

import pandas as pd
import pytest

from quant_metric_research.config import PanelConfig
from quant_metric_research.contracts import (
    DataContractError,
    validate_memberships,
    validate_prices,
)


def test_panel_config_is_immutable_and_validates_boundaries() -> None:
    config = PanelConfig(
        dataset_version="test-v1",
        universe_id="TEST",
        benchmark_symbol="BENCH",
        lookback_sessions=5,
        min_observations=4,
        target_horizon_sessions=2,
        entry_lag_sessions=1,
    )

    assert config.feature_columns[0] == "trailing_return"
    with pytest.raises(AttributeError):
        config.lookback_sessions = 10  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lookback_sessions", 0),
        ("min_observations", 0),
        ("target_horizon_sessions", 0),
        ("entry_lag_sessions", -1),
        ("annualization_sessions", 0),
    ],
)
def test_panel_config_rejects_invalid_numeric_values(field: str, value: int) -> None:
    kwargs = {
        "dataset_version": "test-v1",
        "universe_id": "TEST",
        "benchmark_symbol": "BENCH",
        field: value,
    }
    with pytest.raises(ValueError):
        PanelConfig(**kwargs)


def test_validate_prices_normalizes_a_copy_without_mutating_input() -> None:
    original = pd.DataFrame(
        {
            "date": ["2025-01-02"],
            "symbol": [" aaa "],
            "adjusted_close": [10.0],
        }
    )
    before = original.copy(deep=True)

    validated = validate_prices(original)

    pd.testing.assert_frame_equal(original, before)
    assert validated.loc[0, "symbol"] == "AAA"
    assert pd.api.types.is_datetime64_any_dtype(validated["date"])


def test_validate_prices_rejects_duplicates_and_non_positive_values() -> None:
    duplicated = pd.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-02"],
            "symbol": ["AAA", "AAA"],
            "adjusted_close": [10.0, 11.0],
        }
    )
    with pytest.raises(DataContractError, match="Duplicate"):
        validate_prices(duplicated)

    non_positive = duplicated.iloc[[0]].copy()
    non_positive.loc[:, "adjusted_close"] = 0.0
    with pytest.raises(DataContractError, match="strictly positive"):
        validate_prices(non_positive)


def test_validate_memberships_uses_half_open_non_overlapping_intervals() -> None:
    valid = pd.DataFrame(
        {
            "universe_id": ["TEST", "TEST"],
            "symbol": ["AAA", "AAA"],
            "effective_from": ["2025-01-01", "2025-02-01"],
            "effective_to": ["2025-02-01", None],
            "source": ["history", "history"],
        }
    )
    result = validate_memberships(valid)
    assert result.shape[0] == 2

    overlapping = valid.copy()
    overlapping.loc[0, "effective_to"] = "2025-02-02"
    with pytest.raises(DataContractError, match="overlap"):
        validate_memberships(overlapping)


def test_validate_memberships_rejects_empty_or_reversed_intervals() -> None:
    invalid = pd.DataFrame(
        {
            "universe_id": ["TEST"],
            "symbol": ["AAA"],
            "effective_from": ["2025-02-01"],
            "effective_to": ["2025-02-01"],
            "source": ["history"],
        }
    )
    with pytest.raises(DataContractError, match="after effective_from"):
        validate_memberships(invalid)


@pytest.mark.parametrize("end_date", ["misspelled-date", "", "2025-02-30"])
def test_validate_memberships_rejects_malformed_nonmissing_end_dates(
    end_date: str,
) -> None:
    memberships = pd.DataFrame(
        {
            "universe_id": ["TEST"],
            "symbol": ["AAA"],
            "effective_from": ["2025-01-01"],
            "effective_to": [end_date],
            "source": ["history"],
        }
    )

    with pytest.raises(DataContractError, match="effective_to.*invalid"):
        validate_memberships(memberships)


@pytest.mark.parametrize("end_date", [None, pd.NaT, pd.NA])
def test_validate_memberships_preserves_genuinely_open_intervals(
    end_date: object,
) -> None:
    memberships = pd.DataFrame(
        {
            "universe_id": ["TEST"],
            "symbol": ["AAA"],
            "effective_from": ["2025-01-01"],
            "effective_to": [end_date],
            "source": ["history"],
        }
    )
    original = memberships.copy(deep=True)

    validated = validate_memberships(memberships)

    assert pd.isna(validated.loc[0, "effective_to"])
    pd.testing.assert_frame_equal(memberships, original)
