from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from quant_metric_research import contracts
from quant_metric_research.config import PanelConfig
from quant_metric_research.contracts import (
    DataContractError,
    validate_memberships,
    validate_prices,
)
from quant_metric_research.panel import build_point_in_time_panel

INVALID_DATES = [
    "2025-01-02 12:00:00",
    pd.Timestamp("2025-01-02 00:00:00.000000001"),
    "2025-01-02T00:00:00+00:00",
    pd.Timestamp("2025-01-02", tz="Australia/Sydney"),
    0,
    1735776000000000000,
    1.5,
    True,
    np.bool_(False),
    "2025-02-30",
    "invalid",
    "",
    "NaT",
    ["2025-01-02"],
]


def _prices(value: object) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.Series([value], dtype=object),
            "symbol": ["AAA"],
            "adjusted_close": [10.0],
        }
    )


def _memberships() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "universe_id": ["TEST"],
            "symbol": ["AAA"],
            "effective_from": ["2025-01-01"],
            "effective_to": [None],
            "source": ["fixture"],
        }
    )


@pytest.mark.parametrize("value", INVALID_DATES + [None, pd.NA, pd.NaT, np.nan])
def test_prices_reject_non_daily_dates(value: object) -> None:
    with pytest.raises(DataContractError, match="date"):
        validate_prices(_prices(value))


@pytest.mark.parametrize("column", ["effective_from", "effective_to"])
@pytest.mark.parametrize("value", INVALID_DATES)
def test_membership_boundaries_reject_non_daily_dates(
    column: str, value: object
) -> None:
    frame = _memberships().assign(**{column: pd.Series([value], dtype=object)})

    with pytest.raises(DataContractError, match=column):
        validate_memberships(frame)


@pytest.mark.parametrize(
    "value", [None, pd.NA, pd.NaT, np.nan, np.datetime64("NaT", "ns")]
)
def test_membership_only_end_date_accepts_missing_values(value: object) -> None:
    source = _memberships().assign(effective_to=pd.Series([value], dtype=object))
    before = source.copy(deep=True)

    result = validate_memberships(source)

    assert pd.isna(result.loc[0, "effective_to"])
    pd.testing.assert_frame_equal(source, before)
    with pytest.raises(DataContractError, match="effective_from"):
        validate_memberships(
            source.assign(effective_from=pd.Series([value], dtype=object))
        )


@pytest.mark.parametrize("value", [True, False, np.bool_(True)])
def test_prices_reject_boolean_values_even_in_mixed_object_columns(
    value: object,
) -> None:
    source = pd.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-03"],
            "symbol": ["AAA", "AAA"],
            "adjusted_close": pd.Series([10.0, value], dtype=object),
        }
    )
    before = source.copy(deep=True)

    with pytest.raises(DataContractError, match="adjusted_close"):
        validate_prices(source)

    pd.testing.assert_frame_equal(source, before)


@pytest.mark.parametrize(
    "series",
    [
        pd.Series(pd.to_datetime(["2025-01-02", "2025-01-03"])),
        pd.Series(pd.to_datetime(["2025-01-02", "2025-01-03"], utc=True)),
        pd.Series(pd.to_timedelta([1, 2], unit="D")),
        pd.Series([10 + 1j, 11 + 2j], dtype="complex128"),
        pd.Series([10 + 0j, 11 + 0j], dtype="complex64"),
    ],
)
def test_prices_reject_temporal_and_complex_dtypes(series: pd.Series) -> None:
    source = pd.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-03"],
            "symbol": ["AAA", "AAA"],
            "adjusted_close": series,
        }
    )
    before = source.copy(deep=True)

    with pytest.raises(DataContractError, match="adjusted_close"):
        validate_prices(source)

    pd.testing.assert_frame_equal(source, before)


@pytest.mark.parametrize(
    "value",
    [
        date(2025, 1, 2),
        datetime(2025, 1, 2),
        pd.Timestamp("2025-01-02"),
        np.datetime64("2025-01-02"),
        timedelta(days=1),
        pd.Timedelta(days=1),
        np.timedelta64(1, "D"),
        10 + 1j,
        10 + 0j,
        np.complex64(10 + 2j),
        "10+1j",
        "10+0j",
    ],
)
def test_prices_reject_temporal_and_complex_scalars_in_object_columns(
    value: object,
) -> None:
    source = pd.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-03"],
            "symbol": ["AAA", "AAA"],
            "adjusted_close": pd.Series([10.0, value], dtype=object),
        }
    )
    before = source.copy(deep=True)

    with pytest.raises(DataContractError, match="adjusted_close"):
        validate_prices(source)

    pd.testing.assert_frame_equal(source, before)


@pytest.mark.parametrize(
    "value", ["10.25", "1.025e1", 10.25, Decimal("10.25"), np.float32(10.25)]
)
def test_prices_preserve_positive_real_numeric_inputs(value: object) -> None:
    source = _prices("2025-01-02").assign(
        adjusted_close=pd.Series([value], dtype=object)
    )
    before = source.copy(deep=True)

    result = validate_prices(source)

    assert result.loc[0, "adjusted_close"] == pytest.approx(10.25)
    pd.testing.assert_frame_equal(source, before)


@pytest.mark.parametrize(
    "validator,frame",
    [(validate_prices, _prices("2025-01-02")), (validate_memberships, _memberships())],
)
def test_duplicate_columns_fail_with_contract_error(
    validator, frame: pd.DataFrame
) -> None:
    duplicated = pd.concat([frame, frame.iloc[:, [0]]], axis=1)

    with pytest.raises(DataContractError, match="[Dd]uplicate.*columns"):
        validator(duplicated)


@pytest.mark.parametrize(
    "value",
    [
        "2025-01-02",
        "2025-01-02 00:00:00",
        date(2025, 1, 2),
        datetime(2025, 1, 2),
        pd.Timestamp("2025-01-02"),
        np.datetime64("2025-01-02"),
    ],
)
def test_normalized_daily_dates_are_accepted_without_mutating_inputs(
    value: object,
) -> None:
    prices = _prices(value)
    before = prices.copy(deep=True)

    result = validate_prices(prices)

    assert result.loc[0, "date"] == pd.Timestamp("2025-01-02")
    pd.testing.assert_frame_equal(prices, before)
    assert contracts.validate_as_of_dates([value]) == (pd.Timestamp("2025-01-02"),)


@pytest.mark.parametrize("value", INVALID_DATES + [None, pd.NA, pd.NaT, np.nan])
def test_decision_date_validator_rejects_invalid_values(value: object) -> None:
    with pytest.raises(DataContractError, match="as_of_dates"):
        contracts.validate_as_of_dates([value])


def test_decision_date_validator_preserves_order_and_enforces_sequence_contract() -> (
    None
):
    values = ["2025-01-03", "2025-01-02"]
    assert contracts.validate_as_of_dates(values) == (
        pd.Timestamp("2025-01-03"),
        pd.Timestamp("2025-01-02"),
    )
    assert values == ["2025-01-03", "2025-01-02"]
    for invalid in ([], ["2025-01-02", date(2025, 1, 2)], "2025-01-02", None):
        with pytest.raises(DataContractError, match="as_of_dates"):
            contracts.validate_as_of_dates(invalid)


@pytest.mark.parametrize("value", INVALID_DATES + [None, pd.NaT])
def test_panel_validates_decision_dates_before_calendar_lookup(
    market_fixture, value: object
) -> None:
    prices, memberships, _ = market_fixture
    with pytest.raises(DataContractError, match="as_of_dates"):
        build_point_in_time_panel(
            prices,
            memberships,
            as_of_dates=[value],
            config=PanelConfig(
                dataset_version="test-v1", universe_id="TEST", benchmark_symbol="BENCH"
            ),
        )
