from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from quant_metric_research.contracts import DataContractError
from quant_metric_research.yahoo_import import YahooPriceExport, normalize_yahoo_export


def _export(value: object = 10.25) -> pd.DataFrame:
    return pd.DataFrame(
        {"Date": ["2025-01-02"], "Adj Close": pd.Series([value], dtype=object)}
    )


def test_normalizes_adjusted_close_and_preserves_the_source() -> None:
    source = pd.DataFrame(
        {
            "Date": ["2025-01-06", "2025-01-02", "2025-01-03"],
            "Adj Close": ["12.25", "10.25", "11.25"],
            "Close": [100.0, 100.0, 100.0],
            "Volume": [1, 2, 3],
        },
        index=[8, 4, 4],
    )
    before = source.copy(deep=True)

    result = normalize_yahoo_export(source, symbol=" bhp.ax ")

    assert isinstance(result, YahooPriceExport)
    assert list(result.prices.columns) == ["date", "symbol", "adjusted_close"]
    assert result.prices["date"].tolist() == list(
        pd.bdate_range("2025-01-02", periods=3)
    )
    assert result.prices["symbol"].tolist() == ["BHP.AX"] * 3
    assert result.prices["adjusted_close"].tolist() == [10.25, 11.25, 12.25]
    assert result.prices.index.tolist() == [0, 1, 2]
    assert list(result.missing_prices.columns) == ["date", "symbol", "reason"]
    assert result.missing_prices.empty
    pd.testing.assert_frame_equal(source, before)


def test_missing_adjusted_prices_are_visible_and_never_replaced_with_close() -> None:
    source = pd.DataFrame(
        {
            "Date": ["2025-01-06", "2025-01-02", "2025-01-03"],
            "Adj Close": [None, np.nan, 11.25],
            "Close": [12.25, 10.25, 11.25],
        }
    )
    before = source.copy(deep=True)

    result = normalize_yahoo_export(source, symbol=" ^axjo ")

    assert result.prices["date"].tolist() == [pd.Timestamp("2025-01-03")]
    assert result.prices["symbol"].tolist() == ["^AXJO"]
    assert result.missing_prices.to_dict("records") == [
        {
            "date": pd.Timestamp(day),
            "symbol": "^AXJO",
            "reason": "missing_adjusted_close",
        }
        for day in ["2025-01-02", "2025-01-06"]
    ]
    pd.testing.assert_frame_equal(source, before)


@pytest.mark.parametrize("value", [None, np.nan, pd.NA])
def test_all_missing_export_keeps_every_missing_observation(value: object) -> None:
    result = normalize_yahoo_export(_export(value), symbol="AAA")

    assert result.prices.empty
    assert len(result.missing_prices) == 1
    assert result.missing_prices.loc[0, "reason"] == "missing_adjusted_close"


def test_empty_export_with_required_headers_has_stable_output_schemas() -> None:
    result = normalize_yahoo_export(
        pd.DataFrame(columns=["Date", "Adj Close"]), symbol="AAA"
    )

    assert list(result.prices.columns) == ["date", "symbol", "adjusted_close"]
    assert list(result.missing_prices.columns) == ["date", "symbol", "reason"]
    assert result.prices.empty
    assert result.missing_prices.empty
    assert str(result.prices["date"].dtype) == "datetime64[ns]"
    assert str(result.missing_prices["date"].dtype) == "datetime64[ns]"


@pytest.mark.parametrize("symbol", [None, 123, True, pd.NA, [], "", " \t\n"])
def test_symbol_must_be_a_nonempty_string(symbol: object) -> None:
    with pytest.raises(DataContractError, match="symbol"):
        normalize_yahoo_export(_export(), symbol=symbol)


@pytest.mark.parametrize(
    "frame", [None, [], {"Date": ["2025-01-02"], "Adj Close": [1]}]
)
def test_rejects_non_dataframe_exports(frame: object) -> None:
    with pytest.raises(DataContractError, match="DataFrame"):
        normalize_yahoo_export(frame, symbol="AAA")


@pytest.mark.parametrize("column", ["Date", "Adj Close"])
def test_requires_exact_headers_without_close_fallback(column: str) -> None:
    source = _export().assign(Close=10.25).drop(columns=column)

    with pytest.raises(DataContractError, match=column):
        normalize_yahoo_export(source, symbol="AAA")


@pytest.mark.parametrize("column", ["Date", "Adj Close", "Volume"])
def test_rejects_any_duplicate_column_names(column: str) -> None:
    source = _export().assign(Volume=1)
    duplicated = pd.concat([source, source[[column]]], axis=1)

    with pytest.raises(DataContractError, match="[Dd]uplicate.*columns"):
        normalize_yahoo_export(duplicated, symbol="AAA")


def test_rejects_yfinance_multiindex_columns() -> None:
    source = pd.DataFrame(
        [["2025-01-02", 10.25]],
        columns=pd.MultiIndex.from_tuples([("Date", ""), ("Adj Close", "AAA")]),
    )

    with pytest.raises(DataContractError, match="flat|MultiIndex"):
        normalize_yahoo_export(source, symbol="AAA")


def test_rejects_datetime_index_exports() -> None:
    source = pd.DataFrame({"Adj Close": [10.25]}, index=pd.to_datetime(["2025-01-02"]))

    with pytest.raises(DataContractError, match="Date.*column|DatetimeIndex"):
        normalize_yahoo_export(source, symbol="AAA")


@pytest.mark.parametrize("prices", [[10.25, 11.25], [10.25, None], [None, None]])
def test_rejects_duplicate_dates_including_missing_price_rows(prices: list) -> None:
    source = pd.DataFrame(
        {"Date": ["2025-01-02", date(2025, 1, 2)], "Adj Close": prices}
    )

    with pytest.raises(DataContractError, match="unique|[Dd]uplicate"):
        normalize_yahoo_export(source, symbol="AAA")


@pytest.mark.parametrize(
    "value",
    [
        "2025-01-02 12:00:00",
        "2025-01-02T00:00:00+00:00",
        pd.Timestamp("2025-01-02", tz="Australia/Sydney"),
        0,
        1735776000000000000,
        1.5,
        True,
        "2025-02-30",
        "bad",
        "",
        None,
        pd.NA,
        pd.NaT,
        np.nan,
        ["2025-01-02"],
    ],
)
@pytest.mark.parametrize("price", [10.25, None])
def test_rejects_invalid_dates_even_when_adjusted_price_is_missing(
    value: object, price: object
) -> None:
    source = _export(price).assign(Date=pd.Series([value], dtype=object))

    with pytest.raises(DataContractError, match="date"):
        normalize_yahoo_export(source, symbol="AAA")


@pytest.mark.parametrize(
    "value",
    [
        "bad",
        "",
        " ",
        "null",
        "nan",
        "NaN",
        "Infinity",
        np.inf,
        -np.inf,
        0,
        -0.25,
        True,
        np.bool_(False),
        10 + 0j,
        10 + 1j,
        np.complex64(10 + 1j),
        date(2025, 1, 2),
        pd.Timestamp("2025-01-02"),
        np.datetime64("2025-01-02"),
        timedelta(days=1),
        pd.Timedelta(days=1),
        np.timedelta64(1, "D"),
        [10.25],
        {"price": 10.25},
    ],
)
def test_rejects_malformed_or_invalid_prices_without_coercing_them_to_gaps(
    value: object,
) -> None:
    source = _export(value)
    before = source.copy(deep=True)

    with pytest.raises(DataContractError, match="adjusted_close"):
        normalize_yahoo_export(source, symbol="AAA")

    pd.testing.assert_frame_equal(source, before)


@pytest.mark.parametrize(
    "values",
    [
        pd.Series([True]),
        pd.Series([10 + 1j]),
        pd.Series(pd.to_datetime(["2025-01-02"])),
        pd.Series(pd.to_datetime(["2025-01-02"], utc=True)),
        pd.Series(pd.to_timedelta([1], unit="D")),
    ],
)
def test_rejects_temporal_boolean_and_complex_price_dtypes(values: pd.Series) -> None:
    source = _export().assign(**{"Adj Close": values})

    with pytest.raises(DataContractError, match="adjusted_close"):
        normalize_yahoo_export(source, symbol="AAA")


@pytest.mark.parametrize("value", [10.25, "10.25", "1.025e1", Decimal("10.25")])
def test_accepts_real_numeric_price_values(value: object) -> None:
    result = normalize_yahoo_export(_export(value), symbol="AAA")

    assert result.prices.loc[0, "adjusted_close"] == pytest.approx(10.25)


def test_result_container_is_frozen() -> None:
    result = normalize_yahoo_export(_export(), symbol="AAA")

    with pytest.raises(FrozenInstanceError):
        result.prices = pd.DataFrame()
