from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from quant_metric_research.archive_sample import normalize_mendeley_prices
from quant_metric_research.contracts import DataContractError


def _stocks() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": ["2012-01-03", "2012-01-04", "2012-01-05"],
            "CCC": [30.0, 31.0, 32.0],
            " bbb ": [20.0, 21.0, 22.0],
            "AAA": [10.0, None, 12.0],
        }
    )


def _factors() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": ["1/2/2003", "1/3/2012", "1/4/2012", "1/5/2012"],
            "Mkt-RF": [2.0, 7.0, 1.0, -2.0],
            "RF": [0.0, 0.0, 0.1, 0.0],
        }
    )


def test_fixed_alphabetic_cohort_keeps_missing_member_and_preserves_inputs() -> None:
    stocks, factors = _stocks(), _factors()
    original_stocks, original_factors = stocks.copy(deep=True), factors.copy(deep=True)

    sample = normalize_mendeley_prices(stocks, factors, cohort_size=2)

    assert sample.symbols == ("AAA", "BBB")
    assert list(sample.prices.columns) == ["date", "symbol", "adjusted_close"]
    assert set(sample.prices["symbol"]) == {"AAA", "BBB", "FF_MARKET_PROXY"}
    assert sample.missing_prices.to_dict("records") == [
        {
            "date": pd.Timestamp("2012-01-04"),
            "symbol": "AAA",
            "reason": "missing_adjusted_close",
        }
    ]
    pd.testing.assert_frame_equal(stocks, original_stocks)
    pd.testing.assert_frame_equal(factors, original_factors)
    with pytest.raises(FrozenInstanceError):
        sample.symbols = ()  # type: ignore[misc]


def test_benchmark_chains_percent_components_after_first_day_base() -> None:
    sample = normalize_mendeley_prices(_stocks(), _factors(), cohort_size=2)
    benchmark = sample.prices.loc[sample.prices["symbol"] == "FF_MARKET_PROXY"]

    assert benchmark["adjusted_close"].to_list() == pytest.approx([100, 101.1, 99.078])
    assert benchmark["date"].min() == pd.Timestamp("2012-01-03")
    assert "research" in sample.profile["benchmark"]["construction"].lower()


def test_profile_reports_null_coverage_by_symbol_and_year_without_filling() -> None:
    stocks = _stocks().assign(AAA=[None, None, None])
    sample = normalize_mendeley_prices(stocks, _factors(), cohort_size=2)
    by_symbol = {row["symbol"]: row for row in sample.profile["coverage_by_symbol"]}

    assert sample.symbols == ("AAA", "BBB")
    assert by_symbol["AAA"]["missing_count"] == 3
    assert by_symbol["AAA"]["missing_rate"] == 1.0
    assert sample.profile["coverage_by_symbol_year"][0]["year"] == 2012
    assert sample.profile["empirical_data_provenance_verified"] is False
    assert sample.profile["stage4_eligible"] is False
    assert "forward" not in json.dumps(sample.profile, allow_nan=False).lower()


@pytest.mark.parametrize("cohort_size", [0, -1, True, 1.5, 4])
def test_rejects_invalid_or_oversized_cohort(cohort_size: object) -> None:
    with pytest.raises(DataContractError, match="cohort_size"):
        normalize_mendeley_prices(_stocks(), _factors(), cohort_size=cohort_size)


@pytest.mark.parametrize("dataset", ["stocks", "factors"])
def test_rejects_duplicate_dates_even_on_null_rows(dataset: str) -> None:
    stocks, factors = _stocks(), _factors()
    if dataset == "stocks":
        stocks = stocks.assign(Date=["2012-01-03", "2012-01-03", "2012-01-05"])
    else:
        factors = pd.concat([factors, factors.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataContractError, match="unique|Duplicate"):
        normalize_mendeley_prices(stocks, factors, cohort_size=2)


@pytest.mark.parametrize(
    "bad_date",
    [
        "01/03/2012",
        "2012-1-3",
        "2012-01-03T00:00:00",
        "2012-01-03T12:00:00",
        "2012-02-30",
        None,
        20120103,
    ],
)
def test_rejects_malformed_stock_dates_even_when_all_prices_null(
    bad_date: object,
) -> None:
    stocks = _stocks().assign(AAA=[None, None, None], **{" bbb ": [None, None, None]})
    stocks["Date"] = pd.Series([bad_date, "2012-01-04", "2012-01-05"], dtype=object)
    with pytest.raises(DataContractError, match="date|Date"):
        normalize_mendeley_prices(stocks, _factors(), cohort_size=2)


@pytest.mark.parametrize(
    "bad_date", ["2003-01-02", "2/30/2003", "1/2/2003 00:00:00", None, True]
)
def test_rejects_malformed_factor_dates_outside_sample_window(bad_date: object) -> None:
    factors = _factors()
    factors["Date"] = pd.Series(
        [bad_date, "1/3/2012", "1/4/2012", "1/5/2012"], dtype=object
    )
    with pytest.raises(DataContractError, match="date|Date"):
        normalize_mendeley_prices(_stocks(), factors, cohort_size=2)


@pytest.mark.parametrize("dataset", ["stocks", "factors"])
def test_rejects_duplicate_columns(dataset: str) -> None:
    stocks, factors = _stocks(), _factors()
    if dataset == "stocks":
        stocks = pd.concat([stocks, stocks[["CCC"]]], axis=1)
    else:
        factors = pd.concat([factors, factors[["RF"]]], axis=1)
    with pytest.raises(DataContractError, match="Duplicate"):
        normalize_mendeley_prices(stocks, factors, cohort_size=2)


@pytest.mark.parametrize("alias", ["aaa", " AAA ", "FF_MARKET_PROXY", " ", None, 12])
def test_rejects_header_aliases_invalid_symbols_and_benchmark_collision(
    alias: object,
) -> None:
    stocks = pd.concat([_stocks(), pd.DataFrame({alias: [5.0, 6.0, 7.0]})], axis=1)
    with pytest.raises(DataContractError, match="symbol|Symbol|alias|collision"):
        normalize_mendeley_prices(stocks, _factors(), cohort_size=2)


@pytest.mark.parametrize(
    "invalid",
    [0, -1, np.inf, -np.inf, True, 1 + 2j, "bad", "", pd.Timestamp("2012-01-03")],
)
def test_rejects_present_invalid_selected_prices(invalid: object) -> None:
    stocks = _stocks()
    stocks["AAA"] = pd.Series([10.0, invalid, 12.0], dtype=object)
    with pytest.raises(DataContractError, match="positive|price"):
        normalize_mendeley_prices(stocks, _factors(), cohort_size=2)


@pytest.mark.parametrize("component", ["Mkt-RF", "RF"])
@pytest.mark.parametrize(
    "invalid", [None, np.nan, np.inf, True, "bad", -99.99, -999, 1 + 2j]
)
def test_rejects_missing_invalid_and_sentinel_factor_components(
    component: str, invalid: object
) -> None:
    factors = _factors()
    factors[component] = factors[component].astype(object)
    factors.loc[1, component] = invalid
    with pytest.raises(DataContractError, match="factor|return|sentinel"):
        normalize_mendeley_prices(_stocks(), factors, cohort_size=2)


def test_rejects_total_returns_at_or_below_minus_one_hundred_percent() -> None:
    factors = _factors().assign(**{"Mkt-RF": [0, -90, 1, 0], "RF": [0, -10, 0, 0]})
    with pytest.raises(DataContractError, match="return"):
        normalize_mendeley_prices(_stocks(), factors, cohort_size=2)


def test_rejects_benchmark_compounding_overflow() -> None:
    factors = _factors().assign(**{"Mkt-RF": [0, 0, 1e307, 1e307]})
    with pytest.raises(DataContractError, match="positive|finite|benchmark"):
        normalize_mendeley_prices(_stocks(), factors, cohort_size=2)


def test_missing_benchmark_date_fails_without_calendar_intersection() -> None:
    factors = _factors().drop(index=2)
    with pytest.raises(DataContractError, match="benchmark.*date|calendar"):
        normalize_mendeley_prices(_stocks(), factors, cohort_size=2)


def test_extra_benchmark_date_is_retained_and_compounded() -> None:
    stocks = _stocks().drop(index=1)
    sample = normalize_mendeley_prices(stocks, _factors(), cohort_size=2)
    benchmark = sample.prices.loc[sample.prices["symbol"] == "FF_MARKET_PROXY"]
    assert len(benchmark) == 3
    assert sample.profile["factor_dates_without_stock_rows"] == 1


def test_large_move_diagnostics_do_not_bridge_missing_values_or_clip_prices() -> None:
    stocks = _stocks().assign(AAA=[1.0, None, 100.0], **{" bbb ": [1.0, 100.0, 100.0]})
    sample = normalize_mendeley_prices(stocks, _factors(), cohort_size=2)
    diagnostics = sample.profile["large_daily_moves"]
    assert diagnostics["count"] == 1
    assert diagnostics["examples"][0]["symbol"] == "BBB"
    assert (
        sample.prices.loc[sample.prices["symbol"] == "BBB", "adjusted_close"].max()
        == 100
    )


def test_fixed_window_excludes_outside_rows_without_changing_symbol_selection() -> None:
    stocks = pd.concat(
        [
            _stocks(),
            pd.DataFrame(
                {"Date": ["2017-01-03"], "CCC": [3.0], " bbb ": [2.0], "AAA": [1.0]}
            ),
        ],
        ignore_index=True,
    )
    factors = pd.concat(
        [
            _factors(),
            pd.DataFrame({"Date": ["1/3/2017"], "Mkt-RF": [1.0], "RF": [0.0]}),
        ],
        ignore_index=True,
    )
    sample = normalize_mendeley_prices(stocks, factors, cohort_size=2)
    assert sample.prices["date"].max() == pd.Timestamp("2012-01-05")
    assert sample.profile["source_window"] == {
        "start": "2012-01-03",
        "end": "2016-12-30",
    }


@pytest.mark.parametrize("dataset", ["stocks", "factors"])
def test_empty_or_missing_required_frames_fail(dataset: str) -> None:
    stocks, factors = _stocks(), _factors()
    if dataset == "stocks":
        stocks = stocks.iloc[:0]
    else:
        factors = factors.drop(columns="RF")
    with pytest.raises(DataContractError):
        normalize_mendeley_prices(stocks, factors, cohort_size=2)


@pytest.mark.parametrize("dataset", ["stocks", "factors"])
def test_non_dataframe_inputs_fail_with_contract_error(dataset: str) -> None:
    stocks, factors = _stocks(), _factors()
    with pytest.raises(DataContractError, match="DataFrame"):
        normalize_mendeley_prices(
            stocks.to_dict() if dataset == "stocks" else stocks,
            factors.to_dict() if dataset == "factors" else factors,
            cohort_size=2,
        )


def test_window_with_no_stock_dates_fails() -> None:
    stocks = _stocks().assign(Date=["2003-01-02", "2003-01-03", "2003-01-06"])
    with pytest.raises(DataContractError, match="source window"):
        normalize_mendeley_prices(stocks, _factors(), cohort_size=2)


def test_all_null_selected_members_survive_without_substitution() -> None:
    stocks = _stocks().assign(AAA=[None] * 3, **{" bbb ": [None] * 3})
    sample = normalize_mendeley_prices(stocks, _factors(), cohort_size=2)
    assert sample.symbols == ("AAA", "BBB")
    assert len(sample.missing_prices) == 6
    assert set(sample.prices["symbol"]) == {"FF_MARKET_PROXY"}
    assert sample.profile["large_daily_moves"]["count"] == 0
    json.dumps(sample.profile, allow_nan=False)


@pytest.mark.parametrize("dtype", ["float32", "object"])
def test_rejects_sentinel_at_original_float32_precision(dtype: str) -> None:
    factors = _factors()
    factors["Mkt-RF"] = pd.Series([0.0, np.float32(-99.99), 1.0, 2.0], dtype=dtype)
    with pytest.raises(DataContractError, match="sentinel"):
        normalize_mendeley_prices(_stocks(), factors, cohort_size=2)


def test_dataframe_index_names_cannot_change_source_date_interpretation() -> None:
    stocks = _stocks().rename_axis("Date")
    factors = _factors().rename_axis("date")
    actual = normalize_mendeley_prices(stocks, factors, cohort_size=2)
    expected = normalize_mendeley_prices(_stocks(), _factors(), cohort_size=2)
    pd.testing.assert_frame_equal(actual.prices, expected.prices)
