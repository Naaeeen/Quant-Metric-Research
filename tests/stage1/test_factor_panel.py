from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from quant_metric_research import factor_panel
from quant_metric_research.config import PanelConfig
from quant_metric_research.contracts import DataContractError
from quant_metric_research.panel import build_point_in_time_panel
from quant_metric_research.price_factors import compute_price_factors

NAMES = ("return_21s", "momentum_252s_skip_21s", "ma_distance_63s")
SUFFIXES = (
    "",
    "_available_at",
    "_status",
    "_window_start",
    "_window_end",
    "_required_price_count",
    "_observed_price_count",
    "_formula_version",
)


def _inputs():
    dates = pd.bdate_range("2020-01-02", periods=280)
    prices = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": symbol,
                    "adjusted_close": scale * (100.0 + np.arange(len(dates))),
                }
            )
            for symbol, scale in (("BENCH", 1.1), ("AAA", 1.0), ("BBB", 2.0))
        ],
        ignore_index=True,
    )
    memberships = pd.DataFrame(
        {
            "universe_id": ["TEST"] * 3,
            "symbol": ["AAA", "BBB", "EMPTY"],
            "effective_from": [dates[0], dates[63], dates[0]],
            "effective_to": [pd.NaT, dates[253], pd.NaT],
            "source": ["synthetic"] * 3,
        }
    )
    config = PanelConfig(
        dataset_version="synthetic-factor-panel-v1",
        universe_id="TEST",
        benchmark_symbol="BENCH",
    )
    return prices, memberships, dates, config


def _build(prices, memberships, dates, config, names=NAMES):
    return factor_panel.build_factor_panel(
        prices, memberships, as_of_dates=dates, config=config, factor_names=names
    )


def _extra_columns(names=NAMES):
    return [name + suffix for name in names for suffix in SUFFIXES]


@pytest.mark.parametrize("subset", [None, ("beta", "trailing_return")])
def test_legacy_frame_is_preserved_exactly_and_inputs_are_not_mutated(subset):
    prices, memberships, dates, config = _inputs()
    if subset is not None:
        config = replace(config, feature_columns=subset)
    decisions = [dates[252], dates[21], dates[-1], dates[62]]
    original_prices, original_members = (
        prices.copy(deep=True),
        memberships.copy(deep=True),
    )
    before_decisions = decisions.copy()
    legacy = build_point_in_time_panel(
        prices, memberships, as_of_dates=decisions, config=config
    )

    actual = _build(prices, memberships, decisions, config, NAMES[::-1])

    assert_frame_equal(actual.loc[:, legacy.columns], legacy, check_exact=True)
    assert list(actual.columns) == [*legacy.columns, *_extra_columns(NAMES[::-1])]
    assert_frame_equal(prices, original_prices, check_exact=True)
    assert_frame_equal(memberships, original_members, check_exact=True)
    assert decisions == before_decisions


def test_every_active_row_matches_pure_calculator_and_has_explicit_dtypes():
    prices, memberships, dates, config = _inputs()
    decisions = [dates[21], dates[252], dates[-1]]
    actual = _build(prices, memberships, decisions, config)
    for row in actual.itertuples(index=False):
        series = prices.loc[prices["symbol"].eq(row.symbol)].set_index("date")[
            "adjusted_close"
        ]
        expected = compute_price_factors(
            series, calendar=dates, as_of_date=row.as_of_date, factor_names=NAMES
        )
        for value in expected.factors:
            assert getattr(row, value.name + "_status") == value.status
            assert (
                getattr(row, value.name + "_required_price_count")
                == value.required_price_count
            )
            assert (
                getattr(row, value.name + "_observed_price_count")
                == value.observed_price_count
            )
            assert (
                getattr(row, value.name + "_formula_version") == value.formula_version
            )
            for suffix, timestamp in (
                ("_window_start", value.window_start),
                ("_window_end", value.window_end),
            ):
                stored = getattr(row, value.name + suffix)
                assert pd.isna(stored) if timestamp is None else stored == timestamp
            if value.status == "ok":
                assert getattr(row, value.name) == value.value
                assert getattr(row, value.name + "_available_at") == row.as_of_date
            else:
                assert pd.isna(getattr(row, value.name))
                assert pd.isna(getattr(row, value.name + "_available_at"))
    expected_types = (
        "float64",
        "datetime64[ns]",
        "string",
        "datetime64[ns]",
        "datetime64[ns]",
        "int64",
        "int64",
        "string",
    )
    for name in NAMES:
        assert (
            tuple(str(actual[name + suffix].dtype) for suffix in SUFFIXES)
            == expected_types
        )


def test_membership_boundaries_missing_stocks_and_labels_never_filter_factor_rows():
    prices, memberships, dates, config = _inputs()
    actual = _build(
        prices,
        memberships,
        [dates[21], dates[62], dates[63], dates[252], dates[253], dates[-1]],
        config,
    )
    bbb_dates = actual.loc[actual["symbol"].eq("BBB"), "as_of_date"].tolist()
    assert bbb_dates == [dates[63], dates[252]]
    early = actual.loc[
        actual["symbol"].eq("AAA") & actual["as_of_date"].eq(dates[21])
    ].iloc[0]
    assert not early["feature_eligible"]
    assert early["return_21s_status"] == "ok"
    final = actual.loc[
        actual["symbol"].eq("AAA") & actual["as_of_date"].eq(dates[-1])
    ].iloc[0]
    assert not final["target_available"]
    assert all(final[name + "_status"] == "ok" for name in NAMES)
    empty = actual.loc[
        actual["symbol"].eq("EMPTY") & actual["as_of_date"].eq(dates[252])
    ].iloc[0]
    assert all(empty[name + "_status"] == "missing_required_prices" for name in NAMES)
    assert all(empty[name + "_observed_price_count"] == 0 for name in NAMES)


def test_missing_interior_price_is_not_compressed_or_shared_across_factor_windows():
    prices, memberships, dates, config = _inputs()
    changed = prices.loc[~(prices["symbol"].eq("AAA") & prices["date"].eq(dates[10]))]
    actual = _build(changed, memberships, [dates[252]], config)
    row = actual.loc[actual["symbol"].eq("AAA")].iloc[0]
    assert row["return_21s_status"] == row["ma_distance_63s_status"] == "ok"
    assert row["momentum_252s_skip_21s_status"] == "missing_required_prices"
    assert row["momentum_252s_skip_21s_observed_price_count"] == 231


def test_raw_price_scalars_and_normalized_keys_reach_calculator(monkeypatch):
    prices, memberships, dates, config = _inputs()
    changed = prices.assign(
        date=prices["date"].dt.strftime("%Y-%m-%d"),
        symbol=prices["symbol"].str.lower().map(lambda value: " " + value + " "),
        adjusted_close=prices["adjusted_close"].astype(str),
    )
    changed = changed.sample(frac=1, random_state=7)
    changed.index = [index % 7 for index in range(len(changed))]
    calls = []
    original = compute_price_factors

    def observed(series, **kwargs):
        calls.append(series)
        assert all(isinstance(value, str) for value in series)
        return original(series, **kwargs)

    monkeypatch.setattr(factor_panel, "compute_price_factors", observed)
    actual = _build(changed, memberships, [dates[252]], config)
    assert len(calls) == len(actual) == 3
    assert actual["symbol"].tolist() == ["AAA", "BBB", "EMPTY"]


def test_valid_future_strings_cannot_change_factor_rounding():
    prices, memberships, dates, config = _inputs()
    changed = prices.assign(adjusted_close=prices["adjusted_close"].astype(str))
    aaa = changed["symbol"].eq("AAA")
    changed.loc[aaa, "adjusted_close"] = [
        str(2**60 + index * 13) for index in range(len(dates))
    ]
    baseline = _build(changed, memberships, [dates[252]], config)
    future = changed.copy(deep=True)
    future.loc[aaa & future["date"].eq(dates[273]), "adjusted_close"] = "1.2345"
    actual = _build(future, memberships, [dates[252]], config)
    assert_frame_equal(
        actual[_extra_columns()], baseline[_extra_columns()], check_exact=True
    )
    assert not actual["forward_excess_return"].equals(baseline["forward_excess_return"])


@pytest.mark.parametrize("bad", ["bad", np.nan, 0, True, complex(1, 0), np.inf])
def test_legacy_whole_table_validation_still_rejects_malformed_future_prices(bad):
    prices, memberships, dates, config = _inputs()
    changed = prices.assign(adjusted_close=prices["adjusted_close"].astype(object))
    changed.loc[
        changed["symbol"].eq("AAA") & changed["date"].eq(dates[-1]), "adjusted_close"
    ] = bad
    with pytest.raises(DataContractError):
        _build(changed, memberships, [dates[252]], config)


def test_active_off_calendar_history_is_rejected_without_changing_calendar():
    prices, memberships, dates, config = _inputs()
    extra = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-04")],
            "symbol": ["AAA"],
            "adjusted_close": [100.0],
        }
    )
    with pytest.raises(DataContractError, match="off-calendar"):
        _build(
            pd.concat([prices, extra], ignore_index=True),
            memberships,
            [dates[252]],
            config,
        )


@pytest.mark.parametrize(
    "names", [(), [], NAMES[0], ("unknown",), (NAMES[0], NAMES[0]), (None,)]
)
def test_factor_names_are_explicit_known_unique_and_nonempty(names, monkeypatch):
    prices, memberships, dates, config = _inputs()

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Invalid factor request must fail before panel construction."
        )

    monkeypatch.setattr(factor_panel, "build_point_in_time_panel", forbidden)
    with pytest.raises(DataContractError, match="factor_names"):
        _build(prices, memberships, [dates[252]], config, names)


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_every_appended_column_collision_is_rejected_before_factor_computation(
    suffix, monkeypatch
):
    prices, memberships, dates, config = _inputs()
    legacy = build_point_in_time_panel(
        prices, memberships, as_of_dates=[dates[252]], config=config
    )
    collision = NAMES[0] + suffix
    collided = legacy.assign(**{collision: "existing"})
    monkeypatch.setattr(
        factor_panel, "build_point_in_time_panel", lambda *args, **kwargs: collided
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Collision must fail before factor calculation.")

    monkeypatch.setattr(factor_panel, "compute_price_factors", forbidden)
    with pytest.raises(DataContractError, match="collid"):
        _build(prices, memberships, [dates[252]], config)


def test_legacy_builder_is_called_once_and_returned_index_is_preserved(monkeypatch):
    prices, memberships, dates, config = _inputs()
    original = build_point_in_time_panel
    calls = []

    def observed(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs).set_axis(
            pd.Index([7, 3, 7], name="original_row")
        )

    monkeypatch.setattr(factor_panel, "build_point_in_time_panel", observed)
    actual = _build(prices, memberships, [dates[252]], config)
    assert calls == [1]
    assert actual.index.tolist() == [7, 3, 7]
    assert actual.index.name == "original_row"


def test_non_panel_config_is_rejected():
    prices, memberships, dates, _ = _inputs()
    with pytest.raises(DataContractError, match="PanelConfig"):
        _build(prices, memberships, [dates[252]], None)
