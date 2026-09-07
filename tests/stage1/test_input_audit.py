from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest

from quant_metric_research.config import PanelConfig
from quant_metric_research.input_audit import audit_inputs


@pytest.fixture()
def audit_input():
    dates = pd.bdate_range("2025-01-02", periods=9)
    prices = pd.DataFrame(
        [
            {"date": day, "symbol": symbol, "adjusted_close": 100.0 + position}
            for symbol in ("BENCH", "AAA", "BBB")
            for position, day in enumerate(dates)
            if symbol != "BBB" or 2 <= position < 6
        ]
    )
    memberships = pd.DataFrame(
        {
            "universe_id": ["TEST"] * 3,
            "symbol": ["AAA", "BBB", "GHOST"],
            "effective_from": [dates[0], dates[3], dates[3]],
            "effective_to": [dates[4], None, None],
            "source": ["invented"] * 3,
        }
    )
    config = PanelConfig(
        dataset_version="invented-v1",
        universe_id="TEST",
        benchmark_symbol="BENCH",
        lookback_sessions=3,
        min_observations=2,
        target_horizon_sessions=2,
        entry_lag_sessions=1,
    )
    return prices, memberships, dates, config


def _audit(inputs, **overrides):
    prices, memberships, dates, config = inputs
    return audit_inputs(
        overrides.get("prices", prices),
        overrides.get("memberships", memberships),
        as_of_dates=overrides.get("as_of_dates", [dates[3], dates[4], dates[7]]),
        config=overrides.get("config", config),
    )


def test_audit_reports_historical_members_and_missing_label_endpoints(audit_input):
    report = _audit(audit_input)
    rows = report["coverage_by_date"]
    assert [row["active_members"] for row in rows] == [3, 2, 2]
    assert rows[0]["sufficient_return_history"] == 1
    assert rows[1]["sufficient_return_history"] == 1
    assert rows[2]["sufficient_return_history"] == 0
    # AAA leaves at entry but its future endpoint is still required and available.
    assert rows[0]["label_endpoints_available"] == 1
    assert rows[0]["missing_label_entry"] == 1
    assert rows[0]["missing_label_exit"] == 2
    assert rows[2]["label_calendar_unavailable"] == 2
    assert rows[2]["missing_label_entry"] == 0
    assert report["summary"]["active_member_dates"] == 7
    assert report["summary"]["members_without_any_prices"] == ["GHOST"]
    assert report["calendar"]["source"] == "supplied_benchmark_prices"
    assert report["stage4_eligible"] is False
    assert report["empirical_data_provenance_verified"] is False
    assert all(item["status"] == "unverified" for item in report["external_evidence"])
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def test_audit_never_calculates_features_returns_or_screening(audit_input, monkeypatch):
    import quant_metric_research.features as features
    import quant_metric_research.panel as panel
    import quant_metric_research.screening as screening

    def forbidden(*args, **kwargs):
        pytest.fail("Input audit must not calculate research outcomes.")

    monkeypatch.setattr(features, "compute_price_metrics", forbidden)
    monkeypatch.setattr(panel, "compute_price_metrics", forbidden)
    monkeypatch.setattr(panel, "build_point_in_time_panel", forbidden)
    monkeypatch.setattr(screening, "fit_metric_screen", forbidden)
    result = _audit(audit_input)
    assert result["claim_scope"] == "raw_input_diagnostics_only"
    assert result["no_outcomes_computed"] is True


def test_price_values_change_identity_but_not_presence_diagnostics(audit_input):
    prices, _, _, _ = audit_input
    changed = prices.assign(adjusted_close=prices["adjusted_close"] * 3.0)
    first = _audit(audit_input)
    second = _audit(audit_input, prices=changed)
    assert first["coverage_by_date"] == second["coverage_by_date"]
    assert first["coverage_by_security"] == second["coverage_by_security"]
    assert (
        first["input_fingerprints"]["prices"] != second["input_fingerprints"]["prices"]
    )


def test_report_records_configuration_and_fingerprint_scope(audit_input):
    from quant_metric_research import __version__

    prices, memberships, dates, config = audit_input
    report = _audit(audit_input)
    assert report["package_version"] == __version__
    assert report["config"]["target_horizon_sessions"] == config.target_horizon_sessions
    assert report["config"]["feature_columns"] == list(config.feature_columns)
    # Extra fields are explicitly outside the normalized required-column identity.
    extras = _audit(audit_input, prices=prices.assign(provider_note="not hashed"))
    assert extras == report
    changed = _audit(audit_input, config=replace(config, target_horizon_sessions=1))
    assert (
        changed["input_fingerprints"]["request"]
        != report["input_fingerprints"]["request"]
    )
    changed = _audit(audit_input, memberships=memberships.assign(source="new source"))
    assert (
        changed["input_fingerprints"]["memberships"]
        != report["input_fingerprints"]["memberships"]
    )
    changed = _audit(audit_input, as_of_dates=[dates[3]])
    assert (
        changed["input_fingerprints"]["request"]
        != report["input_fingerprints"]["request"]
    )


def test_audit_is_deterministic_and_does_not_mutate_inputs(audit_input):
    prices, memberships, dates, _ = audit_input
    original_prices, original_memberships = (
        prices.copy(deep=True),
        memberships.copy(deep=True),
    )
    first = _audit(audit_input)
    reordered = _audit(
        audit_input,
        prices=prices.sample(frac=1, random_state=2),
        memberships=memberships.iloc[::-1],
        as_of_dates=[dates[7], dates[3], dates[4]],
    )
    assert first == reordered
    pd.testing.assert_frame_equal(prices, original_prices)
    pd.testing.assert_frame_equal(memberships, original_memberships)


def test_empty_membership_dates_are_visible(audit_input):
    prices, memberships, dates, config = audit_input
    report = _audit(audit_input, config=replace(config, universe_id="NOT_IN_DATA"))
    assert all(row["active_members"] == 0 for row in report["coverage_by_date"])
    assert report["summary"]["active_member_dates"] == 0
    assert "no_active_members" in {warning["code"] for warning in report["warnings"]}
    assert report["coverage_by_security"] == []


def test_off_calendar_prices_are_reported_not_silently_used(audit_input):
    prices, _, _, _ = audit_input
    extra = pd.DataFrame(
        [{"date": pd.Timestamp("2025-01-04"), "symbol": "AAA", "adjusted_close": 1.0}]
    )
    report = _audit(audit_input, prices=pd.concat([prices, extra], ignore_index=True))
    assert report["summary"]["off_calendar_price_rows"] == 1
    assert report["coverage_by_date"] == _audit(audit_input)["coverage_by_date"]
    assert "off_calendar_prices" in {warning["code"] for warning in report["warnings"]}


@pytest.mark.parametrize("bad_kind", ["benchmark", "decision", "config"])
def test_audit_rejects_invalid_inputs(audit_input, bad_kind):
    prices, _, _, _ = audit_input
    overrides = {
        "benchmark": {"prices": prices.loc[prices["symbol"] != "BENCH"]},
        "decision": {"as_of_dates": ["2025-01-04"]},
        "config": {"config": {}},
    }
    with pytest.raises(ValueError):
        _audit(audit_input, **overrides[bad_kind])


def test_audit_adjacent_return_counts_match_stage1_availability(audit_input):
    from quant_metric_research.panel import build_point_in_time_panel

    prices, memberships, dates, config = audit_input
    decisions = [dates[3], dates[4], dates[7]]
    panel = build_point_in_time_panel(
        prices, memberships, as_of_dates=decisions, config=config
    )
    report = _audit(audit_input)
    for row in report["coverage_by_date"]:
        actual = panel.loc[panel["as_of_date"] == pd.Timestamp(row["as_of_date"])]
        assert row["sufficient_return_history"] == int(actual["feature_eligible"].sum())
        assert row["label_endpoints_available"] == int(actual["target_available"].sum())
