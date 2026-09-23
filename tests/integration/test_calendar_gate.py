from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, replace

import pandas as pd
import pytest

from quant_metric_research import (
    PanelConfig,
    audit_inputs,
    build_factor_panel,
    build_point_in_time_panel,
    run_research,
)
from quant_metric_research.cli import main
from quant_metric_research.contracts import DataContractError


@pytest.fixture()
def inputs():
    from quant_metric_research.session_calendar import ExpectedSessionCalendar

    dates = pd.bdate_range("2025-01-02", periods=12)
    prices = pd.DataFrame(
        [
            {"date": day, "symbol": symbol, "adjusted_close": 100.0 + i + k * i**2}
            for k, symbol in enumerate(("BENCH", "AAA", "BBB"))
            for i, day in enumerate(dates)
        ]
    )
    memberships = pd.DataFrame(
        {
            "universe_id": ["TEST", "TEST"],
            "symbol": ["AAA", "BBB"],
            "effective_from": [dates[0], dates[0]],
            "effective_to": [None, None],
            "source": ["invented", "invented"],
        }
    )
    config = PanelConfig(
        dataset_version="calendar-fixture-v1",
        universe_id="TEST",
        benchmark_symbol="BENCH",
        lookback_sessions=4,
        min_observations=2,
        target_horizon_sessions=2,
        entry_lag_sessions=1,
    )
    calendar = ExpectedSessionCalendar(
        sessions=tuple(dates),
        coverage_start=dates[0],
        coverage_end=dates[-1],
        source="invented-session-list",
        version="1",
    )
    return prices, memberships, dates, config, calendar


def _forbidden(*args, **kwargs):
    pytest.fail("Calendar mismatch must stop calculations before they begin.")


@pytest.mark.parametrize("builder", ["legacy", "factors", "research"])
def test_calendar_gap_blocks_every_builder_before_calculation(
    inputs, monkeypatch, builder
):
    prices, memberships, dates, config, calendar = inputs
    prices = prices.loc[prices["date"] != dates[3]]
    monkeypatch.setattr("quant_metric_research.panel.compute_price_metrics", _forbidden)
    monkeypatch.setattr(
        "quant_metric_research.factor_panel.compute_price_factors", _forbidden
    )
    monkeypatch.setattr("quant_metric_research.pipeline.fit_metric_screen", _forbidden)
    kwargs = dict(as_of_dates=[dates[6]], config=config, expected_calendar=calendar)
    functions = {
        "legacy": build_point_in_time_panel,
        "factors": build_factor_panel,
        "research": run_research,
    }
    if builder == "factors":
        kwargs["factor_names"] = ("return_21s",)
    elif builder == "research":
        kwargs.update(
            train_end_date=dates[-1],
            min_cross_section=2,
            minimum_coverage=0.0,
            redundancy_threshold=1.0,
            hac_lags=0,
        )
    with pytest.raises(DataContractError, match="calendar"):
        functions[builder](prices, memberships, **kwargs)


def test_audit_keeps_expected_decision_date_when_benchmark_bar_is_missing(inputs):
    prices, memberships, dates, config, calendar = inputs
    missing = prices.loc[
        ~((prices["symbol"] == "BENCH") & (prices["date"] == dates[6]))
    ]
    report = audit_inputs(
        missing,
        memberships,
        as_of_dates=[dates[6]],
        config=replace(config, min_observations=4),
        expected_calendar=calendar,
    )
    check = report["calendar_check"]
    assert check["missing_benchmark_sessions"] == [dates[6].date().isoformat()]
    assert check["missing_all_price_sessions"] == []
    assert check["status"] == "mismatch"
    row = report["coverage_by_date"][0]
    assert row["as_of_date"] == dates[6].date().isoformat()
    assert row["label_start_date"] == dates[7].date().isoformat()
    assert row["label_end_date"] == dates[9].date().isoformat()
    assert row["sufficient_return_history"] == 0
    assert report["calendar"]["source"] == "declared_expected_sessions"
    assert report["stage4_eligible"] is False
    assert report["calendar"]["independently_verified"] is False
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("position", [0, 3, -1])
def test_audit_detects_whole_market_gaps_without_narrowing_scope(inputs, position):
    prices, memberships, dates, config, calendar = inputs
    report = audit_inputs(
        prices.loc[prices["date"] != dates[position]],
        memberships,
        as_of_dates=[dates[6]],
        config=config,
        expected_calendar=calendar,
    )
    check = report["calendar_check"]
    expected = [dates[position].date().isoformat()]
    assert check["missing_benchmark_sessions"] == expected
    assert check["missing_all_price_sessions"] == expected
    assert check["declaration"] == calendar.to_mapping()
    assert report["input_fingerprints"]["expected_calendar"] == calendar.fingerprint
    assert report["calendar"]["session_count"] == len(dates)
    assert "session_calendar_mismatch" in {row["code"] for row in report["warnings"]}


@pytest.mark.parametrize("factors", [False, True])
def test_matching_calendar_preserves_calculations_and_records_identity(inputs, factors):
    prices, memberships, dates, config, calendar = inputs
    before_prices, before_memberships = (
        prices.copy(deep=True),
        memberships.copy(deep=True),
    )
    builder = build_factor_panel if factors else build_point_in_time_panel
    extra = {"factor_names": ("return_21s",)} if factors else {}
    plain = builder(
        prices, memberships, as_of_dates=list(dates), config=config, **extra
    )
    checked = builder(
        prices,
        memberships,
        as_of_dates=list(dates),
        config=config,
        expected_calendar=calendar,
        **extra,
    )
    pd.testing.assert_frame_equal(plain, checked.loc[:, plain.columns])
    assert checked["session_calendar_sha256"].eq(calendar.fingerprint).all()
    assert checked["session_calendar_source"].eq(calendar.source).all()
    assert checked["session_calendar_version"].eq(calendar.version).all()
    pd.testing.assert_frame_equal(prices, before_prices)
    pd.testing.assert_frame_equal(memberships, before_memberships)


def test_factor_calculator_receives_declared_sessions(inputs, monkeypatch):
    from quant_metric_research import factor_panel

    prices, memberships, dates, config, calendar = inputs
    original = factor_panel.compute_price_factors
    calendars = []

    def record(*args, **kwargs):
        calendars.append(tuple(kwargs["calendar"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(factor_panel, "compute_price_factors", record)
    build_factor_panel(
        prices,
        memberships,
        as_of_dates=[dates[6]],
        config=config,
        expected_calendar=calendar,
        factor_names=("return_21s",),
    )
    assert calendars == [calendar.sessions, calendar.sessions]


def test_audit_can_report_missing_entire_benchmark(inputs):
    prices, memberships, dates, config, calendar = inputs
    report = audit_inputs(
        prices.loc[prices["symbol"] != "BENCH"],
        memberships,
        as_of_dates=[dates[6]],
        config=config,
        expected_calendar=calendar,
    )
    assert len(report["calendar_check"]["missing_benchmark_sessions"]) == len(dates)
    assert report["coverage_by_date"][0]["sufficient_return_history"] == 0


def test_audit_reports_outside_decision_without_fabricated_coverage(inputs):
    prices, memberships, dates, config, calendar = inputs
    outside = dates[-1] + pd.Timedelta(days=20)
    report = audit_inputs(
        prices,
        memberships,
        as_of_dates=[outside],
        config=config,
        expected_calendar=calendar,
    )
    assert report["calendar_check"]["decision_dates_outside_calendar"] == [
        outside.date().isoformat()
    ]
    assert report["coverage_by_date"] == []
    assert report["coverage_status"] == "decision_dates_outside_calendar"


def _cli_files(tmp_path, inputs, *, gap=False):
    prices, memberships, dates, config, calendar = inputs
    if gap:
        prices = prices.loc[prices["date"] != dates[3]]
    prices.to_parquet(tmp_path / "prices.parquet", index=False)
    memberships.to_parquet(tmp_path / "memberships.parquet", index=False)
    pd.DataFrame({"as_of_date": dates}).to_csv(tmp_path / "dates.csv", index=False)
    (tmp_path / "config.json").write_text(json.dumps(asdict(config)), encoding="utf-8")
    (tmp_path / "calendar.json").write_text(
        json.dumps(calendar.to_mapping()), encoding="utf-8"
    )
    return [
        "--prices",
        str(tmp_path / "prices.parquet"),
        "--memberships",
        str(tmp_path / "memberships.parquet"),
        "--as-of-dates",
        str(tmp_path / "dates.csv"),
        "--config",
        str(tmp_path / "config.json"),
        "--calendar",
        str(tmp_path / "calendar.json"),
    ]


def test_calendar_audit_cli_runs_offline_and_reports_gap(inputs, tmp_path):
    args = _cli_files(tmp_path, inputs, gap=True)
    files_before = sorted(path.name for path in tmp_path.iterdir())
    result = subprocess.run(
        [sys.executable, "-m", "quant_metric_research.cli", "audit-inputs", *args],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["calendar_check"]["status"] == "mismatch"
    assert report["no_outcomes_computed"] is True
    assert len(report["coverage_by_date"]) == 12
    assert files_before == sorted(path.name for path in tmp_path.iterdir())


@pytest.mark.parametrize("gap", [False, True])
def test_research_cli_enforces_calendar_before_output(inputs, tmp_path, gap):
    args = _cli_files(tmp_path, inputs, gap=gap)
    destination = tmp_path / "research"
    args = [
        "run",
        *args,
        "--train-end",
        str(inputs[2][-1].date()),
        "--output-dir",
        str(destination),
        "--min-cross-section",
        "2",
        "--minimum-coverage",
        "0",
        "--redundancy-threshold",
        "1",
        "--quantiles",
        "2",
        "--hac-lags",
        "0",
    ]
    if gap:
        with pytest.raises(DataContractError, match="calendar"):
            main(args)
        assert not destination.exists()
    else:
        assert main(args) == 0
        panel = pd.read_parquet(destination / "metric_panel.parquet")
        assert panel["session_calendar_sha256"].eq(inputs[4].fingerprint).all()
        saved_calendar = json.loads(
            (destination / "session_calendar.json").read_text(encoding="utf-8")
        )
        assert saved_calendar == inputs[4].to_mapping()


def test_calendar_cli_rejects_duplicate_declaration_keys(inputs, tmp_path):
    args = _cli_files(tmp_path, inputs)
    path = tmp_path / "calendar.json"
    payload = path.read_text(encoding="utf-8")
    path.write_text(payload[:-1] + ', "source": "different"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        main(["audit-inputs", *args])


def test_research_writer_refuses_reuse_and_preserves_calendar(inputs, tmp_path):
    from quant_metric_research.io import write_research_run

    prices, memberships, dates, config, calendar = inputs
    run = run_research(
        prices,
        memberships,
        as_of_dates=list(dates),
        config=config,
        train_end_date=dates[-1],
        min_cross_section=2,
        minimum_coverage=0.0,
        redundancy_threshold=1.0,
        hac_lags=0,
        quantiles=2,
        expected_calendar=calendar,
    )
    destination = tmp_path / "saved"
    write_research_run(run, destination, panel_format="parquet")
    before = {path.name: path.read_bytes() for path in destination.iterdir()}
    legacy = replace(
        run,
        session_calendar=None,
        panel=run.panel.drop(
            columns=[name for name in run.panel if name.startswith("session_calendar_")]
        ),
    )
    with pytest.raises(FileExistsError):
        write_research_run(legacy, destination, panel_format="parquet")
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == before


def test_research_cli_refuses_existing_output_before_computation(
    inputs, tmp_path, monkeypatch
):
    args = _cli_files(tmp_path, inputs)
    destination = tmp_path / "existing"
    destination.mkdir()
    monkeypatch.setattr("quant_metric_research.cli.run_research", _forbidden)
    with pytest.raises(FileExistsError):
        main(
            [
                "run",
                *args,
                "--train-end",
                str(inputs[2][-1].date()),
                "--output-dir",
                str(destination),
                "--min-cross-section",
                "2",
                "--hac-lags",
                "0",
            ]
        )
    assert list(destination.iterdir()) == []
