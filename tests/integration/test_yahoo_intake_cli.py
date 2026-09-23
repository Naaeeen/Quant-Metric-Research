"""Offline CLI journeys using invented prices, never downloaded Yahoo data."""

from __future__ import annotations

import json
import socket
import subprocess
import sys

import pandas as pd
import pytest

from quant_metric_research.cli import main


def _strict_json(payload: str) -> dict:
    def reject_constant(value: str):
        raise AssertionError(f"Non-finite JSON constant: {value}")

    return json.loads(payload, parse_constant=reject_constant)


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "quant_metric_research.cli", *args],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.fixture()
def yahoo_cli_inputs(tmp_path, market_fixture):
    prices, memberships, dates = market_fixture
    source = tmp_path / "invented-inputs"
    source.mkdir()
    exports = {}
    for symbol, group in prices.groupby("symbol"):
        frame = group.rename(columns={"date": "Date", "adjusted_close": "Adj Close"})
        if symbol == "BBB":
            frame = frame.assign(
                **{"Adj Close": frame["Adj Close"].mask(frame["Date"] == dates[9])}
            )
        frame[["Date", "Adj Close"]].to_csv(source / f"{symbol}.csv", index=False)
        exports[symbol] = f"{symbol}.csv"
    (source / "exports.json").write_text(json.dumps(exports), encoding="utf-8")
    memberships.to_csv(source / "memberships.csv", index=False)
    pd.DataFrame({"as_of_date": dates[8:11]}).to_csv(source / "dates.csv", index=False)
    config = {
        "dataset_version": "invented-yahoo-cli-v1",
        "universe_id": "TEST",
        "benchmark_symbol": "BENCH",
        "lookback_sessions": 4,
        "min_observations": 2,
        "target_horizon_sessions": 2,
    }
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "new-intake"
    args = [
        "import-yahoo",
        "--exports",
        str(source / "exports.json"),
        "--memberships",
        str(source / "memberships.csv"),
        "--as-of-dates",
        str(source / "dates.csv"),
        "--config",
        str(source / "config.json"),
        "--output-dir",
        str(output),
    ]
    return args, source, output


def test_yahoo_cli_end_to_end_offline_and_audit_replay(yahoo_cli_inputs):
    args, _, output = yahoo_cli_inputs

    imported = _run_cli(args)

    assert imported.returncode == 0, imported.stderr
    manifest = _strict_json(imported.stdout)
    assert manifest == _strict_json((output / "intake_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["network_accessed"] is False
    assert manifest["claim_scope"] == "unverified_local_price_import"
    assert manifest["empirical_data_provenance_verified"] is False
    assert manifest["stage4_eligible"] is False
    missing = pd.read_csv(output / "missing_prices.csv")
    assert missing[["symbol", "reason"]].to_dict("records") == [
        {"symbol": "BBB", "reason": "missing_adjusted_close"}
    ]

    replay = _run_cli(
        [
            "audit-inputs",
            "--prices",
            str(output / "prices.parquet"),
            "--memberships",
            str(output / "memberships.parquet"),
            "--as-of-dates",
            str(output / "as_of_dates.csv"),
            "--config",
            str(output / "config.json"),
        ]
    )

    assert replay.returncode == 0, replay.stderr
    audit = _strict_json(replay.stdout)
    assert audit == _strict_json((output / "input_audit.json").read_text())
    assert audit["no_outcomes_computed"] is True
    assert audit["summary"]["active_member_dates"] == 6
    assert audit["stage4_eligible"] is False


def test_yahoo_cli_does_not_train_open_lockbox_or_access_network(
    yahoo_cli_inputs, capsys, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail(
            "Offline intake must not train, open a lockbox, or access a network."
        )

    for name in (
        "run_research",
        "run_stage3_benchmark",
        "write_research_run",
        "write_benchmark_run",
        "ExperimentRegistry",
    ):
        monkeypatch.setattr(f"quant_metric_research.cli.{name}", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)

    assert main(yahoo_cli_inputs[0]) == 0

    manifest = _strict_json(capsys.readouterr().out)
    assert manifest["status"] == "complete"
    assert manifest["network_accessed"] is False


def test_yahoo_cli_existing_output_directory_is_refused_without_changes(
    yahoo_cli_inputs,
):
    args, _, output = yahoo_cli_inputs
    output.mkdir()
    (output / "keep.txt").write_text("original user data", encoding="utf-8")
    before = {path.relative_to(output): path.read_bytes() for path in output.rglob("*")}

    refused = _run_cli(args)

    assert refused.returncode != 0
    assert "already exists" in refused.stderr
    assert refused.stdout == ""
    after = {path.relative_to(output): path.read_bytes() for path in output.rglob("*")}
    assert after == before


def test_yahoo_cli_malformed_export_has_no_success_manifest(yahoo_cli_inputs):
    args, source, output = yahoo_cli_inputs
    (source / "AAA.csv").write_text("Date,Close\n2025-01-02,10\n", encoding="utf-8")

    failed = _run_cli(args)

    assert failed.returncode != 0
    assert "Adj Close" in failed.stderr
    assert failed.stdout == ""
    assert not (output / "intake_manifest.json").exists()
    failure = _strict_json((output / "intake_failure.json").read_text())
    assert failure["status"] == "failed"
    assert list((output / "raw").iterdir())
