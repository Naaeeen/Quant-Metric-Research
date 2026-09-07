from __future__ import annotations

import json
import subprocess
import sys

import pandas as pd
import pytest

from quant_metric_research.cli import main


@pytest.fixture()
def audit_args(tmp_path, market_fixture):
    prices, memberships, dates = market_fixture
    prices.to_csv(tmp_path / "prices.csv.gz", index=False)
    memberships.to_parquet(tmp_path / "memberships.parquet", index=False)
    pd.DataFrame({"as_of_date": [dates[8], dates[-1]]}).to_csv(
        tmp_path / "dates.csv", index=False
    )
    config = {
        "dataset_version": "invented-v1",
        "universe_id": "TEST",
        "benchmark_symbol": "BENCH",
        "lookback_sessions": 4,
        "min_observations": 2,
        "target_horizon_sessions": 2,
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return [
        "audit-inputs",
        "--prices",
        str(tmp_path / "prices.csv.gz"),
        "--memberships",
        str(tmp_path / "memberships.parquet"),
        "--as-of-dates",
        str(tmp_path / "dates.csv"),
        "--config",
        str(tmp_path / "config.json"),
    ]


def test_audit_cli_generates_json_without_training_or_writing(
    audit_args, capsys, monkeypatch, tmp_path
):
    def forbidden(*args, **kwargs):
        pytest.fail("Audit must not invoke a research run or artifact writer.")

    for name in (
        "run_research",
        "run_stage3_benchmark",
        "write_research_run",
        "write_benchmark_run",
    ):
        monkeypatch.setattr(f"quant_metric_research.cli.{name}", forbidden)
    files_before = sorted(path.name for path in tmp_path.iterdir())
    assert main(audit_args) == 0  # Report generated, even with coverage warnings.
    report = json.loads(capsys.readouterr().out)
    assert report["claim_scope"] == "raw_input_diagnostics_only"
    assert report["stage4_eligible"] is False
    assert report["coverage_by_date"][-1]["label_calendar_unavailable"] == 2
    assert report["warnings"]
    assert files_before == sorted(path.name for path in tmp_path.iterdir())


def test_audit_command_runs_end_to_end_offline(audit_args):
    process = subprocess.run(
        [sys.executable, "-m", "quant_metric_research.cli", *audit_args],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    assert report["summary"]["active_member_dates"] == 4
    assert report["no_outcomes_computed"] is True


def test_public_audit_api():
    from quant_metric_research import audit_inputs
    from quant_metric_research.input_audit import audit_inputs as implementation

    assert audit_inputs is implementation
