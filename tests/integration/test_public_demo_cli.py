from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

import quant_metric_research.public_demo as demo
from quant_metric_research.cli import main


@pytest.fixture()
def demo_dependencies(monkeypatch):
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2013-01-02", "2013-01-03"]),
            "symbol": ["FF_MARKET_PROXY", "FF_MARKET_PROXY"],
            "adjusted_close": [100, 101],
        }
    )
    sample = SimpleNamespace(
        prices=prices,
        symbols=("A", "B"),
        profile={"fixture": True},
        missing_prices=pd.DataFrame(columns=["date", "symbol", "reason"]),
    )
    monkeypatch.setattr(demo, "verify_public_archive", lambda path: {"fixture": True})
    monkeypatch.setattr(demo, "read_table", lambda path: pd.DataFrame())
    monkeypatch.setattr(demo, "normalize_mendeley_prices", lambda *a, **kw: sample)
    calls = []

    def workflow(prices, memberships, **kwargs):
        calls.append(kwargs)
        kwargs["output_dir"].mkdir()
        report = {
            "status": "complete",
            "stage4_eligible": False,
            "development_run_id": "fixture-run",
            "development_summary": [],
        }
        (kwargs["output_dir"] / "development_report.json").write_text(
            json.dumps(report)
        )
        return report

    monkeypatch.setattr(demo, "run_development_workflow", workflow)
    return calls


def test_public_demo_persists_plan_profile_and_development_reference(
    tmp_path, demo_dependencies
):
    output = tmp_path / "run"
    report = demo.run_public_demo(
        archive_dir=tmp_path / "archive",
        output_dir=output,
        registry_path=tmp_path / "registry.sqlite3",
    )
    assert report["claim_scope"] == "public_archive_engineering_demo"
    assert report["stage4_eligible"] is False
    assert report["development_run_id"] == "fixture-run"
    assert (output / "missing_prices.csv").is_file()
    assert json.loads((output / "quality_profile.json").read_text()) == {
        "fixture": True
    }
    assert json.loads((output / "public_demo_report.json").read_text()) == report
    call = demo_dependencies[0]
    assert call["benchmark_config"].model_families == ("ridge",)
    assert call["benchmark_config"].ridge_alphas == (1.0,)
    assert call["benchmark_config"].min_cross_section == 20
    assert call["panel_config"].target_horizon_sessions == 20
    assert call["panel_config"].entry_lag_sessions == 1


def test_public_demo_existing_output_does_not_start_work(tmp_path, monkeypatch):
    output = tmp_path / "run"
    output.mkdir()
    monkeypatch.setattr(
        demo, "verify_public_archive", lambda p: pytest.fail("must refuse")
    )
    with pytest.raises(FileExistsError):
        demo.run_public_demo(
            archive_dir=tmp_path / "archive",
            output_dir=output,
            registry_path=tmp_path / "registry.sqlite3",
        )


def test_public_demo_registry_cannot_be_inside_any_run_directory(
    tmp_path, demo_dependencies
):
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="outside"):
        demo.run_public_demo(
            archive_dir=tmp_path / "archive",
            output_dir=output,
            registry_path=output / "registry.sqlite3",
        )
    assert not output.exists()
    assert demo_dependencies == []


def test_public_demo_cli_dispatches_offline_workflow(tmp_path, monkeypatch, capsys):
    called = []
    monkeypatch.setattr(
        "quant_metric_research.cli.run_public_demo",
        lambda **kw: called.append(kw) or {"status": "complete"},
    )
    assert (
        main(
            [
                "public-demo",
                "--archive-dir",
                "archive",
                "--output-dir",
                "new-run",
                "--registry",
                "registry.sqlite3",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
    assert called[0]["archive_dir"] == "archive"


def test_fetch_public_cli_is_an_explicit_separate_network_command(monkeypatch, capsys):
    called = []
    monkeypatch.setattr(
        "quant_metric_research.cli.fetch_public_archive",
        lambda path: called.append(path) or {"status": "complete"},
    )
    assert main(["fetch-public-sample", "--output-dir", "new-archive"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
    assert called == ["new-archive"]
