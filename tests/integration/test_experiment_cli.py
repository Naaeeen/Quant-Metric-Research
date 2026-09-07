from __future__ import annotations

import json

import pytest

from quant_metric_research import cli


def test_full_cli_requires_registry_and_development_reference(capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "benchmark",
                "--panel",
                "unused.csv",
                "--config",
                "unused.json",
                "--output-dir",
                "unused",
                "--evaluate-lockbox",
            ]
        )
    assert error.value.code == 2
    assert "registry" in capsys.readouterr().err


@pytest.mark.parametrize("feasible,expected_exit", [(True, 0), (False, 2)])
def test_preflight_cli_prints_structural_report(
    monkeypatch,
    capsys,
    feasible,
    expected_exit,
):
    monkeypatch.setattr(cli, "read_table", lambda path: object())
    monkeypatch.setattr(cli, "read_json_object", lambda path: {})
    monkeypatch.setattr(cli.BenchmarkConfig, "from_mapping", lambda data: object())
    monkeypatch.setattr(
        cli,
        "preflight_benchmark",
        lambda panel, config: {"schema_version": "1", "feasible": feasible},
    )
    assert cli.main(
        ["preflight", "--panel", "panel.csv", "--config", "config.json"]
    ) == (expected_exit)
    assert json.loads(capsys.readouterr().out)["feasible"] is feasible


def test_history_missing_registry_fails_without_creating_it(tmp_path):
    path = tmp_path / "missing.sqlite3"
    with pytest.raises(FileNotFoundError):
        cli.main(["experiments", "--registry", str(path)])
    assert not path.exists()
