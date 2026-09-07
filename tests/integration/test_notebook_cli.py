from __future__ import annotations

import json

import pytest

import quant_metric_research.cli as cli


def test_seed_notebook_dispatches_existing_registry_and_named_evidence(
    monkeypatch, capsys
):
    calls = []
    monkeypatch.setattr(
        cli,
        "seed_notebook_history",
        lambda **kw: calls.append(kw) or {"status": "seeded"},
    )
    assert (
        cli.main(
            [
                "seed-notebook",
                "--registry",
                "existing.sqlite3",
                "--history-dir",
                "history",
                "--evidence",
                "archive=data/source",
                "--evidence",
                "run=data/run=one",
            ]
        )
        == 0
    )
    assert calls == [
        {
            "registry_path": "existing.sqlite3",
            "history_dir": "history",
            "evidence_dirs": {"archive": "data/source", "run": "data/run=one"},
        }
    ]
    assert json.loads(capsys.readouterr().out) == {"status": "seeded"}


@pytest.mark.parametrize(
    "evidence",
    [["missing-separator"], ["=path"], ["archive="], ["archive=one", "archive=two"]],
)
def test_invalid_evidence_mapping_refused_before_checkpoint(evidence, monkeypatch):
    monkeypatch.setattr(
        cli, "seed_notebook_history", lambda **kw: pytest.fail("must not seed")
    )
    arguments = [
        "seed-notebook",
        "--registry",
        "existing.sqlite3",
        "--history-dir",
        "history",
    ]
    for item in evidence:
        arguments.extend(["--evidence", item])
    with pytest.raises(SystemExit) as caught:
        cli.main(arguments)
    assert caught.value.code == 2


def test_notebook_demo_dispatches_development_only(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        cli,
        "run_checkpointed_public_demo",
        lambda **kw: calls.append(kw) or {"status": "completed"},
    )
    assert (
        cli.main(
            [
                "notebook-demo",
                "--archive-dir",
                "archive",
                "--work-dir",
                "work",
                "--history-dir",
                "history",
            ]
        )
        == 0
    )
    assert calls == [
        {"archive_dir": "archive", "work_dir": "work", "history_dir": "history"}
    ]
    assert json.loads(capsys.readouterr().out) == {"status": "completed"}
    with pytest.raises(SystemExit):
        cli.main(
            [
                "notebook-demo",
                "--archive-dir",
                "archive",
                "--work-dir",
                "work",
                "--history-dir",
                "history",
                "--evaluate-lockbox",
            ]
        )
