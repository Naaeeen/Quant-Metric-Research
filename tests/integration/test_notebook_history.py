from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from quant_metric_research import ExperimentRegistry
from quant_metric_research import _notebook_history_io as history_io
from quant_metric_research import notebook_history as history


def manifest():
    return {
        "panel_fingerprint": "fixture",
        "source_fingerprint": "fixture",
        "configuration": {},
        "package_version": "fixture",
        "python_version": "fixture",
        "numpy_version": "fixture",
        "pandas_version": "fixture",
        "scikit_learn_version": "fixture",
        "scipy_version": "fixture",
        "development_start": "2020-01-01",
        "locked_test_start": "2020-06-01",
        "locked_test_end": "2020-06-20",
        "locked_label_end_max": "2020-07-10",
    }


def add_run(path, status="running"):
    registry = ExperimentRegistry(path)
    run_id = registry.start_development(
        study_id="fixture", hypothesis="Synthetic software test", configuration={}
    )
    registry.record_development_plan(run_id, manifest=manifest())
    if status == "completed":
        registry.complete_development(
            run_id, manifest=manifest(), frozen_family="ridge"
        )
    elif status == "failed":
        registry.fail_run(run_id, error_type="ValueError")
    return run_id


def raw_rows(path):
    with closing(sqlite3.connect(path)) as connection:
        return {
            table: connection.execute(
                f"SELECT * FROM {table} ORDER BY run_id"
            ).fetchall()
            for table in ("runs", "exposures")
        }


@pytest.fixture()
def seeded(tmp_path):
    registry = tmp_path / "canonical.sqlite3"
    add_run(registry)
    add_run(registry, "failed")
    evidence = tmp_path / "old-evidence"
    evidence.mkdir()
    (evidence / "partial.txt").write_text("preserve partial output")
    (evidence / "empty").mkdir()
    archive = tmp_path / "archive"
    archive.mkdir()
    destination = tmp_path / "history"
    history.seed_notebook_history(
        registry_path=registry,
        history_dir=destination,
        evidence_dirs={"old": evidence, "archive": archive},
    )
    return registry, destination, archive


def mock_demo(monkeypatch, *, error=None):
    def run(*, archive_dir, output_dir, registry_path):
        output_dir.mkdir()
        (output_dir / "partial.txt").write_text("current output")
        run_id = add_run(registry_path, "completed" if error is None else "running")
        if error is not None:
            raise error
        return {"status": "complete", "development_run_id": run_id}

    monkeypatch.setattr(history, "run_public_demo", run)


def execute(seeded, tmp_path, name="work"):
    _, directory, archive = seeded
    return history.run_checkpointed_public_demo(
        history_dir=directory,
        archive_dir=archive,
        work_dir=tmp_path / name,
    )


def test_seed_preserves_every_raw_row_exposure_and_partial_evidence(seeded):
    registry, directory, _ = seeded
    assert raw_rows(registry) == raw_rows(directory / "000000/registry.sqlite3")
    assert (directory / "000000/evidence/old/partial.txt").read_text() == (
        "preserve partial output"
    )
    assert (directory / "000000/evidence/old/empty").is_dir()
    checkpoint = json.loads((directory / "000000/checkpoint.json").read_text())
    assert checkpoint["status"] == "seeded"
    assert checkpoint["parent_checkpoint_sha256"] is None


def test_two_runs_restore_and_extend_all_history(seeded, tmp_path, monkeypatch):
    mock_demo(monkeypatch)
    canonical, directory, _ = seeded
    original = canonical.read_bytes()
    first = execute(seeded, tmp_path)
    second = execute(seeded, tmp_path, "work-two")
    assert first["generation"] == "000001"
    assert second["generation"] == "000002"
    assert second["status"] == "completed"
    old = raw_rows(canonical)
    restored = raw_rows(tmp_path / "work-two/registry.sqlite3")
    assert set(old["runs"]) <= set(restored["runs"])
    assert set(old["exposures"]) <= set(restored["exposures"])
    assert len(restored["runs"]) == 4
    assert canonical.read_bytes() == original
    assert (directory / "000001/evidence/demo/partial.txt").is_file()
    assert (directory / "000002/evidence/demo/partial.txt").is_file()


@pytest.mark.parametrize("error", [ValueError("training failed"), KeyboardInterrupt()])
def test_catchable_failure_checkpoints_and_preserves_original_exception(
    seeded,
    tmp_path,
    monkeypatch,
    error,
):
    mock_demo(monkeypatch, error=error)
    with pytest.raises(type(error)) as captured:
        execute(seeded, tmp_path)
    assert captured.value is error
    directory = seeded[1]
    checkpoint = json.loads((directory / "000001/checkpoint.json").read_text())
    assert checkpoint["status"] == "failed"
    assert checkpoint["error_type"] == type(error).__name__
    assert len(raw_rows(directory / "000001/registry.sqlite3")["runs"]) == 3
    mock_demo(monkeypatch)
    assert execute(seeded, tmp_path, "retry")["generation"] == "000002"


def test_start_only_generation_refuses_any_older_fallback(
    seeded, tmp_path, monkeypatch
):
    unresolved = seeded[1] / "000001"
    unresolved.mkdir()
    (unresolved / "start.json").write_text("{}")
    monkeypatch.setattr(history, "run_public_demo", lambda **kw: pytest.fail("no run"))
    with pytest.raises(ValueError):
        execute(seeded, tmp_path)
    assert not (tmp_path / "work").exists()


@pytest.mark.parametrize("change", ["modify", "missing", "extra", "duplicate-json"])
def test_invalid_history_refuses_before_restore(seeded, tmp_path, monkeypatch, change):
    generation = seeded[1] / "000000"
    target = generation / "evidence/old/partial.txt"
    if change == "modify":
        target.write_text("changed")
    elif change == "missing":
        target.unlink()
    elif change == "extra":
        (generation / "extra").write_text("unexpected")
    else:
        checkpoint = generation / "checkpoint.json"
        checkpoint.write_text('{"schema_version":1,"schema_version":1}')
    monkeypatch.setattr(history, "run_public_demo", lambda **kw: pytest.fail("no run"))
    with pytest.raises(ValueError):
        execute(seeded, tmp_path)
    assert not (tmp_path / "work").exists()


@pytest.mark.parametrize("mutation", ["remove", "change", "final"])
def test_registry_regression_or_new_final_record_leaves_unresolved_generation(
    seeded,
    tmp_path,
    monkeypatch,
    mutation,
):
    def invalid_demo(*, registry_path, **kwargs):
        with closing(sqlite3.connect(registry_path)) as connection, connection:
            if mutation == "remove":
                connection.execute("DELETE FROM exposures")
            elif mutation == "change":
                connection.execute("UPDATE runs SET hypothesis = 'changed'")
            else:
                completed = add_run(registry_path, "completed")
                ExperimentRegistry(registry_path).reserve_final(
                    development_run_id=completed,
                    manifest=manifest(),
                    frozen_family="ridge",
                )
        return {"status": "complete"}

    monkeypatch.setattr(history, "run_public_demo", invalid_demo)
    with pytest.raises(ValueError):
        execute(seeded, tmp_path)
    assert (seeded[1] / "000001/start.json").is_file()
    assert not (seeded[1] / "000001/checkpoint.json").exists()


def test_seed_requires_existing_registry_and_new_nonoverlapping_destination(tmp_path):
    missing = tmp_path / "missing.sqlite3"
    with pytest.raises(ValueError):
        history.seed_notebook_history(
            registry_path=missing,
            history_dir=tmp_path / "new",
            evidence_dirs={},
        )
    assert not missing.exists()
    registry = tmp_path / "source/registry.sqlite3"
    add_run(registry)
    with pytest.raises(ValueError):
        history.seed_notebook_history(
            registry_path=registry,
            history_dir=registry.parent / "history",
            evidence_dirs={"parent": registry.parent},
        )
    with pytest.raises(FileExistsError):
        history.seed_notebook_history(
            registry_path=registry,
            history_dir=registry.parent,
            evidence_dirs={},
        )


@pytest.mark.parametrize(
    "location", ["existing", "history-child", "history-parent", "archive"]
)
def test_new_work_directory_must_be_disjoint(seeded, tmp_path, location):
    paths = {
        "existing": seeded[0].parent,
        "history-child": seeded[1] / "new",
        "history-parent": tmp_path,
        "archive": seeded[2] / "new",
    }
    with pytest.raises((ValueError, FileExistsError)):
        history.run_checkpointed_public_demo(
            history_dir=seeded[1],
            archive_dir=seeded[2],
            work_dir=paths[location],
        )


def test_existing_seed_directory_never_overwritten(seeded):
    registry, destination, _ = seeded
    before = (destination / "000000/checkpoint.json").read_bytes()
    with pytest.raises(FileExistsError):
        history.seed_notebook_history(
            registry_path=registry,
            history_dir=destination,
            evidence_dirs={},
        )
    assert (destination / "000000/checkpoint.json").read_bytes() == before


@pytest.mark.parametrize("failure", ["start-write", "start-read", "copy", "checkpoint"])
def test_write_failures_never_report_success_and_leave_blocked_history(
    seeded,
    tmp_path,
    monkeypatch,
    failure,
):
    called = []
    mock_demo(monkeypatch)
    original_demo = history.run_public_demo

    def counted(**kwargs):
        called.append(True)
        return original_demo(**kwargs)

    monkeypatch.setattr(history, "run_public_demo", counted)
    original_write, original_read, original_copy = (
        history.write_json,
        history.read_json,
        history.copy_file,
    )

    def write(value, path):
        if (
            path.parent.name == "000001"
            and path.name
            == ("start.json" if failure == "start-write" else "checkpoint.json")
            and failure in {"start-write", "checkpoint"}
        ):
            path.write_text("{")
            raise OSError("simulated partial write")
        return original_write(value, path)

    def read(path):
        if failure == "start-read" and path.parent.name == "000001":
            raise OSError("simulated read failure")
        return original_read(path)

    def copy(source, destination):
        if failure == "copy" and destination.parent.name == "000001":
            raise OSError("simulated snapshot copy failure")
        return original_copy(source, destination)

    monkeypatch.setattr(history, "write_json", write)
    monkeypatch.setattr(history, "read_json", read)
    monkeypatch.setattr(history, "copy_file", copy)
    with pytest.raises(OSError):
        execute(seeded, tmp_path)
    assert len(called) == (0 if failure.startswith("start") else 1)
    monkeypatch.setattr(history, "read_json", original_read)
    with pytest.raises(ValueError):
        execute(seeded, tmp_path, "retry")


def test_checkpoint_failure_preserves_training_exception_and_adds_note(
    seeded,
    tmp_path,
    monkeypatch,
):
    interruption = KeyboardInterrupt()
    mock_demo(monkeypatch, error=interruption)
    original_copy = history.copy_file

    def fail_copy(source, destination):
        if destination.parent.name == "000001":
            raise OSError("disk unavailable")
        return original_copy(source, destination)

    monkeypatch.setattr(history, "copy_file", fail_copy)
    with pytest.raises(KeyboardInterrupt) as captured:
        execute(seeded, tmp_path)
    assert captured.value is interruption
    assert "Checkpoint unavailable" in interruption.__notes__[0]
    with pytest.raises(ValueError):
        execute(seeded, tmp_path, "retry")


def test_sqlite_never_connects_to_history_storage(seeded, tmp_path, monkeypatch):
    mock_demo(monkeypatch)
    original_connect = sqlite3.connect
    history_uri = seeded[1].as_uri()

    def local_only(database, *args, **kwargs):
        assert not str(database).startswith(history_uri)
        assert seeded[1] not in Path(str(database)).parents
        return original_connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", local_only)
    assert execute(seeded, tmp_path)["status"] == "completed"


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/drive", "a\\b", "a//b"])
def test_unsafe_inventory_names_are_rejected(seeded, tmp_path, name):
    checkpoint = seeded[1] / "000000/checkpoint.json"
    value = json.loads(checkpoint.read_text())
    value["inventory"]["files"][name] = {"size": 0, "sha256": "a" * 64}
    checkpoint.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="unsafe"):
        execute(seeded, tmp_path)


def test_missing_earlier_generation_never_uses_later_snapshot(
    seeded, tmp_path, monkeypatch
):
    mock_demo(monkeypatch)
    execute(seeded, tmp_path)
    (seeded[1] / "000000/start.json").unlink()
    with pytest.raises(ValueError):
        execute(seeded, tmp_path, "retry")


def test_symlinked_evidence_and_work_parent_are_rejected(seeded, tmp_path, monkeypatch):
    link = tmp_path / "linked"
    try:
        link.symlink_to(seeded[2], target_is_directory=True)
    except OSError:
        # Windows may disallow creating symlinks without developer mode. Exercise
        # the same lstat boundary without requiring additional OS permissions.
        import stat
        from types import SimpleNamespace

        original = Path.lstat

        def symbolic_stat(path):
            if path == link:
                return SimpleNamespace(st_mode=stat.S_IFLNK, st_file_attributes=0)
            return original(path)

        monkeypatch.setattr(Path, "lstat", symbolic_stat)
    with pytest.raises(ValueError, match="Symlinks"):
        history.run_checkpointed_public_demo(
            archive_dir=seeded[2],
            history_dir=seeded[1],
            work_dir=link / "work",
        )
    with pytest.raises(ValueError, match="Symlinks"):
        history.seed_notebook_history(
            registry_path=seeded[0],
            history_dir=tmp_path / "newhistory",
            evidence_dirs={"linked": link},
        )


@pytest.mark.parametrize("mutation", ["schema", "metadata", "exposure"])
def test_invalid_existing_registry_is_never_repaired(tmp_path, mutation):
    registry = tmp_path / "invalid.sqlite3"
    add_run(registry)
    with closing(sqlite3.connect(registry)) as connection, connection:
        if mutation == "schema":
            connection.execute("CREATE TABLE surprise(value TEXT)")
        elif mutation == "metadata":
            connection.execute("UPDATE runs SET configuration = ?", ('{"a":1,"a":2}',))
        else:
            connection.execute("DELETE FROM exposures")
    original = registry.read_bytes()
    with pytest.raises(ValueError):
        history.seed_notebook_history(
            registry_path=registry,
            history_dir=tmp_path / "history",
            evidence_dirs={},
        )
    assert registry.read_bytes() == original
    assert not (tmp_path / "history").exists()


def test_checkpoint_write_that_raises_after_writing_still_blocks_retry(
    seeded,
    tmp_path,
    monkeypatch,
):
    mock_demo(monkeypatch)
    original = history.write_json

    def uncertain_write(value, path):
        original(value, path)
        if path.name == "checkpoint.json" and path.parent.name == "000001":
            raise OSError("simulated failed flush after write")

    monkeypatch.setattr(history, "write_json", uncertain_write)
    with pytest.raises(OSError):
        execute(seeded, tmp_path)
    with pytest.raises(ValueError):
        execute(seeded, tmp_path, "retry")


def test_file_copy_is_exclusive_and_detects_corruption(tmp_path, monkeypatch):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_bytes(b"original")
    destination.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        history_io.copy_file(source, destination)
    assert destination.read_bytes() == b"preserve"


def test_real_synthetic_public_demo_workflow_uses_closed_checkpoint(
    seeded,
    tmp_path,
    monkeypatch,
):
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd

    from quant_metric_research import BenchmarkConfig, PanelConfig, public_demo

    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2013-01-02", periods=75)
    symbols = tuple(f"SAMPLE_{index}" for index in range(8))
    prices = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": symbol,
                    "adjusted_close": 100
                    * np.exp(np.cumsum(rng.normal(0.001, 0.012, len(dates)))),
                }
            )
            for symbol in (*symbols, "BENCH")
        ],
        ignore_index=True,
    )
    features = ("trailing_return", "annualized_volatility", "max_drawdown")
    panel = PanelConfig(
        dataset_version="synthetic-history-test-v1",
        universe_id="SYNTHETIC_COHORT",
        benchmark_symbol="BENCH",
        lookback_sessions=10,
        min_observations=8,
        target_horizon_sessions=2,
        entry_lag_sessions=1,
        feature_columns=features,
    )
    benchmark = BenchmarkConfig.from_mapping(
        {
            "feature_columns": features,
            "model_families": ["ridge"],
            "ridge_alphas": [1.0],
            "min_cross_section": 8,
            "quantiles": 2,
            "hac_lags": 1,
            "split": {
                "final_test_date_count": 6,
                "outer_n_splits": 2,
                "outer_test_date_count": 6,
                "outer_min_train_date_count": 18,
                "inner_n_splits": 1,
                "inner_validation_date_count": 4,
                "inner_min_train_date_count": 8,
            },
        }
    )
    sample = SimpleNamespace(
        prices=prices,
        symbols=symbols,
        profile={"synthetic_software_fixture": True},
        missing_prices=pd.DataFrame(columns=["date", "symbol", "reason"]),
    )
    monkeypatch.setattr(
        public_demo, "_prepare_sample", lambda path: (sample, {"synthetic": True})
    )
    monkeypatch.setattr(public_demo, "public_demo_configs", lambda: (panel, benchmark))
    checkpoint = execute(seeded, tmp_path)
    assert checkpoint["status"] == "completed"
    report = json.loads((tmp_path / "work/demo/public_demo_report.json").read_text())
    assert report["final_outcomes_evaluated"] is False
    copied = seeded[1] / "000001/evidence/demo/public_demo_report.json"
    assert json.loads(copied.read_text()) == report
    rows = ExperimentRegistry(tmp_path / "work/registry.sqlite3").list_runs()
    assert len(rows) == 3
    assert all(row["kind"] == "development" for row in rows)
    assert rows[-1]["status"] == "completed"


@pytest.mark.parametrize(
    "value",
    [
        "https://example.test/path",
        "s3://bucket/key",
        "file:a",
        "relative:stream",
        "a.txt:stream",
    ],
)
def test_paths_reject_url_schemes_and_windows_streams(value):
    with pytest.raises(ValueError):
        history_io.checked_path(value)


@pytest.mark.parametrize("value", ["1e999", "-1e999", "NaN", "Infinity"])
def test_json_rejects_nested_nonfinite_numbers(value):
    with pytest.raises(ValueError, match="Nonfinite"):
        history_io.decode_json('{"nested":{"value":' + value + "}}")


def test_mixed_case_evidence_directory_order_is_portable(tmp_path):
    registry = tmp_path / "canonical.sqlite3"
    add_run(registry)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "a").mkdir()
    (evidence / "B").mkdir()
    result = history.seed_notebook_history(
        registry_path=registry,
        history_dir=tmp_path / "history",
        evidence_dirs={"source": evidence},
    )
    assert result["status"] == "seeded"


def test_unavailable_failure_marker_reports_manual_review_without_durability_claim(
    seeded,
    tmp_path,
    monkeypatch,
):
    mock_demo(monkeypatch)
    original = history.write_json

    def unavailable(value, path):
        if path.name == "checkpoint_failed.json":
            raise OSError("storage unavailable")
        original(value, path)
        if path.name == "checkpoint.json" and path.parent.name == "000001":
            raise OSError("uncertain checkpoint flush")

    monkeypatch.setattr(history, "write_json", unavailable)
    with pytest.raises(OSError) as captured:
        execute(seeded, tmp_path)
    assert "stop automatic retries" in captured.value.__notes__[0]
    assert "000001" in captured.value.__notes__[0]
    assert (tmp_path / "work/registry.sqlite3").is_file()
    assert (tmp_path / "work/demo/partial.txt").is_file()


def test_start_records_declared_identity_before_demo(seeded, tmp_path, monkeypatch):
    mock_demo(monkeypatch)
    original = history.run_public_demo

    def inspect_start(**kwargs):
        start = json.loads((seeded[1] / "000001/start.json").read_text())
        identity = start["declared_identity"]
        assert len(identity["source_fingerprint"]) == 64
        assert set(identity["runtime_versions"]) == {
            "python",
            "numpy",
            "pandas",
            "scipy",
            "scikit_learn",
        }
        assert identity["benchmark_configuration"]["model_families"] == ["ridge"]
        assert identity["package_version"]
        return original(**kwargs)

    monkeypatch.setattr(history, "run_public_demo", inspect_start)
    assert execute(seeded, tmp_path)["status"] == "completed"


def test_preexisting_final_reservations_are_preserved_without_new_final_exposure(
    tmp_path,
    monkeypatch,
):
    registry = tmp_path / "canonical.sqlite3"
    completed = add_run(registry, "completed")
    ExperimentRegistry(registry).reserve_final(
        development_run_id=completed,
        manifest=manifest(),
        frozen_family="ridge",
    )
    original = raw_rows(registry)
    directory, archive = tmp_path / "history", tmp_path / "archive"
    archive.mkdir()
    history.seed_notebook_history(
        registry_path=registry,
        history_dir=directory,
        evidence_dirs={},
    )
    assert raw_rows(directory / "000000/registry.sqlite3") == original
    mock_demo(monkeypatch)
    history.run_checkpointed_public_demo(
        history_dir=directory,
        archive_dir=archive,
        work_dir=tmp_path / "work",
    )
    current = raw_rows(tmp_path / "work/registry.sqlite3")
    assert set(original["runs"]) <= set(current["runs"])
    assert set(original["exposures"]) <= set(current["exposures"])


@pytest.mark.parametrize(
    "change", ["parent", "status", "count", "schema", "identity", "inventory"]
)
def test_invalid_checkpoint_metadata_refuses_before_training(
    seeded,
    tmp_path,
    monkeypatch,
    change,
):
    checkpoint = seeded[1] / "000000/checkpoint.json"
    value = json.loads(checkpoint.read_text())
    field, replacement = {
        "parent": ("parent_checkpoint_sha256", "f" * 64),
        "status": ("status", []),
        "count": ("run_count", True),
        "schema": ("schema_version", True),
        "identity": ("generation", "000123"),
        "inventory": ("inventory", {"directories": [], "files": []}),
    }[change]
    checkpoint.write_text(json.dumps({**value, field: replacement}))
    monkeypatch.setattr(history, "run_public_demo", lambda **kw: pytest.fail("no run"))
    with pytest.raises(ValueError):
        execute(seeded, tmp_path)
    assert not (tmp_path / "work").exists()


@pytest.mark.parametrize("mount", ["/content/drive", "/content/gdrive"])
@pytest.mark.parametrize("suffix", ["", "/MyDrive/research/registry.sqlite3"])
def test_standard_drive_mounts_cannot_hold_compute_or_sqlite_paths(mount, suffix):
    with pytest.raises(ValueError, match="Drive"):
        history_io.require_local_compute_path(mount + suffix)


def test_drive_registry_and_work_are_refused_before_history_changes(
    seeded,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        history, "registry_rows", lambda path: pytest.fail("no database")
    )
    monkeypatch.setattr(
        history, "_validate_history", lambda path: pytest.fail("no restore")
    )
    with pytest.raises(ValueError, match="Drive"):
        history.seed_notebook_history(
            registry_path="/content/drive/MyDrive/registry.sqlite3",
            history_dir=tmp_path / "unused",
            evidence_dirs={},
        )
    with pytest.raises(ValueError, match="Drive"):
        history.run_checkpointed_public_demo(
            archive_dir=seeded[2],
            history_dir=seeded[1],
            work_dir="/content/drive/MyDrive/new-work",
        )
    assert not (tmp_path / "unused").exists()
    assert [path.name for path in seeded[1].iterdir()] == ["000000"]


def test_drive_temp_directory_is_refused_before_creation_or_sqlite(
    seeded,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(history, "gettempdir", lambda: "/content/drive/MyDrive/tmp")
    monkeypatch.setattr(
        history, "TemporaryDirectory", lambda **kw: pytest.fail("no temp")
    )
    with pytest.raises(ValueError, match="Drive"):
        execute(seeded, tmp_path)
    assert not (tmp_path / "work").exists()


def test_drive_snapshot_destination_is_refused_before_database_connection(
    seeded,
    monkeypatch,
):
    monkeypatch.setattr(sqlite3, "connect", lambda *a, **kw: pytest.fail("no database"))
    with pytest.raises(ValueError, match="Drive"):
        history_io.snapshot_registry(
            seeded[0],
            Path("/content/drive/MyDrive/closed-snapshot.sqlite3"),
        )
    with pytest.raises(ValueError, match="Drive"):
        history_io.registry_rows(Path("/content/drive/MyDrive/registry.sqlite3"))


@pytest.mark.parametrize(
    "path", ["/content/local-run", "/content/drive-copy", "/tmp/qmr"]
)
def test_local_compute_guard_allows_local_siblings(path):
    assert history_io.require_local_compute_path(path) == history_io.checked_path(path)
