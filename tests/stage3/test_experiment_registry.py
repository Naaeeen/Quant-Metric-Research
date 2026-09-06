from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from copy import deepcopy
from datetime import datetime
from threading import Barrier

import pytest

from quant_metric_research.experiment_registry import ExperimentRegistry


def _manifest(**overrides):
    return {
        "panel_fingerprint": "panel-a",
        "source_fingerprint": "source-a",
        "configuration": {"feature_columns": ["signal"], "hac_lags": 2},
        "package_version": "0.4.0",
        "python_version": "3.11.0",
        "numpy_version": "2.3.0",
        "pandas_version": "2.3.0",
        "scikit_learn_version": "1.9.0",
        "scipy_version": "1.15.0",
        "development_start": "2020-01-01",
        "locked_test_start": "2020-06-01",
        "locked_test_end": "2020-06-20",
        "locked_label_end_max": "2020-07-10",
        "execution_mode": "development",
        "run_fingerprint": "development-hash",
        **overrides,
    }


def _development(registry, *, manifest=None, study="study", complete=True):
    manifest = _manifest() if manifest is None else manifest
    run_id = registry.start_development(
        study_id=study,
        hypothesis="Combining metrics improves unseen-date ranking.",
        configuration=manifest["configuration"],
    )
    registry.record_development_plan(run_id, manifest=manifest)
    if complete:
        registry.complete_development(run_id, manifest=manifest, frozen_family="ridge")
    return run_id


def test_development_lifecycle_is_persistent_and_inputs_are_copied(tmp_path):
    path = tmp_path / "registry.sqlite"
    registry = ExperimentRegistry(path)
    manifest = _manifest()
    original = deepcopy(manifest)
    run_id = _development(registry, manifest=manifest)
    manifest["configuration"]["feature_columns"].append("changed")
    row = ExperimentRegistry(path).get_run(run_id)
    assert row["kind"] == "development"
    assert row["status"] == "completed"
    assert row["manifest"] == original
    assert row["configuration"] == original["configuration"]
    assert row["frozen_family"] == "ridge"
    assert row["development_run_id"] is None
    assert datetime.fromisoformat(row["started_at"]).utcoffset().total_seconds() == 0
    assert row["finished_at"] is not None
    row["configuration"]["feature_columns"].append("local-only")
    assert len(registry.list_runs()) == 1
    assert registry.get_run(run_id)["configuration"] == original["configuration"]


def test_final_uses_mode_independent_identity_and_is_persistent(tmp_path):
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    development_id = _development(registry)
    manifest = _manifest(execution_mode="full", run_fingerprint="final-hash")
    run_id = registry.reserve_final(
        development_run_id=development_id, manifest=manifest, frozen_family="ridge"
    )
    row = registry.get_run(run_id)
    assert row["kind"] == "final"
    assert row["status"] == "running"
    assert row["development_run_id"] == development_id
    assert row["exposure_start"] == "2020-06-01"
    assert row["exposure_end"] == "2020-07-10"
    registry.complete_final(
        run_id, manifest=manifest, acceptance={"model_gate_passed": False}
    )
    assert registry.get_run(run_id)["acceptance"] == {"model_gate_passed": False}
    assert registry.get_run(run_id)["status"] == "completed"


@pytest.mark.parametrize("state", ["running", "failed", "completed"])
def test_every_final_status_stays_consumed(tmp_path, state):
    path = tmp_path / "registry.sqlite"
    registry = ExperimentRegistry(path)
    development_id = _development(registry)
    final_id = registry.reserve_final(
        development_run_id=development_id, manifest=_manifest(), frozen_family="ridge"
    )
    if state == "failed":
        registry.fail_run(final_id, error_type="InterruptedError")
    elif state == "completed":
        registry.complete_final(final_id, manifest=_manifest(), acceptance={})
    reopened = ExperimentRegistry(path)
    with pytest.raises(ValueError, match="overlap"):
        reopened.reserve_final(
            development_run_id=development_id,
            manifest=_manifest(),
            frozen_family="ridge",
        )
    assert len(reopened.list_runs()) == 2


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("2020-06-01", "2020-06-20"),
        ("2020-06-05", "2020-06-10"),
        ("2020-07-01", "2020-07-20"),
        ("2020-07-10", "2020-07-20"),
    ],
)
def test_overlap_is_global_and_includes_label_extension(tmp_path, start, end):
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    first = _development(registry)
    registry.reserve_final(
        development_run_id=first, manifest=_manifest(), frozen_family="ridge"
    )
    changed = _manifest(
        panel_fingerprint="other-dataset",
        configuration={"feature_columns": ["different"], "hac_lags": 3},
        locked_test_start=start,
        locked_test_end=end,
        locked_label_end_max=end,
    )
    second = _development(registry, manifest=changed, study="another-study")
    with pytest.raises(ValueError, match="overlap"):
        registry.reserve_final(
            development_run_id=second, manifest=changed, frozen_family="ridge"
        )
    assert len(registry.list_runs()) == 3


@pytest.mark.parametrize("state", ["running", "failed", "completed"])
def test_prior_development_exposure_cannot_be_relabelled_final(tmp_path, state):
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    first = _development(registry, complete=state == "completed")
    if state == "failed":
        registry.fail_run(first, error_type="ValueError")
    shortened = _manifest(
        panel_fingerprint="shorter-panel",
        locked_test_start="2020-03-01",
        locked_test_end="2020-03-20",
        locked_label_end_max="2020-04-10",
    )
    second = _development(registry, manifest=shortened, study="shortened-study")
    with pytest.raises(ValueError, match="overlap"):
        registry.reserve_final(
            development_run_id=second, manifest=shortened, frozen_family="ridge"
        )


def test_later_disjoint_final_period_can_be_reserved(tmp_path):
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    first = _development(registry)
    registry.reserve_final(
        development_run_id=first, manifest=_manifest(), frozen_family="ridge"
    )
    later = _manifest(
        panel_fingerprint="extended-panel",
        locked_test_start="2020-08-01",
        locked_test_end="2020-08-20",
        locked_label_end_max="2020-09-10",
    )
    second = _development(registry, manifest=later, study="later-study")
    assert registry.reserve_final(
        development_run_id=second, manifest=later, frozen_family="ridge"
    )


@pytest.mark.parametrize(
    "field",
    [
        "panel_fingerprint",
        "source_fingerprint",
        "configuration",
        "package_version",
        "python_version",
        "numpy_version",
        "pandas_version",
        "scikit_learn_version",
        "scipy_version",
        "development_start",
        "locked_test_start",
        "locked_test_end",
        "locked_label_end_max",
    ],
)
def test_stale_identity_is_rejected_without_final_insert(tmp_path, field):
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    development_id = _development(registry)
    changed = _manifest()
    if field == "configuration":
        changed[field] = {"feature_columns": ["other"]}
    elif field.endswith("start"):
        changed[field] = "2020-02-01"
    elif field in {"locked_test_end", "locked_label_end_max"}:
        changed[field] = "2020-06-21"
    else:
        changed[field] = "different"
    with pytest.raises(ValueError):
        registry.reserve_final(
            development_run_id=development_id, manifest=changed, frozen_family="ridge"
        )
    assert len(registry.list_runs()) == 1


def test_failed_unknown_or_wrong_family_development_cannot_be_frozen(tmp_path):
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    running = _development(registry, complete=False)
    for run_id in [running, "unknown"]:
        with pytest.raises(ValueError):
            registry.reserve_final(
                development_run_id=run_id, manifest=_manifest(), frozen_family="ridge"
            )
    registry.fail_run(running, error_type="ValueError")
    with pytest.raises(ValueError):
        registry.complete_development(
            running, manifest=_manifest(), frozen_family="ridge"
        )
    complete = _development(registry)
    with pytest.raises(ValueError, match="frozen"):
        registry.reserve_final(
            development_run_id=complete, manifest=_manifest(), frozen_family="other"
        )


def test_failed_run_can_exist_before_plan_and_hypothesis_is_fixed(tmp_path):
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    run_id = registry.start_development(
        study_id="study", hypothesis="Original hypothesis", configuration={}
    )
    registry.fail_run(run_id, error_type="ValueError")
    row = registry.get_run(run_id)
    assert row["status"] == "failed"
    assert row["manifest"] is None
    assert row["error_type"] == "ValueError"
    with pytest.raises(ValueError, match="hypothesis"):
        registry.start_development(
            study_id="study", hypothesis="Rewritten hypothesis", configuration={}
        )


def test_plan_is_immutable_and_must_match_declared_configuration(tmp_path):
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    run_id = registry.start_development(
        study_id="study", hypothesis="Hypothesis", configuration={}
    )
    with pytest.raises(ValueError, match="configuration"):
        registry.record_development_plan(run_id, manifest=_manifest())
    declared = _manifest(configuration={})
    registry.record_development_plan(run_id, manifest=declared)
    with pytest.raises(ValueError):
        registry.record_development_plan(run_id, manifest=declared)
    with pytest.raises(ValueError):
        registry.complete_development(
            run_id,
            manifest={**declared, "source_fingerprint": "changed"},
            frozen_family="ridge",
        )
    assert registry.get_run(run_id)["status"] == "running"


def test_concurrent_reservations_allow_exactly_one_final(tmp_path):
    path = tmp_path / "registry.sqlite"
    registry = ExperimentRegistry(path)
    development_id = _development(registry)
    barrier = Barrier(2)

    def reserve():
        connection = ExperimentRegistry(path)
        barrier.wait(timeout=10)
        try:
            return connection.reserve_final(
                development_run_id=development_id,
                manifest=_manifest(),
                frozen_family="ridge",
            )
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: reserve(), range(2)))
    assert sum(result is not None for result in results) == 1
    assert len(registry.list_runs()) == 2


@pytest.mark.parametrize("kind", ["empty", "unrelated", "corrupt", "future"])
def test_existing_nonregistry_or_unsupported_database_is_not_reinitialized(
    tmp_path, kind
):
    path = tmp_path / "registry.sqlite"
    if kind == "future":
        ExperimentRegistry(path)
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute("PRAGMA user_version = 999")
    elif kind in {"empty", "unrelated"}:
        with closing(sqlite3.connect(path)) as connection, connection:
            if kind == "unrelated":
                connection.execute("CREATE TABLE unrelated(value TEXT)")
    else:
        path.write_bytes(b"not a sqlite database")
    before = path.read_bytes()
    with pytest.raises(ValueError):
        ExperimentRegistry(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("path", [":memory:", "", "file:memory?mode=memory", None, 3])
def test_invalid_or_nondurable_registry_paths_are_rejected(path):
    with pytest.raises((TypeError, ValueError)):
        ExperimentRegistry(path)


def test_invalid_input_and_state_changes_are_rejected(tmp_path):
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    for study, hypothesis, config in [
        ("", "hypothesis", {}),
        ("s", "", {}),
        ("s", "h", []),
    ]:
        with pytest.raises(ValueError):
            registry.start_development(
                study_id=study, hypothesis=hypothesis, configuration=config
            )
    with pytest.raises(ValueError):
        registry.start_development(
            study_id="study",
            hypothesis="hypothesis",
            configuration={"value": float("nan")},
        )
    complete = _development(registry)
    with pytest.raises(ValueError):
        registry.fail_run(complete, error_type="ValueError")
    with pytest.raises(ValueError):
        registry.fail_run(complete, error_type="Sensitive details: secret token")
    with pytest.raises(ValueError):
        registry.complete_final(complete, manifest=_manifest(), acceptance={})
    with pytest.raises(ValueError):
        registry.get_run("unknown")


@pytest.mark.parametrize(
    "changes",
    [
        {"locked_test_start": "2020-06-01T00:00:00"},
        {"locked_test_end": "not-a-date"},
        {"locked_label_end_max": "2020-06-10"},
        {"development_start": "2020-06-01"},
        {"configuration": []},
    ],
)
def test_invalid_manifest_does_not_record_plan(tmp_path, changes):
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    manifest = _manifest()
    run_id = registry.start_development(
        study_id="study",
        hypothesis="hypothesis",
        configuration=manifest["configuration"],
    )
    with pytest.raises(ValueError):
        registry.record_development_plan(run_id, manifest={**manifest, **changes})
    assert registry.get_run(run_id)["manifest"] is None


def test_final_completion_cannot_change_reserved_identity(tmp_path):
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    development_id = _development(registry)
    run_id = registry.reserve_final(
        development_run_id=development_id, manifest=_manifest(), frozen_family="ridge"
    )
    with pytest.raises(ValueError):
        registry.complete_final(
            run_id, manifest=_manifest(source_fingerprint="changed"), acceptance={}
        )
    assert registry.get_run(run_id)["status"] == "running"


def test_whitespace_run_id_cannot_silently_skip_state_update(tmp_path):
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    run_id = registry.start_development(
        study_id="study", hypothesis="hypothesis", configuration={}
    )
    with pytest.raises(ValueError):
        registry.fail_run(f" {run_id} ", error_type="ValueError")
    assert registry.get_run(run_id)["status"] == "running"


def test_removed_registry_is_not_silently_recreated(tmp_path):
    path = tmp_path / "registry.sqlite"
    registry = ExperimentRegistry(path)
    _development(registry)
    path.unlink()
    with pytest.raises(ValueError):
        registry.list_runs()
    assert not path.exists()


def test_sql_looking_user_text_is_only_data(tmp_path):
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    study = "study'; DROP TABLE runs; --"
    run_id = registry.start_development(
        study_id=study, hypothesis="hypothesis", configuration={}
    )
    assert registry.get_run(run_id)["study_id"] == study
    assert len(registry.list_runs()) == 1


def test_unavailable_acceptance_statistics_are_persisted_as_null(tmp_path):
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    development = _development(registry)
    run_id = registry.reserve_final(
        development_run_id=development, manifest=_manifest(), frozen_family="ridge"
    )
    registry.complete_final(
        run_id,
        manifest=_manifest(),
        acceptance={"ic_t_stat": float("nan"), "p_value": float("inf")},
    )
    assert registry.get_run(run_id)["acceptance"] == {
        "ic_t_stat": None,
        "p_value": None,
    }


@pytest.mark.parametrize(
    "configuration",
    [{1: "non-string-key"}, {"unsupported": object()}, {"nested": [float("inf")]}],
)
def test_nonjson_configuration_is_rejected(tmp_path, configuration):
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    with pytest.raises(ValueError):
        registry.start_development(
            study_id="study", hypothesis="hypothesis", configuration=configuration
        )
    assert registry.list_runs() == []


def test_missing_manifest_identity_and_completion_without_plan_are_rejected(tmp_path):
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    run_id = registry.start_development(
        study_id="study", hypothesis="hypothesis", configuration={}
    )
    with pytest.raises(ValueError):
        registry.record_development_plan(run_id, manifest={})
    with pytest.raises(ValueError):
        registry.complete_development(
            run_id, manifest=_manifest(), frozen_family="ridge"
        )


def test_final_plan_validation_is_nonmutating_and_does_not_consume_holdout(tmp_path):
    path = tmp_path / "registry.sqlite"
    registry = ExperimentRegistry(path)
    development_id = _development(registry)
    before = path.read_bytes()
    assert registry.validate_final_plan(development_id, manifest=_manifest()) is None
    assert path.read_bytes() == before
    assert len(registry.list_runs()) == 1
    assert registry.reserve_final(
        development_run_id=development_id, manifest=_manifest(), frozen_family="ridge"
    )


@pytest.mark.parametrize("change", ["identity", "configuration", "dates", "scipy"])
def test_final_plan_validation_rejects_changed_inputs_before_any_rerun(
    tmp_path, change
):
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    development_id = _development(registry)
    overrides = {
        "identity": {"panel_fingerprint": "different-panel"},
        "configuration": {"configuration": {"feature_columns": ["different"]}},
        "dates": {"development_start": "2019-01-01"},
        "scipy": {"scipy_version": "2.0.0"},
    }
    before = registry.list_runs()
    with pytest.raises(ValueError, match="identity"):
        registry.validate_final_plan(
            development_id, manifest=_manifest(**overrides[change])
        )
    assert registry.list_runs() == before


def test_final_plan_validation_and_reservation_both_recheck_exposure(tmp_path):
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    development_id = _development(registry)
    registry.validate_final_plan(development_id, manifest=_manifest())
    registry.reserve_final(
        development_run_id=development_id, manifest=_manifest(), frozen_family="ridge"
    )
    with pytest.raises(ValueError, match="overlap"):
        registry.validate_final_plan(development_id, manifest=_manifest())
    with pytest.raises(ValueError, match="overlap"):
        registry.reserve_final(
            development_run_id=development_id,
            manifest=_manifest(),
            frozen_family="ridge",
        )
    assert len(registry.list_runs()) == 2


@pytest.mark.parametrize("status", ["running", "failed", "unknown"])
def test_final_plan_validation_requires_completed_development(tmp_path, status):
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    development_id = _development(registry, complete=False)
    if status == "failed":
        registry.fail_run(development_id, error_type="ValueError")
    with pytest.raises(ValueError):
        registry.validate_final_plan(
            "unknown" if status == "unknown" else development_id, manifest=_manifest()
        )
    assert len(registry.list_runs()) == 1
