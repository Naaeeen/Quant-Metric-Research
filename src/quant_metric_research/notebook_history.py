"""Sequential, development-only notebook runs with append-only file checkpoints.

The caller supplies local SQLite/work paths and an existing private history
folder, which may be mounted storage. No database is opened on history storage.
This is a single-user workflow, not a distributed lock or server-durability
guarantee. Every unresolved generation requires manual review; never discard
history or retry from an older checkpoint to bypass recorded exposure.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir

from ._notebook_history_io import (
    checked_path,
    copy_file,
    copy_tree,
    fingerprint,
    inventory,
    read_json,
    registry_rows,
    require_disjoint,
    require_extension,
    require_local_compute_path,
    snapshot_registry,
    validate_inventory,
    write_json,
)
from ._version import __version__
from .public_demo import run_public_demo

_START_KEYS = {
    "schema_version",
    "generation",
    "parent_checkpoint_sha256",
    "operation",
    "started_at",
    "final_outcomes_evaluated",
    "declared_identity",
}
_CHECKPOINT_KEYS = {
    "schema_version",
    "generation",
    "parent_checkpoint_sha256",
    "start_sha256",
    "status",
    "error_type",
    "inventory",
    "run_count",
    "exposure_count",
    "final_outcomes_evaluated",
}


def _declared_identity() -> dict:
    from .public_demo import public_demo_configs

    panel, benchmark = public_demo_configs()
    source_hasher = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        source_hasher.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    identity = {
        "package_version": __version__,
        "source_fingerprint": source_hasher.hexdigest(),
        "runtime_versions": {
            "python": platform.python_version(),
            **{
                name: version(name.replace("_", "-"))
                for name in (
                    "numpy",
                    "pandas",
                    "scipy",
                    "scikit_learn",
                )
            },
        },
        "panel_configuration": asdict(panel),
        "benchmark_configuration": benchmark.to_mapping(),
    }
    return json.loads(json.dumps(identity, allow_nan=False))


def _validate_identity(value: dict | None, *, seed: bool) -> None:
    if seed and value is None:
        return
    if (
        seed
        or not isinstance(value, dict)
        or set(value)
        != {
            "package_version",
            "source_fingerprint",
            "runtime_versions",
            "panel_configuration",
            "benchmark_configuration",
        }
    ):
        raise ValueError("Invalid declared notebook identity.")
    versions = value["runtime_versions"]
    if (
        not isinstance(versions, dict)
        or set(versions) != {"python", "numpy", "pandas", "scipy", "scikit_learn"}
        or any(
            not isinstance(item, str) or not item or len(item) > 256
            for item in [value["package_version"], *versions.values()]
        )
        or not isinstance(value["source_fingerprint"], str)
        or re.fullmatch(r"[0-9a-f]{64}", value["source_fingerprint"]) is None
        or not isinstance(value["panel_configuration"], dict)
        or not isinstance(value["benchmark_configuration"], dict)
    ):
        raise ValueError("Invalid declared notebook configuration or runtime identity.")


def _start(generation: Path, parent: str | None) -> None:
    generation.mkdir(exist_ok=False)
    value = {
        "schema_version": 1,
        "generation": generation.name,
        "parent_checkpoint_sha256": parent,
        "operation": "seed" if parent is None else "public_demo",
        "started_at": datetime.now(UTC).isoformat(),
        "final_outcomes_evaluated": False,
        "declared_identity": None if parent is None else _declared_identity(),
    }
    write_json(value, generation / "start.json")
    if read_json(generation / "start.json") != value:
        raise ValueError("Start marker read-back failed; training was not started.")


def _verify_generation(generation: Path, parent: str | None) -> dict:
    if (generation / "checkpoint_failed.json").exists():
        raise ValueError("Checkpoint failed; generation requires manual review.")
    if (
        not (generation / "evidence").is_dir()
        or not (generation / "registry.sqlite3").is_file()
    ):
        raise ValueError("Generation evidence or registry is missing or invalid.")
    start = read_json(generation / "start.json")
    checkpoint = read_json(generation / "checkpoint.json")
    if set(start) != _START_KEYS or set(checkpoint) != _CHECKPOINT_KEYS:
        raise ValueError(
            "History marker schema is invalid or generation is unresolved."
        )
    for marker in (start, checkpoint):
        if (
            type(marker["schema_version"]) is not int
            or marker["schema_version"] != 1
            or marker["generation"] != generation.name
            or marker["parent_checkpoint_sha256"] != parent
            or marker["final_outcomes_evaluated"] is not False
        ):
            raise ValueError("History generation identity or parent chain is invalid.")
    if start["operation"] != ("seed" if parent is None else "public_demo"):
        raise ValueError("History operation is invalid.")
    _validate_identity(start["declared_identity"], seed=parent is None)
    try:
        timestamp = datetime.fromisoformat(start["started_at"])
        if timestamp.utcoffset() is None:
            raise ValueError("Start time must include a timezone.")
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid history start timestamp.") from error
    allowed = {"seeded"} if parent is None else {"completed", "failed"}
    if not isinstance(checkpoint["status"], str) or checkpoint["status"] not in allowed:
        raise ValueError("Invalid history completion status.")
    error_type = checkpoint["error_type"]
    if (
        checkpoint["status"] == "failed"
        and (
            not isinstance(error_type, str)
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", error_type) is None
        )
    ) or (checkpoint["status"] != "failed" and error_type is not None):
        raise ValueError("Invalid checkpoint error type.")
    if any(
        type(checkpoint[key]) is not int or checkpoint[key] < 0
        for key in ("run_count", "exposure_count")
    ):
        raise ValueError("Invalid checkpoint registry counts.")
    validate_inventory(checkpoint["inventory"])
    actual = inventory(generation, omit_checkpoint=True)
    if (
        checkpoint["inventory"] != actual
        or checkpoint["start_sha256"]
        != fingerprint(generation / "start.json")["sha256"]
    ):
        raise ValueError("History inventory or file fingerprints changed.")
    if set(path.name for path in generation.iterdir()) != {
        "start.json",
        "registry.sqlite3",
        "evidence",
        "checkpoint.json",
    }:
        raise ValueError("Generation files are incomplete or unexpected.")
    return checkpoint


def _validate_history(directory: Path) -> tuple[dict, dict, str]:
    if not directory.is_dir():
        raise ValueError("An existing seeded history directory is required.")
    generations = sorted(directory.iterdir())
    if not generations or len(generations) > 999999:
        raise ValueError("History must contain a bounded nonempty generation chain.")
    parent, previous, latest = None, None, None
    temporary_parent = require_local_compute_path(gettempdir())
    if directory == temporary_parent or directory in temporary_parent.parents:
        raise ValueError("Temporary SQLite files must remain outside history storage.")
    with TemporaryDirectory(
        prefix="qmr-history-validate-", dir=temporary_parent
    ) as temporary:
        local = require_local_compute_path(temporary)
        require_disjoint(local, directory)
        for index, generation in enumerate(generations):
            checked_path(generation)
            if generation.name != f"{index:06d}" or not generation.is_dir():
                raise ValueError(
                    "History generations must be contiguous without extra files."
                )
            latest = _verify_generation(generation, parent)
            restored = local / f"{index:06d}.sqlite3"
            copy_file(generation / "registry.sqlite3", restored)
            current = registry_rows(restored)
            if latest["run_count"] != len(current["runs"]) or latest[
                "exposure_count"
            ] != len(current["exposures"]):
                raise ValueError("Registry counts disagree with the checkpoint.")
            if previous is not None:
                require_extension(previous, current)
            previous = current
            parent = fingerprint(generation / "checkpoint.json")["sha256"]
    return previous, latest, parent


def _write_checkpoint(
    *,
    generation: Path,
    parent: str | None,
    registry: Path,
    evidence: Mapping[str, Path],
    previous: dict | None,
    status: str,
    error_type: str | None = None,
) -> dict:
    registry = require_local_compute_path(registry)
    with TemporaryDirectory(
        prefix="qmr-history-snapshot-", dir=registry.parent
    ) as temporary:
        snapshot = require_local_compute_path(Path(temporary) / "registry.sqlite3")
        current = snapshot_registry(registry, snapshot)
        if previous is not None:
            require_extension(previous, current)
        copy_file(snapshot, generation / "registry.sqlite3")
    evidence_root = generation / "evidence"
    evidence_root.mkdir(exist_ok=False)
    for name, source in evidence.items():
        copy_tree(source, evidence_root / name)
    value = {
        "schema_version": 1,
        "generation": generation.name,
        "parent_checkpoint_sha256": parent,
        "start_sha256": fingerprint(generation / "start.json")["sha256"],
        "status": status,
        "error_type": error_type,
        "inventory": inventory(generation),
        "run_count": len(current["runs"]),
        "exposure_count": len(current["exposures"]),
        "final_outcomes_evaluated": False,
    }
    write_json(value, generation / "checkpoint.json")
    if _verify_generation(generation, parent) != value:
        raise ValueError("Checkpoint read-back verification failed.")
    return value


def _checkpoint(**kwargs) -> dict:
    try:
        return _write_checkpoint(**kwargs)
    except BaseException as error:
        generation = kwargs["generation"]
        try:
            write_json(
                {"status": "checkpoint_failed", "error_type": type(error).__name__},
                generation / "checkpoint_failed.json",
            )
        except BaseException as marker_error:
            error.add_note(
                f"Failure marker unavailable for {generation} "
                f"({type(marker_error).__name__}); stop automatic retries and "
                "retain the local work directory for manual review."
            )
        raise


def seed_notebook_history(
    *,
    registry_path: str | Path,
    history_dir: str | Path,
    evidence_dirs: Mapping[str, Path],
) -> dict:
    """Copy an existing canonical local registry and evidence to NEW history.

    All historical rows, including running/failed attempts and their exposure,
    are retained. Evidence aliases become 000000/evidence/<alias>. Keep the
    original local registry and source evidence; seeding never changes them.
    """
    registry = require_local_compute_path(registry_path)
    destination = checked_path(history_dir)
    if destination.exists():
        raise FileExistsError("History directory already exists; never overwrite it.")
    require_disjoint(registry, destination)
    registry_rows(registry)
    if not isinstance(evidence_dirs, Mapping):
        raise ValueError("Evidence directories must be an alias-to-path mapping.")
    evidence = {}
    for name, path in evidence_dirs.items():
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name) is None
        ):
            raise ValueError("Evidence aliases must be simple bounded directory names.")
        source = checked_path(path)
        require_disjoint(source, destination)
        require_disjoint(source, registry)
        inventory(source)
        evidence[name] = source
    destination.mkdir(parents=True, exist_ok=False)
    generation = destination / "000000"
    _start(generation, None)
    return _checkpoint(
        generation=generation,
        parent=None,
        registry=registry,
        evidence=evidence,
        previous=None,
        status="seeded",
    )


def run_checkpointed_public_demo(
    *,
    archive_dir: str | Path,
    work_dir: str | Path,
    history_dir: str | Path,
) -> dict:
    """Restore the complete validated chain and run the existing offline demo.

    work_dir must be NEW and local, outside both history and archive. A start
    marker is written and reread before training. Catchable BaseException saves
    the current registry and partial demo evidence, then preserves the original
    exception. Abrupt VM/process loss leaves an unresolved generation that
    blocks all subsequent automatic runs. Read-back is not proof of remote
    server durability. Use sequentially with one writer and trusted local inputs.
    """
    work = require_local_compute_path(work_dir)
    archive, directory = map(checked_path, (archive_dir, history_dir))
    require_disjoint(work, directory)
    require_disjoint(work, archive)
    if work.exists():
        raise FileExistsError("Work directory already exists; choose a new directory.")
    if not archive.is_dir():
        raise ValueError("An existing offline archive directory is required.")
    previous, latest, parent = _validate_history(directory)
    work.mkdir(parents=True, exist_ok=False)
    registry = work / "registry.sqlite3"
    copy_file(directory / latest["generation"] / "registry.sqlite3", registry)
    if registry_rows(registry) != previous:
        raise ValueError("Restored registry differs from the validated history.")
    generation = directory / f"{int(latest['generation']) + 1:06d}"
    _start(generation, parent)
    demo = work / "demo"
    try:
        run_public_demo(archive_dir=archive, output_dir=demo, registry_path=registry)
    except BaseException as error:
        try:
            _checkpoint(
                generation=generation,
                parent=parent,
                registry=registry,
                evidence={"demo": demo} if demo.exists() else {},
                previous=previous,
                status="failed",
                error_type=type(error).__name__,
            )
        except BaseException as checkpoint_error:
            error.add_note(
                f"Checkpoint unavailable for {generation}; stop automatic retries "
                "and retain local work for manual review "
                f"({type(checkpoint_error).__name__})."
            )
        raise
    return _checkpoint(
        generation=generation,
        parent=parent,
        registry=registry,
        evidence={"demo": demo} if demo.exists() else {},
        previous=previous,
        status="completed",
    )


__all__ = ["seed_notebook_history", "run_checkpointed_public_demo"]
