"""Private bounded file validation and local SQLite snapshots for notebook history."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import stat
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path, PurePosixPath

from .experiment_registry import _CREATE_EXPOSURES, _CREATE_RUNS, _manifest, _schema
from .intake import _local_path

MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_FILE_BYTES = 2 * 1024**3
MAX_TOTAL_BYTES = 20 * 1024**3
MAX_ENTRIES = 100_000


def checked_path(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("A nonempty filesystem path is required.")
    raw = str(value)
    if "\0" in raw or raw.startswith(("file:", "\\\\", "//")) or raw == ":memory:":
        raise ValueError("Use a filesystem path; SQLite sources must be local.")
    path = _local_path(value).expanduser().absolute()
    non_drive = raw[2:] if re.match(r"^[a-zA-Z]:[\\/]", raw) else raw
    if ":" in non_drive:
        raise ValueError("Windows alternate data streams are not supported.")
    for component in (path, *path.parents):
        try:
            details = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(details.st_mode) or (
            getattr(details, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024)
        ):
            raise ValueError(
                "Symlinks and junctions are not supported in history paths."
            )
    return path.resolve()


def require_disjoint(first: Path, second: Path) -> None:
    if first == second or first in second.parents or second in first.parents:
        raise ValueError("Paths must be outside one another without overlap.")


def require_local_compute_path(value: str | Path) -> Path:
    """Reject standard Colab Drive mounts; other mounts remain caller responsibility."""
    path = checked_path(value)
    normalized = re.sub(r"^[A-Za-z]:", "", path.as_posix())
    if any(
        normalized == mount or normalized.startswith(mount + "/")
        for mount in ("/content/drive", "/content/gdrive")
    ):
        raise ValueError("SQLite and compute paths must be local, outside Colab Drive.")
    return path


def relative_name(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 1024
        or "\\" in value
        or ":" in value
        or "\0" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or PurePosixPath(value).is_absolute()
    ):
        raise ValueError("Inventory contains an unsafe relative path.")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    if len(dict(pairs)) != len(pairs):
        raise ValueError("Duplicate JSON keys are not permitted.")
    return dict(pairs)


def _invalid_constant(value: str) -> None:
    raise ValueError("Nonfinite JSON values are not permitted.")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Nonfinite JSON values are not permitted.")
    return number


def decode_json(text: str) -> dict:
    try:
        result = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
            parse_float=_finite_float,
        )
    except (json.JSONDecodeError, RecursionError) as error:
        raise ValueError("History contains invalid JSON.") from error
    if not isinstance(result, dict):
        raise ValueError("History metadata must be a JSON object.")
    return result


def read_json(path: Path) -> dict:
    checked_path(path)
    if not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("History JSON is missing or exceeds the size limit.")
    try:
        return decode_json(path.read_text(encoding="utf-8"))
    except UnicodeError as error:
        raise ValueError("History JSON must be UTF-8.") from error


def write_json(value: dict, path: Path) -> None:
    encoded = (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    if len(encoded) > MAX_JSON_BYTES:
        raise ValueError("History JSON exceeds the size limit.")
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def fingerprint(path: Path) -> dict:
    checked_path(path)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_FILE_BYTES:
        raise ValueError("History evidence must contain bounded regular files.")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Evidence changed while being read.")
    return {"size": before.st_size, "sha256": digest.hexdigest()}


def inventory(directory: Path, *, omit_checkpoint: bool = False) -> dict:
    checked_path(directory)
    if not directory.is_dir():
        raise ValueError("Evidence directory is missing.")
    paths = sorted(
        directory.rglob("*"), key=lambda path: path.relative_to(directory).as_posix()
    )
    if len(paths) > MAX_ENTRIES:
        raise ValueError("Evidence inventory exceeds the entry limit.")
    directories, files = [], {}
    for path in paths:
        name = relative_name(path.relative_to(directory).as_posix())
        checked_path(path)
        if omit_checkpoint and name == "checkpoint.json":
            continue
        if path.is_dir():
            directories.append(name)
        else:
            files[name] = fingerprint(path)
    if sum(item["size"] for item in files.values()) > MAX_TOTAL_BYTES:
        raise ValueError("Evidence inventory exceeds the total size limit.")
    return {"directories": directories, "files": files}


def validate_inventory(value: dict) -> None:
    if not isinstance(value, dict) or set(value) != {"directories", "files"}:
        raise ValueError("Invalid history inventory schema.")
    directories, files = value["directories"], value["files"]
    if not isinstance(directories, list) or not isinstance(files, dict):
        raise ValueError("Invalid history inventory collections.")
    if len(directories) + len(files) > MAX_ENTRIES:
        raise ValueError("History inventory exceeds the entry limit.")
    for name in [*directories, *files]:
        relative_name(name)
    if directories != sorted(set(directories)) or set(directories) & set(files):
        raise ValueError("Invalid or duplicate inventory paths.")
    for item in files.values():
        if (
            not isinstance(item, dict)
            or set(item) != {"size", "sha256"}
            or type(item["size"]) is not int
            or not 0 <= item["size"] <= MAX_FILE_BYTES
            or not isinstance(item["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None
        ):
            raise ValueError("Invalid inventory size or SHA-256 digest.")
    if sum(item["size"] for item in files.values()) > MAX_TOTAL_BYTES:
        raise ValueError("History inventory exceeds the total size limit.")


def copy_file(source: Path, destination: Path) -> None:
    expected = fingerprint(source)
    checked_path(destination)
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        for block in iter(lambda: incoming.read(1024 * 1024), b""):
            outgoing.write(block)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    if fingerprint(destination) != expected or fingerprint(source) != expected:
        raise ValueError("Evidence copy changed or failed verification.")


def copy_tree(source: Path, destination: Path) -> None:
    expected = inventory(source)
    destination.mkdir(exist_ok=False)
    for name in expected["directories"]:
        (destination / name).mkdir()
    for name in expected["files"]:
        copy_file(source / name, destination / name)
    if inventory(source) != expected or inventory(destination) != expected:
        raise ValueError("Evidence tree changed during copying.")


def _validate_records(runs: dict, exposures: dict) -> None:
    for run_id, row in runs.items():
        record = dict(row)
        if (
            not isinstance(run_id, str)
            or not run_id
            or record["kind"] not in {"development", "final"}
            or record["status"] not in {"running", "failed", "completed"}
            or any(
                value is not None and not isinstance(value, str)
                for value in record.values()
            )
        ):
            raise ValueError("Registry contains invalid run rows.")
        configuration = decode_json(record["configuration"])
        for field in ("manifest", "acceptance"):
            if record[field] is not None:
                decode_json(record[field])
        if record["manifest"] is None:
            if (
                run_id in exposures
                or record["kind"] == "final"
                or record["status"] == "completed"
            ):
                raise ValueError("Registry exposure or completed plan is missing.")
            continue
        plan = _manifest(decode_json(record["manifest"]))
        if plan["configuration"] != configuration:
            raise ValueError("Registry plan configuration is inconsistent.")
        expected = (
            (
                plan["development_start"],
                (
                    date.fromisoformat(plan["locked_test_start"]) - timedelta(days=1)
                ).isoformat(),
            )
            if record["kind"] == "development"
            else (plan["locked_test_start"], plan["locked_label_end_max"])
        )
        if exposures.get(run_id) != (run_id, *expected):
            raise ValueError("Registry exposure is missing or inconsistent.")
    if not exposures.keys() <= runs.keys():
        raise ValueError("Registry contains orphan exposures.")


def registry_rows(path: Path) -> dict:
    """Read only a local database; preserve raw metadata strings for comparison."""
    path = require_local_compute_path(path)
    if not path.is_file():
        raise ValueError("An existing local experiment registry is required.")
    try:
        with closing(
            sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        ) as connection:
            _schema(connection)
            objects = connection.execute(
                "SELECT type, name, sql FROM sqlite_master WHERE sql IS NOT NULL"
            ).fetchall()
            expected = {
                ("table", "runs", " ".join(_CREATE_RUNS.split())),
                ("table", "exposures", " ".join(_CREATE_EXPOSURES.split())),
            }
            if {
                (kind, name, " ".join(sql.split())) for kind, name, sql in objects
            } != expected:
                raise ValueError(
                    "Registry schema must be canonical without extra objects."
                )
            if (
                connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]
                or connection.execute("PRAGMA foreign_key_check").fetchall()
            ):
                raise ValueError("Registry integrity validation failed.")
            connection.row_factory = sqlite3.Row
            runs = {
                row["run_id"]: tuple(dict(row).items())
                for row in connection.execute("SELECT * FROM runs ORDER BY run_id")
            }
            exposures = {
                row["run_id"]: tuple(row)
                for row in connection.execute("SELECT * FROM exposures ORDER BY run_id")
            }
        _validate_records(runs, exposures)
        return {"runs": runs, "exposures": exposures}
    except (sqlite3.Error, TypeError) as error:
        raise ValueError("Local registry validation failed.") from error


def snapshot_registry(source: Path, destination: Path) -> dict:
    source = require_local_compute_path(source)
    destination = require_local_compute_path(destination)
    before = registry_rows(source)
    if destination.exists():
        raise FileExistsError("Local snapshot destination already exists.")
    with (
        closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as incoming,
        closing(sqlite3.connect(destination)) as outgoing,
    ):
        incoming.backup(outgoing)
    after = registry_rows(destination)
    if after != before or registry_rows(source) != before:
        raise ValueError("Registry changed during the local snapshot.")
    return after


def require_extension(previous: dict, current: dict) -> None:
    for table in ("runs", "exposures"):
        if any(current[table].get(key) != row for key, row in previous[table].items()):
            raise ValueError("Registry history regressed or previous rows changed.")
    new_ids = current["runs"].keys() - previous["runs"].keys()
    if any(dict(current["runs"][key])["kind"] != "development" for key in new_ids):
        raise ValueError("Notebook history only permits new development records.")
