"""Local, append-preserving experiment evidence and outcome-exposure guard.

Use one durable SQLite file for related research on one machine. This prevents
accidental reuse within that file, not manual database edits, a different file,
prior human inspection, or direct access to data. Reservations are committed
before evaluation; even failed or interrupted attempts retain their exposure.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from numbers import Integral, Real
from pathlib import Path
from typing import Any
from uuid import uuid4

_APPLICATION_ID = 1364021829
_SCHEMA_VERSION = 1
_IDENTITY_KEYS = (
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
)
_JSON_FIELDS = ("configuration", "manifest", "acceptance")
_RUN_COLUMNS = {
    "run_id",
    "kind",
    "status",
    "study_id",
    "hypothesis",
    "configuration",
    "manifest",
    "frozen_family",
    "development_run_id",
    "error_type",
    "acceptance",
    "started_at",
    "finished_at",
}
_CREATE_RUNS = """
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK(kind IN ('development', 'final')),
    status TEXT NOT NULL CHECK(status IN ('running', 'completed', 'failed')),
    study_id TEXT NOT NULL,
    hypothesis TEXT NOT NULL,
    configuration TEXT NOT NULL,
    manifest TEXT,
    frozen_family TEXT,
    development_run_id TEXT REFERENCES runs(run_id),
    error_type TEXT,
    acceptance TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT
)
"""
_CREATE_EXPOSURES = """
CREATE TABLE exposures (
    run_id TEXT PRIMARY KEY REFERENCES runs(run_id),
    start TEXT NOT NULL,
    end TEXT NOT NULL,
    CHECK(start <= end)
)
"""
_SELECT_RUN = """
SELECT runs.*, exposures.start AS exposure_start, exposures.end AS exposure_end
FROM runs LEFT JOIN exposures ON runs.run_id = exposures.run_id
"""


def _text(value: Any, name: str, *, limit: int = 4096) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > limit
        or "\0" in value
    ):
        raise ValueError(f"{name} must be a non-empty bounded string.")
    return value.strip()


def _plain(value: Any, *, nullable_nonfinite: bool = False) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings.")
        return {
            key: _plain(item, nullable_nonfinite=nullable_nonfinite)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_plain(item, nullable_nonfinite=nullable_nonfinite) for item in value]
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        if not math.isfinite(value):
            if nullable_nonfinite:
                return None
            raise ValueError("Registry metadata must contain finite JSON values.")
        return float(value)
    raise ValueError("Registry metadata must contain JSON-compatible values.")


def _mapping(
    value: Mapping[str, Any], name: str, *, nullable_nonfinite: bool = False
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return _plain(value, nullable_nonfinite=nullable_nonfinite)


def _encode(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    result = _mapping(value, "manifest")
    if any(key not in result for key in _IDENTITY_KEYS):
        raise ValueError("Manifest is missing required reproducibility identity.")
    _mapping(result["configuration"], "manifest configuration")
    for key in _IDENTITY_KEYS:
        if key != "configuration":
            _text(result[key], f"manifest {key}")
    dates: dict[str, date] = {}
    for key in (
        "development_start",
        "locked_test_start",
        "locked_test_end",
        "locked_label_end_max",
    ):
        try:
            parsed = date.fromisoformat(result[key])
        except ValueError:
            raise ValueError(
                "Manifest dates must be normalized ISO calendar dates."
            ) from None
        if parsed.isoformat() != result[key]:
            raise ValueError("Manifest dates must be normalized ISO calendar dates.")
        dates[key] = parsed
    if not (
        dates["development_start"]
        < dates["locked_test_start"]
        <= dates["locked_test_end"]
        <= dates["locked_label_end_max"]
    ):
        raise ValueError("Manifest development and final boundaries are inconsistent.")
    return result


def _match_identity(recorded: Mapping[str, Any], proposed: Mapping[str, Any]) -> None:
    if any(recorded[key] != proposed[key] for key in _IDENTITY_KEYS):
        raise ValueError(
            "Manifest identity differs from the recorded development plan."
        )


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _schema(connection: sqlite3.Connection) -> None:
    application = connection.execute("PRAGMA application_id").fetchone()[0]
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    if (
        application != _APPLICATION_ID
        or version != _SCHEMA_VERSION
        or tables != {"runs", "exposures"}
    ):
        raise ValueError("File is not a supported experiment registry.")
    columns = {row[1] for row in connection.execute("PRAGMA table_info(runs)")}
    exposure_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(exposures)")
    }
    if columns != _RUN_COLUMNS or exposure_columns != {"run_id", "start", "end"}:
        raise ValueError("Experiment registry schema is invalid.")


def _decode(row: sqlite3.Row) -> dict[str, Any]:
    try:
        return {
            key: json.loads(value)
            if key in _JSON_FIELDS and value is not None
            else value
            for key, value in dict(row).items()
        }
    except (json.JSONDecodeError, TypeError):
        raise ValueError("Experiment registry contains invalid metadata.") from None


def _get(connection: sqlite3.Connection, run_id: str) -> dict[str, Any]:
    if _text(run_id, "run_id", limit=256) != run_id:
        raise ValueError("run_id must not contain surrounding whitespace.")
    row = connection.execute(
        _SELECT_RUN + " WHERE runs.run_id = ?", (run_id,)
    ).fetchone()
    if row is None:
        raise ValueError("Unknown experiment run.")
    return _decode(row)


def _running(record: Mapping[str, Any], kind: str) -> None:
    if record["kind"] != kind or record["status"] != "running":
        raise ValueError("Run kind or lifecycle status does not permit this operation.")


def _final_plan(
    connection: sqlite3.Connection,
    development_run_id: str,
    proposed: Mapping[str, Any],
) -> dict[str, Any]:
    development = _get(connection, development_run_id)
    if development["kind"] != "development" or development["status"] != "completed":
        raise ValueError("Final evaluation requires completed development evidence.")
    _match_identity(_manifest(development["manifest"]), proposed)
    prior = connection.execute(
        "SELECT run_id FROM exposures WHERE start <= ? AND end >= ? LIMIT 1",
        (proposed["locked_label_end_max"], proposed["locked_test_start"]),
    ).fetchone()
    if prior is not None:
        raise ValueError("Final period overlaps previously exposed outcome dates.")
    return development


class ExperimentRegistry:
    """Persist local run history and conservatively block exposed final periods."""

    def __init__(self, path: str | Path) -> None:
        if not isinstance(path, (str, Path)):
            raise ValueError("Registry path must name a durable local SQLite file.")
        raw = str(path)
        if not raw.strip() or raw == ":memory:" or raw.startswith("file:"):
            raise ValueError("Registry path must name a durable local SQLite file.")
        self._path = Path(path).expanduser().resolve()
        if self._path.exists() and not self._path.is_file():
            raise ValueError("Registry path must name a file, not a directory.")
        if not self._path.exists():
            self._create()
        with self._connection():
            pass

    def _create(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            connection = sqlite3.connect(self._path, isolation_level=None, timeout=30)
            try:
                connection.execute("BEGIN IMMEDIATE")
                objects = connection.execute(
                    "SELECT name FROM sqlite_master"
                ).fetchall()
                if objects:
                    _schema(connection)
                else:
                    connection.execute(_CREATE_RUNS)
                    connection.execute(_CREATE_EXPOSURES)
                    connection.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
                    connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
                connection.commit()
            finally:
                connection.close()
        except sqlite3.Error:
            raise ValueError(
                "Experiment registry could not be initialized safely."
            ) from None

    @contextmanager
    def _connection(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        try:
            connection = sqlite3.connect(
                self._path.as_uri() + "?mode=rw",
                uri=True,
                isolation_level=None,
                timeout=30,
            )
            connection.row_factory = sqlite3.Row
            try:
                connection.execute("PRAGMA foreign_keys = ON")
                connection.execute("PRAGMA synchronous = FULL")
                if write:
                    connection.execute("BEGIN IMMEDIATE")
                _schema(connection)
                yield connection
                if write:
                    connection.commit()
            finally:
                connection.close()
        except sqlite3.Error:
            raise ValueError("Experiment registry operation failed safely.") from None

    def start_development(
        self, *, study_id: str, hypothesis: str, configuration: Mapping[str, Any]
    ) -> str:
        study = _text(study_id, "study_id", limit=256)
        hypothesis = _text(hypothesis, "hypothesis")
        config = _mapping(configuration, "configuration")
        run_id = str(uuid4())
        with self._connection(write=True) as connection:
            prior = connection.execute(
                "SELECT hypothesis FROM runs WHERE study_id = ? LIMIT 1", (study,)
            ).fetchone()
            if prior is not None and prior["hypothesis"] != hypothesis:
                raise ValueError("An existing study cannot change its hypothesis.")
            connection.execute(
                """INSERT INTO runs
                (run_id, kind, status, study_id, hypothesis, configuration, started_at)
                VALUES (?, 'development', 'running', ?, ?, ?, ?)""",
                (run_id, study, hypothesis, _encode(config), _now()),
            )
        return run_id

    def record_development_plan(
        self, run_id: str, *, manifest: Mapping[str, Any]
    ) -> None:
        proposed = _manifest(manifest)
        exposure_end = (
            date.fromisoformat(proposed["locked_test_start"]) - timedelta(days=1)
        ).isoformat()
        with self._connection(write=True) as connection:
            record = _get(connection, run_id)
            _running(record, "development")
            if record["manifest"] is not None:
                raise ValueError("The development plan has already been recorded.")
            if record["configuration"] != proposed["configuration"]:
                raise ValueError(
                    "Manifest configuration differs from the declared run."
                )
            connection.execute(
                "UPDATE runs SET manifest = ? WHERE run_id = ?",
                (_encode(proposed), run_id),
            )
            connection.execute(
                "INSERT INTO exposures(run_id, start, end) VALUES (?, ?, ?)",
                (run_id, proposed["development_start"], exposure_end),
            )

    def complete_development(
        self, run_id: str, *, manifest: Mapping[str, Any], frozen_family: str
    ) -> None:
        proposed = _manifest(manifest)
        family = _text(frozen_family, "frozen_family", limit=256)
        with self._connection(write=True) as connection:
            record = _get(connection, run_id)
            _running(record, "development")
            if record["manifest"] is None:
                raise ValueError("Development must record its plan before training.")
            _match_identity(record["manifest"], proposed)
            connection.execute(
                """UPDATE runs SET status = 'completed', manifest = ?,
                frozen_family = ?, finished_at = ? WHERE run_id = ?""",
                (_encode(proposed), family, _now(), run_id),
            )

    def validate_final_plan(
        self, development_run_id: str, *, manifest: Mapping[str, Any]
    ) -> None:
        """Check before a development rerun; this does not reserve a final period.

        Reservation must recheck atomically because another process may expose
        dates between this read-only check and the final evaluation.
        """
        proposed = _manifest(manifest)
        with self._connection() as connection:
            _final_plan(connection, development_run_id, proposed)

    def reserve_final(
        self,
        *,
        development_run_id: str,
        manifest: Mapping[str, Any],
        frozen_family: str,
    ) -> str:
        """Commit exposure before returning; no status ever releases it."""
        proposed = _manifest(manifest)
        family = _text(frozen_family, "frozen_family", limit=256)
        run_id = str(uuid4())
        with self._connection(write=True) as connection:
            development = _final_plan(connection, development_run_id, proposed)
            if development["frozen_family"] != family:
                raise ValueError(
                    "Model family differs from the frozen development choice."
                )
            start, end = proposed["locked_test_start"], proposed["locked_label_end_max"]
            self._insert_final(connection, run_id, development, proposed, family)
            connection.execute(
                "INSERT INTO exposures(run_id, start, end) VALUES (?, ?, ?)",
                (run_id, start, end),
            )
        return run_id

    @staticmethod
    def _insert_final(
        connection: sqlite3.Connection,
        run_id: str,
        development: Mapping[str, Any],
        manifest: Mapping[str, Any],
        family: str,
    ) -> None:
        connection.execute(
            """INSERT INTO runs
            (run_id, kind, status, study_id, hypothesis, configuration, manifest,
            frozen_family, development_run_id, started_at)
            VALUES (?, 'final', 'running', ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                development["study_id"],
                development["hypothesis"],
                _encode(development["configuration"]),
                _encode(manifest),
                family,
                development["run_id"],
                _now(),
            ),
        )

    def fail_run(self, run_id: str, *, error_type: str) -> None:
        error = _text(error_type, "error_type", limit=128)
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", error) is None:
            raise ValueError("error_type must be a class name, not exception details.")
        with self._connection(write=True) as connection:
            record = _get(connection, run_id)
            if record["status"] != "running":
                raise ValueError("Only a running run can be marked failed.")
            connection.execute(
                """UPDATE runs SET status = 'failed', error_type = ?, finished_at = ?
                WHERE run_id = ?""",
                (error, _now(), run_id),
            )

    def complete_final(
        self, run_id: str, *, manifest: Mapping[str, Any], acceptance: Mapping[str, Any]
    ) -> None:
        proposed = _manifest(manifest)
        result = _mapping(acceptance, "acceptance", nullable_nonfinite=True)
        with self._connection(write=True) as connection:
            record = _get(connection, run_id)
            _running(record, "final")
            _match_identity(record["manifest"], proposed)
            connection.execute(
                """UPDATE runs SET status = 'completed', manifest = ?, acceptance = ?,
                finished_at = ? WHERE run_id = ?""",
                (_encode(proposed), _encode(result), _now(), run_id),
            )

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._connection() as connection:
            return _get(connection, run_id)

    def list_runs(self) -> list[dict[str, Any]]:
        with self._connection() as connection:
            return [
                _decode(row)
                for row in connection.execute(
                    _SELECT_RUN + " ORDER BY runs.started_at, runs.run_id"
                )
            ]


__all__ = ["ExperimentRegistry"]
