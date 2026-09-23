"""Snapshot and audit locally supplied Yahoo-format exports; never fetch data."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ._version import __version__
from .config import PanelConfig
from .contracts import validate_memberships, validate_prices
from .input_audit import audit_inputs
from .io import read_as_of_dates, read_table
from .yahoo_import import normalize_yahoo_export


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate JSON keys are not allowed.")
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError("Non-finite JSON constants are not allowed.")


def _json_object(payload: bytes) -> dict[str, Any]:
    value = json.loads(
        payload.decode("utf-8-sig"),
        object_pairs_hook=_unique_object,
        parse_constant=_invalid_constant,
    )
    if not isinstance(value, dict):
        raise ValueError("Input JSON must contain an object.")
    return value


def _local_path(value: str | Path) -> Path:
    text = str(value)
    # No URLs, network shares, or Windows device paths. A drive-qualified local
    # path is allowed; file:// and provider URLs are deliberately not adapters.
    scheme = re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", text)
    drive_path = re.match(r"^[a-zA-Z]:[\\/]", text)
    if (
        not text.strip()
        or text.replace("\\", "/").startswith("//")
        or (scheme and not drive_path)
    ):
        raise ValueError("Only local filesystem paths are supported.")
    return Path(value)


def _local_file(value: str | Path) -> Path:
    source = _local_path(value).resolve(strict=True)
    _local_path(source)  # Reject a symlink that resolves onto a network share.
    if not source.is_file():
        raise ValueError("Each source must be a regular local file.")
    return source


def _table_suffix(path: Path) -> str:
    for suffix in (".csv.gz", ".csv", ".parquet", ".pq"):
        if path.name.lower().endswith(suffix):
            return suffix
    raise ValueError("Input tables must be CSV, CSV.GZ, Parquet, or PQ files.")


def _export_paths(payload: bytes, *, parent: Path) -> dict[str, Path]:
    mapping = _json_object(payload)
    if not mapping:
        raise ValueError("Exports JSON must contain at least one symbol.")
    symbols = [symbol.strip().upper() for symbol in mapping]
    if any(not symbol for symbol in symbols) or len(set(symbols)) != len(symbols):
        raise ValueError(
            "Export symbols must be nonempty and unique after normalization."
        )
    if any(not isinstance(value, str) for value in mapping.values()):
        raise ValueError("Each export path must be a local path string.")
    paths = [_local_path(value) for value in mapping.values()]
    sources = [_local_file(parent / path) for path in paths]
    for source in sources:
        _table_suffix(source)
    return dict(zip(symbols, sources, strict=True))


def _snapshot(
    source: Path,
    destination: Path,
    name: str,
    *,
    role: str,
    symbol: str | None = None,
    payload: bytes | None = None,
) -> dict[str, Any]:
    original = source.read_bytes() if payload is None else payload
    relative = f"raw/{name}"
    with (destination / relative).open("xb") as stream:
        stream.write(original)
    return {
        "path": relative,
        "role": role,
        "sha256": hashlib.sha256(original).hexdigest(),
        "byte_count": len(original),
        **({"symbol": symbol} if symbol is not None else {}),
    }


def _write_json(value: dict[str, Any], path: Path) -> None:
    text = json.dumps(value, allow_nan=False, indent=2, sort_keys=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text + "\n")


def _verify_snapshots(destination: Path, source_files: list[dict[str, Any]]) -> None:
    for item in source_files:
        digest = hashlib.sha256((destination / item["path"]).read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise ValueError("Raw snapshot changed during import; completion refused.")


def _save_inputs(
    destination: Path,
    *,
    mapping_path: Path,
    mapping_bytes: bytes,
    exports: dict[str, Path],
    membership_path: Path,
    dates_path: Path,
    config_path: Path,
) -> list[dict[str, Any]]:
    controls = (
        (
            membership_path,
            "memberships" + _table_suffix(membership_path),
            "memberships",
        ),
        (dates_path, "as-of-dates" + _table_suffix(dates_path), "as_of_dates"),
        (config_path, "config.json", "config"),
    )
    mapping = _snapshot(
        mapping_path, destination, "exports.json", role="exports", payload=mapping_bytes
    )
    control_files = [
        _snapshot(source, destination, name, role=role)
        for source, name, role in controls
    ]
    price_files = [
        _snapshot(
            source,
            destination,
            f"price-{index:04d}" + _table_suffix(source),
            role="prices",
            symbol=symbol,
        )
        for index, (symbol, source) in enumerate(sorted(exports.items()))
    ]
    return [mapping, *control_files, *price_files]


def _normalize_snapshot(
    destination: Path, source_files: list[dict[str, Any]]
) -> dict[str, Any]:
    controls = {
        item["role"]: destination / item["path"]
        for item in source_files
        if item["role"] != "prices"
    }
    config = PanelConfig(**_json_object(controls["config"].read_bytes()))
    members = validate_memberships(read_table(controls["memberships"]))
    dates = read_as_of_dates(controls["as_of_dates"])
    exports = {
        item["symbol"]: normalize_yahoo_export(
            read_table(destination / item["path"]), symbol=item["symbol"]
        )
        for item in source_files
        if item["role"] == "prices"
    }
    if config.benchmark_symbol not in exports:
        raise ValueError("Benchmark export is required.")
    prices = validate_prices(pd.concat([item.prices for item in exports.values()]))
    missing = pd.concat(
        [item.missing_prices for item in exports.values()], ignore_index=True
    ).sort_values(["date", "symbol"], kind="stable")
    audit = audit_inputs(prices, members, as_of_dates=dates, config=config)
    prices.to_parquet(destination / "prices.parquet", index=False)
    members.to_parquet(destination / "memberships.parquet", index=False)
    pd.DataFrame({"as_of_date": dates}).to_csv(
        destination / "as_of_dates.csv", index=False
    )
    missing.to_csv(destination / "missing_prices.csv", index=False)
    _write_json(asdict(config), destination / "config.json")
    _write_json(audit, destination / "input_audit.json")
    names = (
        "prices.parquet",
        "memberships.parquet",
        "as_of_dates.csv",
        "missing_prices.csv",
        "config.json",
        "input_audit.json",
    )
    return {
        "empty_price_exports": sorted(
            symbol for symbol, result in exports.items() if result.prices.empty
        ),
        "missing_member_exports": sorted(set(members["symbol"]) - set(exports)),
        "output_fingerprints": {
            name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
            for name in names
        },
    }


def import_yahoo_files(
    exports_path: str | Path,
    *,
    memberships_path: str | Path,
    as_of_dates_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Import supplied files, not provider data rights or historical provenance.

    Only a new directory is accepted. Raw files are snapshotted before parsing;
    on failure they remain for diagnosis. A complete manifest is written last.
    CSV blank/default pandas NA tokens become explicitly reported missing prices.
    Local input files must be trusted: this is not a hostile-filesystem sandbox.
    """
    destination = _local_path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            "Output directory already exists; choose a new directory."
        )
    _local_path(destination.resolve())
    mapping_path = _local_file(exports_path)
    mapping_bytes = mapping_path.read_bytes()
    exports = _export_paths(mapping_bytes, parent=mapping_path.parent)
    membership_path = _local_file(memberships_path)
    dates_path = _local_file(as_of_dates_path)
    config_source = _local_file(config_path)
    _table_suffix(membership_path)
    _table_suffix(dates_path)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        (destination / "raw").mkdir()
        source_files = _save_inputs(
            destination,
            mapping_path=mapping_path,
            mapping_bytes=mapping_bytes,
            exports=exports,
            membership_path=membership_path,
            dates_path=dates_path,
            config_path=config_source,
        )
        outputs = _normalize_snapshot(destination, source_files)
        _verify_snapshots(destination, source_files)
        manifest = {
            "schema_version": 1,
            "package_version": __version__,
            "status": "complete",
            "imported_at": datetime.now(UTC).isoformat(),
            "acquired_at": None,
            "source_format": "yahoo_flat_daily_export",
            "claim_scope": "unverified_local_price_import",
            "network_accessed": False,
            "no_outcomes_computed": True,
            "empirical_data_provenance_verified": False,
            "adjustment_policy_verified": False,
            "usage_rights_verified": False,
            "stage4_eligible": False,
            "source_files": source_files,
            **outputs,
        }
        # A partial write is not a completion marker. Rename only within the new,
        # importer-owned directory after the strict JSON has been closed.
        pending = destination / "intake_manifest.pending.json"
        _write_json(manifest, pending)
        pending.rename(destination / "intake_manifest.json")
        return manifest
    except Exception as error:
        _write_json(
            {"status": "failed", "error_type": type(error).__name__},
            destination / "intake_failure.json",
        )
        raise
