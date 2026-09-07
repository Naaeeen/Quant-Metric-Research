"""Bounded acquisition of one versioned, publisher-licensed research archive."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from ._version import __version__
from .intake import _json_object, _local_file, _local_path, _write_json

DATASET_URL = "https://data.mendeley.com/datasets/ndxfrshm74/3"
SNAPSHOT_URL = "https://data.mendeley.com/public-api/datasets/ndxfrshm74/snapshot/3"
FILES_URL = (
    "https://data.mendeley.com/public-api/datasets/ndxfrshm74/files"
    "?folder_id=root&version=3"
)
_ALLOWED_HOSTS = frozenset(
    {
        "data.mendeley.com",
        "prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com",
    }
)


@dataclass(frozen=True)
class ArchiveFile:
    filename: str
    file_id: str
    sha256: str
    size: int

    @property
    def url(self) -> str:
        return (
            "https://data.mendeley.com/public-files/datasets/ndxfrshm74/files/"
            f"{self.file_id}/file_downloaded"
        )


ARCHIVE_FILES = (
    ArchiveFile(
        "sp500-1216.csv",
        "313b7297-91f9-4b9b-8df1-adefec586f9c",
        "d5695af5c26f35d0fd4016d27b4bf8097a69190d1cd894fa93d56c8eefb46fff",
        6146155,
    ),
    ArchiveFile(
        "FF3-0317.csv",
        "90681207-e931-4b35-94cf-1c8b74ca6a95",
        "72d8efb66928f54931886df0ca4b2eb3bb9611de8c663211bd01ecea48c672c7",
        118569,
    ),
)


def _approved_url(url: str) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in _ALLOWED_HOSTS
        or parsed.port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("Archive requests require an approved HTTPS destination.")


class ArchiveRedirectHandler(HTTPRedirectHandler):
    """Validate redirect targets before making any follow-up request."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _approved_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _read_remote(url: str, *, maximum_bytes: int) -> bytes:
    _approved_url(url)
    request = Request(
        url,
        headers={
            "User-Agent": f"Quant-Metric-Research/{__version__} public-research-sample",
            "Accept": "application/vnd.mendeley-public-dataset.1+json, text/csv",
        },
    )
    # No credential use, automatic retries, or rate-limit workarounds.
    with build_opener(ArchiveRedirectHandler()).open(request, timeout=30) as response:
        _approved_url(response.geturl())
        if response.status != 200:
            raise ValueError("Archive request did not return HTTP 200.")
        declared = response.headers.get("Content-Length")
        if declared is not None and int(declared) > maximum_bytes:
            raise ValueError("Archive response exceeds the declared size limit.")
        body = response.read(maximum_bytes + 1)
    if len(body) > maximum_bytes:
        raise ValueError("Archive response exceeds the declared size limit.")
    return body


def _validate_metadata(snapshot: bytes, inventory: bytes) -> None:
    record = _json_object(snapshot)
    if record.get("id") != "ndxfrshm74" or record.get("version") != 3:
        raise ValueError("Archive dataset/version metadata changed.")
    if record.get("licence", {}).get("short_name") != "CC BY 4.0":
        raise ValueError("Archive publisher license declaration changed.")
    files = json.loads(inventory)
    if not isinstance(files, list):
        raise ValueError("Archive file metadata must be a list.")
    for spec in ARCHIVE_FILES:
        matches = [item for item in files if item.get("id") == spec.file_id]
        if len(matches) != 1:
            raise ValueError("Archive file metadata is absent or ambiguous.")
        item = matches[0]
        content = item.get("content_details", {})
        if (
            item.get("filename") != spec.filename
            or content.get("sha256_hash") != spec.sha256
            or content.get("size") != spec.size
            or content.get("download_url") != spec.url
        ):
            raise ValueError("Pinned archive file metadata changed.")


def _validate_body(spec: ArchiveFile, body: bytes) -> None:
    if len(body) != spec.size:
        raise ValueError("Archive file size differs from the pinned version.")
    if hashlib.sha256(body).hexdigest() != spec.sha256:
        raise ValueError("Archive checksum differs from the pinned version.")


def _save_remote(destination: Path, name: str, url: str, *, limit: int) -> dict:
    body = _read_remote(url, maximum_bytes=limit)
    with (destination / name).open("xb") as stream:
        stream.write(body)
    return {
        "path": name,
        "url": url,
        "sha256": hashlib.sha256(body).hexdigest(),
        "byte_count": len(body),
    }


def fetch_public_archive(output_dir: str | Path) -> dict:
    """Download exactly two pinned CSVs and their public license/file records.

    The publisher's CC BY declaration supports this research demonstration;
    it is not independent verification of underlying third-party rights.
    Original files stay in the requested local directory, never in Git by default.
    """
    destination = _local_path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Archive directory already exists; choose a new one.")
    _local_path(destination.resolve())
    destination.mkdir(parents=True, exist_ok=False)
    try:
        (destination / "raw").mkdir()
        evidence = [
            _save_remote(
                destination, "dataset_record.json", SNAPSHOT_URL, limit=1024 * 1024
            ),
            _save_remote(
                destination, "file_inventory.json", FILES_URL, limit=1024 * 1024
            ),
        ]
        _validate_metadata(
            (destination / "dataset_record.json").read_bytes(),
            (destination / "file_inventory.json").read_bytes(),
        )
        files = []
        for spec in ARCHIVE_FILES:
            item = _save_remote(
                destination, f"raw/{spec.filename}", spec.url, limit=spec.size
            )
            _validate_body(spec, (destination / item["path"]).read_bytes())
            files = [*files, item]
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "package_version": __version__,
            "dataset_url": DATASET_URL,
            "dataset_doi": "10.17632/ndxfrshm74.3",
            "dataset_version": 3,
            "acquired_at": datetime.now(UTC).isoformat(),
            "publisher_declared_license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "attribution": (
                "Pun, Chi Seng (2018), Low- and High-Dimensional Asset Prices Data, "
                "Mendeley Data, V3, doi:10.17632/ndxfrshm74.3."
            ),
            "usage_basis": "publisher_declared_research_archive_license",
            "independent_usage_rights_verified": False,
            "empirical_data_provenance_verified": False,
            "stage4_eligible": False,
            "limitations": [
                "Publisher names Yahoo Finance and Ken French as upstream sources.",
                "Third-party rights remain unverified; no raw publishing.",
                "The 2017 constituent snapshot is not point-in-time membership.",
                "Revisions, adjustment events and delistings are unverified.",
            ],
            "source_files": [*evidence, *files],
        }
        pending = destination / "archive_manifest.pending.json"
        _write_json(manifest, pending)
        pending.rename(destination / "archive_manifest.json")
        return manifest
    except Exception as error:
        _write_json(
            {"status": "failed", "error_type": type(error).__name__},
            destination / "archive_failure.json",
        )
        raise


def verify_public_archive(archive_dir: str | Path) -> dict:
    """Verify an existing local acquisition before offline normalization."""
    directory = _local_path(archive_dir)
    manifest = _json_object(
        _local_file(directory / "archive_manifest.json").read_bytes()
    )
    if manifest.get("status") != "complete":
        raise ValueError("Archive acquisition is incomplete.")
    expected = {
        "dataset_record.json": SNAPSHOT_URL,
        "file_inventory.json": FILES_URL,
        **{f"raw/{spec.filename}": spec.url for spec in ARCHIVE_FILES},
    }
    items = manifest.get("source_files", [])
    if len(items) != len(expected) or {item.get("path") for item in items} != set(
        expected
    ):
        raise ValueError("Archive source inventory changed.")
    for item in items:
        if item.get("url") != expected[item["path"]]:
            raise ValueError("Archive source URL changed.")
        body = _local_file(directory / item["path"]).read_bytes()
        if len(body) != item.get("byte_count") or hashlib.sha256(
            body
        ).hexdigest() != item.get("sha256"):
            raise ValueError("Archive local file checksum or size changed.")
    _validate_metadata(
        (directory / "dataset_record.json").read_bytes(),
        (directory / "file_inventory.json").read_bytes(),
    )
    for spec in ARCHIVE_FILES:
        _validate_body(spec, (directory / "raw" / spec.filename).read_bytes())
    return manifest
