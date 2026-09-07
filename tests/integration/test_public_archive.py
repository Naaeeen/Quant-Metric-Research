"""Offline transport fixtures for the version-pinned public archive."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from io import BytesIO
from types import SimpleNamespace
from urllib.request import Request

import pytest

import quant_metric_research.public_archive as archive


@pytest.fixture()
def archive_transport(monkeypatch):
    bodies = (b"Date,A\n2012-01-03,10\n", b"Date,Mkt-RF,RF\n1/3/2012,1,0\n")
    specs = tuple(
        replace(spec, sha256=hashlib.sha256(body).hexdigest(), size=len(body))
        for spec, body in zip(archive.ARCHIVE_FILES, bodies, strict=True)
    )
    monkeypatch.setattr(archive, "ARCHIVE_FILES", specs)
    metadata = {
        "id": "ndxfrshm74",
        "version": 3,
        "name": "Low- and High-Dimensional Asset Prices Data",
        "licence": {"short_name": "CC BY 4.0"},
    }
    files = [
        {
            "id": spec.file_id,
            "filename": spec.filename,
            "content_details": {
                "sha256_hash": spec.sha256,
                "size": spec.size,
                "download_url": spec.url,
            },
        }
        for spec in specs
    ]
    payloads = {
        archive.SNAPSHOT_URL: json.dumps(metadata).encode(),
        archive.FILES_URL: json.dumps(files).encode(),
        **{spec.url: body for spec, body in zip(specs, bodies, strict=True)},
    }
    calls = []

    def remote(url, *, maximum_bytes):
        calls.append(url)
        assert maximum_bytes > 0
        return payloads[url]

    monkeypatch.setattr(archive, "_read_remote", remote)
    return payloads, calls


def test_fetch_retains_source_bytes_and_license_evidence(tmp_path, archive_transport):
    payloads, calls = archive_transport
    output = tmp_path / "archive"
    manifest = archive.fetch_public_archive(output)
    assert len(calls) == 4
    assert manifest["status"] == "complete"
    assert manifest["dataset_doi"] == "10.17632/ndxfrshm74.3"
    assert manifest["publisher_declared_license"] == "CC BY 4.0"
    assert manifest["independent_usage_rights_verified"] is False
    assert manifest["empirical_data_provenance_verified"] is False
    assert manifest["stage4_eligible"] is False
    assert manifest["acquired_at"]
    for item in manifest["source_files"]:
        assert (output / item["path"]).read_bytes() == payloads[item["url"]]
    assert archive.verify_public_archive(output) == manifest


def test_existing_destination_refused_before_network(tmp_path, archive_transport):
    _, calls = archive_transport
    output = tmp_path / "archive"
    output.mkdir()
    (output / "keep").write_text("unchanged")
    with pytest.raises(FileExistsError):
        archive.fetch_public_archive(output)
    assert calls == []
    assert (output / "keep").read_text() == "unchanged"


def test_changed_download_has_no_completion_marker(tmp_path, archive_transport):
    payloads, _ = archive_transport
    payloads[archive.ARCHIVE_FILES[0].url] = b"changed"
    output = tmp_path / "archive"
    with pytest.raises(ValueError, match="checksum|size"):
        archive.fetch_public_archive(output)
    assert not (output / "archive_manifest.json").exists()
    assert (
        json.loads((output / "archive_failure.json").read_text())["status"] == "failed"
    )
    assert (
        output / "raw" / archive.ARCHIVE_FILES[0].filename
    ).read_bytes() == b"changed"


def test_changed_publisher_metadata_fails_before_price_download(
    tmp_path, archive_transport
):
    payloads, calls = archive_transport
    changed = json.loads(payloads[archive.FILES_URL])
    changed[0]["content_details"]["sha256_hash"] = "0" * 64
    payloads[archive.FILES_URL] = json.dumps(changed).encode()
    with pytest.raises(ValueError, match="metadata"):
        archive.fetch_public_archive(tmp_path / "archive")
    assert len(calls) == 2


def test_unsupported_license_fails_before_price_download(tmp_path, archive_transport):
    payloads, calls = archive_transport
    record = json.loads(payloads[archive.SNAPSHOT_URL])
    record["licence"]["short_name"] = "All rights reserved"
    payloads[archive.SNAPSHOT_URL] = json.dumps(record).encode()
    with pytest.raises(ValueError, match="license"):
        archive.fetch_public_archive(tmp_path / "archive")
    assert len(calls) == 2


def test_verify_detects_local_tampering_without_network(tmp_path, archive_transport):
    _, calls = archive_transport
    output = tmp_path / "archive"
    archive.fetch_public_archive(output)
    (output / "raw" / archive.ARCHIVE_FILES[0].filename).write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum|size"):
        archive.verify_public_archive(output)
    assert len(calls) == 4


@pytest.mark.parametrize(
    "target",
    [
        "http://data.mendeley.com/file",
        "https://example.com/file",
        "file:///etc/passwd",
        "https://user:pass@data.mendeley.com/file",
    ],
)
def test_redirect_rejects_unapproved_destination_before_access(target):
    redirect = archive.ArchiveRedirectHandler()
    with pytest.raises(ValueError, match="HTTPS"):
        redirect.redirect_request(
            Request(archive.SNAPSHOT_URL), None, 302, "", {}, target
        )


def _response(monkeypatch, *, body=b"data", declared=None, status=200, url=None):
    stream = BytesIO(body)
    stream.status = status
    stream.headers = {} if declared is None else {"Content-Length": declared}
    stream.geturl = lambda: url or archive.SNAPSHOT_URL
    calls = []

    def open_request(request, timeout):
        calls.append((request, timeout))
        return stream

    monkeypatch.setattr(
        archive, "build_opener", lambda handler: SimpleNamespace(open=open_request)
    )
    return calls


@pytest.mark.parametrize("declared", [None, "4"])
def test_transport_preserves_bytes_and_uses_bounded_honest_request(
    monkeypatch, declared
):
    calls = _response(monkeypatch, declared=declared)
    assert archive._read_remote(archive.SNAPSHOT_URL, maximum_bytes=4) == b"data"
    request, timeout = calls[0]
    assert timeout == 30
    assert "Quant-Metric-Research/" in request.get_header("User-agent")
    assert request.get_header("Authorization") is None


@pytest.mark.parametrize(
    "response,match",
    [
        ({"declared": "5"}, "size limit"),
        ({"body": b"extra"}, "size limit"),
        ({"declared": "invalid"}, "invalid literal"),
        ({"status": 206}, "HTTP 200"),
        ({"url": "https://example.com"}, "HTTPS"),
    ],
)
def test_transport_rejects_oversize_partial_or_unapproved_response(
    monkeypatch, response, match
):
    _response(monkeypatch, **response)
    with pytest.raises(ValueError, match=match):
        archive._read_remote(archive.SNAPSHOT_URL, maximum_bytes=4)


def test_transport_rejects_bad_origin_before_network(monkeypatch):
    calls = _response(monkeypatch)
    with pytest.raises(ValueError, match="HTTPS"):
        archive._read_remote("https://example.com", maximum_bytes=4)
    assert calls == []
