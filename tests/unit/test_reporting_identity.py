from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from quant_metric_research import benchmark_reporting


def test_shared_source_identity_matches_existing_byte_algorithm():
    expected = sha256()
    package = Path(benchmark_reporting.__file__).resolve().parent
    for path in sorted(package.glob("*.py")):
        expected.update(path.name.encode("utf-8"))
        expected.update(b"\0")
        expected.update(path.read_bytes())
        expected.update(b"\0")

    assert benchmark_reporting._source_fingerprint() == expected.hexdigest()


def test_source_identity_keeps_order_name_and_top_level_python_scope(
    tmp_path, monkeypatch
):
    (tmp_path / "z.py").write_bytes(b"last\n")
    (tmp_path / "a.py").write_bytes(b"first\n")
    (tmp_path / "notes.md").write_bytes(b"not source")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "ignored.py").write_bytes(b"outside existing scope")
    monkeypatch.setattr(
        benchmark_reporting, "__file__", str(tmp_path / "benchmark_reporting.py")
    )

    expected = sha256(b"a.py\0first\n\0z.py\0last\n\0").hexdigest()
    assert benchmark_reporting._source_fingerprint() == expected

    (tmp_path / "a.py").write_bytes(b"changed\n")
    assert benchmark_reporting._source_fingerprint() != expected
