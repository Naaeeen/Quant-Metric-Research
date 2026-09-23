from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from quant_metric_research.intake import import_yahoo_files


@pytest.fixture()
def intake_files(tmp_path, market_fixture):
    prices, memberships, dates = market_fixture
    source = tmp_path / "inputs"
    source.mkdir()
    exports = {}
    for symbol, group in prices.groupby("symbol"):
        frame = group.rename(columns={"date": "Date", "adjusted_close": "Adj Close"})
        if symbol == "BBB":
            frame = frame.assign(**{"Adj Close": float("nan")})
        frame[["Date", "Adj Close"]].to_csv(source / f"{symbol}.csv", index=False)
        exports[symbol] = f"{symbol}.csv"
    (source / "exports.json").write_text(json.dumps(exports), encoding="utf-8")
    memberships.to_csv(source / "memberships.csv", index=False)
    pd.DataFrame({"as_of_date": dates[8:11]}).to_csv(source / "dates.csv", index=False)
    config = {
        "dataset_version": "invented-yahoo-test",
        "universe_id": "TEST",
        "benchmark_symbol": "BENCH",
        "lookback_sessions": 4,
        "min_observations": 2,
        "target_horizon_sessions": 2,
    }
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return {
        "exports_path": source / "exports.json",
        "memberships_path": source / "memberships.csv",
        "as_of_dates_path": source / "dates.csv",
        "config_path": source / "config.json",
        "output_dir": tmp_path / "intake",
    }


def test_import_snapshots_exact_bytes_and_retains_missing_security(intake_files):
    manifest = import_yahoo_files(**intake_files)
    output = intake_files["output_dir"]
    assert manifest["status"] == "complete"
    assert manifest["network_accessed"] is False
    assert manifest["empirical_data_provenance_verified"] is False
    assert manifest["stage4_eligible"] is False
    assert manifest["acquired_at"] is None
    assert manifest["claim_scope"] == "unverified_local_price_import"
    for item in manifest["source_files"]:
        payload = (output / item["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == item["sha256"]
    first = next(
        item for item in manifest["source_files"] if item.get("symbol") == "AAA"
    )
    assert (output / first["path"]).read_bytes() == (
        intake_files["exports_path"].parent / "AAA.csv"
    ).read_bytes()
    prices = pd.read_parquet(output / "prices.parquet")
    members = pd.read_parquet(output / "memberships.parquet")
    assert set(prices.symbol) == {"AAA", "BENCH"}
    assert set(members.symbol) == {"AAA", "BBB"}
    missing = pd.read_csv(output / "missing_prices.csv")
    assert set(missing.symbol) == {"BBB"}
    audit = json.loads((output / "input_audit.json").read_text())
    assert audit["summary"]["members_without_any_prices"] == ["BBB"]
    assert audit["summary"]["active_member_dates"] == 6
    assert manifest["empty_price_exports"] == ["BBB"]
    assert manifest["missing_member_exports"] == []
    assert json.loads((output / "intake_manifest.json").read_text()) == manifest
    for name, digest in manifest["output_fingerprints"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest


def test_import_checks_existing_destination_before_reading_inputs(intake_files):
    output = intake_files["output_dir"]
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep")
    intake_files["exports_path"].unlink()
    with pytest.raises(FileExistsError):
        import_yahoo_files(**intake_files)
    assert marker.read_text() == "keep"


@pytest.mark.parametrize(
    "mapping",
    [
        "{}",
        '{"AAA":"AAA.csv","aaa":"BBB.csv"}',
        '{"AAA":"AAA.csv","AAA":"BBB.csv"}',
        '{"AAA":true}',
        '{"AAA":"https://example.com/price.csv"}',
        "[]",
    ],
)
def test_import_rejects_ambiguous_or_nonlocal_maps(intake_files, mapping):
    intake_files["exports_path"].write_text(mapping, encoding="utf-8")
    with pytest.raises(ValueError):
        import_yahoo_files(**intake_files)
    assert not intake_files["output_dir"].exists()


def test_import_failure_preserves_input_snapshot_without_success_manifest(intake_files):
    source = intake_files["exports_path"].parent
    (source / "AAA.csv").write_text("Date,Close\n2025-01-02,10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Adj Close"):
        import_yahoo_files(**intake_files)
    output = intake_files["output_dir"]
    assert list((output / "raw").iterdir())
    assert not (output / "intake_manifest.json").exists()
    failure = json.loads((output / "intake_failure.json").read_text())
    assert failure["status"] == "failed"
    assert "error_type" in failure
    assert "error_message" not in failure
    with pytest.raises(FileExistsError):
        import_yahoo_files(**intake_files)


def test_import_records_absent_export_without_dropping_memberships(intake_files):
    source = intake_files["exports_path"]
    source.write_text('{"AAA":"AAA.csv","BENCH":"BENCH.csv"}', encoding="utf-8")
    result = import_yahoo_files(**intake_files)
    assert result["missing_member_exports"] == ["BBB"]
    assert set(
        pd.read_parquet(intake_files["output_dir"] / "memberships.parquet").symbol
    ) == {"AAA", "BBB"}


def test_import_requires_valid_benchmark_before_success(intake_files):
    source = intake_files["exports_path"]
    source.write_text('{"AAA":"AAA.csv","BBB":"BBB.csv"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Benchmark"):
        import_yahoo_files(**intake_files)
    assert not (intake_files["output_dir"] / "intake_manifest.json").exists()


@pytest.mark.parametrize("prefix", ["\\/", "/\\"])
@pytest.mark.parametrize(
    "argument",
    [
        "exports_path",
        "memberships_path",
        "as_of_dates_path",
        "config_path",
        "output_dir",
    ],
)
def test_import_rejects_mixed_network_paths_before_filesystem_probes(
    intake_files, monkeypatch, prefix, argument
):
    original_exists = Path.exists
    original_resolve = Path.resolve

    def forbid_network(path):
        if str(path).replace("\\", "/").startswith("//"):
            raise AssertionError("Network filesystem must not be probed.")

    def checked_exists(path):
        forbid_network(path)
        return original_exists(path)

    def checked_resolve(path, *args, **kwargs):
        forbid_network(path)
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "exists", checked_exists)
    monkeypatch.setattr(Path, "resolve", checked_resolve)
    with pytest.raises(ValueError, match="local"):
        import_yahoo_files(
            **{**intake_files, argument: prefix + "server/share/export.csv"}
        )


def test_import_uses_saved_bytes_not_a_second_read_of_live_source(
    intake_files, monkeypatch
):
    import quant_metric_research.intake as intake

    original = intake.read_table
    source = intake_files["exports_path"].parent / "AAA.csv"
    previous = source.read_bytes()

    def tamper_after_snapshot(path):
        if Path(path).name.startswith("price-"):
            source.write_text("Date,Close\nbroken,1\n", encoding="utf-8")
        return original(path)

    monkeypatch.setattr(intake, "read_table", tamper_after_snapshot)
    result = import_yahoo_files(**intake_files)
    aaa = next(item for item in result["source_files"] if item.get("symbol") == "AAA")
    assert aaa["sha256"] == hashlib.sha256(previous).hexdigest()
    assert result["status"] == "complete"


@pytest.mark.parametrize("suffix", [".csv.gz", ".parquet", ".pq"])
def test_import_supported_export_formats(intake_files, suffix):
    source = intake_files["exports_path"].parent
    frame = pd.read_csv(source / "AAA.csv")
    path = source / ("AAA" + suffix)
    if suffix == ".csv.gz":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)
    mapping = json.loads(intake_files["exports_path"].read_text())
    intake_files["exports_path"].write_text(
        json.dumps({**mapping, "AAA": path.name}), encoding="utf-8"
    )
    result = import_yahoo_files(**intake_files)
    saved = next(item for item in result["source_files"] if item.get("symbol") == "AAA")
    assert (
        intake_files["output_dir"] / saved["path"]
    ).read_bytes() == path.read_bytes()
    assert result["status"] == "complete"


@pytest.mark.parametrize("bad_value", ["oops", "-2", "inf", "True"])
def test_import_rejects_present_invalid_price_without_coercion(intake_files, bad_value):
    path = intake_files["exports_path"].parent / "AAA.csv"
    path.write_text(f"Date,Adj Close\n2025-01-02,{bad_value}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="positive"):
        import_yahoo_files(**intake_files)
    assert not (intake_files["output_dir"] / "intake_manifest.json").exists()


@pytest.mark.parametrize("token", ["", "null", "NaN", "NA"])
def test_import_csv_na_tokens_are_visible_missing_prices(intake_files, token):
    path = intake_files["exports_path"].parent / "AAA.csv"
    path.write_text(f"Date,Adj Close\n2025-01-02,{token}\n", encoding="utf-8")
    manifest = import_yahoo_files(**intake_files)
    assert manifest["empty_price_exports"] == ["AAA", "BBB"]
    missing = pd.read_csv(intake_files["output_dir"] / "missing_prices.csv")
    assert len(missing.loc[missing.symbol == "AAA"]) == 1


@pytest.mark.parametrize(
    "mapping",
    [
        '{"AAA":"file:///data.csv"}',
        '{"AAA":"//server/share/AAA.csv"}',
        '{"AAA":"\\\\\\\\server\\\\share\\\\AAA.csv"}',
        '{"AAA":"AAA.csv","BBB":NaN}',
        '{"   ":"AAA.csv"}',
    ],
)
def test_import_rejects_unsupported_paths_and_symbols_before_output(
    intake_files, mapping
):
    intake_files["exports_path"].write_text(mapping, encoding="utf-8")
    with pytest.raises(ValueError):
        import_yahoo_files(**intake_files)
    assert not intake_files["output_dir"].exists()


def test_import_rejects_duplicate_csv_header(intake_files):
    path = intake_files["exports_path"].parent / "AAA.csv"
    path.write_text("Date,Adj Close,Adj Close\n2025-01-02,10,20\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        import_yahoo_files(**intake_files)


def test_import_all_missing_benchmark_cannot_complete(intake_files):
    path = intake_files["exports_path"].parent / "BENCH.csv"
    path.write_text("Date,Adj Close\n2025-01-02,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Benchmark"):
        import_yahoo_files(**intake_files)
    assert not (intake_files["output_dir"] / "intake_manifest.json").exists()


def test_import_header_only_export_remains_in_inventory(intake_files):
    path = intake_files["exports_path"].parent / "BBB.csv"
    path.write_text("Date,Adj Close\n", encoding="utf-8")
    manifest = import_yahoo_files(**intake_files)
    assert manifest["empty_price_exports"] == ["BBB"]
    assert manifest["missing_member_exports"] == []
    audit = json.loads((intake_files["output_dir"] / "input_audit.json").read_text())
    assert audit["summary"]["members_without_any_prices"] == ["BBB"]


def test_import_partial_completion_write_never_publishes_success(
    intake_files, monkeypatch
):
    import quant_metric_research.intake as intake

    write_json = intake._write_json

    def interrupt_manifest(value, path):
        if path.name == "intake_manifest.pending.json":
            path.write_text('{"status":', encoding="utf-8")
            raise OSError("private-path-or-provider-error")
        return write_json(value, path)

    monkeypatch.setattr(intake, "_write_json", interrupt_manifest)
    with pytest.raises(OSError):
        import_yahoo_files(**intake_files)
    output = intake_files["output_dir"]
    assert not (output / "intake_manifest.json").exists()
    failure = (output / "intake_failure.json").read_text()
    assert json.loads(failure) == {"status": "failed", "error_type": "OSError"}
    assert "private-path" not in failure


def test_import_detects_saved_snapshot_change_before_completion(
    intake_files, monkeypatch
):
    import quant_metric_research.intake as intake

    read_table = intake.read_table

    def change_saved_file(path):
        if Path(path).name == "price-0000.csv":
            frame = pd.read_csv(path)
            frame.assign(**{"Adj Close": frame["Adj Close"] * 2}).to_csv(
                path, index=False
            )
        return read_table(path)

    monkeypatch.setattr(intake, "read_table", change_saved_file)
    with pytest.raises(ValueError, match="snapshot"):
        import_yahoo_files(**intake_files)
    assert not (intake_files["output_dir"] / "intake_manifest.json").exists()
