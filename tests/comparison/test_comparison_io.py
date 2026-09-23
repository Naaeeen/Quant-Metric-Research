"""Persist saved descriptive reports without recomputation or overwriting evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_metric_research.benchmark_comparison import compare_feature_bundles

TABLES = ("daily_metrics", "fold_metrics", "summary", "daily_deltas", "delta_summary")


def _module():
    return import_module("quant_metric_research.comparison_io")


@pytest.fixture
def comparison(comparison_runs):
    return compare_feature_bundles(*comparison_runs, model_family="ridge")


def _changed_table(comparison, name, frame):
    return replace(comparison, **{f"_{name}": frame})


def test_writer_preserves_tables_and_reports_exact_written_bytes(comparison, tmp_path):
    result = _module().write_feature_bundle_comparison(comparison, tmp_path / "out")
    assert isinstance(result, _module().ComparisonArtifacts)
    assert set(result.files) == {*TABLES, "comparison_manifest"}
    manifest = json.loads(result.files["comparison_manifest"].read_text("utf-8"))
    assert manifest["artifact_format_version"] == "1"
    assert manifest["status"] == "completed"
    assert manifest["report_metadata"]["comparison_definition_version"] == "1"
    assert manifest["report_metadata"]["authenticity_verified"] is False
    assert manifest["writer_package_version"]
    for name in TABLES:
        path = result.files[name]
        payload = path.read_bytes()
        assert b"\r\n" not in payload
        original = getattr(comparison, name)
        actual = pd.read_csv(
            path, parse_dates=["as_of_date"] if "as_of_date" in original else []
        )
        pd.testing.assert_frame_equal(actual, original, check_dtype=False)
        item = manifest["tables"][name]
        assert item == {
            "filename": f"{name}.csv",
            "row_count": len(original),
            "columns": list(original.columns),
            "byte_count": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    assert not (result.output_dir / "comparison_manifest.pending.json").exists()


def test_writer_preserves_accepted_column_and_row_order(comparison, tmp_path):
    frames = {f"_{name}": getattr(comparison, name).iloc[::-1, ::-1] for name in TABLES}
    changed = replace(comparison, **frames)
    result = _module().write_feature_bundle_comparison(changed, tmp_path / "out")
    for name in TABLES:
        original = getattr(changed, name).reset_index(drop=True)
        actual = pd.read_csv(
            result.files[name],
            parse_dates=["as_of_date"] if "as_of_date" in original else [],
        )
        pd.testing.assert_frame_equal(actual, original, check_dtype=False)


def test_missing_pairs_and_negative_numbers_are_not_rewritten(comparison, tmp_path):
    daily = comparison.daily_deltas.assign(rank_ic_delta=np.nan, spread_delta=-0.25)
    summary = comparison.delta_summary.assign(
        paired_rank_ic_date_count=0,
        paired_rank_ic_date_coverage=0.0,
        mean_paired_rank_ic_delta=np.nan,
    )
    changed = replace(comparison, _daily_deltas=daily, _delta_summary=summary)
    result = _module().write_feature_bundle_comparison(changed, tmp_path / "out")
    actual = pd.read_csv(result.files["daily_deltas"])
    assert actual["rank_ic_delta"].isna().all()
    assert actual["spread_delta"].eq(-0.25).all()
    actual_summary = pd.read_csv(result.files["delta_summary"])
    assert actual_summary["mean_paired_rank_ic_delta"].isna().all()
    assert actual_summary["paired_rank_ic_date_count"].eq(0).all()


def test_writer_does_not_mutate_inputs_and_return_metadata_is_frozen(
    comparison, tmp_path
):
    before = {name: getattr(comparison, name) for name in TABLES}
    result = _module().write_feature_bundle_comparison(comparison, tmp_path / "out")
    for name in TABLES:
        pd.testing.assert_frame_equal(before[name], getattr(comparison, name))
    with pytest.raises(TypeError):
        result.files["summary"] = Path("changed")
    with pytest.raises(TypeError):
        result.manifest["report_metadata"]["metric_settings"]["quantiles"] = 99
    with pytest.raises(FrozenInstanceError):
        result.output_dir = Path("changed")


def test_artifact_constructor_defensively_owns_nested_metadata(tmp_path):
    files = {"summary": tmp_path / "summary.csv"}
    manifest = {"nested": {"items": [1, 2]}}
    result = _module().ComparisonArtifacts(tmp_path, files, manifest)
    files.clear()
    manifest["nested"]["items"].append(3)
    assert result.files["summary"].name == "summary.csv"
    assert result.manifest["nested"]["items"] == (1, 2)


@pytest.mark.parametrize(
    "field,value",
    [
        ("comparison_definition_version", "2"),
        ("comparison_definition_version", 1),
        ("claim_scope", "final"),
        ("caller_owned_inputs", False),
        ("caller_owned_inputs", 1),
        ("final_outcomes_evaluated", True),
        ("inference_reported", True),
        ("authenticity_verified", True),
        ("history_verified", True),
        ("history_verified", 0),
    ],
)
def test_wrong_definition_or_scope_fails_before_any_creation(
    comparison, tmp_path, field, value
):
    changed = replace(comparison, metadata={**comparison.metadata, field: value})
    with pytest.raises((TypeError, ValueError)):
        _module().write_feature_bundle_comparison(changed, tmp_path / "parent" / "out")
    assert not (tmp_path / "parent").exists()


@pytest.mark.parametrize("name", TABLES)
@pytest.mark.parametrize("change", ["extra", "missing", "duplicate", "nonstring"])
def test_exact_schema_rejected_before_any_creation(comparison, tmp_path, name, change):
    frame = getattr(comparison, name)
    if change == "extra":
        frame["p_value"] = 0.01
    elif change == "missing":
        frame = frame.iloc[:, 1:]
    elif change == "duplicate":
        frame.columns = [frame.columns[1], *frame.columns[1:]]
    else:
        frame.columns = [7, *frame.columns[1:]]
    with pytest.raises((TypeError, ValueError)):
        _module().write_feature_bundle_comparison(
            _changed_table(comparison, name, frame), tmp_path / "out"
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "value",
    [
        "=1+1",
        "+1",
        "-2",
        "@SUM(A1)",
        "\t=1",
        "0.25",
        "NaN",
        "",
        np.inf,
        -np.inf,
        True,
        1j,
        pd.Timestamp("2025-01-01"),
    ],
)
def test_numeric_cells_reject_text_and_unsupported_scalars(comparison, tmp_path, value):
    frame = comparison.daily_deltas.astype({"spread_delta": object})
    frame.at[0, "spread_delta"] = value
    with pytest.raises((TypeError, ValueError)):
        _module().write_feature_bundle_comparison(
            _changed_table(comparison, "daily_deltas", frame), tmp_path / "out"
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "column,value",
    [
        ("phase", "locked_test"),
        ("model", "=1+1"),
        ("evaluation_scope", "other"),
        ("phase", None),
    ],
)
def test_text_cells_require_fixed_enums(comparison, tmp_path, column, value):
    frame = comparison.daily_metrics.astype({column: object})
    frame.at[0, column] = value
    with pytest.raises((TypeError, ValueError)):
        _module().write_feature_bundle_comparison(
            _changed_table(comparison, "daily_metrics", frame), tmp_path / "out"
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "value",
    [
        "=TODAY()",
        "bad",
        "2025-02-30",
        123,
        None,
        pd.Timestamp("2025-01-01 12:00"),
        pd.Timestamp("2025-01-01", tz="UTC"),
    ],
)
def test_dates_require_valid_normalized_daily_values(comparison, tmp_path, value):
    frame = comparison.daily_deltas.astype({"as_of_date": object})
    frame.at[0, "as_of_date"] = value
    with pytest.raises((TypeError, ValueError)):
        _module().write_feature_bundle_comparison(
            _changed_table(comparison, "daily_deltas", frame), tmp_path / "out"
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "value",
    [
        None,
        3,
        "",
        " ",
        "https://example.com/out",
        "file:///tmp/out",
        "//server/share/out",
        "\\\\server\\share\\out",
        "out:stream",
        "bad\0path",
    ],
)
def test_unsafe_paths_are_refused(comparison, value):
    with pytest.raises((TypeError, ValueError)):
        _module().write_feature_bundle_comparison(comparison, value)


@pytest.mark.parametrize("kind", ["directory", "file"])
def test_existing_destinations_are_never_modified(comparison, tmp_path, kind):
    destination = tmp_path / "out"
    if kind == "directory":
        destination.mkdir()
        sentinel = destination / "keep.txt"
    else:
        sentinel = destination
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _module().write_feature_bundle_comparison(comparison, destination)
    assert sentinel.read_text("utf-8") == "keep"


@pytest.mark.parametrize(
    "filename",
    [*(f"{name}.csv" for name in TABLES), "comparison_manifest.pending.json"],
)
def test_write_failure_preserves_partial_evidence(
    comparison, tmp_path, monkeypatch, filename
):
    destination = tmp_path / "out"
    original_open = Path.open
    failure = OSError("injected write failure")

    def failing_open(path, mode="r", *args, **kwargs):
        if path.name == filename and "x" in mode:
            raise failure
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)
    with pytest.raises(OSError) as caught:
        _module().write_feature_bundle_comparison(comparison, destination)
    assert caught.value is failure
    assert destination.is_dir()
    assert not (destination / "comparison_manifest.json").exists()
    assert any("Inspect" in note for note in failure.__notes__)
    with pytest.raises(FileExistsError):
        _module().write_feature_bundle_comparison(comparison, destination)


@pytest.mark.parametrize("failure_mode", ["raise_after_prefix", "short_write"])
def test_partial_later_table_preserves_completed_and_partial_bytes(
    comparison, tmp_path, monkeypatch, failure_mode
):
    destination = tmp_path / "out"
    original_open = Path.open
    failure = OSError("injected failure after writing a prefix")
    expected = {
        name: getattr(comparison, name)
        .to_csv(index=False, na_rep="", lineterminator="\n")
        .encode("utf-8")
        for name in ("daily_metrics", "fold_metrics")
    }
    prefix = expected["fold_metrics"][:17]

    class PartialStream:
        def __init__(self, stream):
            self.stream = stream

        def write(self, payload):
            assert payload == expected["fold_metrics"]
            written = self.stream.write(payload[: len(prefix)])
            self.stream.flush()
            if failure_mode == "raise_after_prefix":
                raise failure
            return written

        def close(self):
            self.stream.close()

    def partial_open(path, mode="r", *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        if path.name == "fold_metrics.csv" and "x" in mode:
            return PartialStream(stream)
        return stream

    monkeypatch.setattr(Path, "open", partial_open)
    with pytest.raises(OSError) as caught:
        _module().write_feature_bundle_comparison(comparison, destination)
    if failure_mode == "raise_after_prefix":
        assert caught.value is failure
    else:
        assert "incomplete" in str(caught.value)
    assert any("Inspect" in note for note in caught.value.__notes__)
    assert (destination / "daily_metrics.csv").read_bytes() == expected["daily_metrics"]
    assert (destination / "fold_metrics.csv").read_bytes() == prefix
    assert {path.name for path in destination.iterdir()} == {
        "daily_metrics.csv",
        "fold_metrics.csv",
    }
    assert not (destination / "comparison_manifest.json").exists()


@pytest.mark.parametrize("published", [False, True])
def test_publication_error_is_ambiguous_without_cleanup_or_success(
    comparison, tmp_path, monkeypatch, published
):
    destination = tmp_path / "out"
    original_rename = Path.rename
    failure = OSError("injected publication failure")

    def failed_rename(path, target):
        if path.name == "comparison_manifest.pending.json":
            if published:
                original_rename(path, target)
            raise failure
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", failed_rename)
    with pytest.raises(OSError) as caught:
        _module().write_feature_bundle_comparison(comparison, destination)
    assert caught.value is failure
    assert (destination / "comparison_manifest.json").exists() is published
    assert all((destination / f"{name}.csv").exists() for name in TABLES)
    assert any("Inspect" in note for note in failure.__notes__)


def test_writer_never_evaluates_fits_or_opens_registry(
    comparison, tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("Writer attempted research execution or input loading")

    monkeypatch.setattr(
        "quant_metric_research.benchmark_comparison.evaluate_prediction_frame",
        forbidden,
    )
    monkeypatch.setattr(
        "quant_metric_research.benchmark.run_stage3_benchmark", forbidden
    )
    monkeypatch.setattr(
        "quant_metric_research.experiment_registry.ExperimentRegistry.__init__",
        forbidden,
    )
    monkeypatch.setattr("quant_metric_research.io.read_table", forbidden)
    _module().write_feature_bundle_comparison(comparison, tmp_path / "out")


def test_missing_parent_is_not_created(comparison, tmp_path):
    with pytest.raises(FileNotFoundError):
        _module().write_feature_bundle_comparison(
            comparison, tmp_path / "missing" / "out"
        )
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize("kind", ["file", "directory", "dangling_link"])
def test_any_final_manifest_entry_blocks_publication(
    comparison, tmp_path, monkeypatch, kind
):
    module = _module()
    destination = tmp_path / "out"
    marker = destination / "comparison_manifest.json"
    original_write = module._write_exclusive
    original_lstat = Path.lstat

    def write_then_collide(path, payload):
        original_write(path, payload)
        if path.name == "comparison_manifest.pending.json":
            if kind == "file":
                marker.write_bytes(b"keep")
            elif kind == "directory":
                marker.mkdir()

    def dangling_lstat(path, *args, **kwargs):
        if path == marker and kind == "dangling_link":
            return original_lstat(destination)
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(module, "_write_exclusive", write_then_collide)
    monkeypatch.setattr(Path, "lstat", dangling_lstat)
    with pytest.raises(FileExistsError):
        module.write_feature_bundle_comparison(comparison, destination)
    assert (destination / "comparison_manifest.pending.json").exists()
    if kind == "file":
        assert marker.read_bytes() == b"keep"
    elif kind == "directory":
        assert marker.is_dir()


def test_exclusive_table_creation_preserves_an_existing_child(
    comparison, tmp_path, monkeypatch
):
    destination = tmp_path / "out"
    original_open = Path.open
    collided = False

    def collide(path, mode="r", *args, **kwargs):
        nonlocal collided
        if path.name == "daily_metrics.csv" and "x" in mode and not collided:
            collided = True
            with original_open(path, "xb") as stream:
                stream.write(b"keep")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", collide)
    with pytest.raises(FileExistsError):
        _module().write_feature_bundle_comparison(comparison, destination)
    assert (destination / "daily_metrics.csv").read_bytes() == b"keep"
    assert not (destination / "comparison_manifest.json").exists()


@pytest.mark.parametrize("ancestor", [False, True])
def test_destination_or_ancestor_symlink_is_refused(
    comparison, tmp_path, monkeypatch, ancestor
):
    import stat
    from types import SimpleNamespace

    destination = tmp_path / "parent" / "out"
    intercepted = destination.parent if ancestor else destination
    original_lstat = Path.lstat

    def symlink_lstat(path, *args, **kwargs):
        if path == intercepted:
            return SimpleNamespace(st_mode=stat.S_IFLNK, st_file_attributes=0)
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", symlink_lstat)
    with pytest.raises(ValueError, match="Symlink|junction"):
        _module().write_feature_bundle_comparison(comparison, destination)
    assert not (tmp_path / "parent").exists()


def test_serialization_failure_creates_no_output(comparison, tmp_path, monkeypatch):
    failure = UnicodeError("injected serialization failure")

    def forbidden(*args, **kwargs):
        raise failure

    monkeypatch.setattr(pd.DataFrame, "to_csv", forbidden)
    with pytest.raises(UnicodeError) as caught:
        _module().write_feature_bundle_comparison(comparison, tmp_path / "out")
    assert caught.value is failure
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("value", [None, pd.NA, float("nan")])
def test_genuine_missing_numeric_cells_remain_blank(comparison, tmp_path, value):
    frame = comparison.daily_deltas.astype({"spread_delta": object})
    frame.at[0, "spread_delta"] = value
    result = _module().write_feature_bundle_comparison(
        _changed_table(comparison, "daily_deltas", frame), tmp_path / "out"
    )
    assert pd.isna(pd.read_csv(result.files["daily_deltas"]).at[0, "spread_delta"])


def test_json_strings_are_preserved_not_csv_escaped(comparison, tmp_path):
    changed = replace(
        comparison, metadata={**comparison.metadata, "note": "=formula-like 中文"}
    )
    result = _module().write_feature_bundle_comparison(changed, tmp_path / "out")
    raw = result.files["comparison_manifest"].read_text("utf-8")
    assert json.loads(raw)["report_metadata"]["note"] == "=formula-like 中文"
    assert "NaN" not in raw and "Infinity" not in raw


def test_wrong_result_type_fails_before_output_creation(tmp_path):
    with pytest.raises(TypeError, match="FeatureBundleComparison"):
        _module().write_feature_bundle_comparison(object(), tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("write_fails", [False, True])
def test_close_failure_preserves_the_primary_exception(
    tmp_path, monkeypatch, write_fails
):
    write_failure = OSError("primary write failed")
    close_failure = OSError("close failed")

    class FailingStream:
        def write(self, payload):
            if write_fails:
                raise write_failure
            return len(payload)

        def close(self):
            raise close_failure

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: FailingStream())
    with pytest.raises(OSError) as caught:
        _module()._write_exclusive(tmp_path / "out.csv", b"payload")
    assert caught.value is (write_failure if write_fails else close_failure)
    if write_fails:
        assert any("close" in note.lower() for note in write_failure.__notes__)
