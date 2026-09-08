"""Exclusive local inspection bundles for already-computed descriptive comparisons."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from ._notebook_history_io import checked_path
from ._version import __version__
from .benchmark_comparison import DESCRIPTIVE_SUMMARY_COLUMNS, FeatureBundleComparison
from .contracts import _daily_dates

_TABLE_COLUMNS = MappingProxyType(
    {
        "daily_metrics": (
            "phase",
            "fold",
            "as_of_date",
            "model",
            "evaluation_scope",
            "rank_ic",
            "spread",
            "scoring_universe_count",
            "scored_count",
            "evaluation_count",
            "rank_ic_count",
            "spread_count",
            "eligible_target_count",
            "eligible_realized_return_count",
            "score_coverage",
            "rank_ic_coverage",
            "spread_coverage",
            "tied_score_fraction",
        ),
        "fold_metrics": (
            "phase",
            "fold",
            "model",
            "evaluation_scope",
            "date_count",
            "spread_date_count",
            "mean_rank_ic",
            "median_rank_ic",
            "positive_date_rate",
            "mean_spread",
            "mean_score_coverage",
            "mean_rank_ic_coverage",
            "mean_spread_coverage",
        ),
        "summary": DESCRIPTIVE_SUMMARY_COLUMNS,
        "daily_deltas": (
            "evaluation_scope",
            "contrast",
            "as_of_date",
            "rank_ic_delta",
            "spread_delta",
        ),
        "delta_summary": (
            "evaluation_scope",
            "contrast",
            "scheduled_date_count",
            "paired_rank_ic_date_count",
            "paired_rank_ic_date_coverage",
            "mean_paired_rank_ic_delta",
            "paired_spread_date_count",
            "paired_spread_date_coverage",
            "mean_paired_spread_delta",
        ),
    }
)
_TEXT_ENUMS = MappingProxyType(
    {
        "phase": frozenset({"development"}),
        "evaluation_scope": frozenset({"native", "common"}),
        "model": frozenset({"legacy_equal_rank", "legacy_model", "candidate_model"}),
        "contrast": frozenset(
            {
                "candidate_model_minus_legacy_model",
                "candidate_model_minus_legacy_equal_rank",
            }
        ),
    }
)
_SCOPE_FLAGS = MappingProxyType(
    {
        "caller_owned_inputs": True,
        "authenticity_verified": False,
        "history_verified": False,
        "final_outcomes_evaluated": False,
        "inference_reported": False,
    }
)


def _json_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Comparison JSON requires string mapping keys.")
        return {key: _json_copy(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_copy(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError("Comparison JSON requires finite JSON-compatible values.")


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class ComparisonArtifacts:
    """Defensively owned paths and metadata; materialized files remain mutable."""

    output_dir: Path
    files: Mapping[str, Path]
    manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.output_dir, Path):
            raise TypeError("output_dir must be a Path.")
        if not isinstance(self.files, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, Path)
            for key, value in self.files.items()
        ):
            raise TypeError("files must map string names to Paths.")
        if not isinstance(self.manifest, Mapping):
            raise TypeError("manifest must be a mapping.")
        object.__setattr__(self, "files", MappingProxyType(dict(self.files)))
        object.__setattr__(self, "manifest", _freeze(_json_copy(self.manifest)))


def _numeric_cell(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, (bool, np.bool_, np.timedelta64)):
        return False
    if isinstance(value, (int, np.integer)):
        return True
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value) or np.isnan(value))
    if isinstance(value, Decimal):
        return value.is_finite() or value.is_qnan()
    return False


def _validate_table(frame: pd.DataFrame, name: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")
    if (
        any(not isinstance(column, str) for column in frame.columns)
        or frame.columns.has_duplicates
        or set(frame.columns) != set(_TABLE_COLUMNS[name])
    ):
        raise ValueError(
            f"{name} requires exactly the definition-1 descriptive columns."
        )
    for column in frame:
        values = frame[column]
        if column in _TEXT_ENUMS:
            valid = all(
                isinstance(value, str) and value in _TEXT_ENUMS[column]
                for value in values
            )
        elif column == "as_of_date":
            if any(
                isinstance(value, str)
                and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None
                for value in values
            ):
                raise ValueError("Date text must use normalized YYYY-MM-DD values.")
            _daily_dates(values, field=f"{name}.as_of_date")
            valid = True
        else:
            valid = all(_numeric_cell(value) for value in values)
        if not valid:
            raise ValueError(f"Unsupported values in {name}.{column}.")


def _snapshot(result: FeatureBundleComparison) -> tuple[dict, dict[str, bytes], dict]:
    if not isinstance(result, FeatureBundleComparison):
        raise TypeError("result must be a FeatureBundleComparison.")
    metadata = _json_copy(result.metadata)
    if (
        metadata.get("comparison_definition_version") != "1"
        or metadata.get("claim_scope") != "development_descriptive_comparison"
        or any(
            metadata.get(key) is not expected for key, expected in _SCOPE_FLAGS.items()
        )
    ):
        raise ValueError(
            "Only definition-1 descriptive development comparisons are supported."
        )
    frames = {name: getattr(result, name) for name in _TABLE_COLUMNS}
    for name, frame in frames.items():
        _validate_table(frame, name)
    payloads = {
        name: frame.to_csv(index=False, na_rep="", lineterminator="\n").encode("utf-8")
        for name, frame in frames.items()
    }
    inventory = {
        name: {
            "filename": f"{name}.csv",
            "row_count": len(frames[name]),
            "columns": list(frames[name].columns),
            "byte_count": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for name, payload in payloads.items()
    }
    return metadata, payloads, inventory


def _encoded_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _write_exclusive(path: Path, payload: bytes) -> None:
    stream = path.open("xb")
    try:
        if stream.write(payload) != len(payload):
            raise OSError("Comparison output write was incomplete.")
    except BaseException as error:
        try:
            stream.close()
        except BaseException as closing_error:
            error.add_note(f"Secondary close failure ({type(closing_error).__name__}).")
        raise
    else:
        stream.close()


def _require_absent(path: Path) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        return
    raise FileExistsError(f"Comparison output entry already exists: {path}")


def write_feature_bundle_comparison(
    result: FeatureBundleComparison, output_dir: str | Path
) -> ComparisonArtifacts:
    """Save five inspection tables and publish a strict-JSON manifest last.

    Requires one writer on a non-hostile local filesystem. Path checks and rename
    are not a distributed lock or race-proof publication protocol. Failed writes
    retain partial evidence; a post-publication exception requires inspection.
    CSV is not exact dtype replay; digests certify neither history nor authenticity.
    No model evaluation, input-artifact loading or registry access is performed.
    """
    metadata, payloads, inventory = _snapshot(result)
    destination = checked_path(output_dir)
    _require_absent(destination)
    files = {name: destination / f"{name}.csv" for name in _TABLE_COLUMNS}
    files["comparison_manifest"] = destination / "comparison_manifest.json"
    manifest = {
        "artifact_format_version": "1",
        "status": "completed",
        "writer_package_version": __version__,
        "report_metadata": metadata,
        "tables": inventory,
    }
    encoded = _encoded_json(manifest)
    artifacts = ComparisonArtifacts(destination, files, manifest)
    destination.mkdir(exist_ok=False)
    try:
        for name, payload in payloads.items():
            _write_exclusive(files[name], payload)
        pending = destination / "comparison_manifest.pending.json"
        _write_exclusive(pending, encoded)
        _require_absent(files["comparison_manifest"])
        pending.rename(files["comparison_manifest"])
    except BaseException as error:
        error.add_note(
            "Inspect the comparison output directory and completion manifest; partial "
            "or already-published evidence is preserved. Do not auto-retry; use a new "
            "destination after inspection."
        )
        raise
    return artifacts


__all__ = ["ComparisonArtifacts", "write_feature_bundle_comparison"]
