from __future__ import annotations

import json
import math
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from .benchmark import BenchmarkRun

_PREDICTION_FORMATS = frozenset({"csv", "parquet"})


@dataclass(frozen=True, slots=True)
class BenchmarkArtifacts:
    """Paths and manifest for one materialized benchmark run."""

    output_dir: Path
    files: MappingProxyType[str, Path]
    manifest: MappingProxyType[str, Any]


def _json_value(value: Any, *, location: str) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, pd.Timedelta):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, pd.Period):
        return str(value)
    if isinstance(value, pd.Interval):
        return str(value)
    if isinstance(value, np.ndarray):
        return [
            _json_value(item, location=f"{location}[{index}]")
            for index, item in enumerate(value.tolist())
        ]
    if isinstance(value, np.generic):
        return _json_value(value.item(), location=location)
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            string_key = str(key)
            if string_key in converted:
                raise TypeError(
                    f"JSON mapping at {location} contains colliding keys: {key!r}."
                )
            converted[string_key] = _json_value(
                item,
                location=f"{location}.{string_key}",
            )
        return converted
    if isinstance(value, (tuple, list)):
        return [
            _json_value(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    raise TypeError(
        f"Value at {location} is not JSON serializable: {type(value).__name__}."
    )


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _write_json(value: Any, destination: Path) -> None:
    destination.write_text(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _stable_json_cell(value: Any) -> str:
    normalized = _json_value(value, location="parameters")
    return json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _serialized_trials(frame: pd.DataFrame) -> pd.DataFrame:
    serialized = frame.copy(deep=True)
    if "parameters" in serialized.columns:
        serialized["parameters"] = serialized["parameters"].map(_stable_json_cell)
    return serialized


def _artifact_paths(
    destination: Path,
    *,
    prediction_format: str,
) -> dict[str, Path]:
    return {
        "benchmark_manifest": destination / "benchmark_manifest.json",
        "data_gate": destination / "data_gate.json",
        "fold_assignments": destination / "fold_assignments.parquet",
        "hyperparameter_trials": destination / "hyperparameter_trials.csv",
        "screening_by_fold": destination / "screening_by_fold.csv",
        "oos_predictions": destination / f"oos_predictions.{prediction_format}",
        "daily_metrics": destination / "daily_metrics.csv",
        "fold_summary": destination / "fold_summary.csv",
        "benchmark_summary": destination / "benchmark_summary.csv",
        "acceptance": destination / "acceptance.json",
    }


def write_benchmark_run(
    result: BenchmarkRun,
    output_dir: str | Path,
    *,
    prediction_format: str = "parquet",
) -> BenchmarkArtifacts:
    """Write the complete, reproducible artifact bundle for a Stage 3 run."""

    if not isinstance(result, BenchmarkRun):
        raise TypeError("result must be a BenchmarkRun.")
    if (
        not isinstance(prediction_format, str)
        or prediction_format not in _PREDICTION_FORMATS
    ):
        raise ValueError("prediction_format must be either 'csv' or 'parquet'.")
    try:
        destination = Path(output_dir)
    except TypeError as error:
        raise TypeError("output_dir must be a path-like value.") from error
    if destination.exists():
        if not destination.is_dir():
            raise NotADirectoryError(f"Output path is not a directory: {destination}")
        raise FileExistsError(f"Output directory already exists: {destination}")

    manifest = _json_value(result.manifest, location="manifest")
    data_gate = _json_value(result.data_gate, location="data_gate")
    acceptance = _json_value(result.acceptance, location="acceptance")
    tuning_trials = _serialized_trials(result.tuning_trials)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            dir=destination.parent,
        )
    )
    temporary_paths = _artifact_paths(temporary, prediction_format=prediction_format)
    try:
        _write_json(manifest, temporary_paths["benchmark_manifest"])
        _write_json(data_gate, temporary_paths["data_gate"])
        _write_json(acceptance, temporary_paths["acceptance"])
        result.fold_assignments.to_parquet(
            temporary_paths["fold_assignments"], index=False
        )
        tuning_trials.to_csv(temporary_paths["hyperparameter_trials"], index=False)
        result.screening_by_fold.to_csv(
            temporary_paths["screening_by_fold"], index=False
        )
        if prediction_format == "parquet":
            result.predictions.to_parquet(
                temporary_paths["oos_predictions"], index=False
            )
        else:
            result.predictions.to_csv(temporary_paths["oos_predictions"], index=False)
        result.daily_metrics.to_csv(temporary_paths["daily_metrics"], index=False)
        result.fold_metrics.to_csv(temporary_paths["fold_summary"], index=False)
        result.summary.to_csv(temporary_paths["benchmark_summary"], index=False)
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    paths = _artifact_paths(destination, prediction_format=prediction_format)

    return BenchmarkArtifacts(
        output_dir=destination,
        files=MappingProxyType(dict(paths)),
        manifest=_freeze_json(manifest),
    )


__all__ = ["BenchmarkArtifacts", "write_benchmark_run"]
