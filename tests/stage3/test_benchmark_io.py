from __future__ import annotations

import json
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from quant_metric_research.benchmark import BenchmarkRun
from quant_metric_research.benchmark_io import write_benchmark_run


def _benchmark_run() -> BenchmarkRun:
    return BenchmarkRun(
        data_gate=MappingProxyType(
            {
                "provider_verified": np.bool_(False),
                "reasons": ("point_in_time_provider_unverified",),
                "checked_at": pd.Timestamp("2024-06-30T12:30:00Z"),
                "missing_value": pd.NA,
            }
        ),
        fold_assignments=pd.DataFrame(
            {
                "phase": ["development"],
                "fold": [np.int64(1)],
                "evaluation_start": [pd.Timestamp("2024-05-01")],
            }
        ),
        predictions=pd.DataFrame(
            {
                "phase": ["development"],
                "fold": [1],
                "as_of_date": [pd.Timestamp("2024-05-01")],
                "symbol": ["AAA"],
                "model": ["ridge"],
                "score": [0.25],
            }
        ),
        daily_metrics=pd.DataFrame(
            {
                "phase": ["development"],
                "fold": [1],
                "model": ["ridge"],
                "rank_ic": [0.4],
            }
        ),
        fold_metrics=pd.DataFrame(
            {
                "phase": ["development"],
                "fold": [1],
                "model": ["ridge"],
                "mean_rank_ic": [0.4],
            }
        ),
        tuning_trials=pd.DataFrame(
            {
                "phase": ["development", "development"],
                "family": ["ridge", "ridge"],
                "parameters": [
                    {"features": ("value", "quality"), "alpha": np.float64(1.0)},
                    MappingProxyType({"alpha": np.float64(0.1)}),
                ],
            }
        ),
        screening_by_fold=pd.DataFrame(
            {
                "phase": ["development"],
                "fold": [1],
                "feature": ["value"],
                "selected": [True],
            }
        ),
        summary=pd.DataFrame(
            {
                "phase": ["development"],
                "model": ["ridge"],
                "mean_rank_ic": [0.4],
            }
        ),
        acceptance=MappingProxyType(
            {
                "locked_test_used_once": np.bool_(True),
                "frozen_model_family": "ridge",
            }
        ),
        manifest=MappingProxyType(
            {
                "implementation_version": "0.2.0",
                "run_fingerprint": "abc123",
                "seeds": (np.int64(17),),
            }
        ),
    )


def test_write_benchmark_run_creates_complete_deterministic_bundle(tmp_path) -> None:
    run = _benchmark_run()
    original_parameters = run.tuning_trials["parameters"].tolist()
    destination = tmp_path / "benchmark-run"

    artifacts = write_benchmark_run(run, destination)

    expected_names = {
        "acceptance.json",
        "benchmark_manifest.json",
        "benchmark_summary.csv",
        "daily_metrics.csv",
        "data_gate.json",
        "fold_assignments.parquet",
        "fold_summary.csv",
        "hyperparameter_trials.csv",
        "oos_predictions.parquet",
        "screening_by_fold.csv",
    }
    assert {path.name for path in artifacts.files.values()} == expected_names
    assert {path.name for path in destination.iterdir()} == expected_names
    assert artifacts.output_dir == destination

    manifest_text = (destination / "benchmark_manifest.json").read_text(
        encoding="utf-8"
    )
    assert manifest_text.index('"implementation_version"') < manifest_text.index(
        '"run_fingerprint"'
    )
    assert json.loads(manifest_text) == {
        "implementation_version": "0.2.0",
        "run_fingerprint": "abc123",
        "seeds": [17],
    }
    assert artifacts.manifest["run_fingerprint"] == "abc123"
    assert artifacts.manifest["seeds"] == (17,)
    assert json.loads((destination / "data_gate.json").read_text("utf-8")) == {
        "checked_at": "2024-06-30T12:30:00+00:00",
        "missing_value": None,
        "provider_verified": False,
        "reasons": ["point_in_time_provider_unverified"],
    }

    pd.testing.assert_frame_equal(
        pd.read_parquet(destination / "fold_assignments.parquet"),
        run.fold_assignments,
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(destination / "oos_predictions.parquet"),
        run.predictions,
    )
    trials = pd.read_csv(destination / "hyperparameter_trials.csv")
    assert trials["parameters"].tolist() == [
        '{"alpha":1.0,"features":["value","quality"]}',
        '{"alpha":0.1}',
    ]
    assert run.tuning_trials["parameters"].tolist() == original_parameters

    with pytest.raises(TypeError):
        artifacts.files["new"] = destination / "new.csv"
    with pytest.raises(TypeError):
        artifacts.manifest["run_fingerprint"] = "changed"


def test_write_benchmark_run_supports_csv_predictions(tmp_path) -> None:
    destination = tmp_path / "csv-run"

    artifacts = write_benchmark_run(
        _benchmark_run(), destination, prediction_format="csv"
    )

    assert artifacts.files["oos_predictions"] == (destination / "oos_predictions.csv")
    assert not (destination / "oos_predictions.parquet").exists()
    loaded = pd.read_csv(destination / "oos_predictions.csv")
    assert loaded.loc[0, "model"] == "ridge"


def test_write_benchmark_run_produces_identical_text_artifacts(tmp_path) -> None:
    first = tmp_path / "first-run"
    second = tmp_path / "second-run"

    write_benchmark_run(_benchmark_run(), first, prediction_format="csv")
    write_benchmark_run(_benchmark_run(), second, prediction_format="csv")

    text_names = {
        "acceptance.json",
        "benchmark_manifest.json",
        "benchmark_summary.csv",
        "daily_metrics.csv",
        "data_gate.json",
        "fold_summary.csv",
        "hyperparameter_trials.csv",
        "oos_predictions.csv",
        "screening_by_fold.csv",
    }
    assert {name: (first / name).read_bytes() for name in text_names} == {
        name: (second / name).read_bytes() for name in text_names
    }


@pytest.mark.parametrize("prediction_format", ["json", "PARQUET", ""])
def test_write_benchmark_run_rejects_unsupported_prediction_format(
    tmp_path, prediction_format: str
) -> None:
    destination = tmp_path / "invalid-run"

    with pytest.raises(
        ValueError, match="prediction_format must be either 'csv' or 'parquet'"
    ):
        write_benchmark_run(
            _benchmark_run(),
            destination,
            prediction_format=prediction_format,
        )

    assert not destination.exists()


def test_write_benchmark_run_rejects_file_as_output_directory(tmp_path) -> None:
    destination = tmp_path / "already-a-file"
    destination.write_text("occupied", encoding="utf-8")

    with pytest.raises(NotADirectoryError, match="Output path is not a directory"):
        write_benchmark_run(_benchmark_run(), destination)

    assert destination.read_text(encoding="utf-8") == "occupied"


@pytest.mark.parametrize("with_existing_file", [False, True])
def test_write_benchmark_run_rejects_existing_directory(
    tmp_path,
    with_existing_file: bool,
) -> None:
    destination = tmp_path / "existing-run"
    destination.mkdir()
    if with_existing_file:
        (destination / "old-result.csv").write_text("stale", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        write_benchmark_run(_benchmark_run(), destination)

    expected_names = {"old-result.csv"} if with_existing_file else set()
    assert {path.name for path in destination.iterdir()} == expected_names


def test_write_failure_does_not_publish_partial_bundle(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "failed-run"

    def fail_csv(*args: object, **kwargs: object) -> None:
        raise OSError("simulated write failure")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_csv)

    with pytest.raises(OSError, match="simulated write failure"):
        write_benchmark_run(_benchmark_run(), destination)

    assert not destination.exists()
    assert not list(tmp_path.glob(".failed-run.*"))
