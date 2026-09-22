"""Synthetic orchestration checks for the fixed two-bundle example."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from quant_metric_research import (
    FEATURE_BUNDLES,
    BenchmarkConfig,
    ExperimentRegistry,
    NestedSplitConfig,
)

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "feature_ablation.py"
LEGACY, EXTENDED = "legacy10_v1", "legacy10_plus_price3_v1"


@pytest.fixture
def example():
    spec = importlib.util.spec_from_file_location("feature_ablation", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def one_thread():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture
def inputs(tmp_path):
    rng = np.random.default_rng(57)
    dates = pd.bdate_range("2022-01-03", periods=34)
    rows = []
    for day, date in enumerate(dates[:32]):
        for stock in range(8):
            values = rng.normal(size=13)
            rows.append(
                {
                    "as_of_date": date,
                    "symbol": f"S{stock}",
                    "label_end_date": dates[day + 2],
                    "forward_excess_return": 0.02 * values[0] + 0.01 * values[10]
                    if stock != 7
                    else np.nan,
                    **dict(
                        zip(
                            FEATURE_BUNDLES[EXTENDED].feature_columns,
                            values,
                            strict=True,
                        )
                    ),
                }
            )
    panel = pd.DataFrame(rows)
    panel.loc[panel.symbol == "S7", "return_21s"] = np.nan
    # Normalize the entire enriched panel once so both runs fingerprint it equally.
    panel["ma_distance_63s"] = panel["ma_distance_63s"].round().astype(int)
    panel.to_parquet(tmp_path / "panel.parquet", index=False)
    config = BenchmarkConfig(
        feature_columns=FEATURE_BUNDLES[LEGACY].feature_columns,
        split=NestedSplitConfig(4, 1, 6, 10, 1, 4, 6),
        min_cross_section=6,
        quantiles=2,
        hac_lags=1,
        ridge_alphas=(1.0,),
        hist_max_iter=8,
        hist_min_samples_leaf=3,
    )
    (tmp_path / "config.json").write_text(json.dumps(config.to_mapping()))
    registry = ExperimentRegistry(tmp_path / "history.sqlite3")
    prior = registry.start_development(
        study_id="prior", hypothesis="Retain history", configuration={}
    )
    registry.fail_run(prior, error_type="InterruptedError")
    return dict(
        panel_path=tmp_path / "panel.parquet",
        config_path=tmp_path / "config.json",
        registry_path=tmp_path / "history.sqlite3",
        output_dir=tmp_path / "ablation",
        study_id="synthetic-features",
        hypothesis="Compare fixed features on invented data.",
    )


def test_real_runs_share_panel_and_only_change_features(example, inputs, monkeypatch):
    registry = ExperimentRegistry(inputs["registry_path"])
    prior = registry.list_runs()[0]
    before = inputs["panel_path"].read_bytes()
    observed = []
    run = example.run_stage3_benchmark

    def capture(panel, **kwargs):
        assert (inputs["output_dir"] / "declaration.json").is_file()
        assert (inputs["output_dir"] / "preflight.json").is_file()
        assert kwargs["evaluate_lockbox"] is False
        observed.append((panel.copy(deep=True), kwargs["config"]))
        return run(panel, **kwargs)

    monkeypatch.setattr(example, "run_stage3_benchmark", capture)
    report = example.run_feature_ablation(**inputs)
    assert report["status"] == "complete" and report["evaluate_lockbox"] is False
    assert json.loads((inputs["output_dir"] / "completion.json").read_text()) == report
    assert len(observed) == 2
    pd.testing.assert_frame_equal(observed[0][0], observed[1][0], check_exact=True)
    first, second = (item[1].to_mapping() for item in observed)
    assert first.pop("feature_columns") == FEATURE_BUNDLES[LEGACY].feature_columns
    assert second.pop("feature_columns") == FEATURE_BUNDLES[EXTENDED].feature_columns
    assert first == second
    assert registry.get_run(prior["run_id"]) == prior
    assert len(registry.list_runs()) == 3
    assert all(row["kind"] == "development" for row in registry.list_runs())
    assert inputs["panel_path"].read_bytes() == before
    for bundle in (LEGACY, EXTENDED):
        assert (inputs["output_dir"] / bundle / "benchmark_manifest.json").is_file()
        assert (inputs["output_dir"] / f"{bundle}_timing.json").is_file()
    preflight = json.loads((inputs["output_dir"] / "preflight.json").read_text())
    assert (
        preflight[LEGACY]["summary"]["development"]["feature_coverage"]
        != preflight[EXTENDED]["summary"]["development"]["feature_coverage"]
    )
    for family in ("ridge", "hist_gradient_boosting"):
        directory = inputs["output_dir"] / f"comparison_{family}"
        assert (directory / "comparison_manifest.json").is_file()
        summary = pd.read_csv(directory / "summary.csv")
        assert set(summary.model) == {
            "legacy_equal_rank",
            "legacy_model",
            "candidate_model",
        }
        assert set(summary.evaluation_scope) == {"native", "common"}
        assert not any("p_value" in column for column in summary)


@pytest.mark.parametrize("drift", ["locked", "outer", "inner", "infeasible"])
def test_preflight_drift_refused_before_fitting(example, inputs, monkeypatch, drift):
    original = example.preflight_benchmark

    def change(panel, *, config):
        result = original(panel, config=config)
        if config.feature_columns == FEATURE_BUNDLES[EXTENDED].feature_columns:
            if drift == "locked":
                result["summary"]["locked_interval"]["start"] = "1900-01-01"
            elif drift == "outer":
                result["folds"][0]["evaluation_start"] = "1900-01-01"
            elif drift == "inner":
                result["folds"][0]["inner_folds"][0]["training"]["label_end_max"] = (
                    "1900-01-01"
                )
            else:
                result["feasible"] = False
        return result

    monkeypatch.setattr(example, "preflight_benchmark", change)
    monkeypatch.setattr(
        example, "run_stage3_benchmark", lambda *a, **k: pytest.fail("Must not fit.")
    )
    with pytest.raises(ValueError, match="(preflight|schedule)"):
        example.run_feature_ablation(**inputs)
    assert len(ExperimentRegistry(inputs["registry_path"]).list_runs()) == 1
    assert not inputs["output_dir"].exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("ridge_alphas", None),
        ("model_families", ["ridge"]),
        ("include_pca_model", True),
        ("target_column", "other"),
        ("realized_return_column", "other"),
        ("primary_baseline", "best_metric"),
        ("feature_columns", list(FEATURE_BUNDLES[EXTENDED].feature_columns)),
    ],
)
def test_invalid_config_never_fits(example, inputs, monkeypatch, field, value):
    config = json.loads(inputs["config_path"].read_text())
    if value is None:
        config.pop(field)
    else:
        config[field] = value
    inputs["config_path"].write_text(json.dumps(config))
    monkeypatch.setattr(
        example, "run_stage3_benchmark", lambda *a, **k: pytest.fail("Must not fit.")
    )
    with pytest.raises(ValueError):
        example.run_feature_ablation(**inputs)
    assert not inputs["output_dir"].exists()


@pytest.mark.parametrize(
    "case", ["missing_feature", "missing_registry", "empty_registry", "existing_output"]
)
def test_invalid_inputs_never_fit(example, inputs, monkeypatch, case):
    if case == "missing_feature":
        panel = pd.read_parquet(inputs["panel_path"]).drop(columns="return_21s")
        panel.to_parquet(inputs["panel_path"], index=False)
    elif case == "missing_registry":
        inputs["registry_path"] = inputs["registry_path"].with_name("absent.sqlite3")
    elif case == "empty_registry":
        inputs["registry_path"] = inputs["registry_path"].with_name("empty.sqlite3")
        ExperimentRegistry(inputs["registry_path"])
    else:
        inputs["output_dir"].mkdir()
    monkeypatch.setattr(
        example, "run_stage3_benchmark", lambda *a, **k: pytest.fail("Must not fit.")
    )
    with pytest.raises((ValueError, OSError)):
        example.run_feature_ablation(**inputs)


def test_second_run_failure_retains_first_bundle_and_failed_history(
    example, inputs, monkeypatch
):
    from quant_metric_research import experiment_workflow

    original = experiment_workflow._execute_benchmark

    def fail(*args, **kwargs):
        if (
            kwargs["config"].feature_columns
            == FEATURE_BUNDLES[EXTENDED].feature_columns
        ):
            raise RuntimeError("Synthetic second-run failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(experiment_workflow, "_execute_benchmark", fail)
    with pytest.raises(RuntimeError, match="second-run"):
        example.run_feature_ablation(**inputs)
    assert (inputs["output_dir"] / LEGACY / "benchmark_manifest.json").is_file()
    assert (inputs["output_dir"] / "failure.json").is_file()
    assert not (inputs["output_dir"] / "completion.json").exists()
    statuses = [
        row["status"] for row in ExperimentRegistry(inputs["registry_path"]).list_runs()
    ]
    assert statuses.count("completed") == 1 and statuses.count("failed") == 2


@pytest.mark.parametrize("stage", ["bundle", "comparison"])
def test_publication_failure_retains_completed_work(
    example, inputs, monkeypatch, stage
):
    original = example.write_feature_bundle_comparison

    def fail(*args, **kwargs):
        if stage == "comparison" and args[1].name == "comparison_ridge":
            return original(*args, **kwargs)
        raise OSError("Synthetic publication failure")

    monkeypatch.setattr(
        example,
        "write_benchmark_run"
        if stage == "bundle"
        else "write_feature_bundle_comparison",
        fail,
    )
    with pytest.raises(OSError, match="publication"):
        example.run_feature_ablation(**inputs)
    assert (inputs["output_dir"] / f"{LEGACY}_timing.json").is_file()
    assert (inputs["output_dir"] / "failure.json").is_file()
    assert not (inputs["output_dir"] / "completion.json").exists()
    statuses = [
        row["status"] for row in ExperimentRegistry(inputs["registry_path"]).list_runs()
    ]
    assert statuses.count("completed") == (1 if stage == "bundle" else 2)
    if stage == "comparison":
        assert (
            inputs["output_dir"] / "comparison_ridge" / "comparison_manifest.json"
        ).is_file()


def test_changed_source_fails_before_fitting(example, inputs, monkeypatch):
    original = example.read_table

    def change(path):
        frame = original(path)
        inputs["config_path"].write_text(inputs["config_path"].read_text() + " ")
        return frame

    monkeypatch.setattr(example, "read_table", change)
    monkeypatch.setattr(
        example, "run_stage3_benchmark", lambda *a, **k: pytest.fail("Must not fit.")
    )
    with pytest.raises(ValueError, match="changed"):
        example.run_feature_ablation(**inputs)
    assert len(ExperimentRegistry(inputs["registry_path"]).list_runs()) == 1


def test_direct_cli_runs_and_refuses_overwrite(inputs):
    command = [sys.executable, str(EXAMPLE)]
    for field, flag in (
        ("panel_path", "panel"),
        ("config_path", "config"),
        ("registry_path", "registry"),
        ("output_dir", "output-dir"),
        ("study_id", "study-id"),
        ("hypothesis", "hypothesis"),
    ):
        command.extend([f"--{flag}", str(inputs[field])])
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "complete"
    repeated = subprocess.run(
        command, capture_output=True, text=True, timeout=60, check=False
    )
    assert repeated.returncode == 2
    assert "already exists" in repeated.stderr
