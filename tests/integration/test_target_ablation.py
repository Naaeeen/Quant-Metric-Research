"""Exercise the target-only example using real, small model fits."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from quant_metric_research import BenchmarkConfig, ExperimentRegistry, NestedSplitConfig
from quant_metric_research.config import DEFAULT_FEATURE_COLUMNS

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "target_ablation.py"


@pytest.fixture()
def example():
    spec = importlib.util.spec_from_file_location("target_ablation", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def one_training_thread():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture()
def inputs(tmp_path):
    random = np.random.default_rng(57)
    dates = pd.bdate_range("2022-01-03", periods=34)
    records = []
    for day, date in enumerate(dates[:32]):
        for stock in range(8):
            values = random.normal(size=10)
            records.append(
                {
                    "as_of_date": date,
                    "symbol": f"S{stock}",
                    "label_end_date": dates[day + 2],
                    "forward_excess_return": (
                        0.02 * values[0]
                        + 0.01 * values[1] ** 2
                        + random.normal(0, 0.003)
                        if stock != 7
                        else np.nan
                    ),
                    **dict(zip(DEFAULT_FEATURE_COLUMNS, values, strict=True)),
                }
            )
    panel = pd.DataFrame(records)
    config = BenchmarkConfig(
        feature_columns=DEFAULT_FEATURE_COLUMNS,
        split=NestedSplitConfig(4, 1, 6, 10, 1, 4, 6),
        min_cross_section=6,
        quantiles=2,
        hac_lags=1,
        ridge_alphas=(1.0,),
        hist_max_iter=8,
        hist_min_samples_leaf=3,
    )
    panel.to_parquet(tmp_path / "panel.parquet", index=False)
    (tmp_path / "config.json").write_text(json.dumps(config.to_mapping()))
    registry = ExperimentRegistry(tmp_path / "history.sqlite3")
    prior = registry.start_development(
        study_id="prior", hypothesis="Retained attempt", configuration={}
    )
    registry.fail_run(prior, error_type="InterruptedError")
    return {
        "panel_path": tmp_path / "panel.parquet",
        "config_path": tmp_path / "config.json",
        "registry_path": tmp_path / "history.sqlite3",
        "output_dir": tmp_path / "ablation",
        "study_id": "synthetic-target-test",
        "hypothesis": "Compare return and rank labels on invented data.",
    }


def test_real_training_compares_five_arms_and_retains_history(
    example, inputs, monkeypatch, capsys
):
    runs = []
    original = example.run_stage3_benchmark

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        runs.append(result)
        return result

    monkeypatch.setattr(example, "run_stage3_benchmark", capture)
    arguments = [str(EXAMPLE)]
    for field, option in (
        ("panel_path", "panel"),
        ("config_path", "config"),
        ("registry_path", "registry"),
        ("output_dir", "output-dir"),
        ("study_id", "study-id"),
        ("hypothesis", "hypothesis"),
    ):
        arguments.extend([f"--{option}", str(inputs[field])])
    monkeypatch.setattr(sys, "argv", arguments)
    assert example.main() == 0
    report = json.loads(capsys.readouterr().out)
    output = inputs["output_dir"]
    assert report == json.loads((output / "completion.json").read_text())
    assert report["status"] == "complete"
    assert report["evaluate_lockbox"] is False
    assert len(report["runs"]) == 2
    assert all(row["elapsed_seconds"] >= 0 for row in report["runs"])
    summary = pd.read_csv(output / "summary.csv")
    assert set(summary["model"]) == {
        "equal_weight_rank",
        "raw_ridge",
        "rank_ridge",
        "raw_hist_gradient_boosting",
        "rank_hist_gradient_boosting",
    }
    assert set(summary["evaluation_scope"]) == {"native", "common"}
    assert summary["mean_rank_ic"].notna().all()
    assert summary["mean_spread"].abs().max() < 0.2
    assert not any("p_value" in column for column in summary)
    predictions = pd.read_parquet(output / "comparison_predictions.parquet")
    assert set(predictions["phase"]) == {"development"}
    pd.testing.assert_series_equal(
        predictions["target"], predictions["realized_return"], check_names=False
    )
    assert predictions["target"].isna().any()
    assert (predictions["train_label_end_max"] < predictions["evaluation_start"]).all()
    rows = ExperimentRegistry(inputs["registry_path"]).list_runs()
    assert len(rows) == 3
    assert [row["status"] for row in rows].count("failed") == 1
    assert {row["kind"] for row in rows} == {"development"}
    assert sum(row["status"] == "completed" for row in rows) == 2
    assert (output / "raw" / "benchmark_manifest.json").is_file()
    assert (output / "rank" / "benchmark_manifest.json").is_file()
    # Missing predictions shrink common coverage, without changing other scores.
    rank_predictions = runs[1].predictions.copy(deep=True)
    missing = (rank_predictions["model"] == "ridge") & (
        rank_predictions["symbol"] == "S0"
    )
    rank_predictions.loc[missing, "score"] = np.nan
    alternate = replace(runs[1], predictions=rank_predictions)
    reduced = example.comparison_predictions(runs[0], alternate)
    assert len(reduced) == len(predictions)
    assert reduced.loc[reduced["model"] == "rank_ridge", "score"].isna().sum() > 0
    pd.testing.assert_series_equal(
        reduced.loc[reduced["model"] == "raw_ridge", "score"].reset_index(drop=True),
        predictions.loc[predictions["model"] == "raw_ridge", "score"].reset_index(
            drop=True
        ),
    )
    config = BenchmarkConfig.from_mapping(json.loads(inputs["config_path"].read_text()))
    alternate_output = inputs["output_dir"].parent / "missing-prediction"
    alternate_output.mkdir()
    example._save_comparison(alternate_output, runs[0], alternate, config)
    daily = pd.read_csv(alternate_output / "daily_metrics.csv")
    native = daily.loc[
        (daily["model"] == "raw_ridge") & (daily["evaluation_scope"] == "native")
    ]
    common = daily.loc[
        (daily["model"] == "raw_ridge") & (daily["evaluation_scope"] == "common")
    ]
    assert native["score_coverage"].eq(1).all()
    assert common["score_coverage"].eq(7 / 8).all()
    removed = runs[1].predictions.loc[runs[1].predictions["model"].eq("ridge")].index[0]
    changed_keys = replace(runs[1], predictions=runs[1].predictions.drop(index=removed))
    with pytest.raises(ValueError, match="(baseline evidence|scoring rows)"):
        example.comparison_predictions(runs[0], changed_keys)
    before = (output / "completion.json").read_bytes()
    with pytest.raises(SystemExit) as error:
        example.main()
    assert error.value.code == 2
    assert (output / "completion.json").read_bytes() == before
    _assert_evaluation_outcomes_do_not_change_scores(example, inputs, runs[1])


def _assert_evaluation_outcomes_do_not_change_scores(example, inputs, original):
    frame = pd.read_parquet(inputs["panel_path"])
    evaluation_dates = original.predictions["as_of_date"].unique()
    evaluated = frame["as_of_date"].isin(evaluation_dates)
    frame.loc[evaluated & frame["symbol"].eq("S0"), "forward_excess_return"] = np.nan
    frame.loc[evaluated & frame["symbol"].eq("S1"), "forward_excess_return"] *= -100
    config = replace(
        example._config(inputs["config_path"].read_bytes()),
        target_column="forward_excess_rank",
    )
    changed = example.run_stage3_benchmark(
        example.prepare_panel(frame),
        config=config,
        evaluate_lockbox=False,
        registry=ExperimentRegistry(inputs["registry_path"]),
        study_id="synthetic-perturbation-test",
        hypothesis="Evaluation outcomes cannot affect their own fold's scores.",
    )
    columns = ["phase", "fold", "as_of_date", "symbol", "model", "score"]
    pd.testing.assert_frame_equal(
        original.predictions.loc[:, columns],
        changed.predictions.loc[:, columns],
        check_exact=True,
    )
    assert not original.predictions["target"].equals(changed.predictions["target"])


def test_ranks_use_observed_same_date_outcomes_and_preserve_rows(example, inputs):
    frame = pd.read_parquet(inputs["panel_path"])
    first_date = frame["as_of_date"].min()
    frame.loc[frame["as_of_date"] == first_date, "forward_excess_return"] = [
        0.01,
        0.01,
        0.03,
        np.nan,
        0.02,
        0.04,
        0.05,
        0.06,
    ]
    original = frame.copy(deep=True)
    ranked = example.prepare_panel(frame)
    first = ranked.loc[ranked["as_of_date"] == first_date]
    assert first["forward_excess_rank"].tolist() == pytest.approx(
        [1.5 / 7, 1.5 / 7, 4 / 7, np.nan, 3 / 7, 5 / 7, 6 / 7, 1],
        nan_ok=True,
    )
    changed = frame.assign(
        forward_excess_return=frame["forward_excess_return"].where(
            frame["as_of_date"] == first_date, np.nan
        )
    )
    pd.testing.assert_frame_equal(
        first,
        example.prepare_panel(changed).loc[
            lambda rows: rows["as_of_date"] == first_date
        ],
    )
    pd.testing.assert_frame_equal(frame, original)
    assert len(ranked) == len(frame)


@pytest.mark.parametrize("kind", ["different_maturity", "missing_maturity"])
def test_invalid_rank_label_maturity_fails_before_training(example, inputs, kind):
    panel = pd.read_parquet(inputs["panel_path"])
    panel.loc[0, "label_end_date"] = (
        pd.NaT
        if kind == "missing_maturity"
        else panel.loc[0, "label_end_date"] + pd.Timedelta(days=1)
    )
    with pytest.raises(ValueError, match="(maturity|label_end_date)"):
        example.prepare_panel(panel)


@pytest.mark.parametrize(
    "case",
    [
        "missing_registry",
        "empty_registry",
        "blank_history",
        "existing_output",
        "config",
    ],
)
def test_invalid_request_cannot_start_training(example, inputs, case, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid inputs must fail before fitting.")

    monkeypatch.setattr(example, "run_stage3_benchmark", forbidden)
    if case == "missing_registry":
        inputs = {
            **inputs,
            "registry_path": inputs["registry_path"].with_name("absent.sqlite3"),
        }
    elif case in {"empty_registry", "blank_history"}:
        path = inputs["registry_path"].with_name("empty.sqlite3")
        if case == "empty_registry":
            path.touch()
        else:
            ExperimentRegistry(path)
        inputs = {**inputs, "registry_path": path}
    elif case == "existing_output":
        inputs["output_dir"].mkdir()
    else:
        config = json.loads(inputs["config_path"].read_text())
        config.pop("ridge_alphas")
        inputs["config_path"].write_text(json.dumps(config))
    with pytest.raises((ValueError, FileNotFoundError, FileExistsError)):
        example.run_target_ablation(**inputs)


def test_changed_source_is_rejected_before_training(example, inputs, monkeypatch):
    original = example.read_table

    def change_config(path):
        frame = original(path)
        config = json.loads(inputs["config_path"].read_text())
        config["ridge_alphas"] = [99.0]
        inputs["config_path"].write_text(json.dumps(config))
        return frame

    monkeypatch.setattr(example, "read_table", change_config)
    with pytest.raises(ValueError, match="changed"):
        example.run_target_ablation(**inputs)
    assert len(ExperimentRegistry(inputs["registry_path"]).list_runs()) == 1
    assert not (inputs["output_dir"] / "completion.json").exists()


def test_second_fit_failure_retains_first_bundle_and_failed_record(
    example, inputs, monkeypatch
):
    from quant_metric_research import experiment_workflow

    original = experiment_workflow._execute_benchmark

    def fail_rank(*args, **kwargs):
        if kwargs["config"].target_column == "forward_excess_rank":
            raise RuntimeError("Synthetic training interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(experiment_workflow, "_execute_benchmark", fail_rank)
    with pytest.raises(RuntimeError, match="interruption"):
        example.run_target_ablation(**inputs)
    output = inputs["output_dir"]
    assert (output / "raw" / "benchmark_manifest.json").is_file()
    assert (output / "failure.json").is_file()
    assert not (output / "completion.json").exists()
    rows = ExperimentRegistry(inputs["registry_path"]).list_runs()
    assert len(rows) == 3
    assert sum(row["status"] == "failed" for row in rows) == 2


def test_publication_failure_preserves_completed_training(example, inputs, monkeypatch):
    def fail_write(*args, **kwargs):
        raise OSError("Synthetic publication failure")

    monkeypatch.setattr(example, "write_benchmark_run", fail_write)
    with pytest.raises(OSError, match="publication"):
        example.run_target_ablation(**inputs)
    assert not (inputs["output_dir"] / "completion.json").exists()
    rows = ExperimentRegistry(inputs["registry_path"]).list_runs()
    assert sum(row["status"] == "completed" for row in rows) == 1
