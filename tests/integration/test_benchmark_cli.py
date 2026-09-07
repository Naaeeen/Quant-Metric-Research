from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import quant_metric_research.cli as cli
from quant_metric_research.benchmark import BenchmarkRun
from quant_metric_research.benchmark_config import BenchmarkConfig


def _write_panel(path: Path) -> pd.DataFrame:
    panel = pd.DataFrame(
        {
            "as_of_date": ["2025-01-02", "2025-01-02"],
            "symbol": ["AAA", "BBB"],
            "label_end_date": ["2025-01-09", "2025-01-09"],
            "metric_a": [1.0, 2.0],
            "forward_excess_return": [0.01, -0.01],
        }
    )
    if path.suffix == ".csv":
        panel.to_csv(path, index=False)
    else:
        panel.to_parquet(path, index=False)
    return panel


def _write_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "feature_columns": ["metric_a"],
                "min_cross_section": 2,
                "quantiles": 2,
                "hac_lags": 0,
                "split": {
                    "final_test_date_count": 1,
                    "outer_n_splits": 1,
                    "outer_test_date_count": 1,
                    "outer_min_train_date_count": 1,
                    "inner_n_splits": 1,
                    "inner_validation_date_count": 1,
                    "inner_min_train_date_count": 1,
                },
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    (
        "panel_suffix",
        "format_arguments",
        "expected_prediction_format",
        "evaluate_lockbox",
    ),
    [
        (".csv", [], "parquet", False),
        (".parquet", ["--prediction-format", "csv", "--evaluate-lockbox"], "csv", True),
    ],
)
def test_benchmark_command_loads_panel_runs_and_delegates_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    panel_suffix: str,
    format_arguments: list[str],
    expected_prediction_format: str,
    evaluate_lockbox: bool,
) -> None:
    panel_path = tmp_path / f"panel{panel_suffix}"
    expected_panel = _write_panel(panel_path)
    config_path = tmp_path / "benchmark.json"
    _write_config(config_path)
    output_dir = tmp_path / "benchmark-results"
    expected_run = object()
    observed: dict[str, object] = {}

    def fake_run(
        panel: pd.DataFrame,
        *,
        config: BenchmarkConfig,
        evaluate_lockbox: bool = False,
        **experiment_options: object,
    ) -> object:
        observed["panel"] = panel
        observed["config"] = config
        observed["evaluate_lockbox"] = evaluate_lockbox
        observed["experiment_options"] = experiment_options
        return expected_run

    def fake_write(
        result: BenchmarkRun,
        destination: str | Path,
        *,
        prediction_format: str = "parquet",
    ) -> object:
        observed["result"] = result
        observed["destination"] = destination
        observed["prediction_format"] = prediction_format
        return object()

    monkeypatch.setattr(cli, "run_stage3_benchmark", fake_run)
    monkeypatch.setattr(cli, "write_benchmark_run", fake_write)

    arguments = [
        "benchmark",
        "--panel",
        str(panel_path),
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        *format_arguments,
    ]
    if evaluate_lockbox:
        arguments.extend(
            [
                "--registry",
                str(tmp_path / "registry.sqlite3"),
                "--development-run-id",
                "registered-reference",
            ]
        )
    exit_code = cli.main(arguments)

    assert exit_code == 0
    pd.testing.assert_frame_equal(observed["panel"], expected_panel)
    assert isinstance(observed["config"], BenchmarkConfig)
    assert observed["config"].feature_columns == ("metric_a",)
    assert observed["result"] is expected_run
    assert observed["destination"] == output_dir
    assert observed["prediction_format"] == expected_prediction_format
    assert observed["evaluate_lockbox"] is evaluate_lockbox


def test_existing_output_is_rejected_before_training(tmp_path, monkeypatch) -> None:
    output = tmp_path / "existing"
    output.mkdir()

    def forbidden(*args, **kwargs):
        raise AssertionError("An invalid destination must not consume the lockbox.")

    monkeypatch.setattr(cli, "run_stage3_benchmark", forbidden)
    with pytest.raises(FileExistsError, match="already exists"):
        cli.main(
            [
                "benchmark",
                "--panel",
                "unused.csv",
                "--config",
                "unused.json",
                "--output-dir",
                str(output),
                "--evaluate-lockbox",
                "--registry",
                str(tmp_path / "registry.sqlite3"),
                "--development-run-id",
                "registered-reference",
            ]
        )


def test_benchmark_command_reports_missing_panel_before_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "benchmark.json"
    _write_config(config_path)
    run_called = False

    def fail_if_called(*args: object, **kwargs: object) -> object:
        nonlocal run_called
        run_called = True
        return object()

    monkeypatch.setattr(cli, "run_stage3_benchmark", fail_if_called)

    with pytest.raises(FileNotFoundError, match="Input table does not exist"):
        cli.main(
            [
                "benchmark",
                "--panel",
                str(tmp_path / "missing.csv"),
                "--config",
                str(config_path),
                "--output-dir",
                str(tmp_path / "results"),
            ]
        )

    assert not run_called


def test_benchmark_command_rejects_invalid_prediction_format() -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "benchmark",
                "--panel",
                "panel.csv",
                "--config",
                "benchmark.json",
                "--output-dir",
                "results",
                "--prediction-format",
                "xlsx",
            ]
        )

    assert error.value.code == 2
