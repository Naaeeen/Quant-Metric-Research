from __future__ import annotations

import json

import pandas as pd
import pytest

from quant_metric_research.benchmark import BenchmarkConfig, NestedSplitConfig
from quant_metric_research.cli import main


def _panel() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=36)
    rows: list[dict[str, object]] = []
    for date_index, as_of_date in enumerate(dates):
        for symbol_index in range(10):
            centered = float(symbol_index) - 4.5
            momentum = centered + 0.03 * date_index
            value = float((symbol_index * 3 + date_index) % 11) - 5.0
            quality = -centered + 0.02 * date_index
            noise = float((symbol_index * 7 + date_index * 5) % 13) - 6.0
            target = 0.03 * momentum + 0.008 * value - 0.002 * quality
            rows.append(
                {
                    "dataset_version": "synthetic-v1",
                    "as_of_date": as_of_date,
                    "symbol": f"S{symbol_index:02d}",
                    "feature_available_at": as_of_date,
                    "label_end_date": as_of_date + pd.offsets.BDay(2),
                    "momentum": momentum,
                    "value": value,
                    "quality": quality,
                    "noise": noise,
                    "forward_excess_return": target,
                    "realized_return": target,
                }
            )
    return pd.DataFrame(rows)


def _config() -> BenchmarkConfig:
    return BenchmarkConfig(
        feature_columns=("momentum", "value", "quality", "noise"),
        target_column="forward_excess_return",
        realized_return_column="realized_return",
        split=NestedSplitConfig(
            final_test_date_count=3,
            outer_n_splits=2,
            outer_test_date_count=3,
            outer_min_train_date_count=12,
            inner_n_splits=2,
            inner_validation_date_count=2,
            inner_min_train_date_count=6,
        ),
        min_cross_section=6,
        minimum_coverage=0.8,
        redundancy_threshold=0.95,
        quantiles=3,
        hac_lags=1,
        ridge_alphas=(0.1, 1.0),
        hist_learning_rates=(0.05,),
        hist_max_leaf_nodes=(7,),
        hist_l2_regularization=(1.0,),
        hist_max_iter=30,
        hist_min_samples_leaf=3,
        random_seed=17,
    )


def test_benchmark_command_writes_stage3_artifacts(tmp_path) -> None:
    panel_path = tmp_path / "panel.parquet"
    config_path = tmp_path / "benchmark-config.json"
    output_dir = tmp_path / "benchmark-artifacts"
    _panel().to_parquet(panel_path, index=False)
    config_path.write_text(
        json.dumps(_config().to_mapping(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "benchmark",
            "--panel",
            str(panel_path),
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--prediction-format",
            "csv",
        ]
    )

    assert exit_code == 0
    assert output_dir.is_dir()
    assert {
        path.name for path in output_dir.iterdir()
    } == {
        "acceptance.json",
        "benchmark_manifest.json",
        "benchmark_summary.csv",
        "daily_metrics.csv",
        "data_gate.json",
        "fold_assignments.parquet",
        "fold_summary.csv",
        "hyperparameter_trials.csv",
        "oos_predictions.csv",
        "screening_by_fold.csv",
    }


def test_run_command_requires_complete_walk_forward_group(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main(
            [
                "run",
                "--prices",
                "prices.parquet",
                "--memberships",
                "memberships.parquet",
                "--as-of-dates",
                "dates.csv",
                "--config",
                "config.json",
                "--train-end",
                "2024-06-30",
                "--output-dir",
                "artifacts/run-001",
                "--min-cross-section",
                "6",
                "--hac-lags",
                "1",
                "--walk-forward-splits",
                "2",
            ]
        )

    assert error.value.code == 2
    captured = capsys.readouterr()
    assert "All three walk-forward arguments must be provided together." in captured.err
