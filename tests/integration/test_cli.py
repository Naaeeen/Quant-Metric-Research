from __future__ import annotations

import json

import pandas as pd

from quant_metric_research.cli import main


def test_cli_run_writes_panel_and_stage_two_reports(
    tmp_path,
    market_fixture,
) -> None:
    prices, memberships, dates = market_fixture
    prices_path = tmp_path / "prices.csv"
    memberships_path = tmp_path / "memberships.csv"
    dates_path = tmp_path / "as_of_dates.csv"
    config_path = tmp_path / "config.json"
    output_path = tmp_path / "results"

    prices.to_csv(prices_path, index=False)
    memberships.to_csv(memberships_path, index=False)
    pd.DataFrame({"as_of_date": dates[6:11]}).to_csv(
        dates_path,
        index=False,
    )
    config_path.write_text(
        json.dumps(
            {
                "dataset_version": "cli-v1",
                "universe_id": "TEST",
                "benchmark_symbol": "BENCH",
                "lookback_sessions": 5,
                "min_observations": 4,
                "target_horizon_sessions": 2,
                "entry_lag_sessions": 1,
                "annualization_sessions": 5,
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "run",
            "--prices",
            str(prices_path),
            "--memberships",
            str(memberships_path),
            "--as-of-dates",
            str(dates_path),
            "--config",
            str(config_path),
            "--train-end",
            str(dates[10].date()),
            "--output-dir",
            str(output_path),
            "--min-cross-section",
            "2",
            "--hac-lags",
            "1",
            "--panel-format",
            "csv",
            "--with-pca",
            "--walk-forward-splits",
            "1",
            "--walk-forward-test-date-count",
            "1",
            "--walk-forward-min-train-date-count",
            "1",
        ]
    )

    assert exit_code == 0
    assert (output_path / "metric_panel.csv").exists()
    assert (output_path / "feature_quality.csv").exists()
    assert (output_path / "rank_ic_summary.csv").exists()
    assert (output_path / "feature_selection.json").exists()
    assert (output_path / "pca_scores.parquet").exists()
    assert (output_path / "walk_forward_metric_summary.csv").exists()

    selection = json.loads(
        (output_path / "feature_selection.json").read_text(encoding="utf-8")
    )
    assert selection["selected_features"]
    assert selection["fitted_through"] == str(dates[10].date())
