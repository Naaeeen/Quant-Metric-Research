from __future__ import annotations

import json

import pandas as pd
import pytest

from quant_metric_research.config import PanelConfig
from quant_metric_research.io import (
    read_as_of_dates,
    read_json_object,
    read_table,
    write_research_run,
)
from quant_metric_research.pipeline import run_research
from quant_metric_research.validation import WalkForwardMetricConfig


def _config() -> PanelConfig:
    return PanelConfig(
        dataset_version="io-v1",
        universe_id="TEST",
        benchmark_symbol="BENCH",
        lookback_sessions=5,
        min_observations=4,
        target_horizon_sessions=2,
        entry_lag_sessions=1,
        annualization_sessions=5,
    )


def test_read_json_object_rejects_non_object(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="object"):
        read_json_object(path)


def test_read_as_of_dates_rejects_duplicates(tmp_path) -> None:
    path = tmp_path / "dates.csv"
    pd.DataFrame({"as_of_date": ["2025-01-01", "2025-01-01"]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="unique"):
        read_as_of_dates(path)


def test_read_table_rejects_unsupported_extension(tmp_path) -> None:
    path = tmp_path / "prices.txt"
    path.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="CSV"):
        read_table(path)


def test_write_research_run_emits_optional_pca_artifacts(
    tmp_path, market_fixture
) -> None:
    prices, memberships, dates = market_fixture
    # Allow BBB's first two-session forward label to mature before screening.
    result = run_research(
        prices,
        memberships,
        as_of_dates=tuple(dates[4:12]),
        config=_config(),
        train_end_date=dates[11],
        min_cross_section=2,
        minimum_coverage=0.8,
        redundancy_threshold=0.9,
        hac_lags=1,
        run_pca=True,
        pca_variance_to_keep=0.95,
        walk_forward_config=WalkForwardMetricConfig(
            n_splits=1,
            test_date_count=2,
            min_train_date_count=2,
        ),
    )

    write_research_run(result, tmp_path / "out", panel_format="parquet")

    assert (tmp_path / "out" / "metric_panel.parquet").exists()
    assert (tmp_path / "out" / "feature_selection.json").exists()
    assert (tmp_path / "out" / "pca_scores.parquet").exists()
    assert (tmp_path / "out" / "pca_component_weights.csv").exists()
    assert (tmp_path / "out" / "walk_forward_metric_summary.csv").exists()
    summary = json.loads(
        (tmp_path / "out" / "feature_selection.json").read_text(encoding="utf-8")
    )
    assert summary["selected_features"]


def test_write_research_run_rejects_bad_panel_format(tmp_path, market_fixture) -> None:
    prices, memberships, dates = market_fixture
    result = run_research(
        prices,
        memberships,
        as_of_dates=tuple(dates[8:11]),
        config=_config(),
        train_end_date=dates[10],
        min_cross_section=2,
        minimum_coverage=0.8,
        redundancy_threshold=0.9,
        hac_lags=1,
        run_pca=False,
    )

    with pytest.raises(ValueError, match="csv or parquet"):
        write_research_run(result, tmp_path / "out", panel_format="json")
