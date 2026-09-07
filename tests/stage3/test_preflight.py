from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest

from quant_metric_research.benchmark_config import BenchmarkConfig, NestedSplitConfig
from quant_metric_research.preflight import preflight_benchmark


def _panel() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=40)
    return pd.DataFrame(
        [
            {
                "as_of_date": date,
                "symbol": f"S{stock}",
                "label_end_date": date + pd.offsets.BDay(2),
                "momentum": float(stock + day),
                "value": float(stock - day),
                "target": float(stock) / 100.0,
                "realized": float(stock) / 200.0,
            }
            for day, date in enumerate(dates)
            for stock in range(6)
        ]
    )


def _config() -> BenchmarkConfig:
    return BenchmarkConfig(
        feature_columns=("momentum", "value"),
        target_column="target",
        realized_return_column="realized",
        min_cross_section=4,
        quantiles=2,
        hac_lags=3,
        split=NestedSplitConfig(
            final_test_date_count=4,
            outer_n_splits=2,
            outer_test_date_count=4,
            outer_min_train_date_count=12,
            inner_n_splits=2,
            inner_validation_date_count=3,
            inner_min_train_date_count=8,
        ),
    )


def test_preflight_is_serializable_and_never_fits_or_scores(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("Preflight must not fit or calculate predictive outcomes")

    for module, name in (
        ("benchmark", "fit_fold_screen"),
        ("benchmark", "fit_candidate"),
        ("benchmark", "fit_non_ml_baselines"),
        ("benchmark_tuning", "fit_fold_screen"),
        ("benchmark_tuning", "fit_candidate"),
        ("screening", "fit_metric_screen"),
        ("reduction", "fit_pca_baseline"),
        ("signals", "compute_daily_rank_ic"),
        ("signals", "compute_quantile_spreads"),
    ):
        monkeypatch.setattr(f"quant_metric_research.{module}.{name}", forbidden)
    panel = _panel()
    original = panel.copy(deep=True)

    result = preflight_benchmark(panel, config=_config())

    assert result["schema_version"] == "1"
    assert result["feasible"] is True
    assert result["errors"] == []
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    pd.testing.assert_frame_equal(panel, original)
    assert result["summary"]["no_training_performed"] is True
    assert result["summary"]["no_predictive_outcomes_computed"] is True
    assert result["summary"]["provider_provenance_verified"] is False
    assert result["summary"]["stage4_eligible"] is False
    assert len(result["folds"]) == 3
    assert [len(fold["inner_folds"]) for fold in result["folds"]] == [2, 2, 2]
    assert result["folds"][-1]["scope"] == "final_training"


def test_preflight_reports_purging_and_strict_temporal_boundaries() -> None:
    result = preflight_benchmark(_panel(), config=_config())

    for fold in result["folds"]:
        training = fold["training"]
        assert training["purged_label_overlap_count"] > 0
        assert training["label_end_max"] < fold["evaluation_start"]
        for inner in fold["inner_folds"]:
            assert inner["training"]["label_end_max"] < inner["evaluation"]["start"]
            assert inner["training"]["supervised_row_count"] > 0


def test_reserved_outcome_values_and_features_do_not_change_report() -> None:
    panel = _panel()
    report = preflight_benchmark(panel, config=_config())
    locked_start = pd.Timestamp(report["summary"]["locked_interval"]["start"])
    changed = panel.copy(deep=True)
    reserved = changed["as_of_date"] >= locked_start
    changed.loc[reserved, ["target", "realized", "momentum", "value"]] = 98765.0

    assert preflight_benchmark(changed, config=_config()) == report
    assert set(report["summary"]["locked_interval"]) == {
        "start",
        "end",
        "date_count",
        "row_count",
    }
    assert "symbol" not in json.dumps(report)


def test_development_missingness_is_visible_without_dropping_universe_rows() -> None:
    panel = _panel()
    panel.loc[panel["symbol"] == "S0", "momentum"] = float("nan")
    panel.loc[panel["symbol"] == "S1", "target"] = float("nan")
    panel.loc[panel["symbol"] == "S2", "realized"] = float("nan")

    report = preflight_benchmark(panel, config=_config())

    assert report["feasible"] is True
    development = report["summary"]["development"]
    assert development["feature_coverage"]["momentum"] == pytest.approx(5 / 6)
    for fold in report["folds"]:
        training = fold["training"]
        assert training["feature_only_row_count"] > 0
        assert training["row_count"] > training["supervised_row_count"]
    evaluation = report["folds"][0]["evaluation"]
    assert evaluation["target_row_count"] == evaluation["date_count"] * 5
    assert evaluation["realized_return_row_count"] == evaluation["date_count"] * 5


@pytest.mark.parametrize(
    "change", ["missing_feature", "bad_return", "duplicate", "empty"]
)
def test_invalid_panels_return_actionable_failure(change: str) -> None:
    panel = _panel()
    if change == "missing_feature":
        panel = panel.drop(columns="momentum")
    elif change == "bad_return":
        panel = panel.astype({"realized": object})
        panel.loc[0, "realized"] = "not a return"
    elif change == "duplicate":
        panel = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    else:
        panel = panel.iloc[0:0]

    report = preflight_benchmark(panel, config=_config())

    assert report["feasible"] is False
    assert report["errors"][0]["scope"] == "panel"
    assert report["errors"][0]["message"]
    json.dumps(report, allow_nan=False)


def test_insufficient_outer_dates_returns_split_failure() -> None:
    report = preflight_benchmark(_panel().iloc[:90], config=_config())

    assert report["feasible"] is False
    assert report["errors"][0]["scope"] == "split_plan"


def test_inner_split_failure_is_found_even_when_outer_splits_exist() -> None:
    config = _config()
    config = replace(config, split=replace(config.split, inner_min_train_date_count=24))

    report = preflight_benchmark(_panel(), config=config)

    assert report["feasible"] is False
    assert any(
        error["scope"] == "development_fold_1.inner" for error in report["errors"]
    )
    assert report["folds"][-1]["scope"] == "final_training"
    assert len(report["folds"][-1]["inner_folds"]) == 2


def test_final_training_inner_split_feasibility_is_also_checked() -> None:
    config = _config()
    config = replace(config, split=replace(config.split, inner_min_train_date_count=35))

    report = preflight_benchmark(_panel(), config=config)

    assert report["feasible"] is False
    assert any(error["scope"] == "final_training.inner" for error in report["errors"])


def test_hac_warnings_are_bounds_not_inference_results() -> None:
    report = preflight_benchmark(_panel(), config=replace(_config(), hac_lags=19))

    assert report["feasible"] is True
    assert any(
        warning["code"] == "hac_lag_truncation" for warning in report["warnings"]
    )
    assert any(warning["code"] == "short_hac_period" for warning in report["warnings"])
    hac = report["folds"][0]["evaluation"]["hac"]
    assert hac["requested_lags"] == 19
    assert hac["maximum_effective_lags"] == 3
    assert hac["eligible_date_count_upper_bound"] == 4
    scheduled = report["summary"]["hac"]
    assert scheduled["locked_scheduled_date_count"] == 4
    assert scheduled["locked_maximum_effective_lags"] == 3
    assert any(
        warning["scope"] == "locked_schedule" and warning["code"] == "short_hac_period"
        for warning in report["warnings"]
    )
    assert "t_stat" not in json.dumps(report)
    assert "p_value" not in json.dumps(report)


def test_wrong_config_type_is_a_programming_error() -> None:
    with pytest.raises(ValueError, match="BenchmarkConfig"):
        preflight_benchmark(_panel(), config={})


@pytest.mark.parametrize("change", ["wrong_type", "duplicate_columns", "no_realized"])
def test_unsupported_panel_structures_are_reported(change: str) -> None:
    panel = _panel()
    if change == "wrong_type":
        panel = []
    elif change == "duplicate_columns":
        panel = pd.concat([panel, panel[["momentum"]]], axis=1)
    else:
        panel = panel.drop(columns="realized")

    report = preflight_benchmark(panel, config=_config())

    assert report["feasible"] is False
    assert report["errors"][0]["scope"] == "panel"


def test_low_coverage_and_unavailable_return_dates_are_only_structural_warnings() -> (
    None
):
    panel = _panel()
    panel["momentum"] = float("nan")
    panel["realized"] = float("nan")

    report = preflight_benchmark(panel, config=_config())

    assert report["feasible"] is True
    warning_codes = {warning["code"] for warning in report["warnings"]}
    assert {"low_feature_coverage", "no_return_evaluable_dates"} <= warning_codes


def test_sparse_development_labels_warn_that_hac_is_unavailable() -> None:
    panel = _panel()
    last_development_date = panel["as_of_date"].drop_duplicates().iloc[-5]
    panel.loc[
        (panel["as_of_date"] <= last_development_date) & (panel["symbol"] != "S0"),
        "target",
    ] = float("nan")

    report = preflight_benchmark(panel, config=_config())

    assert report["feasible"] is True
    assert any(
        warning["code"] == "insufficient_hac_dates" for warning in report["warnings"]
    )
    assert report["folds"][0]["evaluation"]["hac"]["maximum_effective_lags"] is None
