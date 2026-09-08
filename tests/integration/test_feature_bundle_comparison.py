"""Actual synthetic producer compatibility; no empirical or final-test runs."""

import builtins
import io
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _enriched_panel(qmr):
    # Same deterministic price process as test_factor_panel_contract.inputs.
    dates = pd.bdate_range("2020-01-01", periods=320)
    position = np.arange(len(dates))
    prices = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": symbol,
                    "adjusted_close": 100
                    * np.exp(
                        0.0005 * position
                        + (number + 1) * 0.002 * np.sin(position / (number + 3))
                    ),
                }
            )
            for number, symbol in enumerate(("BENCH", "A", "B", "C", "D", "E", "F"))
        ],
        ignore_index=True,
    )
    memberships = pd.DataFrame(
        [
            {
                "universe_id": "INVENTED",
                "symbol": symbol,
                "effective_from": dates[0],
                "effective_to": None,
                "source": "invented fixture",
            }
            for symbol in ("A", "B", "C", "D", "E", "F", "MISSING")
        ]
    )
    panel_config = qmr.PanelConfig(
        dataset_version="synthetic-comparison-v1",
        universe_id="INVENTED",
        benchmark_symbol="BENCH",
        target_horizon_sessions=2,
        entry_lag_sessions=1,
    )
    return qmr.build_factor_panel(
        prices,
        memberships,
        as_of_dates=list(dates[252:276]),
        config=panel_config,
        factor_names=qmr.FEATURE_BUNDLES["legacy10_plus_price3_v1"].price_factor_names,
    )


def _development_pair(qmr):
    panel = _enriched_panel(qmr)
    original = panel.copy(deep=True)
    base = qmr.BenchmarkConfig(
        feature_columns=qmr.FEATURE_BUNDLES["legacy10_v1"].feature_columns,
        target_column="forward_excess_return",
        realized_return_column="forward_return",
        split=qmr.NestedSplitConfig(3, 1, 3, 8, 1, 2, 4),
        min_cross_section=4,
        quantiles=2,
        hac_lags=1,
        model_families=("ridge",),
        ridge_alphas=(1.0,),
    )
    expanded = replace(
        base,
        feature_columns=qmr.FEATURE_BUNDLES["legacy10_plus_price3_v1"].feature_columns,
    )
    runs = tuple(
        qmr.run_stage3_benchmark(panel, config=config) for config in (base, expanded)
    )
    assert len(panel) == 168 and panel["as_of_date"].nunique() == 24
    pd.testing.assert_frame_equal(panel, original)
    return runs


def _selected_scores(legacy, candidate):
    columns = [
        "phase",
        "fold",
        "as_of_date",
        "symbol",
        "model",
        "score",
        "target",
        "realized_return",
    ]
    return pd.concat(
        [
            run.predictions.loc[run.predictions["model"] == source, columns].assign(
                model=role
            )
            for run, source, role in (
                (legacy, "equal_weight_rank", "legacy_equal_rank"),
                (legacy, "ridge", "legacy_model"),
                (candidate, "ridge", "candidate_model"),
            )
        ],
        ignore_index=True,
    )


def _forbid_comparator_side_effects(guard, qmr):
    from sklearn.pipeline import Pipeline

    from quant_metric_research import (
        baselines,
        benchmark,
        benchmark_io,
        benchmark_models,
        benchmark_tuning,
    )
    from quant_metric_research import (
        io as artifact_io,
    )

    def forbidden(*args, **kwargs):
        pytest.fail(
            "Comparison must not fit, load input artifacts, or access a registry."
        )

    for owner, name in (
        (Pipeline, "fit"),
        (benchmark_models, "fit_candidate"),
        (benchmark, "fit_candidate"),
        (benchmark_tuning, "fit_candidate"),
        (baselines, "fit_non_ml_baselines"),
        (benchmark, "fit_non_ml_baselines"),
        (qmr.ExperimentRegistry, "__init__"),
        (qmr.ExperimentRegistry, "get_run"),
        (qmr.ExperimentRegistry, "list_runs"),
        (sqlite3, "connect"),
        (artifact_io, "read_table"),
        (artifact_io, "read_json_object"),
        (pd, "read_csv"),
        (pd, "read_parquet"),
        (pd, "read_pickle"),
        (benchmark_io, "write_benchmark_run"),
        (qmr, "write_benchmark_run"),
    ):
        guard.setattr(owner, name, forbidden)

    # Reporter identity may read installed source; caller artifacts stay out of scope.
    package_directory = Path(qmr.__file__).resolve().parent
    for owner in (builtins, io):
        original_open = owner.open

        def source_only_open(file, mode="r", *args, _open=original_open, **kwargs):
            path = Path(file).resolve()
            if (
                path.parent != package_directory
                or path.suffix != ".py"
                or mode not in {"r", "rb"}
            ):
                pytest.fail(
                    "Comparison attempted file access outside installed source."
                )
            return _open(file, mode, *args, **kwargs)

        guard.setattr(owner, "open", source_only_open)


def test_real_development_pair_compares_saved_scores_without_fitting(monkeypatch):
    import quant_metric_research as qmr
    from quant_metric_research import (
        FeatureBundleComparison,
        benchmark_comparison,
        compare_feature_bundles,
    )

    legacy, candidate = _development_pair(qmr)
    for run in (legacy, candidate):
        assert run.manifest["artifact_schema_version"] == "6"
        assert run.manifest["package_version"] == "0.13.0"
        assert run.manifest["execution_mode"] == "development"
        assert run.manifest["experiment"]["registered"] is False
        assert set(run.predictions["phase"]) == {"development"}
    for name in ("panel_fingerprint", "source_fingerprint", "evaluation_schedule"):
        assert legacy.manifest[name] == candidate.manifest[name]
    for name in ("run_fingerprint", "model_input_fingerprint"):
        assert legacy.manifest[name] != candidate.manifest[name]
    configurations = [
        dict(run.manifest["configuration"]) for run in (legacy, candidate)
    ]
    assert {
        key: value
        for key, value in configurations[0].items()
        if key != "feature_columns"
    } == {
        key: value
        for key, value in configurations[1].items()
        if key != "feature_columns"
    }

    frame_names = (
        "fold_assignments",
        "predictions",
        "daily_metrics",
        "fold_metrics",
        "tuning_trials",
        "screening_by_fold",
        "summary",
    )
    snapshots = [
        {name: getattr(run, name).copy(deep=True) for name in frame_names}
        for run in (legacy, candidate)
    ]
    manifests = [
        json.dumps(dict(run.manifest), default=dict, sort_keys=True)
        for run in (legacy, candidate)
    ]
    expected_scores = _selected_scores(legacy, candidate)
    captured = []
    original_evaluate = benchmark_comparison.evaluate_prediction_frame

    def capture_scores(predictions, **kwargs):
        captured.append(predictions.copy(deep=True))
        assert tuple(kwargs["primary_models"]) == (
            "legacy_equal_rank",
            "legacy_model",
            "candidate_model",
        )
        assert (
            tuple(kwargs["expected_dates_by_phase"]["development"])
            == legacy.manifest["evaluation_schedule"]["dates_by_phase"]["development"]
        )
        return original_evaluate(predictions, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(benchmark_comparison, "evaluate_prediction_frame", capture_scores)
        _forbid_comparator_side_effects(guard, qmr)
        report = compare_feature_bundles(legacy, candidate, model_family="ridge")

    assert isinstance(report, FeatureBundleComparison)
    assert len(captured) == 1
    keys = ["phase", "fold", "as_of_date", "symbol", "model"]
    pd.testing.assert_frame_equal(
        captured[0]
        .loc[:, expected_scores.columns]
        .sort_values(keys)
        .reset_index(drop=True),
        expected_scores.sort_values(keys).reset_index(drop=True),
        check_dtype=False,
        check_exact=True,
    )
    missing = captured[0].loc[captured[0]["symbol"] == "MISSING"]
    assert len(missing) == 9  # Three development dates, three saved-score roles.
    assert missing[["score", "target", "realized_return"]].isna().all().all()
    observed = captured[0].dropna(subset=["target", "realized_return"])
    assert (observed["target"] != observed["realized_return"]).any()
    daily = report.daily_metrics
    assert set(daily["model"]) == {
        "legacy_equal_rank",
        "legacy_model",
        "candidate_model",
    }
    assert set(daily["evaluation_scope"]) == {"native", "common"}
    assert daily["scoring_universe_count"].eq(7).all()
    assert daily["scored_count"].eq(6).all()
    assert daily["score_coverage"].tolist() == pytest.approx([6 / 7] * len(daily))
    assert daily["eligible_target_count"].eq(6).all()
    assert daily["eligible_realized_return_count"].eq(6).all()
    assert daily["rank_ic_coverage"].eq(1.0).all()
    assert daily["spread_coverage"].eq(1.0).all()
    assert not {"newey_west_t_stat", "p_value", "bh_q_value", "inference_status"} & set(
        report.summary
    )
    assert set(report.delta_summary["contrast"]) == {
        "candidate_model_minus_legacy_model",
        "candidate_model_minus_legacy_equal_rank",
    }

    assert report.metadata["claim_scope"] == "development_descriptive_comparison"
    assert report.metadata["caller_owned_inputs"] is True
    for key in (
        "authenticity_verified",
        "history_verified",
        "final_outcomes_evaluated",
        "inference_reported",
    ):
        assert report.metadata[key] is False
    assert report.metadata["model_family"] == "ridge"
    assert report.metadata["comparison_identity"]["package_version"] == "0.13.0"
    with pytest.raises(TypeError):
        report.metadata["history_verified"] = True
    with pytest.raises(TypeError):
        report.metadata["comparison_identity"]["package_version"] = "changed"
    for name in (
        "daily_metrics",
        "fold_metrics",
        "summary",
        "daily_deltas",
        "delta_summary",
    ):
        original_table = getattr(report, name)
        editable = getattr(report, name)
        column = "contrast" if "contrast" in editable else "model"
        editable.loc[editable.index[0], column] = "changed"
        pd.testing.assert_frame_equal(getattr(report, name), original_table)
    for index, run in enumerate((legacy, candidate)):
        for name in frame_names:
            pd.testing.assert_frame_equal(getattr(run, name), snapshots[index][name])
        assert (
            json.dumps(dict(run.manifest), default=dict, sort_keys=True)
            == manifests[index]
        )
