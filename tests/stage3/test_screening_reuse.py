from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_metric_research import benchmark, benchmark_tuning, screening
from quant_metric_research.benchmark_config import BenchmarkConfig, NestedSplitConfig
from quant_metric_research.experiment_workflow import _execute_benchmark
from quant_metric_research.signals import SPREAD_COLUMNS, compute_quantile_spreads


@pytest.fixture()
def reuse_config() -> BenchmarkConfig:
    return BenchmarkConfig(
        feature_columns=("signal", "clone", "other", "constant"),
        target_column="target",
        realized_return_column="realized_return",
        split=NestedSplitConfig(
            final_test_date_count=3,
            outer_n_splits=2,
            outer_test_date_count=2,
            outer_min_train_date_count=10,
            inner_n_splits=2,
            inner_validation_date_count=2,
            inner_min_train_date_count=4,
        ),
        min_cross_section=4,
        quantiles=3,
        hac_lags=1,
        ridge_alphas=(0.1, 1.0),
        hist_max_iter=5,
        hist_min_samples_leaf=3,
        random_seed=17,
    )


@pytest.fixture()
def reuse_panel() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "as_of_date": date,
                "symbol": f"S{stock}",
                "feature_available_at": date,
                "label_end_date": date + pd.offsets.BDay(2),
                "signal": float(stock),
                "clone": float(stock * 100 + day),
                "other": float((stock * 3 + day) % 7),
                "constant": 1.0,
                "target": 0.02 * stock + 0.003 * ((stock * 3 + day) % 7),
                "realized_return": 0.02 * stock,
            }
            for day, date in enumerate(pd.bdate_range("2025-01-02", periods=30))
            for stock in range(6)
        ]
    )


def _screen_kwargs(panel, config):
    return {
        "feature_columns": config.feature_columns,
        "target_column": config.target_column,
        "train_end_date": panel["as_of_date"].max(),
        "min_cross_section": config.min_cross_section,
        "minimum_coverage": config.minimum_coverage,
        "redundancy_threshold": config.redundancy_threshold,
        "hac_lags": config.hac_lags,
        "quantiles": config.quantiles,
    }


def _full_screen(training, *, config):
    return screening.fit_metric_screen(training, **_screen_kwargs(training, config))


def _reference_evaluation_block(training, evaluation, **kwargs):
    """Previous orchestration: prepare and screen again for each model family."""
    config = kwargs["config"]
    phase, fold = kwargs["phase"], kwargs["fold"]
    screen = _full_screen(training, config=config)
    prediction_kwargs = {
        "screen": screen,
        "config": config,
        "phase": phase,
        "fold": fold,
        "evaluation_start": kwargs["evaluation_start"],
    }
    predictions = [
        benchmark._baseline_predictions(training, evaluation, **prediction_kwargs)
    ]
    trials = []
    screens = [
        benchmark_tuning.screen_records(
            screen, phase=phase, outer_fold=fold, inner_fold="final_fit"
        ).assign(family="all", fit_kind="final_fit")
    ]
    for family in kwargs["families"]:
        tuned = benchmark_tuning.tune_model_family(
            training, config=config, family=family, phase=phase, outer_fold=fold
        )
        predictions.append(
            benchmark._model_predictions(
                training,
                evaluation,
                candidate=tuned.selected_candidate,
                **prediction_kwargs,
            )
        )
        trials.append(tuned.trials)
        screens.append(tuned.screening.assign(family=family, fit_kind="inner_tuning"))
    return tuple(
        pd.concat(frames, ignore_index=True)
        for frames in (predictions, trials, screens)
    )


def test_stage3_skips_unused_training_spreads(
    reuse_panel, reuse_config, monkeypatch
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("Stage 3 must not calculate unused training spreads.")

    monkeypatch.setattr(screening, "compute_quantile_spreads", forbidden)
    screen = benchmark_tuning.fit_fold_screen(reuse_panel, config=reuse_config)
    assert screen.quantile_spreads.empty
    assert tuple(screen.quantile_spreads.columns) == SPREAD_COLUMNS
    result = benchmark.run_stage3_benchmark(reuse_panel, config=reuse_config)
    assert result.tuning_trials["validation_mean_spread"].notna().all()
    assert result.daily_metrics["spread"].notna().any()


def test_default_screen_retains_exact_spreads_and_selection(
    reuse_panel, reuse_config
) -> None:
    original = reuse_panel.copy(deep=True)
    kwargs = _screen_kwargs(reuse_panel, reuse_config)
    full = screening.fit_metric_screen(reuse_panel, **kwargs)
    fast = screening.fit_metric_screen(
        reuse_panel, **kwargs, include_quantile_spreads=False
    )
    expected_spreads = compute_quantile_spreads(
        reuse_panel,
        feature_columns=reuse_config.feature_columns,
        target_column=reuse_config.target_column,
        min_cross_section=reuse_config.min_cross_section,
        quantiles=reuse_config.quantiles,
    )
    pd.testing.assert_frame_equal(
        full.quantile_spreads, expected_spreads, check_exact=True
    )
    assert not full.quantile_spreads.empty
    for name in ("quality", "daily_rank_ic", "ic_summary", "redundancy"):
        pd.testing.assert_frame_equal(getattr(full, name), getattr(fast, name))
    for name in (
        "selected_features",
        "dropped_features",
        "fitted_through",
        "training_row_count",
    ):
        assert getattr(full, name) == getattr(fast, name)
    pd.testing.assert_frame_equal(reuse_panel, original)


@pytest.mark.parametrize("quantiles", [True, 1, 1.5, "3", None])
@pytest.mark.parametrize("include_spreads", [False, True])
def test_optional_spreads_preserve_invalid_quantile_errors(
    reuse_panel, reuse_config, quantiles, include_spreads
) -> None:
    kwargs = {**_screen_kwargs(reuse_panel, reuse_config), "quantiles": quantiles}
    with pytest.raises(ValueError, match="quantiles must be an integer of at least 2"):
        screening.fit_metric_screen(
            reuse_panel, **kwargs, include_quantile_spreads=include_spreads
        )


def test_inner_screens_are_fitted_once_per_outer_block(
    reuse_panel, reuse_config, monkeypatch
) -> None:
    calls = []
    original_fit = benchmark_tuning.fit_fold_screen

    def capture(training, *, config):
        calls.append(training.copy(deep=True))
        return original_fit(training, config=config)

    monkeypatch.setattr(benchmark_tuning, "fit_fold_screen", capture)
    monkeypatch.setattr(benchmark, "fit_fold_screen", capture)
    result = benchmark.run_stage3_benchmark(reuse_panel, config=reuse_config)

    assert len(calls) == 6  # Two outer blocks, each with final + two inner screens.
    assert all(len({len(frame) for frame in calls[i : i + 3]}) == 3 for i in (0, 3))
    inner = result.screening_by_fold.query("fit_kind == 'inner_tuning'")
    assert len(inner) == 2 * 2 * 2 * len(reuse_config.feature_columns)
    assert set(inner["family"]) == set(reuse_config.model_families)


@pytest.mark.parametrize("edge_cases", [False, True])
def test_reused_screening_matches_previous_complete_benchmark_exactly(
    reuse_panel, reuse_config, monkeypatch, edge_cases
) -> None:
    panel = reuse_panel.copy(deep=True)
    config = replace(reuse_config, include_pca_model=True)
    if edge_cases:
        # Keep unlabeled features, unequal outcome masks, sparse inputs and ties.
        panel.loc[panel["symbol"] == "S1", "target"] = np.nan
        panel.loc[panel["symbol"] == "S2", "realized_return"] = np.nan
        panel.loc[panel.index[0], "label_end_date"] = panel["label_end_date"].max()
        missing_features = (panel["symbol"] == "S3") & (panel.index % 4 == 1)
        panel.loc[missing_features, list(config.feature_columns)] = np.nan
        config = replace(config, model_families=tuple(reversed(config.model_families)))
    original = panel.copy(deep=True)
    # Synthetic final-block coverage also checks the single-family reuse path.
    arguments = {
        "config": config,
        "evaluate_lockbox": edge_cases,
        "before_lockbox": lambda manifest, family: None,
    }
    actual = _execute_benchmark(panel, **arguments)
    with monkeypatch.context() as reference_patch:
        reference_patch.setattr(
            benchmark, "_fit_evaluation_block", _reference_evaluation_block
        )
        reference_patch.setattr(benchmark_tuning, "fit_fold_screen", _full_screen)
        expected = _execute_benchmark(panel, **arguments)

    for name in (
        "fold_assignments",
        "predictions",
        "daily_metrics",
        "fold_metrics",
        "tuning_trials",
        "screening_by_fold",
        "summary",
    ):
        pd.testing.assert_frame_equal(
            getattr(actual, name), getattr(expected, name), check_exact=True
        )
    for name in ("acceptance", "data_gate"):
        assert getattr(actual, name) == getattr(expected, name)
    source_dependent = {"source_fingerprint", "run_fingerprint"}
    assert {
        key: value
        for key, value in actual.manifest.items()
        if key not in source_dependent
    } == {
        key: value
        for key, value in expected.manifest.items()
        if key not in source_dependent
    }
    pd.testing.assert_frame_equal(panel, original)


def test_inner_preparation_preserves_training_masks_and_purging(
    reuse_panel, reuse_config
) -> None:
    changed = reuse_panel.copy(deep=True)
    changed.loc[changed["symbol"] == "S1", "target"] = np.nan
    changed.loc[changed.index[0], "label_end_date"] = changed["label_end_date"].max()
    original = changed.copy(deep=True)

    for training, validation, screen in benchmark_tuning._inner_folds(
        changed, config=reuse_config
    ):
        assert set(training["symbol"]) == set(changed["symbol"])
        assert training["label_end_date"].max() < validation["as_of_date"].min()
        unlabeled = training.loc[training["symbol"] == "S1"]
        assert unlabeled["target"].isna().all()
        # Mature labels with missing targets retain separate realized outcomes.
        assert unlabeled[["realized_return", "label_end_date"]].notna().all().all()
        assert unlabeled["signal"].notna().all()
        purged = training.loc[
            (training["as_of_date"] == changed["as_of_date"].min())
            & (training["symbol"] == "S0")
        ]
        outcome_columns = ["target", "realized_return", "label_end_date"]
        assert purged[outcome_columns].isna().all().all()
        assert screen.training_row_count == len(training)
    pd.testing.assert_frame_equal(changed, original)
