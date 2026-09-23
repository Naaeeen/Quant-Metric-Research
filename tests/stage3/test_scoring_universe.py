from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_metric_research import benchmark
from quant_metric_research.benchmark_config import BenchmarkConfig, NestedSplitConfig
from quant_metric_research.benchmark_data import build_stage3_data_plan
from quant_metric_research.benchmark_models import candidate_specs, fit_candidate
from quant_metric_research.benchmark_tuning import _inner_folds


@pytest.fixture()
def scoring_config() -> BenchmarkConfig:
    return BenchmarkConfig(
        feature_columns=("signal", "other"),
        target_column="target",
        realized_return_column="realized_return",
        split=NestedSplitConfig(
            final_test_date_count=3,
            outer_n_splits=1,
            outer_test_date_count=3,
            outer_min_train_date_count=8,
            inner_n_splits=1,
            inner_validation_date_count=2,
            inner_min_train_date_count=4,
        ),
        model_families=("ridge",),
        min_cross_section=3,
        quantiles=3,
        hac_lags=1,
        ridge_alphas=(1.0,),
    )


@pytest.fixture()
def scoring_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=30)
    return pd.DataFrame(
        [
            {
                "as_of_date": date,
                "symbol": f"S{stock}",
                "feature_available_at": date,
                "label_end_date": date + pd.offsets.BDay(2),
                "signal": float(stock),
                "other": float((stock * 3 + day) % 7),
                "target": 0.02 * stock + 0.003 * ((stock * 3 + day) % 7),
                "realized_return": 0.02 * stock,
            }
            for day, date in enumerate(dates)
            for stock in range(6)
        ]
    )


def _plan(panel: pd.DataFrame, config: BenchmarkConfig):
    return build_stage3_data_plan(
        panel,
        feature_columns=config.feature_columns,
        target_column=config.target_column,
        locked_test_date_count=config.split.final_test_date_count,
        locked_min_cross_section=config.min_cross_section,
        n_splits=config.split.outer_n_splits,
        evaluation_date_count=config.split.outer_test_date_count,
        min_train_date_count=config.split.outer_min_train_date_count,
    )


def _scores(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ["as_of_date", "symbol", "model"]
    return predictions.sort_values(keys).loc[:, [*keys, "score"]].reset_index(drop=True)


def test_locked_scores_do_not_depend_on_future_target_availability(
    scoring_panel: pd.DataFrame, scoring_config: BenchmarkConfig
) -> None:
    first_plan = _plan(scoring_panel, scoring_config)
    changed = scoring_panel.copy(deep=True)
    hidden = (changed["as_of_date"] >= first_plan.locked_test.test_start_date) & (
        changed["symbol"] == "S1"
    )
    changed.loc[hidden, "target"] = np.nan

    first, _, _ = benchmark._locked_run(
        first_plan, config=scoring_config, frozen_family="ridge"
    )
    second, _, _ = benchmark._locked_run(
        _plan(changed, scoring_config), config=scoring_config, frozen_family="ridge"
    )

    pd.testing.assert_frame_equal(_scores(first), _scores(second))
    hidden_predictions = second.loc[second["symbol"] == "S1"]
    assert not hidden_predictions.empty
    assert hidden_predictions["target"].isna().all()
    assert hidden_predictions["realized_return"].notna().all()


def test_development_scores_keep_rows_whose_labels_overlap_lockbox(
    scoring_panel: pd.DataFrame, scoring_config: BenchmarkConfig
) -> None:
    first_plan = _plan(scoring_panel, scoring_config)
    changed = scoring_panel.copy(deep=True)
    evaluation_date = first_plan.development_folds[0].evaluation_start_date
    hidden = (changed["as_of_date"] == evaluation_date) & (changed["symbol"] == "S1")
    changed.loc[hidden, "label_end_date"] = first_plan.locked_test.test_start_date

    first, _, _ = benchmark._development_run(first_plan, config=scoring_config)
    second, _, _ = benchmark._development_run(
        _plan(changed, scoring_config), config=scoring_config
    )

    pd.testing.assert_frame_equal(_scores(first), _scores(second))
    hidden_predictions = second.loc[
        (second["as_of_date"] == evaluation_date) & (second["symbol"] == "S1")
    ]
    assert not hidden_predictions.empty
    assert hidden_predictions[["target", "realized_return"]].isna().all().all()


@pytest.mark.parametrize("exclusion", ["missing_target", "label_overlap"])
def test_outer_training_preserves_unlabeled_feature_cross_section(
    scoring_panel: pd.DataFrame,
    scoring_config: BenchmarkConfig,
    monkeypatch: pytest.MonkeyPatch,
    exclusion: str,
) -> None:
    changed = scoring_panel.copy(deep=True)
    first_date = changed["as_of_date"].min()
    hidden = (changed["as_of_date"] == first_date) & (changed["symbol"] == "S1")
    if exclusion == "missing_target":
        changed.loc[hidden, "target"] = np.nan
    else:
        changed.loc[hidden, "label_end_date"] = changed["label_end_date"].max()
    captured: list[pd.DataFrame] = []

    def capture(training, evaluation, **kwargs):
        captured.append(training.copy(deep=True))
        return evaluation.copy(deep=True), pd.DataFrame(), pd.DataFrame()

    monkeypatch.setattr(benchmark, "_fit_evaluation_block", capture)
    benchmark._development_run(_plan(changed, scoring_config), config=scoring_config)

    rows = captured[0].loc[captured[0]["as_of_date"] == first_date]
    assert len(rows) == 6
    assert rows.loc[rows["symbol"] == "S1", "target"].isna().all()
    assert rows.loc[rows["symbol"] == "S1", "realized_return"].isna().all()
    assert rows.loc[rows["symbol"] == "S1", "label_end_date"].isna().all()


def test_inner_training_preserves_purged_rows_for_feature_ranking(
    scoring_panel: pd.DataFrame, scoring_config: BenchmarkConfig
) -> None:
    changed = scoring_panel.copy(deep=True)
    first_date = changed["as_of_date"].min()
    hidden = (changed["as_of_date"] == first_date) & (changed["symbol"] == "S1")
    changed.loc[hidden, "target"] = np.nan
    changed.loc[hidden, "label_end_date"] = pd.NaT

    training, _, _ = _inner_folds(changed, config=scoring_config)[0]
    rows = training.loc[training["as_of_date"] == first_date]
    assert len(rows) == 6
    assert rows.loc[rows["symbol"] == "S1", "target"].isna().all()


def test_training_ranks_complete_universe_before_filtering_supervised_labels(
    scoring_panel: pd.DataFrame,
    scoring_config: BenchmarkConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quant_metric_research.benchmark_models as models

    captured: list[pd.DataFrame] = []

    class CapturingPipeline:
        named_steps = {"model": SimpleNamespace(get_params=lambda **kwargs: {})}

        def fit(self, features, target, **kwargs):
            captured.append(features.copy(deep=True))

    monkeypatch.setattr(
        models, "_build_pipeline", lambda *args, **kw: CapturingPipeline()
    )
    changed = scoring_panel.copy(deep=True)
    changed.loc[changed["symbol"] == "S1", "target"] = np.nan
    fit_candidate(
        changed,
        feature_columns=("signal",),
        target_column="target",
        candidate=candidate_specs(scoring_config, family="ridge")[0],
        rank_features=True,
        random_seed=1,
    )

    first_date = changed["as_of_date"].min()
    retained = (changed["as_of_date"] == first_date) & changed["target"].notna()
    np.testing.assert_allclose(
        captured[0].loc[changed.index[retained], "signal"],
        [1 / 6, 3 / 6, 4 / 6, 5 / 6, 1.0],
    )


@pytest.mark.parametrize("rank_features", [False, True])
def test_model_predict_never_scores_rows_with_no_observed_features(
    scoring_panel: pd.DataFrame,
    scoring_config: BenchmarkConfig,
    rank_features: bool,
) -> None:
    fitted = fit_candidate(
        scoring_panel,
        feature_columns=scoring_config.feature_columns,
        target_column="target",
        candidate=candidate_specs(scoring_config, family="ridge")[0],
        rank_features=rank_features,
        random_seed=1,
    )
    evaluation = scoring_panel.loc[
        scoring_panel["as_of_date"] == scoring_panel["as_of_date"].max()
    ].copy(deep=True)
    evaluation.loc[evaluation.index[0], ["signal", "other"]] = np.nan

    scores = fitted.predict(evaluation)

    assert pd.isna(scores.iloc[0])
    assert scores.iloc[1:].notna().all()


def test_supervised_filter_preserves_row_alignment_with_repeated_index(
    scoring_panel: pd.DataFrame, scoring_config: BenchmarkConfig
) -> None:
    training = scoring_panel.copy(deep=True)
    training.loc[training["symbol"] == "S1", "target"] = np.nan
    repeated = training.set_axis([0] * len(training))
    arguments = {
        "feature_columns": scoring_config.feature_columns,
        "target_column": "target",
        "candidate": candidate_specs(scoring_config, family="ridge")[0],
        "rank_features": True,
        "random_seed": 1,
    }

    expected = fit_candidate(training, **arguments).predict(scoring_panel)
    actual = fit_candidate(repeated, **arguments).predict(scoring_panel)

    pd.testing.assert_series_equal(actual, expected)
