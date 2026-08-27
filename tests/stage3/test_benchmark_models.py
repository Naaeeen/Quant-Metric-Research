from __future__ import annotations

import pandas as pd

from quant_metric_research.benchmark_config import BenchmarkConfig, NestedSplitConfig
from quant_metric_research.benchmark_models import candidate_specs, fit_candidate


def _config() -> BenchmarkConfig:
    return BenchmarkConfig(
        feature_columns=("signal", "other"),
        split=NestedSplitConfig(
            final_test_date_count=2,
            outer_n_splits=1,
            outer_test_date_count=2,
            outer_min_train_date_count=4,
            inner_n_splits=1,
            inner_validation_date_count=2,
            inner_min_train_date_count=2,
        ),
        min_cross_section=3,
        quantiles=3,
        hac_lags=1,
        ridge_alphas=(0.1, 1.0),
        hist_learning_rates=(0.05,),
        hist_max_leaf_nodes=(5, 7),
        hist_l2_regularization=(0.0, 1.0),
        hist_max_iter=20,
        hist_min_samples_leaf=2,
        random_seed=9,
    )


def _training() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=6)
    rows: list[dict[str, object]] = []
    for date_index, date in enumerate(dates):
        for symbol_index in range(5):
            signal = float(symbol_index) + 0.01 * date_index
            rows.append(
                {
                    "as_of_date": date,
                    "symbol": f"S{symbol_index}",
                    "label_end_date": date + pd.offsets.BDay(1),
                    "signal": signal,
                    "other": float((symbol_index + date_index) % 3),
                    "target": 0.5 * signal,
                }
            )
    return pd.DataFrame(rows)


def test_candidate_grid_is_small_explicit_and_deterministic() -> None:
    config = _config()

    ridge = candidate_specs(config, family="ridge")
    hist = candidate_specs(config, family="hist_gradient_boosting")

    assert [candidate.parameters["alpha"] for candidate in ridge] == [0.1, 1.0]
    assert len(hist) == 4
    assert len({candidate.candidate_id for candidate in hist}) == 4


def test_ridge_fit_is_fold_local_and_prediction_uses_cross_sectional_ranks() -> None:
    training = _training()
    candidate = candidate_specs(_config(), family="ridge")[0]
    fitted = fit_candidate(
        training,
        feature_columns=("signal", "other"),
        target_column="target",
        candidate=candidate,
        rank_features=True,
        random_seed=9,
    )
    evaluation = training.loc[training["as_of_date"] == training["as_of_date"].max()]
    scaled = evaluation.assign(signal=evaluation["signal"] * 1000.0)

    pd.testing.assert_series_equal(
        fitted.predict(evaluation),
        fitted.predict(scaled),
    )
    assert fitted.fitted_through == pd.Timestamp(training["as_of_date"].max())
    assert fitted.train_label_end_max == pd.Timestamp(training["label_end_date"].max())


def test_histogram_boosting_disables_internal_random_validation() -> None:
    config = _config()
    candidate = candidate_specs(config, family="hist_gradient_boosting")[0]

    first = fit_candidate(
        _training(),
        feature_columns=("signal", "other"),
        target_column="target",
        candidate=candidate,
        rank_features=True,
        random_seed=config.random_seed,
    )
    second = fit_candidate(
        _training(),
        feature_columns=("signal", "other"),
        target_column="target",
        candidate=candidate,
        rank_features=True,
        random_seed=config.random_seed,
    )

    assert first.model_parameters["early_stopping"] is False
    pd.testing.assert_series_equal(
        first.predict(_training()), second.predict(_training())
    )
