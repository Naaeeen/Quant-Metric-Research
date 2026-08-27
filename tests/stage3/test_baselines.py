from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from quant_metric_research.baselines import (
    FittedNonMLBaselines,
    fit_non_ml_baselines,
    predict_non_ml_baselines,
)


def _training_panel() -> pd.DataFrame:
    dates = pd.to_datetime(["2025-01-02", "2025-01-09", "2025-01-16"])
    rows: list[dict[str, object]] = []
    for date_index, date in enumerate(dates):
        for symbol_index in range(5):
            alpha = float(symbol_index)
            rows.append(
                {
                    "as_of_date": date,
                    "symbol": f"S{symbol_index}",
                    "alpha": alpha,
                    "zeta_inverse": -alpha,
                    "partial": (
                        float((symbol_index + date_index) % 5)
                        if symbol_index != 4
                        else np.nan
                    ),
                    "target": alpha * 0.02 + date_index * 0.001,
                }
            )
    return pd.DataFrame(rows)


def test_fit_learns_per_date_ic_direction_and_deterministic_best_metric() -> None:
    fitted = fit_non_ml_baselines(
        _training_panel(),
        feature_columns=("zeta_inverse", "alpha"),
        target_column="target",
        min_cross_section=4,
    )

    assert isinstance(fitted, FittedNonMLBaselines)
    assert fitted.feature_signs == {"zeta_inverse": -1, "alpha": 1}
    assert fitted.mean_rank_ic["zeta_inverse"] == pytest.approx(-1.0)
    assert fitted.mean_rank_ic["alpha"] == pytest.approx(1.0)
    assert fitted.best_feature == "alpha"
    assert fitted.selected_features == ("zeta_inverse", "alpha")
    assert fitted.fitted_through == pd.Timestamp("2025-01-16")
    assert fitted.training_row_count == 15


def test_fitted_baseline_state_is_immutable() -> None:
    fitted = fit_non_ml_baselines(
        _training_panel(),
        feature_columns=("alpha",),
        target_column="target",
        min_cross_section=4,
    )

    with pytest.raises(FrozenInstanceError):
        fitted.best_feature = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        fitted.feature_signs["alpha"] = -1  # type: ignore[index]
    with pytest.raises(TypeError):
        fitted.mean_rank_ic["alpha"] = -1.0  # type: ignore[index]


def test_prediction_returns_individual_best_and_equal_weight_models() -> None:
    fitted = fit_non_ml_baselines(
        _training_panel(),
        feature_columns=("alpha", "zeta_inverse", "partial"),
        target_column="target",
        min_cross_section=4,
    )
    scoring = pd.DataFrame(
        {
            "as_of_date": pd.to_datetime(["2025-02-03"] * 5),
            "symbol": ["A", "B", "C", "D", "E"],
            "alpha": [0.0, 1.0, 2.0, 3.0, 4.0],
            "zeta_inverse": [0.0, -1.0, -2.0, -3.0, -4.0],
            "partial": [4.0, 3.0, np.nan, 1.0, 0.0],
        }
    )
    unchanged = scoring.copy(deep=True)

    predictions = predict_non_ml_baselines(fitted, scoring)

    assert predictions.columns.tolist() == [
        "as_of_date",
        "symbol",
        "model",
        "score",
        "baseline_feature",
        "feature_count",
    ]
    assert set(predictions["model"]) == {
        "metric:alpha",
        "metric:zeta_inverse",
        "metric:partial",
        "best_metric",
        "equal_weight_rank",
    }
    assert predictions.shape[0] == scoring.shape[0] * 5
    assert predictions["score"].dropna().between(0.0, 1.0).all()
    assert (
        predictions.loc[predictions["model"] == "best_metric", "baseline_feature"]
        == "alpha"
    ).all()
    composite_counts = predictions.loc[
        predictions["model"] == "equal_weight_rank",
        ["symbol", "feature_count"],
    ].set_index("symbol")["feature_count"]
    assert composite_counts.to_dict() == {"A": 3, "B": 3, "C": 2, "D": 3, "E": 3}

    symbol_c = predictions.loc[
        (predictions["symbol"] == "C") & (predictions["model"] == "equal_weight_rank"),
        "score",
    ].item()
    alpha_c = predictions.loc[
        (predictions["symbol"] == "C") & (predictions["model"] == "metric:alpha"),
        "score",
    ].item()
    inverse_c = predictions.loc[
        (predictions["symbol"] == "C")
        & (predictions["model"] == "metric:zeta_inverse"),
        "score",
    ].item()
    assert alpha_c == pytest.approx(0.6)
    assert inverse_c == pytest.approx(0.4)
    assert symbol_c == pytest.approx((alpha_c + inverse_c) / 2.0)
    pd.testing.assert_frame_equal(scoring, unchanged)


def test_future_targets_cannot_change_fitted_state_or_predictions() -> None:
    training = _training_panel()
    fitted = fit_non_ml_baselines(
        training,
        feature_columns=("alpha", "zeta_inverse"),
        target_column="target",
        min_cross_section=4,
    )
    future = training.loc[training["as_of_date"] == training["as_of_date"].max()].copy(
        deep=True
    )
    future["as_of_date"] = pd.Timestamp("2025-03-03")

    original = predict_non_ml_baselines(fitted, future)
    changed = future.copy(deep=True)
    changed["target"] *= -1_000_000.0
    changed["zeta_inverse"] += 100.0
    changed_predictions = predict_non_ml_baselines(fitted, changed)

    assert fitted.feature_signs == {"alpha": 1, "zeta_inverse": -1}
    assert fitted.best_feature == "alpha"
    pd.testing.assert_series_equal(
        original.loc[original["model"] == "metric:alpha", "score"].reset_index(
            drop=True
        ),
        changed_predictions.loc[
            changed_predictions["model"] == "metric:alpha", "score"
        ].reset_index(drop=True),
    )


@pytest.mark.parametrize(
    ("feature_columns", "expected_message"),
    [
        ((), "at least one feature"),
        (("alpha", "alpha"), "must be unique"),
        (("constant",), "cannot learn a non-zero direction"),
    ],
)
def test_fit_rejects_invalid_or_unlearnable_feature_sets(
    feature_columns: tuple[str, ...],
    expected_message: str,
) -> None:
    panel = _training_panel().assign(constant=1.0)

    with pytest.raises(ValueError, match=expected_message):
        fit_non_ml_baselines(
            panel,
            feature_columns=feature_columns,
            target_column="target",
            min_cross_section=4,
        )


def test_fit_and_predict_validate_required_columns_and_row_keys() -> None:
    panel = _training_panel()
    with pytest.raises(ValueError, match="Missing baseline columns: target"):
        fit_non_ml_baselines(
            panel.drop(columns="target"),
            feature_columns=("alpha",),
            target_column="target",
            min_cross_section=4,
        )

    with pytest.raises(ValueError, match="target_column must not be a feature"):
        fit_non_ml_baselines(
            panel,
            feature_columns=("alpha", "target"),
            target_column="target",
            min_cross_section=4,
        )

    duplicated_training = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate as_of_date-symbol.*training"):
        fit_non_ml_baselines(
            duplicated_training,
            feature_columns=("alpha",),
            target_column="target",
            min_cross_section=4,
        )

    fitted = fit_non_ml_baselines(
        panel,
        feature_columns=("alpha",),
        target_column="target",
        min_cross_section=4,
    )
    scoring = panel.loc[:, ["as_of_date", "symbol", "alpha"]].head(2)
    duplicated = pd.concat([scoring, scoring.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate as_of_date-symbol"):
        predict_non_ml_baselines(fitted, duplicated)

    with pytest.raises(ValueError, match="Missing prediction columns: symbol"):
        predict_non_ml_baselines(fitted, scoring.drop(columns="symbol"))


def test_invalid_minimum_cross_section_is_rejected() -> None:
    with pytest.raises(ValueError, match="integer of at least 2"):
        fit_non_ml_baselines(
            _training_panel(),
            feature_columns=("alpha",),
            target_column="target",
            min_cross_section=True,  # type: ignore[arg-type]
        )
