from __future__ import annotations

import pandas as pd

from quant_metric_research.benchmark_metrics import (
    choose_frozen_model,
    evaluate_prediction_frame,
)


def _predictions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date_index, date in enumerate(pd.bdate_range("2025-01-02", periods=2)):
        for symbol_index in range(4):
            target = float(symbol_index)
            for model, score in (
                ("ridge", target),
                ("hist_gradient_boosting", target + 0.01 * date_index),
                (
                    "equal_weight_rank",
                    None if symbol_index == 0 else target,
                ),
            ):
                rows.append(
                    {
                        "phase": "development",
                        "fold": 1,
                        "as_of_date": date,
                        "symbol": f"S{symbol_index}",
                        "model": model,
                        "score": score,
                        "target": target,
                        "realized_return": target,
                    }
                )
    return pd.DataFrame(rows)


def test_prediction_evaluation_reports_native_and_common_intersections() -> None:
    result = evaluate_prediction_frame(
        _predictions(),
        min_cross_section=3,
        quantiles=3,
        hac_lags=1,
        primary_models=("ridge", "equal_weight_rank"),
    )

    daily = result.daily_metrics
    ridge_native = daily.loc[
        (daily["model"] == "ridge") & (daily["evaluation_scope"] == "native")
    ]
    ridge_common = daily.loc[
        (daily["model"] == "ridge") & (daily["evaluation_scope"] == "common")
    ]

    assert set(ridge_native["evaluation_count"]) == {4}
    assert set(ridge_common["evaluation_count"]) == {3}
    assert (ridge_native["rank_ic"] == 1.0).all()
    assert (ridge_common["rank_ic"] == 1.0).all()
    assert not result.fold_metrics.empty
    assert not result.summary.empty


def test_rank_ic_and_spread_use_separate_complete_case_samples() -> None:
    predictions = _predictions()
    missing_realized = (
        predictions["as_of_date"] == predictions["as_of_date"].min()
    ) & (predictions["symbol"] == "S3")
    predictions.loc[missing_realized, "realized_return"] = float("nan")

    result = evaluate_prediction_frame(
        predictions,
        min_cross_section=3,
        quantiles=3,
        hac_lags=1,
        primary_models=("ridge", "equal_weight_rank"),
    )

    row = result.daily_metrics.loc[
        (result.daily_metrics["as_of_date"] == predictions["as_of_date"].min())
        & (result.daily_metrics["model"] == "ridge")
        & (result.daily_metrics["evaluation_scope"] == "native")
    ].iloc[0]
    assert row["rank_ic_count"] == 4
    assert row["spread_count"] == 3
    assert row["eligible_target_count"] == 4
    assert row["eligible_realized_return_count"] == 3
    assert row["rank_ic_coverage"] == 1.0
    assert row["spread_coverage"] == 1.0
    assert row["rank_ic"] == 1.0
    assert pd.notna(row["spread"])


def test_frozen_model_uses_development_common_rank_ic_and_simple_tie_break() -> None:
    result = evaluate_prediction_frame(
        _predictions(),
        min_cross_section=3,
        quantiles=3,
        hac_lags=1,
        primary_models=(
            "ridge",
            "hist_gradient_boosting",
            "equal_weight_rank",
        ),
    )

    selected = choose_frozen_model(
        result.summary,
        model_families=("ridge", "hist_gradient_boosting"),
    )

    assert selected == "ridge"
