from __future__ import annotations

import numpy as np
import pandas as pd

from quant_metric_research.reduction import fit_pca_baseline


def _panel() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=8)
    rows = []
    for date_index, date in enumerate(dates):
        for symbol_index in range(4):
            rows.append(
                {
                    "as_of_date": date,
                    "symbol": f"S{symbol_index}",
                    "a": float(symbol_index + date_index),
                    "b": float(2 * symbol_index + date_index),
                    "c": np.nan
                    if symbol_index == 0
                    else float(symbol_index - date_index),
                }
            )
    return pd.DataFrame(rows)


def test_pca_baseline_is_fitted_only_on_training_period() -> None:
    panel = _panel()
    train_end = panel["as_of_date"].sort_values().unique()[4]
    original = fit_pca_baseline(
        panel,
        feature_columns=("a", "b", "c"),
        train_end_date=train_end,
        variance_to_keep=0.9,
    )

    changed = panel.copy(deep=True)
    changed.loc[changed["as_of_date"] > train_end, ["a", "b", "c"]] *= -100.0
    rebuilt = fit_pca_baseline(
        changed,
        feature_columns=("a", "b", "c"),
        train_end_date=train_end,
        variance_to_keep=0.9,
    )

    np.testing.assert_allclose(original.components, rebuilt.components)
    np.testing.assert_allclose(
        original.explained_variance_ratio,
        rebuilt.explained_variance_ratio,
    )


def test_pca_transform_preserves_keys_and_returns_components() -> None:
    panel = _panel()
    fitted = fit_pca_baseline(
        panel,
        feature_columns=("a", "b", "c"),
        train_end_date="2025-01-08",
        variance_to_keep=0.9,
    )

    transformed = fitted.transform(panel)

    assert transformed[["as_of_date", "symbol"]].equals(
        panel[["as_of_date", "symbol"]].reset_index(drop=True)
    )
    assert any(column.startswith("pc_") for column in transformed.columns)
    assert fitted.fitted_through == pd.Timestamp("2025-01-08")
