from __future__ import annotations

import pandas as pd
import pytest

from quant_metric_research.splits import build_purged_walk_forward_splits


def _panel() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=10)
    rows = []
    for date_index, as_of in enumerate(dates):
        for symbol in ("AAA", "BBB"):
            rows.append(
                {
                    "as_of_date": as_of,
                    "symbol": symbol,
                    "label_end_date": dates[min(date_index + 2, len(dates) - 1)],
                }
            )
    return pd.DataFrame(rows).sample(frac=1.0, random_state=7)


def test_walk_forward_split_purges_training_labels_overlapping_test() -> None:
    panel = _panel()

    folds = build_purged_walk_forward_splits(
        panel,
        n_splits=2,
        test_date_count=2,
        min_train_date_count=3,
    )

    assert len(folds) == 2
    for fold in folds:
        train = panel.loc[list(fold.train_indices)]
        test = panel.loc[list(fold.test_indices)]
        assert train["as_of_date"].max() < test["as_of_date"].min()
        assert train["label_end_date"].max() < test["as_of_date"].min()
        assert set(train.index).isdisjoint(test.index)


def test_walk_forward_split_preserves_original_row_indices() -> None:
    panel = _panel()
    folds = build_purged_walk_forward_splits(
        panel,
        n_splits=1,
        test_date_count=2,
        min_train_date_count=3,
    )

    all_indices = set(panel.index)
    assert set(folds[0].train_indices).issubset(all_indices)
    assert set(folds[0].test_indices).issubset(all_indices)


def test_walk_forward_split_rejects_impossible_request() -> None:
    with pytest.raises(ValueError, match="Not enough"):
        build_purged_walk_forward_splits(
            _panel().iloc[:4],
            n_splits=3,
            test_date_count=2,
            min_train_date_count=3,
        )
