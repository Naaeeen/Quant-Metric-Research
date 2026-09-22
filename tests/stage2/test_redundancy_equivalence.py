from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from scipy.stats import ConstantInputWarning

from quant_metric_research import redundancy
from quant_metric_research.screening import fit_metric_screen


def _reference_redundancy(
    panel: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    min_cross_section: int,
    as_of_date_column: str = "as_of_date",
) -> pd.DataFrame:
    """Pre-optimization algorithm: convert complete cases separately per pair/date."""
    normalized = panel.copy(deep=True)
    normalized[as_of_date_column] = pd.to_datetime(
        normalized[as_of_date_column], errors="coerce"
    )
    rows = []
    for left, right in combinations(feature_columns, 2):
        correlations = []
        for _, group in normalized.groupby(as_of_date_column, sort=True):
            paired = (
                group.loc[:, [left, right]]
                .apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if paired.shape[0] < min_cross_section:
                continue
            if paired[left].nunique() <= 1 or paired[right].nunique() <= 1:
                continue
            correlation = paired[left].corr(paired[right], method="spearman")
            if pd.notna(correlation):
                correlations.append(float(correlation))
        if correlations:
            clipped = np.clip(
                np.asarray(correlations, dtype=float), -1.0 + 1e-12, 1.0 - 1e-12
            )
            mean_spearman = float(np.tanh(np.arctanh(clipped).mean()))
            mean_abs_spearman = float(np.abs(clipped).mean())
        else:
            mean_spearman = float("nan")
            mean_abs_spearman = float("nan")
        rows.append(
            {
                "left": left,
                "right": right,
                "mean_spearman": mean_spearman,
                "mean_abs_spearman": mean_abs_spearman,
                "date_count": len(correlations),
            }
        )
    return pd.DataFrame(rows, columns=redundancy.REDUNDANCY_COLUMNS)


def _mixed_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "as_of_date": np.repeat(pd.bdate_range("2024-01-02", periods=3), 8),
            "a": [1, 2, None, 4, 5, 6, 7, 8] * 3,
            "b": ["8", "bad", "6", "5", np.inf, "3", "2", "1"] * 3,
            "c": [1, 1, 2, None, 2, -np.inf, 3, 4] * 3,
            "d": [1, 1, 1, 1, 2, 2, 2, 2] * 3,
        }
    )


@pytest.mark.parametrize("minimum", [2, 3, 4, 5, 6, 7, 8, 9])
@pytest.mark.parametrize("order", ["original", "permuted"])
def test_pairwise_results_match_reference_exactly(minimum, order):
    panel = _mixed_panel()
    if order == "permuted":
        panel = panel.sample(frac=1, random_state=23)
        panel.index = [index % 5 for index in range(len(panel))]
    before = panel.copy(deep=True)
    kwargs = {"feature_columns": ("d", "b", "a", "c"), "min_cross_section": minimum}

    actual = redundancy.compute_feature_redundancy(panel, **kwargs)
    expected = _reference_redundancy(panel, **kwargs)

    assert_frame_equal(actual, expected, check_exact=True)
    assert_frame_equal(panel, before, check_exact=True)
    for threshold in (0.0, 0.5, 0.9, 1.0):
        assert (
            actual["mean_abs_spearman"]
            .ge(threshold)
            .equals(expected["mean_abs_spearman"].ge(threshold))
        )


def test_constant_detection_happens_after_each_pair_missingness_filter():
    panel = pd.DataFrame(
        {
            "as_of_date": ["2024-01-02"] * 4,
            "a": [1, 1, 1, 2],
            "b": [2, 3, 4, None],
            "c": [1, 2, 3, 4],
        }
    )
    kwargs = {"feature_columns": ("a", "b", "c"), "min_cross_section": 3}

    actual = redundancy.compute_feature_redundancy(panel, **kwargs)

    assert_frame_equal(actual, _reference_redundancy(panel, **kwargs), check_exact=True)
    assert actual.iloc[0]["date_count"] == 0
    assert actual.iloc[2]["date_count"] == 1


def test_conversion_remains_date_local_for_large_integer_strings():
    panel = pd.DataFrame(
        {
            "decision_date": ["2024-01-02"] * 4 + ["2024-01-03"] * 4,
            "a": [str(2**53 + number) for number in range(4)] + ["0.1"] * 4,
            "b": ["1", "2", "3", "4", "4", "3", "2", "1"],
            "c": ["4", "1", "3", "2", None, "1.5", "2.5", "3.5"],
        }
    )
    kwargs = {
        "feature_columns": ("a", "b", "c"),
        "min_cross_section": 3,
        "as_of_date_column": "decision_date",
    }

    assert_frame_equal(
        redundancy.compute_feature_redundancy(panel, **kwargs),
        _reference_redundancy(panel, **kwargs),
        check_exact=True,
    )


@pytest.mark.parametrize("feature_columns", [(), ("a",), ("a", "b")])
@pytest.mark.parametrize("empty", [True, False])
def test_empty_panels_and_zero_or_one_feature_preserve_output_schema(
    feature_columns, empty
):
    panel = _mixed_panel().iloc[:0] if empty else _mixed_panel()
    kwargs = {"feature_columns": feature_columns, "min_cross_section": 2}

    assert_frame_equal(
        redundancy.compute_feature_redundancy(panel, **kwargs),
        _reference_redundancy(panel, **kwargs),
        check_exact=True,
    )


def test_grouping_and_numeric_conversion_are_not_repeated_for_every_pair(monkeypatch):
    panel = _mixed_panel()
    numeric_calls = []
    grouping_calls = []
    original_numeric = pd.to_numeric
    original_groupby = pd.DataFrame.groupby

    def observed_numeric(values, *args, **kwargs):
        numeric_calls.append(values.name)
        return original_numeric(values, *args, **kwargs)

    def observed_groupby(frame, *args, **kwargs):
        grouping_calls.append(len(frame))
        return original_groupby(frame, *args, **kwargs)

    monkeypatch.setattr(pd, "to_numeric", observed_numeric)
    monkeypatch.setattr(pd.DataFrame, "groupby", observed_groupby)

    redundancy.compute_feature_redundancy(
        panel, feature_columns=("a", "b", "c", "d"), min_cross_section=2
    )

    assert len(grouping_calls) == 1
    assert sorted(numeric_calls) == sorted(["a", "b", "c", "d"] * 3)


@pytest.mark.parametrize("minimum", [None, True, 0, 1, 2.0, "2"])
def test_invalid_minimum_still_fails_before_computation(minimum):
    with pytest.raises(ValueError, match="integer of at least 2"):
        redundancy.compute_feature_redundancy(
            _mixed_panel(), feature_columns=("a", "b"), min_cross_section=minimum
        )


def test_missing_columns_still_fail_before_computation():
    with pytest.raises(ValueError, match="Missing redundancy columns: absent"):
        redundancy.compute_feature_redundancy(
            _mixed_panel(), feature_columns=("a", "absent"), min_cross_section=2
        )


def test_invalid_dates_still_fail_even_without_feature_pairs():
    panel = pd.DataFrame({"as_of_date": ["not-a-date"], "a": [1]})
    with pytest.raises(ValueError, match="as_of_date contains invalid values"):
        redundancy.compute_feature_redundancy(
            panel, feature_columns=("a",), min_cross_section=2
        )


def test_nullable_dtypes_match_reference_without_input_mutation():
    panel = pd.DataFrame(
        {
            "as_of_date": np.repeat(pd.bdate_range("2024-01-02", periods=2), 6),
            "a": pd.Series([1, 2, pd.NA, 4, 5, 6] * 2, dtype="Int64"),
            "b": pd.Series([1.5, pd.NA, 2.5, 4.5, 3.5, 6.5] * 2, dtype="Float64"),
            "c": pd.Series(
                [True, False, pd.NA, True, False, True] * 2, dtype="boolean"
            ),
        }
    )
    before = panel.copy(deep=True)
    kwargs = {"feature_columns": ("a", "b", "c"), "min_cross_section": 2}

    assert_frame_equal(
        redundancy.compute_feature_redundancy(panel, **kwargs),
        _reference_redundancy(panel, **kwargs),
        check_exact=True,
    )
    assert_frame_equal(panel, before, check_exact=True)


@pytest.mark.parametrize("kind", ["declared", "equal", "below", "above"])
def test_scalar_boundary_and_downstream_selection_are_exact(monkeypatch, kind):
    panel = pd.DataFrame(
        {
            "as_of_date": np.repeat(pd.bdate_range("2024-01-02", periods=4), 5),
            "symbol": [f"S{i}" for i in range(5)] * 4,
            "a": [0, 1, 2, 3, 4] * 4,
            "b": [0, 1, 2, 4, 3] * 4,
            "target": [0, 1, 2, 3, 4] * 4,
        }
    )
    before = panel.copy(deep=True)
    kwargs = {"feature_columns": ("a", "b"), "min_cross_section": 3}
    expected_pairs = _reference_redundancy(panel, **kwargs)
    actual_pairs = redundancy.compute_feature_redundancy(panel, **kwargs)
    assert_frame_equal(actual_pairs, expected_pairs, check_exact=True)
    value = expected_pairs.iloc[0]["mean_abs_spearman"]
    # Matrix Spearman returns 0.9 and would change this screening decision.
    assert value < 0.9
    threshold = {
        "declared": 0.9,
        "equal": value,
        "below": np.nextafter(value, -np.inf),
        "above": np.nextafter(value, np.inf),
    }[kind]
    kwargs.update(
        target_column="target",
        train_end_date="2024-01-05",
        minimum_coverage=0.8,
        redundancy_threshold=threshold,
        hac_lags=1,
        quantiles=2,
        include_quantile_spreads=False,
    )
    actual = fit_metric_screen(panel, **kwargs)
    monkeypatch.setattr(
        "quant_metric_research.screening.compute_feature_redundancy",
        _reference_redundancy,
    )
    expected = fit_metric_screen(panel, **kwargs)
    assert actual.selected_features == expected.selected_features
    assert actual.dropped_features == expected.dropped_features
    assert ("b" in actual.selected_features) == (kind in {"declared", "above"})
    for name in ("quality", "daily_rank_ic", "ic_summary", "redundancy"):
        assert_frame_equal(
            getattr(actual, name), getattr(expected, name), check_exact=True
        )
    assert_frame_equal(panel, before, check_exact=True)


def test_asymmetric_missingness_reranks_ties_within_complete_pairs():
    panel = pd.DataFrame(
        {
            "as_of_date": ["2024-01-02"] * 6,
            "a": [1, 1, 2, 3, 4, 5],
            "b": [3, None, 1, 1, 2, None],
            "c": [None, 2, 2, 1, 3, 4],
        }
    )
    kwargs = {"feature_columns": ("c", "a", "b"), "min_cross_section": 3}
    assert_frame_equal(
        redundancy.compute_feature_redundancy(panel, **kwargs),
        _reference_redundancy(panel, **kwargs),
        check_exact=True,
    )
    paired = panel[["a", "b"]].dropna()
    wrong = panel[["a", "b"]].rank().dropna()
    assert paired.a.corr(paired.b, method="spearman") != wrong.a.corr(wrong.b)


def test_raw_uniqueness_precedes_float_precision_collapse():
    panel = pd.DataFrame(
        {"as_of_date": ["2024-01-02"] * 2, "a": [2**53, 2**53 + 1], "b": [1, 2]}
    )
    before = panel.copy(deep=True)
    kwargs = {"feature_columns": ("a", "b"), "min_cross_section": 2}
    # Raw values vary, so Spearman runs and warns after its float conversion.
    # Testing float uniqueness first would silently skip this call.
    with pytest.warns(ConstantInputWarning):
        expected = _reference_redundancy(panel, **kwargs)
    with pytest.warns(ConstantInputWarning):
        actual = redundancy.compute_feature_redundancy(panel, **kwargs)
    assert_frame_equal(actual, expected, check_exact=True)
    assert actual.iloc[0]["date_count"] == 0
    assert_frame_equal(panel, before, check_exact=True)


def test_pairwise_complete_cases_do_not_allocate_dataframes(monkeypatch):
    panel = _mixed_panel()
    kwargs = {"feature_columns": ("a", "b", "c", "d"), "min_cross_section": 3}
    expected = _reference_redundancy(panel, **kwargs)
    calls = []
    original = pd.DataFrame.dropna

    def observed_dropna(frame, *args, **kwargs):
        calls.append(frame.shape)
        return original(frame, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "dropna", observed_dropna)
    actual = redundancy.compute_feature_redundancy(panel, **kwargs)
    assert_frame_equal(actual, expected, check_exact=True)
    assert calls == [], "Pair/date completeness must use arrays, not DataFrame.dropna."


@pytest.mark.parametrize("dtype", ["Int64", "UInt64"])
def test_nullable_large_integer_uniqueness_survives_missingness(dtype):
    panel = pd.DataFrame(
        {
            "as_of_date": ["2024-01-02"] * 3,
            "a": pd.Series([2**53, 2**53 + 1, pd.NA], dtype=dtype),
            "b": [1, 2, 3],
        }
    )
    before = panel.copy(deep=True)
    kwargs = {"feature_columns": ("a", "b"), "min_cross_section": 2}
    with pytest.warns(ConstantInputWarning):
        expected = _reference_redundancy(panel, **kwargs)
    with pytest.warns(ConstantInputWarning):
        actual = redundancy.compute_feature_redundancy(panel, **kwargs)
    assert_frame_equal(actual, expected, check_exact=True)
    assert actual.iloc[0]["date_count"] == 0
    assert_frame_equal(panel, before, check_exact=True)
