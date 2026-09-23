from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from quant_metric_research._comparison_contract import validate_comparison_inputs
from quant_metric_research._comparison_predictions import _exact_outcome


def _manifest(run, **changes):
    return replace(run, manifest=MappingProxyType({**run.manifest, **changes}))


def _prediction(run, column, value, *, index=0):
    frame = run.predictions.copy(deep=True)
    frame[column] = frame[column].astype(object)
    frame.loc[index, column] = value
    return replace(run, predictions=frame)


def test_valid_inputs_preserve_values_and_return_copied_immutable_evidence(
    comparison_runs,
):
    legacy, candidate = comparison_runs
    before = [run.predictions.copy(deep=True) for run in comparison_runs]
    result = validate_comparison_inputs(legacy, candidate, model_family="ridge")
    assert set(result.predictions["model"]) == {
        "legacy_equal_rank",
        "legacy_model",
        "candidate_model",
    }
    assert len(result.predictions) == 48
    assert result.expected_dates[2] == "2025-01-08"
    assert result.producer_identity["package_version"] == "fixture-producer-v1"
    with pytest.raises(TypeError):
        result.producer_identity["package_version"] = "changed"
    result.predictions.loc[:, "score"] = 99.0
    for run, saved in zip(comparison_runs, before, strict=True):
        pd.testing.assert_frame_equal(run.predictions, saved)


class Untouchable:
    def __getattribute__(self, name):
        raise AssertionError("Frame accessed before scope validation")


@pytest.mark.parametrize(
    "change",
    [
        "full_manifest",
        "schema",
        "final_acceptance",
        "final_flag",
        "promoted",
        "data_gate",
        "final_experiment",
    ],
)
def test_scope_failures_happen_before_any_frame_access(comparison_runs, change):
    legacy, candidate = comparison_runs
    broken = replace(
        candidate, predictions=Untouchable(), fold_assignments=Untouchable()
    )
    if change == "full_manifest":
        broken = _manifest(broken, execution_mode="full")
    elif change == "schema":
        broken = _manifest(broken, artifact_schema_version="5")
    elif change == "final_experiment":
        broken = _manifest(broken, experiment={"kind": "final"})
    elif change == "data_gate":
        broken = replace(broken, data_gate={"claim_scope": "benchmark_engine_only"})
    else:
        key, value = {
            "final_acceptance": ("acceptance_status", "passed"),
            "final_flag": ("lockbox_evaluated_once_in_this_run", True),
            "promoted": ("stage4_eligible", True),
        }[change]
        broken = replace(broken, acceptance={**broken.acceptance, key: value})
    with pytest.raises(ValueError):
        validate_comparison_inputs(legacy, broken, model_family="ridge")


@pytest.mark.parametrize(
    "key,value",
    [
        ("source_fingerprint", "c" * 64),
        ("panel_fingerprint", "d" * 64),
        ("numpy_version", "different"),
        ("locked_test_start", "2025-01-14"),
        ("dataset_versions", ["different"]),
    ],
)
def test_producer_evidence_must_match_between_inputs(comparison_runs, key, value):
    legacy, candidate = comparison_runs
    with pytest.raises(ValueError):
        validate_comparison_inputs(
            legacy, _manifest(candidate, **{key: value}), model_family="ridge"
        )


@pytest.mark.parametrize(
    "change", ["missing_field", "changed_setting", "wrong_features"]
)
def test_complete_configs_may_differ_only_in_literal_bundle_features(
    comparison_runs, change
):
    legacy, candidate = comparison_runs
    config = deepcopy(candidate.manifest["configuration"])
    if change == "missing_field":
        del config["random_seed"]
    elif change == "changed_setting":
        config["random_seed"] += 1
    else:
        config["feature_columns"] = ("beta",)
    with pytest.raises(ValueError):
        validate_comparison_inputs(
            legacy, _manifest(candidate, configuration=config), model_family="ridge"
        )


@pytest.mark.parametrize(
    "change", ["version", "source", "locked_phase", "interior", "empty", "non_iso"]
)
def test_exact_supported_schedule_is_required(comparison_runs, change):
    legacy, candidate = comparison_runs
    schedule = deepcopy(candidate.manifest["evaluation_schedule"])
    if change == "version":
        schedule["definition_version"] = "2"
    elif change == "source":
        schedule["source"] = "predictions"
    elif change == "locked_phase":
        schedule["dates_by_phase"]["locked_test"] = ("2025-01-13",)
    elif change == "interior":
        schedule["dates_by_phase"]["development"] = (
            "2025-01-06",
            "2025-01-07",
            "2025-01-08",
            "2025-01-09",
            "2025-01-11",
        )
    elif change == "empty":
        schedule["dates_by_phase"]["development"] = ()
    else:
        schedule["dates_by_phase"]["development"] = ("2025-1-6",)
    with pytest.raises(ValueError):
        validate_comparison_inputs(
            legacy,
            _manifest(candidate, evaluation_schedule=schedule),
            model_family="ridge",
        )


@pytest.mark.parametrize(
    "column,value",
    [
        ("phase", "locked_test"),
        ("fold", 2),
        ("as_of_date", "2025-01-08"),
        ("row_id", None),
        ("symbol", " A "),
        ("score", True),
        ("score", complex(1, 2)),
        ("target", "nan"),
        ("realized_return", np.inf),
        ("target", 99),
        ("realized_return", None),
        ("fit_end_date", "2025-01-01"),
        ("train_label_end_max", "2025-01-02"),
        ("evaluation_start", "2025-01-05"),
        ("feature_count", True),
        ("feature_count", 1.0),
        ("feature_count", -1),
        ("feature_count", 4),
        ("selected_feature_count", 2),
        ("zero_observed_features", 1),
        ("zero_observed_features", True),
        ("candidate_id", 'ridge:{"alpha":99}'),
        ("selected_features", '["beta","beta","return_21s"]'),
        ("selected_features", '["beta","trailing_return","unknown"]'),
    ],
)
def test_selected_prediction_contract_failures(comparison_runs, column, value):
    legacy, candidate = comparison_runs
    with pytest.raises(ValueError):
        validate_comparison_inputs(
            legacy, _prediction(candidate, column, value), model_family="ridge"
        )


@pytest.mark.parametrize(
    "change",
    ["duplicate", "missing_row", "missing_day", "assignment", "assignment_bound"],
)
def test_keys_and_assignment_evidence_must_be_complete(comparison_runs, change):
    legacy, candidate = comparison_runs
    if change == "duplicate":
        candidate = replace(
            candidate,
            predictions=pd.concat(
                [candidate.predictions, candidate.predictions.iloc[:1]]
            ),
        )
    elif change == "missing_row":
        candidate = replace(candidate, predictions=candidate.predictions.iloc[1:])
    elif change == "missing_day":
        candidate = replace(
            candidate,
            predictions=candidate.predictions.loc[
                candidate.predictions.as_of_date.ne(pd.Timestamp("2025-01-06"))
            ],
        )
    elif change == "assignment":
        candidate = replace(
            candidate, fold_assignments=candidate.fold_assignments.iloc[1:]
        )
    else:
        altered = candidate.fold_assignments.copy()
        altered.loc[0, "train_end_date"] = pd.Timestamp("2025-01-06")
        candidate = replace(candidate, fold_assignments=altered)
    with pytest.raises(ValueError):
        validate_comparison_inputs(legacy, candidate, model_family="ridge")


def test_positive_feature_count_with_missing_score_is_visible_coverage(comparison_runs):
    legacy, candidate = comparison_runs
    result = validate_comparison_inputs(
        legacy, _prediction(candidate, "score", None), model_family="ridge"
    )
    assert result.predictions["score"].isna().sum() == 1


def test_zero_count_requires_missing_score_and_boolean_flag(comparison_runs):
    legacy, candidate = comparison_runs
    candidate = _prediction(
        _prediction(
            _prediction(candidate, "feature_count", 0), "zero_observed_features", True
        ),
        "score",
        None,
    )
    result = validate_comparison_inputs(legacy, candidate, model_family="ridge")
    assert result.predictions["zero_observed_features"].sum() == 1


def test_unused_model_metadata_does_not_apply_selected_arm_count_rules(comparison_runs):
    legacy, candidate = comparison_runs
    unused = legacy.predictions.iloc[:1].assign(
        model="best_metric",
        selected_feature_count=1,
        feature_count=1,
        candidate_id=None,
    )
    legacy = replace(
        legacy, predictions=pd.concat([legacy.predictions, unused], ignore_index=True)
    )
    assert (
        len(
            validate_comparison_inputs(
                legacy, candidate, model_family="ridge"
            ).predictions
        )
        == 48
    )


@pytest.mark.parametrize("family", ["", " ridge", "unknown", None])
def test_family_is_explicit_and_configured(comparison_runs, family):
    with pytest.raises(ValueError):
        validate_comparison_inputs(*comparison_runs, model_family=family)


@pytest.mark.parametrize("column", ["target", "realized_return"])
def test_tiny_outcome_difference_is_not_tolerated(comparison_runs, column):
    legacy, candidate = comparison_runs
    # A nonzero reference exercises pandas' default relative tolerance.
    index = candidate.predictions.index[candidate.predictions.symbol.eq("B")][0]
    candidate = _prediction(candidate, column, 0.010000000001, index=index)
    with pytest.raises(ValueError, match="outcomes"):
        validate_comparison_inputs(legacy, candidate, model_family="ridge")


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("bad", [0, "false", None, True])
def test_both_frames_are_untouched_until_both_scopes_pass(comparison_runs, side, bad):
    runs = [
        replace(run, predictions=Untouchable(), fold_assignments=Untouchable())
        for run in comparison_runs
    ]
    runs[side] = replace(
        runs[side],
        acceptance={**runs[side].acceptance, "lockbox_evaluated_once_in_this_run": bad},
    )
    with pytest.raises(ValueError):
        validate_comparison_inputs(*runs, model_family="ridge")


class OutcomeSentinel(pd.DataFrame):
    def __getitem__(self, key):
        if isinstance(key, str) and key in {"score", "target", "realized_return"}:
            raise AssertionError("Outcome accessed before all structural checks")
        return super().__getitem__(key)


@pytest.mark.parametrize(
    "column,value",
    [("phase", "locked_test"), ("as_of_date", pd.Timestamp("2025-01-08")), ("fold", 2)],
)
def test_all_phase_and_date_scope_checks_precede_outcome_access(
    comparison_runs, column, value
):
    legacy, candidate = comparison_runs
    broken = _prediction(candidate, column, value)
    legacy = replace(legacy, predictions=OutcomeSentinel(legacy.predictions))
    with pytest.raises(ValueError):
        validate_comparison_inputs(legacy, broken, model_family="ridge")


@pytest.mark.parametrize(
    "change",
    [
        "duplicate_assignment",
        "bad_role",
        "unpurged",
        "wrong_lag_unit",
        "duplicate_schedule",
        "incomplete_config",
    ],
)
def test_identically_malformed_evidence_does_not_pass_by_matching(
    comparison_runs, change
):
    changed = []
    for run in comparison_runs:
        if change == "duplicate_assignment":
            run = replace(
                run,
                fold_assignments=pd.concat(
                    [run.fold_assignments, run.fold_assignments.iloc[:1]]
                ),
            )
        elif change in {"bad_role", "unpurged"}:
            frame = run.fold_assignments.copy()
            if change == "bad_role":
                frame.loc[0, "role"] = "unknown"
            else:
                frame.loc[frame.fold.eq(1), "train_label_end_max"] = pd.Timestamp(
                    "2025-01-06"
                )
            run = replace(run, fold_assignments=frame)
        elif change == "incomplete_config":
            config = deepcopy(run.manifest["configuration"])
            del config["hist_max_iter"]
            run = _manifest(run, configuration=config)
        else:
            schedule = deepcopy(run.manifest["evaluation_schedule"])
            if change == "wrong_lag_unit":
                schedule["lag_unit"] = "calendar_days"
            else:
                schedule["dates_by_phase"]["development"] = ("2025-01-06", "2025-01-06")
            run = _manifest(run, evaluation_schedule=schedule)
        changed.append(run)
    with pytest.raises(ValueError):
        validate_comparison_inputs(*changed, model_family="ridge")


def test_same_missing_evaluation_row_in_every_arm_is_rejected(comparison_runs):
    changed = [
        replace(
            run, predictions=run.predictions.loc[run.predictions.row_id.ne("row-0-A")]
        )
        for run in comparison_runs
    ]
    with pytest.raises(ValueError, match="assignment"):
        validate_comparison_inputs(*changed, model_family="ridge")


@pytest.mark.parametrize(
    "reason", ["missing_label", "missing_target", "locked_test_overlap"]
)
def test_excluded_rows_and_matching_missing_outcomes_remain_scored(
    comparison_runs, reason
):
    changed = []
    for run in comparison_runs:
        assignments, predictions = run.fold_assignments.copy(), run.predictions.copy()
        selected = assignments.fold.eq(1) & assignments.row_id.eq("row-0-A")
        assignments.loc[selected, ["role", "exclusion_reason"]] = [
            "excluded",
            reason,
        ]
        predictions.loc[predictions.row_id.eq("row-0-A"), "target"] = np.nan
        if reason in {"missing_label", "locked_test_overlap"}:
            predictions.loc[predictions.row_id.eq("row-0-A"), "realized_return"] = (
                np.nan
            )
        changed.append(
            replace(run, fold_assignments=assignments, predictions=predictions)
        )
    result = validate_comparison_inputs(*changed, model_family="ridge")
    scored = result.predictions.loc[result.predictions.row_id.eq("row-0-A")]
    assert (
        len(scored) == 3 and scored.score.notna().all() and scored.target.isna().all()
    )


def test_non_equal_primary_baseline_and_partial_observed_features_are_valid(
    comparison_runs,
):
    changed = []
    for run in comparison_runs:
        config = deepcopy(run.manifest["configuration"])
        config["primary_baseline"] = "best_metric"
        changed.append(_manifest(run, configuration=config))
    changed[1] = _prediction(changed[1], "feature_count", 1)
    assert (
        len(validate_comparison_inputs(*changed, model_family="ridge").predictions)
        == 48
    )


@pytest.mark.parametrize("model", ["ridge", "equal_weight_rank"])
@pytest.mark.parametrize(
    "column,value", [("row_id", "unknown"), ("target", 0.5), ("realized_return", None)]
)
def test_both_legacy_arms_receive_alignment_checks(
    comparison_runs, model, column, value
):
    legacy, candidate = comparison_runs
    index = legacy.predictions.index[legacy.predictions.model.eq(model)][0]
    with pytest.raises(ValueError):
        validate_comparison_inputs(
            _prediction(legacy, column, value, index=index),
            candidate,
            model_family="ridge",
        )


@pytest.mark.parametrize("registered", [False, True])
def test_actual_producer_development_annotation_without_kind_is_supported(
    comparison_runs, registered
):
    from quant_metric_research.experiment_workflow import _annotate

    runs = [
        _annotate(run, run_id=f"development-{i}" if registered else None, final=False)
        for i, run in enumerate(comparison_runs)
    ]
    assert (
        len(validate_comparison_inputs(*runs, model_family="ridge").predictions) == 48
    )


def test_final_development_reference_is_rejected_before_frames(comparison_runs):
    legacy, candidate = comparison_runs
    candidate = _manifest(
        candidate,
        experiment={
            "registered": True,
            "run_id": "final-run",
            "development_run_id": "prior-development",
            "registry_scope": "local_file_date_envelopes",
        },
    )
    candidate = replace(
        candidate, predictions=Untouchable(), fold_assignments=Untouchable()
    )
    with pytest.raises(ValueError):
        validate_comparison_inputs(legacy, candidate, model_family="ridge")


@pytest.mark.parametrize(
    "change", ["split_id", "train_before_panel", "label_before_training"]
)
def test_identical_invalid_fold_identity_or_training_bounds_fail(
    comparison_runs, change
):
    runs = []
    for run in comparison_runs:
        frame, predictions = run.fold_assignments.copy(), run.predictions.copy()
        if change == "split_id":
            frame.loc[frame.fold.eq(1), "split_id"] = "incorrect-fold"
        elif change == "train_before_panel":
            frame.loc[frame.fold.eq(1), "train_end_date"] = pd.Timestamp("2024-12-31")
            predictions.loc[predictions.fold.eq(1), "fit_end_date"] = pd.Timestamp(
                "2024-12-31"
            )
        else:
            frame.loc[frame.fold.eq(1), "train_label_end_max"] = pd.Timestamp(
                "2025-01-02"
            )
            predictions.loc[predictions.fold.eq(1), "train_label_end_max"] = (
                pd.Timestamp("2025-01-02")
            )
        runs.append(replace(run, fold_assignments=frame, predictions=predictions))
    with pytest.raises(ValueError):
        validate_comparison_inputs(*runs, model_family="ridge")


def test_same_fold_training_ids_cannot_be_scored_as_evaluation(comparison_runs):
    runs = []
    for run in comparison_runs:
        assignments = run.fold_assignments.copy()
        assignments.loc[
            assignments.fold.eq(1) & assignments.row_id.eq("row-0-A"), "role"
        ] = "training"
        runs.append(replace(run, fold_assignments=assignments))
    with pytest.raises(ValueError, match="assignment"):
        validate_comparison_inputs(*runs, model_family="ridge")


@pytest.mark.parametrize(
    "reason", ["future_of_evaluation", "unknown", "label_overlap", "locked_final_test"]
)
def test_scored_exclusion_reasons_must_allow_in_window_evaluation(
    comparison_runs, reason
):
    runs = []
    for run in comparison_runs:
        assignments = run.fold_assignments.copy()
        assignments.loc[
            assignments.fold.eq(1) & assignments.row_id.eq("row-0-A"),
            ["role", "exclusion_reason"],
        ] = ["excluded", reason]
        runs.append(replace(run, fold_assignments=assignments))
    with pytest.raises(ValueError, match="assignment"):
        validate_comparison_inputs(*runs, model_family="ridge")


@pytest.mark.parametrize(
    "reason,column",
    [
        ("missing_label", "target"),
        ("missing_label", "realized_return"),
        ("locked_test_overlap", "target"),
        ("locked_test_overlap", "realized_return"),
        ("missing_target", "target"),
    ],
)
def test_excluded_outcomes_cannot_contradict_assignment_authorization(
    comparison_runs, reason, column
):
    runs = []
    for run in comparison_runs:
        assignments, predictions = run.fold_assignments.copy(), run.predictions.copy()
        assignments.loc[
            assignments.fold.eq(1) & assignments.row_id.eq("row-0-A"),
            ["role", "exclusion_reason"],
        ] = ["excluded", reason]
        selected = predictions.row_id.eq("row-0-A")
        predictions.loc[selected, ["target", "realized_return"]] = np.nan
        predictions.loc[selected, column] = 0.01
        runs.append(replace(run, fold_assignments=assignments, predictions=predictions))
    with pytest.raises(ValueError, match="assignment"):
        validate_comparison_inputs(*runs, model_family="ridge")


def test_evaluation_role_requires_observed_target(comparison_runs):
    runs = []
    for run in comparison_runs:
        predictions = run.predictions.copy()
        predictions.loc[predictions.row_id.eq("row-0-A"), "target"] = np.nan
        runs.append(replace(run, predictions=predictions))
    with pytest.raises(ValueError, match="assignment"):
        validate_comparison_inputs(*runs, model_family="ridge")


@pytest.mark.parametrize("column", ["target", "realized_return"])
def test_lossy_float_conversion_cannot_hide_distinct_integer_outcomes(
    comparison_runs, column
):
    runs = []
    for offset, run in enumerate(comparison_runs):
        predictions = run.predictions.copy()
        predictions[column] = predictions[column].astype(object)
        predictions.loc[predictions.row_id.eq("row-0-A"), column] = 2**60 + offset
        runs.append(replace(run, predictions=predictions))
    with pytest.raises(ValueError, match="outcomes"):
        validate_comparison_inputs(*runs, model_family="ridge")


def test_exact_outcome_uses_compact_decimal_evidence():
    # Bounded exponent for the RED check: the old Fraction path expands it.
    result = _exact_outcome("1e-1000")
    assert isinstance(result, Decimal)
    assert result.as_tuple().exponent == -1000


def test_exact_outcome_does_not_round_to_ambient_decimal_precision():
    with localcontext() as context:
        context.prec = 2
        assert _exact_outcome(2**60) != _exact_outcome(2**60 + 1)
        assert _exact_outcome("0.1") != _exact_outcome(0.1)
        assert _exact_outcome(1) == _exact_outcome(1.0)


@pytest.mark.parametrize(
    "value",
    [np.int64(1), np.uint64(1), np.float32(1), np.float64(1), "1", Decimal("1")],
)
def test_exact_outcome_supports_standard_real_scalar_representations(value):
    assert _exact_outcome(value) == Decimal(1)


def test_exact_outcome_rejects_unsupported_numeric_representation():
    with pytest.raises(ValueError, match="representation"):
        _exact_outcome(Fraction(1, 3))


@pytest.mark.parametrize("equal", [False, True])
def test_extreme_underflowing_exponents_retain_compact_exact_outcome_identity(
    comparison_runs, equal
):
    runs = []
    for index, run in enumerate(comparison_runs):
        predictions = run.predictions.copy()
        predictions["target"] = predictions["target"].astype(object)
        value = "1e-100000000" if equal or index == 0 else "2e-100000000"
        predictions.loc[predictions.row_id.eq("row-0-A"), "target"] = value
        runs.append(replace(run, predictions=predictions))
    if equal:
        result = validate_comparison_inputs(*runs, model_family="ridge")
        assert (
            result.predictions.loc[result.predictions.row_id.eq("row-0-A"), "target"]
            .eq(0.0)
            .all()
        )
    else:
        with pytest.raises(ValueError, match="outcomes"):
            validate_comparison_inputs(*runs, model_family="ridge")
