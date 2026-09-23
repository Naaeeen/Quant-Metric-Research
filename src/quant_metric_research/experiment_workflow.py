from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Any

import pandas as pd

from . import benchmark as engine
from .benchmark_config import BenchmarkConfig
from .benchmark_data import build_stage3_data_plan
from .benchmark_metrics import (
    PredictionEvaluation,
    choose_frozen_model,
    evaluate_prediction_frame,
)
from .benchmark_reporting import _data_gate, _fingerprints
from .experiment_registry import ExperimentRegistry


def _execute_benchmark(
    panel: pd.DataFrame,
    *,
    config: BenchmarkConfig,
    evaluate_lockbox: bool = False,
    on_plan: Callable[[Mapping[str, Any]], None] | None = None,
    before_lockbox: Callable[[Mapping[str, Any], str], None] | None = None,
) -> engine.BenchmarkRun:
    """Internal computation kernel; public final runs require registry governance."""
    if not isinstance(config, BenchmarkConfig):
        raise ValueError("config must be a BenchmarkConfig.")
    if not isinstance(evaluate_lockbox, bool):
        raise ValueError("evaluate_lockbox must be a boolean.")
    engine._validate_realized_return(panel, config)
    plan = build_stage3_data_plan(
        panel,
        feature_columns=config.feature_columns,
        target_column=config.target_column,
        locked_test_date_count=config.split.final_test_date_count,
        locked_min_cross_section=config.min_cross_section,
        n_splits=config.split.outer_n_splits,
        evaluation_date_count=config.split.outer_test_date_count,
        min_train_date_count=config.split.outer_min_train_date_count,
    )
    manifest = _fingerprints(plan, config=config, evaluate_lockbox=evaluate_lockbox)
    if on_plan is not None:
        on_plan(manifest)
    development_predictions, development_trials, development_screens = (
        engine._development_run(plan, config=config)
    )
    development_evaluation = engine._evaluate_development(
        development_predictions,
        config=config,
        expected_dates=engine._phase_schedule(plan, phase="development"),
    )
    frozen_family = choose_frozen_model(
        development_evaluation.summary,
        model_families=config.model_families,
    )
    if not evaluate_lockbox:
        return engine._development_result(
            plan,
            config=config,
            predictions=development_predictions,
            trials=development_trials,
            screening=development_screens,
            evaluation=development_evaluation,
            frozen_family=frozen_family,
        )
    if before_lockbox is None:
        raise ValueError("Final-test reservation callback is required.")
    before_lockbox(manifest, frozen_family)
    locked_predictions, locked_trials, locked_screens = engine._locked_run(
        plan,
        config=config,
        frozen_family=frozen_family,
    )
    locked_schedule = engine._phase_schedule(plan, phase="locked_test")
    locked_evaluation = evaluate_prediction_frame(
        locked_predictions,
        min_cross_section=config.min_cross_section,
        quantiles=config.quantiles,
        hac_lags=config.hac_lags,
        primary_models=(frozen_family, config.primary_baseline),
        expected_dates_by_phase={"locked_test": locked_schedule},
    )
    predictions = engine._sort_predictions(
        pd.concat([development_predictions, locked_predictions], ignore_index=True)
    )
    evaluation = PredictionEvaluation(
        daily_metrics=pd.concat(
            [
                development_evaluation.daily_metrics,
                locked_evaluation.daily_metrics,
            ],
            ignore_index=True,
        ),
        fold_metrics=pd.concat(
            [
                development_evaluation.fold_metrics,
                locked_evaluation.fold_metrics,
            ],
            ignore_index=True,
        ),
        summary=pd.concat(
            [development_evaluation.summary, locked_evaluation.summary],
            ignore_index=True,
        ),
    )
    tuning_trials = (
        pd.concat([development_trials, locked_trials], ignore_index=True)
        .sort_values(
            ["phase", "outer_fold", "family", "candidate_order"], kind="stable"
        )
        .reset_index(drop=True)
    )
    screening = (
        pd.concat([development_screens, locked_screens], ignore_index=True)
        .sort_values(
            ["phase", "outer_fold", "fit_kind", "family", "inner_fold", "feature"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    return engine.BenchmarkRun(
        data_gate=_data_gate(plan, config),
        fold_assignments=engine._assignment_frame(plan, config),
        predictions=predictions,
        daily_metrics=evaluation.daily_metrics,
        fold_metrics=evaluation.fold_metrics,
        tuning_trials=tuning_trials,
        screening_by_fold=screening,
        summary=evaluation.summary,
        acceptance=engine._acceptance(
            evaluation,
            config=config,
            frozen_family=frozen_family,
            expected_locked_dates=locked_schedule,
        ),
        manifest=_fingerprints(plan, config=config),
    )


def _annotate(
    result: engine.BenchmarkRun,
    *,
    run_id: str | None,
    final: bool,
    development_run_id: str | None = None,
) -> engine.BenchmarkRun:
    return replace(
        result,
        manifest=MappingProxyType(
            {
                **result.manifest,
                "experiment": {
                    "registered": run_id is not None,
                    "run_id": run_id,
                    "development_run_id": development_run_id,
                    "registry_scope": "local_file_date_envelopes" if run_id else None,
                },
            }
        ),
        acceptance=MappingProxyType(
            {
                **result.acceptance,
                "lockbox_reuse_registry_enforced": bool(final and run_id),
            }
        ),
    )


def _mark_failed(
    registry: ExperimentRegistry, run_id: str, error: BaseException
) -> None:
    try:
        registry.fail_run(run_id, error_type=type(error).__name__)
    except Exception as marking_error:
        error.add_note(
            "Registry status update failed; the run may remain unresolved. "
            f"Reservation is not released ({type(marking_error).__name__})."
        )


def run_experiment(
    panel: pd.DataFrame,
    *,
    config: BenchmarkConfig,
    evaluate_lockbox: bool,
    registry: ExperimentRegistry | None,
    study_id: str | None,
    hypothesis: str | None,
    development_run_id: str | None,
) -> engine.BenchmarkRun:
    if not isinstance(config, BenchmarkConfig):
        raise ValueError("config must be a BenchmarkConfig.")
    if not isinstance(evaluate_lockbox, bool):
        raise ValueError("evaluate_lockbox must be a boolean.")
    if registry is not None and not isinstance(registry, ExperimentRegistry):
        raise ValueError("registry must be an ExperimentRegistry.")
    if evaluate_lockbox:
        if registry is None or not development_run_id:
            raise ValueError(
                "Final evaluation requires registry and development_run_id."
            )
        if study_id is not None or hypothesis is not None:
            raise ValueError(
                "Final evaluation uses the registered study and hypothesis."
            )
        reference = registry.get_run(development_run_id)
        if reference["kind"] != "development" or reference["status"] != "completed":
            raise ValueError(
                "Final evaluation requires completed development evidence."
            )
        return _run_final(
            panel,
            config=config,
            registry=registry,
            development_run_id=development_run_id,
        )
    if development_run_id is not None:
        raise ValueError("development_run_id is only valid for final evaluation.")
    if registry is None:
        if study_id is not None or hypothesis is not None:
            raise ValueError("study_id and hypothesis require a registry.")
        return _annotate(
            _execute_benchmark(panel, config=config), run_id=None, final=False
        )
    return _run_development(
        panel,
        config=config,
        registry=registry,
        study_id=study_id,
        hypothesis=hypothesis,
    )


def _run_development(
    panel: pd.DataFrame,
    *,
    config: BenchmarkConfig,
    registry: ExperimentRegistry,
    study_id: str | None,
    hypothesis: str | None,
) -> engine.BenchmarkRun:
    run_id = registry.start_development(
        study_id=study_id,
        hypothesis=hypothesis,
        configuration=config.to_mapping(),
    )
    try:
        result = _execute_benchmark(
            panel,
            config=config,
            on_plan=lambda manifest: registry.record_development_plan(
                run_id,
                manifest=manifest,
            ),
        )
        registry.complete_development(
            run_id,
            manifest=result.manifest,
            frozen_family=result.acceptance["frozen_model_family"],
        )
    except BaseException as error:
        _mark_failed(registry, run_id, error)
        raise
    return _annotate(result, run_id=run_id, final=False)


def _run_final(
    panel: pd.DataFrame,
    *,
    config: BenchmarkConfig,
    registry: ExperimentRegistry,
    development_run_id: str,
) -> engine.BenchmarkRun:
    reservation: str | None = None

    def reserve(manifest: Mapping[str, Any], family: str) -> None:
        nonlocal reservation
        reservation = registry.reserve_final(
            development_run_id=development_run_id,
            manifest=manifest,
            frozen_family=family,
        )

    try:
        result = _execute_benchmark(
            panel,
            config=config,
            evaluate_lockbox=True,
            on_plan=lambda manifest: registry.validate_final_plan(
                development_run_id=development_run_id,
                manifest=manifest,
            ),
            before_lockbox=reserve,
        )
        if reservation is None:
            raise RuntimeError("Final evaluation finished without a reservation.")
        annotated = _annotate(
            result,
            run_id=reservation,
            final=True,
            development_run_id=development_run_id,
        )
        registry.complete_final(
            reservation,
            manifest=result.manifest,
            acceptance=annotated.acceptance,
        )
    except BaseException as error:
        if reservation is not None:
            _mark_failed(registry, reservation, error)
        raise
    return annotated
