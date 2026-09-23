from __future__ import annotations

import json
from dataclasses import replace
from types import MappingProxyType

import pandas as pd
import pytest

from quant_metric_research.benchmark import BenchmarkRun
from quant_metric_research.benchmark_config import BenchmarkConfig, NestedSplitConfig
from quant_metric_research.feature_bundles import FEATURE_BUNDLES


@pytest.fixture
def comparison_runs():
    """Saved-score fixtures only: no source panel, fitting or registry required."""
    dates = pd.to_datetime(["2025-01-06", "2025-01-07", "2025-01-09", "2025-01-10"])
    schedule = ("2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09", "2025-01-10")
    config = BenchmarkConfig(
        feature_columns=FEATURE_BUNDLES["legacy10_v1"].feature_columns,
        split=NestedSplitConfig(2, 2, 2, 2, 1, 1, 1),
        min_cross_section=2,
        quantiles=2,
        hac_lags=1,
        model_families=("ridge",),
        ridge_alphas=(0.1, 1.0),
    )
    bounds = {
        1: (pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03"), dates[0], dates[1]),
        2: (pd.Timestamp("2025-01-06"), pd.Timestamp("2025-01-07"), dates[2], dates[3]),
    }
    rows = [
        (i // 2 + 1, date, symbol, f"row-{i}-{symbol}")
        for i, date in enumerate(dates)
        for symbol in ("A", "B", "C", "D")
    ]
    assignments = pd.DataFrame(
        [
            {
                "phase": "development",
                "fold": fold,
                "split_id": f"development_fold_{fold - 1}",
                "row_id": row_id,
                "role": "evaluation" if ownfold == fold else "excluded",
                "exclusion_reason": None if ownfold == fold else "future_of_evaluation",
                "train_end_date": bounds[fold][0],
                "train_label_end_max": bounds[fold][1],
                "evaluation_start": bounds[fold][2],
                "evaluation_end": bounds[fold][3],
            }
            for fold in (1, 2)
            for ownfold, date, symbol, row_id in rows
        ]
    )
    identity = {
        "artifact_schema_version": "6",
        "execution_mode": "development",
        "source_fingerprint": "a" * 64,
        "panel_fingerprint": "b" * 64,
        "implementation_version": "fixture-producer-v1",
        "package_version": "fixture-producer-v1",
        "python_version": "3.12.0",
        "numpy_version": "2.2.0",
        "pandas_version": "2.3.0",
        "scipy_version": "1.15.0",
        "scikit_learn_version": "1.9.0",
        "dataset_versions": ["synthetic-comparison-v1"],
        "fingerprint_scope": (
            "source_code+configuration+validated_panel_contract+execution_mode"
        ),
        "development_start": "2025-01-01",
        "locked_test_start": "2025-01-13",
        "locked_test_end": "2025-01-14",
        "locked_label_end_max": "2025-01-15",
        "evaluation_schedule": {
            "definition_version": "1",
            "source": "validated_panel_within_planned_phase_bounds",
            "lag_unit": "scheduled_observations",
            "dates_by_phase": {"development": schedule},
        },
    }
    runs = []
    for candidate in (False, True):
        current = replace(
            config,
            feature_columns=FEATURE_BUNDLES[
                "legacy10_plus_price3_v1" if candidate else "legacy10_v1"
            ].feature_columns,
        )
        features = (
            ["beta", "trailing_return", "return_21s"]
            if candidate
            else ["beta", "trailing_return"]
        )
        predictions = pd.DataFrame(
            [
                {
                    "phase": "development",
                    "fold": fold,
                    "as_of_date": date,
                    "symbol": symbol,
                    "row_id": row_id,
                    "model": model,
                    "score": float((ord(symbol) - 65 + int(date.day)) % 4)
                    if model == "ridge"
                    else float(ord(symbol) - 65) / 4,
                    "target": float(ord(symbol) - 65) / 100,
                    "realized_return": float(ord(symbol) - 65) / 100,
                    "candidate_id": (
                        'ridge:{"alpha":1.0}' if candidate else 'ridge:{"alpha":0.1}'
                    )
                    if model == "ridge"
                    else None,
                    "fit_end_date": bounds[fold][0],
                    "train_label_end_max": bounds[fold][1],
                    "evaluation_start": bounds[fold][2],
                    "selected_features": json.dumps(features, separators=(",", ":")),
                    "selected_feature_count": len(features),
                    "feature_count": len(features),
                    "zero_observed_features": False,
                    "baseline_feature": None,
                    "best_metric_feature": None if model == "ridge" else "beta",
                }
                for fold, date, symbol, row_id in rows
                for model in ("ridge", "equal_weight_rank")
            ]
        )
        manifest = {
            **identity,
            "configuration": current.to_mapping(),
            "run_fingerprint": ("c" if candidate else "d") * 64,
            "model_input_fingerprint": ("e" if candidate else "f") * 64,
        }
        runs.append(
            BenchmarkRun(
                data_gate=MappingProxyType({"claim_scope": "development_only"}),
                fold_assignments=assignments.copy(deep=True),
                predictions=predictions,
                daily_metrics=pd.DataFrame(),
                fold_metrics=pd.DataFrame(),
                tuning_trials=pd.DataFrame(),
                screening_by_fold=pd.DataFrame(),
                summary=pd.DataFrame(),
                acceptance=MappingProxyType(
                    {
                        "acceptance_status": "not_evaluated",
                        "lockbox_evaluated_once_in_this_run": False,
                        "model_gate_passed": False,
                        "eligible_for_stage4_data_review": False,
                        "stage4_eligible": False,
                    }
                ),
                manifest=MappingProxyType(manifest),
            )
        )
    return tuple(runs)
