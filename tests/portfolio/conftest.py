from types import MappingProxyType

import pandas as pd
import pytest

from quant_metric_research.benchmark import BenchmarkRun
from quant_metric_research.benchmark_config import BenchmarkConfig, NestedSplitConfig
from quant_metric_research.session_calendar import ExpectedSessionCalendar


@pytest.fixture
def ledger_case():
    dates = pd.bdate_range("2025-01-02", periods=12)
    calendar = ExpectedSessionCalendar(
        tuple(dates), dates[0], dates[-1], "fixture", "1"
    )
    config = BenchmarkConfig(
        feature_columns=("signal",),
        split=NestedSplitConfig(1, 1, 3, 2, 1, 1, 1),
        min_cross_section=2,
        quantiles=2,
        hac_lags=1,
        model_families=("ridge",),
        ridge_alphas=(1.0,),
    )
    predictions = pd.DataFrame(
        [
            {
                "phase": "development",
                "fold": 1,
                "as_of_date": day,
                "symbol": symbol,
                "row_id": f"{day.date()}-{symbol}",
                "model": model,
                "score": score,
                "fit_end_date": dates[0],
                "train_label_end_max": dates[1],
                "evaluation_start": dates[2],
                "feature_count": 1,
                "selected_feature_count": 1,
                "zero_observed_features": False,
                "target": 999.0,
                "realized_return": -999.0,
            }
            for day in dates[2:5]
            for model in ("ridge", "equal_weight_rank")
            for symbol, score in (("A", 3.0), ("B", 2.0), ("C", 1.0))
        ]
    )
    assignments = pd.DataFrame(
        [
            {
                "phase": "development",
                "fold": 1,
                "split_id": "development_fold_0",
                "row_id": row.row_id,
                "role": "evaluation",
                "exclusion_reason": None,
                "train_end_date": dates[0],
                "train_label_end_max": dates[1],
                "evaluation_start": dates[2],
                "evaluation_end": dates[4],
            }
            for row in predictions.loc[predictions.model.eq("ridge")].itertuples()
        ]
    )
    manifest = {
        "artifact_schema_version": "6",
        "execution_mode": "development",
        "configuration": config.to_mapping(),
        "package_version": "fixture",
        "implementation_version": "fixture",
        "python_version": "3.12.0",
        "numpy_version": "2.2.0",
        "pandas_version": "2.3.0",
        "scipy_version": "1.15.0",
        "scikit_learn_version": "1.9.0",
        "dataset_versions": ["fixture"],
        "fingerprint_scope": (
            "source_code+configuration+validated_panel_contract+execution_mode"
        ),
        **{
            key: "a" * 64
            for key in (
                "source_fingerprint",
                "panel_fingerprint",
                "run_fingerprint",
                "model_input_fingerprint",
            )
        },
        "development_start": str(dates[0].date()),
        "locked_test_start": str(dates[9].date()),
        "locked_test_end": str(dates[10].date()),
        "locked_label_end_max": str(dates[11].date()),
        "evaluation_schedule": {
            "definition_version": "1",
            "source": "validated_panel_within_planned_phase_bounds",
            "lag_unit": "scheduled_observations",
            "dates_by_phase": {"development": tuple(str(d.date()) for d in dates[2:5])},
        },
    }
    empty = pd.DataFrame()
    run = BenchmarkRun(
        data_gate=MappingProxyType({"claim_scope": "development_only"}),
        fold_assignments=assignments,
        predictions=predictions,
        daily_metrics=empty,
        fold_metrics=empty,
        tuning_trials=empty,
        screening_by_fold=empty,
        summary=empty,
        acceptance=MappingProxyType(
            {
                "acceptance_status": "not_evaluated",
                "lockbox_evaluated_once_in_this_run": False,
            }
        ),
        manifest=MappingProxyType(manifest),
    )
    prices = pd.DataFrame(
        [
            {"date": day, "symbol": symbol, "adjusted_close": 100.0}
            for day in dates
            for symbol in ("A", "B", "C", "MARKET")
        ]
    )
    return dates, calendar, run, prices
