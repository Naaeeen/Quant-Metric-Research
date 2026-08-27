from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .pipeline import ResearchRun


def read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Input table does not exist: {source}")
    lowered_name = source.name.lower()
    if lowered_name.endswith((".csv", ".csv.gz")):
        return pd.read_csv(source)
    if lowered_name.endswith((".parquet", ".pq")):
        return pd.read_parquet(source)
    raise ValueError("Input tables must be CSV, CSV.GZ, Parquet, or PQ files.")


def read_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Configuration JSON must contain an object.")
    return value


def read_as_of_dates(path: str | Path) -> tuple[pd.Timestamp, ...]:
    frame = read_table(path)
    if "as_of_date" not in frame.columns:
        raise ValueError("as-of-date input must contain an as_of_date column.")
    parsed = pd.to_datetime(frame["as_of_date"], errors="coerce")
    if parsed.isna().any() or parsed.empty:
        raise ValueError("as_of_date contains invalid or empty values.")
    dates = tuple(pd.Timestamp(value) for value in parsed)
    if len(set(dates)) != len(dates):
        raise ValueError("as_of_date values must be unique.")
    return dates


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)


def write_research_run(
    run: ResearchRun,
    output_dir: str | Path,
    *,
    panel_format: str,
) -> None:
    if panel_format not in {"csv", "parquet"}:
        raise ValueError("panel_format must be csv or parquet.")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    _write_frame(
        run.panel,
        destination / f"metric_panel.{panel_format}",
    )
    report_frames = {
        "feature_quality.csv": run.screen.quality,
        "daily_rank_ic.csv": run.screen.daily_rank_ic,
        "rank_ic_summary.csv": run.screen.ic_summary,
        "quantile_spreads.csv": run.screen.quantile_spreads,
        "feature_redundancy.csv": run.screen.redundancy,
    }
    for filename, frame in report_frames.items():
        frame.to_csv(destination / filename, index=False)

    selection = {
        "fitted_through": str(run.screen.fitted_through.date()),
        "training_row_count": run.screen.training_row_count,
        "selected_features": list(run.screen.selected_features),
        "dropped_features": dict(run.screen.dropped_features),
    }
    (destination / "feature_selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if run.walk_forward is not None:
        run.walk_forward.fold_metrics.to_csv(
            destination / "walk_forward_metric_folds.csv",
            index=False,
        )
        run.walk_forward.summary.to_csv(
            destination / "walk_forward_metric_summary.csv",
            index=False,
        )

    if run.pca is None or run.pca_scores is None:
        return
    run.pca_scores.to_parquet(
        destination / "pca_scores.parquet",
        index=False,
    )
    component_weights = pd.DataFrame(
        run.pca.components,
        columns=run.pca.feature_columns,
    )
    component_weights.insert(
        0,
        "component",
        [f"pc_{position + 1}" for position in range(component_weights.shape[0])],
    )
    component_weights.to_csv(
        destination / "pca_component_weights.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "component": [
                f"pc_{position + 1}"
                for position in range(run.pca.explained_variance_ratio.shape[0])
            ],
            "explained_variance_ratio": (run.pca.explained_variance_ratio),
        }
    ).to_csv(
        destination / "pca_explained_variance.csv",
        index=False,
    )
