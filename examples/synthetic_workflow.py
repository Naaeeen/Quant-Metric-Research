"""Exercise the public research CLI using invented, deterministic observations."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_HYPOTHESIS = "Synthetic plumbing check, no alpha claim"


def _synthetic_panel() -> pd.DataFrame:
    """Invent a two-feature relationship; this is not a simulated price history."""
    random = np.random.default_rng(20260907)
    dates = pd.bdate_range("2024-01-02", periods=38)
    records = []
    for date_index, as_of_date in enumerate(dates[:36]):
        for stock_index in range(8):
            signal_a, signal_b = random.normal(size=2)
            target = 0.015 * signal_a + 0.01 * signal_b + random.normal(0.0, 0.003)
            records.append(
                {
                    "as_of_date": as_of_date,
                    "symbol": f"SYNTHETIC_{stock_index:02d}",
                    "label_end_date": dates[date_index + 2],
                    "signal_a": signal_a,
                    "signal_b": signal_b,
                    "forward_excess_return": target if date_index < 34 else np.nan,
                    "dataset_version": "synthetic-workflow-v1",
                    "universe_id": "synthetic-eight-stocks",
                }
            )
    return pd.DataFrame.from_records(records)


def _benchmark_config() -> dict[str, Any]:
    return {
        "feature_columns": ["signal_a", "signal_b"],
        "model_families": ["ridge"],
        "ridge_alphas": [1.0],
        "min_cross_section": 8,
        "quantiles": 2,
        "hac_lags": 1,
        "split": {
            "final_test_date_count": 6,
            "outer_n_splits": 1,
            "outer_test_date_count": 6,
            "outer_min_train_date_count": 14,
            "inner_n_splits": 1,
            "inner_validation_date_count": 4,
            "inner_min_train_date_count": 8,
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _cli(*arguments: str, expect_success: bool = True) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        [sys.executable, "-m", "quant_metric_research.cli", *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if expect_success and completed.returncode != 0:
        raise RuntimeError(
            f"CLI {arguments[0]} failed:\n{completed.stdout}{completed.stderr}"
        )
    return completed


def _manifest_run_id(output: Path) -> str:
    manifest = json.loads((output / "benchmark_manifest.json").read_text("utf-8"))
    return str(manifest["experiment"]["run_id"])


def _check_non_empirical_outputs(output: Path) -> None:
    for phase in ("development", "final"):
        acceptance = json.loads((output / phase / "acceptance.json").read_text("utf-8"))
        if (
            acceptance["stage4_eligible"] is not False
            or acceptance["empirical_data_provenance_verified"] is not False
        ):
            raise RuntimeError("Synthetic evidence must not be eligible for Stage 4.")


def _run_final(
    output: Path, common: tuple[str, ...], development_run_id: str
) -> subprocess.CompletedProcess:
    final_arguments = (
        "benchmark",
        *common,
        "--evaluate-lockbox",
        "--development-run-id",
        development_run_id,
    )
    _cli(*final_arguments, "--output-dir", str(output / "final"))
    repeated = _cli(
        *final_arguments,
        "--output-dir",
        str(output / "refused-attempt"),
        expect_success=False,
    )
    if (
        repeated.returncode == 0
        or "previously exposed outcome dates" not in repeated.stderr
        or (output / "refused-attempt").exists()
    ):
        raise RuntimeError("The registry did not refuse a repeated final evaluation.")
    return repeated


def run_example(output: Path) -> dict[str, Any]:
    # Never delete or overwrite an old experiment, even if an earlier run failed.
    output.mkdir(parents=True, exist_ok=False)
    panel_path, config_path = output / "panel.csv", output / "benchmark-config.json"
    _synthetic_panel().to_csv(panel_path, index=False)
    _write_json(config_path, _benchmark_config())
    inputs = ("--panel", str(panel_path), "--config", str(config_path))
    preflight = json.loads(_cli("preflight", *inputs).stdout)
    _write_json(output / "preflight.json", preflight)
    registry = output / "experiments.sqlite3"
    common = (*inputs, "--registry", str(registry), "--prediction-format", "csv")
    _cli(
        "benchmark",
        *common,
        "--output-dir",
        str(output / "development"),
        "--study-id",
        "synthetic-workflow-demo",
        "--hypothesis",
        _HYPOTHESIS,
    )
    development_run_id = _manifest_run_id(output / "development")
    repeated = _run_final(output, common, development_run_id)
    history = json.loads(_cli("experiments", "--registry", str(registry)).stdout)
    _write_json(output / "experiments.json", history)
    if len(history) != 2 or any(run["status"] != "completed" for run in history):
        raise RuntimeError("Expected exactly two completed registered experiments.")
    _check_non_empirical_outputs(output)
    report = {
        "synthetic_data": True,
        "empirical_alpha_claim": False,
        "stage4_eligible": False,
        "development_run_id": development_run_id,
        "final_run_id": _manifest_run_id(output / "final"),
        "repeat_final_refused": True,
        "repeat_final_exit_code": repeated.returncode,
        "repeat_final_message": repeated.stderr.strip(),
        "output_dir": str(output),
        "warning": "Invented data tests plumbing, not alpha or provider provenance.",
    }
    _write_json(output / "demo-report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        parser.error(f"Output directory already exists: {output}")
    try:
        report = run_example(output)
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        parser.exit(2, f"Synthetic workflow failed: {error}\n")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
