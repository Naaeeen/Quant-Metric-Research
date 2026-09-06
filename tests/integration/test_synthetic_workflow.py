from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

_EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "synthetic_workflow.py"


def test_offline_example_exercises_registered_cli_workflow(tmp_path: Path) -> None:
    output = tmp_path / "synthetic-demo"
    completed = subprocess.run(
        [sys.executable, str(_EXAMPLE), "--output-dir", str(output)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(completed.stdout)
    assert report["synthetic_data"] is True
    assert report["empirical_alpha_claim"] is False
    assert report["stage4_eligible"] is False
    assert report["repeat_final_refused"] is True
    assert "previously exposed outcome dates" in report["repeat_final_message"]
    assert report["development_run_id"] != report["final_run_id"]
    assert (output / "preflight.json").is_file()
    assert (output / "experiments.json").is_file()
    assert not (output / "refused-attempt").exists()

    development = pd.read_csv(output / "development" / "oos_predictions.csv")
    final = pd.read_csv(output / "final" / "oos_predictions.csv")
    assert set(development["phase"]) == {"development"}
    assert "locked_test" in set(final["phase"])
    for phase in ("development", "final"):
        acceptance = json.loads((output / phase / "acceptance.json").read_text("utf-8"))
        assert acceptance["stage4_eligible"] is False
        assert acceptance["empirical_data_provenance_verified"] is False

    original_panel = (output / "panel.csv").read_bytes()
    repeated = subprocess.run(
        [sys.executable, str(_EXAMPLE), "--output-dir", str(output)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert repeated.returncode == 2
    assert "already exists" in repeated.stderr
    assert (output / "panel.csv").read_bytes() == original_panel
