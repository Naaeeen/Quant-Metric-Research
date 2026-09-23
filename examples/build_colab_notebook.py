"""Generate an output-free tutorial pinned to an already published source commit."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from textwrap import dedent

import nbformat


def build_notebook(source_revision: str):
    if re.fullmatch(r"[0-9a-f]{40}", source_revision) is None:
        raise ValueError("Use a complete, lowercase 40-character Git commit SHA.")
    sections = [
        (
            "markdown",
            """
        # Goal: a reproducible, development-only stock-ranking experiment

        Run the existing fixed 30-stock, ten-metric Ridge experiment on CPU.
        This notebook calls the tested library; it does not implement another model.
        It preserves earlier attempts in one private append-only checkpoint history.
        **This is an engineering demo, with no proven alpha.** In the earlier
        version 0.7/0.8 real-data development runs, Ridge mean Rank IC was -0.0477
        versus -0.0370 for equal-weight ranks (not a new result from this source pin).
        Retrospective membership creates survivorship bias; provider adjustments,
        delistings and point-in-time availability remain unverified.

        ## Setup

        Use a CPU runtime (no GPU purchase needed). First upload the privately
        prepared seeded history folder to your own Drive as `qmr-colab-history`.
        Keep all generations. Its `000000/evidence/archive` must contain the
        verified archive, and its registry must include the earlier attempts.
        See the repository's examples guide for the one-time local preparation.
        **Do not create a blank registry or discard a failed generation.**

        Run all cells in order with one writer. Mounting Drive requires your
        authorization; this notebook can then access the mounted files.
        The source and dependencies are public, but data/history and executed
        outputs remain private. Clear outputs before sharing a notebook.

        This refreshed source has local isolated-install and synthetic workflow
        checks plus Linux CI. The earlier five-cell real-data execution used the
        0.8 pin; the full refreshed notebook and hosted Colab/Drive interaction
        still require your execution. Follow every cell in order with the complete
        existing history; do not reuse an older seed or erase prior attempts.
        """,
        ),
        (
            "code",
            f"""
        import json
        import os
        import subprocess
        import sys
        import tempfile
        from pathlib import Path

        SOURCE_REVISION = "{source_revision}"
        SOURCE_URL = "https://github.com/Naaeeen/Quant-Metric-Research.git"
        HISTORY_DIR = Path(os.environ.get(
            "QMR_HISTORY_DIR", "/content/drive/MyDrive/qmr-colab-history"
        )).expanduser()
        SESSION_PARENT = Path(os.environ.get("QMR_SESSION_PARENT", "/content"))
        LOCAL_REPOSITORY = os.environ.get("QMR_LOCAL_REPOSITORY")
        def is_drive_path(path):
            normalized = str(path).replace("\\\\", "/")
            return any(normalized == mount or normalized.startswith(mount + "/")
                       for mount in ("/content/drive", "/content/gdrive"))

        PROCESS_ENV = {{**{{key: value for key, value in os.environ.items()
                          if key.upper() not in ("PYTHONPATH", "PYTHONHOME")}},
                       "OMP_NUM_THREADS": "1",
                       "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}}
        # QMR_* overrides support documented local validation; not model tuning.
        print("Pinned source:", SOURCE_REVISION)
        print("CPU only; one sequential writer; existing private history required.")
        """,
        ),
        (
            "markdown",
            """
        ## Steps — 1. Connect private history and create local working storage

        The live database, source checkout and Python environment stay on the
        runtime's local disk, not Drive. Only closed snapshots and evidence files
        go to history. Drive read-back does not prove remote server durability.
        Never launch two copies against the same history.
        """,
        ),
        (
            "code",
            """
        if str(HISTORY_DIR).replace("\\\\", "/").startswith("/content/drive/"):
            from google.colab import drive
            drive.mount("/content/drive")
        HISTORY_DIR = HISTORY_DIR.resolve(strict=True)
        SESSION_PARENT = SESSION_PARENT.expanduser().resolve(strict=True)
        if SESSION_PARENT == HISTORY_DIR or HISTORY_DIR in SESSION_PARENT.parents:
            raise ValueError("Compute must remain outside the history directory.")
        if is_drive_path(SESSION_PARENT):
            raise ValueError("Use local VM storage for compute, not mounted Drive.")
        if not (HISTORY_DIR / "000000" / "checkpoint.json").is_file():
            raise ValueError("Upload the complete seeded history before running.")
        ARCHIVE_DIR = HISTORY_DIR / "000000" / "evidence" / "archive"
        if not (ARCHIVE_DIR / "archive_manifest.json").is_file():
            raise ValueError("The seeded history is missing its offline archive.")
        SESSION_DIR = Path(tempfile.mkdtemp(prefix="qmr-session-", dir=SESSION_PARENT))
        WORK_DIR = SESSION_DIR / "run"
        print("New local session:", SESSION_DIR.name)
        """,
        ),
        (
            "markdown",
            """
        ### 2. Install the exact source into a separate Python environment

        The immutable commit prevents a changing branch from silently changing
        this run. Exact research-library versions come from the constraints file;
        this is not a full OS/hardware or package-hash lock. The dedicated venv
        avoids changing Colab's preloaded notebook kernel. Installation requires
        network access; the subsequent fixed experiment uses the offline archive.
        Keep this cell's dependency report with the private executed notebook.
        """,
        ),
        (
            "code",
            """
        def command(arguments, *, cwd=None):
            result = subprocess.run(
                [str(item) for item in arguments], cwd=cwd, env=PROCESS_ENV,
                check=False, capture_output=True, text=True,
            )
            if result.returncode:
                raise RuntimeError(
                    f"Command failed ({result.returncode}). Keep the local session "
                    "and history; do not blindly retry. Private diagnostic tail:\\n"
                    + result.stderr[-4000:]
                )
            return result.stdout.strip()

        if LOCAL_REPOSITORY:
            local_source = Path(LOCAL_REPOSITORY).expanduser().resolve(strict=True)
            if is_drive_path(local_source):
                raise ValueError("Local validation source must not be on Drive.")
            clone_source = local_source
        else:
            clone_source = SOURCE_URL
        # A fresh clone excludes local untracked or modified build-control files.
        REPO_DIR = SESSION_DIR / "source"
        command(["git", "clone", "--no-local", "--no-checkout",
                 clone_source, REPO_DIR])
        command(["git", "checkout", "--detach", SOURCE_REVISION], cwd=REPO_DIR)
        if command(["git", "rev-parse", "HEAD"], cwd=REPO_DIR) != SOURCE_REVISION:
            raise ValueError("Source identity mismatch.")
        VENV_DIR = SESSION_DIR / "venv"
        command([sys.executable, "-I", "-m", "venv", VENV_DIR])
        PYTHON = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        command([PYTHON, "-I", "-m", "pip", "install", "--upgrade", "pip==26.2.1"])
        command([PYTHON, "-I", "-m", "pip", "install", REPO_DIR,
                 "-c", REPO_DIR / "examples" / "colab-constraints.txt"])
        print(command([PYTHON, "-I", "-m", "pip", "check"]))
        print(command([PYTHON, "-I", "-m", "pip", "freeze"]))
        """,
        ),
        (
            "markdown",
            """
        ### 3. Restore history, train and checkpoint the fixed development run

        This step trains Ridge (`alpha=1`) with fold-local screening and purged,
        nested chronological validation. It compares with existing simple
        baselines; it does not search additional factors, models or horizons.
        Allow several minutes (runtime-dependent). The command validates every
        checkpoint, restores the latest registry locally, writes a start marker
        before training, then appends the new closed registry and evidence.

        On a catchable failure it tries to retain partial evidence. An abrupt VM
        loss or incomplete checkpoint blocks another automatic run. Preserve
        history and the local session, then inspect the failed generation; do
        not delete it, use an older snapshot, or blindly rerun this cell.
        """,
        ),
        (
            "code",
            """
        completed = command([
            PYTHON, "-I", "-m", "quant_metric_research.cli", "notebook-demo",
            "--archive-dir", ARCHIVE_DIR, "--work-dir", WORK_DIR,
            "--history-dir", HISTORY_DIR,
        ], cwd=REPO_DIR)
        checkpoint = json.loads(completed)
        if checkpoint["status"] != "completed":
            raise RuntimeError("The development checkpoint did not complete.")
        print(json.dumps({key: checkpoint[key] for key in
              ("generation", "status", "run_count", "exposure_count",
               "final_outcomes_evaluated")}, indent=2))
        """,
        ),
        (
            "markdown",
            """
        ## Checks — read bounded development summaries, not raw observations

        Rank IC measures whether the score ordering agrees with forward returns.
        Compare Ridge with equal-weight ranks across folds, not only one average.
        The spread is an overlapping 20-session descriptive signal statistic,
        not an annual return, a net-of-cost backtest or a tradable portfolio.
        Reserved final outcomes are not evaluated here; the supplied panel does
        contain forward labels, so this is not a physically sealed holdout.
        """,
        ),
        (
            "code",
            """
        import csv

        report_path = WORK_DIR / "demo" / "public_demo_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (report["status"] != "complete"
                or report["final_outcomes_evaluated"]
                or report["stage4_eligible"]
                or report["empirical_data_provenance_verified"]):
            raise RuntimeError("Unexpected completion or claim-scope state.")
        selected = [
            {key: row[key] for key in ("model", "date_count", "mean_rank_ic",
                                      "mean_rank_ic_coverage", "mean_spread")}
            for row in report["development_summary"]
            if row["evaluation_scope"] == "common"
            and row["model"] in ("ridge", "equal_weight_rank")
        ]
        if len(selected) != 2:
            raise RuntimeError("Expected both comparable development summaries.")
        print(json.dumps(selected, indent=2, allow_nan=False))
        fold_path = WORK_DIR / "demo/research/development/fold_summary.csv"
        with fold_path.open(encoding="utf-8", newline="") as stream:
            folds = [row for row in csv.DictReader(stream)
                     if row["model"] in ("ridge", "equal_weight_rank")
                     and row["evaluation_scope"] == "common"]
        print("Development fold summaries:")
        print(json.dumps(folds[:6], indent=2, allow_nan=False))
        print("Final outcomes not evaluated; provenance unverified; "
              "Stage 4 ineligible.")
        """,
        ),
        (
            "markdown",
            """
        ## Next steps

        Keep the entire private history directory, including the new generation,
        and verify it is available after Drive sync. Clear notebook outputs before
        sharing. Do not treat rerunning an observed dataset as fresh evidence.

        The library now includes an opt-in price-only factor catalog and a
        development-only same-family feature-bundle comparison. This notebook
        does not invoke them or change its fixed ten-metric experiment. A separate
        predeclared study would test candidate factors while retaining every
        attempt and checking availability, leakage and coverage. Additional models
        or PCA are comparisons to earn, not a
        remedy assumed to fix a negative result. Independent historical evidence,
        costs, turnover and portfolio tests are needed before any trading claim.

        Sources: [Colab FAQ](https://research.google.com/colaboratory/faq.html),
        [runtime guidance](https://research.google.com/colaboratory/runtime-version-faq.html),
        [SQLite network-storage caveats](https://sqlite.org/useovernet.html),
        [Qlib workflow](https://qlib.readthedocs.io/en/latest/component/workflow.html).
        """,
        ),
    ]
    cells = []
    for index, (kind, source) in enumerate(sections):
        factory = (
            nbformat.v4.new_code_cell
            if kind == "code"
            else nbformat.v4.new_markdown_cell
        )
        cells.append(factory(dedent(source).strip(), id=f"qmr-colab-{index:02d}"))
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
    )
    nbformat.validate(notebook)
    return notebook


def write_notebook(source_revision: str, output: Path) -> None:
    if output.suffix != ".ipynb":
        raise ValueError("Output must have the .ipynb extension.")
    notebook = build_notebook(source_revision)
    with output.open("x", encoding="utf-8") as stream:
        nbformat.write(notebook, stream)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    write_notebook(arguments.source_revision, arguments.output)
