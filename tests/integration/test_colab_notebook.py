"""Executable notebook structure and reproducible-source contract."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[2]


def builder():
    spec = importlib.util.spec_from_file_location(
        "colab_builder", ROOT / "examples" / "build_colab_notebook.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("revision", ["main", "", "a" * 39, "../source", "G" * 40])
def test_mutable_or_invalid_source_revision_refused(revision):
    with pytest.raises(ValueError, match="40"):
        builder().build_notebook(revision)


def test_notebook_is_valid_clean_deterministic_and_compiles():
    module = builder()
    notebook = module.build_notebook("a" * 40)
    nbformat.validate(notebook)
    assert notebook == module.build_notebook("a" * 40)
    assert notebook.metadata.kernelspec.name == "python3"
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(code) >= 5
    for cell in code:
        assert cell.outputs == []
        assert cell.execution_count is None
        compile(cell.source, f"<cell-{cell.id}>", "exec")
    sources = "\n".join(cell.source for cell in notebook.cells)
    for required in (
        "Goal",
        "Setup",
        "Steps",
        "Checks",
        "Next steps",
        "SOURCE_REVISION",
        "a" * 40,
        "notebook-demo",
        "colab-constraints.txt",
        "QMR_HISTORY_DIR",
        "QMR_LOCAL_REPOSITORY",
        "mean_rank_ic",
        "final_outcomes_evaluated",
        "stage4_eligible",
        "survivorship",
        "no proven alpha",
    ):
        assert required in sources
    assert "--evaluate-lockbox" not in sources
    assert "seed-notebook" not in "\n".join(cell.source for cell in code)


def test_writer_refuses_overwrite_and_invalid_extension(tmp_path):
    module = builder()
    output = tmp_path / "colab.ipynb"
    module.write_notebook("b" * 40, output)
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        module.write_notebook("c" * 40, output)
    assert output.read_bytes() == original
    with pytest.raises(ValueError, match="ipynb"):
        module.write_notebook("b" * 40, tmp_path / "result.txt")
    nbformat.validate(nbformat.read(output, as_version=4))


def test_generated_notebook_passes_repository_lint(tmp_path):
    output = tmp_path / "generated.ipynb"
    builder().write_notebook("b" * 40, output)
    checked = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--config",
            str(ROOT / "pyproject.toml"),
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr


def code_cells():
    return [
        cell.source
        for cell in builder().build_notebook("a" * 40).cells
        if cell.cell_type == "code"
    ]


def test_subprocess_failure_shows_bounded_recovery_diagnostic():
    source = code_cells()[2].split("if LOCAL_REPOSITORY:", 1)[0]
    failed = subprocess.CompletedProcess(
        ["qmr"], returncode=1, stdout="", stderr="private" * 1000 + "manual review"
    )
    namespace = {
        "subprocess": SimpleNamespace(run=lambda *args, **kwargs: failed),
        "PROCESS_ENV": {},
    }
    exec(source, namespace)
    with pytest.raises(RuntimeError, match="manual review") as caught:
        namespace["command"](["qmr"])
    assert "do not blindly retry" in str(caught.value)
    assert len(str(caught.value)) < 4200


def test_setup_creates_only_local_session_and_retains_seed(tmp_path, monkeypatch):
    history = tmp_path / "history"
    archive = history / "000000/evidence/archive"
    archive.mkdir(parents=True)
    (history / "000000/checkpoint.json").write_text("{}")
    (archive / "archive_manifest.json").write_text("{}")
    monkeypatch.setenv("QMR_HISTORY_DIR", str(history))
    monkeypatch.setenv("QMR_SESSION_PARENT", str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", "unrelated-source")
    monkeypatch.setenv("PYTHONHOME", "unrelated-runtime")
    namespace = {}
    exec(code_cells()[0], namespace)
    assert "PYTHONPATH" not in namespace["PROCESS_ENV"]
    assert "PYTHONHOME" not in namespace["PROCESS_ENV"]
    exec(code_cells()[1], namespace)
    assert namespace["SESSION_DIR"].is_dir()
    assert not namespace["WORK_DIR"].exists()
    assert namespace["ARCHIVE_DIR"] == archive
    assert (history / "000000/checkpoint.json").read_text() == "{}"


def test_installation_clones_clean_source_and_uses_isolated_python():
    source = code_cells()[2]
    assert '"clone", "--no-local", "--no-checkout"' in source
    assert '"checkout", "--detach", SOURCE_REVISION' in source
    assert '[sys.executable, "-I", "-m", "venv"' in source
    assert '[PYTHON, "-I", "-m", "pip", "install", REPO_DIR' in source
    assert 'PYTHON, "-I", "-m", "quant_metric_research.cli"' in code_cells()[3]
    assert '"pip", "install", "--upgrade", "pip==26.2.1"' in source


@pytest.mark.parametrize("mount", ["/content/drive", "/content/gdrive"])
def test_notebook_recognizes_only_exact_or_descendant_drive_paths(mount):
    namespace = {}
    exec(code_cells()[0], namespace)
    predicate = namespace["is_drive_path"]
    assert predicate(mount)
    assert predicate(mount + "/MyDrive/work")
    assert not predicate(mount + "-copy")


@pytest.mark.parametrize(
    "unsafe_flag", [None, "final_outcomes_evaluated", "stage4_eligible"]
)
def test_summary_cell_checks_scope_and_displays_only_aggregate(
    tmp_path, capsys, unsafe_flag
):
    destination = tmp_path / "demo/research/development"
    destination.mkdir(parents=True)
    report = {
        "status": "complete",
        "final_outcomes_evaluated": False,
        "stage4_eligible": False,
        "empirical_data_provenance_verified": False,
        "development_summary": [
            {
                "model": model,
                "evaluation_scope": "common",
                "date_count": 189,
                "mean_rank_ic": -0.04,
                "mean_rank_ic_coverage": 1,
                "mean_spread": -0.005,
            }
            for model in ("ridge", "equal_weight_rank")
        ],
    }
    if unsafe_flag:
        report[unsafe_flag] = True
    (tmp_path / "demo/public_demo_report.json").write_text(json.dumps(report))
    (destination / "fold_summary.csv").write_text(
        "model,evaluation_scope,mean_rank_ic\nridge,common,-0.04\n"
    )
    namespace = {"WORK_DIR": tmp_path, "json": json}
    if unsafe_flag:
        with pytest.raises(RuntimeError, match="scope"):
            exec(code_cells()[-1], namespace)
    else:
        exec(code_cells()[-1], namespace)
        output = capsys.readouterr().out
        assert "ridge" in output and "equal_weight_rank" in output
        assert "Final outcomes not evaluated" in output
