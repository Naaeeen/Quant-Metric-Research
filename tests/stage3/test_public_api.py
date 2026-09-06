from __future__ import annotations

import tomllib
from pathlib import Path

import quant_metric_research as qmr
from quant_metric_research.benchmark import IMPLEMENTATION_VERSION


def test_stage3_version_has_one_package_source() -> None:
    project_root = Path(__file__).resolve().parents[2]
    configuration = tomllib.loads(
        (project_root / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert qmr.__version__ == IMPLEMENTATION_VERSION == "0.5.0"
    assert configuration["project"]["dynamic"] == ["version"]
    assert configuration["tool"]["setuptools"]["dynamic"]["version"] == {
        "attr": "quant_metric_research._version.__version__"
    }


def test_stage3_workflow_is_available_from_package_root() -> None:
    assert qmr.BenchmarkConfig is not None
    assert qmr.NestedSplitConfig is not None
    assert qmr.BenchmarkRun is not None
    assert qmr.BenchmarkArtifacts is not None
    assert callable(qmr.run_stage3_benchmark)
    assert callable(qmr.write_benchmark_run)
