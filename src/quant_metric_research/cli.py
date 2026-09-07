from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .benchmark import run_stage3_benchmark
from .benchmark_config import BenchmarkConfig
from .benchmark_io import write_benchmark_run
from .config import PanelConfig
from .experiment_registry import ExperimentRegistry
from .input_audit import audit_inputs
from .intake import import_yahoo_files
from .io import read_as_of_dates, read_json_object, read_table, write_research_run
from .pipeline import run_research
from .preflight import preflight_benchmark
from .public_archive import fetch_public_archive
from .public_demo import run_public_demo
from .validation import WalkForwardMetricConfig


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _minimum_two_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("value must be an integer of at least 2")
    return parsed


def _non_negative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def _unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


def _positive_unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be in (0, 1]")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qmr")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser(
        "fetch-public-sample",
        help="Download two pinned Mendeley CSVs and license records (network access).",
    )
    fetch_parser.add_argument("--output-dir", required=True)
    demo_parser = subparsers.add_parser(
        "public-demo", help="Run the declared public-archive development demo offline."
    )
    for option in ("archive-dir", "output-dir", "registry"):
        demo_parser.add_argument(f"--{option}", required=True)

    import_parser = subparsers.add_parser(
        "import-yahoo",
        help="Snapshot and audit local Yahoo-format exports; no download.",
    )
    for option in ("exports", "memberships", "as-of-dates", "config", "output-dir"):
        import_parser.add_argument(f"--{option}", required=True)

    audit_parser = subparsers.add_parser(
        "audit-inputs",
        help="Inspect raw-input coverage without computing outcomes or training.",
    )
    for option in ("prices", "memberships", "as-of-dates", "config"):
        audit_parser.add_argument(f"--{option}", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--prices", required=True)
    run_parser.add_argument("--memberships", required=True)
    run_parser.add_argument("--as-of-dates", required=True)
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--train-end", required=True)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument(
        "--min-cross-section",
        type=_positive_integer,
        required=True,
    )
    run_parser.add_argument(
        "--minimum-coverage",
        type=_unit_interval,
        default=0.8,
    )
    run_parser.add_argument(
        "--redundancy-threshold",
        type=_unit_interval,
        default=0.9,
    )
    run_parser.add_argument(
        "--hac-lags",
        type=_non_negative_integer,
        required=True,
    )
    run_parser.add_argument(
        "--quantiles",
        type=_minimum_two_integer,
        default=5,
    )
    run_parser.add_argument(
        "--panel-format",
        choices=("csv", "parquet"),
        default="parquet",
    )
    run_parser.add_argument("--with-pca", action="store_true")
    run_parser.add_argument(
        "--pca-variance-to-keep",
        type=_positive_unit_interval,
        default=0.95,
    )
    run_parser.add_argument(
        "--walk-forward-splits",
        type=_positive_integer,
    )
    run_parser.add_argument(
        "--walk-forward-test-date-count",
        type=_positive_integer,
    )
    run_parser.add_argument(
        "--walk-forward-min-train-date-count",
        type=_positive_integer,
    )

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--panel", required=True)
    benchmark_parser.add_argument("--config", required=True)
    benchmark_parser.add_argument("--output-dir", required=True)
    benchmark_parser.add_argument("--registry")
    benchmark_parser.add_argument("--study-id")
    benchmark_parser.add_argument("--hypothesis")
    benchmark_parser.add_argument("--development-run-id")
    benchmark_parser.add_argument(
        "--evaluate-lockbox",
        action="store_true",
        help="Evaluate final outcomes using matching registered development evidence.",
    )
    benchmark_parser.add_argument(
        "--prediction-format",
        choices=("csv", "parquet"),
        default="parquet",
    )
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--panel", required=True)
    preflight_parser.add_argument("--config", required=True)
    history_parser = subparsers.add_parser("experiments")
    history_parser.add_argument("--registry", required=True)
    history_parser.add_argument("--run-id")
    return parser


def _walk_forward_config(
    args: argparse.Namespace,
) -> WalkForwardMetricConfig | None:
    values = (
        args.walk_forward_splits,
        args.walk_forward_test_date_count,
        args.walk_forward_min_train_date_count,
    )
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("All three walk-forward arguments must be provided together.")
    return WalkForwardMetricConfig(
        n_splits=args.walk_forward_splits,
        test_date_count=args.walk_forward_test_date_count,
        min_train_date_count=args.walk_forward_min_train_date_count,
    )


def _validate_cli_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    if args.command == "benchmark":
        if args.evaluate_lockbox:
            if not args.registry or not args.development_run_id:
                parser.error(
                    "--evaluate-lockbox requires --registry and --development-run-id."
                )
            if args.study_id or args.hypothesis:
                parser.error(
                    "Final evaluation uses the registered study and hypothesis."
                )
        elif args.development_run_id:
            parser.error("--development-run-id requires --evaluate-lockbox.")
        elif args.registry:
            if not args.study_id or not args.hypothesis:
                parser.error(
                    "Registered development requires --study-id and --hypothesis."
                )
        elif args.study_id or args.hypothesis:
            parser.error("--study-id and --hypothesis require --registry.")
        return
    if args.command != "run":
        return
    values = (
        args.walk_forward_splits,
        args.walk_forward_test_date_count,
        args.walk_forward_min_train_date_count,
    )
    if all(value is None for value in values):
        return
    if any(value is None for value in values):
        parser.error("All three walk-forward arguments must be provided together.")


def _audit_command(args: argparse.Namespace) -> int:
    report = audit_inputs(
        read_table(Path(args.prices)),
        read_table(Path(args.memberships)),
        as_of_dates=read_as_of_dates(Path(args.as_of_dates)),
        config=PanelConfig(**read_json_object(Path(args.config))),
    )
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 0  # Report generated; not a research-readiness or provenance approval.


def _import_command(args: argparse.Namespace) -> int:
    manifest = import_yahoo_files(
        args.exports,
        memberships_path=args.memberships,
        as_of_dates_path=args.as_of_dates,
        config_path=args.config,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True))
    return 0  # Successful import is not data-provenance or research approval.


def _run_command(args: argparse.Namespace) -> int:
    prices = read_table(Path(args.prices))
    memberships = read_table(Path(args.memberships))
    as_of_dates = read_as_of_dates(Path(args.as_of_dates))
    config = PanelConfig(**read_json_object(Path(args.config)))

    result = run_research(
        prices,
        memberships,
        as_of_dates=as_of_dates,
        config=config,
        train_end_date=args.train_end,
        min_cross_section=args.min_cross_section,
        minimum_coverage=args.minimum_coverage,
        redundancy_threshold=args.redundancy_threshold,
        hac_lags=args.hac_lags,
        quantiles=args.quantiles,
        run_pca=bool(args.with_pca),
        pca_variance_to_keep=args.pca_variance_to_keep,
        walk_forward_config=_walk_forward_config(args),
    )

    write_research_run(
        result,
        Path(args.output_dir),
        panel_format=args.panel_format,
    )

    return 0


def _benchmark_command(args: argparse.Namespace) -> int:
    destination = Path(args.output_dir)
    if destination.exists():
        raise FileExistsError(f"Output directory already exists: {destination}")
    if args.evaluate_lockbox:
        print(
            "Final evaluation consumes a durable reservation in this local registry, "
            "including on failure. It does not prove data provenance or tradability.",
            file=sys.stderr,
        )
    panel = read_table(Path(args.panel))
    config = BenchmarkConfig.from_mapping(read_json_object(Path(args.config)))
    result = run_stage3_benchmark(
        panel,
        config=config,
        evaluate_lockbox=args.evaluate_lockbox,
        registry=ExperimentRegistry(args.registry) if args.registry else None,
        study_id=args.study_id,
        hypothesis=args.hypothesis,
        development_run_id=args.development_run_id,
    )

    write_benchmark_run(
        result,
        Path(args.output_dir),
        prediction_format=args.prediction_format,
    )

    return 0


def _preflight_command(args: argparse.Namespace) -> int:
    report = preflight_benchmark(
        read_table(Path(args.panel)),
        config=BenchmarkConfig.from_mapping(read_json_object(Path(args.config))),
    )
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 0 if report["feasible"] else 2


def _history_command(args: argparse.Namespace) -> int:
    path = Path(args.registry)
    if not path.is_file():
        raise FileNotFoundError(f"Registry does not exist: {path}")
    registry = ExperimentRegistry(path)
    records = registry.get_run(args.run_id) if args.run_id else registry.list_runs()
    print(json.dumps(records, allow_nan=False, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_cli_args(args, parser)
    if args.command == "fetch-public-sample":
        print(
            json.dumps(fetch_public_archive(args.output_dir), allow_nan=False, indent=2)
        )
        return 0
    if args.command == "public-demo":
        report = run_public_demo(
            archive_dir=args.archive_dir,
            output_dir=args.output_dir,
            registry_path=args.registry,
        )
        print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
        return 0
    if args.command == "import-yahoo":
        return _import_command(args)
    if args.command == "audit-inputs":
        return _audit_command(args)
    if args.command == "run":
        return _run_command(args)
    if args.command == "benchmark":
        return _benchmark_command(args)
    if args.command == "preflight":
        return _preflight_command(args)
    if args.command == "experiments":
        return _history_command(args)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
