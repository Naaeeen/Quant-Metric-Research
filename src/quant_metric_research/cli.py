from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark import run_stage3_benchmark
from .benchmark_config import BenchmarkConfig
from .benchmark_io import write_benchmark_run
from .config import PanelConfig
from .io import read_as_of_dates, read_json_object, read_table, write_research_run
from .pipeline import run_research
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
    benchmark_parser.add_argument(
        "--prediction-format",
        choices=("csv", "parquet"),
        default="parquet",
    )
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
    panel = read_table(Path(args.panel))
    config = BenchmarkConfig.from_mapping(read_json_object(Path(args.config)))
    result = run_stage3_benchmark(panel, config=config)

    write_benchmark_run(
        result,
        Path(args.output_dir),
        prediction_format=args.prediction_format,
    )

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_cli_args(args, parser)
    if args.command == "run":
        return _run_command(args)
    if args.command == "benchmark":
        return _benchmark_command(args)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
