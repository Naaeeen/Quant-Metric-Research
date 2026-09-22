# Quant Metric Research

Build equity metric panels, screen individual signals, and compare stock-ranking
models on purged walk-forward folds. The research question is whether combining
historical metrics improves rankings on unseen dates.

The pipeline has four parts:

1. Build a dated stock panel from adjusted-close prices, universe membership,
   trailing metrics, and forward targets.
2. Check coverage, measure Rank IC and quantile spreads, and identify redundant
   metrics. Optional walk-forward and PCA comparisons test the selected metrics.
3. Compare individual metrics and equal-weight ranks with Ridge, histogram
   gradient boosting, and optional Ridge+PCA using nested purged folds.
4. Evaluate saved development rankings with a cash-funded holdings ledger,
   delayed execution, drift-aware turnover and explicit costs.

## Current status

Both declared development studies are complete. Expanded raw-return histogram
boosting improved over the original ten inputs, but a simple moving-average
signal scored higher. The next experiment must test what ML adds beyond that
signal and whether the result survives costs. These findings come from a
retrospective 30-stock sample; see the [target comparison](docs/research-decisions.md#corrected-panel-target-study-results)
and [feature comparison](docs/research-decisions.md#feature-bundle-study-results).

Version 0.17 adds [development holdings accounting](docs/stage3-benchmark.md#development-holdings-and-cost-accounting)
and an offline example that trains Ridge on invented prices and evaluates
continuous accounts across two folds. Qualified broader data, negative controls
and a declared market-data cost comparison remain open. Final evaluation has not run.
The Colab walkthrough remains pinned
to the separately verified 0.13 source; its fixed ten-metric Ridge experiment is
unchanged. Local notebook checks and hosted Colab execution have separate
[verification records](examples/README.md#verification-status).

## Install

Use Python 3.11 or later. Create and activate a virtual environment from the
repository directory, then install the package.

Windows PowerShell:

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
~~~

macOS or Linux:

~~~sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
~~~

For a deterministic offline check:

~~~text
python examples/synthetic_workflow.py --output-dir artifacts/synthetic-demo-001
~~~

This [synthetic workflow](examples/README.md#offline-synthetic-workflow) trains a
small Ridge model on invented data and checks that a repeated final evaluation
is refused. Use a fresh output directory for each run.

## Choose a workflow

| Starting point | Guide |
| --- | --- |
| Continue the existing experiment in Colab | [Colab walkthrough](examples/README.md#colab-development-walkthrough) |
| Run the declared public US archive experiment | [Public-archive demo](examples/README.md#public-archive-development-demo) |
| Import authorized local Yahoo-format exports | [Offline intake](examples/README.md#offline-yahoo-format-intake) |
| Supply normalized prices and dated memberships | [Data contract](docs/data-contract.md) |
| Compare models or feature bundles | [Stage 3 benchmark](docs/stage3-benchmark.md) |
| Compare raw-return and rank training targets | [Target ablation](examples/README.md#target-ablation) |
| Run the fixed ten/thirteen-feature comparison | [Feature ablation](examples/README.md#feature-ablation) |
| Train and check a synthetic holdings/cost workflow | [Portfolio example](examples/README.md#synthetic-holdings-and-cost-workflow) |

The Colab workflow requires a private seed containing the existing canonical
registry, archive, and prior-run evidence. Keep the live SQLite database on local
VM storage and use the documented checkpoints for persistence. Preserve the
experiment history when moving environments.

To run the public-archive development demo locally:

~~~text
qmr fetch-public-sample --output-dir data/raw/mendeley-v3-001
qmr public-demo --archive-dir data/raw/mendeley-v3-001 --output-dir artifacts/public-demo-001 --registry artifacts/research-registry.sqlite3
~~~

The download verifies pinned file sizes and hashes. The demo retains the first
30 stock labels alphabetically, audits coverage, builds the panel, checks the
evaluation schedule, and runs registered Ridge development. Its final output is
`public_demo_report.json`. The guide lists fixed settings and recovery steps.

For your own inputs, the [data contract](docs/data-contract.md) specifies prices,
historical memberships, decision dates, and configuration. Review the provider
evidence and coverage before building a panel. For a prepared Stage 3 panel,
check the schedule and register development:

~~~text
qmr preflight --panel artifacts/run-001/metric_panel.parquet --config benchmark-config.json
qmr benchmark --panel artifacts/run-001/metric_panel.parquet --config benchmark-config.json --output-dir artifacts/benchmark-001 --registry artifacts/research-registry.sqlite3 --study-id metrics-v1 --hypothesis "Combined metrics improve unseen-date ranking over equal-weight ranks."
~~~

These commands require your panel and benchmark configuration. The
[benchmark guide](docs/stage3-benchmark.md#run-and-outputs) explains configuration,
outputs, and final evaluation. Preflight checks structural feasibility; model
fitting and statistical evaluation happen during the benchmark.

## Research assumptions and limits

Features use observations available at the decision time. Targets begin after
the configured entry lag and use the declared sessions when supplied, otherwise
the observed benchmark dates. New empirical studies should supply a calendar
from an independent source covering the full input period. Missing observations
and immature labels remain visible in coverage reports. Screening, preprocessing,
and model fitting use training partitions; purging removes training labels that
overlap evaluation dates. Rankings use the full scoring universe before outcome
availability is considered. See the [data contract](docs/data-contract.md) for
the precise timing, missingness, and weighting rules.

Keep one durable registry for related research. Final evaluation requires
`--evaluate-lockbox`, a matching completed `--development-run-id`, and a reservation
of the final outcome dates. Failed or interrupted reservations remain consumed.
The registry tracks recorded exposure; prior inspection and unregistered research
still affect the independence of a final test. Follow the
[final-evaluation procedure](docs/stage3-benchmark.md#3-explicitly-evaluate-the-referenced-final-test)
before opening it.

The public demo uses a retrospective January 2017 constituent snapshot, which
introduces survivorship bias, and a constructed `FF_MARKET_PROXY` return index.
The publisher declares CC BY 4.0; underlying third-party rights and historical
provenance still need independent review. Keep raw data and row-level artifacts
local. The proposed ASX cohort also needs provider access and historical evidence;
current constituents cannot establish historical membership.

Rank IC and quantile spreads describe ranking quality. Assessing tradability also
requires costs, turnover, liquidity, risk exposures, and execution assumptions.
Develop those checks alongside model experiments and freeze the complete procedure
before final evaluation. Promotion beyond development requires verified
point-in-time data and a passed, predeclared real-data final evaluation. See the
[roadmap](docs/roadmap.md) for the plan and completion criteria.

## Further reading

- [Data contract](docs/data-contract.md): input formats, timing, and provider evidence.
- [Benchmark guide](docs/stage3-benchmark.md): models, inference, registry, and reports.
- [Research decisions](docs/research-decisions.md): declarations, results, and sources.
- [Roadmap](docs/roadmap.md): completed capabilities and next experiments.
- [Examples](examples/README.md): runnable workflows and verification records.
