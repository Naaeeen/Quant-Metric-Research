# Quant Metric Research

A standalone, leakage-aware research pipeline for testing whether historical
equity metrics contain stable cross-sectional signal.

The repository now covers three research stages:

1. Build a point-in-time `(as_of_date, symbol)` panel from adjusted-close data,
   dated universe membership, trailing metrics, and explicit forward targets.
2. Audit metric quality, measure per-date Rank IC and quantile spreads, identify
   redundant metrics, optionally validate them on purged walk-forward folds,
   and optionally compare a train-only PCA baseline.
3. Benchmark non-ML rank composites against supervised cross-sectional models
   with nested purged development folds and an explicitly opened final test.

It does not claim to be a complete trading system. Portfolio construction,
cost-aware backtesting, risk controls, execution, and live monitoring come
after these research stages.

## Design boundaries

- Historical universe membership is required for research-valid results.
- Features use only observations at or before `as_of_date`.
- Targets start after a configurable entry lag and use benchmark trading
  sessions, not spreadsheet row counts or calendar-day offsets.
- Missing or immature labels remain visible instead of being silently filled.
- Screening, imputation, scaling, PCA, and model fitting are fold-local.
- Training rows are purged whenever their labels overlap an evaluation fold.
- Model-loss weights give every training date equal total weight, so dates with
  more listed securities do not dominate the supervised objective. Fold-local
  imputation, scaling, and PCA remain row-weighted preprocessing steps.
- Stage 3 defaults to development only. Opening the final test requires
  `--evaluate-lockbox`, a durable local `--registry`, and a matching completed
  `--development-run-id`. Recorded outcome-date exposure blocks overlapping
  final tests within that registry, even after a failed or interrupted run.
  This guards accidental reuse, not deliberate bypass or prior human inspection.
- Lockbox dates must meet the configured minimum target cross-section. The
  model gate also requires native score and spread coverage, enough valid
  Rank-IC and spread dates, strict improvement over the baseline, and the
  predeclared paired-improvement inference bound.
- PCA is an optional compression comparator, not the definition of a useful
  metric or a required production step.

See `docs/data-contract.md` and `docs/research-decisions.md` before adding a
data provider or model.

## What “useful” means here

A metric is a research candidate when it has sufficient point-in-time
coverage, non-trivial cross-sectional variation, and a reasonably stable
relationship with the forward target. A stronger candidate should also retain
Rank IC or top-minus-bottom spread on unseen walk-forward dates and should not
be only a duplicate of a better metric.

That is still not proof of tradable alpha. Costs, turnover, liquidity, risk
exposures, data provenance, and model-selection bias are separate gates.

## Install

~~~text
python -m venv .venv
python -m pip install -e ".[dev]"
~~~

## Inputs

The Stage 1 CLI accepts CSV, compressed CSV, or Parquet tables:

- prices: date, symbol, adjusted_close;
- memberships: universe_id, symbol, effective_from, effective_to, source;
- decision dates: one as_of_date column;
- config: a JSON object matching PanelConfig.

Example config:

~~~json
{
  "dataset_version": "asx-v1",
  "universe_id": "ASX200",
  "benchmark_symbol": "^AXJO",
  "lookback_sessions": 252,
  "min_observations": 126,
  "target_horizon_sessions": 20,
  "entry_lag_sessions": 1,
  "annualization_sessions": 252,
  "annual_risk_free_rate": 0.04
}
~~~

The existing application's current constituent file is not historical
membership data. It can support a demo, but not an unbiased historical claim.

Stage 3 accepts a versioned panel containing `as_of_date`, `symbol`,
`label_end_date`, the configured feature columns, and the target/realized-return
columns. See `docs/data-contract.md` before treating any panel as research-grade.

## Run

Start with the version 0.5 offline raw-input audit:

~~~text
qmr audit-inputs --prices data/prices.parquet --memberships data/memberships.parquet --as-of-dates data/as_of_dates.csv --config config.json
~~~

This prints JSON coverage by date and security: active historical members,
adjacent historical price pairs, missing decision prices, and future label
endpoint availability. It does not calculate returns, screen metrics, train a
model, or write artifacts. Exit code 0 means the report was generated, **not**
that the data is approved. Invalid contracts fail; coverage gaps remain warnings.
Daily inputs must be normalized timezone-naive dates; ambiguous prices and
duplicate CSV column names are rejected instead of silently repaired.

Fingerprints identify normalized required columns and the request, not raw file
bytes or provider provenance. Future price presence is inspected, so this is
not a sealed holdout. The benchmark calendar is inferred from supplied prices,
not independently verified. All external-evidence checks stay unverified and
Stage 4 stays ineligible. Complete the human evidence checklist in
[the data contract](docs/data-contract.md) before an empirical experiment;
keep licensed input data and reports out of public commits.

Once that intake review is complete, build the panel and run Stage 2:

~~~text
qmr run --prices data/prices.parquet --memberships data/memberships.parquet --as-of-dates data/as_of_dates.csv --config config.json --train-end 2024-12-31 --output-dir artifacts/run-001 --min-cross-section 50 --hac-lags 19 --walk-forward-splits 5 --walk-forward-test-date-count 20 --walk-forward-min-train-date-count 252 --with-pca
~~~

hac-lags must match the overlap implied by the target horizon and sampling
frequency; 19 is only an example for a heavily overlapping 20-session target.

This run writes the metric panel, coverage report, daily and summary Rank IC,
quantile spreads, redundancy pairs, selected/dropped metrics, optional
walk-forward reports, and optional PCA scores/loadings.

The single-cutoff screen uses only labels matured by the end of `--train-end`.
The returned panel still contains all supplied outcomes, and optional
walk-forward evaluation uses its own purged folds over the supplied panel.
Do not use `qmr run` as a no-outcome intake check.

For Stage 3, check structural feasibility before training, then record the
hypothesis and run development without scoring final-test outcomes:

~~~text
qmr preflight --panel artifacts/run-001/metric_panel.parquet --config benchmark-config.json
qmr benchmark --panel artifacts/run-001/metric_panel.parquet --config benchmark-config.json --output-dir artifacts/benchmark-001 --registry artifacts/research-registry.sqlite3 --study-id metrics-v1 --hypothesis "Combined metrics improve unseen-date ranking over equal-weight ranks."
~~~

Version 0.4 adds preflight, registered evidence, referenced final evaluation,
and `qmr experiments` history. Keep one registry for related research: a new
registry filename does not make previously seen outcomes independent. Preflight
feasibility does not guarantee successful fitting or a useful signal. Validation
and hashing still read the full input; this is not a sealed data store.

The benchmark atomically publishes a new, non-overwriting output directory with
a reproducibility manifest and data gate, fold assignments,
tuning and screening records, out-of-sample predictions, daily and fold
metrics, summary comparisons, and the acceptance decision. See
[the Stage 3 guide](docs/stage3-benchmark.md) for final-run commands, the schema 4
artifact contract, and registry limitations. A registry status of `completed`
means calculation completed; verify the artifact bundle was also published.

For a deterministic offline smoke check, run the
[synthetic example](examples/README.md):

~~~text
python examples/synthetic_workflow.py --output-dir artifacts/synthetic-demo-001
~~~

It trains a tiny Ridge model on invented data and verifies final-test reuse is
refused. It is a software check, not evidence of market alpha.

Scores use the full contemporaneous stock cross-section, before filtering
future outcomes. Daily prediction coverage is separate from labeled-pair
coverage. Quantile spreads share bucket weight equally across boundary ties;
these descriptive spreads are not executable portfolio returns.

## Stage 3 status and the next gate

The Stage 3 engine is implemented. It compares every usable metric, a best
train-only metric, and an equal-weight oriented-rank baseline with Ridge,
histogram gradient boosting, and optional Ridge+PCA. Scores are evaluated as
rankings; they are not calibrated expected-return forecasts.

The repository still makes no empirical alpha claim. Promotion to Stage 4 is
blocked until a research-grade point-in-time provider and its identifier,
corporate-action, universe-membership, and delisting policies are independently
verified, followed by a pre-declared real-data lockbox run. A passing model
gate alone is not enough.

See docs/roadmap.md for the proposed model, portfolio, and production gates.
