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
   with nested purged development folds and one locked final test.

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
- Stage 3 selects a model family using development results, then evaluates that
  frozen family on the final lockbox once within a run. A real experiment also
  needs an external experiment ID/reuse registry so reruns cannot be presented
  as a fresh lockbox.
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

~~~text
qmr run --prices data/prices.parquet --memberships data/memberships.parquet --as-of-dates data/as_of_dates.csv --config config.json --train-end 2024-12-31 --output-dir artifacts/run-001 --min-cross-section 50 --hac-lags 19 --walk-forward-splits 5 --walk-forward-test-date-count 20 --walk-forward-min-train-date-count 252 --with-pca
~~~

hac-lags must match the overlap implied by the target horizon and sampling
frequency; 19 is only an example for a heavily overlapping 20-session target.

This run writes the metric panel, coverage report, daily and summary Rank IC,
quantile spreads, redundancy pairs, selected/dropped metrics, optional
walk-forward reports, and optional PCA scores/loadings.

To run the implemented Stage 3 benchmark:

~~~text
qmr benchmark --panel artifacts/run-001/metric_panel.parquet --config benchmark-config.json --output-dir artifacts/benchmark-001 --prediction-format parquet
~~~

The benchmark atomically publishes a new, non-overwriting output directory with
a reproducibility manifest and data gate, outer/lockbox fold assignments,
tuning and screening records, out-of-sample predictions, daily and fold
metrics, summary comparisons, and the acceptance decision. See
`docs/stage3-benchmark.md` for the config and artifact contract.

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
