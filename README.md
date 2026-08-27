# Quant Metric Research

A standalone, leakage-aware research pipeline for testing whether historical
equity metrics contain stable cross-sectional signal.

The first release deliberately covers only two stages:

1. Build a point-in-time `(as_of_date, symbol)` panel from adjusted-close data,
   dated universe membership, trailing metrics, and explicit forward targets.
2. Audit metric quality, measure per-date Rank IC and quantile spreads, identify
   redundant metrics, optionally validate them on purged walk-forward folds,
   and optionally compare a train-only PCA baseline.

It does not claim to be a complete trading system. Portfolio construction,
cost-aware backtesting, risk controls, execution, and live monitoring come
after these research stages.

## Design boundaries

- Historical universe membership is required for research-valid results.
- Features use only observations at or before `as_of_date`.
- Targets start after a configurable entry lag and use benchmark trading
  sessions, not spreadsheet row counts or calendar-day offsets.
- Missing or immature labels remain visible instead of being silently filled.
- Screening and PCA are fitted only through an explicit training end date.
- Walk-forward training rows are purged when their labels overlap a test fold.
- PCA is an optional compression benchmark, not the definition of a useful
  metric.

See `docs/data-contract.md` and `docs/research-decisions.md` before adding a
data provider or model.

## What “useful” means here

A metric is a research candidate when it has sufficient point-in-time
coverage, non-trivial cross-sectional variation, and a reasonably stable
relationship with the forward target. A stronger candidate should also retain
Rank IC or top-minus-bottom spread on unseen walk-forward dates and should not
be only a duplicate of a better metric.

That is still not proof of tradable alpha. Costs, turnover, liquidity, risk
exposures, and model-selection bias are later gates.

## Install

~~~text
python -m venv .venv
python -m pip install -e ".[dev]"
~~~

## Inputs

The CLI accepts CSV, compressed CSV, or Parquet tables:

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

## Run

~~~text
qmr run --prices data/prices.parquet --memberships data/memberships.parquet --as-of-dates data/as_of_dates.csv --config config.json --train-end 2024-12-31 --output-dir artifacts/run-001 --min-cross-section 50 --hac-lags 19 --walk-forward-splits 5 --walk-forward-test-date-count 20 --walk-forward-min-train-date-count 252 --with-pca
~~~

hac-lags must match the overlap implied by the target horizon and sampling
frequency; 19 is only an example for a heavily overlapping 20-session target.

The run writes the metric panel, coverage report, daily and summary Rank IC,
quantile spreads, redundancy pairs, selected/dropped metrics, optional
walk-forward reports, and optional PCA scores/loadings.

## ML and the next stages

Stages 1 and 2 do not require model training. Training becomes useful in Stage
3 only after the panel and out-of-sample metric baselines are trustworthy. The
first model should be a simple cross-sectional baseline and must be compared
against individual metrics and an equal-weight rank before a more complex
model is justified.

See docs/roadmap.md for the proposed model, portfolio, and production gates.
