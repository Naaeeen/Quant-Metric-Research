# Stage 3 benchmark

## Purpose and status

Stage 3 asks one narrow question: does a supervised combination of the metrics
rank future cross-sectional excess returns more consistently than transparent
non-ML baselines?

The benchmark engine and artifact workflow are implemented. This is not an
empirical alpha result. Until the historical provider, point-in-time universe,
stable identifiers, corporate actions, and delisting returns are independently
verified, every run has `claim_scope: benchmark_engine_only` and remains
ineligible for Stage 4.

## What the engine compares

For each outer development fold, the engine learns metric directions and fits:

- every usable individual metric;
- `best_metric`, chosen only from that fold's training data;
- `equal_weight_rank`, an equal-weight average of training-oriented
  cross-sectional metric ranks;
- Ridge as the regularized linear model;
- a small histogram gradient boosting model as the nonlinear comparison;
- optional Ridge+PCA as a compression comparator.

PCA is not feature usefulness. It preserves directions of feature variance,
which may or may not predict the target, so it remains an optional model family
and is judged against the same out-of-sample baselines.

The model target defaults to raw `forward_excess_return`. Predictions are
ranking scores, not calibrated expected-return forecasts. The main metrics are
per-date Spearman Rank IC and the raw realized-return spread between top and
bottom score buckets.

## Validation design

The final contiguous configured block before the immature tail is locked before
model selection. Every locked date must have at least `min_cross_section`
labelled targets, while row-level missingness remains visible. Development uses
expanding outer walk-forward folds. Each outer
training window runs inner purged walk-forward validation to choose a small,
explicit hyperparameter grid. Any row whose `label_end_date` reaches an
evaluation start is removed from training.

Screening, imputation, scaling, rank transforms, PCA, and fitting are local to
the relevant training fold. Model-loss weights give each date equal total
weight; imputation, scaling, and PCA remain row-weighted preprocessing. The
model family is frozen from development `common` Rank IC, then evaluated on the
final lockbox once within that run. Preventing reuse across runs requires an
external experiment ID/reuse registry and remains a Stage 4 entry control.

Reports contain two scopes:

- `native`: all usable predictions from that model, to show real coverage;
- `common`: one security-date intersection shared by all configured model
  families and the primary baseline in development, then by the frozen model
  and primary baseline in the lockbox.

This follows the separation between train/validation/test used by
[Qlib's public benchmark workflow](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/XGBoost/workflow_config_xgboost_Alpha158.yaml),
the fold-local preprocessing guidance in
[scikit-learn's leakage documentation](https://scikit-learn.org/stable/common_pitfalls.html),
and the model-selection/evaluation distinction in
[scikit-learn's nested-CV example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

## Configuration

The config is a JSON object accepted by `BenchmarkConfig.from_mapping`.
`feature_columns`, `hac_lags`, and every field in `split` are required; other
values below illustrate explicit research choices rather than universal
defaults.

~~~json
{
  "feature_columns": [
    "trailing_return",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "benchmark_correlation",
    "beta",
    "capm_alpha",
    "information_ratio",
    "historical_var_5pct"
  ],
  "split": {
    "final_test_date_count": 20,
    "outer_n_splits": 5,
    "outer_test_date_count": 20,
    "outer_min_train_date_count": 252,
    "inner_n_splits": 3,
    "inner_validation_date_count": 20,
    "inner_min_train_date_count": 126
  },
  "target_column": "forward_excess_return",
  "realized_return_column": "forward_excess_return",
  "min_cross_section": 50,
  "minimum_coverage": 0.8,
  "redundancy_threshold": 0.9,
  "quantiles": 5,
  "hac_lags": 19,
  "ridge_alphas": [0.1, 1.0, 10.0, 100.0],
  "hist_learning_rates": [0.05],
  "hist_max_leaf_nodes": [7],
  "hist_l2_regularization": [1.0],
  "include_pca_model": false,
  "rank_features": true,
  "random_seed": 42,
  "primary_baseline": "equal_weight_rank",
  "minimum_rank_ic_improvement": 0.0,
  "minimum_development_win_rate": 0.5,
  "minimum_coverage_ratio": 0.95,
  "minimum_locked_test_date_count": 20,
  "minimum_locked_score_coverage": 0.8,
  "minimum_locked_spread_date_count": 20,
  "minimum_locked_spread_coverage": 0.8,
  "maximum_locked_rank_ic_improvement_p_value": 0.05
}
~~~

`hac_lags` is required. `19` is only an example for overlapping 20-session
labels sampled each session. Choose it from the actual horizon and sampling
schedule before looking at results. The split also needs enough complete dates
to satisfy all outer and inner windows after label purging.

## Run and outputs

The panel may be CSV, compressed CSV, Parquet, or PQ. Parquet is the default
prediction format. The output directory must not already exist; the writer
builds a temporary sibling bundle and publishes it atomically only after every
file succeeds:

~~~text
qmr benchmark --panel artifacts/run-001/metric_panel.parquet --config benchmark-config.json --output-dir artifacts/benchmark-001 --prediction-format parquet
~~~

The output directory contains:

- `benchmark_manifest.json`: artifact schema version 2, full validated-panel and
  model-input fingerprints, actual package-source fingerprint, configuration,
  date boundaries, dataset versions, and library versions;
- `data_gate.json`: structural result, locked target/realized-return coverage
  and cross-section counts, plus separate external-data verification fields;
- `fold_assignments.parquet`: every row's role and exclusion reason in outer
  development and locked-test splits;
- `hyperparameter_trials.csv`: inner-validation results and stable JSON model
  parameters;
- `screening_by_fold.csv`: fold-local feature-screening record;
- `oos_predictions.parquet` or `.csv`: development and locked-test scores;
- `daily_metrics.csv`, `fold_summary.csv`, and `benchmark_summary.csv`;
- `acceptance.json`: frozen family, model checks, coverage comparison, and the
  Stage 4 blocker.

The row-level inner tuning assignments are deterministic from the source panel
and manifest configuration, but are not written as a separate table. Their
fold-local screening and candidate outcomes are retained in
`screening_by_fold.csv` and `hyperparameter_trials.csv`.

## Artifact schemas

Column order is stable:

- `fold_assignments`: `phase`, `fold`, `split_id`, `row_id`, `role`,
  `exclusion_reason`, `train_end_date`, `train_label_end_max`,
  `evaluation_start`, `evaluation_end`;
- `oos_predictions`: `phase`, `fold`, `as_of_date`, `symbol`, `row_id`,
  `model`, `score`, `target`, `realized_return`, `candidate_id`, fit dates,
  canonical-JSON `selected_features`, baseline metadata, observed selected-input
  `feature_count`, `selected_feature_count`, and `zero_observed_features`;
- `daily_metrics`: phase/fold/date/model/scope plus Rank IC, spread, evaluation
  count, separate Rank-IC/spread counts and eligible counts, their two coverage
  rates, and tied-score fraction;
- `fold_summary` and `benchmark_summary`: phase/model/scope aggregation,
  stability, coverage, and inference fields;
- `hyperparameter_trials`: outer fold, family/candidate, canonical-JSON
  parameters, validation Rank IC/spread/count, and selection flag;
- `screening_by_fold`: outer/inner fold, feature, selected/drop reason,
  training Rank IC, fit cutoff, row count, family, and fit kind.

JSON uses sorted UTF-8 keys, ISO timestamps, and strict finite values. CSV
parameter cells and prediction feature lists use compact, sorted/canonical JSON
rather than delimiter-dependent text.

## Reading the decision

First verify the manifest and data gate, then audit fold exclusions and coverage.
Interpret the development and locked rows separately. A model gate passes only
when all configured Rank-IC, spread, fold-win-rate, native relative/absolute
score coverage, native spread coverage, valid Rank-IC/spread lockbox-date
counts, and paired daily Rank-IC-improvement inference checks pass. Equality
with the baseline does not count as improvement.

Passing that gate means the model earned a data/research review; it does not
mean the signal is tradable. Stage 4 requires verified provenance plus a real,
pre-declared lockbox run. Only then should portfolio construction test turnover,
liquidity, costs, constraints, exposures, capacity, and negative controls.

For context, [Qlib evaluates signals with IC and Rank IC](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md),
while the broad comparison of regularized linear and nonlinear methods in
[Gu, Kelly, and Xiu](https://dachxiu.chicagobooth.edu/download/ML_BKP.pdf)
supports treating model complexity as an empirical comparison, not an assumed
improvement.
