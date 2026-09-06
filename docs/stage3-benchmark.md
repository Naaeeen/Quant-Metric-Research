# Stage 3 benchmark

## Purpose and status

Stage 3 asks one narrow question: does a supervised combination of the metrics
rank future cross-sectional excess returns more consistently than transparent
non-ML baselines?

The benchmark engine and artifact workflow are implemented. This is not an
empirical alpha result. Until the historical provider, point-in-time universe,
stable identifiers, corporate actions, and delisting returns are independently
verified, full runs have `claim_scope: benchmark_engine_only`, development
runs have `claim_scope: development_only`, and both remain
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
final lockbox only when `evaluate_lockbox=True` is explicitly supplied with an
`ExperimentRegistry` and a matching completed `development_run_id`. The CLI
requires `--evaluate-lockbox --registry PATH --development-run-id RUN_ID`.
Version 0.4 implements the local outcome-exposure guard described below; this
does not replace an independently frozen research plan or data-provenance review.

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

Twenty locked dates with nineteen HAC lags is a deliberately small configuration
illustration, not persuasive statistical evidence for an overlapping 20-session
signal. Declare a sufficiently informative test period and sensitivity checks;
no universal date-count threshold guarantees reliable small-sample inference.

## Run and outputs

The panel may be CSV, compressed CSV, Parquet, or PQ. Parquet is the default
prediction format. The output directory must not already exist; the writer
builds a temporary sibling bundle and publishes it atomically only after every
file succeeds. All commands below use the same panel, config, and durable local
registry. Substitute actual paths and a hypothesis declared before inspection.

### 1. Check structural feasibility without training

~~~text
qmr preflight --panel artifacts/run-001/metric_panel.parquet --config benchmark-config.json
~~~

The JSON report includes `feasible`, scoped errors/warnings, development feature
coverage and label availability, purged outer/inner window summaries, and planned
final dates. It flags short HAC windows and unreachable acceptance date-count
thresholds. The CLI returns zero for a feasible report and two for structural
infeasibility; input-file or config loading can fail before a report is built.

This performs no screening, model fitting, Rank IC calculation, or final-outcome
scoring. `feasible: true` only means the requested split schedules exist; constant
features, fitting errors, or weak signal can still prevent a successful benchmark.
Warnings are diagnostics, not universal statistical adequacy rules. The report
contains no per-security rows or final-test performance, but structural checks
still read the entire panel and use label availability to reserve final dates.

### 2. Register and run development

~~~text
qmr benchmark --panel artifacts/run-001/metric_panel.parquet --config benchmark-config.json --output-dir artifacts/benchmark-001 --registry artifacts/research-registry.sqlite3 --study-id metrics-v1 --hypothesis "Combined metrics improve unseen-date ranking over equal-weight ranks."
~~~

Registered development requires all three of `--registry`, `--study-id`, and
`--hypothesis`. An existing study cannot silently change its hypothesis. The run
is recorded before calculation, its exposure interval before development fitting,
and its selected model family on successful completion. Failed calculations remain
in the run history. Inner-candidate outcomes are retained in the artifact bundle;
the registry records run-level status, configuration, identity, and frozen choice.

Read the development reference from
`benchmark_manifest.json` at `experiment.run_id`, or inspect history:

~~~text
qmr experiments --registry artifacts/research-registry.sqlite3
qmr experiments --registry artifacts/research-registry.sqlite3 --run-id DEVELOPMENT_RUN_ID
~~~

The first command prints a JSON list; the second prints one run. Records include
`kind` (`development` or `final`), `status` (`running`, `completed`, or `failed`),
study/hypothesis, timestamps, configuration, manifest, and exposure boundaries.
`completed` means the calculation completed, not that the CLI subsequently
published its artifact directory. Check both the registry and bundle; a write
failure after calculation cannot undo exposure or make a final test reusable.

Unregistered development remains available by omitting the three registration
flags, but it cannot authorize a final evaluation and is invisible to this
registry's exposure history. Use registered development for governed research.

### 3. Explicitly evaluate the referenced final test

After reviewing development evidence and freezing the experiment, replace the
placeholder below with the completed development run's ID:

~~~text
qmr benchmark --panel artifacts/run-001/metric_panel.parquet --config benchmark-config.json --output-dir artifacts/final-001 --registry artifacts/research-registry.sqlite3 --development-run-id DEVELOPMENT_RUN_ID --evaluate-lockbox
~~~

Do not supply a new study or hypothesis on this command: both come from the
reference. The engine validates completed, matching development evidence before
repeating the deterministic development procedure. It verifies the repeated
family selection, then atomically checks and reserves the outcome interval before
locked-model fitting or scoring. It does not load a saved fitted model. Concurrent
or repeated attempts cannot both reserve overlapping final intervals in the same
registry. The manifest records the final run ID and its development reference.

Evidence identity requires the same validated panel, actual package source,
configuration, package version, date boundaries, and Python, NumPy, pandas,
SciPy, and scikit-learn versions. This is not a full environment lock: operating
system, hardware, BLAS implementation, and other dependencies are not completely
captured. Changes to the bound identity require new registered development;
they do not clear existing outcome exposure.

Development mode does not fit, score, or export outcomes for the reserved test
period. Validation, split feasibility and full-panel hashing still inspect the
input, so this is not a physical secrecy boundary. Perturbing reserved outcome
values with the dates and availability pattern unchanged leaves development
scores and selection unchanged, but intentionally changes the input fingerprint.

### Registry scope and failure semantics

Keep one durable SQLite registry for related research. Final reservations are
checked against every prior development and final exposure in that file, across
all study names, datasets, securities, model choices, and configurations. This is
deliberately conservative calendar-envelope matching, not a claim that all these
datasets have identical outcomes:

- Development records the inclusive envelope from `development_start` through
  the calendar day before `locked_test_start`.
- Final evaluation reserves `locked_test_start` through `locked_label_end_max`,
  the maximum `label_end_date` across the reserved rows, not merely the last
  final decision date. This protects the forward-outcome tail too.
- Once an exposure is recorded, failure, interruption, or a `running` status
  does not release it. A new output folder or study ID cannot reopen it. An error
  before a final reservation is committed does not itself consume that final
  interval; existing development exposures still remain.

The guard is local to one registry file, not tamper-proof governance. A fresh
registry filename, manual database changes, unregistered experiments, or earlier
human inspection can bypass its knowledge. Renaming/copying files does not restore
statistical independence. It neither conceals raw labels nor establishes provider
provenance, empirical validity, or tradability. Record and independently review
research performed outside this workflow rather than presenting it as unseen.

### Artifact bundle

The output directory contains:

- `benchmark_manifest.json`: artifact schema version 4, `execution_mode`
  (`development` or `full`), full validated-panel and
  model-input fingerprints, actual package-source fingerprint, configuration,
  date boundaries including the final label-end envelope, dataset versions,
  library versions, and `experiment` registration/run-reference metadata;
- `data_gate.json`: structural result and external-data verification fields;
  locked target/return coverage and cross-section counts appear only in full mode;
- `fold_assignments.parquet`: every row's role and exclusion reason in outer
  development splits, plus locked-test splits only in full mode. These are
  labeled/purged eligibility assignments, not the complete scoring universe;
- `hyperparameter_trials.csv`: inner-validation results and stable JSON model
  parameters;
- `screening_by_fold.csv`: fold-local feature-screening record;
- `oos_predictions.parquet` or `.csv`: full scoring-universe development rows,
  including missing outcomes, plus locked-test rows only in full mode;
- `daily_metrics.csv`, `fold_summary.csv`, and `benchmark_summary.csv`;
- `acceptance.json`: frozen family, model checks, coverage comparison, and the
  Stage 4 blocker. Development mode reports `acceptance_status: not_evaluated`
  and cannot pass the model or Stage 4 gate.

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
  rates, scoring-universe/scored counts, absolute prediction coverage, and
  tied-score fraction;
- `fold_summary` and `benchmark_summary`: phase/model/scope aggregation,
  stability, coverage, and inference fields;
- `hyperparameter_trials`: outer fold, family/candidate, canonical-JSON
  parameters, validation Rank IC/spread/count, and selection flag;
- `screening_by_fold`: outer/inner fold, feature, selected/drop reason,
  training Rank IC, fit cutoff, row count, family, and fit kind.

JSON uses sorted UTF-8 keys, ISO timestamps, and strict finite values. CSV
parameter cells and prediction feature lists use compact, sorted/canonical JSON
rather than delimiter-dependent text.

Schema 4 retains the distinction introduced in schema 3 between
`score_coverage` (finite scores / scoring universe) and
`rank_ic_coverage` (score-target pairs / non-null targets). The former feeds
the native prediction-coverage acceptance gate. `spread_coverage` independently
measures score-return pairs. Future outcome missingness never selects the
cross-section used to compute scores.

Stage 2, inner tuning and Stage 3 use the same spread convention: fixed
`floor(n / quantiles)` bucket mass, shared fractionally by all boundary ties.
Reordering or renaming equal-score stocks cannot change the reported spread.
This convention describes a statistic, not a tradeable portfolio.

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
