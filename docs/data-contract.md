# Data contract

## Decision time

The MVP assumes signals are formed after the `as_of_date` market close. An
`entry_lag_sessions` value of one therefore starts the label on the next
benchmark session. A horizon of 20 means 20 close-to-close benchmark-session
intervals from that entry point. Both dates are stored in every output row.

## Price input

Required columns:

- `date`: normalized trading date.
- `symbol`: stable security identifier used by this dataset version.
- `adjusted_close`: finite, strictly positive adjusted close.

Rows must be unique by `(date, symbol)`. The benchmark must be present in the
same table. The core library does not download or silently repair market data;
provider-specific ingestion belongs in an adapter with its own provenance.

## Universe membership input

Required columns:

- `universe_id`
- `symbol`
- `effective_from`
- `effective_to` (nullable)
- `source`

Intervals are half-open: `effective_from <= as_of_date < effective_to`. Null
`effective_to` means open-ended. Intervals for the same universe and symbol may
not overlap. A current constituent list without dated intervals is allowed only
as an explicitly labelled demo and cannot support an unbiased historical claim.

## Panel output

One row represents one active security at one decision date. Metadata includes:

- dataset version and universe identifier;
- feature availability, window start/end, observation count, and eligibility;
- benchmark symbol and metric configuration;
- label start/end, label availability, forward return, benchmark return,
  excess return, and cross-sectional target rank.

The stored configuration includes lookback, minimum observations, target
horizon, entry lag, annualization frequency, and annual risk-free rate so a
panel row can be traced to the assumptions that produced it.

The initial feature family mirrors reusable per-security metrics from the
existing application: trailing return, Sharpe, Sortino, volatility, maximum
drawdown, beta, CAPM alpha, information ratio, benchmark correlation, and
historical VaR. Efficient frontier output is excluded because it is a
portfolio-level optimization result, not a stable per-security feature.

## Missingness

Rows are retained when prices, history, or labels are unavailable. Numeric
fields remain null and a reason column records why. This prevents silent
survivor filtering and makes coverage part of Stage 2 analysis.

## Stage 3 benchmark input

Stage 3 consumes a panel, not raw price files. Required columns are:

- `as_of_date` and `symbol`, unique after normalization;
- `label_end_date`, which must be strictly later than `as_of_date` when present;
- every name in `feature_columns`;
- `target_column` and `realized_return_column` (both default to
  `forward_excess_return`).

Dates must be valid, normalized, timezone-naive calendar dates. Features and
targets may be null, because missingness is handled inside each fold, but any
non-null value must be finite and numeric. If `decision_time` or per-feature
availability columns are supplied, the validator rejects features that became
available after the decision time. The loader creates a deterministic `row_id`
from the normalized security-date key; callers should not use a DataFrame index
as identity.

The default target is the raw forward excess return. Models may learn from that
continuous target, but their output is treated as a ranking score rather than a
calibrated expected-return estimate.

## Split and fitting boundaries

All training folds must end before the first test decision date, and every
training row must also have `label_end_date < first_test_as_of_date`. This
purges overlapping forward labels without assuming a fixed sampling frequency.

Stage 2's optional walk-forward report records the train end, maximum retained
label end, test start/end, train selection status, test Rank IC, and test
quantile spread for every feature and fold. It also applies the training
direction to test Rank IC, so a stable negative predictor is not mistaken for
an unstable or useless one.

Stage 3 reserves the final configured count of complete, labelled dates before
any model selection. Development then uses expanding outer walk-forward folds;
each model candidate is selected with inner purged walk-forward validation.
The final model family is frozen from development common-sample Rank IC and the
lockbox is evaluated once. Screening, imputation, scaling, rank transforms,
PCA, and fitting are repeated from training data inside the relevant fold.

Supervised model-loss weights give each decision date equal total weight and
are normalized to mean one across rows. This prevents larger cross-sections
from quietly controlling the fitted objective. Imputation, scaling, and PCA
are still row-weighted preprocessing operations; they are fold-local, but the
engine does not claim that every preprocessing statistic is date-weighted.

## Stage 3 evaluation and artifacts

The engine fits three types of non-ML comparison: every usable individual
metric, the best metric selected on training data, and an equal-weight
composite of training-oriented cross-sectional ranks. Model families are
Ridge, histogram gradient boosting, and optional Ridge+PCA.

Evaluation reports per-date Rank IC, raw top-minus-bottom realized-return
spread, score coverage, and tied-score fraction. `native` results use each
model's available rows. `common` uses one security-date intersection across all
configured model families plus the primary baseline during development, then
across the frozen model and primary baseline in the lockbox. Model selection
and the main acceptance comparison use the common scope so missing predictions
cannot create an unfair sample advantage.

`qmr benchmark` writes:

- `benchmark_manifest.json` and `data_gate.json`;
- `fold_assignments.parquet`;
- `hyperparameter_trials.csv` and `screening_by_fold.csv`;
- `oos_predictions.parquet` (or `.csv`);
- `daily_metrics.csv`, `fold_summary.csv`, and `benchmark_summary.csv`;
- `acceptance.json`.

The manifest fingerprints the validated panel contract, model inputs,
configuration, and actual package source files, and records the artifact schema
and major library versions. The writer refuses an existing destination and
publishes a completed bundle by renaming a temporary sibling directory, so a
failed run cannot leave a partial result that looks final. The data gate reports
locked target/realized-return coverage and cross-section counts, and deliberately
keeps provider, stable-identifier,
corporate-action, and delisting-policy verification false until those policies
are independently audited. Therefore an engine run cannot by itself support an
empirical alpha claim or Stage 4 promotion.
