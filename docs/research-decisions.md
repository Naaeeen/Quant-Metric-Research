# Research decisions

Last reviewed: 2026-09-07

## Entry-gate evidence

- Microsoft Qlib separates data handling, dated train/validation/test segments,
  signal analysis, and portfolio analysis. Its public Alpha158 benchmarks use
  IC, Rank IC, ICIR, returns, information ratio, and drawdown.
  <https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md>
- Qlib's published XGBoost workflow uses non-overlapping dated train,
  validation, and test segments rather than a random row split.
  <https://github.com/microsoft/qlib/blob/main/examples/benchmarks/XGBoost/workflow_config_xgboost_Alpha158.yaml>
- Qlib defines learnable labels from future returns and evaluates signals with
  IC/Rank IC; that supports using raw future excess return as the training
  target while treating output as a ranking score.
  <https://github.com/microsoft/qlib/blob/main/docs/advanced/alpha.rst>
- Numerai models each row as a stock at an era, keeps features point-in-time,
  defines explicit 20/60-day future-relative-return targets, and warns that
  overlapping targets need special cross-validation treatment.
  <https://docs.numer.ai/numerai-tournament/data>
- QuantConnect recommends dynamic point-in-time universes to reduce look-ahead
  and survivorship bias and recommends walk-forward research.
  <https://www.quantconnect.com/docs/v2/writing-algorithms/key-concepts/research-guide>
- Scikit-learn requires preprocessing and feature selection to be fitted on
  training data only; the warning explicitly includes scaling, selection, and
  PCA.
  <https://scikit-learn.org/stable/common_pitfalls.html>
- Scikit-learn's nested-cross-validation example separates hyperparameter
  selection from generalization evaluation; without that separation, the
  reported score can inherit selection bias.
  <https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html>
- Scikit-learn's time-series splitter exposes a gap between training and test
  samples; this repo instead purges by each row's actual label end date because
  the panel may not use a fixed sampling interval.
  <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html>
- Harvey, Liu, and Zhu show why testing many candidate factors makes a
  conventional unadjusted significance threshold unreliable.
  <https://www.nber.org/papers/w20592>
- Gu, Kelly, and Xiu compare regularized linear and nonlinear ML methods for
  empirical asset pricing using large panels of stock characteristics. This
  supports a simple-linear-versus-small-tree benchmark, not skipping directly
  to a complex model.
  <https://dachxiu.chicagobooth.edu/download/ML_BKP.pdf>

## Decisions changed by research

1. A current ASX200 file is not accepted as historical membership. The input
   contract requires effective dates; a static list is demo-only.
2. Label offsets use benchmark sessions and an explicit entry lag. They are not
   calendar days and are not inferred from row positions inside each symbol.
3. Evaluation is cross-sectional by decision date. Pooled correlations can be
   dominated by time trends and are not used as Rank IC.
4. Feature scaling, supervised screening, redundancy representative selection,
   and PCA are fitted through an explicit training cutoff.
5. PCA is optional. With the initial small, interpretable metric set, the MVP
   first reports coverage, stability, Rank IC, quantile spread, and redundancy.
6. Multiple metrics create selection risk. Rank-IC summaries include a
   Newey-West-style overlap adjustment and Benjamini-Hochberg q-values, while
   the report still avoids claiming that statistical significance guarantees
   tradable alpha.
7. The single-cutoff report is candidate screening, not an out-of-sample
   result. Optional purged walk-forward validation now repeats train-only
   selection and reports unseen-date Rank IC, spread, and selection stability.
8. The reusable Alpha, Sharpe, Sortino, and information-ratio features preserve
   the existing application's annualized conventions. Efficient frontier stays
   excluded because it is a portfolio output rather than a stock-date feature.
9. Stage 3 predicts raw forward excess return but evaluates cross-sectional
   ranking quality. Scores are not presented as calibrated return forecasts.
10. The non-ML bar is explicit: each usable metric, the best metric chosen only
    from training data, and an equal-weight composite of rank-oriented metrics.
11. Supervised model-loss weights give each date equal total weight. Otherwise
    a date with a larger investable cross-section would receive more influence
    simply because it has more rows. Fold-local imputation, scaling, and PCA
    remain row-weighted; this distinction is reported rather than hidden.
12. Model complexity is bounded. Ridge is the regularized linear baseline and
    histogram gradient boosting is the small nonlinear comparison. Ridge+PCA
    is optional; PCA is only a compression comparator because variance
    preservation is not the same as predictive usefulness.
13. The final dates are locked before tuning. Outer purged folds estimate
    development performance; inner purged folds select parameters. The frozen
    model family is chosen from development common-sample Rank IC, then the
    lockbox is evaluated only after an explicit opt-in in that engine run.
14. Both native and common-sample metrics are retained. Native results expose
    practical coverage; common results make the primary model-versus-baseline
    comparison on identical security-date rows.
15. A benchmark result is an immutable artifact bundle. The destination must
    not already exist, publication is atomic, feature lists and hyperparameters
    use canonical JSON, and the manifest fingerprints the actual package source
    plus the full validated panel contract and execution mode—not only a
    manually bumped version.
16. Rank IC and economic spread use separate complete-case samples. Their
    counts and coverage are reported separately because a distinct realized
    return column may be missing when the learning target is still valid.
17. Acceptance coverage uses each model's native sample, not the deliberately
    identical common intersection. The gate requires both relative and absolute
    score coverage, native spread coverage, minimum valid Rank-IC and spread
    lockbox date counts, a strict improvement over the baseline, and a
    predeclared HAC p-value bound on the paired daily Rank-IC improvement.
18. `lockbox_evaluated_once_in_this_run` is deliberately narrow. Preventing a
    team from rerunning the same lockbox needs cross-run evidence. Version 0.4
    adds a local experiment registry and reports its enforcement separately;
    this is not a claim to enforce an organization-wide access-control policy.

## September 2026 audit: keep the engine, repair the evidence path

The audit reproduced defects before implementation and added regression tests.
No result justified a wholesale rebuild or a more complex model.

- Future target missingness previously removed stocks before feature ranking,
  changing other stocks' scores. Score full contemporaneous date cross-sections
  first; mask unauthorized outcomes separately. Apply this distinction in
  training transforms, inner validation and final scoring.
- Keep daily decision timestamps within their as-of day and reject malformed
  membership ends. A later timestamp must not authorize future features under
  an earlier decision-date split.
- Replace arbitrary stable-sort tie slicing with equal fractional weight across
  boundary ties. Keep the old bucket mass and undefined flat-score behavior.
  This is our explicitly tested descriptive-statistic convention, not a claim
  that every quant platform uses it.
- Default Stage 3 to development only, require explicit final-test opt-in, and
  distinguish unperformed acceptance from empirical failure. Schema 3 records
  the execution mode and separates prediction coverage from outcome coverage.

The methodological basis remains training-only model choices and preprocessing,
as described by [scikit-learn](https://scikit-learn.org/stable/common_pitfalls.html).
[Qlib Recorder](https://qlib.readthedocs.io/en/stable/component/recorder.html)
records experiment/run identity, parameters, metrics and artifacts; that informs
the version 0.4 registry described below. [Bailey et al., The Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)
explains why holdouts alone do not account for repeated strategy searches.
Accordingly, an opt-in flag and one-run p-value are insufficient evidence of
unbiased discovery after an unrecorded research search.

## Version 0.4: preflight and durable experiment evidence

Keep the existing numerical benchmark and add evidence controls before adding
models. The no-training preflight shares the benchmark's temporal split builders,
checks all outer/inner/final-training schedules after purging, and reports
development coverage and sample-size limitations. It deliberately does not
screen features, calculate IC, or promise that a model can pass acceptance.

Qlib Recorder supplies the experiment/run separation. [MLflow's backend-store
documentation](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/)
separates run metadata from artifacts and supports a local SQLite backend.
We use standard-library SQLite instead of adding an experiment server dependency;
this is a scope/cost decision, not a claim of feature parity with either system.

[SQLite's transaction documentation](https://www.sqlite.org/lang_transaction.html)
explains how `BEGIN IMMEDIATE` obtains the write transaction and serializes
competing writers. The registry checks overlap and inserts the reservation in
one transaction, then commits before final model fitting or evaluation. Keeping
that transaction open through training would let a crash roll back the protection.
Failed/interrupted runs retain their exposure; no reset/unreserve API is provided.

Research and independent review changed the initial implementation in three ways:

1. Protect against previously exposed development periods, not only earlier final
   runs. Otherwise a shortened panel could rename development dates as a fresh
   final test. Overlap checks span all studies/configurations in the same registry.
2. Extend final exposure through maximum locked `label_end_date` so a shifted
   decision block cannot silently reuse the forward-return tail. Development
   exposure conservatively ends the calendar day before the lockbox.
3. Validate the referenced plan before repeating any development fitting, then
   repeat validation atomically when reserving the final test. This prevents a
   stale configuration from evaluating unregistered development dates before
   being rejected. Match panel/source/config/boundaries and recorded runtime
   versions, then require the same development-selected model family.

These envelope rules are our conservative safeguards, not a universal industry
standard. They may reject unrelated markets sharing dates in the same registry.
Registry `completed` means computation finished; subsequent artifact publication
can still fail without releasing the consumed interval. Unknown schemas and
invalid/corrupt files fail closed. Metadata uses parameterized SQL, validated
finite JSON and exception class names instead of potentially sensitive error text.

The runnable offline synthetic example verifies the CLI lifecycle and refusal
behavior. It proves no empirical signal. Neither the registry nor a one-run
p-value corrects for an entire history of strategy searches: multiple-testing,
independent review and a genuinely unobserved period remain research obligations.
Full-panel validation and hashing still read the underlying data, and local files
cannot prevent manual edits, alternative registries or prior human inspection.

Provider acquisition, a new market choice, net-cost portfolio testing and
production promotion remain outside this software-correctness milestone.

## Exit-gate conclusion

The first two stages and the Stage 3 benchmark engine are logically aligned
with the cited research workflows: point-in-time contracts, explicit future
labels, cross-sectional evaluation, nested purged temporal validation,
train-only transforms, a locked final test, and separate signal versus
portfolio analysis.

This is an implementation conclusion, not an empirical alpha conclusion. The
engine is implemented and tested, but no research-grade real-data lockbox run
has passed. Before any Stage 4 promotion, independently verify the historical
provider, stable identifiers, universe membership, corporate actions, and
delisting-return policy; pre-declare the experiment; then review one lockbox
run against the non-ML baselines.

Even a passing Stage 3 model gate only makes the result eligible for data and
research review. Portfolio construction must still test turnover, costs,
liquidity, exposure drift, capacity, and negative controls.

## Recheck triggers

Repeat the research gate before changing any of these items:

- data vendor, security identifiers, or corporate-action treatment;
- universe definition or region;
- decision time, entry lag, target horizon, or rebalance frequency;
- feature-selection threshold or statistical test;
- dimensionality-reduction method;
- model family, tuning grid, acceptance threshold, or lockbox boundary;
- experiment identifier or lockbox-reuse policy;
- transition to portfolio backtesting or live use.
