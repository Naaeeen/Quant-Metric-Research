# Research roadmap

## Current boundary

Stages 1 and 2 build the point-in-time panel and screen individual metrics. The
Stage 3 benchmark engine is also implemented: it trains supervised ranking
models, evaluates purged development folds, and selects a model family. Opening
the reserved final test is now explicit, not the default. The repository still
does not claim empirical alpha or
simulate a portfolio.

That boundary is intentional. A model cannot repair survivorship bias,
look-ahead leakage, unstable identifiers, incorrect corporate actions, or
missing delisting returns.

## Stage 3: supervised cross-sectional benchmark

Goal: test whether combining metrics produces a more stable unseen-date stock
ranking than transparent non-ML baselines.

Implemented in the engine:

- individual-metric, best train-only metric, and equal-weight oriented-rank
  baselines;
- Ridge and small histogram-gradient-boosting families, plus optional
  Ridge+PCA as a controlled dimensionality-reduction comparator;
- fold-local screening, imputation, scaling, rank transforms, PCA, and fitting;
- inner purged validation inside outer purged development folds;
- model-family selection from development common-sample Rank IC, with optional
  explicit final-test evaluation;
- native-coverage and common-sample Rank IC/spread reports, reproducibility
  fingerprints, outer/lockbox assignments, tuning records, and explicit
  acceptance/data gates. Inner tuning windows are reproducible from the input
  panel and manifest configuration; their screening and trial results are
  persisted, but their row-level assignments are not a separate artifact.

The September 2026 audit retained this direction and corrected four hazards:
future-label filtering before ranking, unconstrained decision timestamps,
arbitrary tie-bucket selection, and malformed membership-end dates. Coverage
now distinguishes actual prediction coverage from labeled-pair coverage.
CI checks a clean install, tests with branch tracking, lint, CLI and packaging.

Next engineering milestone (before a new model family): a no-training preflight
and persistent study/holdout reservation. Record hypothesis, data/code/config
fingerprints, trial history (including failures), fixed boundaries, frozen
selection and status. Reserve before final evaluation; treat interruptions as
potentially consumed. Changing a config hash must not reset the same holdout.
This local guard would prevent accidents, not intentional bypass or earlier
human inspection. Also add a runnable synthetic example; it must make no alpha
claim. These controls are not implemented by the current opt-in flag.

Remaining before Stage 3 can make an empirical conclusion:

1. Select and independently audit a versioned historical provider for prices,
   point-in-time membership, stable identifiers, corporate actions, and
   delisting returns.
2. Freeze the universe, decision time, rebalance schedule, target horizon,
   transaction-price assumption, feature list, evaluation period, acceptance
   thresholds, experiment ID, and lockbox-reuse policy before inspecting
   benchmark results.
3. Build and quality-check the real panel, including missingness by date,
   security lifecycle coverage, identifier changes, and target maturity.
4. Run the declared experiment once, inspect the manifest and fold assignment
   audit, and compare the frozen model with the equal-rank and individual-metric
   baselines on both common and native samples.
5. Require the model acceptance gate and independent data-provenance review to
   pass. If either fails, document the result and revise only through a new
   declared experiment with a new lockbox.

The model gate requires strict baseline improvement, minimum native score and
spread coverage, enough valid Rank-IC and spread lockbox dates, and the
predeclared HAC threshold on paired daily Rank-IC improvement. It is not allowed
to pass from a tied one-date result.

Training is useful here because the experiment tests a combined ranking. The
model remains a hypothesis to benchmark, not evidence that the ranking is
tradable. `stage4_eligible` therefore remains false while provenance checks are
unverified, even if the statistical model gate passes.

## Stage 4: portfolio and execution backtest

Goal: determine whether signal quality survives implementation.

Entry gate: a passed real-data Stage 3 lockbox, verified data policies, and a
documented independent review. Development results alone cannot enter Stage 4.

- Convert scores into a clearly specified long-only or long-short portfolio.
- Add sector, size, beta, concentration, liquidity, and position constraints.
- Model commissions, spread, slippage, market impact, rebalance delay, and
  turnover.
- Include delisted securities and unavailable trades rather than silently
  filtering them.
- Compare gross versus net return, information ratio, drawdown, capacity, and
  exposure drift.
- Stress assumptions and run negative controls, such as shuffled targets and
  intentionally delayed signals.

## Stage 5: repeatable research product

Goal: make every score and report reproducible and reviewable.

- Version datasets, configurations, code, models, and output artifacts.
- Record feature and label availability timestamps and data-quality incidents.
- Monitor live coverage, drift, IC decay, turnover, costs, and exposure limits.
- Define retraining and shutdown rules before live use.
- Require review before a research score can affect a portfolio.

The later production gate should also define ownership, incident response,
data-vendor change control, champion/challenger promotion, and a reproducible
rollback path.

## Final objective

The practical end product is a reproducible signal engine that converts
research-grade point-in-time data into a stock ranking, shows why the ranking
exists, and demonstrates where it does and does not work. After the portfolio
gate, the same evidence should show whether signal quality survives constraints
and realistic costs. Its value is faster, reviewable research and better
decision support—not an automatic guarantee of profit.
