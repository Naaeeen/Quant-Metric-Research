# Research roadmap

## Current boundary

Stages 1 and 2 build the point-in-time panel and screen individual metrics. The
Stage 3 benchmark engine is also implemented: it trains supervised ranking
models, evaluates purged development folds, freezes a model family, and touches
one final lockbox. The repository still does not claim empirical alpha or
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
- model-family freezing from development common-sample Rank IC, followed by one
  locked final-test evaluation;
- native-coverage and common-sample Rank IC/spread reports, reproducibility
  fingerprints, full assignments, tuning records, and explicit acceptance/data
  gates.

Remaining before Stage 3 can make an empirical conclusion:

1. Select and independently audit a versioned historical provider for prices,
   point-in-time membership, stable identifiers, corporate actions, and
   delisting returns.
2. Freeze the universe, decision time, rebalance schedule, target horizon,
   transaction-price assumption, feature list, evaluation period, and
   acceptance thresholds before inspecting benchmark results.
3. Build and quality-check the real panel, including missingness by date,
   security lifecycle coverage, identifier changes, and target maturity.
4. Run the declared experiment once, inspect the manifest and fold assignment
   audit, and compare the frozen model with the equal-rank and individual-metric
   baselines on both common and native samples.
5. Require the model acceptance gate and independent data-provenance review to
   pass. If either fails, document the result and revise only through a new
   declared experiment with a new lockbox.

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
