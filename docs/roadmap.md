# Roadmap after Stages 1 and 2

## Current boundary

The current repository builds the research panel and evaluates individual
metrics. It does not yet train a prediction model or simulate a portfolio.
That boundary is intentional: a model cannot repair survivorship bias,
look-ahead leakage, an ambiguous decision time, or a weak target.

## Stage 3: supervised cross-sectional benchmark

Goal: produce a score that ranks stocks by expected forward excess return and
test whether combining metrics improves unseen-date Rank IC and spread.

1. Connect a versioned historical provider for prices, corporate actions,
   security identifiers, universe membership, and delistings.
2. Freeze the universe, decision time, rebalance frequency, target horizon,
   transaction-price assumption, and evaluation period before tuning.
3. Establish non-ML baselines: each individual metric, the best train-only
   metric, and an equal-weight rank composite.
4. Train simple models first. A regularized linear model is the interpretability
   baseline; a small tree-based model is the nonlinear comparison. Fit
   imputation, scaling, selection, and any PCA inside each training fold only.
5. Use purged walk-forward train/validation/test windows. Report Rank IC,
   Rank-IC stability, quantile spread, turnover, coverage, and performance by
   market regime. Hyperparameters are chosen on validation folds, never on the
   final test period.
6. Keep the model only if it improves the pre-declared baselines consistently,
   not just the pooled average or one favorable period.

Training is needed at this stage because the artifact is now a combined stock
ranking, not merely a report about each metric. The model is a hypothesis to
benchmark, not proof that the signal is tradable.

## Stage 4: portfolio and execution backtest

Goal: determine whether signal quality survives implementation.

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

## Final objective

The practical end product is a reproducible signal engine that converts
point-in-time data into a stock ranking, shows why the ranking exists, and
demonstrates where it does and does not work. Its value is faster,
evidence-based research and better decision support. It is not an automatic
guarantee of profit.
