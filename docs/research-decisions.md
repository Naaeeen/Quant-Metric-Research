# Research decisions

Last reviewed: 2026-08-27

## Entry-gate evidence

- Microsoft Qlib separates data handling, dated train/validation/test segments,
  signal analysis, and portfolio analysis. Its public Alpha158 benchmarks use
  IC, Rank IC, ICIR, returns, information ratio, and drawdown.
  <https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md>
- Qlib's published XGBoost workflow uses non-overlapping dated train,
  validation, and test segments rather than a random row split.
  <https://github.com/microsoft/qlib/blob/main/examples/benchmarks/XGBoost/workflow_config_xgboost_Alpha158.yaml>
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
- Scikit-learn's time-series splitter exposes a gap between training and test
  samples; this repo instead purges by each row's actual label end date because
  the panel may not use a fixed sampling interval.
  <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html>
- Harvey, Liu, and Zhu show why testing many candidate factors makes a
  conventional unadjusted significance threshold unreliable.
  <https://www.nber.org/papers/w20592>

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

## Exit-gate conclusion

The implemented first two stages are logically aligned with the cited quant
research workflows: point-in-time inputs, explicit future labels,
cross-sectional evaluation, purged temporal validation, train-only transforms,
and separate signal versus portfolio analysis.

The current outputs identify candidates; they do not establish economic value.
Before any claim of usefulness, use a real historical universe, choose the
rebalance/horizon before inspecting results, examine walk-forward stability,
and then test turnover, costs, liquidity, and risk exposures.

PCA remains an optional global-standardization baseline. Before using it in a
model, compare it with interpretable selected features and consider
cross-sectional normalization and industry/size neutralization when those
point-in-time fields become available.

## Recheck triggers

Repeat the research gate before changing any of these items:

- data vendor, security identifiers, or corporate-action treatment;
- universe definition or region;
- decision time, entry lag, target horizon, or rebalance frequency;
- feature-selection threshold or statistical test;
- dimensionality-reduction method;
- transition to model training, portfolio backtesting, or live use.
