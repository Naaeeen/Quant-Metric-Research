# Research roadmap

## Goal

Build a stock-ranking research system that can answer three questions:
which inputs help, which model combines them best, and whether the resulting
ranking remains useful after realistic trading constraints.

The work covers data, targets, features, models, evaluation, compute, software
quality and delivery. Success means a reproducible comparison with enough
evidence to accept or reject a candidate. A larger model or a longer feature
list is useful only if it improves that comparison.

This is the working plan following the September 22, 2026 review. Change it
when code inspection, a measured experiment or a primary source contradicts a
decision. Record the reason in [research decisions](research-decisions.md).
Completed release details remain there and in Git history.

## Current position

The repository builds price-derived stock-date panels, screens metrics and
trains Ridge, histogram gradient boosting and optional Ridge+PCA on nested,
purged time splits. It records experiments, scores, coverage, uncertainty and
development feature-bundle comparisons. The CPU Colab walkthrough pins 0.13;
the current package is 0.15, with a separate executable target-ablation example.

The only completed market-data study used 30 alphabetically selected stocks
from a retrospective 2017 constituent snapshot, with 2012-2016 prices.
Development mean Rank IC was -0.04767 for Ridge and -0.03701 for equal-weight
ranks. The result does not favor Ridge. The added three-factor bundle and the
other model families still need a declared market-data comparison.

Keep the panel, temporal validation, baseline and experiment-history design.
Shift effort from additional report wrappers to data qualification and
completed experiments. The full audit and source comparison are recorded in
[the September 22 review](research-decisions.md#september-22-2026-direction-review).

## How each stage runs

1. Review the relevant code, requirements, evidence and current upstream sources.
   Write the hypothesis, alternatives, costs and completion checks before editing.
2. Get an independent assessment. For major decisions, compare a fresh-view
   review with one informed by project history; reconcile disagreements using
   code, tests and sources.
3. Implement a small, testable change. For behavior changes, demonstrate a
   failing regression first, then pass it and review the diff.
4. Run the declared experiment or verification. Keep failures and unfavorable
   results. Compare accuracy, stability, coverage and resource use together.
5. Commit and push a coherent, reviewed milestone in English. Check remote
   revision and CI. Update this plan before choosing the next milestone.

Review gates apply to individual milestones, not just releases. Stop expanding
a method when its benefit disappears, its data assumptions fail, or a simpler
alternative gives equivalent results.

## 1. Define the decision and research target

**Question:** What should the ranking help someone decide?

- Keep named-stock ranking as the main product. Specify market, eligible stocks,
  decision timestamp, execution delay, holding period and baseline.
- Start with the existing 20-session outcome. Compare raw forward excess return
  with within-date return ranks while keeping realized returns for economics.
- Treat 5-, 20- and 60-session horizons as separate hypotheses. Label maturity,
  purging, rebalance frequency and uncertainty assumptions must change together.
- Distinguish benchmark subtraction from risk neutralization: subtracting the
  same market return does not change a day's stock order. Sector/beta residual
  targets require dated exposure inputs.
- Choose primary and secondary metrics before reading candidate results.
  Specify both an effect worth pursuing and a result that would reject it.

**Complete when:** one experiment specification fixes the decision, target,
universe, baseline, development/final periods and acceptance criteria.

## 2. Qualify and scale the data

**Question:** Does the dataset represent the opportunity available at the time?

- Audit historical membership, security identifiers, listings/delistings,
  corporate actions, publication times, revisions and permitted use.
- Compare expected exchange sessions with observed bars. A date missing from
  every series must not silently shorten feature windows or target horizons.
- Measure missingness by date, stock and lifecycle; distinguish missing current
  prices from insufficient warm-up history. Define a freshness policy for
  legacy metrics whose price endpoints are missing.
- Retain the pinned archive as an inexpensive development demonstration.
  More names from its retrospective constituent list do not remove its bias.
- Evaluate a broader, longer named-stock source against the
  [provider checklist](data-contract.md#human-provider-evidence-gate).
  More market periods matter as well as more stock-date rows.
- Use Numerai's published methods as a reference, not an approved data source.
  Its current terms restrict data use to tournament participation. Defer data
  adoption until the intended use and permissions are established; do not
  substitute obfuscated IDs for named-stock portfolio histories.
- Use Qlib as an implementation reference. Recheck data availability and rights
  before adopting a download: the official dataset was disabled at this review.
- Add filing-derived fundamentals only with historical availability and
  security mapping. Current restated facts are not historical observations.

**Complete when:** a versioned source report identifies covered periods,
verified assumptions, unresolved gaps, row/feature counts, missingness and
estimated memory. A small sample reproduces the full transformation.

## 3. Test features and representations

**Question:** Which information helps beyond the existing baseline?

- Compare the existing ten metrics against the ten plus the three implemented
  price factors. Keep formulas, lookbacks, feature availability and missingness
  rules explicit.
- Test quality-only screening against the current supervised redundancy screen;
  correlated features can still contain useful conditional information.
- Compare raw features, cross-sectional ranks and train-fitted scaling where
  appropriate. Measure whether preprocessing choices change coverage.
- Keep PCA as a compression comparator. Consider supervised reduction such as
  PLS only as a separate, fold-fitted candidate.
- Add factor families in small groups: price horizons first, then OHLCV,
  fundamentals, exposures or sentiment when their required inputs exist.
- Use held-out ablations and grouped importance to examine contribution;
  in-sample feature importance does not establish usefulness.
- Retain simple individual signals and the original-ten equal-rank baseline
  across feature-bundle comparisons.

**Complete when:** an ablation report isolates each representation change and
shows foldwise effect, coverage, redundancy and compute cost.

## 4. Compare model and training choices

**Question:** Which learner earns its extra complexity?

- First exercise the implemented Ridge and histogram boosting families.
  Compare raw-return and rank-target training without simultaneously changing
  features, dates or sample membership.
- Add LightGBM regression as the next optional challenger if the existing
  comparison warrants it. Match tuning effort and record dependency versions.
- Consider LambdaRank only with date groups, a declared relevance/gain mapping,
  and a top-k objective. It is not a drop-in replacement for a regressor.
- Consider CatBoost where it tests a useful alternative; verify group versus
  row-weight semantics for ranking objectives.
- On a larger dataset, compare a small MLP or TabM under the same temporal
  protocol. Run several declared seeds and measure variation as well as mean.
- Keep pretrained tabular models as a separate comparator. Check checkpoint
  licenses, access, memory and the distinction from training a model ourselves.
- Test rolling versus expanding training, retraining frequency, robust losses
  and simple ensembles one at a time.
- Bound the parameter search. Tree early stopping must use an inner temporal
  validation window, never a random split or the outer/final test.

**Complete when:** each candidate has identical evaluation dates, a documented
search budget, reproducible parameters and results against fixed baselines.
Promote a more expensive model only for a useful, repeatable improvement.

## 5. Strengthen statistical evaluation

**Question:** Would the conclusion survive a fair unseen-period test?

- Retain label-end purging, nested temporal tuning and fold-local preprocessing.
  Keep scoring universes independent of future outcome availability.
- Report daily and foldwise Rank IC, raw-return spreads, prediction coverage,
  outcome coverage and tied scores on native and common samples.
- Check market periods, cohort slices and concentration of gains in one fold.
  Compare methods on paired dates, not unrelated summary means.
- Choose dependence-aware uncertainty from the outcome horizon and sampling
  schedule. Report missing scheduled dates; assess longer blocks and sensitivity
  checks where a short overlapping sample is uninformative.
- Track all trials, including failed and unfavorable ones. Set a finite
  development search budget and a rule for moving to genuinely new evaluation
  data when continued adaptation has exhausted the original test.
- Add negative controls: within-date outcome permutation, deliberately delayed
  signals, known-null synthetic data and tests for future-data perturbations.
- Freeze the entire selected pipeline, including portfolio rules, before final
  evaluation. Keep final outcomes out of ongoing model selection.

**Complete when:** independent review reproduces the split, target maturity,
comparison population and selection decision from saved evidence.

## 6. Evaluate trading assumptions during development

**Question:** Is the ranking useful once implemented?

This work starts alongside model development, before opening the final test.
The earlier roadmap placed all portfolio work after a passed final holdout;
that ordering is replaced here.

- Begin with a declared long-only selection/rebalance rule and benchmark.
- Use a chronological holdings ledger with execution delay, cash, drift-aware
  turnover, entries, exits and explicit handling of unavailable trades.
- Evaluate commissions, spreads and slippage over a stated cost grid.
  Add liquidity, size, sector, beta and concentration constraints as inputs allow.
- Report gross/net returns, drawdown, turnover and exposures from actual
  period-by-period holdings. Do not compound overlapping 20-session signal
  spreads as if they were daily portfolio returns.
- Stress execution delay, rebalance frequency, missing prices and delistings.
  Compare a simple baseline portfolio using the same assumptions.
- Keep final model/portfolio evaluation separate from exploratory development
  simulations. Paper trading and live deployment require additional review.

**Complete when:** a tested development simulator explains every position
change and reconciles gross return, costs, cash and net return on hand-worked
fixtures and a declared dataset.

## 7. Make Colab runs practical

**Question:** Can a new environment reproduce the result within its resources?

- Provide separate smoke, development and larger-study configurations.
  Increase rows, history and features in measured steps.
- Record preprocessing, training and evaluation time separately, plus memory,
  hardware, thread count, data shape and model/search settings.
- Use projected Parquet reads and reuse deterministic preprocessing where
  equivalence tests show that reuse does not cross fold boundaries.
- Pin the tested source and constraints. Check a clean CPU environment first;
  add GPU support only for a model that benefits from it.
- Keep active SQLite and training files on local VM storage; checkpoint the
  existing private history to persistent storage between completed units.
- Test interrupted runs and reruns. Do not create a blank registry to recover
  from a failed experiment.
- Execute every notebook cell from a fresh runtime. Record local, Linux CI and
  hosted Colab verification separately.

**Complete when:** the walkthrough completes from its documented inputs,
reproduces the declared report and survives a tested interruption without
losing experiment history.

## 8. Keep the implementation maintainable

**Question:** Can the team understand, test and change it?

- Keep data, model, evaluation and storage responsibilities separate; reuse
  existing interfaces before adding another abstraction.
- Test formulas, missingness, dates, leakage and end-to-end CLI/example behavior.
  Use small fixtures with outcomes a reviewer can calculate.
- Run lint, tests with at least 80% aggregate coverage including branch tracking,
  dependency checks/audit and package builds before a code release.
- Review performance changes against identical outputs and measured runtime.
- Validate external inputs once at the right boundary. Preserve useful error
  context and caller-owned data; avoid speculative fallback layers.
- Write short English docs and comments. Explain non-obvious decisions and
  operating requirements; keep release history out of the quick start.
- Review the complete diff for secrets, private data and unintended changes.
  Keep changes scoped to this repository.

**Complete when:** a clean installation passes the checks, the example works,
and an independent reviewer can follow the implementation and its evidence.

## 9. Turn the result into a reusable research tool

**Question:** What remains useful after the first experiment?

- Produce comparable experiment summaries with configurations, data versions,
  fold results, runtime and reasons for accepting or rejecting candidates.
- Add new factors through a versioned catalog and a standard ablation process.
- Define retraining, monitoring, champion/challenger review and rollback before
  any model affects a portfolio.
- Monitor data coverage, feature drift, ranking stability, realized IC, costs
  and exposures; distinguish data failure from model deterioration.
- Make the handoff usable by the client: one reproducible run, readable results
  and a clear list of remaining data or operational requirements.

**Complete when:** a second researcher can add a candidate, reproduce the
baseline, interpret the comparison and resume the same research history.

## Next milestones

| Order | Deliverable | Status and completion check |
| --- | --- | --- |
| 1 | Direction audit and revised working plan | Complete: independent findings reconciled; sources and priorities recorded. |
| 2 | Correct target/return separation in inner tuning | Complete: failing regressions reproduced and fixed; rank-target training reports return-unit spreads. |
| 3 | Executable target/model ablation | Software complete: registered synthetic development runs, fixed features/folds, runtime and coverage reports. Market-data comparison remains open. |
| 4 | Data qualification and empirical ablations | Next: resolve calendar/freshness assumptions, complete declared target and 10/13-feature studies, and document a qualified larger-data choice. |
| 5 | Development economic diagnostics and negative controls | Open: tested holdings/cost accounting and a frozen final-evaluation procedure. |
| 6 | Stronger challenger and larger Colab study | Open: fair tuning budget, measured resources, independent review and reproducible run. |

Calendar and feature-freshness corrections belong before market-data conclusions
that depend on them. Their priority is not a reason to delay unrelated model
unit tests or the synthetic experiment workflow.

The longer-term stages remain open until their completion checks are met.
New sources or measured results can change this order; preserve the decision
and evidence rather than silently replacing the previous study.
