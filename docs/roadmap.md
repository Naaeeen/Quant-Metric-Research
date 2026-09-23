# Research roadmap

## Goal

Build and critically evaluate a complete, reusable stock-ranking research
system in this standalone repository. Start with authorized historical data,
train models ourselves, compare their rankings on unseen periods, and measure
what those rankings would mean under explicit trading assumptions. Deliver a
workflow that the team can run in Colab, inspect, extend and reproduce.

The system should answer three questions: which inputs add useful information,
which training approach combines them best, and whether the improvement survives
changes in market conditions, missing data, trading costs and resource limits.
The goal covers all nine workstreams below: the research decision, data,
features, models, statistical validation, trading economics, compute,
engineering and handoff. Every workstream needs a verified deliverable.

Judge performance by out-of-sample ranking quality, stability, coverage and
economic value together. Declare numerical thresholds and a finite experiment
budget for each study before observing its results. Compare stronger models
under matched evaluation rules and measured compute budgets. Select complexity
when it earns a useful improvement; retain a simpler method when it performs
equally well.

Reconsider the market, data source, target, horizon, feature set, model and
training procedure when evidence supports a better choice. Record the changed
hypothesis and its dependencies before implementation or training, and preserve
earlier results and outcome exposure. Changes to the plan must keep the full
research objective in view.

This is the working plan following the September 22, 2026 review. Change it
when code inspection, a measured experiment or a primary source contradicts a
decision. Record the reason in [research decisions](research-decisions.md).
Completed release details remain there and in Git history.

### What completes the goal

The final handoff must include all of the following:

1. **A defined decision and qualified data.** A versioned specification fixes
   the stock universe, information available at decision time, execution delay,
   target, baseline and evaluation periods. Source evidence covers historical
   membership, identifiers, corporate actions, missing observations, revisions
   and permitted use. A qualified broader-data study must test more than the
   current retrospective demonstration cohort. If no candidate source passes,
   record the unmet requirement and next action; a source comparison alone does
   not complete this deliverable.
2. **Completed, attributable experiments.** Run the corrected-input raw/rank
   target comparison and the separate original-ten/expanded-factor comparison.
   Test feature screening and representations, and make an evidence-backed
   decision about the next model challenger. Save baselines, all declared
   trials, parameters, folds, coverage, paired effects and measured resources.
   Each new comparison changes one question at a time.
3. **An independently checked conclusion.** Review leakage, target maturity,
   sample selection, dependence, negative controls and concentration of gains.
   Use the declared criteria to accept, reject or call each result inconclusive.
   Preserve final outcomes for the frozen evaluation procedure; state any
   outstanding final-evaluation requirement explicitly. If no candidate meets
   the development and data criteria, record that conclusion and retain the
   unopened final period. An eligible candidate still needs the frozen final
   evaluation before a promotion decision.
4. **Tested economic accounting.** Reconcile positions, cash, execution delays,
   turnover and costs on hand-worked fixtures and development market data.
   Compare the candidate with a baseline portfolio under identical assumptions.
   Explain whether ranking improvements survive the stated cost and risk cases.
5. **A reproducible Colab and scoring workflow.** Pin a coherent source/runtime,
   execute a fresh hosted notebook, measure time and peak memory, and verify
   interruption recovery with the same durable research history. Reproduce
   saved-model scores from decision-time inputs without requiring future labels.
   Record any unsupported environment separately.
6. **A maintainable client handoff.** Deliver reviewed code, passing release
   checks, concise English instructions, comparable experiment reports and a
   worked example of adding a candidate. Include tested offline monitoring and
   retraining procedures, ownership of operational decisions, and the remaining
   requirements for any later paper-trading or production system.

A sound negative finding can complete a research study. It does not complete
unfinished data, economic, portability or handoff work. Finish the overall goal
only when the required deliverables have evidence and an independent review.
Model promotion is a separate decision that requires the documented data and
final-evaluation gates; profitable alpha is a hypothesis to test.

### Required work and conditional extensions

The nine workstreams and the handoff above are required. LightGBM, ranking
losses, neural models, pretrained models, new horizons, fundamentals, sentiment
and ensembles are candidates, not a requirement to implement every method.
For each candidate, record the question it tests, prerequisites, comparison
budget and decision to run or defer it. A deferred candidate needs a reason and
a condition for reconsideration. Required deliverables stay open when their
prerequisites are missing.

Use existing authorization for local research, implementation, review and
English commits/pushes to the public ML repository. Keep private data private
and FIT unchanged. New spending, provider commitments, restricted-data uploads,
final evaluation and trading follow their respective approval and research
gates. This plan does not authorize those actions by itself.

## Current position

The repository builds price-derived stock-date panels, screens metrics and
trains Ridge, histogram gradient boosting and optional Ridge+PCA on nested,
purged time splits. It records experiments, scores, coverage, uncertainty and
development feature-bundle comparisons. The CPU Colab walkthrough pins 0.13;
the current package is 0.16.1, with a separate executable target-ablation example,
declared-calendar checks, corrected trailing-return endpoints and faster pairwise
screening with exact-equivalence tests.

Completed market-data studies use 30 alphabetically selected stocks from a
retrospective 2017 constituent snapshot, with 2012-2016 prices. The corrected
target comparison found mean Rank IC of 0.01506 for raw-return histogram
boosting, -0.05001 for raw-return Ridge and -0.03701 for equal-weight ranks.
Only raw-return boosting met the declared continuation criteria against the
baseline. Neither rank-target transformation met its criteria. Boosting's
second-fold IC was negative; this is a development lead, not a stable signal.
The separate three-factor addition and qualified broader study remain open.
See the [full comparison](research-decisions.md#corrected-panel-target-study-results).

Keep the panel, temporal validation, baseline and experiment-history design.
Shift effort from additional report wrappers to data qualification and
completed experiments. The full audit and source comparison are recorded in
[the September 22 review](research-decisions.md#september-22-2026-direction-review).

Track implementation and research evidence separately:

| Evidence level | Current position | Next evidence needed |
| --- | --- | --- |
| Implemented and tested | Target/return separation, target ablation, calendar/endpoint corrections and exact-preserving array screening; corrected target experiment completed. | Separate factor comparison and end-to-end resource measurements at larger scale. |
| Data checked | Archive session dates match pinned XNYS; normalized date keys were checked. | Per-security completeness and qualified historical membership, provenance and rights. |
| Market-data evaluated | Corrected raw-return boosting met continuation criteria; Ridge and rank transformations did not. All five arms covered the 189 development dates. | Factor comparison, independent negative controls, economics and a qualified broader study. |
| Environment reproduced | The walkthrough pins 0.13; current package capabilities are 0.16.1. Local and Linux checks have their own records. | A coherent updated notebook and fresh hosted Colab execution. |

For each milestone, retain its hypothesis, prerequisite, deliverable,
verification method, acceptance rule, evidence path and current status in the
existing plan and research records. A passing software test, a completed market
experiment and a successful hosted run establish different things.

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

At the start of a stage and after any contradictory result, ask:

- Is this still the right decision, target and dataset for the client?
- Which assumption can invalidate the conclusion, and how will it be checked?
- What does the current primary research or upstream implementation support?
- Is there a simpler or better alternative that answers the same question?
- What evidence is missing, and will the next action produce it?
- What will make the experiment useful, inconclusive or worth stopping?

When changing direction, record the old choice, new evidence, revised choice,
affected results and next verification. Fix a demonstrated correctness problem
before dependent experiments. Continue independent work where possible. A
failed predictive comparison prompts diagnosis of data, labels and economics
as well as models; it does not automatically justify a larger search.

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
- Add batch scoring from decision-time inputs without future labels. Version
  the fitted preprocessing and model together; test score parity after saving
  and reloading them, including missing-feature and changed-schema cases.
- Exercise monitoring and retraining offline with delayed labels, stale inputs
  and interrupted runs. Define who reviews a warning or candidate replacement;
  alerts and new models must not silently change portfolio decisions.

**Complete when:** a second researcher can add a candidate, reproduce the
baseline, interpret the comparison and resume the same research history.

## Next milestones

| Order | Deliverable | Status and completion check |
| --- | --- | --- |
| 1 | Direction audit and revised working plan | Complete: independent findings reconciled; sources and priorities recorded. |
| 2 | Correct target/return separation in inner tuning | Complete: failing regressions reproduced and fixed; rank-target training reports return-unit spreads. |
| 3 | Executable target/model ablation | Complete for the declared small development study: both targets, Ridge/boosting, all six paired comparisons, independent review and measured resources. |
| 4 | Data qualification and empirical ablations | Calendar/endpoint checks, input rebuild and target study complete. Next: separate 10/13-feature study and a qualified larger-data choice. |
| 5 | Development economic diagnostics and negative controls | Open: tested holdings/cost accounting and a frozen final-evaluation procedure. |
| 6 | Stronger challenger and larger Colab study | Open: fair tuning budget, measured resources, independent review and reproducible run. |

Calendar and feature-freshness corrections belong before market-data conclusions
that depend on them. Their priority is not a reason to delay unrelated model
unit tests or the synthetic experiment workflow.

The target-only study is complete and the canonical history now contains five
development records, including the original interrupted run. Preserve this
lineage; the three-record pre-study backup and older two-record registry are
historical evidence, not restart points.

The array-screening implementation preserves exact synthetic outputs and
selection boundaries. Three fresh-process component comparisons measured a
5.49x median speed ratio on 120 dates, 30 stocks and ten features. This does not
measure a whole-study speedup or establish hosted Colab performance.
Next, declare the separate 10/13-feature comparison before fitting,
retaining fixed baselines and the original feature bundle. Keep raw-return
boosting as a candidate for further tests, not an approved portfolio model.
Qualify broader data and develop economic checks in parallel. Historical
membership and source rights remain unresolved for the current archive.

The longer-term stages remain open until their completion checks are met.
New sources or measured results can change this order; preserve the decision
and evidence rather than silently replacing the previous study.

## Dependencies and progress checks

- **Corrected demonstration:** the target study passed its input, schedule,
  history and resource checks. Continue the resulting five-record history,
  unchanged final-period boundaries and explicit declarations for later studies.
- **Parallel development:** run the separate factor study while qualifying
  larger data and building economic accounting and negative controls. Shared
  datasets, schedules and registries must remain consistent across the work.
- **Larger study:** require a passed source gate, tested transformations, a
  selected comparison protocol and measured resource estimates before scaling.
  Refresh Colab to the coherent release used by that study.
- **Final evaluation:** freeze data, preprocessing, model choice, holdings,
  costs and acceptance rules before any authorized final evaluation. A changed
  source or target needs a declared exposure-aware plan, not a silently renewed
  holdout. Do not open the final block merely to finish a milestone.
- **Handoff:** reproduce the chosen workflow, scoring and recovery from its
  documented inputs. Review all nine workstreams against the completion contract
  and name any open requirement before reporting the goal complete.

Report progress by evidence produced: a corrected assumption, tested behavior,
completed experiment, qualified data decision or reproduced workflow. Track
remaining work and the next dependency alongside it. More commits, model names,
training hours or consumed tokens are not measures of research success.
