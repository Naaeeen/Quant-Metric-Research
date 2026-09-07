# Research roadmap

## Current boundary

Stages 1 and 2 build the point-in-time panel and screen individual metrics. The
Stage 3 benchmark engine is also implemented: it trains supervised ranking
models, evaluates purged development folds, and selects a model family. Opening
the reserved final test requires completed matching development evidence and a
durable local outcome-date reservation, not just an opt-in flag. The repository still
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

The next engineering milestone is now implemented in version 0.4:

- no-training preflight reconstructs the purged outer/inner schedules and reports
  coverage and sample-size warnings without ranking metrics or screening features;
- a local SQLite registry keeps hypothesis, data/code/config/runtime identity,
  fixed boundaries, frozen family and run status, including failed attempts;
- referenced final runs validate identity before development is repeated, then
  atomically reserve the final outcome envelope before final fitting;
- all recorded development and final outcome envelopes block a proposed
  overlapping final test, including failed/interrupted reservations and forward
  label tails. A renamed study or changed configuration does not reset dates;
- the offline synthetic CLI example exercises preflight, development, one final
  evaluation, experiment history and a refused second evaluation.

This guard prevents local workflow accidents, not intentional bypass, earlier
human inspection or unregistered research. It does not establish empirical alpha.

Version 0.5 adds the raw-input readiness milestone: `qmr audit-inputs` checks
coverage and future endpoint presence without calculating research outcomes.
Its normalized-input fingerprints support comparison, not historical
provenance certification. The same audit cycle tightened daily dates and price
types, rejected duplicate CSV headers, and excluded immature targets from the
single-cutoff Stage 2 screen while retaining the original panel.

Version 0.6.0 adds `qmr import-yahoo`, an offline adapter for flat, per-symbol
`Date` and `Adj Close` exports. It stores exact source bytes and SHA-256 hashes,
normalizes supplied inputs, preserves missing-price evidence, and writes an
input audit. It does not download data, add a yfinance dependency, reconstruct
membership or substitute `Close`. The final intake manifest marks completed
processing; failures retain partial evidence and require a fresh destination.
Neither a success marker nor `imported_at` verifies provider provenance or the
original acquisition date. All external data gates remain unverified.

The selected ASX demonstration cohort and VAS adjusted ETF proxy are documented
in [the provider decision](research-decisions.md#september-2026-provider-decision-prepare-an-offline-asx-demonstration).
This specification was chosen before inspecting price outcomes. It accepts
present-selection bias as a demonstration limitation and does not claim
historical ASX200 coverage. No real prices were downloaded and no model training
or final-test evaluation was performed for this intake release.

Version 0.7.0 adds a separate public US archive demonstration to make development
possible without supplying ASX exports. `qmr fetch-public-sample` acquires the two
pinned Mendeley V3 files and license records; `qmr public-demo` verifies them,
audits coverage, builds the daily metric panel, runs preflight and registers a
fixed Ridge development experiment. It uses the first 30 stock labels
alphabetically, retains missing histories and constructs `FF_MARKET_PROXY` from
archived market-factor returns. The declared final block is not evaluated.
See [the frozen declaration](research-decisions.md#september-2026-public-archive-development-declaration)
and [the runnable example](../examples/README.md#public-archive-development-demo).

This milestone exercises real historical observations under a publisher-declared
CC BY license; it does not certify underlying third-party rights or historical
provenance. The retrospective January 2017 constituent snapshot is unsuitable
for an unbiased-universe claim. Raw data and row-level artifacts remain local
and ignored. ASX access remains unresolved, and neither a completed development
run nor favorable statistics can satisfy the independent data-evidence gate.

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

## Next engineering sequence: Colab and controlled factor research

The September 8 review prioritizes portability and reproducibility before more
model complexity. The declared real-data run is complete; the portability
release carries its evidence forward, while factor expansion remains future work:

1. Preserve the completed real-data run and earlier interrupted attempt.
   Ridge did not improve aggregate development Rank IC over equal-weight ranks;
   retain that result without changing the declared model or cohort to improve it.
2. Remove measured duplicate screening work only behind output-equivalence
   tests. Keep Stage 2 diagnostics, pairwise missingness, fold-local fitting,
   label purging, and all out-of-sample evaluation semantics unchanged.
3. Version 0.8.0 adds a CPU-only Colab walkthrough around the existing library:
   pinned source, a constrained isolated environment and bounded result displays.
   Seed the private handoff from the existing canonical registry and archive/
   prior-run evidence; never silently replace it with a blank history. Keep live
   SQLite on local VM storage, with append-only evidence checkpoints in
   user-selected persistent storage. Unresolved interruptions block reuse.
   Local and Python 3.12 Linux verification are distinct from a hosted Colab run;
   the walkthrough does not claim hosted execution or offer final evaluation.
   The local five-cell notebook run and all three Linux Python versions now pass;
   see the [portability evidence](research-decisions.md#verified-portability-result-2026-09-08).
4. Define a small factor catalog before computing additional features: formula,
   required inputs, lookback, availability, adjustment assumptions and missingness
   rules. Start with price-only candidates supported by the archive; OHLCV,
   publication-lagged fundamentals and sentiment require separate source evidence.
5. Predeclare ablations and comparisons with the existing PCA and tree families;
   retain all attempts and exposure history. Additional horizons and later
   portfolio work remain gated by the data and evaluation requirements below.

Colab can discard VM files and changes preinstalled libraries over time; its
[FAQ](https://research.google.com/colaboratory/faq.html) and
[runtime guidance](https://research.google.com/colaboratory/runtime-version-faq.html)
motivate environment isolation and explicit persistence. SQLite warns against
assuming reliable locking/synchronization on
[network filesystems](https://sqlite.org/useovernet.html), so a mounted Drive
database is not the intended live-registry design. These safeguards are
single-user accident prevention, not distributed or tamper-proof governance.
The [Colab example](../examples/README.md#colab-development-walkthrough) gives
the private-folder handoff and failure procedure; the
[checkpoint contract](data-contract.md#notebook-history-checkpoints) defines
what is verified. File read-back is not a guarantee of remote durability.

The modular data/model/evaluation separation follows the public
[Qlib workflow](https://qlib.readthedocs.io/en/latest/component/workflow.html).
Its [factor definitions](https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py)
are implementation references, not a reason to add unsupported inputs or claim
that a larger factor set will improve this sample.

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
