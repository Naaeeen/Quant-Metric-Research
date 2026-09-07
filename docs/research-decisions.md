# Research decisions

Last reviewed: 2026-09-08

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

## Real-data entry preparation: diagnostics before an adapter

The next useful increment is an offline input audit plus a human provider-evidence
checklist, not another model or a general certification system. A market, vendor,
access agreement and independent sample have not yet been established. This is
our bounded engineering choice, not a claim that one vendor or workflow is best.

Research exposed distinctions that the input contract alone cannot verify:

- [Qlib's PIT database](https://qlib.readthedocs.io/en/stable/advanced/PIT.html)
  separates a financial statement's period from publication date and retains
  amendments. Downloading today's corrected history and hashing it does not
  reconstruct what was available at an earlier decision time.
- [Norgate's FAQ](https://norgatedata.com/data-package-faq.php) distinguishes
  effective-date membership and stable asset IDs from ticker strings. It also
  documents continuously applied corrections without versioning, and no
  delisting-return/reason or post-delisting-event data. Its suggested final-bar
  liquidation is a vendor approximation, not proof of a fixed-horizon terminal
  return. These are documented product limitations, not findings from our samples.
- [Norgate's content tables](https://norgatedata.com/data-content-tables.php)
  identify synthetic pre-March-2000 ASX membership histories and certain
  Canadian histories using current methodology. Record coverage by period and
  distinguish reconstructed history from observed eligibility.
- [QuantConnect's security identifiers](https://www.quantconnect.com/docs/v2/writing-algorithms/key-concepts/security-identifiers)
  distinguish permanent security identity from mutable tickers. Its
  [security master](https://www.quantconnect.com/docs/v2/writing-algorithms/datasets/quantconnect/us-equity-security-master)
  represents corporate-action events separately from underlying equity prices.
  Provider documentation informs the checks; it does not verify a QMR dataset.
- [Datasheets for Datasets](https://arxiv.org/html/1803.09010v8) motivates
  documenting composition, collection, preprocessing, uses, distribution and
  maintenance, including explicit unknowns. We adopt a small factual checklist
  rather than treating filled fields as a certificate. Usage rights require
  separate evidence; for example, [Norgate's EULA](https://norgatedata.com/subscribe/eula.php)
  restricts redistribution and commercial use.

The resulting `audit_inputs` API and `qmr audit-inputs` command validate existing
raw-input contracts and report coverage by date/security, members with enough
adjacent historical price pairs and future label-endpoint presence. They calculate no returns,
signals or models. Normalized-required-column/request fingerprints identify the
audited inputs but exclude extra fields and original file bytes. All external
checks stay unverified and Stage 4 remains blocked. Exit code zero confirms
report generation only; the checklist in `data-contract.md` still needs a human
record of provider claims, independent sample checks and unresolved gaps.

This audit cannot establish session completeness from benchmark bars alone:
compare against an independent exchange calendar. It also reads future price
presence and hashes supplied values, so it is not a sealed holdout. Acquisition
timestamps and reproducible snapshots help trace later changes, not establish
missing historical vintages. No provider was selected or empirically audited.

The same boundary review corrected these earlier-path issues:

1. Daily Stage 1 dates and the Stage 2 cutoff now reject intraday, timezone-aware
   and numeric epoch inputs instead of silently normalizing them. Duplicate or
   empty decision-date requests fail explicitly.
2. The `run_research` screening copy masks outcomes with unknown label ends or
   label ends after its end-of-day cutoff. Same-day matured labels are allowed.
   The original panel and contemporaneous feature rows remain available for
   coverage, redundancy and PCA. This correction is local to that workflow:
   Stage 3 and walk-forward folds retain their stricter label-end-before-test
   purging, and the generic screening helper is unchanged.
3. Reject boolean, temporal and complex prices before pandas can reinterpret
   them as numbers, and reject raw duplicate CSV headers before pandas can
   rename them. Independently reviewed real-file and mixed-type regressions
   cover these input boundaries.

After agreeing the research specification and access rights, inspect a permitted
sample across dated membership, identifier changes, actions, delisting and
revision cases. Only then implement the necessary adapter and lifecycle tests.

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

## September 2026 provider decision: prepare an offline ASX demonstration

The next release prepares a Yahoo-format file import for a fixed, presently
selected ASX cohort: `BHP.AX`, `CBA.AX`, `CSL.AX`, `NAB.AX`, `RIO.AX`, `TLS.AX`,
`WES.AX` and `WOW.AX`. These selections precede inspection of price outcomes.
They define a demonstration cohort, with selection and survivorship bias; they
do not assert historical ASX200 membership. The
[offline intake example](../examples/README.md#declared-intake-only-demonstration-scope)
declares a 2015-01-01 through 2026-02-28 inclusive export window and weekly
decision dates during 2016-2025, before data review. Actual coverage,
an independent exchange calendar, historical identifiers and terminal returns
remain unverified. This increment imports and audits inputs; it opens no training
or final-test evaluation and establishes no empirical alpha.

`VAS.AX` supplies the proposed adjusted ETF return proxy. Vanguard states that
VAS seeks to track the S&P/ASX 300 before fees, expenses and tax. An adjusted ETF
price series is not the official ASX200 total-return index, and provider adjustment
quality remains a separate check.
[Vanguard's product description](https://www.vanguard.com.au/adviser/invest/etf?portId=8205&productType=etf)

The existing Financial-Investment-Tool source was inspected on 2026-09-07:
`server/requirements.txt` pins `yfinance==0.2.65`; `server/src/metrics.py` adds one
calendar day to the inclusive UI end date and calls `yf.download` with
`group_by='ticker'`, `auto_adjust=False`, `threads=False` and `progress=False`.
Its short-lived in-memory cache is not an archival data source. Its separate
price-field helper can select `Close` when `Adj Close` is absent. The undated
`data/asx200.csv` has 198 rows; the universe-sync script excludes `IFL.AX` and
`NSR.AX`. This seed and its filtered output do not establish dated membership.
No market-data download or original price snapshot was produced by this inspection.

The QMR importer requires flat, per-symbol `Date` and `Adj Close` exports and
never substitutes `Close`. It preserves original file bytes with a manifest and
an audit so later transformations can be traced. This verifies file identity and
schema, not the adjustment method: upstream yfinance 0.2.65 itself can substitute
Close when the source omits adjusted-close values. Yahoo describes adjusted close
as reflecting splits and distributions, but a column heading is not independent
evidence that those events were correctly applied.
[Pinned yfinance parser](https://raw.githubusercontent.com/ranaroussi/yfinance/0.2.65/yfinance/utils.py),
[Yahoo adjustment explanation](https://help.yahoo.com/kb/SLN28256.html)

The chosen file format does not select or authorize an acquisition method.
Yahoo's Australian terms restrict automated collection without prior permission;
its download help documents subscription-dependent historical CSV access.
yfinance's software license does not grant rights to Yahoo data. These sources
leave access, local reuse and any sharing for this project to be established;
they do not support a blanket claim that every Yahoo use is prohibited. No
automatic downloader, new data-provider dependency, purchase or redistribution
is included, and no authorized real dataset is available in this release.
[Yahoo AU terms](https://legal.yahoo.com/au/en/yahoo/terms/otos/index.html),
[Yahoo export help](https://help.yahoo.com/kb/sln2311.html),
[yfinance's upstream notice](https://github.com/ranaroussi/yfinance)

Nasdaq WIKI was investigated but not selected. Its legacy documentation describes
public-domain data and the `WIKI/<ticker>` time-series route; this is distinct from
the `WIKI/PRICES` Tables route. The legacy getting-started page requires a key for
each request, while current Tables documentation both requires a key and publishes
anonymous-call limits. The WIKI product documentation and retirement FAQ returned
404 during this review. Current endpoint availability and applicable access terms
remain unresolved; Tables requirements alone do not settle the time-series route.
No price endpoint was probed. A separately verified frozen archive could support
a historical engineering example, but would not resolve this ASX cohort's needs.
[Legacy WIKI route and metadata](https://docs.data.nasdaq.com/v1.0/docs/in-depth-usage),
[Legacy authentication](https://docs.data.nasdaq.com/v1.0/docs/getting-started),
[Tables authentication](https://docs.data.nasdaq.com/docs/api-and-analysis-tools-for-tables-data),
[Tables anonymous limits](https://docs.data.nasdaq.com/docs/rate-limits-1)

The next evidence gate is an authorized export plus its source/access record,
followed by observed date, action, missingness and identity checks. Original-byte
snapshots cannot recover historical provider vintages or certify delisting returns.
Stage 4 remains blocked pending the existing data and research review requirements.

## September 2026 public-archive development declaration

The user delegated source and implementation choices and requested autonomous
progress beyond the unavailable local ASX exports. The ASX intake remains
supported; this is a separate **US engineering/development demonstration**, not
fulfilment of historical ASX membership or permission to trade.

Before calculating features or inspecting model outcomes, the declared source is
Chi Seng Pun's 2018 Mendeley Data V3 archive, DOI `10.17632/ndxfrshm74.3`:
`sp500-1216.csv` (2012-2016 daily adjusted prices) and `FF3-0317.csv`.
The public metadata identifies a January 2017 constituent snapshot and labels
the dataset CC BY 4.0, while noting that third-party content may require further
permission. This supports the publisher-declared research archive use here, not
independent verification of Yahoo/French redistribution rights. Raw exports and
row-level derived artifacts stay local and ignored by Git.
[Dataset](https://data.mendeley.com/datasets/ndxfrshm74/3),
[versioned metadata](https://data.mendeley.com/public-api/datasets/ndxfrshm74/snapshot/3).

Fixed choices for the first run:

- Select the first 30 normalized stock-column labels alphabetically from the
  frozen file, before screening values; retain sparse histories, with no
  replacement based on coverage or performance. This arbitrary bounded cohort
  and the archive's survivor selection prevent an unbiased-universe claim.
- Use daily observations from 2012-01-03 through 2016-12-30, with decision dates
  from 2013-01-02 through 2016-12-30. The immature price-history tail stays visible.
- Construct `FF_MARKET_PROXY` from archived daily `(Mkt-RF + RF) / 100`, starting
  at 100 on the first included session. It is a derived broad-market return
  index, not an ETF or the official S&P 500 total-return index. Do not use future
  returns to define features. The benchmark calendar is supplied, not certified.
- Keep the ten existing trailing metrics, a 252-session lookback, minimum 126
  observations, 20-session forward excess return, one-session entry lag,
  252-session annualization and zero assumed risk-free rate for trailing ratios.
- Train only Ridge (`alpha=1`), against all individual-metric, best train-only
  metric and equal-weight oriented-rank baselines. Use fold-local screening,
  imputation, scaling and ranks; no model-family or parameter search after results.
- Reserve 63 final decision dates without evaluating their outcomes. Development
  uses three outer 63-date windows (minimum 252 training dates), each with two
  inner 42-date windows (minimum 126 training dates), and purges label overlap.
  Require a minimum 20-stock cross-section, three quantiles, coverage 0.8,
  redundancy threshold 0.9, HAC lag 19 and random seed 42.
- Audit raw coverage and build the panel without the separate `qmr run` screening
  path. Persist no-training preflight before registered development, and retain
  one durable local registry outside individual run directories.

The benchmark construction follows
[French's market-factor definition](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_factors.html);
fold-local transforms follow
[scikit-learn's leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).
This run can demonstrate data handling and actual model training. Its acceptance
remains `not_evaluated` and Stage 4 remains false; retrospective revisions,
survivorship, corporate actions, stable identifiers and delistings are not
certified. Never present the reserved block as physically sealed or historically
unseen merely because the workflow does not evaluate it.

### Observed development result, 2026-09-08

The declared experiment completed without changing its cohort, model, features
or split settings. An earlier usage-interrupted attempt remains in the same
registry as run `a9dc0e7b-4240-4205-b393-ba33a5bdab6d`; its partial output was
not overwritten or treated as a completed result. The successful new-directory
retry is `4fdd4eba-4c8f-459c-9e0f-981d2ebf47d0`.

The source sample has 1,258 sessions and 722 explicit missing stock prices
(ABBV: 250; ALLE: 472). No sparse member was replaced. The resulting panel has
30,240 stock-date rows over 1,008 decision dates. Development evaluation covers
189 dates; the primary baseline and Ridge have 100% prediction and evaluation
coverage on both native and common scopes.

| Development mean Rank IC | Equal-weight rank | Ridge |
| --- | ---: | ---: |
| Outer fold 1 | 0.03615 | 0.08545 |
| Outer fold 2 | -0.13397 | -0.23891 |
| Outer fold 3 | -0.01320 | 0.01044 |
| Overall | -0.03701 | -0.04767 |

Ridge is better in two folds but substantially worse in the second. It does
not improve the predeclared aggregate ranking criterion. The mean raw
top-minus-bottom 20-session spread is also negative for both methods
(-0.005446 for equal ranks, -0.004336 for Ridge); these overlapping signal
statistics are not annual returns, implemented portfolios, or net PnL.

Conclusion: the real-data engineering path works, but this experiment provides
no stable positive-signal or model-improvement result. Do not reverse directions,
change the sample, tune parameters or open the final block to manufacture a win.
This does not establish that all metrics or ML are universally ineffective.

Independent review matched all 24 recorded artifact hashes, the registry link,
and source fingerprint
`7d91f3c65c811e1ecf1357159d6b594cce6ba04a496cfc19fb293493c022e5f0`.
The final 63 eligible decision dates (2016-09-01 to 2016-11-30) were not evaluated;
there is no final experiment record or exposure. Acceptance remains
`not_evaluated`, with provenance and Stage 4 eligibility false. The raw files,
row-level results, and canonical registry remain local. Version 0.7.0 code was
released in commit `eff6bd14b753f4aad7640c5c022ccca9878534ca`, with 695 local
tests passing, 89.76% aggregate branch-tracked coverage, and successful Linux
Python 3.11/3.13 CI. The next engineering milestones are in the roadmap.

## September 2026 Colab portability decision

The user requested later Colab execution. Version 0.8.0 makes the existing
development experiment portable rather than changing its research hypothesis.
The cohort, ten metrics, target, Ridge parameter and purged folds remain those
of the frozen public-archive declaration. The negative result above is not
reversed, retuned or replaced, and the reserved final block remains unevaluated.

The selected workflow uses a CPU runtime, immutable source revision and
dedicated constrained virtual environment. Google documents that Colab VMs
have limited lifetimes and their files/libraries are not included when a notebook
is shared; its runtime guidance recommends installing required library versions.
Those platform constraints motivate explicit setup and preservation of research
evidence, not an assumption that every future Colab runtime is compatible.
[Colab FAQ](https://research.google.com/colaboratory/faq.html),
[runtime guidance](https://research.google.com/colaboratory/runtime-version-faq.html).

The existing local canonical registry is exported with the archive and prior
run evidence under aliases `archive`, `run001` and `run002`. The user privately
uploads that intact seed folder to Drive; the notebook does not start a blank
registry. Running and failed records, including the interrupted first attempt,
remain part of the history. The whole checkpoint chain is validated before
restoring a live registry into a new local VM work directory.

SQLite's backup API supplies a consistent local snapshot; a closed snapshot is
then copied as a file to checkpoint storage. SQLite warns that filesystem
locking and synchronization over a network can fail, so the design does not
open the live database on Drive. Checksummed file copies and read-back checks
are evidence of what was observed, not proof of remote server durability.
[SQLite backup](https://sqlite.org/backup.html),
[network-filesystem cautions](https://sqlite.org/useovernet.html).

Append-only start and completion markers preserve successful and catchably
failed attempts. An incomplete or corrupt generation blocks the next automatic
run rather than falling back to older history. This bounded single-user helper
has no distributed lock, automatic repair, background job or final-evaluation
entry point. The notebook refuses recognized `/content/drive` work paths; callers
remain responsible for identifying other nonlocal mounts and keeping source/live
SQLite local.

Reusing configuration-driven data, model and signal-analysis components is
consistent with Microsoft's public
[Qlib workflow](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml).
That is an architectural reference, not evidence that this archive is point-in-time
or that more factors/models will improve it. No new model-selection exploration
is part of the portability release.

The notebook and its builder live under `examples/`; the
[walkthrough](../examples/README.md#colab-development-walkthrough) describes
private upload and interruption handling. Local tests, a pinned environment
and Linux verification must be reported separately from hosted execution.
No actual hosted Colab execution, empirical alpha, final acceptance, independently
verified provider provenance or Stage 4 readiness is claimed by this decision.

### Verified portability result: 2026-09-08

The local notebook executed all five code cells sequentially in 667.41 seconds,
including a fresh source clone, isolated installation, real-data training and
checkpoint publication. Its library source is pinned to commit
`7740dee1b4c6aeed463d6440511d7b606c6fe406`. The later notebook formatting/import
correction does not change computation; independent syntax-tree review confirmed
equivalence apart from the placement of the standard-library CSV import.

The installed runtime was Python 3.14.3, NumPy 2.5.3, pandas 3.0.5,
scikit-learn 1.9.0 and SciPy 1.18.1. Separately,
[Linux CI on 68a6de1](https://github.com/Naaeeen/Quant-Metric-Research/actions/runs/34162757260)
passed Python 3.11, 3.12 and 3.13; Python 3.12 used the research constraints,
passed all 833 tests and reported 89.38% aggregate coverage with branch tracking.
The original local full suite passed 832 tests; the generated-notebook lint
regression was then added and verified, with all 16 notebook tests passing.
Lint, formatting, package build, dependency consistency and an independent
third-party vulnerability audit also passed. The unpublished local package is
not advisory-indexed and was reviewed as source instead.

Independent consistency checks found no numerical change in the existing
development outputs. Prior registry entries and evidence were preserved.
This is reproducibility evidence, not a new independent alpha experiment.
Run identifiers, local handoff state and the executed notebook remain private.
Final outcomes were not evaluated, acceptance is `not_evaluated` and Stage 4
eligibility remains false. Hosted Colab, Drive authorization and remote-storage
durability were not tested by this local execution.

## September 2026 fixed-window factor capability decision

The next milestone adds an opt-in catalog and pure calculator, not another model
search on the observed public archive. Historical source availability, membership,
identifiers, adjustments and delistings remain the larger obstacles to an alpha
claim. A small software capability is still useful for exercising explicit,
testable factor definitions without pretending those data gaps are solved.

The three declared formulas are `return_21s`,
`momentum_252s_skip_21s` and `ma_distance_63s`, as specified in the
[fixed-window contract](data-contract.md#opt-in-fixed-window-price-factors).
They use only adjusted close; no OHLCV, fundamentals, sentiment, new source
acquisition or paid resource is introduced. Their signs stay raw; any later
orientation must be learned inside the appropriate training fold.

French's daily momentum construction separates prior 2–12-month performance,
while its short-term reversal construction uses recent performance. Those are
size/return-sorted portfolios requiring inputs this calculator does not have.
They motivate distinguishing horizons, not calling these stock-level formulas
French factors or reproducing their missingness conventions.
[Daily momentum definition](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_mom_factor_daily.html),
[daily short-term reversal definition](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_st_rev_factor_daily.html).

Microsoft Qlib includes lagged-close and moving-average/close ratios with explicit
rolling windows. Our names/formulas are explicit and are not exact Alpha158
replicas; OHLCV-dependent features are excluded.
[Qlib feature implementation](https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py).
The 21/63/252-session choices and complete-interval rule are this project's
declared engineering conventions, not claims about universal best settings.

Historical input validation is global, but numerical conversion for arithmetic
is local to each required factor window. pandas can choose an integer or floating
conversion dtype based on the supplied values. Converting the entire history
first could therefore let an excluded decimal string change rounding inside an
otherwise unchanged window. A regression test covers this exact dependency.
[pandas numeric conversion](https://pandas.pydata.org/docs/reference/api/pandas.to_numeric.html).

These factors are additional representations of the same prices. With complete
matching endpoints, `1 + return_252s` equals
`(1 + momentum_252s_skip_21s) * (1 + return_21s)`. Splitting horizons does not
create independent information. The existing legacy `trailing_return` uses its
first/last nonmissing prices, so this identity is not promised for sparse legacy
windows. Its existing behavior is preserved, not silently redefined.

Keep `DEFAULT_FEATURE_COLUMNS`, `PanelConfig`, the ten legacy metric formulas,
the declared public experiment and the existing Colab source pin unchanged.
Adding package files legitimately changes source identity; old experiments stay
reproducible through their pinned revision and original manifests, not by hiding
new source files from fingerprinting.

The release gate is synthetic arithmetic, boundary, missingness, future-perturbation,
extreme-value and immutability tests plus independent review and normal software
checks. Automatic panel merging and empirical comparisons are deferred. A later
comparison needs an explicit adapter and predeclared feature bundle, unchanged
cohort/target/folds, native/common coverage reporting and retained exposure history.
It must not treat reused development dates as fresh evidence or open final outcomes.

## September 2026 panel enrichment decision

The next capability is an opt-in wrapper around the existing panel builder, not
a second membership/label engine or an additional model runner. This keeps the
new factor metadata aligned with the exact legacy rows and targets. Rebuilding
both from the same input avoids accepting an unrelated panel and price history.
The [adapter contract](data-contract.md#opt-in-factor-panel-enrichment) documents
the strict whole-table intake inherited from Stage 1 and the original-scalar
window arithmetic used by the new factors.

Qlib distinguishes inference processing from learning-only operations that may
depend on labels. That boundary supports retaining missing-target rows here,
without adopting its whole processing stack.
[Qlib data-handler documentation](https://qlib.readthedocs.io/en/latest/component/data.html#data-handler).
Learned selection, scaling, imputation and PCA remain inside the relevant training
fold, consistent with [scikit-learn's leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).

Append results in the existing builder's row order with explicit collision checks.
An implicit join could remove unmatched rows or multiply keys; pandas also matches
null join keys, so any future keyed implementation needs explicit nonnull keys and
one-to-one cardinality checks.
[pandas merge contract](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html).

The two immutable feature bundles are candidate schemas, not full experiment
preregistrations. Stage 3 accepts their feature tuples without changing PanelConfig
or the fixed public experiment. A panel built with a smaller legacy subset is not
silently expanded to satisfy a full-ten schema.

A fair future cross-bundle comparison is a separate gate. Existing `common`
metrics align models within one run, not feature bundles from different runs;
each run also fits its own equal-rank baseline. Comparing summary rows alone is
therefore insufficient. Later comparisons must use the same enriched panel,
source/runtime, targets and split settings; verify matching prediction keys,
outcomes and scoring universes; retain native coverage and identify the original-
ten baseline explicitly. Never regenerate model scores or input cross-sectional
transforms after filtering on outcome availability. Spearman still ranks its
evaluated pairs by definition; the existing descriptive spread uses available
outcome pairs and is not a realized portfolio backtest. Preserve every attempt
and prior exposure.

This milestone tests software on invented observations only. It performs no new
empirical model search, final scoring, source acquisition or data certification.
The previous Colab source pin and private history remain unchanged. Adding code
and columns changes current fingerprints normally; it does not rewrite old
manifests or exempt new files from identity checks. Data provenance remains a
larger prerequisite for an alpha claim than the number of candidate factors.

The follow-up code audit prioritizes inference handling before cross-bundle
significance tests: existing HAC paths remove unavailable daily metrics and apply
lags to the remaining sequence. This does not establish a fixed-session lag
interpretation when dates are missing. Stage 2 can also omit unavailable dates
entirely, so a correction must preserve or explicitly supply the expected schedule,
not merely remove one missing-value filter. Consecutive, equally spaced periods
are an explicit assumption in the
[statsmodels HAC documentation](https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_hac.html).
Do not infer exchange sessions from generic business-day dates. A conservative
unavailable-inference status is preferable to silently presenting unsupported
gap-aware significance; this is the next methodology-hardening task, not a new
empirical search in this release.

## September 2026 scheduled-inference correction

Version 0.11 implements the prior audit's smallest methodological correction:
retain a supplied observation schedule and withhold unsupported HAC inference.
The change spans the numerical primitive, Stage 2 summaries, Stage 3 summaries
and paired final acceptance; changing only the final missing-value filter would
not recover dates already omitted upstream. The detailed
[contract and limitations](stage3-benchmark.md#scheduled-inference) are normative.

This is deliberately not a new missing-data estimator. The declared lag remains
in scheduled-observation units, and gaps are not compressed, imputed or silently
bridged. Descriptive available-pair statistics and feature selection retain their
meaning; an unavailable p-value is reported explicitly and cannot pass the final
significance gate. A schedule assembled from supplied panel dates cannot prove
that the panel itself contains every intended trading session.

Verification uses invented data, including dates with no valid feature IC,
missing paired model results and duplicate dates across folds. No empirical model
search, final-outcome evaluation or historical result rewrite is part of this
correction. Package source identity and artifact schema advance normally; the
earlier Colab source pin and retained private evidence remain unchanged.

The frozen local verification passed 1,230 tests in 187.62 seconds with 90.35%
aggregate coverage and branch tracking. The new scheduled-inference module has
100% coverage; the numerical statistics module has 96%. Independent numerical
review also checked 765 complete-case synthetic examples against the old
calculation with exact results. Repository lint/format, dependency consistency,
dependency vulnerability audit and source/wheel builds passed. These are software
checks, not a new alpha experiment or hosted Colab result. Linux CI is recorded
separately on the release pull request.
