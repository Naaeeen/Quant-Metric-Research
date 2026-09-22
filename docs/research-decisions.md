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
As revised in the September 22 roadmap, these development diagnostics precede
final evaluation.

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

## September 2026 schedule-evidence prerequisite

Before comparing feature bundles, inspection found that schema 5 retained HAC
counts and statuses but not the actual phase schedule. A later report cannot
recover dates absent from all prediction or metric rows. Version 0.12 therefore
persists the exact sequence using the same shared derivation as inference,
instead of immediately adding a comparison runner or an arbitrary artifact loader.
See the [schedule evidence contract](stage3-benchmark.md#stored-evaluation-schedules).

Nested immutable mappings and ISO-string tuples fit the existing JSON writer and
registry metadata format. Development contains only its own sequence; full mode
adds the planned locked sequence. The whole mapping is deliberately not inserted
into registry identity equality because legitimate development/full mappings
differ. Existing source, panel, package, configuration, runtime and boundary
identity continues to constrain the derivation. Exposure guards are unchanged.

The next comparison should explicitly separate original-ten equal ranks,
original-ten same-family model and candidate-feature same-family model, then
align keys, outcomes, native/common coverage and exact development schedules.
Saved metadata supports consistency checks, not authenticated predictions,
complete outside research history or independence from earlier selection.
No real-data experiment, historical rewrite, final-outcome evaluation or Colab
repinning is part of this prerequisite.

The frozen local verification passed 1,256 tests in 197.88 seconds with 90.38%
aggregate coverage and branch tracking. The shared schedule module and benchmark
reporting module both have 100% coverage. Synthetic replay tests preserve missing
scheduled dates through JSON and Parquet, and a synthetic registry lifecycle
checks that final reservation still precedes final evaluation. Independent review,
repository lint/format, dependency consistency, dependency vulnerability audit
and source/wheel builds passed. These are software checks, not a new empirical
result, provenance certification or hosted Colab run. Linux CI is recorded
separately on the release pull request.

## September 2026 controlled development comparison

The next question is whether additional features help the same model family,
not simply whether a different model wins on a different sample. Version 0.13
therefore separates three saved-score arms: the original-bundle equal-rank
baseline, the original-bundle model and the extended-bundle same-family model.
The family is an explicit caller choice; no new fitting or automatic selection
occurs inside the comparison. The equal-rank arm uses features surviving the
original bundle's fold-local screening and orientation, not necessarily all ten.

Reusing saved predictions for signal analysis follows the separation seen in
[Qlib's signal evaluation source](https://raw.githubusercontent.com/microsoft/qlib/main/qlib/contrib/eva/alpha.py)
and [Recorder design](https://qlib.readthedocs.io/en/latest/component/recorder.html).
The project retains its own spread convention and scheduled-date contract; it
does not adopt Qlib's optional missing-date removal, half-gross spread scaling
or adjacent-date autocorrelation behavior. No Qlib dependency or pickle loader
is needed.

Producer inspection and independent review sharpened two safeguards. A
development-labeled manifest cannot override explicit final-evaluation markers
elsewhere in the object; those scope checks precede prediction access. Prediction
dates must exactly equal the saved schedule inside each fold's evaluation window:
identical extra rows in between-fold gaps would otherwise pass cross-arm key
equality. Scoring rows excluded from supervised evaluation because of unavailable
outcomes must still remain visible. Without the original panel, jointly omitted
excluded rows cannot be reconstructed from the assignment table alone.

Matching complete keys and outcomes precedes the three-arm finite-score
intersection. This avoids silently dropping unequal rows; pandas also explicitly
warns that [null join keys match](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html).
Native metrics remain visible because the common population also depends on
the candidate model's score availability. IC and spread retain independent
outcome masks. Comparative means use paired daily differences, with separate
paired-date counts and coverage, not differences between independently averaged
arm summaries.

The comparison exports descriptive statistics only. Choosing a bundle from
development results is still model selection, not fresh validation, consistent
with [scikit-learn's nested-CV explanation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
Its shuffled classification example is not adopted for time-ordered equity data.
Matching producer identities support a conditional check of supplied objects;
the current comparison source/runtime identity is recorded separately. Neither
provenance, authenticity nor complete prior research history is certified. No
new empirical experiment, final evaluation, historical rewrite or Colab repin
is part of this release.

Implementation review also found that default tolerant table equality and
float64 conversion could hide distinct supplied outcomes. The comparator checks
their original supported numeric values exactly, separately from evaluation
arithmetic; decimal construction/comparison does not round through an ambient
arithmetic context. See [Python's decimal contract](https://docs.python.org/3.11/library/decimal.html).
This can conservatively reject a decimal string versus a nearby binary float.
For finite paired daily means, the standard-library
[arithmetic mean](https://docs.python.org/3.11/library/statistics.html#statistics.mean)
replaces custom scaled summation, with cancellation and overflow regressions.
These are numerical consistency checks, not new estimators or market evidence.

Final independent review also checked the comparator against the producer's
`evaluation_cross_section` and `_fold_assignment`, not only matching-arm fixtures.
An explicit missing-label or locked-test-overlap exclusion requires both saved
outcomes to remain masked; a missing-target exclusion masks the target, while an
evaluation-role row retains its target. Contradictory masks now fail rather than
being silently repaired. Scores and independently missing realized returns remain
visible. This prevents matching but internally impossible evidence from bypassing
the producer's label-time boundary; it still does not authenticate caller objects.

Final local verification after this correction: **1,409 tests passed** in about
217 seconds with **90.44% aggregate coverage including branch tracking**. The
genuine synthetic two-run comparison passed independently, with no fitting,
artifact loading or registry access inside the comparator. Independent review,
127-file lint/format checks, dependency consistency and vulnerability audit,
installed API/command checks and source/wheel builds passed. These software
checks add no empirical result, final outcome exposure or hosted Colab claim.
Linux verification is recorded separately on the release pull request.

## September 2026 explicit Colab source refresh

The Colab entrypoint now explicitly pins the independently reviewed
[0.13 source](https://github.com/Naaeeen/Quant-Metric-Research/commit/dadcb6c699f8d3ab2cf512669b74abff64c98f75).
The fixed CPU ten-metric Ridge experiment and dependency constraints are unchanged;
new price factors and bundle comparisons remain opt-in outside this notebook.
This adopts the newer schedule/consistency safeguards, not a new hypothesis or a
claim that rerunning observed data supplies independent evidence.

Direct comparison with the old 0.8 pin found the history helpers, experiment
registry, public-demo entrypoint and constraints unchanged. History schema 1 is
separate from benchmark artifact schema 6: existing generations are retained,
not migrated, rewritten or relabeled under a new identity. A synthetic declared
old identity can test retention but is not execution under the old interpreter.

This is an explicit compatibility milestone, not automatic repinning after every
release. [Colab runtime guidance](https://research.google.com/colaboratory/runtime-version-faq.html)
supports installing required library versions as runtimes change. The existing
local-compute/private-checkpoint separation remains appropriate given Colab's
[temporary runtime and Drive limitations](https://research.google.com/colaboratory/faq.html).
No remote durability or distributed-locking guarantee is inferred.

The target public source passed a fresh isolated constrained bootstrap/API/CLI
check and [all three Linux Python jobs](https://github.com/Naaeeen/Quant-Metric-Research/actions/runs/34172921551),
including constrained Python 3.12. The earlier real-data five-cell/11-minute
execution belongs to the 0.8 pin. Neither a full refreshed five-cell real-data
run nor hosted Colab/Drive execution is claimed; the examples guide records the
remaining private-history, Drive authorization and run-in-order steps.

Final frozen refresh verification: **1,410 tests passed** in about 225 seconds
with **90.44% aggregate coverage including branch tracking**. The extended
existing synthetic workflow preserves prior file/directory entries and bytes,
raw completed/failed/running records and an existing final reservation, appending
exactly one current-identity development record/exposure. Both notebook summary
sources read that genuine synthetic output and its schema-6 schedule evidence.
No old interpreter or real archive acquisition was exercised by that fixture.
Independent ten-file review, 128-file lint/format, dependency consistency/audit
and source/wheel builds passed. Production source, package version, experiment
defaults and dependency constraints were not changed by this refresh. Its Linux
CI result is recorded separately on the refresh pull request.

## September 2026 descriptive comparison persistence

Version 0.14 adds a narrow persistence boundary around `FeatureBundleComparison`,
not another training runner. Registered development can already return the two
comparison-ready runs; the existing comparison checks their conditional
consistency. Saving that result should not repeat model selection or evaluation.
[Qlib's record templates](https://raw.githubusercontent.com/microsoft/qlib/main/qlib/workflow/record_temp.py)
also separate prediction records from saved signal-analysis outputs. We adopt
that separation, not pickle loading, optional skipping, a new experiment manager,
or annualized claims from overlapping returns.

The writer saves five descriptive tables and strict JSON metadata. It validates
exact definition-1 table schemas and fixed textual labels before any write, so
direct construction cannot smuggle an inference column into a descriptive
export. [OWASP's CSV injection guidance](https://owasp.org/www-community/attacks/CSV_Injection)
explains why CSV quoting alone is insufficient. The narrow schema rejects
formula-like text instead of modifying legitimate numeric values; it is not a
universal spreadsheet sanitization facility. Original metadata strings remain
JSON data, and genuine missing paired means are not replaced with zero.

[pandas' CSV interface](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html)
supports in-memory serialization before creation, explicit missing-value
representation and stable newline choice. Files use exclusive creation in a
fresh local destination. The completion manifest is published last and any
existing marker is refused. [Python's rename documentation](https://docs.python.org/3.11/library/pathlib.html#pathlib.Path.rename)
distinguishes Unix replacement behavior from Windows refusal, so a last-moment
path check is not advertised as race-proof publication. This is a single-writer,
non-hostile-local-filesystem contract, not distributed storage or durability.

Failure handling retains partial evidence and the original exception, including
the ambiguous case where publication succeeds and then reports an error. No
automatic cleanup or retry occurs. Output digests are byte-consistency metadata,
not signatures or proof of authenticated history. The artifact-format version
is independent of benchmark schema 6 and comparison definition 1; both stay
unchanged. The separately verified Colab pin remains at 0.13. No new empirical
study, final evaluation, data acquisition, history rewrite, model/default change
or dependency is introduced. Provider provenance still gates the next empirical
study; persistence is not evidence that any added factor improves a ranking.

Final frozen local verification: **1,508 tests passed** in about 246 seconds,
with **90.48% aggregate coverage including branch tracking**. The 97 focused
writer checks include physically preserved partial bytes, short writes,
write-plus-close errors, pre-existing entries and publication-then-error cases.
The existing genuine synthetic two-run comparison is reused for export parity;
no additional model pair or empirical study is introduced. Independent review,
130-file lint/format, dependency consistency and vulnerability audit, installed
0.14 API/command checks and source/wheel builds passed. Linux results are recorded
separately on the release pull request. No hosted Colab or remote-durability claim
is added by these software checks.

## September 22, 2026 direction review

### Decision

Keep the stock-ranking engine. Redirect the next milestones toward completed
target, feature and model experiments, a stronger dataset, and development-time
economic diagnostics. A rebuild is not justified by this review.

Two GPT-6 assessments examined the method independently: one started without
conversation history; the other received the current implementation and known
results. A third reviewed data/model alternatives, and a fourth reviewed writing.
The assessments agreed on the main priorities. Their agreement is a review
input; the evidence below is the basis for the changes.

The completed public study used 30 alphabetically selected stocks from a
retrospective 2017 membership list. Its 189 development dates gave mean Rank IC
of -0.04767 for Ridge and -0.03701 for equal-weight ranks. This rejects an
improvement claim for that run, not Ridge or ML in general. The three additional
price factors, HGB and PCA are implemented but have not yet established an
improvement on market data.

### Findings and actions

| Area | Finding | Action |
| --- | --- | --- |
| Research direction | Stock-date ranking, chronological evaluation and transparent baselines match the question. | Retain the architecture and test new hypotheses through it. |
| Engineering balance | Several releases improved evidence storage without completing new model comparisons. | Prioritize an executable ablation over another report wrapper. |
| Data | A small retrospective survivor cohort cannot settle the broader modeling question. | Keep it as a development demo; qualify a broader named-stock dataset separately. |
| Target | Squared-error fitting on raw returns is a legitimate baseline, but may emphasize extreme-return dates. | Compare it with within-date rank targets under identical folds and search budgets. |
| Target reporting | Inner tuning calculates its spread from the training target rather than the separately configured realized return. | Fix the units and independent missingness masks before a rank-target experiment. |
| Calendar | Benchmark observations define sessions; a date absent from every series disappears. | Add independent calendar comparison before conclusions that depend on session offsets. |
| Feature freshness | Legacy trailing return drops missing endpoints and can report an older endpoint as an eligible window. | Specify endpoint/freshness behavior and add a regression before changing the legacy metric contract. |
| Validation | Purging, fold-local fitting, contemporaneous scoring universes and common/native reports address real failure modes. | Retain them and add target-transformation and negative-control tests. |
| Economics | The earlier roadmap postponed all portfolio work until after final evaluation. | Develop and freeze holdings, costs and risk rules before opening the final test. |
| Compute | Current models fit a CPU workflow, but historical whole-notebook time is not model-only training time. | Measure preprocessing, training and evaluation separately before scaling. |
| Writing | README mixes release history, instructions and repeated interpretation limits. | Lead with purpose, results and runnable paths; keep detailed contracts at their point of use. |

The target-reporting bug is diagnostic: candidate selection uses mean Rank IC,
not the incorrectly reported spread. A synthetic check with rank targets
[0.25, 0.5, 0.75, 1.0], ascending scores and returns [0.01, 0.02, 0.03, 0.04]
reported 0.5 instead of the correct two-bucket return spread of 0.02.

Subtracting one benchmark return from every stock on a date preserves that
date's ordering. It does not neutralize beta or sector exposures. Raw excess
returns and their within-date ranks therefore offer a clean training-target
comparison while the evaluation can stay in raw-return units.

### Immediate implementation plan

1. Publish the direction review, comprehensive roadmap and shorter README.
   Keep behavioral changes in a separate commit.
2. Correct inner-tuning spreads. Test distinct target/return columns, their
   independent missingness, same-column compatibility, unchanged predictions
   and real rank-target training.
3. Add a small target-ablation example using the existing panel validator,
   preflight, registry, benchmark and evaluator. Run raw-return and rank-target
   arms, each with Ridge and HGB, over the same original-ten feature panel.
   Save each ordinary benchmark bundle and report a five-arm common/native
   comparison with one retained equal-rank baseline.
4. Require a common label-end date within each decision-date cross-section
   before computing rank targets. With heterogeneous maturity, an immature
   peer's outcome could otherwise influence a retained training rank.
5. Record explicit configs, input/script identity and elapsed time before and
   around the two runs. Use existing history, preserve partial results, and
   leave the final block untouched. Compare saved, already-masked outcomes;
   joining full panel outcomes back would undo the development boundary.
6. Review, test and publish the workflow. Then declare its market-data run
   separately, after checking the source and feature-freshness assumptions.

The example compares training targets, not feature bundles. Its report does
not use the existing feature-bundle writer, whose contract requires matching
target configurations. Longer-term work and completion criteria are in the
[roadmap](roadmap.md).

### Data and model choices

Named-stock research remains the main track. Numerai provides useful published
methods, but its data is not yet qualified for this project's use. Its IDs change
across eras, and its targets and CORR scoring have their own definitions. Any
future adapter would need to preserve those meanings; it must not invent ticker
histories, trading dates or portfolio returns.
Current documentation identifies v5.3 and explicit Ender-20/Ender-60 targets;
the generic target alias means Ender-60 in that version.

A follow-up check changed the initial data recommendation. The
[terms dated August 31, 2026](https://crypto.numer.ai/terms), section 4.2, limit
data use to tournament participation, prohibit redistribution and impose
conditions on service-provider access. Public API access is not permission for
an unrelated research benchmark. Defer adoption until the intended use and
permissions are established. No dataset download, account creation, submission
or cloud upload was performed during this review.

Qlib provides reference workflows and factor definitions, but its current
README says the official dataset is temporarily disabled. A community download
is a new data-source decision, not a verified substitute. Alpha158 also needs
inputs beyond this archive's adjusted closes. SEC filings can later enrich
features if accession, filing availability, revisions and security mapping are
handled explicitly.

Exercise Ridge and HGB before adding another library. LightGBM regression is
the next tree challenger. LambdaRank requires date groups, integer relevance
and a declared gain mapping; CatBoost ranking has different group-weight
semantics. Neither is a drop-in replacement for the current row-weighted loss.

For larger-data neural work, compare a small MLP or TabM under a matched
temporal protocol. TabPFN is a separate pretrained comparator with checkpoint
access and licensing requirements. Generic tabular benchmark results motivate
candidates, not expected equity returns.

### Primary sources checked

| Source | What informs this plan |
| --- | --- |
| [Qlib LightGBM workflow](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml) | Separate train/validation/test periods, MSE regression, and specified portfolio/cost analysis alongside signal reports. |
| [Qlib label processors](https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py) | Cross-sectional label normalization is a concrete alternative to raw-return fitting. |
| [Qlib data preparation](https://github.com/microsoft/qlib#data-preparation) | Current download availability differs from older instructions. |
| [Qlib factor loader](https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py) | Factor definitions require declared price/volume inputs and windows. |
| [Gu, Kelly and Xiu](https://dachxiu.chicagobooth.edu/download/ML_BKP.pdf) | Chronological estimation, tuning and testing; nonlinear models studied on a much broader asset-pricing dataset. |
| [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) | Fit preprocessing on training data only. |
| [Nested model selection](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html) | Separate tuning from evaluation; this repository uses time-based, not the example's classification, folds. |
| [Histogram gradient boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html) | Current estimator behavior, weighting and early-stopping controls. |
| [Numerai data](https://docs.numer.ai/numerai-tournament/data) | Versioned eras, changing IDs, feature subsets and explicit target horizons. |
| [Numerai terms](https://crypto.numer.ai/terms) | Section 4.2 changes the data recommendation: tournament-limited use is not a general research-data license. |
| [Numerai CORR](https://docs.numer.ai/numerai-tournament/scoring/correlation-corr) | The official metric transforms predictions and targets; it is not Spearman Rank IC. |
| [SEC APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) | Filing-level facts and frames have different availability/revision semantics. |
| [LightGBM ranker](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html) and [parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html) | Query groups, integer relevance, gain mappings and deterministic CPU settings. |
| [CatBoost ranking objectives](https://catboost.ai/docs/en/concepts/loss-functions-ranking) | Group and object weights differ across ranking losses. |
| [Revisiting tabular deep learning](https://arxiv.org/abs/2106.11959) | Matched evaluation protocols and strong simple neural baselines; no universal winner. |
| [Tree models on tabular data](https://arxiv.org/abs/2207.08815) | Trees are an important baseline under constrained tuning budgets. |
| [TabReD](https://arxiv.org/abs/2406.19380) | Temporal rather than random splits can change relative model rankings. |
| [TabM](https://arxiv.org/abs/2410.24210) | A parameter-efficient MLP ensemble candidate for a later larger-data experiment. |
| [TabPFN repository](https://github.com/PriorLabs/TabPFN) | Current checkpoint defaults, access and model-weight licensing need explicit review. |
| [Colab FAQ](https://research.google.com/colaboratory/faq.html) | Resources and runtime duration vary; measure and checkpoint bounded runs. |
| [Google writing guide](https://developers.google.com/style/tone) and [code review guide](https://google.github.io/eng-practices/review/reviewer/looking-for.html) | Direct prose, useful comments and review for unnecessary complexity. |

These are published methods and interfaces. They describe what can be adopted
and tested here, not the private production practices of trading firms.

### Version 0.15 implementation and verification

Inner tuning now calculates spreads from the configured realized-return column,
independently of target missingness. Candidate selection still uses Rank IC.
Ten regression cases cover separate and shared columns, missing outcomes,
column-name collisions, unchanged inputs, and genuine rank-target training.
Scaling realized returns by ten scales the reported spreads by ten without
changing predictions or selected candidates.

The target-ablation example trains Ridge and histogram boosting against raw
returns and within-date ranks. It retains the original-ten equal-rank baseline,
existing research history, identical folds and already-masked raw outcomes.
The five-arm report includes native and common coverage; elapsed time is measured
per target run, including both learners and nested tuning. Input snapshots,
declarations, standard run bundles and partial failures remain available.

Twelve integration cases exercise the real training workflow, target maturity,
future-outcome perturbations, common coverage, input changes, reruns and failure
recovery. These runs use synthetic fixtures. The next market-data study remains
separate from software verification, with calendar and freshness checks first.

Final frozen local verification: **1,530 tests passed** in 255 seconds with
**90.47% aggregate coverage including branch tracking**. The executable example
also passed its separate coverage run at **91.30%**. Lint and formatting checks,
dependency consistency and vulnerability audit, command checks, independent
review, and source/wheel builds passed. Linux CI is recorded on the pull request.
The existing Colab notebook stays pinned to 0.13; hosted execution of the new
example and predictive improvement have not been demonstrated.

## Version 0.16: session completeness and price endpoints

The direction review found two data issues worth fixing before the next market
comparison: a date absent from every series was invisible to a benchmark-derived
calendar, and trailing return could replace a missing endpoint with an older
observation. This release fixes those assumptions without changing the model
families, declared public demo or pinned Colab notebook.

`ExpectedSessionCalendar` records sessions, inclusive coverage bounds, source
and version. Its full declaration is fingerprinted. The input audit reports
calendar differences; panel, factor and research builders reject mismatches
before calculation. Registered development snapshots the declaration and checks
its bytes before registration and after training. The old optional behavior
remains available for reproducing existing demonstrations.

Trailing return now uses the original window endpoints or remains missing.
Two endpoint flags distinguish missing current prices from the minimum paired
history requirement. Complete-data values, the other nine formulas, active
members and forward labels are preserved. This changes sparse-input behavior;
new studies must rebuild both comparison arms from the same corrected inputs.

An independent review also found that reusing a Stage 1 output directory could
leave an old calendar beside new results. The writer now requires a fresh
destination, and the CLI checks it before starting work. Regression tests retain
every prior artifact byte when a rerun is refused.

The implementation follows explicit exchange sessions rather than generic
weekdays. [exchange_calendars](https://github.com/gerrymanoim/exchange_calendars)
provides an upstream session interface; the [NYSE calendar](https://www.nyse.com/trade/hours-calendars)
distinguishes full closures from early closes. Neither a holiday list nor the
mere presence of benchmark bars proves the completeness of a supplied dataset.
[pandas percentage changes](https://pandas.pydata.org/docs/reference/api/pandas.Series.pct_change.html)
operate on supplied positions, so an absent row must be detected before return
calculation. No calendar dependency or automatic download was added.

Verification includes missing first/middle/last market dates, malformed daily
declarations, missing benchmark history, endpoint arithmetic, unchanged complete
data, CLI persistence and genuine synthetic Ridge/SQLite development. Registered
runs preserve earlier history. Tamper tests block registration or refuse
completion while retaining any completed development record. Independent source
review found no remaining actionable issue. The local full suite passed
**1,606 tests** with **90.87% aggregate coverage
including branch tracking**; release tooling and Linux results are recorded with
the pull request.

Next, qualify the calendar for the existing archive and run the declared
target-only development comparison. A date-only check found 1,258 matching dates
in its stock and factor files, including the absence of both 2012 Sandy closure
dates; full equality against an independent calendar still needs verification.
The latest checkpoint and live registry contain three matching development
records, whereas the older original registry has two. Continue the latest
lineage. No additional market-data training or final evaluation was performed
for this release; historical universe and source-rights gaps remain unchanged.

### Archive calendar qualification, September 22, 2026

The complete in-scope date sets in both pinned archive files match XNYS from
`exchange_calendars==4.13.2`: **1,258 sessions** over 2012-01-03 through
2016-12-30, with no missing, extra or duplicate dates. There are 250 sessions in
2012 and 252 in each subsequent year. Both Sandy closure dates are absent.
An independent second calculation reproduced the ordered declaration and both
date-set comparisons. The four archived file hashes still match their manifest.

The calendar was generated in an isolated environment with explicit bounds:

~~~python
import exchange_calendars

exchange = exchange_calendars.get_calendar("XNYS", start="2012-01-03", end="2016-12-30")
sessions = exchange.sessions_in_range("2012-01-03", "2016-12-30")
~~~

Its declaration uses source `https://github.com/gerrymanoim/exchange_calendars`
and version `4.13.2:XNYS`. The canonical calendar fingerprint is
`45de2bd427aaa02ce5cbbe1b388564d80eaa7b143b67300c7935e4ca31ec83ba`.
The [pinned XNYS implementation](https://github.com/gerrymanoim/exchange_calendars/blob/4.13.2/exchange_calendars/exchange_calendar_xnys.py)
includes irregular closures as well as regular holidays and early closes.

Projected reads of the latest checkpoint's normalized inputs also passed:
38,276 price keys contain no duplicates or null keys, the benchmark retains all
1,258 sessions, and all 1,008 unique decision dates belong to the declaration.
These checks read date/symbol columns and hash file bytes; they do not calculate
returns, fit models or write experiment records. Scripts, runtime versions,
the declaration and the two reports are retained locally under
`artifacts/calendar-qualification-20260922-001/`.

This establishes agreement with the pinned community calendar, not per-security
price completeness or independently certified historical provenance. The next
step is the corrected-panel target comparison using the latest registry lineage.

## September 22, 2026 comprehensive goal contract

The user requested a more detailed goal covering the whole research system.
The nine roadmap workstreams already cover that scope. A fresh independent
review identified the missing pieces as completion criteria, dependencies and
clear evidence levels, rather than additional model families.

The [expanded goal](roadmap.md#goal) now requires qualified data, completed
controlled experiments, an independently reviewed conclusion, tested trading
accounting, reproducible Colab/scoring workflows and a maintainable handoff.
It separates software verification, market-data findings and hosted execution.
Conditional methods need a hypothesis and budget; required deliverables remain
open until their evidence exists. A negative result can resolve a research
question without supporting portfolio promotion.

The review also added decision-time batch scoring and offline monitoring tests
to the handoff workstream. A saved model needs reproducible preprocessing and
must score without future labels. Monitoring should be exercised with stale
inputs, delayed outcomes and interruptions before operational use.

Primary sources rechecked for this change:

- [Google's Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml)
  separates infrastructure tests from model evaluation and addresses
  training/serving differences. It informs the scoring-parity requirement.
- [The ML Test Score](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/)
  treats testing and monitoring as work beyond an offline model experiment.
- [Qlib's published workflow](https://github.com/microsoft/qlib#auto-quant-research-workflow)
  connects dataset construction, training, signal evaluation and portfolio
  analysis, including results with and without costs.
- [Google's writing guide](https://developers.google.com/style/tone)
  supports direct instructions, concrete wording and short explanations.

These sources inform the project-specific completion contract; they do not
establish that any candidate will improve this dataset. The order remains the
corrected target study, separate factor study, qualified larger-data comparison
and reproducible handoff, with economic checks developed alongside the models.
This documentation change performs no training, final evaluation or registry
operation.

## September 22, 2026 corrected-panel target study declaration

Study `mendeley30-targets-20260922-001` tests whether learning within-date
percentile returns improves development ranking relative to learning return
magnitudes. It uses the existing 30-stock retrospective demonstration, not a
newly qualified investment universe. This declaration precedes model fitting.

### Fixed design

- Rebuild the original ten metrics from checkpoint `000001` inputs using 0.16
  and the independently checked XNYS declaration. Preserve 252-session lookback,
  126 minimum paired observations, zero annual risk-free rate, 252 annualization
  sessions, one-session entry delay and a 20-session holding horizon.
- Compare raw forward excess returns with average percentile ranks among
  observed outcomes on each date. Retain missing labels and every feature row;
  require a shared label maturity within each cross-section. Raw returns remain
  the economic evaluation outcome.
- Use Ridge with alpha 1 and histogram boosting with learning rate 0.05,
  seven leaves, L2 penalty 1, 200 iterations, minimum leaf size 20 and seed 42.
  Disable boosting early stopping and PCA. Each family has one parameter choice.
- Keep cross-sectional feature ranks, fold-local screening at 0.8 coverage and
  0.9 redundancy, date-balanced model loss, and the original-ten equal-rank
  baseline. Require 20 stocks for a daily metric; use three return buckets.
- Use three outer 63-date folds with at least 252 training dates and two inner
  42-date folds with at least 126 training dates. Purge by actual label maturity.
  Retain HAC lag 19 for existing diagnostics; decisions below are descriptive.
- Keep final decisions at 2016-09-01 through 2016-11-30, with maximum final label
  date 2016-12-30. Assert both preflights have identical outer/inner schedules
  and training-label maxima before fitting. Save the exact schedules and all
  configuration fields with input, source and calendar hashes.
- Continue the latest three-record registry, retaining the interrupted record.
  Save before/after SQLite backups and prove that every prior row is unchanged.
  Add development records only; do not invoke final evaluation.

The existing example runs raw then rank targets, each with both model families:
36 supervised fits across inner training and outer refits. Use one CPU thread.
The execution ceiling is two wall-clock hours from launching the ablation
process, including loading, preflight, fitting and output writing, and 4 GiB
peak process working set. Input preparation is timed separately. These are
limits, not runtime estimates. Preserve partial evidence after an interruption;
do not retry automatically or expand the search in response to scores.

### Declared comparisons and decisions

Report all six contrasts on the saved five-arm common population: rank minus
raw within each family, and each of the four learned arms minus equal-rank.
Also retain native coverage and all individual arm summaries.

A contrast needs at least 180 of 189 scheduled dates with finite paired IC
and 180 with finite paired raw-return spreads. Each fold must have at least
60 of its 63 dates for each measure. Keep the original schedule and missing
dates; calculate IC and spread differences on their respective paired dates.

For a target transformation to merit further investigation, require mean paired
IC improvement of at least 0.01, positive mean IC improvement in at least two
of three folds, and nonnegative mean paired spread improvement. A learned-arm
comparison against equal-rank additionally requires positive absolute IC and
spread for the candidate on the same paired dates. These are project-specific
continuation thresholds, not significance or economic-materiality claims.

Classify coverage failure as `inconclusive`; with sufficient coverage classify
all conditions passing as `promising`, otherwise `not_supported`. Unfinished
computations are `incomplete`. A promising rank transformation may still leave
both models ineffective; report that distinction. Selecting later work from
these development contrasts does not make the selected result unbiased.

Report feature/endpoint coverage, zero-input scoring rows, ties, fold effects,
paired-date counts and runtime. Keep final-period numerical outcomes out of
inspection and selection. The local input snapshots contain full panel data;
they are private research files, not a physically sealed holdout.

### Research basis and next decision

[Qlib's cross-sectional rank processor](https://github.com/microsoft/qlib/blob/main/qlib/data/dataset/processor.py)
provides a concrete rank-normalization reference; this study uses percentiles
without Qlib's subsequent centering and rescaling.
[Scikit-learn's nested-validation guidance](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
supports keeping selection separate from evaluation. The current
[boosting API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)
confirms why early stopping is explicitly disabled here. Fresh and
history-informed reviews found no prerequisite implementation blocker.

After the run, apply these rules before choosing further modeling work. Keep
the separate ten/thirteen-feature study, larger-data qualification and economic
accounting on the roadmap regardless of whether this target change helps.

### Corrected-panel target study results

The declared study completed on September 22. Raw-return histogram boosting
met the continuation criteria against equal-weight ranks. Neither rank-target
transformation met its criteria. Ridge remained below the baseline in aggregate.
The results below are development findings from the fixed retrospective cohort.

The rebuilt panel contains 30,240 rows over 1,008 decision dates and 30 stocks.
Its SHA-256 is
`de2062b1d80ffe503bc69106b838f36d172b9444e450123e03551810442432f5`.
Both saved preflights equal the preparation records. Both model runs have the
same source, panel, runtime and evaluation schedule; configurations differ only
in the training target. Snapshot hashes match the declaration.

All five arms have 5,670 finite scores: 30 stocks on each of 189 development
dates. Native and five-arm common results coincide. Every contrast has 189
paired IC dates and 189 paired spread dates, with 63 of each in every fold.
There are no zero-observed-input rows in these evaluation predictions.

The spread is the mean top-minus-bottom three-bucket **20-session outcome**,
shown in percentage points. It is not a daily portfolio return and includes
neither trading costs nor a holdings simulation.

| Arm | Mean Rank IC | Mean spread (pp) | Dates with tied scores |
| --- | ---: | ---: | ---: |
| Equal-weight ranks | -0.03701 | -0.5446 | 183 |
| Raw-return Ridge | -0.05001 | -0.4572 | 0 |
| Rank-target Ridge | -0.05764 | -0.8061 | 0 |
| Raw-return histogram boosting | 0.01506 | 0.3501 | 38 |
| Rank-target histogram boosting | -0.01270 | 0.1158 | 13 |

Each contrast below is candidate minus reference on paired dates. The fold
count reports positive mean IC differences, not the candidate's absolute IC.

| Contrast | Mean IC change | Mean spread change (pp) | Positive IC folds | Decision |
| --- | ---: | ---: | ---: | --- |
| Rank Ridge minus raw Ridge | -0.00763 | -0.3489 | 1/3 | not_supported |
| Rank boosting minus raw boosting | -0.02775 | -0.2343 | 2/3 | not_supported |
| Raw Ridge minus equal-rank | -0.01301 | 0.0874 | 2/3 | not_supported |
| Raw boosting minus equal-rank | 0.05207 | 0.8947 | 2/3 | promising |
| Rank Ridge minus equal-rank | -0.02063 | -0.2615 | 1/3 | not_supported |
| Rank boosting minus equal-rank | 0.02431 | 0.6604 | 1/3 | not_supported |

The rank-target boosting arm illustrates why an improved aggregate difference
is insufficient: its absolute IC remains negative and it beats equal-rank IC
in only one fold. Raw-return boosting meets the declared rule, but its positive
average also conceals a weak middle period:

| Development fold | Equal-rank IC | Raw Ridge IC | Rank Ridge IC | Raw boosting IC | Rank boosting IC |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2015-11-02 to 2016-02-02 | 0.03615 | 0.08550 | 0.07106 | 0.11924 | 0.12894 |
| 2016-02-03 to 2016-05-03 | -0.13397 | -0.24722 | -0.21473 | -0.15548 | -0.14241 |
| 2016-05-04 to 2016-08-02 | -0.01320 | 0.01168 | -0.02924 | 0.08141 | -0.02462 |

Raw boosting's corresponding spread means are 1.3975, -1.1264 and 0.7792 pp.
Its paired spread improvement over equal-rank is positive in each fold. This
short, overlapping sample and retrospective membership do not establish a
stable or tradable signal. No significance claim is attached to the declared
continuation threshold.

All outer screens retain the same six feature names: benchmark correlation,
trailing return, annualized volatility, max drawdown, beta and historical 5%
VaR. Their order changes in the third fold; raw and rank arms agree exactly
within folds. The equal-rank baseline averages training-oriented ranks of those
six retained features. The declaration's "original-ten" refers to its candidate
bundle, not an unscreened average of all ten metrics. The implemented screened
baseline was fixed before this study; these results do not test an all-ten average.

Feature availability is lower before the evaluation interval. Across 27,720
pre-final development rows, trailing-return coverage is 97.381%; the other nine
features have 98.290% coverage. There are 726 missing window-start endpoints,
222 missing decision-price endpoints and 474 zero-input rows. These are retained
and reported, not removed to improve the evaluation coverage figures above.

#### Runtime, history and verification

Preparation took 93.12 seconds. The raw and rank runs took 571.43 and 575.96
seconds respectively, each including both families and nested validation.
The full ablation process took 1,185.61 seconds (19.76 minutes), including
loading, preflight and output generation, within its two-hour ceiling.
The actual training interpreter's observed peak working set was 508,157,952
bytes (484.62 MiB), below 4 GiB; observed CPU time was 1,166.19 seconds.

The initial execution observer measured the Windows virtualenv launcher rather
than its training child. Its memory/CPU numbers are not used. A separate
observer attached to the verified child PID and sampled the OS lifetime peak
every second, including memory used before attachment. The final subsecond
interval can be missed. Both original records and the correction are retained.
Independent synthetic profiling ran concurrently, so these timings describe
this run, not an isolated comparison of model speed.

The runtime was Windows, Python 3.14.3, package 0.16.0, NumPy 2.5.3,
pandas 3.0.5, SciPy 1.18.1 and scikit-learn 1.9.0, with one training thread.
Training used source revision `1b36ecd265a1b091a3d1159686f5dc5d4ea2d5d7`.
This was a local run, not hosted Colab verification.

The registry retains every field of all three prior runs/exposures unchanged,
including the interrupted record. It adds two completed development runs:

- Raw target: `99de0c72-8a1f-4d88-b9fb-e223f9c04e59`.
- Rank target: `8caeb19e-49b7-4cc2-b626-44586779ca68`.

The after-study backup and live registry match at five runs/exposures, with
zero final runs or exposures. Final-period boundaries remain unchanged and
final numerical outcomes were not used in this comparison. An independent
audit reproduced the registry and manifest identity checks.

The paired postprocessor passed 18 synthetic tests before use. Independent
calculations from saved development metrics reproduced all six decisions.
Local evidence is under `artifacts/target-ablation-20260922-001/`: preparation
and preflight records, immutable input snapshots, both runs, all native/common
metrics, resource observations, before/after history and `paired-report-001/`.
Only aggregate findings are published; inputs and row-level outputs stay local.

#### Next decision

Retain raw-return boosting for further controlled tests; do not adopt rank
targets or promote a model from this result. Complete the separate 10/13-factor
comparison, negative controls, economic accounting and broader-data
qualification. Before another screening-heavy run, test whether reducing
small-table overhead can preserve exact selections while improving runtime.
The broader goal and final-evaluation requirements remain unchanged.

### Screening allocation optimization plan

An independent invented-data profile identified repeated pairwise table
operations as the next measured optimization target. On 600 dates, 30 stocks
and ten features, profiled redundancy took 92.44 seconds, daily IC 26.42 seconds
and IC summary/HAC 0.20 seconds. These include profiler overhead and concurrent
work; they are not estimates of the full training runtime.

Keep the scalar Spearman calculation and replace per-pair table slicing and
missing-row removal with date-local arrays and availability masks. Preserve
pair-specific reranking, ties, original-dtype uniqueness checks, pair/date order,
clipping and aggregation. Public input validation and caller-owned data stay
unchanged. No values may be reused across training folds.

A private prototype matched 40 oracle cases and two downstream screen cases.
One unprofiled 120-date, 30-stock, ten-feature comparison took 8.71 seconds for
the existing implementation and 2.20 seconds for the prototype. Validate the
integrated implementation with a failing allocation-count regression, exact
output/selection checks, independent review and the full suite. Then compare
three fresh-process runs per implementation on the same synthetic fixture,
alternating order, recording wall/CPU time and whole-process peak memory.

A direct replacement with `DataFrame.corr` was rejected. The five-value pair
`[0,1,2,3,4]` and `[0,1,2,4,3]` produces `0.8999999999999998` through the existing
scalar path but `0.9` through the matrix path in the tested runtime. That changes
selection at the configured threshold. Mathematical agreement alone is not
enough for a behavior-preserving optimization.

[Pandas documents pairwise complete observations and index alignment](https://pandas.pydata.org/docs/reference/api/pandas.Series.corr.html).
The two arrays here come from the same ordered rows, after their shared mask.
[SciPy documents the scalar Spearman calculation and constant-input behavior](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html).
Tests must preserve the current conversion and threshold behavior, including
large integers and nullable inputs, instead of accepting a numerical tolerance
that changes selected features.

This change does not rerun the market study or change its conclusion. The
package/source identity changes with the implementation, so saved study
identities remain attached to their original revision. Keep the notebook pin
unchanged until its separate reproducibility milestone.

### Screening allocation optimization results

Version 0.16.1 uses date-local value arrays and availability masks instead of
allocating a two-column table for every pair/date. It preserves scalar SciPy
arithmetic and pair-major aggregation. Extension-backed columns keep exact
object values until after pairwise filtering and raw uniqueness checks.

The allocation regression first failed with 18 per-pair `DataFrame.dropna`
calls; 40 other cases passed. Independent review then found that nullable
large integers with missing values could convert to float before uniqueness.
Two new signed/unsigned regressions reproduced the mismatch in warning
behavior, and object extraction corrected it. The resulting 73 focused tests
passed. Independent checks also reproduced exact results on 80 randomized
panels and warning behavior for NumPy, nullable and Arrow-backed integer cases.

The planned fresh-process measurement used one thread, three alternating runs
per implementation, and 120 dates x 30 stocks x ten tied/missing numeric features.
Every pairwise output matched exactly across all six runs.

| Implementation | Wall seconds, all runs | Median wall seconds | Maximum process peak (MiB) |
| --- | --- | ---: | ---: |
| Before, revision `fb60080` | 9.177, 8.826, 8.708 | 8.826 | 180.79 |
| Array screening | 1.607, 1.610, 1.605 | 1.607 | 179.70 |

The median wall-time ratio is 5.49. Timing includes date-local conversion and
redundancy calculation, but excludes imports and fixture setup. Memory is the
OS lifetime peak of the actual child interpreter, including imports and inputs,
not incremental allocation by the function. Local machine activity was not
isolated. This is a synthetic component comparison, not a measured reduction
in total training time or a claim about larger datasets.

Measurement used Windows, Python 3.14.3, NumPy 2.5.3, pandas 3.0.5 and SciPy
1.18.1. The integrated redundancy source hash is
`0aa070bdea206f795d67e61bd3c4cdfac88c23d40be5437936e9f18789e32698`.
The private measurement script, individual outputs, timings and process peaks
are retained in `artifacts/screening-validation-20260922-001/`. No empirical
inputs, predictions, experiment records or final outcomes changed.

Release checks passed locally: 1,615 tests in 269.49 seconds with 90.90%
aggregate coverage and branch tracking, lint/format, dependency consistency,
dependency vulnerability audit, source/wheel builds and the installed command.
The editable project is excluded from the dependency audit. The built wheel
contains the measured redundancy source. The first full run caught a stale
version assertion; it was updated to 0.16.1 before the successful full rerun.
Independent review approved the code, regression cases and measurement claims.

## September 22, 2026 feature-bundle study declaration

Study `mendeley30-features-20260922-001` asks whether the three fixed price
factors improve raw-return histogram boosting under the existing training and
screening procedure. The completed target study motivates HGB as the primary
family; Ridge remains a fixed linear control. This is further development on
already-used periods, not independent confirmation of the earlier finding.

### Inputs and fixed training procedure

Build one enriched panel from the same normalized archive prices, memberships,
decision dates and `PanelConfig` as the corrected target study. Supply the pinned
XNYS calendar with fingerprint
`45de2bd427aaa02ce5cbbe1b388564d80eaa7b143b67300c7935e4ca31ec83ba`.
Verify every original column, dtype, row and order against the saved corrected
panel. Retain missing observations, warm-up rows and immature outcomes. Both
new runs must use the same enriched panel; the older ten-feature result cannot
substitute for a new arm with matching source, runtime and full-panel identity.

The expected shape is 30,240 stock-date rows, 30 stocks and 1,008 decision dates.
This is still the retrospective 2017 cohort with 2012-2016 prices. It does not
resolve historical membership, identifiers, delisting outcomes or source rights.

| Setting | Declaration |
| --- | --- |
| Candidate inputs | `legacy10_v1` versus `legacy10_plus_price3_v1` |
| Added factors | `return_21s`, `momentum_252s_skip_21s`, `ma_distance_63s`, formula version 1 |
| Target and realized outcome | `forward_excess_return`; next-session close entry, 20-session holding label |
| Ridge | Alpha 1 |
| Histogram boosting | Learning rate 0.05, seven leaves, L2 1, 200 iterations, minimum leaf size 20, seed 42; early stopping disabled |
| Preprocessing | Cross-sectional feature ranks; train-fold screening, imputation and scaling; date-balanced model loss |
| Screening/evaluation | Coverage 0.8, redundancy 0.9, minimum cross-section 20, three spread buckets, HAC lag 19 |
| Outer development | Three 63-date folds; minimum 252 training dates |
| Inner validation | Two 42-date folds; minimum 126 training dates; labels mature before each evaluation window |
| Budget | Two registered development runs, 36 supervised fits; one training thread; two hours and 4 GiB peak working set |

The two-hour clock starts when the ablation process launches and includes loading,
preflight, training, comparison and output writes. Raw-panel preparation is measured
separately. The private execution observer enforces these limits; the reusable
example itself only fixes the training thread count and records per-arm elapsed time.

The [factor contract](data-contract.md#opt-in-fixed-window-price-factors) fixes
complete required windows: 22 prices for return, 232 prices in the older momentum
interval requiring 253 calendar positions, and 63 prices for moving-average
distance. These are not the legacy 126-observation eligibility rule. Screening
may remove any candidate inside a training fold; the study tests bundle value
through that complete procedure, not an unconditional contribution by each factor.

Both configurations differ only in `feature_columns`. Keep all other exported
settings unchanged, including inactive final-test settings. Both no-training
preflights must agree on temporal boundaries and nested fold evidence; their
feature-coverage diagnostics may differ. Development evaluation remains:

1. November 2, 2015 through February 2, 2016: 63 dates.
2. February 3 through May 3, 2016: 63 dates.
3. May 4 through August 2, 2016: 63 dates.

Final decisions remain September 1 through November 30, 2016, with maximum label
date December 30, 2016. `evaluate_lockbox` stays false. Preserve all five existing
development records/exposures, including the interrupted attempt. A successful
study appends exactly two development records to that same history. No automatic
retry, parameter expansion or final evaluation follows a failure.

### Comparisons and continuation decision

Produce the existing feature-bundle report separately for HGB and Ridge. Each
uses its own three-arm finite-score intersection: extended model, legacy model
and the legacy run's screened, training-oriented equal-rank baseline. The
baseline is not necessarily an average of all ten candidate metrics. Do not use
the extended run's equal-rank baseline or manually freeze the prior six selected
features. Retain native results alongside each common-sample comparison.

There are four declared contrasts: extended minus legacy model and extended
minus legacy equal-rank, separately for HGB and Ridge. For each of the two HGB
contrasts, require all of the following:

- At least 180 of 189 paired dates for IC and independently for spread, with at
  least 60 of 63 pairs for each measure in every fold.
- Mean paired IC improvement at least 0.01, with positive mean paired IC
  improvement in at least two of three folds.
- Mean paired raw-return spread improvement at least zero.
- Positive candidate mean IC on that contrast's IC-paired dates and positive
  candidate mean spread on its independently paired spread dates.

An unfinished run or evidence bundle is `incomplete`. Insufficient paired
coverage in either HGB contrast makes the study `inconclusive`. With adequate
coverage, both contrasts passing gives `promising`; otherwise the result is
`not_supported`. Apply the same calculations to Ridge as secondary diagnostics;
they cannot rescue an HGB failure. Keep this study decision in a separate sidecar,
outside the generic descriptive comparison writer.

These are project-specific continuation thresholds, not significance tests.
Overlapping 20-session spreads are fractional signal outcomes, not daily
portfolio returns. A promising result justifies further development on qualified
broader data and economic checks. Otherwise retain the legacy HGB candidate and
diagnose the fixed bundle's coverage and selections without expanding this study.

### Implementation and verification plan

Add a small prepared-panel example around the existing benchmark and comparison
APIs. Test the two registered synthetic runs, configuration and schedule equality,
input preservation, new-output rule and failure retention before market training.
Keep acquisition, raw-price preparation and private history backup in the study
procedure. Do not add a new model, report schema or general experiment framework.

Before fitting, publish this declaration, verify the software, freeze the tested
revision/runtime and retain source/input hashes. Measure the actual training
interpreter rather than the Windows virtualenv launcher. Save before/after
registry snapshots, all native/common comparisons, fold selections and resources.
Independently check the decisions and history before publishing aggregate results.

The source review supports the representations and experiment discipline:

- [Qlib Alpha158](https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py)
  includes lagged-price and moving-average ratios. Our windows and ratio direction
  differ; these factors are not an Alpha158 replication.
- [French's daily momentum construction](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_mom_factor_daily.html)
  excludes recent returns from longer momentum. Our fixed-session stock
  characteristic is not its size-sorted portfolio factor.
- [Cawley and Talbot](https://www.jmlr.org/papers/v11/cawley10a.html) analyze selection
  bias in model comparison. [Scikit-learn's leakage guide](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)
  places learned feature selection and preprocessing inside training. Neither
  source validates the project's numerical continuation thresholds.

An accompanying consistency review corrected the benchmark guide's old sequence:
holdings, costs and negative controls belong in development, before freezing final
evaluation. The nine-workstream goal remains open after this bounded factor study.

### Feature-study implementation and input preparation

The prepared-panel example is implemented without changing the package's model,
factor or comparison APIs. The package remains 0.16.1; the new example requires
the source checkout. It normalizes all thirteen candidate columns once, checks
matching temporal preflight evidence, and runs both bundles through the existing
development registry and comparison writer. Study-specific continuation decisions
remain a separate analysis.

The initial test failed because the example did not exist. The completed 21-case
suite exercises real small synthetic fits, input/configuration identity, preflight
drift, history retention, failures and direct CLI execution. A comparison-output
failure after the first report preserves that report and both completed runs.
Standalone coverage is 87.60% of statements and 86.75% aggregate with branch
tracking. Subprocess CLI checks run separately from the parent coverage measure.
The full suite passed 1,636 tests in 304.64 seconds with 90.90% aggregate package
coverage and branch tracking. Lint/format, dependency consistency, vulnerability
audit and source/wheel builds passed; the editable project is excluded from the
dependency audit. Independent review approved the software and declaration.

Private no-training preparation took 721.73 seconds while synthetic verification
also ran. The 30,240-row enriched panel preserves every original column, dtype and
row exactly. Both preflights pass with identical temporal evidence and unchanged
final boundaries. Input and calendar hashes match the prior study; the base
benchmark configuration is byte-identical. The new panel SHA-256 is
`35a580f8fce7a191e9eed2a59e2989f5edd821a6988cf4e751fee8e81ef0787e`.

Across 27,720 pre-final rows, new-factor availability is 99.048% for `return_21s`,
97.179% for skipped-month momentum, and 98.752% for moving-average distance.
Zero-observed-input rows decrease from 474 under the legacy bundle to 264 under
the extended bundle. All rows remain present. This is input-coverage evidence,
not predictive performance; the study's native/common comparisons must retain
the distinction. Preparation leaves all five existing records/exposures unchanged.

Preparation evidence stays under `artifacts/feature-ablation-20260922-001/`.
No market-data feature comparison has trained yet. Before fitting, verify the
actual-interpreter resource observer and the declared-decision analysis, then
freeze the execution revision. Preserve these prepared inputs rather than
rebuilding them merely to repeat the preparation timing.
Independent read-only checks reproduced the data hashes, preflight boundaries and
five-record history equality without decoding final outcomes. The preparation
report records input hashes, not its start-time package/script identity. Bind the
reviewed package and preparation-script hashes in the execution declaration; keep
that post-preparation verification distinct from a contemporaneous capture.

### Full-lifecycle goal and completion checks

The user requested improvement across the entire project, including choices
of data, target, scale and model. The roadmap retains nine required workstreams
and six handoff deliverables. An independent scope review found that several
workstream completion rules were weaker than that overall contract. They now
require a qualified broader-data source, executed negative controls and period
sensitivity checks, fresh hosted Colab execution, and tested label-free scoring,
monitoring, retraining and rollback. These are requirements, not completed work.

The review also adds training/scoring feature parity and measured learning
curves to the plan. Cross-workstream reviews occur after input qualification,
after the first pilot, after each study and before environment/client handoff.
They can change the next hypothesis or priority; they do not change the declared
10/13-feature experiment, its thresholds or its finite training budget.

Google's [ML Test Score paper](https://research.google.com/pubs/archive/45742.pdf)
motivates checks of feature costs, training/scoring consistency, reproducibility,
staleness and recovery beyond a model-quality score. Qlib's
[workflow documentation](https://qlib.readthedocs.io/en/stable/component/workflow.html)
and [online-serving design](https://qlib.readthedocs.io/en/latest/component/online.html)
provide references for separating recorded experiments from later model updates
and prediction routines. We use these ideas for an offline research handoff;
they neither validate this project's signal nor require a live-trading service.

## Feature-bundle study results

The declared 10/13-feature study is complete. Adding the fixed price-factor
bundle improved HGB and Ridge on these development periods; both HGB contrasts
passed the published continuation rules. The study decision is `promising`.
An important simple-signal diagnostic changes the next priority: the individual
63-session moving-average-distance signal outperformed both learned combinations.
The next comparison must test what ML adds beyond that signal, alongside costs
and period stability, before spending more on model complexity.

### Execution and independent checks

The study used source revision `9a36c761f1eba059419ad3baa63513be2106cd8d`, package
0.16.1, Python 3.14.3, NumPy 2.5.3, pandas 3.0.5, SciPy 1.18.1 and scikit-learn
1.9.0. Source, prepared inputs, configuration and script hashes matched before
and after execution. The input and calendar identities are those recorded in
the declaration and preparation sections above.

The private Windows observer binds the actual interpreter's PID and creation
time before releasing it, samples its lifetime peak working set, and merges a
final self-measurement. Its tests cover budget stops, short allocations between
polls, failed/abrupt exits, ownership, PID reuse and cleanup. This follows the
process identity and Windows memory fields in the pinned
[psutil 7.2.2 documentation](https://raw.githubusercontent.com/giampaolo/psutil/release-7.2.2/docs/index.rst).
The private execution, observation and decision components passed 85 synthetic
tests in 14.81 seconds with 84% aggregate branch-tracked coverage. Independent
review approved each component before fitting. The six public CI jobs passed
on the frozen source revision.

The run took **745.47 seconds (12.42 minutes)**, with **734.59 CPU seconds** and
**585.92 MiB** peak working set in the actual training interpreter. The ten-input
arm took 283.59 seconds to fit; the thirteen-input arm took 393.25 seconds. The
total also includes imports, loading, preflight, comparisons and writes.
Preparation took a separate 721.73 seconds. These are Windows measurements,
not hosted Colab timings or a controlled end-to-end speedup comparison.
Resource limits were a sampled watchdog, not OS allocation caps; the final
self-sample excludes its own write and interpreter shutdown.

Completed trial/fold records support the declared 36 supervised fits. Exactly
two completed development runs were appended:

- Ten inputs: `e23036a0-b528-468a-9b54-e0470f713c8f`.
- Thirteen inputs: `5f7c9fce-9612-4f88-9812-ef74aca85303`.

Independent review confirmed all five earlier records and exposures are
unchanged, the after snapshot equals the live seven-record history, and there
are no final runs. Reserved final boundaries remain unchanged. A separate
software-equivalence check found the new ten-input development predictions
byte-identical to the earlier raw-target arm: 56,700 rows across all ten model
and baseline outputs, SHA-256
`2d82f7c7d103a3d8ac42cec378b41ad6c8e4154c8a2660abecbbdd98620f4d09`.
The earlier result was not substituted for the new registered arm.

### Declared comparisons

Each contrast has all 189 paired IC dates and 189 independently paired spread
dates, including 63 of each in every fold. Native and common daily metrics are
identical in this study; every evaluation cross-section contains 30 scored
stocks. Spreads below are mean top-minus-bottom **20-session outcomes**, expressed
as percentages, not compounded portfolio returns.

| Arm | Mean Rank IC | Mean spread (%) |
| --- | ---: | ---: |
| Legacy screened equal-rank | -0.03701 | -0.54459 |
| Ten-input HGB | 0.01506 | 0.35010 |
| Thirteen-input HGB | 0.06714 | 1.10735 |
| Ten-input Ridge | -0.05001 | -0.45717 |
| Thirteen-input Ridge | 0.06011 | 0.52902 |

| Paired contrast | Mean IC change | Spread change (percentage points) | Positive IC-change folds |
| --- | ---: | ---: | ---: |
| Thirteen-input HGB minus ten-input HGB | 0.05208 | 0.75725 | 2/3 |
| Thirteen-input HGB minus legacy equal-rank | 0.10415 | 1.65194 | 3/3 |
| Thirteen-input Ridge minus ten-input Ridge | 0.11012 | 0.98619 | 3/3 |
| Thirteen-input Ridge minus legacy equal-rank | 0.09711 | 1.07361 | 2/3 |

All four contrasts pass their descriptive continuation calculations; only the
two HGB contrasts determine the primary decision. The expanded HGB fold ICs are
0.11777, -0.04571 and 0.12935. Fold 2 improves substantially but still has negative
absolute IC and spread; fold 1 slightly underperforms the ten-input HGB. The
expanded Ridge also remains negative in fold 2. The gains are not uniform.

Legacy outer screens retain six features. Expanded screens retain nine in all
three folds: all three additions plus volatility, Sortino, maximum drawdown,
benchmark correlation, beta and historical VaR. Trailing return leaves the
selection and Sortino enters. The result therefore measures the entire bundle
through the declared screening procedure, not each new factor's isolated effect.

### Simple signals and the next decision

The existing individual-signal reports give `metric:ma_distance_63s` mean IC
**0.17806** and mean spread **2.47420%**, both above the expanded HGB. Its direction
is learned from each training fold. `metric:return_21s` also has higher IC
(0.07445), but lower spread (0.78526%). Both cover all 189 evaluation dates.
The training-selected `best_metric` is benchmark correlation, with negative
IC; it is distinct from the individual signal that looks best after evaluation.

These diagnostics leave the frozen `promising` decision intact, but do not
establish that combining factors beats all simple alternatives. Declare a
separate simple-factor/ML comparison before further evaluation. Retain the
expanded HGB and Ridge, the legacy baseline and the moving-average-distance
signal as research candidates; do not select a new winner from this table.
The immediate implementation priority is a tested development holdings/cost
ledger and negative controls. Broader data qualification and a fresh Colab run
remain required. The current retrospective cohort and repeatedly used periods
support a development lead, not independent confirmation or net-profit evidence.

Private evidence is under `artifacts/feature-ablation-20260922-001/`: `study/`
contains both runs and comparisons, `analysis/` the frozen decision and paired
tables, and `execution/` the identity declaration, resources and after snapshot.
Only aggregate findings are published; input prices, predictions and private
history remain outside Git.

### Broader-data and economic follow-up

A fresh source-code review confirms that Open Source Asset Pricing's public
[signal packaging](https://raw.githubusercontent.com/OpenSourceAP/CrossSection/master/Shipping/Code/1_pack_signals.r)
joins characteristics on `permno` and `yyyymm`, excluding several CRSP-derived
fields. Its separate
[CRSP retrieval](https://raw.githubusercontent.com/OpenSourceAP/CrossSection/master/Portfolios/Code/10_DownloadCRSP.R)
uses authenticated WRDS access for individual returns, delisting returns and
dated ticker/eligibility records. The public characteristics alone therefore
do not close the broader named-stock data gate. An authorized CRSP-equivalent
export would enable a separately declared monthly study; it is not an automatic
substitute for the existing 20-session experiment.

For economics, implement one development-only, fractional adjusted-price-unit
ledger. Fix target weights at decision time and execute at a declared later
close; value actual holdings daily and charge costs from cash. Preserve unused
allocations when new purchases lack prices, and stop with retained holdings
when a held asset cannot be valued. Use the same rules for each candidate and
baseline. Include entry/exit costs and drift-aware two-way traded notional.
Adjusted units are return exposures, not broker shares or a verified fill model.

This design draws on [Qlib's separation of strategy and daily accounting](https://qlib.readthedocs.io/en/latest/component/strategy.html),
[QuantConnect's fee-aware sizing and reduction-first orders](https://www.quantconnect.com/docs/v2/writing-algorithms/trading-and-orders/position-sizing),
and its [distinction between adjusted prices and dividend cash credits](https://www.quantconnect.com/docs/v2/writing-algorithms/securities/asset-classes/us-equity/corporate-actions).
Verify delayed entry, cash-funded fees, drift, ties, missing valuations and
dividend non-duplication on hand-worked fixtures before a separately declared
market-data cost comparison. No portfolio results have been computed yet.

## Development holdings ledger implementation plan

Implement the next economic check in three small parts: a score-only adapter,
a pure allocation/accounting kernel, and a chronological account. Reuse the
declared calendar and benchmark evidence; do not build another experiment
manager. This stage uses hand-worked and synthetic data only. A market-data
comparison still needs its own declaration, arms and cost grid.

The adapter must retain the native decision-time scoring universe, including
rows whose future outcomes are unavailable. Validate saved development scope,
fold timing and row assignments without reading targets or realized returns.
Fingerprint only the consumed score/price projections. Metadata consistency is
not authentication of the original fitting procedure or proof of omitted rows.

Use fixed top-k slots, with each slot worth 1/k of post-cost NAV. Split a cutoff
tie across all tied names; fewer than k finite scores leave unused slots as cash.
This makes differing score coverage visible instead of silently concentrating
the surviving names. Flat scores can produce equal allocations. Decisions are
after-close events, execution follows explicit calendar-session offsets, and
all entries must precede mandatory liquidation before the reserved final block.
Carry cash and holdings across fold boundaries.

Charge directional commission, one-way spread and slippage as additive cash
debits on mark-price traded notional. A quoted full spread must be converted to
the intended one-way cost by the caller. Do not also displace prices by the same
cost. Solve self-financing post-cost NAV in normalized units; verify cash,
holdings and cost reconciliation. A zero-cost counterfactual is a separate run.

Require failing tests before implementation for entry lag, cash-funded entry,
rotation and exit fees, drift, ties, missing scores/prices, and final boundaries.
Then add multi-fold integration, label-deletion and future-data perturbation
tests, independent review, full release checks and a runnable synthetic example.
Keep any partial account explicitly incomplete, with retained holdings, cash,
last valuation date and blocked date. No empirical net-return claim follows
from passing these software tests.

### Ledger implementation and review

Version 0.17.0 implements the score adapter, self-financing kernel and continuous
account, with a public `evaluate_long_only` API. The adapter supports learned
models and the individual, best-metric and equal-rank baselines. It verifies
configured fold lengths and single-metric feature counts as well as training
timing. The ledger records all cost components, cash, positions, target weights,
rejected purchases and score coverage; partial accounts retain unresolved state.

Independent review produced regressions for extreme numeric inputs, aggregate
return overflow, contradictory fold lengths/feature counts and missing saved
blocked-account state. Each reproduced failure was corrected before release.
The focused verification comprises 63 kernel cases, 39 adapter cases, 21 ledger
cases and five trained-workflow integration tests. The public API test also
checks the new exports and release version.

Release verification passed 1,765 tests in 305.65 seconds with 91.26% aggregate
coverage and branch tracking enabled. Ruff, dependency compatibility and the
vulnerability audit passed. The standard isolated build produced the source
archive and wheel; an import from the wheel confirmed the public ledger API.
The first full run exposed only a stale 0.16.1 version assertion, which was
updated for this release before repeating the complete suite.

The standalone synthetic command completed one registered two-fold Ridge
experiment and four continuous accounts in 4.51 seconds on this Windows host.
It used 64 invented sessions, eight stocks, three inputs and one training thread.
This wall time was measured while other verification work ran; it is not a
controlled performance comparison or hosted Colab measurement. Its private
outputs are under `artifacts/synthetic-portfolio-20260922-001/`.
The real seven-record history still matches the feature-study after snapshot
exactly, including exposures and the earlier interrupted record. No empirical
portfolio result or final outcome was evaluated in this implementation stage.

### Next falsification checks

Start with a declared, zero-fit saved-score alignment diagnostic, then a small
full-retraining control. Shuffling saved scores tests whether evaluation depends
on their security alignment; it does not test how the original scores were
learned. Use shared permutations across arms within matching score-availability
strata, retain all scheduled dates/masks, and report each replicate descriptively.
Fix arm identities, seeds and metrics before execution.

Full-pipeline target permutations instead require refitting screening, feature
orientation, preprocessing, tuning and models. This distinction follows the
[Ojala-Garriga permutation framework](https://www.jmlr.org/papers/volume11/ojala10a/ojala10a.pdf)
and [scikit-learn's implementation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.permutation_test_score.html).
Our adaptation would preserve date/maturity/missingness strata and paired
target/return fields. Independent daily shuffles break the serial structure of
overlapping financial outcomes, so treat them as destructive diagnostics, not a
calibrated financial permutation test. Three two-family replicas of the fixed
three-outer/two-inner schedule would require 54 fits; none have run yet.

Do not infer that a signal is leakage-free because a shuffled control loses
performance. Nor does another diagnostic make the repeatedly used development
period unseen: [Cawley-Talbot](https://www.jmlr.org/papers/v11/cawley10a.html)
documents overfitting to model-selection criteria. Known-null synthetic data,
future-availability rejection tests and a qualified broader evaluation remain
separate requirements. Freeze the actual control declaration before running it.

## Full-lifecycle goal acceptance review

The user requested a more detailed, all-round improvement goal. Keep the nine
workstreams and six handoff requirements in the [roadmap](roadmap.md); make their
acceptance evidence and remaining milestones concrete. This planning change
does not change an existing experiment, read new outcomes or reserve final data.

An independent review identified gaps in model-bundle lineage, measurable hosted
reproduction, offline monitoring outcomes and the distinction between authorized
data use and historical eligibility. The roadmap now specifies a seven-dimension
improvement scorecard, explicit scoring/operations/handoff milestones, runtime
and numerical reproduction checks, and expected behavior for five offline
monitoring and rollback scenarios. Model selection can retain a simple method;
adding another model family is not itself a completion requirement.

The supporting references were checked on September 22, 2026:

- [Qlib workflow documentation](https://qlib.readthedocs.io/en/latest/component/workflow.html)
  separates dataset, model and experiment records and records signal and
  portfolio analysis. Use that separation as a design reference, not its example
  parameters as evidence for our market or cost assumptions.
- [Scikit-learn model persistence](https://scikit-learn.org/stable/model_persistence.html)
  describes runtime compatibility, training/source metadata and the trust
  requirements of executable serialization. The scoring milestone must test
  the chosen artifact format and supported runtime before handoff.
- [The Colab FAQ](https://research.google.com/colaboratory/faq.html)
  describes variable hardware and runtime availability and recommends limiting
  small Drive operations. Measure the assigned environment and test local-work
  checkpoint recovery rather than promising a fixed hosted runtime.
- [Cawley and Talbot](https://www.jmlr.org/papers/v11/cawley10a.html)
  show how selection on noisy evaluation criteria can overfit. Retain bounded
  declared comparisons and outcome-exposure history as the project expands.

Next empirical work remains the declared simple-signal/ML economic comparison
and negative controls, with broader source qualification in parallel. No change
to the trained candidates or their development/final boundaries follows from
this plan revision. Source evidence, hosted execution, saved-model scoring and
offline operations remain open until their separate checks are completed.

## September 22, 2026 development cost-study declaration

Study `mendeley30-costs-20260922-001` asks whether the expanded HGB ranking earns
an economic advantage over the simple moving-average-distance signal under
fixed long-only accounting. Reuse saved feature-study predictions; perform zero
fits. This is further development on examined periods, including a simple signal
chosen after its earlier results were seen. It is not independent confirmation.

Use the original feature-study runs `e23036a0-b528-468a-9b54-e0470f713c8f`
(`legacy10_v1`) and `5f7c9fce-9612-4f88-9812-ef74aca85303`
(`legacy10_plus_price3_v1`). Evaluate these five fixed arms:

1. Original-ten `equal_weight_rank`.
2. Original-ten `hist_gradient_boosting`.
3. Expanded `hist_gradient_boosting` (primary candidate).
4. Expanded `ridge`.
5. Expanded `metric:ma_distance_63s` (primary reference).

Retain saved training-fold directions, score missingness and the full saved
decision-time universe. Do not orient signals again, select names by subsequent
returns or intersect populations using available outcomes. Read only score and
timing/coverage columns from predictions; no target or realized-return columns.

### Fixed accounts and observations

- Value all 189 independently declared XNYS sessions from 2015-11-02 through
  2016-08-02. Begin with 100,000 cash and no positions, use zero cash yield, and
  liquidate at the final close. No prices after that date enter calculation.
- Use ten equal top-k slots, split boundary ties, retain unused slots as cash,
  decide after close and execute one declared session later. Carry holdings and
  cash continuously across folds. A missing new-buy mark leaves cash; a missing
  held mark produces the existing incomplete-account result, without deletion
  or a substitute price.
- Primary rebalance spacing is 20 sessions, anchored at the first date without
  resetting at folds: November 2, December 1 and December 30, 2015; January 29,
  February 29, March 29, April 26, May 24, June 22 and July 21, 2016.
  The fixed sensitivity is every five sessions from the same anchor (38
  decisions). Keep the shortened final holding interval in both schedules.
- Simulate 0, 5 and 20 basis points total cost per side, charged once on traded
  mark notional, including initial purchases and terminal sales. These are
  uncalibrated scenarios, not estimated fees or fills. Store the total scenario
  charge in the ledger's commission field, with spread and slippage zero; this
  field therefore represents the assumed all-in charge in this study.
- Run all five arms at both frequencies and all three cost rates: exactly 30
  accounts. Keep separate zero-cost counterfactual accounts; do not treat the
  carried account's daily gross series as that counterfactual.
- Report terminal NAV/return, signed maximum drawdown, cumulative traded
  notional, cash costs, two-way turnover, score coverage, rejected purchases,
  cash allocation and maximum single-position weight. Show incomplete accounts
  with their retained state and no terminal-return estimate.

The primary contrast is expanded HGB minus MA63 at 20-session rebalancing and
5 bps per side. Record the other five frequency/cost contrasts in the same
fixed matrix, plus all arm summaries. No alternative case replaces the primary.
The market proxy supplies context only; it is not a tradable benchmark portfolio.

### Decision rule and execution evidence

Call the primary result worth further investigation only if both accounts
complete, every scheduled decision has at least 95% finite-score coverage, HGB's
net return is positive, its terminal-return advantage is at least 0.01, and its
signed maximum drawdown is no worse than MA63's minus 0.02. These are project
screening choices, not calibrated significance or investor suitability tests.
Treat differences within 1e-12 of numerical thresholds as equality: inclusive
conditions accept equality, while strictly positive conditions do not. This
arithmetic tolerance was fixed on synthetic boundary tests before outcomes.

Also require that no single period is necessary for a positive advantage.
Partition paired daily dollar P&L differences by the original three 63-session
valuation-date folds and divide by initial capital. Their sum must reconcile
to the terminal-return difference, and the sum outside each individual fold
must remain positive. These are calendar-period contributions of continuous
accounts, not independent fold portfolios or a refit attribution.

A blocked primary account or insufficient primary coverage makes the result
inconclusive. Otherwise, unmet continuation conditions mean the primary
criterion was not met. Report nonpositive return differences and drawdown
breaches in every sensitivity case as fragility; do not search for a replacement
setting. Completing this diagnostic does not complete negative controls or
qualify the retrospective cohort for model promotion.

Commit this declaration before reading portfolio outcomes. Freeze the private
comparison code, input-file identities, score projection, calendar and runtime
before execution, and verify them afterward. Use the version 0.17.0 ledger.
Retain all seven canonical research records and exposures unchanged through
read-only SQL comparisons against the completed feature-study snapshot. No new
model run or final exposure is created. Keep the final block starting 2016-09-01
untouched, and keep prices and row-level artifacts private.

The execution budget is 30 accounts, zero fits, 15 minutes and 2 GiB process peak
working set, checked between accounts. Record actual elapsed time and observed
peak memory; preserve partial outputs if a check stops execution. Test projected
loading, account persistence, contribution arithmetic and the decision rule on
synthetic fixtures first, then obtain independent review of the private runner.

This protocol uses [Qlib's separation of daily accounting and trading frequency](https://qlib.readthedocs.io/en/latest/component/strategy.html)
and [QuantConnect's fee-aware sizing](https://www.quantconnect.com/docs/v2/writing-algorithms/trading-and-orders/position-sizing)
as design references. The ledger uses fractional adjusted-price units and no
extra dividend cash credits, consistent with the
[adjusted-price distinction](https://www.quantconnect.com/docs/v2/writing-algorithms/securities/asset-classes/us-equity/corporate-actions).
None of those sources validates this archive's provenance, cost levels or fills.
