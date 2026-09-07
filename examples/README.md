# Offline examples

## Offline Yahoo-format intake

Version 0.6.0 imports files already acquired with appropriate source rights.
There is no bundled real-price dataset, automatic downloader, new yfinance
dependency, model training or final-test evaluation in this intake workflow.
The [ASX provider decision](../docs/research-decisions.md#september-2026-provider-decision-prepare-an-offline-asx-demonstration)
records the selected demonstration cohort, its VAS adjusted ETF proxy and the
remaining access and historical-evidence gaps.

### Declared intake-only demonstration scope

The cohort was selected on 2026-09-07, before inspecting its price outcomes.
For a later authorized export, request daily observations from **2015-01-01
through 2026-02-28 inclusive**. The extra history supports trailing features and
the extra tail supports forward-label availability checks; actual coverage must
still be audited.

Proposed decision dates run from **2016-01-01 through 2025-12-31**: within that
range, use the last supplied VAS benchmark session in each ISO week. This is a
demonstration schedule, not an independently verified ASX calendar. Supply the
resulting daily dates in `as_of_dates.csv`; the importer does not generate them.
Use the README's 252-session lookback, 126-observation minimum, 20-session target,
one-session entry lag, 252-session annualization and zero risk-free-rate config.

If using a fixed-cohort membership file for this demonstration, retain all eight
stocks across the requested interval and identify its source as a
`present-selected-cohort-demo-2026-09-07`. These intervals define the demonstration
selection, **not historical index membership or listing availability**. Missing
histories must remain visible. VAS is the benchmark, not a ninth ranked stock.
This release declares only intake and coverage checks: no training split,
acceptance threshold or final holdout is opened by this specification.

### Import supplied files

Create `data/raw/exports.json` to map each ticker to one local export. For the
declared demonstration cohort, a mapping using CSV files is:

~~~json
{
  "BHP.AX": "BHP.AX.csv",
  "CBA.AX": "CBA.AX.csv",
  "CSL.AX": "CSL.AX.csv",
  "NAB.AX": "NAB.AX.csv",
  "RIO.AX": "RIO.AX.csv",
  "TLS.AX": "TLS.AX.csv",
  "WES.AX": "WES.AX.csv",
  "WOW.AX": "WOW.AX.csv",
  "VAS.AX": "VAS.AX.csv"
}
~~~

The paths resolve relative to `exports.json`, so this example expects the files
in `data/raw/`. Local `.csv.gz`, `.parquet` and `.pq` exports are also accepted.
Each file must contain a flat, single-symbol table with explicit `Date` and
`Adj Close` columns. A date index alone or a yfinance MultiIndex table is not
accepted. Dates must be timezone-naive midnight dates, with no duplicates.
`Close` is never a fallback and missing prices are never imputed.

CSV imports use pandas' default missing-value tokens, including empty fields,
`NaN`, `N/A` and `NULL`. Those missing adjusted-close observations remain visible
in the intake evidence; other malformed non-missing price text fails validation.
See [pandas' documented missing-value behavior](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html).
The `Adj Close` name alone does not verify the provider's adjustment method.

Supply `memberships.csv`, `as_of_dates.csv` and `panel_config.json` separately
under [the input contract](../docs/data-contract.md). Memberships are supplied
evidence, not reconstructed from this export mapping. Keep members with absent
or unusable price coverage in that evidence so the audit can expose their gaps.
The configured benchmark requires usable price observations.

From the installed repository, run:

~~~text
qmr import-yahoo --exports data/raw/exports.json --memberships data/raw/memberships.csv --as-of-dates data/raw/as_of_dates.csv --config data/raw/panel_config.json --output-dir artifacts/yahoo-intake-001
~~~

The output contains original-byte snapshots with SHA-256 hashes, normalized
prices, memberships, decision dates and configuration, `missing_prices.csv`,
`input_audit.json`, and the final success marker `intake_manifest.json`.
Review the missingness and coverage evidence even when the command succeeds.
All provider-provenance checks and `stage4_eligible` remain false; `imported_at`
records the local import time, not the provider's acquisition date. This audit
inspects supplied future price presence and is not a sealed holdout.

An existing output directory is refused without replacement. If processing
fails after directory creation, partial artifacts and `intake_failure.json`
remain for diagnosis. A directory without `intake_manifest.json` is incomplete;
use a fresh name such as `artifacts/yahoo-intake-002` after resolving the error.
Keep source exports and intake evidence private unless their sharing rights
have been established. Authorized data and independent lifecycle checks remain
necessary before progressing to an empirical experiment.

## Offline synthetic workflow

This example checks the Stage 3 experiment workflow through the public CLI.
It downloads nothing, buys no data, and uses invented observations. A successful
run demonstrates working software and a local holdout-reuse guard, not empirical
alpha, verified data provenance, or a tradable strategy.
It starts from an invented metric panel; it does not test raw-price ingestion
or the Stage 1 calculation of historical metrics.

After installing the repository into your active Python environment:

~~~text
python examples/synthetic_workflow.py --output-dir artifacts/synthetic-demo-001
~~~

Use the repository virtual environment (on Windows,
`.venv\Scripts\python.exe`; on Unix, `.venv/bin/python`). The output directory
must not exist, including after a failed earlier attempt. The example never
deletes or overwrites prior output; use a new name to rerun the entire synthetic
demo. New synthetic demos are separate software checks, not new empirical tests.

The example generates a fixed-seed panel with 36 business dates, eight invented
stock identifiers, two features, two-session labels, and an immature two-date
tail. The target deliberately depends on the features. This relationship is
manufactured, not discovered in market prices. One Ridge candidate and small
nested walk-forward windows keep the smoke check fast; these windows are not
recommended statistical research settings.

The script sequentially runs:

1. `qmr preflight`: validate structural feasibility without model training.
2. `qmr benchmark`: register a stated synthetic hypothesis and run development.
3. `qmr benchmark --evaluate-lockbox`: reference the completed development run
   and reserve the final test before opening it.
4. The same final command into a new destination: confirm reuse is refused.
5. `qmr experiments`: export the two completed runs for inspection.

It invokes these commands with `python -m quant_metric_research.cli`, using the
same interpreter as the example. The directory contains inputs, `preflight.json`,
`development/`, `final/`, the SQLite registry, `experiments.json`, and
`demo-report.json`. Standard output is the JSON demo report. A refusal must leave
no `refused-attempt/` result bundle. Both benchmark runs must retain
`stage4_eligible: false`; even a passing model statistic cannot verify provenance.

The registry helps prevent accidental reuse within this local registry. It is
not a physically sealed holdout or protection against intentional bypass by
deleting/copying a registry or inspecting source data. A real-data experiment
still needs audited provider policies and an independently frozen research plan.
