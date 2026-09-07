# Examples

## Colab development walkthrough

Version 0.8.0 provides [colab_public_demo.ipynb](colab_public_demo.ipynb), generated
by [build_colab_notebook.py](build_colab_notebook.py). It reruns the same fixed
public-archive development experiment on CPU. It is not a parameter search,
a fresh unseen-market test, or a final-evaluation notebook. Ridge's
[previous negative aggregate result](../docs/research-decisions.md#observed-development-result-2026-09-08)
is retained.

### Prepare the private handoff once

Use the existing canonical local registry, not an empty replacement. For this
research, include all three evidence aliases: `archive` for the downloaded
Mendeley archive, `run001` for the interrupted first attempt and `run002` for the
completed retry. Run the version 0.8.0 command locally with the actual existing
paths and a new private history directory:

~~~text
qmr seed-notebook --registry /path/to/canonical/research-registry.sqlite3 --history-dir /path/to/new/qmr-colab-history --evidence archive=/path/to/mendeley-v3-001 --evidence run001=/path/to/public-demo-001 --evidence run002=/path/to/public-demo-002
~~~

The Python equivalent is `seed_notebook_history(registry_path=...,
history_dir=..., evidence_dirs=...)` from `quant_metric_research.notebook_history`.
It snapshots the whole existing SQLite registry, including running/failed
records and recorded exposure, and copies the named evidence without changing
the originals. The seed is under `000000/`, with `registry.sqlite3`,
`evidence/archive/`, `evidence/run001/`, `evidence/run002/`, `start.json` and the
final `checkpoint.json`.

Upload the entire `qmr-colab-history` folder to a private Google Drive folder
using Drive's folder-upload interface. Keep every generation and its original
names together. No ZIP extraction or per-file reconstruction is required.
Do not upload this handoff to GitHub or share it publicly: it contains raw
prices, row-level results and research history. Retain the original local copy.
If related local research continues before the upload, prepare a new seed from
the updated canonical history. After handoff, continue one authoritative history
sequentially; do not independently branch research from an older registry copy.

### Run the notebook in order

1. Open the notebook and select a CPU runtime. Its bootstrap uses an immutable
   source revision and a dedicated virtual environment constrained by
   [colab-constraints.txt](colab-constraints.txt), rather than upgrading the
   notebook kernel's preinstalled libraries. Keep source and constraints fixed
   for the run.
2. Mount the private Drive folder and point the notebook at the complete seeded
   history. The notebook does not create a new blank registry. It reads the small
   archived input from `000000/evidence/archive` as ordinary files; the source
   checkout, environment, training and live SQLite stay on local VM storage.
3. Run `notebook-demo` with a new local work directory. The helper validates the
   full checkpoint chain, restores its latest registry locally, writes and
   rereads a new start marker, then calls the existing offline `public-demo`.
4. Inspect the development-only summary and the new checkpoint. Keep the whole
   updated history for later sessions. Clear saved notebook outputs before
   sharing the notebook.

The underlying command, invoked with the dedicated environment's `qmr`, is:

~~~text
qmr notebook-demo --archive-dir "/content/drive/MyDrive/qmr-colab-history/000000/evidence/archive" --work-dir /content/qmr-work-001 --history-dir "/content/drive/MyDrive/qmr-colab-history"
~~~

The defaults are history at `/content/drive/MyDrive/qmr-colab-history` and a
fresh session directory under `/content`. For local validation, environment
variables `QMR_HISTORY_DIR` and `QMR_SESSION_PARENT` change those locations.
`QMR_LOCAL_REPOSITORY` selects an existing local repository as the clone source.
A new clone checks out the pinned commit, excluding untracked or modified local
build files. Child Python uses isolated mode and omits inherited Python path/home
overrides. The bootstrap also updates its dedicated installer to the pinned,
audited version before installing research dependencies.
These overrides select storage/source locations, not research parameters.

The fixed cohort, ten metrics, 252-session lookback, 20-session target, Ridge
`alpha=1` and nested purged folds are unchanged from the
[public-archive declaration](#public-archive-development-demo).
There is no final-evaluation option. New runtime/source identities remain
recorded, and changing machines does not erase prior outcome exposure.

### Checkpoints and interruptions

Each new generation contains a start marker, a closed SQLite snapshot, copied
demo evidence and a checkpoint manifest. Live computation and SQLite are under
the new local work directory; persistent storage holds only ordinary files.
A completed checkpoint confirms its recorded files passed read-back checks,
not that a signal is useful.

Catchable failures preserve current registry state and partial evidence when
checkpointing succeeds, then re-raise the original error. An abrupt VM/process
loss, failed checkpoint or incomplete generation blocks the next automatic run.
Missing or changed evidence also fails validation. Do not delete the blocked
generation, restore an older checkpoint, or choose a blank registry to continue.
Preserve available local work and review the interrupted state.

Use one writer, sequentially. The notebook rejects known `/content/drive` and
`/content/gdrive` work
locations; arbitrary mounted/network filesystems cannot be identified reliably,
so the caller must keep the live work directory and SQLite source on genuinely
local storage. Read-back is not proof of remote server durability or a distributed
lock, and
the helper cannot detect deliberate deletion of an entire history tail.
See the [checkpoint contract](../docs/data-contract.md#notebook-history-checkpoints).

The notebook is supplied for later execution by the user. Local checks and
Linux CI are not evidence of an actual hosted Colab run. Colab can discard VM
files and its available resources can change; see its
[FAQ](https://research.google.com/colaboratory/faq.html).
No hosted Colab execution, final-test evaluation, verified historical provenance
or trading readiness is claimed.

## Public-archive development demo

Version 0.7.0 can fetch a small, pinned real-data research archive and run the
declared development experiment. No data account or API key is required for
these public archive endpoints. Use the installed repository virtual environment
(Windows `.venv\Scripts\python.exe`; Unix `.venv/bin/python`). With that
environment active:

~~~text
qmr fetch-public-sample --output-dir data/raw/mendeley-v3-001
qmr public-demo --archive-dir data/raw/mendeley-v3-001 --output-dir artifacts/public-demo-001 --registry artifacts/research-registry.sqlite3
~~~

Only `fetch-public-sample` accesses the network. It downloads `sp500-1216.csv`
and `FF3-0317.csv` from Chi Seng Pun's
[Mendeley Data V3 archive](https://data.mendeley.com/datasets/ndxfrshm74/3),
plus the dataset record and file inventory. It validates the publisher's CC BY
4.0 declaration, exact file identities, byte counts and SHA-256 hashes. The
archive directory contains `raw/`, `dataset_record.json`, `file_inventory.json`
and the final success marker `archive_manifest.json`.

`public-demo` rechecks the local evidence and performs these fixed steps:

1. Select the first 30 normalized stock-column labels alphabetically, retaining
   sparse members, over 2012-01-03 through 2016-12-30.
2. Build `FF_MARKET_PROXY` from `(Mkt-RF + RF) / 100`: first included session
   base 100, then compound subsequent daily returns. This constructed research
   index is neither an ETF nor the official S&P 500 total-return index.
3. Audit inputs and calculate the ten trailing metrics on daily decision dates
   from 2013-01-02 through 2016-12-30, with a 252-session lookback, minimum
   126 observations, one-session entry lag and 20-session excess-return target.
4. Save no-training preflight, then run registered Ridge (`alpha=1`) development
   against individual metrics, the best training-only metric and equal-weight
   oriented ranks. Three outer 63-date windows contain two inner 42-date windows;
   fitting and screening remain fold-local and labels are purged at boundaries.

The [frozen declaration](../docs/research-decisions.md#september-2026-public-archive-development-declaration)
records the full settings. The final 63 eligible decision dates remain reserved
without final-outcome evaluation; this command has no final-evaluation option.
The saved panel still contains forward outcomes, and preflight inspects their
availability, so the reserved period is not a physically sealed holdout.

On completion, `public_demo_report.json` links the development summary and run
ID. `quality_profile.json` reports symbol/year missingness and large daily moves;
`missing_prices.csv` retains explicit null observations. Under `research/` are
normalized inputs and configs, `input_audit.json`, `metric_panel.parquet`,
`preflight.json`, the `development/` benchmark bundle, `registry_record.json`
and `development_report.json`. Inspect history with the same registry:

~~~text
qmr experiments --registry artifacts/research-registry.sqlite3
~~~

Both output directories must be new. After a failure, retain partial evidence
and any failure record; a missing completion marker means the bundle is
incomplete. Retry into a new output directory using the same durable registry,
kept outside individual run directories. Do not recreate the registry to reset
recorded outcome exposure.

Attribution: Pun, Chi Seng (2018), *Low- and High-Dimensional Asset Prices Data*,
Mendeley Data, V3, DOI `10.17632/ndxfrshm74.3`. The publisher declares
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and names Yahoo Finance
and Ken French as upstream sources. That declaration is not an independent
guarantee of third-party rights. The January 2017 constituent snapshot introduces
survivorship bias; historical membership, identifiers, adjustments and delistings
remain unverified. Keep raw and row-level derived data in the ignored locations
above. A completed run is development evidence, with `acceptance_status` still
`not_evaluated` and `stage4_eligible: false`. It does not resolve the separate ASX
intake's source-access or historical-evidence gaps.

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
