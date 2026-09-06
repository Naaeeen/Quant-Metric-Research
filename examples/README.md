# Offline synthetic workflow

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
