# Project agent guide

## Scope and planning

- This is a standalone stock-ranking research repository. Do not modify the
  separate Financial-Investment-Tool repository or global agent configuration.
- Before a stage or material methodological change, read README.md,
  docs/roadmap.md, docs/data-contract.md, and the relevant benchmark documentation.
  Check the current code, tests, client requirements, and known limitations.
- Write a bounded plan before editing. Recheck it at stage boundaries and when
  findings contradict assumptions. Verify methodological or API claims against
  primary papers, official documentation, or upstream source. Record material
  choices and sources in the existing research-decisions documentation.
- Prefer a falsifiable baseline and a focused correction to extra complexity.
  Do not silently expand the universe, target, dataset, costs, or publication scope.

## Research invariants

- Use point-in-time membership, stable security identifiers, and availability
  timestamps. Retain missing observations with visible coverage diagnostics.
- Split by decision date, never randomly across stock-date rows. Training labels
  must mature strictly before each evaluation window starts. Apply the same
  purging rules to inner tuning and outer development folds.
- Fit screening, feature orientation, imputation, scaling, PCA, hyperparameters,
  and supervised models using the appropriate training partition only.
- Compute a day's ranking over the scoring universe available at decision time.
  Do not let future target or realized-return missingness alter another stock's
  score. Apply outcome-availability masks only when evaluating outcomes.
- Use development results for model selection. Final holdout evaluation must be
  explicit and follow a frozen experiment plan; opening it repeatedly invalidates
  its independence. Use the same durable experiment registry for related work.
  Validate frozen development identity before rerunning development, then commit
  the final outcome-date reservation before fitting or evaluating final models.
  Failed/interrupted reservations stay consumed; do not delete or replace the
  registry to obtain a fresh result. This guard does not detect untracked research.
- Run the no-training preflight before a declared experiment. It checks the
  schedule and data contract, not predictive usefulness or provider provenance.
  Keep source, panel, configuration, and recorded runtime versions unchanged
  between registered development and final evaluation.
- Keep native coverage and common-sample comparisons visible. Report label
  overlap and uncertainty assumptions rather than treating observations as IID.
- Synthetic fixtures prove software behavior, not alpha, provider provenance, or
  tradability. Do not promote a model to portfolio use without the documented
  real-data and execution gates.

## Implementation and verification

- Use the repository's virtual environment, not an unrelated global interpreter.
  On Windows use .venv\Scripts\python.exe; on Unix use .venv/bin/python.
  Bootstrap a fresh environment from pyproject.toml with pip install ".[dev]".
- For behavioral changes, first add a regression test that fails for the right
  reason, implement the smallest correction, then refactor. Include boundary,
  missingness, temporal-leakage, and CLI/artifact cases where applicable.
- Preserve caller-owned inputs; make explicit copies before transformations.
  Validate external inputs and fail clearly rather than silently repairing data.
- Run focused tests during development. Before release run python -m ruff check .,
  python -m pytest, python -m pip check, a dependency vulnerability audit, and a
  package build in the intended environment. Install the build/audit tools
  separately if needed; they are not runtime dependencies.
- The pytest configuration enforces at least 80% aggregate coverage with branch
  tracking enabled. Report the actual measure and result, not a branch-only
  coverage claim. Explain unavailable checks instead of claiming they passed.
- Review the complete diff and generated artifact contracts before publication.
  Never commit credentials, private data, local environment files, or provider
  exports whose redistribution rights have not been confirmed.

## Shared-worktree coordination

- Assign every worker explicit file ownership. Workers are not alone: preserve
  other changes, communicate overlap, and never revert another worker's edits.
- Only the coordinating agent handles branches, commits, pushes, and merges.
  Workers must not perform these actions. Branch names must not contain codex.
- Pause all file and Git mutations before the final full-suite run. Record the
  tested revision/diff, and reverify affected checks if it changes afterward.
- Commit and push only within the user's authorized scope, after review and
  verification. Check the remote revision and CI result before claiming release.
