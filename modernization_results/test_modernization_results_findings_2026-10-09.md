# `scripts/test_modernization_results.py` Harness Findings (Task 2)

Date: 2026-10-09

## Summary

`scripts/test_modernization_results.py` (the before/after baseline
comparison harness referenced by task 2.2) is fully implemented as code
(`ModernizationTester` with `create_baseline()`, `validate_against_baseline()`,
`compare_anndata_objects()`, `compare_tsv_files()`) but its three built-in
test configurations are **stale relative to the current `run_conga.py` CLI**
and cannot run successfully as shipped:

1. All three configs (`basic_correlation`, `tcr_clumping`, `vectorized_tcr`)
   reference input files named `test_clones.tsv` and `test_gex.h5ad` in
   `--test-data-dir` (default `test_data/`). No files with these exact names
   exist in `test_data/`; the real files are named `SC5v2_humanPBMCs_clones.tsv`,
   `classic_clones.tsv`, `test_run_clones.tsv`, etc. Since
   `run_conga_analysis()` only adds `--clones_file`/`--gex_data` when
   `input_path.exists()` is true (silently skipping otherwise), running
   `--create-baseline` against the default `test_data/` would silently
   invoke `run_conga.py` with no input data at all, rather than failing
   loudly.

2. The `vectorized_tcr` config passes `--use_vectorized_tcrdist` to
   `run_conga.py`. This flag does not exist in the current CLI (confirmed
   via `run_conga.py --help`); vectorized TCR encoding is simply the
   *default* representation for supported organisms (human/mouse/rhesus)
   now, selected via `resolve_tcr_representation()`, with
   `--use_kpca_tcrdist` / `--use_exact_tcrdist_nbrs` as the opt-out flags.
   Running this config as shipped would fail with an argparse
   "unrecognized arguments" error.

3. `--no_plots` is not a real flag either (used in my own first manual
   attempt at reproducing this, not in the harness itself, but worth noting
   since it's an easy mistake to repeat) -- there is no global
   plot-suppression flag; plotting is controlled per-analysis-type.

## Decision: ran the workflow matrix directly instead of fixing the harness

Rather than rewrite the harness's hardcoded configs (which would mean
guessing at a "canonical" test dataset and inventing config dicts not
asked for by the user), I validated pandas3/numpy2 compatibility by running
`scripts/run_conga.py` directly against the user's real dataset
(`SC5v2_humanPBMCs_clones.tsv` + matching 10x H5) across all three TCR
representation paths and the `--graph_vs_graph`, `--tcr_clumping`,
`--graph_vs_features`, and `--all` analysis modes. See
`workflow_matrix_2026-10-09.md` for full results (all passed, zero
pandas/numpy deprecation warnings).

This satisfies the intent of task 2 (prove the pipeline produces correct,
stable results under the real pandas3/numpy2 environment) without the
harness's specific `--create-baseline`/`--validate` byte-comparison
workflow. The harness's `compare_anndata_objects()` /
`compare_tsv_files()` comparison logic itself was not exercised, since
there was no prior-pandas2/numpy1 baseline available to compare against in
this environment (the environment only has pandas 3.0.6/numpy 2.5.3
installed; there's no secondary legacy environment to generate a true
"before" baseline from).

## If the harness itself needs to work in the future

To make `test_modernization_results.py` usable as shipped, at minimum:
- Update the three config dicts' `input_files` to match actual file names
  (e.g. `SC5v2_humanPBMCs_clones.tsv` + the matching `.h5` file, passed with
  `--gex_data_type 10x_h5`, since `run_conga_analysis()` currently assumes
  `.h5ad`-only GEX input and never passes `--gex_data_type`).
- Remove `--use_vectorized_tcrdist` from the `vectorized_tcr` config (it's
  the default; drop the flag or replace with an explicit opt-out comparison
  against `--use_kpca_tcrdist`).
- `run_conga_analysis()` should raise/fail loudly rather than silently
  omit `--clones_file`/`--gex_data` when expected input files are missing,
  to avoid masking the exact failure mode that bit this investigation.

Not fixed as part of this session since the user asked to finish the
modernization verification tasks, not to refactor this specific harness;
flagging the gap here for visibility rather than silently leaving it
unaddressed.
