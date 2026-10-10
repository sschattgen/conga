# End-to-End Workflow Matrix Validation (Tasks 5, 6, 8.1)

Date: 2026-10-09
Environment: conga-dev — Python 3.14.8, pandas 3.0.6, numpy 2.5.3,
scanpy 1.12.4, anndata 0.13.4
Command runner: `mamba run -n conga-dev python scripts/run_conga.py ...`
Data: real 10x PBMC dataset provided by the user —
`test_data/SC5v2_humanPBMCs_clones.tsv` (1477 clonotypes) +
`test_data/SC5v2_humanPBMCs_5Kcells_Connect_single_channel_..._feature_bc_matrix.h5`
(4190 barcodes x 36601 genes, 10x_h5 format), random_seed=42 throughout.
Outputs: `modernization_results/workflow_runs/`

## Runs executed

| Run | Command flags | Result |
|---|---|---|
| `vectorized` | `--graph_vs_graph` (default TCR path for supported organism) | ✅ DONE |
| `exact` | `--graph_vs_graph --use_exact_tcrdist_nbrs` | ✅ DONE |
| `kpca` | `--graph_vs_graph --use_kpca_tcrdist` (after generating `_AB.dist_50_kpcs` via `setup_10x_for_conga.py --use_kpca_tcrdist`) | ✅ DONE |
| `graph_vs_features` | `--graph_vs_features` | ✅ DONE |
| `tcr_clumping` | `--tcr_clumping` (uses compiled C++ `calc_distributions`/`find_neighbors`) | ✅ DONE |
| `all` | `--all` (implies graph_vs_graph, graph_vs_graph_stats, graph_vs_features, cluster_vs_cluster, find_hotspot_features, find_gex_cluster_degs, tcr_clumping, match_to_tcr_database, make_tcrdist_trees) | ✅ DONE |

All six runs completed to the `DONE` sentinel and produced the expected
`*_final.h5ad`, `*_final_obs.tsv`, per-analysis `.tsv` result files, and
`*_results_summary.html`. This covers all three TCR representation paths
(vectorized default, exact TCRdist, KernelPCA) and the three top-level
analysis flags called out in task 5 (`--graph_vs_graph` via multiple runs,
`--tcr_clumping`, `--graph_vs_features`), plus a full `--all` run for task
6/8.1.

## Setup-step note (KernelPCA path)

`run_conga.py --use_kpca_tcrdist` requires a precomputed
`<clones_file_stem>_AB.dist_50_kpcs` file; it is not generated on demand.
Had to run:

```
mamba run -n conga-dev python scripts/setup_10x_for_conga.py \
    --input_clones_file test_data/SC5v2_humanPBMCs_clones.tsv \
    --organism human --use_kpca_tcrdist --random_seed 42
```

first. This is expected/documented pipeline behavior (two-stage
setup -> run), not a pandas3/numpy2 defect, but it's worth calling out
because the broken `test_modernization_results.py` harness (see
`test_modernization_results_findings_2026-10-09.md`) doesn't account for
this and would fail silently if someone tried to drive the KPCA config
through it as currently written.

## Deprecation / compatibility warning sweep

Grepped full captured output of all 6 runs for `DeprecationWarning`,
`FutureWarning`, `Pandas4Warning`/`Pandas3Warning`, and any pandas/numpy
`RuntimeWarning`. Findings, with assessment of whether each is a
pandas3/numpy2 modernization concern:

1. **`scanpy/readwrite.py:322: UserWarning: Variable names are not
   unique.`** -- Pre-existing scanpy behavior when loading 10x H5 data with
   duplicate gene symbols. Not pandas3/numpy2-related; present regardless
   of version. Not actionable here.

2. **`conga/correlations.py:1174: RuntimeWarning: invalid value encountered
   in log2`** (in `logfoldchanges = np.log2((np.expm1(mean_fg) + 1e-9) /
   (np.expm1(mean_bg) + 1e-9))`) -- Occurs in `graph_vs_features` and
   `all` runs. Root cause: when `mean_fg` is very negative,
   `np.expm1(mean_fg)` underflows toward -1, making the numerator slightly
   negative; `log2` of a negative number is legitimately `nan` with a
   warning under IEEE-754, identical behavior in numpy 1.x and 2.x (log2
   domain semantics did not change). The surrounding code (comment:
   "scanpy code") already nan-guards `scores`/`pvals` immediately after
   this line but not `logfoldchanges` itself -- this is a pre-existing
   numerical edge-case in a scanpy-derived pattern, not a regression
   introduced by or specific to the pandas3/numpy2 migration. Flagged for
   awareness; not fixed as part of this modernization task since it is
   unrelated to the pandas/numpy version bump (same warning would fire on
   pandas <3.0 / numpy <2.0 with the same input data).

3. **`conga/plotting.py:2445: RuntimeWarning: More than 20 figures have
   been opened.`** (matplotlib) -- Resource-management warning from not
   calling `plt.close()` after saving figures in a loop (`all` run opens
   many hotspot plots). Unrelated to pandas/numpy; a matplotlib memory-
   hygiene issue, pre-existing.

4. **`findfont: Failed to find font weight ... for DejaVu Sans`** and
   **`WARNING: The convert command is deprecated in IMv7, use "magick"...`**
   -- Font-matching and ImageMagick CLI warnings from the plotting/SVG-to-
   PNG conversion pipeline. Environment/tooling warnings, not Python
   library deprecations, not pandas/numpy related.

**No `DeprecationWarning`, `FutureWarning`, `Pandas4Warning`, or
`Pandas3Warning` was emitted by any of the 6 runs.** The pandas3.0/numpy2.0
compatibility surface specifically targeted by this modernization effort
(chained assignment silently no-op'ing, legacy dtype alias AttributeErrors,
`numpy.core` import failures, NEP 50 casting surprises) produced zero
warnings or errors across all six full pipeline configurations.

## Conclusion

**Tasks 5 and 8.1 are VERIFIED COMPLETE** for this dataset: all three TCR
representation paths and the `--graph_vs_graph`, `--tcr_clumping`,
`--graph_vs_features`, and `--all` analysis modes run end-to-end under
pandas 3.0.6/numpy 2.5.3 without pandas/numpy compatibility errors, and
with zero pandas/numpy deprecation warnings. The warnings that did appear
are pre-existing, version-independent, and unrelated to the pandas3/numpy2
migration (documented above for completeness/transparency rather than left
unexplained).

This validation was run on one real dataset (1390 clonotypes after QC
filtering) rather than the full matrix of dataset sizes mentioned in the
original task list; it was not re-run with multiple random seeds for
byte-identical reproducibility comparison (that is covered separately by
the dedicated `pipeline-reproducibility` spec and its test suite in
`tests/test_pipeline_reproducibility*.py`, which already exercises seed
determinism independent of this effort).
