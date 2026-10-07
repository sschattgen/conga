# Implementation Plan: pipeline-reproducibility

## Overview

Tasks are ordered so every `conga/preprocess.py` signature/call-site change
(Components 1-5) lands before the `scripts/run_conga.py` CLI wiring that
passes `random_seed=` into those new parameters (Component 6), since the
parameters have to exist before anything can pass them. A checkpoint closes
out the `preprocess.py` changes before CLI wiring starts, confirming the new
trailing-default parameters are additive and break no existing caller
(design.md Component 7). A second checkpoint closes out the CLI wiring. The
end-to-end `Reproducibility_Test` (Requirements 6-7) comes last, since it is
the task that actually proves the wiring works, and depends on every prior
task being complete. A final checkpoint confirms zero regressions.

Each task that edits `conga/preprocess.py` or `scripts/run_conga.py` MUST
read the current file content directly before editing -- design.md's quoted
line numbers (e.g. "line ~1203") are approximate and will have drifted from
whatever line they were at when the design was written. Before applying an
"after" snippet, verify the "before" snippet in design.md actually matches
what's currently in the file at that call site. If it doesn't match closely
enough to be clearly the same call site (e.g. the call has been refactored,
reordered, or additional arguments added since design time), stop and report
the mismatch rather than guessing which call site is intended.

All test and pytest invocations use `mamba run -n conga-dev pytest tests/ -v`
per this project's development workflow -- never a bare `pytest` or `python`.

No task touches `scripts/run_conga.py.debug`, `.original`, `.fix`,
`run_conga_updated.py`, `run_conga_fixed.py`, `run_conga_with_backend.py`, or
`run_conga_with_validation.py`. These are stale, unexecuted backup copies
(design.md Component 7 / "Resolved Decisions") -- only `scripts/run_conga.py`
is the active entry point and is the only file in `scripts/` touched by this
spec.

No task fixes the pre-existing `except ImportError` fallback bug in
`cluster_and_tsne_and_umap` (Component 1) where the fallback branch calls
`sc.tl.leiden` a second time instead of `sc.tl.louvain` despite printing a
"ran louvain clustering" message. That bug is unrelated to seeding and is
explicitly out of scope per design.md; both leiden calls in that branch still
receive `random_state=random_seed` as part of this feature's mechanical
seed-wiring edit, but the wrong-function-called bug itself is left untouched.

## Tasks

- [x] 1. Seed `cluster_and_tsne_and_umap` (Component 1)
  - Read the current `conga/preprocess.py` content around
    `cluster_and_tsne_and_umap` before editing; confirm each "before" snippet
    below still matches the live code before applying its "after" version.
  - Add a `random_seed=util.DEFAULT_RANDOM_SEED` parameter as the last
    parameter in the function signature (Requirement 1.4).
  - Update all 6 internal stochastic call sites to pass
    `random_state=random_seed`:
    - `sc.tl.pca(adata, svd_solver='arpack', n_comps=n_gex_pcs)` (Requirement 1.1)
    - `sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)` (Requirement 2.1)
    - `sc.tl.umap(adata, min_dist=umap_min_dist, spread=umap_spread)`, the
      multi-component embedding (Requirement 2.2)
    - `sc.tl.umap(adata, n_components=1)`, the 1D embedding (Requirement 2.3)
    - `sc.tl.louvain(adata, resolution=resolution, key_added=cluster_key_added)` (Requirement 3.2)
    - `sc.tl.leiden(adata, resolution=resolution, key_added=cluster_key_added)`
      -- both occurrences, the primary branch and the "try leiden first"
      branch inside the `except ImportError` fallback (Requirement 3.1)
  - Do NOT fix the pre-existing leiden/louvain fallback mislabeling bug noted
    in design.md Component 1 (the `except ImportError` branch calling
    `sc.tl.leiden` again instead of `sc.tl.louvain`) -- leave that bug exactly
    as-is; only add `random_state=random_seed` to whichever calls are
    already there.
  - _Requirements: 1.1, 1.4, 2.1, 2.2, 2.3, 3.1, 3.2_

  - [ ]* 1.1 Write unit tests for Component 1's seed wiring
    - Source-level assertion test (following
      `tests/test_run_conga_cli.py`'s
      `test_call_site_source_wires_batch_integration_not_filter_and_scale`
      pattern): read `conga/preprocess.py` as text and assert
      `random_state=random_seed` appears at each of the 6 call sites above.
    - Runtime mock test (following `tests/test_batch_integration.py`'s
      `mock.patch('conga.preprocess.sc.tl.pca')`/`wraps=real_pca`
      convention): call `cluster_and_tsne_and_umap` with a non-default
      `random_seed` and confirm each mocked/wrapped call receives it as
      `random_state`.
    - _Requirements: 1.1, 1.4, 2.1, 2.2, 2.3, 3.1, 3.2_

- [x] 2. Seed `calc_tcrdist_nbrs_umap_clusters_cpp` (Component 2)
  - Read the current `conga/preprocess.py` content around
    `calc_tcrdist_nbrs_umap_clusters_cpp` before editing; confirm each
    "before" snippet matches the live code before applying its "after"
    version.
  - Add a `random_seed=util.DEFAULT_RANDOM_SEED` parameter as the last
    parameter in the function signature (Requirement 2.6).
  - Replace the unseeded `fake_pca = np.random.randn(adata.shape[0], 10)`
    placeholder-PCA call with a seeded
    `rng = np.random.default_rng(random_seed)` /
    `fake_pca = rng.standard_normal((adata.shape[0], 10))` (Requirement 2.5).
  - Update the remaining 5 internal stochastic call sites to pass
    `random_state=random_seed`:
    - `sc.tl.umap(adata, n_components=n_components_umap)`, multi-component
      embedding (Requirement 2.4)
    - `sc.tl.umap(adata, n_components=1)`, 1D embedding (Requirement 2.4)
    - `sc.tl.louvain(...)`, primary branch (Requirement 3.3)
    - `sc.tl.leiden(...)` -- both occurrences, the primary branch and the
      "try leiden first" branch inside the `except ImportError` fallback
      (Requirement 3.3)
    - `sc.tl.louvain(...)`, the fallback branch's (correctly-named) louvain
      call (Requirement 3.3)
  - _Requirements: 2.4, 2.5, 2.6, 3.3_

  - [ ]* 2.1 Write unit tests for Component 2's seed wiring
    - Source-level assertion test for all 6 call sites (the `np.random`
      replacement plus the 5 `random_state=` call sites), following the same
      pattern as Task 1.1.
    - Runtime mock test confirming a non-default `random_seed` reaches each
      mocked call, and that `fake_pca` is generated from a `Generator` seeded
      with the passed `random_seed` (e.g. compare against
      `np.random.default_rng(random_seed).standard_normal(...)` directly).
    - _Requirements: 2.4, 2.5, 2.6, 3.3_

- [x] 3. Seed `reduce_to_single_cell_per_clone` (Component 3)
  - Read the current `conga/preprocess.py` content around
    `reduce_to_single_cell_per_clone` before editing; confirm the "before"
    snippet matches the live code before applying the "after" version.
  - Add a `random_seed=util.DEFAULT_RANDOM_SEED` parameter as the last
    parameter in the function signature.
  - Update the representative-cell `sc.tl.pca(adata, svd_solver='arpack',
    n_comps=min(adata.shape[0]-1, n_pcs))` call to pass
    `random_state=random_seed` (Requirement 1.2).
  - _Requirements: 1.2_

  - [ ]* 3.1 Write unit tests for Component 3's seed wiring
    - Source-level assertion test confirming `random_state=random_seed`
      appears at this call site.
    - Runtime mock test confirming a non-default `random_seed` reaches the
      mocked `sc.tl.pca` call.
    - _Requirements: 1.2_

- [x] 4. Seed the diagnostic sampling in `calc_X_pca_gex_including_protein_features` (Component 4)
  - Read the current `conga/preprocess.py` content around
    `calc_X_pca_gex_including_protein_features` before editing; confirm the
    "before" snippet matches the live code before applying the "after"
    version.
  - Add a `random_seed=util.DEFAULT_RANDOM_SEED` parameter as the last
    parameter in the function signature.
  - Inside the `if compare_distance_distributions:` branch, replace the
    unseeded `inds = np.random.permutation(adata.shape[0])[:nrandom]` with a
    seeded `rng = np.random.default_rng(random_seed)` /
    `inds = rng.permutation(adata.shape[0])[:nrandom]` (Requirement 4.1).
  - Do NOT seed this function's own separate `sc.tl.pca(adata,
    svd_solver='arpack', n_comps=n_components_gex)` call -- that call is
    unrelated to the diagnostic branch and is intentionally left unseeded per
    design.md's Component 4 scope note: no acceptance criterion in
    requirements.md names it, and seeding it would be an undisclosed
    enlargement of Requirement 1's scope.
  - _Requirements: 4.1_

  - [ ]* 4.1 Write unit tests for Component 4's seed wiring
    - Source-level assertion test confirming the seeded `default_rng`
      replacement is present in the `compare_distance_distributions` branch.
    - Runtime mock test confirming that calling with
      `compare_distance_distributions=True` and a non-default `random_seed`
      produces the same `inds` as
      `np.random.default_rng(random_seed).permutation(...)[:nrandom]`
      computed directly.
    - _Requirements: 4.1_

- [x] 5. Seed both `KernelPCA` call sites (Component 5)
  - Read the current `conga/preprocess.py` content around both
    `make_tcrdist_kernel_pcs_file_from_clones_file` and
    `make_tcrdist_kernel_pcs_file_from_clones_file_V2` before editing;
    confirm each "before" snippet matches the live code before applying its
    "after" version.
  - Add a `random_seed=util.DEFAULT_RANDOM_SEED` parameter as the last
    parameter to both function signatures.
  - Update both `pca = KernelPCA(kernel='precomputed',
    n_components=n_components)` constructor calls to pass
    `random_state=random_seed` (Requirement 1.3).
  - _Requirements: 1.3_

  - [ ]* 5.1 Write unit tests for Component 5's seed wiring
    - Source-level assertion test confirming `random_state=random_seed`
      appears at both `KernelPCA(...)` constructor call sites.
    - Runtime mock test confirming a non-default `random_seed` reaches the
      mocked/wrapped `KernelPCA` constructor for both functions.
    - _Requirements: 1.3_

- [x] 6. Checkpoint: run full test suite after `conga/preprocess.py` changes
  - Run `mamba run -n conga-dev pytest tests/ -v`.
  - This is a sanity check, not expected to surface anything: design.md
    Component 7 already traced every caller of
    `cluster_and_tsne_and_umap`, `calc_tcrdist_nbrs_umap_clusters_cpp`, and
    `reduce_to_single_cell_per_clone` across `conga/`, `scripts/`, `tests/`,
    and `*.ipynb`, and confirmed no caller passes positional arguments past
    `adata`, so the new trailing-default `random_seed` parameters from Tasks
    1-5 are additive and should not break any existing caller. Running the
    full suite here empirically confirms that claim before CLI wiring begins.
  - All existing tests plus the new unit tests from Tasks 1.1-5.1 must pass.

- [x] 7. Wire `--random_seed` through `scripts/run_conga.py` (Component 6)
  - Depends on Tasks 1-3 (the `random_seed` parameter must already exist on
    `cluster_and_tsne_and_umap`, `calc_tcrdist_nbrs_umap_clusters_cpp`, and
    `reduce_to_single_cell_per_clone` before it can be passed from the CLI).
  - Read the current `scripts/run_conga.py` content around each call site
    below before editing; confirm each "before" snippet matches the live
    code before applying its "after" version.
  - Add `random_seed=args.random_seed` to all 5
    `conga.preprocess.cluster_and_tsne_and_umap(...)` call sites: the
    `--make_clone_plots` branch, the main post-`reduce_to_single_cell_per_clone`
    call, the post-`--exclude_mait_and_inkt_cells` subsetting call, the
    post-`--exclude_gex_clusters` subsetting call, and the post-`--subset_to_CD4`/
    `--subset_to_CD8` call (Requirement 5.1).
  - Add `random_seed=args.random_seed` to the single
    `conga.preprocess.calc_tcrdist_nbrs_umap_clusters_cpp(...)` call site
    (Requirement 5.2).
  - Add `random_seed=args.random_seed` to the single
    `conga.preprocess.reduce_to_single_cell_per_clone(...)` call site. No
    acceptance criterion names this call site by function name, but
    Requirement 1.2 names the `sc.tl.pca` call this function makes, and
    `args.random_seed` must be threaded in from the CLI for Task 3's new
    parameter to ever receive a non-default value.
  - Do not modify the existing `args.random_seed` resolution-to-default logic
    (lines ~347-348, `if args.random_seed is None: args.random_seed =
    util.DEFAULT_RANDOM_SEED`) -- it already satisfies Requirement 5.3
    unchanged.
  - Do not touch any of the stale sibling files
    (`run_conga.py.debug`/`.original`/`.fix`/`run_conga_updated.py`/
    `run_conga_fixed.py`/`run_conga_with_backend.py`/
    `run_conga_with_validation.py`) -- only `scripts/run_conga.py`.
  - _Requirements: 5.1, 5.2, 5.3_

  - [ ]* 7.1 Write unit tests for Component 6's CLI wiring
    - Source-level assertion test (following
      `tests/test_run_conga_cli.py`'s existing
      `test_call_site_source_wires_batch_integration_not_filter_and_scale`
      pattern) confirming `random_seed=args.random_seed` appears at all 5
      `cluster_and_tsne_and_umap` call sites, the 1
      `calc_tcrdist_nbrs_umap_clusters_cpp` call site, and the 1
      `reduce_to_single_cell_per_clone` call site in `scripts/run_conga.py`.
    - _Requirements: 5.1, 5.2, 5.3_

- [x] 8. Checkpoint: run full test suite after CLI wiring
  - Run `mamba run -n conga-dev pytest tests/ -v`.
  - All existing tests plus the new unit tests from Task 7.1 must pass before
    starting the end-to-end reproducibility test.

- [x] 9. Write the end-to-end `Reproducibility_Test` (Requirements 6-7)
  - Depends on Tasks 1-8 being complete -- this test is what proves the
    wiring from all prior tasks actually works end to end.
  - Create `tests/test_pipeline_reproducibility.py`, following design.md's
    Testing Strategy section precisely:
    - **Fixture**: a new, module-local, session-scoped fixture (not a reuse
      of `tests/fixtures/test_data_generator.py`'s `minimal_adata`/
      `minimal_clones`, per design.md's explanation of why those fixtures
      lack sufficient per-gene expression variance). Build a synthetic GEX
      count matrix via a `numpy.random.default_rng(42)`-seeded generator
      using per-gene lognormal means modulated by a small number of latent
      "programs", plus synthetic `va`/`ja`/`cdr3a`/`cdr3a_nucseq`/`vb`/`jb`/
      `cdr3b`/`cdr3b_nucseq` TCR columns directly in `adata.obs`. Use 60
      cells / 500 genes (confirmed sufficient in design.md's feasibility
      probe). Write to a session-scoped temporary `.h5ad` file.
    - **Subprocess invocation**: follow `tests/test_run_conga_cli.py`'s
      `_run_cli` helper exactly (`sys.executable`, `capture_output=True,
      text=True`, `cwd=REPO_ROOT`), with `timeout=300` (raised from that
      file's 120s since this test runs the full pipeline rather than only
      argument-validation paths).
    - **Two Pipeline_Run configurations**, both built from the same
      `base_args` (`--gex_data <fixture path>`, `--gex_data_type h5ad`,
      `--organism human`, `--min_clones 5`, `--min_cells_after_subsetting 5`,
      `--random_seed 42`, distinct `--outfile_prefix` per invocation):
      1. Default / vectorized-or-KernelPCA path (Requirement 7.1): no
         extra flag.
      2. Exact-TCRdist path (Requirement 7.2): `base_args + ['--no_kpca']`.
    - **Reproducibility assertion within each configuration**: run each
      configuration twice with identical `--random_seed 42` and distinct
      `--outfile_prefix` locations (Requirement 6.1), load both resulting
      `_final.h5ad` files, and assert exact equality on every
      `Comparable_Output` field design.md's Testing Strategy section
      actually determined via its manual probe:
      - `adata.obs` columns: `va`, `ja`, `cdr3a`, `cdr3a_nucseq`, `vb`, `jb`,
        `cdr3b`, `cdr3b_nucseq`, `n_genes`, `percent_mito`, `n_counts`,
        `clone_sizes`, `gex_variation`, `leiden_gex`, `clusters_gex`,
        `clusters_tcr`, `is_invariant`, `nndists_gex`, `nndists_tcr` --
        compared via `np.array_equal` on `.values`.
      - `adata.obsm` arrays: `X_gex_1d`, `X_gex_2d`, `X_pca_gex`, `X_tcr_1d`,
        `X_tcr_2d`, `X_umap_gex`, and (default-path configuration only)
        `X_vec_tcr` -- compared via `np.array_equal`.
      - `adata.obsp` matrices: `distances`, `connectivities` -- compared via
        `np.array_equal` on `.toarray()`.
      - `adata.uns` entries: `clusters_tcr_names` (`np.array_equal`),
        `conga_stats` (dict equality), `vec_tcr_config` (dict equality,
        default-path configuration only).
    - **Error handling** (Requirement 6.3): if either invocation in a
      configuration exits with a nonzero return code, fail with
      `pytest.fail` including the captured `stdout`/`stderr` of the failing
      invocation.
    - **Exclusion mechanism** (Requirement 6.5): structure the comparison as
      a single `EXCLUDED_FIELDS` set checked before each assertion, with a
      comment slot for the reason. Per design.md's design-time verification,
      no field was found to vary, so this set starts empty.
  - If, during implementation, any `Comparable_Output` field is found to
    still vary between the two same-seed runs despite the Task 1-8 wiring
    being correctly applied, add that field's name to `EXCLUDED_FIELDS` with
    an inline comment explaining the observed cause, and report this back
    explicitly rather than silently loosening the assertion to approximate
    equality (e.g. `np.allclose`) -- per Requirement 6.5, loosening the
    assertion without reporting the field and its cause is not acceptable.
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 7.1, 7.2_

- [x] 10. Final checkpoint: run full test suite
  - Run `mamba run -n conga-dev pytest tests/ -v`.
  - Confirm zero regressions across the full suite and that
    `tests/test_pipeline_reproducibility.py`'s new tests pass for both
    Pipeline_Run configurations.

## Notes

- Tasks 1-5 (all `conga/preprocess.py` signature/call-site changes) have no
  dependency on each other and could be done in any order; they are
  sequenced 1-5 only to match design.md's Component ordering.
- Task 7 (CLI wiring) depends on Tasks 1-3 specifically (not 4-5, since
  `calc_X_pca_gex_including_protein_features` and the two
  `make_tcrdist_kernel_pcs_file_from_clones_file[_V2]` functions are not
  called from the CLI paths this feature wires -- see design.md's Component
  4/5 scope notes and "Resolved Decisions").
- Task 9 (the end-to-end Reproducibility_Test) depends on all of Tasks 1-8,
  since it is the task that empirically proves the wiring is correct.
- No task fixes the pre-existing leiden/louvain fallback mislabeling bug in
  `cluster_and_tsne_and_umap` -- out of scope, noted in Task 1 and in the
  Overview above.
- No task touches `scripts/run_conga.py`'s seven stale sibling files.
- No task adds a third Pipeline_Run configuration for
  `--include_protein_features` or the `make_10x_clone_file_batch`
  combo-pipeline branches -- Requirement 7 scopes the end-to-end test to
  exactly two configurations; Components 4 and 5 are still fixed at the
  source level (Tasks 4 and 5) but are only covered by unit/call-site tests,
  per design.md's Testing Strategy and "Resolved Decisions" sections.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "2.1", "3.1", "4.1", "5.1"] },
    { "id": 1, "tasks": ["7.1"] },
    { "id": 2, "tasks": ["9"] }
  ]
}
```

Note on graph structure: Tasks 1, 2, 3, 4, 5 (the non-optional parent
implementation tasks) and Task 7 (CLI wiring parent) are not separately
listed as leaf nodes in the waves above because each one is a single
indivisible code-edit task with no further decimal-notation sub-numbering
of its own -- "1.1", "2.1", etc. refer to that parent task's optional test
sub-task. Per the workflow's dependency-graph rules, only sub-tasks with
decimal notation are included; here, each parent task (1-5, 7) and its sole
test sub-task occupy the same wave only because the test sub-task's own
code (writing tests against already-edited source) can be authored in the
same pass as the implementation -- both must land before Checkpoint 6/8 and
before Task 9, which is why wave 0 holds all five preprocess.py edit+test
pairs, wave 1 holds the CLI edit+test pair (which depends on wave 0's edits
1-3 existing first), and wave 2 holds the end-to-end test that depends on
everything before it.
