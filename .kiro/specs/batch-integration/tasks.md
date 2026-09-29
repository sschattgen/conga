# Implementation Plan: Batch Integration

## Overview

This implementation adds a `batch_integration()` entry point to `conga/preprocess.py` that runs batch-aware HVG selection followed by a Harmony- or scVI-corrected GEX PCA representation, driven by a single shared `batch_key`. It also confirms and closes small behavior gaps in the existing `force_variable_genes` mechanism (the Fixed_HVG_Pathway) so that automatic HVG detection and batch integration are both provably skipped when a caller supplies a fixed gene panel, and adds a `Counts_Layer` capture step that neither pathway needs today but that `scvi` integration cannot run without.

The work has two independent foundational pieces (shared constants in `conga/util.py`, and the `Counts_Layer` capture patch to `filter_normalize_and_hvg`) that unlock the rest. The Fixed_HVG_Pathway fix and the `Counts_Layer` capture both modify `filter_normalize_and_hvg`, but in genuinely non-overlapping regions of the function (the `Counts_Layer` capture sits between the antibody-feature-removal block and `sc.pp.normalize_total`/`log1p`, roughly lines 546-562; the Fixed_HVG_Pathway guard sits in the separate HVG-selection block roughly lines 565-597) — they are listed as parallel sub-tasks for planning purposes, but because both edit the same file, execute them as sequential commits in practice to avoid merge conflicts. `batch_integration()` itself splits along its two `Integration_Method` branches: the `harmony` path only needs the shared constants, while the `scvi` path additionally needs the `Counts_Layer` to be reachable, so Harmony and scVI implementation are split into separate sub-tasks mirroring that asymmetric dependency, the same way the vectorized-tcrdist plan split its TCR-side and GEX-side tracks.

Per this design's explicit Testing Strategy section, `batch_integration` is pipeline wiring around two third-party libraries, not a pure function with a generative input space — the design omits a Correctness Properties section and specifies unit and integration tests only, with no property-based testing. Rather than deferring all of the design's itemized unit/integration tests into one late-running test phase, each test is placed as a regular (non-optional) sub-task inside the implementation phase that produces the behavior it checks. This keeps a test close to the code it exercises (catching regressions the moment the relevant function lands, the same rationale the vectorized-tcrdist plan used for placing property tests next to implementation) and lets the file-level test tasks for `_validate_batch_key`, the Harmony path, the scVI path, and the CLI run in parallel with each other once their respective implementation sub-tasks land, instead of serializing all testing behind every implementation phase. Cross-cutting checks that only make sense once multiple pieces exist together (the `.h5ad` round-trip, the `cluster_and_tsne_and_umap` guard-reuse spy, the full CLI smoke test) remain grouped in a final integration-and-checkpoint phase.

## Tasks

### Phase A: Shared Constants and Dependency Declaration

- [x] 1. Add shared constants and dependency declarations
  - [x] 1.1 Add batch integration constants to `conga/util.py`
    - Add `OBSM_KEY_PCA_GEX_UNINTEGRATED: str = 'X_pca_gex_unintegrated'` and `OBSM_KEY_PCA_GEX_INTEGRATED: str = 'X_pca_gex_integrated'`, following the existing `OBSM_KEY_VEC_TCR`/`OBSM_KEY_PCA_TCR` convention
    - Add `UNS_KEY_BATCH_INTEGRATION_CONFIG: str = 'batch_integration_config'`, following the existing `UNS_KEY_ACTIVE_TCR_REP` convention
    - Add `BATCH_INTEGRATION_METHODS: frozenset[str] = frozenset({'harmony', 'scvi'})`
    - _Requirements: 4.3, 4.4, 2.1_

  - [x] 1.2 Add the `batch-integration` optional-dependency group to `pyproject.toml`
    - Add `[project.optional-dependencies]` group `batch-integration = ["harmonypy>=0.0.10", "scvi-tools>=1.1.0"]`
    - Update the `all` and `all-gpu` groups to include `batch-integration` alongside the existing `performance`/`batch`/`scvi`/`dev` entries
    - Leave the pre-existing `batch` (`bbknn`) and `scvi` extras untouched; do not add `scanorama` or `bbknn` to the new group
    - _Requirements: 7.1, 7.2, 7.3_

### Phase B: Counts_Layer Capture and Fixed_HVG_Pathway Fixes (sequential within `filter_normalize_and_hvg`)

- [x] 2. Capture the Counts_Layer in `filter_normalize_and_hvg`
  - [x] 2.1 Add unconditional `adata.layers['counts']` capture
    - Insert `adata.layers['counts'] = adata.raw.X.copy()` immediately after the antibody-feature-removal block and immediately before `sc.pp.normalize_total`/`sc.pp.log1p` (existing lines ~546-562), so gene/cell ordering matches the final `adata.raw.X` used downstream
    - Confirm the line is unconditional (no parameter, no branch) and runs for every call regardless of `hvg_batch_key` or `force_variable_genes`
    - _Requirements: 3.1, 3.2, 3.3_

  - [x] 2.2 Write unit tests for Counts_Layer capture
    - Test that `adata.layers['counts']` equals the pre-normalization `adata.raw.X` values elementwise, for both the case where antibody features are present and where they are absent
    - Test that `'counts' in adata.layers` holds when `filter_normalize_and_hvg` is called with no `hvg_batch_key` and no `force_variable_genes` set (Requirement 3.2)
    - Place in `tests/test_counts_layer.py`
    - _Requirements: 3.1, 3.2, 3.3_

- [x] 3. Fix the Fixed_HVG_Pathway guard in `filter_normalize_and_hvg`
  - [x] 3.1 Skip automatic HVG detection when `force_variable_genes` or a caller-set `highly_variable` column is present
    - Restructure the HVG-selection block (existing lines ~565-597) into the three-state guard from the design: no caller input (run `sc.pp.highly_variable_genes` as today), `adata.uns['force_variable_genes']` set (build `hvg_mask` from the gene list, skip auto-HVG), or `adata.var['highly_variable']` pre-set with neither of the other two (use the caller's mask verbatim, skip auto-HVG)
    - Ensure all three states fall through to the same unconditional TR/IG and sex-linked exclusion code that follows
    - _Requirements: 5.1, 5.2, 5.3_

  - [x] 3.2 Log excluded gene symbols and record Fixed_Gene_List/mask sizes
    - When `force_variable_genes` names a symbol absent from `adata.var_names`, exclude it from `hvg_mask` and log the excluded symbols and their count at `logging.WARNING`
    - Record `adata.uns['conga_stats']['fixed_hvg_list_size']` (caller-supplied list length, or mask size for the direct `highly_variable` assignment state) and `adata.uns['conga_stats']['fixed_hvg_mask_size']` (resulting mask size after TR/IG and sex-linked exclusion, reusing the existing `num_highly_variable_genes` computation)
    - _Requirements: 5.6, 5.8_

  - [x] 3.3 Write unit tests for the Fixed_HVG_Pathway
    - Test that a gene symbol absent from `adata.var_names` is excluded from `hvg_mask` and logs a `logging.WARNING` with the correct count (Requirement 5.6)
    - Test that `sc.pp.highly_variable_genes` is not called when `force_variable_genes` is set, via mock/spy assertion (Requirement 5.3)
    - Test that a direct `adata.var['highly_variable']` assignment survives `filter_normalize_and_hvg` unmodified when neither `force_variable_genes` nor `hvg_batch_key` is set (Requirement 5.2)
    - Test that TR/IG and sex-linked exclusion still reduce the caller-supplied mask (Requirement 5.7)
    - Test that `conga_stats['fixed_hvg_list_size']` and `conga_stats['fixed_hvg_mask_size']` are recorded with correct values (Requirement 5.8)
    - Place in `tests/test_fixed_hvg_pathway.py`
    - _Requirements: 5.2, 5.3, 5.6, 5.7, 5.8_

### Phase C: `batch_integration()` Core Implementation (Harmony and scVI tracks in parallel)

- [ ] 4. Implement shared validation and entry point scaffold
  - [x] 4.1 Implement `_validate_batch_key`
    - Raise `ValueError` naming the missing column if `batch_key` is absent from `adata.obs`
    - Raise `ValueError` naming the column if `adata.obs[batch_key].nunique() < 2`
    - _Requirements: 1.4, 1.5_

  - [x] 4.2 Write unit tests for `_validate_batch_key`
    - Test column absent from `adata.obs` (Requirement 1.4)
    - Test column present with 0 or 1 distinct values (Requirement 1.5)
    - Test column present with >= 2 distinct values raises no error
    - Place in `tests/test_batch_integration.py`
    - _Requirements: 1.4, 1.5_

  - [x] 4.3 Implement the `batch_integration()` signature, method validation, and mutual-exclusion check
    - Implement the public signature `batch_integration(adata, batch_key, method, *, n_gex_pcs=40, hvg_min_mean=0.0125, hvg_max_mean=3, hvg_min_disp=0.5, min_genes_per_cell=None, max_genes_per_cell=None, max_percent_mito=None, normalize_antibody_features_CLR=True, scvi_max_epochs=None, random_seed=util.DEFAULT_RANDOM_SEED) -> AnnData`
    - Call `_validate_batch_key`, then raise `ValueError` naming the supplied value and `harmony`/`scvi` if `method not in util.BATCH_INTEGRATION_METHODS`
    - Raise `ValueError` stating the two pathways are mutually exclusive if `adata.uns.get('force_variable_genes')` is present at entry
    - Call `filter_normalize_and_hvg(adata, hvg_batch_key=batch_key, ...)` passing through the relevant preprocessing parameters
    - _Requirements: 1.1, 1.2, 1.3, 1.6, 2.1, 2.2, 5.5_

  - [~] 4.4 Write unit tests for method validation and mutual exclusion
    - Test `'scanorama'`, `'bbknn'`, and an arbitrary string all raise `ValueError` naming the supported set (Requirement 2.2)
    - Test that calling `batch_integration()` with `adata.uns['force_variable_genes']` already set raises `ValueError` (Requirement 5.5)
    - Place in `tests/test_batch_integration.py`
    - _Requirements: 2.2, 5.5_

- [ ] 5. Implement the Harmony integration track
  - [~] 5.1 Implement `_regress_out_technical_covariates`
    - Call `sc.pp.regress_out(adata, ['n_counts', 'percent_mito'])` unconditionally, relying on both columns already being populated by the preceding `filter_normalize_and_hvg` call
    - Do not call `sc.pp.scale()` anywhere in this helper
    - _Requirements: (supports 2.3, per Overview point 4 / Resolved Decisions on regress-out-but-unscaled PCA input)_

  - [~] 5.2 Implement `_run_harmony_integration` and wire the `harmony` branch of `batch_integration()`
    - Cast `adata.obs[batch_key]` to a pandas categorical dtype
    - Import `harmonypy` only inside this function; on `ImportError`, raise an `ImportError` naming `harmonypy` and the `conga[batch-integration]` extra
    - Call `_regress_out_technical_covariates`, then `sc.tl.pca(adata, svd_solver='arpack', n_comps=n_gex_pcs)`, store the result under `util.OBSM_KEY_PCA_GEX_UNINTEGRATED`
    - Call `scanpy.external.pp.harmony_integrate(adata, key, basis='X_pca', adjusted_basis=util.OBSM_KEY_PCA_GEX_INTEGRATED)`, which mutates `adata.obsm` in place and returns `None`
    - Set `adata.obsm['X_pca_gex'] = adata.obsm[util.OBSM_KEY_PCA_GEX_INTEGRATED]` and record `method`, `batch_key`, and `n_batches` in `adata.uns[util.UNS_KEY_BATCH_INTEGRATION_CONFIG]`
    - _Requirements: 2.3, 2.5, 2.7, 4.1, 4.2, 4.4_

  - [~] 5.3 Write unit and integration tests for the Harmony path
    - Integration test: on a small synthetic `AnnData` (a few hundred cells, two batches) with `harmonypy` installed, assert `OBSM_KEY_PCA_GEX_UNINTEGRATED`, `OBSM_KEY_PCA_GEX_INTEGRATED`, and `X_pca_gex` are all present, `X_pca_gex` equals `OBSM_KEY_PCA_GEX_INTEGRATED` elementwise, and the two representations are not elementwise-equal to each other (Requirement 4.1, 4.2)
    - Integration test: with `harmonypy` uninstalled (via `sys.modules` patching), `method='harmony'` raises `ImportError` naming `conga[batch-integration]` (Requirement 2.5)
    - Unit test: `adata.uns[util.UNS_KEY_BATCH_INTEGRATION_CONFIG]` contents (method, batch_key, n_batches) match the run's actual inputs (Requirement 4.4)
    - Place in `tests/test_batch_integration.py`
    - _Requirements: 2.5, 4.1, 4.2, 4.4_

- [ ] 6. Implement the scVI integration track
  - [~] 6.1 Implement `_run_scvi_integration` and wire the `scvi` branch of `batch_integration()`
    - Raise `ValueError` stating that the `scvi` Integration_Method requires the Counts_Layer and naming `filter_normalize_and_hvg`, if `'counts' not in adata.layers`
    - Import `scvi` only inside this function; on `ImportError`, raise an `ImportError` naming `scvi-tools` and the `conga[batch-integration]` extra
    - Call `_regress_out_technical_covariates`, then `sc.tl.pca(adata, svd_solver='arpack', n_comps=n_gex_pcs)`, store the result under `util.OBSM_KEY_PCA_GEX_UNINTEGRATED`
    - Call `scvi.model.SCVI.setup_anndata(adata, layer='counts', batch_key=batch_key, continuous_covariate_keys=['percent_mito'])`, train an `scvi.model.SCVI(adata)` model (passing `max_epochs=scvi_max_epochs`), and read `model.get_latent_representation()`
    - Store the latent representation under `util.OBSM_KEY_PCA_GEX_INTEGRATED`, set `adata.obsm['X_pca_gex']` to it, and record `method`, `batch_key`, and `n_batches` in `adata.uns[util.UNS_KEY_BATCH_INTEGRATION_CONFIG]`
    - _Requirements: 2.4, 2.6, 2.7, 3.4, 3.5, 4.1, 4.2, 4.4_

  - [~] 6.2 Write unit and integration tests for the scVI path
    - Unit test: `_run_scvi_integration` raises `ValueError` when `adata.layers['counts']` is absent, using a fixture that skips `filter_normalize_and_hvg` (Requirement 3.4)
    - Integration test (marked `slow`): on a small synthetic `AnnData` with `scvi-tools` installed, assert the same `obsm` structure as the Harmony test, and assert `scvi.model.SCVI.setup_anndata` was called with `layer='counts'`, the correct `batch_key`, and `continuous_covariate_keys=['percent_mito']` via a spy/mock rather than asserting on trained-model output values (Requirement 3.5)
    - Integration test: with `scvi-tools` uninstalled (via `sys.modules` patching), `method='scvi'` raises `ImportError` naming `conga[batch-integration]` (Requirement 2.6)
    - Place in `tests/test_batch_integration.py`
    - _Requirements: 2.6, 3.4, 3.5, 4.1, 4.2_

- [~] 7. Checkpoint - Ensure Phase C tests pass
  - Ensure all tests pass, ask the user if questions arise.

### Phase D: CLI Integration

- [ ] 8. Add and validate `--batch_key`/`--batch_integration_method` CLI flags
  - [~] 8.1 Add the two new flags to `scripts/run_conga.py`
    - Add `parser.add_argument('--batch_key', type=str, default=None, ...)` and `parser.add_argument('--batch_integration_method', type=str, default=None, ...)`, placed immediately after the existing `--force_variable_genes` flag and before `--batch_keys`
    - Do not use `choices=` at `add_argument` time (so `None` is not rejected before the pairing check runs)
    - _Requirements: 6.1, 6.2_

  - [~] 8.2 Add post-import validation checks
    - Immediately after `import conga`/`from conga import util`, add: a check that `--batch_key` and `--batch_integration_method` are supplied together, exiting nonzero naming both flags if not (Requirement 6.3)
    - A check that `--batch_integration_method` is in `util.BATCH_INTEGRATION_METHODS` when supplied, exiting nonzero naming the supplied value and the supported set if not (Requirement 6.6)
    - A check that `--force_variable_genes` is not supplied together with `--batch_key` or `--batch_integration_method`, exiting nonzero naming the conflicting flags if it is (Requirement 6.5)
    - _Requirements: 6.3, 6.4, 6.5, 6.6_

  - [~] 8.3 Wire the call site
    - After the existing `if args.force_variable_genes:` block and before the `filter_and_scale` call, add an `if args.batch_key:` branch that calls `conga.preprocess.batch_integration(adata, batch_key=args.batch_key, method=args.batch_integration_method)` instead of `filter_and_scale`
    - Keep the existing `filter_and_scale` call as the `else` branch for the Fixed_HVG_Pathway and Default_Pathway
    - _Requirements: 6.1, 6.2_

  - [~] 8.4 Write CLI validation and smoke tests
    - Unit tests: `--batch_integration_method` without `--batch_key` exits nonzero naming both flags (Requirement 6.3); `--batch_key` without `--batch_integration_method` likewise; `--force_variable_genes` with either new flag exits nonzero (Requirement 6.5); an unsupported `--batch_integration_method` value exits nonzero naming the supported set (Requirement 6.6)
    - Integration test: invoke `scripts/run_conga.py` as a subprocess with `--batch_key`+`--batch_integration_method=harmony` on a small fixture dataset, assert exit code 0 and the expected `obsm` keys in the output `.h5ad`
    - Place in `tests/test_run_conga_cli.py` (new file)
    - _Requirements: 6.3, 6.5, 6.6_

### Phase E: Final Integration, Round-Trip, and Compatibility Checks

- [ ] 9. Verify cross-cutting correctness and backward compatibility
  - [~] 9.1 Write the `.h5ad` round-trip test
    - Run the Full_Integration_Pathway, write to a temp `.h5ad`, read it back, assert `OBSM_KEY_PCA_GEX_UNINTEGRATED`, `OBSM_KEY_PCA_GEX_INTEGRATED`, and `UNS_KEY_BATCH_INTEGRATION_CONFIG` all round-trip elementwise/value-equal
    - Place in `tests/test_batch_integration.py`
    - _Requirements: 4.5_

  - [~] 9.2 Write the downstream-consumption guard-reuse test
    - After the Full_Integration_Pathway populates `adata.obsm['X_pca_gex']`, call `cluster_and_tsne_and_umap(adata)` with `recompute_pca_gex=False` and assert, via a spy on `sc.tl.pca`, that `sc.tl.pca` is not called again
    - Place in `tests/test_batch_integration.py`
    - _Requirements: 4.2_

  - [~] 9.3 Final checkpoint - full suite, backward compatibility, and optional-dependency isolation
    - Run the full test suite and confirm all tests pass
    - Confirm a no-argument `filter_and_scale`/`filter_normalize_and_hvg` preprocessing run is unaffected (no new required parameters, only an added `adata.layers['counts']` key)
    - Confirm the Full_Integration_Pathway and Fixed_HVG_Pathway are mutually exclusive in practice by running both the CLI flag-pairing test and the programmatic `batch_integration()` mutual-exclusion test together, not by code review alone
    - Confirm `import conga` succeeds with neither `harmonypy` nor `scvi-tools` installed (run in an environment/venv lacking both, or simulate via `sys.modules` patching covering the whole import chain)
    - Ask the user if questions arise.

## Notes

- Use `mamba run -n conga-dev python ...` / `mamba run -n conga-dev pytest ...` for all Python execution, per this project's development-workflow steering doc. Do not invoke a bare `python`, `pip`, or `pytest`.
- Install the new optional dependencies before running Harmony/scVI-path tests: `mamba run -n conga-dev pip install -e ".[batch-integration]"`. The `ImportError`-path tests (Requirement 2.5, 2.6) intentionally exercise the *absence* of these packages and should be run in an environment or mocked context where they are not importable.
- Behavior change to flag explicitly: `filter_normalize_and_hvg` now unconditionally adds `adata.layers['counts']` for every caller, including existing callers who never touch batch integration. This is additive only (no existing key, column, or return value changes) but increases per-call memory usage by one copy of the raw count matrix; note this if profiling memory regressions elsewhere.
- Behavior change to flag explicitly: when `force_variable_genes` is set, `sc.pp.highly_variable_genes` is no longer called at all (previously it ran and its result was discarded). Any code that relied on `adata.var['highly_variable']` reflecting the *automatic* detection result even when `force_variable_genes` was also set will observe a different value after this change; this is the intended fix for Requirement 5.3.
- `batch_integration()` does not call `filter_and_scale` and therefore does not run `sc.pp.scale()` on either the `harmony` or `scvi` path — this is a deliberate, reviewed design decision (see design.md Overview point 4 and Resolved Decisions), not an oversight. Do not "fix" this by adding a `sc.pp.scale()` call during implementation.
- The scVI integration test is expected to be marked `slow` (per `pyproject.toml`'s existing `slow` pytest marker) since even a tiny fixture's training loop is not instantaneous; do not attempt to make it fast by mocking away the actual `SCVI.train()` call in the integration test — that is what the separate `setup_anndata`-spy assertion is for.
- Test files referenced above (`tests/test_batch_integration.py`, `tests/test_fixed_hvg_pathway.py`, `tests/test_counts_layer.py`, `tests/test_run_conga_cli.py`) are new files; none exist in the current `tests/` directory.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["2.1", "3.1"] },
    { "id": 2, "tasks": ["2.2", "3.2"] },
    { "id": 3, "tasks": ["3.3", "4.1"] },
    { "id": 4, "tasks": ["4.2", "4.3"] },
    { "id": 5, "tasks": ["4.4", "5.1", "6.1"] },
    { "id": 6, "tasks": ["5.2", "6.2"] },
    { "id": 7, "tasks": ["5.3"] },
    { "id": 8, "tasks": ["8.1"] },
    { "id": 9, "tasks": ["8.2"] },
    { "id": 10, "tasks": ["8.3"] },
    { "id": 11, "tasks": ["8.4", "9.1", "9.2"] },
    { "id": 12, "tasks": ["9.3"] }
  ]
}
```

## Success Metrics

### Correctness and Behavioral Guarantees
- Single shared `batch_key` drives both HVG selection and the Integration_Method with no independently-settable second key (Requirement 1)
- `method` is restricted to exactly `{'harmony', 'scvi'}`; all other values, including `scanorama` and `bbknn`, are rejected with `ValueError` (Requirement 2)
- `harmonypy` and `scvi-tools` are imported only inside their respective helper functions; `import conga.preprocess` succeeds with neither installed (Requirement 2.7)
- `adata.layers['counts']` is captured unconditionally, matches `adata.raw.X`'s ordering at capture time, and is present regardless of which pathway (if any) runs afterward (Requirement 3)
- `OBSM_KEY_PCA_GEX_UNINTEGRATED` and `OBSM_KEY_PCA_GEX_INTEGRATED` are both populated after the Full_Integration_Pathway, `X_pca_gex` equals the latter elementwise, and both representations differ from each other elementwise (Requirement 4.1, 4.2)
- `adata.uns[UNS_KEY_BATCH_INTEGRATION_CONFIG]` and both new `obsm` arrays round-trip through `.h5ad` elementwise/value-equal (Requirement 4.5)
- The Fixed_HVG_Pathway skips `sc.pp.highly_variable_genes` entirely (not merely discards its result), logs excluded gene symbols, still applies TR/IG and sex-linked exclusion, and records both `conga_stats` size keys (Requirement 5)
- The Full_Integration_Pathway and Fixed_HVG_Pathway are mutually exclusive, enforced both at the CLI layer and inside `batch_integration()` itself (Requirement 5.5, 6.5)
- `cluster_and_tsne_and_umap` and `calc_nbrs` require no code changes and correctly reuse a pre-populated `X_pca_gex`, confirmed by a spy on `sc.tl.pca` rather than by code inspection alone (Requirement 4.2)
- A no-argument preprocessing run (no `batch_key`, no `force_variable_genes`) is behaviorally identical to today's output apart from the added `adata.layers['counts']` key (backward compatibility)

### Dependency Installation Verification
- `conga[batch-integration]` installs `harmonypy>=0.0.10` and `scvi-tools>=1.1.0` and nothing else new; neither `scanorama` nor `bbknn` is added to this group (Requirement 7)
- The pre-existing `batch` (`bbknn`) and `scvi` extras remain unchanged and installable independently of `batch-integration`
- `ImportError` messages raised by `_run_harmony_integration` and `_run_scvi_integration` correctly name `conga[batch-integration]` as the extra to install
