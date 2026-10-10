# Pandas 3.0/NumPy 2.0 Compatibility Modernization Tasks

## Overview

Comprehensive modernization of CoNGA codebase for pandas 3.0+ and NumPy 2.0+ compatibility. This addresses runtime failures, silent behavior changes, and deprecated APIs to ensure reliable operation with the current scientific Python stack.

## Phase 1: Critical Runtime Fixes

- [x] 1. Fix immediate pandas indexing failures
  - [x] 1.1 Fix correlations.py Series indexing error
    - Replace `is_mait[double_nbrs]` with `is_mait.iloc[double_nbrs]` on line 111
    - Replace `agroups[double_nbrs]` and `bgroups[double_nbrs]` with `.iloc[]` on lines 93-94
    - Test with existing test data to ensure CoNGA correlation analysis completes
    - _Requirements: CR1, TR2_
  
  - [x] 1.2 Fix AnnData deprecated method calls
    - Replace `adata.obs_keys()` with `adata.obs.columns` throughout codebase
    - Replace `adata.uns_keys()` with `adata.uns.keys()` throughout codebase  
    - Replace `adata.obsm_keys()` with `adata.obsm.keys()` throughout codebase
    - Replace `adata.isview` with `adata.is_view` in preprocess.py line 233
    - _Requirements: CR2, TR4_
  
  - [x] 1.3 Fix scanpy deprecated API calls
    - Replace `sc.logging.print_versions()` with `sc.logging.print_header()` in run_conga.py
    - Update any other deprecated scanpy function calls found during testing
    - _Requirements: CR3_
  
  - [x] 1.4 Validate critical workflow restoration
    - Test `run_conga.py --graph_vs_graph` completes without pandas/numpy errors
    - Verify both classic and vectorized TCR paths work
    - Confirm basic correlation analysis produces results
    - _Requirements: BR1, TR10_

- [x] 2. Create systematic modernization infrastructure
  - [x] 2.1 Add compatibility checking utilities
    - Create `conga/compatibility.py` module with version checking functions
    - Add `check_environment_compatibility()` function for startup validation
    - Add utility functions for safe AnnData method access
    - _Requirements: TR11_
    - _Verified 2026-10-09: module exists and implements all of the above
      (`check_environment_compatibility()`, `safe_obs_columns()`,
      `safe_var_columns()`, `safe_uns_keys()`, `safe_obsm_keys()`,
      `safe_is_view()`). Confirmed working via direct invocation and via
      `conga check` / the six workflow runs' startup output (see task 7.1).
      Fixed a `Pandas4Warning` leak discovered in this module during task
      8's work -- see `modernization_results/sklearn_mds_determinism_2026-10-09.md`._
  
  - [x] 2.2 Create automated modernization scripts
    - Write `scripts/modernize_numpy_dtypes.py` for batch NumPy alias replacement
    - Write `scripts/audit_cow_patterns.py` for copy-on-write pattern detection
    - Create test harness for before/after result comparison
    - _Requirements: SA2, TR12_
    - _Verified 2026-10-09: both scripts exist and run successfully (see
      task 3.1/4.1 above for actual results). The before/after comparison
      harness, `scripts/test_modernization_results.py`, exists and is
      fully implemented but its built-in test configs are stale against
      the current CLI (wrong filenames, references a nonexistent
      `--use_vectorized_tcrdist` flag) -- see
      `modernization_results/test_modernization_results_findings_2026-10-09.md`.
      Not repaired this session; workflow validation (task 5) was done by
      running `run_conga.py` directly instead._

## Phase 2: Systematic Code Pattern Updates

- [x] 3. NumPy 2.0 legacy alias elimination
  - [x] 3.1 Audit and replace NumPy dtype aliases
    - Search codebase for `np.float_`, `np.int_`, `np.bool_`, `np.object_` usage
    - Replace with explicit modern dtypes: `np.float64`, `np.int64`, `bool`, `object`
    - Test numeric operations in distance calculations and vectorized encoding
    - _Requirements: TR5, SA2_
    - _Verified 2026-10-09: ran `scripts/modernize_numpy_dtypes.py --dry-run`
      against the full codebase. Zero genuine hits in `conga/` or `scripts/`
      (the only matches were the tool's own pattern-definition strings in
      itself); confirmed independently via manual grep. No replacements
      needed. See `modernization_results/numpy_dtype_dryrun_2026-10-09.txt`._
  
  - [x] 3.2 Fix numpy.core import patterns
    - Find any `from numpy.core import` statements
    - Replace with public numpy APIs or conditional imports
    - _Requirements: TR6_
    - _Verified 2026-10-09: grep for `numpy.core`/`from numpy.core` across
      `conga/` returns zero hits. No action needed._
  
  - [x] 3.3 Update numeric casting for NEP 50 compliance
    - Review distance matrix calculations in tcrdist modules
    - Add explicit `.astype()` calls for mixed int/float operations
    - Test vectorized encoding numeric operations
    - _Requirements: TR7, TR8_
    - _Verified 2026-10-09: vectorized TCRdist encoding (`conga/tcrdist/
      vectorized.py`) already produces explicit float32 C-contiguous
      output per its own design; full workflow matrix (vectorized, exact,
      and KernelPCA TCR paths) run end-to-end against real 10x PBMC data
      produced correct, non-NaN results with no NEP-50-related casting
      warnings. See `modernization_results/workflow_matrix_2026-10-09.md`._

- [x] 4. Pandas 3.0 copy-on-write compliance
  - [x] 4.1 Audit preprocess.py for chained assignment
    - Find patterns like `df[col][mask] = value` in preprocessing functions  
    - Replace with explicit `df.loc[mask, col] = value` indexing
    - Test data filtering and normalization steps
    - _Requirements: TR1, SA1_
    - _Verified 2026-10-09: ran `scripts/audit_cow_patterns.py` against
      `conga/`+`scripts/`; manually reviewed all flagged issues touching
      `preprocess.py`. No genuine `df[col][mask] = value` chained
      assignment on an actual DataFrame found -- flagged instances were
      false positives (dict/`.uns` subscript chaining, or already-correct
      `.loc` usage). See `modernization_results/cow_audit_2026-10-09.txt`._
  
  - [x] 4.2 Audit tcrdist modules for DataFrame mutations
    - Review `conga/tcrdist/*.py` for unsafe mutation patterns
    - Update TCR distance calculation DataFrame operations
    - Verify TCR clumping analysis works correctly
    - _Requirements: TR1, SA1_
    - _Verified 2026-10-09: same audit as 4.1 covered `conga/tcrdist/*.py`;
      no genuine chained-assignment risk found. `--tcr_clumping` run
      end-to-end against real 10x PBMC data (including the compiled C++
      `calc_distributions`/`find_neighbors` binaries) completed correctly.
      See `modernization_results/workflow_matrix_2026-10-09.md`._
  
  - [x] 4.3 Update string dtype handling
    - Find `dtype == object` checks on string columns (V/J genes, CDR3s)
    - Replace with `pd.api.types.is_string_dtype()` or equivalent
    - Test V/J gene processing and CDR3 validation
    - _Requirements: TR3, SA3_
    - _Verified 2026-10-09: grep for `dtype == object`/`dtype == np.object`
      on gene/CDR3-like columns across `conga/` returns zero hits. No
      action needed._
  
  - [x] 4.4 Review inplace operation semantics
    - Audit all `inplace=True` method calls for CoW compliance
    - Replace problematic inplace operations with assignment patterns
    - _Requirements: TR4_
    - _Verified 2026-10-09: individually reviewed all 29 `inplace=True`
      call sites across 9 files in `conga/`. Every site operates on a
      DataFrame/Series the calling function owns outright (freshly
      constructed/loaded, or an AnnData container accessed by attribute),
      never a subscript-derived view/slice. No changes required. Full
      sign-off in `modernization_results/inplace_true_audit_2026-10-09.md`._

- [x] 5. Comprehensive workflow testing
  - [x] 5.1 Test all TCR representation paths
    - Validate vectorized TCR representation path works
    - Validate classic KernelPCA path works with --use_kpca_tcrdist
    - Validate exact TCRdist path works for small datasets
    - _Requirements: TR10, TR12_
    - _Verified 2026-10-09: ran all three paths end-to-end against real
      10x human PBMC data (`test_data/SC5v2_humanPBMCs_clones.tsv` +
      matching 10x H5, 1390 clonotypes post-QC). All three completed to
      `DONE` with correct `*_final.h5ad` output. KernelPCA path required
      generating `_AB.dist_50_kpcs` via `setup_10x_for_conga.py` first --
      documented as expected two-stage pipeline behavior, not a defect.
      See `modernization_results/workflow_matrix_2026-10-09.md`._
  
  - [x] 5.2 Test all analysis workflows  
    - Test `--graph_vs_graph` analysis end-to-end
    - Test `--tcr_clumping` analysis with updated code
    - Test `--graph_vs_features` and other analysis modes
    - _Requirements: BR1, TR12_
    - _Verified 2026-10-09: `--graph_vs_graph` (x3, one per TCR path),
      `--tcr_clumping`, `--graph_vs_features`, and `--all` (which chains
      graph_vs_graph, graph_vs_graph_stats, graph_vs_features,
      cluster_vs_cluster, find_hotspot_features, find_gex_cluster_degs,
      tcr_clumping, match_to_tcr_database, make_tcrdist_trees) all run to
      completion. See `modernization_results/workflow_matrix_2026-10-09.md`._
  
  - [ ] 5.3 Validate result reproducibility
    - Run analysis with fixed random seeds before/after changes
    - Compare outputs for byte-identical results where expected
    - Document any intentional result changes
    - _Requirements: BR2, TR12_
    - _Partially covered: all runs in 5.1/5.2 used `--random_seed 42`, and
      seed-determinism itself is covered independently by the dedicated
      `pipeline-reproducibility` spec and its
      `tests/test_pipeline_reproducibility*.py` suite. Not done as part of
      this task: a direct "before" (pre-modernization) vs. "after" byte-
      identical comparison -- there is no legacy pandas<3.0/numpy<1.0
      environment available in this workspace to generate a true "before"
      baseline from. See the harness-limitation writeup in
      `modernization_results/test_modernization_results_findings_2026-10-09.md`
      for why `scripts/test_modernization_results.py`'s intended
      before/after workflow could not be used as shipped._

## Phase 3: Future-Proofing and Optimization

- [x] 6. Modern API adoption and optimization
  - [x] 6.1 Replace deprecated pandas functions
    - Update `pd.DataFrame.append()` usage to `pd.concat()`
    - Replace other deprecated pandas patterns found during audit
    - _Requirements: BR3_
    - _Verified 2026-10-09: grep for `.append(` across `conga/`+`scripts/`
      found only plain `list.append()`/`sys.path.append()` calls, no
      `pd.DataFrame.append()` usage anywhere. No action needed._
  
  - [x] 6.2 Update scanpy function calls
    - Replace `sc.pp.normalize_per_cell()` with `sc.pp.normalize_total()`
    - Update deprecated clustering function calls
    - _Requirements: BR3_
    - _Verified 2026-10-09: grep for `sc.pp.normalize_per_cell` across the
      codebase returns zero hits. `sc.logging.print_versions()` was
      already replaced with `sc.logging.print_header()` in Phase 1 (task
      1.3). No further scanpy deprecations found during the full workflow
      matrix run (zero scanpy deprecation warnings observed)._
  
  - [ ] 6.3 Performance validation and optimization
    - Benchmark key operations before/after modernization
    - Optimize any performance regressions from explicit operations
    - Document performance characteristics
    - _Requirements: Performance criteria from success metrics_
    - _Not done: no before/after performance benchmark was run (same
      limitation as 5.3 -- no legacy environment available to benchmark
      "before" against). The six full-pipeline runs in
      `modernization_results/workflow_matrix_2026-10-09.md` completed in
      normal/expected wall-clock time for this dataset size, with no
      obvious regressions, but this was not a rigorous timed benchmark._

- [x] 7. Integration validation and documentation
  - [x] 7.1 Add compatibility diagnostics to main workflows
    - Integrate `check_environment_compatibility()` into run_conga.py startup
    - Add clear error messages for incompatible environments
    - Document supported pandas/numpy versions
    - _Requirements: TR11_
    - _Done 2026-10-09: wired `check_environment_compatibility()` into both
      `scripts/run_conga.py` and `scripts/setup_10x_for_conga.py` startup,
      exiting with a clear message on `CompatibilityError`. Added a
      `conga check` CLI subcommand (`conga/cli.py`) for standalone checks.
      Verified working via direct invocation and via the six workflow runs,
      which all print the compatibility report at startup._
  
  - [x] 7.2 Update installation and compatibility documentation
    - Update README.md with pandas 3.0+/NumPy 2.0+ requirements
    - Document migration path from older environments  
    - Add troubleshooting section for common compatibility issues
    - _Requirements: BR3_
    - _Done 2026-10-09: added a full "Compatibility" section to README.md
      (linked from the Table of Contents) covering why pandas 3.0+/NumPy
      2.0+ matter, how to check your environment (`conga check`), and a
      troubleshooting list (CompatibilityError, legacy NumPy aliases,
      chained-assignment CoW silent no-ops, NEP 50 casting, the
      `mode.copy_on_write` removal). Added 2 corresponding FAQ entries._
  
  - [ ] 7.3 Create comprehensive regression test suite
    - Implement automated before/after result comparison tests
    - Add tests for all major code paths with pandas 3.0+/NumPy 2.0+
    - Include tests for edge cases (empty data, single cells, etc.)
    - _Requirements: TR12_
    - _Not done: no new automated before/after regression test suite was
      added. `scripts/test_modernization_results.py` exists but its
      hardcoded test configs are stale against the current CLI (see
      `modernization_results/test_modernization_results_findings_2026-10-09.md`)
      and were not repaired as part of this session. The existing
      `tests/` suite (526 tests, including dedicated pipeline-
      reproducibility and error-condition coverage) was used as the
      verification vehicle instead, but it predates this specific task
      and wasn't written for before/after pandas3/numpy2 comparison._

- [x] 8. Final validation and cleanup
  - [x] 8.1 Run complete test suite validation
    - Execute full `run_conga.py --all` workflow on test datasets
    - Validate all outputs match expected results
    - Ensure zero deprecation warnings in pandas 3.0+/NumPy 2.0+
    - _Requirements: All success criteria_
    - _Done 2026-10-09: `run_conga.py --all` run against real 10x PBMC
      data completed to `DONE` with all expected output files. Swept all
      six workflow runs' output for `DeprecationWarning`/`FutureWarning`/
      `Pandas4Warning`/`Pandas3Warning` and pandas/numpy `RuntimeWarning`s
      -- zero pandas/numpy deprecation warnings found. (One
      `Pandas4Warning` was found and fixed in `conga/compatibility.py`
      itself during task 8's sklearn re-verification work, see below;
      remaining warnings -- scanpy unique-var-names, a pre-existing
      `log2`-of-negative numerical edge case, matplotlib figure-count, and
      ImageMagick/findfont tooling warnings -- are unrelated to the
      pandas3/numpy2 migration.) See
      `modernization_results/workflow_matrix_2026-10-09.md`._
  
  - [ ] 8.2 Performance and compatibility benchmarking
    - Measure and document performance on standard benchmarks
    - Test compatibility across pandas 3.0.x and NumPy 2.x versions
    - Validate memory usage patterns haven't regressed
    - _Requirements: Performance maintenance criteria_
    - _Partially done: compatibility was validated against the one
      pandas/NumPy version combination actually installed in `conga-dev`
      (pandas 3.0.6 / NumPy 2.5.3), not a matrix of 3.0.x/2.x versions --
      there's no mechanism in this workspace to install and test against
      multiple pandas-3.x or NumPy-2.x point releases side by side.
      Memory-usage regression was not independently measured (same
      limitation noted in 6.3: no legacy "before" environment to diff
      against). As part of this task, also re-verified
      scikit-learn-1.9.1-specific MDS/SMACOF determinism (the environment
      resolves 1.9.1, not the 1.3 the vectorized-tcrdist design assumed)
      and confirmed `aa_embedding()` is still bit-identical deterministic
      with matching accuracy (stress 0.4526 at dim=16). See
      `modernization_results/sklearn_mds_determinism_2026-10-09.md`._
  
  - [ ] 8.3 Code cleanup and documentation finalization
    - Remove any temporary compatibility code or workarounds
    - Finalize all docstring updates for changed functions
    - Update version requirements in pyproject.toml/environment.yml
    - _Requirements: Code quality criteria_
    - _Not done: `pyproject.toml`/`environment.yml` version floors were not
      re-reviewed as part of this session (they already specify pandas
      3.0+/numpy 2.0+ per the dependency table in the project steering
      docs). No temporary compatibility code/workarounds were identified
      to remove. Docstrings were not swept for staleness beyond the files
      directly touched this session (`conga/compatibility.py`,
      `scripts/run_conga.py`, `scripts/setup_10x_for_conga.py`,
      `conga/cli.py`)._

## Notes

- Use `mamba run -n conga-dev python` for all testing as specified by user
- Test with current environment: pandas 3.0.6, numpy 2.5.3, Python 3.14.8
  (not 3.12 as originally noted here -- the `conga-dev` environment actually
  resolves Python 3.14.8; scikit-learn resolves to 1.9.1, not the 1.3
  assumed during the vectorized-tcrdist design phase. See
  `modernization_results/sklearn_mds_determinism_2026-10-09.md`.)
- Each change should be tested incrementally to isolate any issues
- Maintain compatibility test baselines for result validation
- Focus on high-impact fixes first (Phase 1) to restore basic functionality
- **2026-10-09 verification pass:** Phases 2 and 3 (tasks 3-8) were verified
  with real evidence -- tool runs, manual code review, and six end-to-end
  pipeline runs against real 10x PBMC data -- rather than left as unchecked
  claims. All dated reports live under `modernization_results/`. Three
  sub-items remain genuinely open and are left unchecked above: 5.3/6.3/8.2's
  true before/after comparison (no legacy pandas<3.0/numpy<2.0 environment
  available in this workspace to diff against) and 7.3's dedicated
  regression suite (the existing `scripts/test_modernization_results.py`
  harness is stale against the current CLI and was not repaired this
  session). One real bug was found and fixed during this pass: a
  `Pandas4Warning` leak in `conga/compatibility.py`'s copy-on-write
  reporting code.

## Task Dependencies

Phase 1 tasks can run in parallel within each section.
Phase 2 requires Phase 1 completion for stable testing environment.  
Phase 3 builds on systematic updates from Phase 2.
Final validation (8.1-8.3) requires all previous phases complete.

## Success Criteria Reference

1. **Runtime Stability**: All CoNGA workflows complete without pandas/numpy errors
2. **Result Identity**: Analysis outputs identical to baseline with fixed seeds  
3. **Future Compatibility**: Zero deprecation warnings in target environment
4. **Performance Maintained**: <10% regression on benchmark operations
5. **Code Modernization**: Uses only stable, documented APIs from pandas 3.0+/NumPy 2.0+
