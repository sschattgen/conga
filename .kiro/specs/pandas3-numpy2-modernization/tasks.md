# Pandas 3.0/NumPy 2.0 Compatibility Modernization Tasks

## Overview

Comprehensive modernization of CoNGA codebase for pandas 3.0+ and NumPy 2.0+ compatibility. This addresses runtime failures, silent behavior changes, and deprecated APIs to ensure reliable operation with the current scientific Python stack.

## Phase 1: Critical Runtime Fixes

- [ ] 1. Fix immediate pandas indexing failures
  - [ ] 1.1 Fix correlations.py Series indexing error
    - Replace `is_mait[double_nbrs]` with `is_mait.iloc[double_nbrs]` on line 111
    - Replace `agroups[double_nbrs]` and `bgroups[double_nbrs]` with `.iloc[]` on lines 93-94
    - Test with existing test data to ensure CoNGA correlation analysis completes
    - _Requirements: CR1, TR2_
  
  - [ ] 1.2 Fix AnnData deprecated method calls
    - Replace `adata.obs_keys()` with `adata.obs.columns` throughout codebase
    - Replace `adata.uns_keys()` with `adata.uns.keys()` throughout codebase  
    - Replace `adata.obsm_keys()` with `adata.obsm.keys()` throughout codebase
    - Replace `adata.isview` with `adata.is_view` in preprocess.py line 233
    - _Requirements: CR2, TR4_
  
  - [ ] 1.3 Fix scanpy deprecated API calls
    - Replace `sc.logging.print_versions()` with `sc.logging.print_header()` in run_conga.py
    - Update any other deprecated scanpy function calls found during testing
    - _Requirements: CR3_
  
  - [ ] 1.4 Validate critical workflow restoration
    - Test `run_conga.py --graph_vs_graph` completes without pandas/numpy errors
    - Verify both classic and vectorized TCR paths work
    - Confirm basic correlation analysis produces results
    - _Requirements: BR1, TR10_

- [ ] 2. Create systematic modernization infrastructure
  - [ ] 2.1 Add compatibility checking utilities
    - Create `conga/compatibility.py` module with version checking functions
    - Add `check_environment_compatibility()` function for startup validation
    - Add utility functions for safe AnnData method access
    - _Requirements: TR11_
  
  - [ ] 2.2 Create automated modernization scripts
    - Write `scripts/modernize_numpy_dtypes.py` for batch NumPy alias replacement
    - Write `scripts/audit_cow_patterns.py` for copy-on-write pattern detection
    - Create test harness for before/after result comparison
    - _Requirements: SA2, TR12_

## Phase 2: Systematic Code Pattern Updates

- [ ] 3. NumPy 2.0 legacy alias elimination
  - [ ] 3.1 Audit and replace NumPy dtype aliases
    - Search codebase for `np.float_`, `np.int_`, `np.bool_`, `np.object_` usage
    - Replace with explicit modern dtypes: `np.float64`, `np.int64`, `bool`, `object`
    - Test numeric operations in distance calculations and vectorized encoding
    - _Requirements: TR5, SA2_
  
  - [ ] 3.2 Fix numpy.core import patterns
    - Find any `from numpy.core import` statements
    - Replace with public numpy APIs or conditional imports
    - _Requirements: TR6_
  
  - [ ] 3.3 Update numeric casting for NEP 50 compliance
    - Review distance matrix calculations in tcrdist modules
    - Add explicit `.astype()` calls for mixed int/float operations
    - Test vectorized encoding numeric operations
    - _Requirements: TR7, TR8_

- [ ] 4. Pandas 3.0 copy-on-write compliance
  - [ ] 4.1 Audit preprocess.py for chained assignment
    - Find patterns like `df[col][mask] = value` in preprocessing functions  
    - Replace with explicit `df.loc[mask, col] = value` indexing
    - Test data filtering and normalization steps
    - _Requirements: TR1, SA1_
  
  - [ ] 4.2 Audit tcrdist modules for DataFrame mutations
    - Review `conga/tcrdist/*.py` for unsafe mutation patterns
    - Update TCR distance calculation DataFrame operations
    - Verify TCR clumping analysis works correctly
    - _Requirements: TR1, SA1_
  
  - [ ] 4.3 Update string dtype handling
    - Find `dtype == object` checks on string columns (V/J genes, CDR3s)
    - Replace with `pd.api.types.is_string_dtype()` or equivalent
    - Test V/J gene processing and CDR3 validation
    - _Requirements: TR3, SA3_
  
  - [ ] 4.4 Review inplace operation semantics
    - Audit all `inplace=True` method calls for CoW compliance
    - Replace problematic inplace operations with assignment patterns
    - _Requirements: TR4_

- [ ] 5. Comprehensive workflow testing
  - [ ] 5.1 Test all TCR representation paths
    - Validate vectorized TCR representation path works
    - Validate classic KernelPCA path works with --use_kpca_tcrdist
    - Validate exact TCRdist path works for small datasets
    - _Requirements: TR10, TR12_
  
  - [ ] 5.2 Test all analysis workflows  
    - Test `--graph_vs_graph` analysis end-to-end
    - Test `--tcr_clumping` analysis with updated code
    - Test `--graph_vs_features` and other analysis modes
    - _Requirements: BR1, TR12_
  
  - [ ] 5.3 Validate result reproducibility
    - Run analysis with fixed random seeds before/after changes
    - Compare outputs for byte-identical results where expected
    - Document any intentional result changes
    - _Requirements: BR2, TR12_

## Phase 3: Future-Proofing and Optimization

- [ ] 6. Modern API adoption and optimization
  - [ ] 6.1 Replace deprecated pandas functions
    - Update `pd.DataFrame.append()` usage to `pd.concat()`
    - Replace other deprecated pandas patterns found during audit
    - _Requirements: BR3_
  
  - [ ] 6.2 Update scanpy function calls
    - Replace `sc.pp.normalize_per_cell()` with `sc.pp.normalize_total()`
    - Update deprecated clustering function calls
    - _Requirements: BR3_
  
  - [ ] 6.3 Performance validation and optimization
    - Benchmark key operations before/after modernization
    - Optimize any performance regressions from explicit operations
    - Document performance characteristics
    - _Requirements: Performance criteria from success metrics_

- [ ] 7. Integration validation and documentation
  - [ ] 7.1 Add compatibility diagnostics to main workflows
    - Integrate `check_environment_compatibility()` into run_conga.py startup
    - Add clear error messages for incompatible environments
    - Document supported pandas/numpy versions
    - _Requirements: TR11_
  
  - [ ] 7.2 Update installation and compatibility documentation
    - Update README.md with pandas 3.0+/NumPy 2.0+ requirements
    - Document migration path from older environments  
    - Add troubleshooting section for common compatibility issues
    - _Requirements: BR3_
  
  - [ ] 7.3 Create comprehensive regression test suite
    - Implement automated before/after result comparison tests
    - Add tests for all major code paths with pandas 3.0+/NumPy 2.0+
    - Include tests for edge cases (empty data, single cells, etc.)
    - _Requirements: TR12_

- [ ] 8. Final validation and cleanup
  - [ ] 8.1 Run complete test suite validation
    - Execute full `run_conga.py --all` workflow on test datasets
    - Validate all outputs match expected results
    - Ensure zero deprecation warnings in pandas 3.0+/NumPy 2.0+
    - _Requirements: All success criteria_
  
  - [ ] 8.2 Performance and compatibility benchmarking
    - Measure and document performance on standard benchmarks
    - Test compatibility across pandas 3.0.x and NumPy 2.x versions
    - Validate memory usage patterns haven't regressed
    - _Requirements: Performance maintenance criteria_
  
  - [ ] 8.3 Code cleanup and documentation finalization
    - Remove any temporary compatibility code or workarounds
    - Finalize all docstring updates for changed functions
    - Update version requirements in pyproject.toml/environment.yml
    - _Requirements: Code quality criteria_

## Notes

- Use `mamba run -n conga-dev python` for all testing as specified by user
- Test with current environment: pandas 3.0.6, numpy 2.5.3, Python 3.12
- Each change should be tested incrementally to isolate any issues
- Maintain compatibility test baselines for result validation
- Focus on high-impact fixes first (Phase 1) to restore basic functionality

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
