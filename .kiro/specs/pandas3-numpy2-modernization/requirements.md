# Pandas 3.0/NumPy 2.0 Compatibility Requirements

## Overview

CoNGA must be fully compatible with pandas 3.0+ and NumPy 2.0+ to support modern Python 3.12 environments. The current codebase has compatibility issues that cause both runtime failures and silent behavior changes. This comprehensive modernization ensures reliable operation across the updated scientific Python stack.

## Business Requirements

### BR1: Runtime Stability
**Requirement**: All CoNGA workflows must complete without pandas/NumPy compatibility errors
- **Rationale**: Current pandas 3.0.6/numpy 2.5.3 environment causes KeyError exceptions in correlation analysis
- **Success Criteria**: Full `run_conga.py --all` pipeline completes on test datasets

### BR2: Result Reproducibility  
**Requirement**: Analysis results must be identical across pandas/NumPy versions
- **Rationale**: Silent behavior changes in pandas 3.0 copy-on-write and NumPy 2.0 casting can alter results without warning
- **Success Criteria**: Deterministic outputs with fixed random seeds produce byte-identical results

### BR3: Future Compatibility
**Requirement**: Codebase uses modern, supported APIs to avoid future deprecation issues
- **Rationale**: Removed NumPy legacy aliases and pandas deprecation warnings indicate technical debt
- **Success Criteria**: No deprecation warnings in pandas 3.0+ and NumPy 2.0+ environments

## Technical Requirements

### Pandas 3.0 Copy-on-Write Compatibility

#### TR1: Eliminate Chained Assignment Patterns
**Requirement**: Remove all chained assignment operations that rely on pre-CoW semantics
- **Location**: `conga/preprocess.py` and `conga/tcrdist/*.py` contain legacy patterns
- **Issue**: Operations like `df.column[mask] = value` silently fail under copy-on-write
- **Solution**: Use explicit `.loc[]` indexing for all DataFrame mutations

#### TR2: Fix Pandas Series Indexing
**Requirement**: Replace positional indexing of Series with integer arrays
- **Location**: `conga/correlations.py` lines 93-94, 111
- **Issue**: `series[int_array]` fails when int_array contains indices not in series.index  
- **Solution**: Use `.iloc[]` for positional access or ensure index alignment

#### TR3: Handle String Dtype Changes
**Requirement**: Explicitly specify string handling for V/J gene and CDR3 columns
- **Issue**: pandas 3.0 defaults to string[pyarrow] dtype instead of object
- **Solution**: Validate assumptions about string column dtypes and update comparisons

#### TR4: Update Inplace Operation Semantics
**Requirement**: Review all `inplace=True` method calls for CoW compliance
- **Issue**: Some inplace operations have tightened semantics under CoW
- **Solution**: Test or replace with explicit assignment patterns

### NumPy 2.0 Compatibility

#### TR5: Remove Legacy Dtype Aliases
**Requirement**: Replace all usage of removed NumPy legacy dtype aliases
- **Targets**: `np.float_`, `np.int_`, `np.bool_`, `np.object_`
- **Solution**: Use explicit dtypes: `np.float64`, `np.int64`, `bool`, `object`
- **Scope**: All Python files in conga/ and scripts/

#### TR6: Update Core Import Patterns  
**Requirement**: Replace privatized numpy.core imports
- **Issue**: `numpy.core` moved to `numpy._core` (private)
- **Solution**: Use public numpy APIs or conditional imports with fallbacks

#### TR7: Handle Casting Rule Changes (NEP 50)
**Requirement**: Make integer/float casting explicit in distance calculations
- **Issue**: Mixed int/float operations may produce different dtypes than before
- **Location**: Distance matrix and vectorized encoding code most likely affected
- **Solution**: Add explicit `.astype()` calls for numeric operations

#### TR8: Validate Array Interface Changes
**Requirement**: Test array creation and manipulation patterns against NumPy 2.0
- **Issue**: Some array interface behaviors changed subtly
- **Solution**: Add explicit validation for array shapes and dtypes in critical paths

### Dependency Integration

#### TR9: Update Scikit-learn Integration
**Requirement**: Verify scikit-learn 1.9+ behavior with NumPy 2.0
- **Issue**: The vectorized-tcrdist design assumed sklearn 1.3; environment resolves to 1.9.1
- **Action**: Re-verify MDS/SMACOF determinism and random_state behavior

#### TR10: Test AnnData/Scanpy Compatibility  
**Requirement**: Ensure AnnData and scanpy work correctly with pandas 3.0/NumPy 2.0
- **Issue**: These are core dependencies that may have their own compatibility issues
- **Action**: Test full CoNGA workflows with current versions

### Error Handling and Diagnostics

#### TR11: Add Compatibility Diagnostics
**Requirement**: Implement version checking and compatibility warnings
- **Feature**: Startup checks that warn about known compatibility issues
- **Feature**: Clear error messages when version conflicts are detected

#### TR12: Comprehensive Test Coverage
**Requirement**: Test all major code paths against pandas 3.0/NumPy 2.0
- **Coverage**: All modules in conga/ package
- **Coverage**: All analysis workflows (graph_vs_graph, tcr_clumping, etc.)
- **Coverage**: All three TCR representation paths (vectorized, KernelPCA, exact)

## Specific Issue Fixes

### Critical Runtime Errors

#### CR1: Fix correlations.py Indexing Error
**Error**: `KeyError: "None of [Index([...], dtype='int64')] are in the [index]"`
**Location**: `conga/correlations.py` lines 93-94, 111  
**Root Cause**: Pandas Series `is_mait[double_nbrs]` indexing assumes integer positional access
**Fix**: Use `is_mait.iloc[double_nbrs]` for positional indexing

#### CR2: Fix Deprecated AnnData Methods
**Error**: Multiple `FutureWarning` messages about deprecated methods
**Locations**: `obs_keys()`, `uns_keys()`, `obsm_keys()`, `isview()`, etc.
**Fix**: Replace with modern equivalents (`obs.columns`, `uns.keys()`, `is_view`, etc.)

#### CR3: Fix Scanpy Deprecations
**Error**: `FutureWarning: The function print_versions is deprecated`
**Location**: `scripts/run_conga.py` line 415
**Fix**: Replace with `sc.logging.print_header()`

### Systematic Audits Required

#### SA1: Audit Data Frame Mutation Patterns
**Scope**: All files in `conga/preprocess.py`, `conga/tcrdist/*.py`
**Pattern**: `df[col][mask] = value` or similar chained operations
**Action**: Convert to `df.loc[mask, col] = value`

#### SA2: Audit NumPy Legacy Usage
**Scope**: All Python files in project
**Pattern**: Regular expression search for `np\.(float_|int_|bool_|object_)`
**Action**: Replace with explicit modern dtypes

#### SA3: Audit String Column Assumptions
**Scope**: V/J gene processing, CDR3 handling code
**Pattern**: `dtype == object` checks on string columns
**Action**: Use pandas string accessors or explicit string dtype checks

## Success Criteria

### Functional Validation
1. **Complete Pipeline Success**: Full `run_conga.py --all` workflow completes without errors
2. **Result Identity**: Analysis outputs identical to pandas 2.x/NumPy 1.x baseline (with fixed seeds)
3. **Performance Maintained**: No significant performance regressions from compatibility fixes

### Code Quality  
1. **Warning-Free**: No deprecation or compatibility warnings in pandas 3.0+/NumPy 2.0+
2. **Future-Proof APIs**: All APIs used are documented as stable in current versions
3. **Test Coverage**: All modified code paths covered by tests

### Compatibility Matrix
| Component | Pandas 3.0+ | NumPy 2.0+ | Status |
|-----------|-------------|------------|---------|
| preprocess.py | ✅ | ✅ | Updated |
| correlations.py | ✅ | ✅ | Fixed |
| tcr_scoring.py | ✅ | ✅ | Updated |
| plotting.py | ✅ | ✅ | Updated |
| tcrdist/* | ✅ | ✅ | Audited |
| vectorized.py | ✅ | ✅ | Verified |

## Constraints and Assumptions

### Version Requirements
- **Pandas**: >= 3.0.0 (current: 3.0.6)
- **NumPy**: >= 2.0.0 (current: 2.5.3) 
- **Python**: >= 3.12 (fixed requirement)
- **Scikit-learn**: >= 1.8 (current: 1.9.1)

### Backward Compatibility
- **Support older versions**: No - this is a modernization to current stack
- **Migration path**: Users must update to pandas 3.0+/NumPy 2.0+ 
- **Documentation**: Provide clear upgrade instructions

### Testing Environment
- **Primary**: mamba environment with pandas 3.0.6, numpy 2.5.3
- **Validation**: Same test datasets used for vectorized-tcrdist validation
- **Benchmarking**: Compare against known-good results from pandas 2.x baseline

## Risk Assessment

### High Risk
- **Silent behavior changes**: CoW and casting changes may alter results without errors
- **Third-party dependencies**: scanpy/AnnData may have their own compatibility issues

### Medium Risk  
- **Performance impact**: Explicit operations may be slower than previous implicit ones
- **API surface**: Large codebase means many potential compatibility issues

### Mitigation Strategies
- **Comprehensive testing**: Test all workflows before/after on same data
- **Staged rollout**: Fix critical errors first, then systematic modernization
- **Version pinning**: Document exact compatible dependency versions
