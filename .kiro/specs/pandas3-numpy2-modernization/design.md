# Pandas 3.0/NumPy 2.0 Compatibility Design

## Overview

This design addresses comprehensive modernization of the CoNGA codebase for pandas 3.0+ and NumPy 2.0+ compatibility. The approach prioritizes fixing runtime failures first, then systematically updating code patterns that may cause silent behavior changes, followed by modernizing deprecated APIs for future compatibility.

## Architecture Strategy

### Three-Phase Modernization Approach

```
Phase 1: Critical Runtime Fixes
├── Fix pandas Series indexing errors (correlations.py)
├── Replace deprecated AnnData method calls  
├── Update scanpy API usage
└── Immediate workflow restoration

Phase 2: Systematic Code Pattern Updates
├── Copy-on-write compliance audit
├── NumPy legacy alias elimination
├── String dtype handling modernization  
└── Explicit casting implementation

Phase 3: Future-Proofing and Optimization
├── Modern API adoption
├── Performance optimization
├── Compatibility diagnostics
└── Documentation updates
```

### Compatibility Layer Strategy

Rather than maintaining backward compatibility, this modernization commits fully to pandas 3.0+/NumPy 2.0+ and provides clear migration guidance for users.

## Detailed Design

### Phase 1: Critical Runtime Fixes

#### Fix 1.1: Pandas Series Indexing (correlations.py)

**Problem**: `is_mait[double_nbrs]` fails because `double_nbrs` contains positional indices but `is_mait.index` contains cell barcodes.

**Current Code Pattern**:
```python
# Line 111 in correlations.py - FAILS
mait_fraction=np.sum(is_mait[double_nbrs])/overlap

# Lines 93-94 - FAILS  
overlap_corrected = min(len(set(agroups[double_nbrs])),
                        len(set(bgroups[double_nbrs])))
```

**Solution Strategy**:
```python
# Option A: Positional indexing (if double_nbrs are positions)
mait_fraction = np.sum(is_mait.iloc[double_nbrs])/overlap

# Option B: Index alignment (if we need to map positions to barcodes)
cell_indices = adata.obs_names[double_nbrs]  
mait_fraction = np.sum(is_mait.loc[cell_indices])/overlap

# Option C: Convert to numpy array to avoid index semantics
is_mait_values = is_mait.values
mait_fraction = np.sum(is_mait_values[double_nbrs])/overlap
```

**Implementation Decision**: Use Option A (`.iloc[]`) as it preserves the intended positional semantics with minimal code change.

#### Fix 1.2: AnnData Method Deprecations

**Problem**: Multiple deprecated method calls throughout codebase.

**Systematic Replacement Pattern**:
```python
# Old → New
adata.obs_keys()     → list(adata.obs.columns) or adata.obs.columns
adata.var_keys()     → list(adata.var.columns) or adata.var.columns  
adata.uns_keys()     → list(adata.uns.keys()) or adata.uns.keys()
adata.obsm_keys()    → list(adata.obsm.keys()) or adata.obsm.keys()
adata.isview         → adata.is_view
```

**Implementation**: Create utility functions to centralize the transition:
```python
# In conga/util.py
def safe_obs_columns(adata):
    """Get obs column names in pandas 3.0+ compatible way."""
    return adata.obs.columns

def safe_uns_keys(adata):  
    """Get uns keys in AnnData compatible way."""
    return adata.uns.keys()
```

#### Fix 1.3: Scanpy API Updates

**Problem**: `sc.logging.print_versions()` is deprecated.

**Solution**: Replace with `sc.logging.print_header()` in `scripts/run_conga.py`.

### Phase 2: Systematic Pattern Updates

#### Copy-on-Write Compliance Audit

**Strategy**: Systematic review of DataFrame mutation patterns in critical files.

**Target Files**:
1. `conga/preprocess.py` - Data preprocessing and filtering
2. `conga/correlations.py` - Analysis computations  
3. `conga/tcrdist/*.py` - TCR distance calculations
4. `conga/plotting.py` - Result visualization
5. `scripts/run_conga.py` - Main pipeline

**Pattern Detection**:
```python
# Problematic patterns to find and fix:
df[col][mask] = value                    # Chained assignment
df.query('condition')[col] = value       # Chained with query
df.groupby('key').col.transform(func)    # May create copies
df[mask].loc[:, col] = value            # Mixed indexing
```

**Replacement Strategy**:
```python
# Safe patterns for copy-on-write:
df.loc[mask, col] = value               # Explicit loc indexing
df = df.assign(**{col: df[col].where(~mask, new_value)})  # Functional
with pd.option_context('mode.copy_on_write', False):     # Escape hatch
    # Legacy code block
```

#### NumPy 2.0 Legacy Alias Elimination

**Detection Strategy**: Use regex search across codebase:
```bash
grep -r "np\.\(float_\|int_\|bool_\|object_\)" conga/ scripts/
```

**Replacement Map**:
```python
np.float_   → np.float64    # or np.float32 where appropriate
np.int_     → np.int64      # or np.int32 where appropriate  
np.bool_    → bool          # Python built-in
np.object_  → object        # Python built-in
```

**Implementation**: Create automated replacement script:
```python
# scripts/modernize_numpy_dtypes.py
import re
import glob

REPLACEMENTS = {
    r'np\.float_': 'np.float64',
    r'np\.int_': 'np.int64', 
    r'np\.bool_': 'bool',
    r'np\.object_': 'object'
}

def modernize_file(filepath):
    """Apply NumPy 2.0 dtype modernization to a Python file."""
    # Implementation details...
```

#### String Dtype Handling Modernization

**Issue**: pandas 3.0 defaults string columns to `string[pyarrow]` instead of `object`.

**Audit Strategy**: 
1. Find all string column operations in V/J gene and CDR3 processing
2. Test assumptions about `.dtype == object` comparisons
3. Update to use pandas string accessors where appropriate

**Code Patterns to Update**:
```python
# Potentially problematic:
if col.dtype == object:  # May fail with string[pyarrow] dtype

# More robust:
if pd.api.types.is_string_dtype(col):  # Handles both object and string dtypes
if col.dtype.kind in ['O', 'S', 'U']:  # Multiple string representations
```

### Phase 3: Future-Proofing

#### Compatibility Diagnostics Framework

**Design**: Add startup compatibility checking to catch version issues early.

```python
# conga/compatibility.py
import pandas as pd
import numpy as np
import warnings

MIN_PANDAS_VERSION = "3.0.0"
MIN_NUMPY_VERSION = "2.0.0"

def check_environment_compatibility():
    """Check pandas/numpy versions and warn about known issues."""
    pandas_version = pd.__version__
    numpy_version = np.__version__
    
    # Version checks
    if Version(pandas_version) < Version(MIN_PANDAS_VERSION):
        warnings.warn(f"pandas {pandas_version} < {MIN_PANDAS_VERSION}: upgrade recommended")
    
    # Known issue detection  
    if Version(pandas_version) >= Version("3.0.0"):
        # Enable copy-on-write checking if available
        pd.options.mode.copy_on_write = True
        
    return True  # or raise if critical incompatibilities found
```

#### Modern API Adoption

**Strategy**: Replace deprecated patterns with modern equivalents proactively.

**Examples**:
```python
# Old scanpy patterns → Modern equivalents
sc.pp.normalize_per_cell()  →  sc.pp.normalize_total()
sc.tl.louvain()            →  sc.tl.leiden()

# Old pandas patterns → Modern equivalents  
df.append()                →  pd.concat()
df.iteritems()             →  df.items()
```

## Testing and Validation Strategy

### Regression Testing Framework

**Approach**: Capture known-good outputs from pandas 2.x environment as baselines, then validate that pandas 3.0+ produces identical results.

```python
# tests/test_pandas3_compatibility.py
class TestPandas3Compatibility:
    
    def test_correlation_analysis_identical_results(self):
        """Verify correlations.py produces identical results in pandas 3.0+."""
        # Load baseline results from pandas 2.x
        # Run same analysis in current environment  
        # Assert byte-identical outputs
        
    def test_preprocessing_deterministic(self):
        """Verify preprocessing steps are deterministic."""
        # Run preprocessing twice with same seed
        # Assert identical AnnData objects
```

### Compatibility Test Matrix

| Test Category | Scope | Success Criteria |
|---------------|-------|------------------|
| Runtime Stability | All workflows complete without pandas/numpy errors | ✅ PASS |
| Result Identity | Outputs identical to baseline (fixed seed) | ✅ PASS |
| Performance | No >20% performance regression | ✅ PASS |  
| API Modernization | No deprecation warnings | ✅ PASS |

### Validation Datasets

**Primary**: Use existing test datasets from vectorized-tcrdist validation:
- `test_data/SC5v2_humanPBMCs_5Kcells_*` (10x human PBMC)
- Known-good baseline outputs for comparison

**Secondary**: Generate synthetic test cases for edge conditions:
- Empty DataFrames, single-cell datasets, missing values
- Various string encodings, mixed dtypes

## Implementation Plan

### Phase 1: Critical Fixes (Week 1)
1. **Fix correlations.py indexing** - Replace `series[int_array]` with `.iloc[]`
2. **Update AnnData method calls** - Replace deprecated methods systematically
3. **Fix scanpy deprecations** - Update to modern API calls
4. **Validate basic workflows** - Ensure `run_conga.py` completes

### Phase 2: Systematic Updates (Week 2)
1. **Copy-on-write audit** - Review and fix DataFrame mutation patterns
2. **NumPy legacy elimination** - Replace all `np.float_`/`np.int_` etc. usage
3. **String dtype modernization** - Update string column handling
4. **Comprehensive testing** - Validate all modules and workflows

### Phase 3: Future-Proofing (Week 3) 
1. **Compatibility diagnostics** - Add version checking and warnings
2. **Performance optimization** - Optimize any performance regressions
3. **Documentation updates** - Update installation and compatibility docs
4. **Final validation** - Complete test suite against pandas 3.0+/NumPy 2.0+

## Risk Mitigation

### Silent Behavior Changes
**Risk**: pandas CoW and NumPy casting changes alter results without errors.
**Mitigation**: Comprehensive before/after testing with deterministic seeds.

### Third-Party Dependencies  
**Risk**: scanpy/AnnData compatibility issues beyond our control.
**Mitigation**: Version pinning and compatibility testing in CI environment.

### Performance Regressions
**Risk**: Explicit operations slower than previous implicit behavior.  
**Mitigation**: Performance benchmarking and optimization of critical paths.

## Success Metrics

1. **Zero Runtime Errors**: All CoNGA workflows complete successfully
2. **Deterministic Results**: Identical outputs with fixed random seeds  
3. **Warning-Free Execution**: No deprecation or compatibility warnings
4. **Performance Maintained**: <10% regression on benchmark datasets
5. **Future Compatibility**: Uses only stable, documented APIs

This comprehensive modernization ensures CoNGA works reliably with the current scientific Python stack while positioning it for continued compatibility with future versions.
