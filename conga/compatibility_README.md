# CoNGA Compatibility Module

## Overview

The `conga.compatibility` module provides utilities for checking pandas 3.0+ and NumPy 2.0+ compatibility, detecting environment issues, and accessing AnnData methods safely across different package versions.

## Key Features

- **Environment Validation**: Comprehensive checking of pandas/NumPy versions and compatibility issues
- **Safe AnnData Access**: Utility functions that work across different AnnData versions  
- **Copy-on-Write Detection**: Automatic detection and configuration for pandas 3.0+ copy-on-write semantics
- **NumPy 2.0 Support**: Handles removed legacy aliases and new casting rules
- **Startup Diagnostics**: Automatic compatibility checking when importing CoNGA

## Quick Start

```python
import conga

# Check full environment compatibility
status = conga.check_environment_compatibility()
print(f"Environment compatible: {status['compatible']}")

# Use safe AnnData access functions
obs_columns = conga.safe_obs_columns(adata)
uns_keys = conga.safe_uns_keys(adata) 
is_view = conga.safe_is_view(adata)
```

## Main Functions

### Environment Checking

#### `check_environment_compatibility(verbose=True)`
Performs comprehensive compatibility validation for the CoNGA environment.

**Parameters:**
- `verbose` (bool): Whether to print detailed compatibility report

**Returns:**
- `dict`: Complete compatibility status with warnings and errors

**Example:**
```python
import conga.compatibility as compat

# Full check with detailed output
status = compat.check_environment_compatibility(verbose=True)

# Silent check for programmatic use
status = compat.check_environment_compatibility(verbose=False)
if not status['compatible']:
    print("Environment has compatibility issues!")
```

### Safe AnnData Access

These functions provide safe access to AnnData methods across different package versions:

#### `safe_obs_columns(adata)`
Get observation (cell) metadata column names safely.

#### `safe_var_columns(adata)` 
Get variable (gene) metadata column names safely.

#### `safe_uns_keys(adata)`
Get unstructured metadata keys safely.

#### `safe_obsm_keys(adata)`
Get observation matrix keys safely.

#### `safe_is_view(adata)`
Check if AnnData object is a view safely.

**Example:**
```python
import conga
import anndata as ad

adata = ad.read_h5ad('data.h5ad')

# Safe access across AnnData versions
cell_metadata_cols = conga.safe_obs_columns(adata)
gene_metadata_cols = conga.safe_var_columns(adata) 
analysis_results = conga.safe_uns_keys(adata)

# Check if it's a view before modifications
if not conga.safe_is_view(adata):
    adata.obs['new_column'] = values
```

### Version Checking

#### `check_pandas_compatibility()`
Detailed pandas version and configuration checking.

#### `check_numpy_compatibility()`
Detailed NumPy version and feature checking.

### String Validation

#### `validate_string_dtype(series, column_name="column")`
Validates pandas Series contains string data in a pandas 3.0+ compatible way.

**Example:**
```python
import conga.compatibility as compat

# Validate V gene column contains strings
compat.validate_string_dtype(adata.obs['v_gene'], 'v_gene')
```

## Version Requirements

The compatibility module enforces these minimum versions:

- **Python**: >= 3.12.0
- **pandas**: >= 3.0.0  
- **NumPy**: >= 2.0.0
- **scanpy**: >= 1.9.0
- **AnnData**: >= 0.9.0

## Compatibility Features

### Pandas 3.0+ Support

- **Copy-on-Write**: Automatically enables pandas copy-on-write mode for safe DataFrame operations
- **String Dtypes**: Handles new string dtype defaults in pandas 3.0+
- **Deprecated Methods**: Provides safe access to replaced AnnData methods

### NumPy 2.0+ Support  

- **Legacy Aliases**: Detects removal of `np.float_`, `np.int_`, etc.
- **NEP 50 Casting**: Warns about new numeric casting behavior
- **API Changes**: Handles privatization of `numpy.core` module

## Error Handling

The module uses a structured approach to compatibility issues:

- **Critical Errors**: Raise `CompatibilityError` for environment issues that prevent CoNGA from running
- **Warnings**: Use `CompatibilityWarning` for non-critical issues that may affect behavior
- **Graceful Fallbacks**: Provide alternative implementations for deprecated methods

## Integration with CoNGA

The compatibility module is automatically loaded when importing CoNGA and performs basic compatibility checking. For full integration:

```python
# In run_conga.py or other scripts
import conga

# Check environment at startup
try:
    status = conga.check_environment_compatibility(verbose=True)
    if not status['compatible']:
        print("Warning: Environment compatibility issues detected")
        for warning in status['warnings']:
            print(f"  - {warning}")
except conga.compatibility.CompatibilityError as e:
    print(f"Critical compatibility error: {e}")
    sys.exit(1)

# Use safe functions throughout codebase
def process_data(adata):
    # Safe AnnData access
    cell_types = adata.obs.columns if 'cell_type' in conga.safe_obs_columns(adata) else []
    
    # Safe string validation
    if 'v_gene' in conga.safe_obs_columns(adata):
        conga.compatibility.validate_string_dtype(adata.obs['v_gene'], 'v_gene')
```

## Best Practices

1. **Always use safe functions** when accessing AnnData methods that may be deprecated
2. **Check environment compatibility** at the start of analysis workflows  
3. **Handle warnings appropriately** - compatibility warnings indicate potential issues
4. **Use explicit dtypes** instead of relying on NumPy legacy aliases
5. **Test with fixed seeds** to ensure reproducible results across pandas/NumPy versions

## Troubleshooting

### Common Issues

**"Copy-on-write enabled" warning**: This is expected with pandas 3.0+ and ensures safe DataFrame operations.

**"Legacy aliases removed" warning**: Update code to use explicit dtypes like `np.float64` instead of `np.float_`.

**"NEP 50 casting rules" warning**: Verify that numeric operations produce expected dtypes by adding explicit `.astype()` calls.

### Getting Help

For compatibility issues:
1. Run `conga.check_environment_compatibility(verbose=True)` to see detailed status
2. Check warnings and errors for specific guidance
3. Consult the CoNGA documentation for upgrade instructions
4. Report persistent issues on the CoNGA GitHub repository
