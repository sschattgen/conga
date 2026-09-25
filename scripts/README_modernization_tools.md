# CoNGA Pandas 3.0/NumPy 2.0 Modernization Tools

This directory contains automated tools to modernize the CoNGA codebase for compatibility with pandas 3.0+ and NumPy 2.0+. These tools address the requirements specified in Task 2.2 of the pandas3-numpy2-modernization spec.

## Tools Overview

### 1. `modernize_numpy_dtypes.py` - NumPy Legacy Dtype Modernizer

**Purpose**: Systematically finds and replaces NumPy legacy dtype aliases that were removed in NumPy 2.0.

**What it fixes**:
- `np.float_` → `np.float64`
- `np.int_` → `np.int64` 
- `np.bool_` → `bool`
- `np.object_` → `object`
- `np.complex_` → `np.complex128`
- `np.str_` → `str`
- `np.unicode_` → `str`
- Plus `numpy.` prefixed variants

**Usage**:
```bash
# Preview changes (dry run)
python scripts/modernize_numpy_dtypes.py --dry-run

# Apply changes with backups
python scripts/modernize_numpy_dtypes.py --backup --report-file numpy_changes.txt

# Process specific directory only
python scripts/modernize_numpy_dtypes.py --target-dir conga/tcrdist
```

**Features**:
- ✅ Dry-run mode for safe preview
- ✅ Automatic backup creation (`.numpy_modernize_backup` extension)
- ✅ Comprehensive reporting with before/after context
- ✅ Regex-based pattern matching for accuracy
- ✅ Handles both `np.` and `numpy.` prefixed aliases

### 2. `audit_cow_patterns.py` - Copy-on-Write Pattern Auditor

**Purpose**: Detects pandas copy-on-write unsafe patterns that cause silent failures in pandas 3.0+.

**What it detects**:
- Chained assignment: `df[col][mask] = value`
- Query chaining: `df.query(...)[col] = value`
- Mixed indexing: `df[mask].loc[:, col] = value`
- Unsafe inplace operations on selections
- Groupby transform assignments

**Usage**:
```bash
# Audit all Python files
python scripts/audit_cow_patterns.py

# Generate detailed text report
python scripts/audit_cow_patterns.py --report-file cow_audit.txt

# Generate JSON report for programmatic processing
python scripts/audit_cow_patterns.py --output-format json --report-file cow_audit.json

# Audit specific directory
python scripts/audit_cow_patterns.py --target-dir conga/preprocess.py
```

**Risk Levels**:
- 🔴 **HIGH**: Almost certainly breaks with CoW (e.g., `df[col][mask] = value`)
- 🟡 **MEDIUM**: May break depending on context
- 🟢 **LOW**: Potentially problematic
- ℹ️ **INFO**: Worth noting but likely safe

**Note**: This tool reports many false positives for dictionary operations like `adata.uns['key']['subkey'] = value`, which are not pandas operations and are safe. Focus on patterns involving DataFrame/Series objects.

### 3. `test_modernization_results.py` - Result Validation Framework

**Purpose**: Validates that modernization changes don't alter scientific accuracy by comparing analysis results before and after modifications.

**What it validates**:
- AnnData object consistency (X matrix, obs, obsm, uns)
- TSV file numerical accuracy  
- CoNGA analysis result reproducibility
- Key output file byte-identical comparison

**Usage**:
```bash
# Create baseline from current codebase (run before modernization)
python scripts/test_modernization_results.py --create-baseline

# Validate current results against baseline (run after modernization)  
python scripts/test_modernization_results.py --validate --report-file validation_report.txt

# Compare two specific files directly
python scripts/test_modernization_results.py --compare-files before.h5ad after.h5ad
```

**Test Configurations**:
- Basic TCR-GEX correlation analysis
- TCR clumping statistical analysis
- Vectorized TCR representation validation
- Configurable numerical tolerance (default: 1e-10)

### 4. `modernize_codebase.py` - Complete Workflow Orchestrator

**Purpose**: Orchestrates all three tools in a comprehensive modernization workflow.

**Usage**:
```bash
# Complete modernization workflow (audit → modernize → validate)
python scripts/modernize_codebase.py --full-modernization

# Audit-only mode (no changes made)
python scripts/modernize_codebase.py --audit-only

# Validation-only mode
python scripts/modernize_codebase.py --validate-only

# Preview all changes (dry-run mode)
python scripts/modernize_codebase.py --full-modernization --dry-run
```

**Workflow Phases**:
1. **Audit Phase**: Detect CoW patterns and preview NumPy changes
2. **Modernization Phase**: Apply NumPy dtype fixes (if not dry-run)
3. **Validation Phase**: Create baseline and validate results

## Recommended Workflow

### Step 1: Initial Assessment
```bash
# Get overview of issues without making changes
python scripts/modernize_codebase.py --audit-only
```

### Step 2: Create Baseline
```bash
# Create baseline before any changes (important!)
python scripts/test_modernization_results.py --create-baseline
```

### Step 3: Apply NumPy Modernization
```bash
# Apply NumPy fixes with backups
python scripts/modernize_numpy_dtypes.py --backup --report-file numpy_modernization.txt
```

### Step 4: Manual CoW Fixes
Review the CoW audit results and manually fix high-risk patterns:

```python
# Change from (unsafe):
df[col][mask] = value

# To (safe):
df.loc[mask, col] = value
```

Focus on DataFrame/Series operations, ignore dictionary operations like `adata.uns[key][subkey] = value`.

### Step 5: Validate Results
```bash
# Ensure modernization doesn't change analysis results
python scripts/test_modernization_results.py --validate --report-file validation.txt
```

### Step 6: Full Workflow Validation
```bash
# Run complete validation to ensure everything works
python scripts/modernize_codebase.py --validate-only
```

## Output Files

All tools generate timestamped output files in `modernization_results/`:

- `cow_audit_YYYYMMDD_HHMMSS.txt` - Copy-on-write audit report
- `cow_audit_YYYYMMDD_HHMMSS.json` - Machine-readable CoW audit
- `numpy_modernization_YYYYMMDD_HHMMSS.txt` - NumPy modernization report
- `validation_report_YYYYMMDD_HHMMSS.txt` - Result validation report
- `modernization_summary_YYYYMMDD_HHMMSS.json` - Complete workflow summary
- `baseline_results/` - Baseline analysis outputs for comparison

## Technical Notes

### Environment Setup
All tools use the mamba environment:
```bash
mamba run -n conga-dev python scripts/[tool_name].py [args]
```

### Backup Files
- NumPy modernizer creates `.numpy_modernize_backup` files
- Always keep backups until validation passes
- Use `git status` to review changes before committing

### False Positives in CoW Audit
The CoW auditor intentionally casts a wide net and reports many false positives. Common safe patterns that are incorrectly flagged:

```python
# SAFE - Dictionary operations (not pandas)
adata.uns['conga_results'][table_tag] = results_df
overall_status['python']['compatible'] = True
neighbors['params']['metric'] = 'tcrdist'

# UNSAFE - DataFrame operations (real CoW issues)  
df['column'][mask] = value           # Use df.loc[mask, 'column'] = value
df.query('condition')['col'] = val   # Use boolean indexing instead
```

### Validation Tolerance
The result validator uses a numerical tolerance of `1e-10` by default. If legitimate differences are expected due to algorithm improvements, adjust with `--tolerance`.

### Performance Notes
- NumPy modernizer: ~1-2 seconds for full codebase
- CoW auditor: ~5-10 seconds for full codebase  
- Result validator: ~2-5 minutes per test configuration (depends on analysis complexity)

## Integration with Development Workflow

These tools integrate with the existing CoNGA development workflow:

1. **Before major changes**: Create baseline with result validator
2. **During development**: Use CoW auditor to catch unsafe patterns
3. **Before release**: Run full modernization workflow to ensure compatibility
4. **CI/CD integration**: Add validation checks to prevent pandas/numpy regressions

## Troubleshooting

### "No baseline found" Error
```bash
# Create baseline first
python scripts/test_modernization_results.py --create-baseline
```

### Validation Failures
- Check if input data files exist in `test_data/`
- Verify mamba environment is activated: `mamba activate conga-dev`
- Review detailed error messages in validation report

### Large Number of CoW Issues
- Focus on HIGH risk issues first
- Ignore dictionary operations (false positives)
- Use `--target-dir` to process files incrementally

### Permission Errors
```bash
# Make scripts executable
chmod +x scripts/modernize_*.py scripts/audit_*.py scripts/test_*.py
```

## Development Notes

### Adding New Test Configurations
Edit `test_modernization_results.py` and add to `_init_test_configs()`:

```python
configs.append({
    'name': 'my_new_test',
    'description': 'Description of what this tests',
    'script': 'scripts/run_conga.py',
    'args': ['--my_analysis'],
    'output_files': ['*.h5ad'],
    'key_outputs': ['uns.my_results']
})
```

### Extending Pattern Detection
Add new CoW patterns to `audit_cow_patterns.py` in `_init_detection_patterns()`:

```python
new_patterns = [
    {
        'regex': re.compile(r'pattern_regex'),
        'description': 'What this pattern does wrong',
        'risk_level': RiskLevel.HIGH,
        'suggestion': 'How to fix it'
    }
]
```

### Custom NumPy Patterns  
Add to `modernize_numpy_dtypes.py` in `ADDITIONAL_PATTERNS`:

```python
'custom_pattern': {
    'pattern': r'regex_pattern',
    'replacement': 'replacement_string',
    'description': 'What this fixes'
}
```

## Requirements Satisfied

This implementation satisfies the following requirements from the modernization spec:

- **SA2**: NumPy legacy alias elimination with automated replacement
- **TR12**: Comprehensive test coverage with before/after result comparison
- **CR1-CR3**: Detection and guidance for fixing critical runtime errors
- **TR1, TR11**: Copy-on-write pattern detection and modernization guidance

The tools provide both automated fixes (NumPy dtypes) and detection/guidance (CoW patterns) while ensuring scientific accuracy is maintained through comprehensive validation.