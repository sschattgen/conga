# Error Condition Test Coverage

This document summarizes the comprehensive error condition tests implemented for Task 11.2 of the vectorized TCRdist feature.

## Files Created

- `tests/test_vectorized_errors.py` - Primary error condition tests
- `tests/test_representation_selection.py` - Representation selection error tests

## Coverage Summary

### Requirements Validated: 10.2
*Every error condition from Requirements 3, 4, and 8 as specified in the Error Handling table*

## Test Coverage by Requirement Category

### Organism Validation Errors (Requirements 3.3, 3.4)

**Test Class: `TestOrganismValidationErrors`**

✅ **Unsupported organism error (Requirement 3.3)**
- Tests gamma-delta organisms (`human_gd`, `mouse_gd`)
- Tests Ig organisms (`human_ig`, `mouse_ig`) 
- Tests completely unknown organisms
- Verifies error message contains rejected organism and alternative paths

✅ **Organism/chain with no V records (Requirement 3.4)**
- Tests error when Gene_Database has no records for organism/chain combination
- Verifies error message names organism and chain

### V Gene Validation Errors (Requirement 4.4)

**Test Class: `TestVGeneValidationErrors`**

✅ **Missing V gene in Gene_Database**
- Tests with invalid V gene identifiers
- Verifies error message contains offending gene identifier
- Verifies error message includes count of affected clonotypes
- Tests multiple missing genes to verify count accuracy

### CDR3 Validation Errors (Requirements 4.5, 4.6, 4.7)

**Test Class: `TestCDR3ValidationErrors`**

✅ **CDR3 contains non-standard residue (Requirement 4.5)**
- Tests invalid characters: `X`, `B`, `Z`, numbers, dashes, spaces
- Verifies error message contains the offending CDR3 sequence

✅ **CDR3 too short (Requirement 4.6)**
- Tests CDR3s shorter than `n_trim + c_trim + 1`
- Tests with default and custom trim parameters
- Verifies error message contains CDR3 and trim values

✅ **CDR3 too long warning (Requirement 4.7)**
- Verifies long CDR3s generate WARNING, not ERROR
- Confirms encoding succeeds with overly long CDR3s
- Checks warning message mentions count or sequences

### Validation Ordering (Requirement 4.8)

**Test Class: `TestValidationOrderingError`**

✅ **Validation before allocation**
- Verifies all validation completes before memory allocation
- Uses mocking to ensure `numpy.hstack` is never called on validation failure

### AnnData Storage Errors (Requirement 7.5)

**Test Class: `TestAnnDataStorageErrors`**

✅ **X_vec_tcr overwrite warning**
- Tests warning when `X_vec_tcr` key already exists
- Verifies warning message mentions key and overwrite

### Representation Selection Errors (Requirements 8.10, 8.11)

**Test Classes: `TestRepresentationSelectionErrors`, `TestSelectionTableCoverage`**

✅ **KernelPCA override above limit (Requirement 8.10)**
- Tests error when KernelPCA requested above `KPCA_REDUCTION_LIMIT`
- Tests with default and custom limits
- Verifies error message contains observation count, limit, and parameter name

✅ **Both overrides requested (Requirement 8.11)**
- Tests error when both KernelPCA and exact overrides requested
- Verifies error message names both override types

✅ **Complete selection table coverage**
- Tests every row of the Requirements 8 selection table
- Covers all organism/count/override combinations
- Verifies correct `TcrRepresentation` values returned

### Binary Dependency Errors (Requirements 8.13, 8.14)

**Test Class: `TestBinaryDependencyErrors`**

✅ **Exact path binaries missing warning (Requirement 8.13)**
- Tests warning when exact path used but `tcrdist_cpp` binaries unavailable
- Verifies fallback to Python implementation

✅ **Exact path projection needs binaries (Requirement 8.14)**
- Tests detection of missing binaries for projection/clustering operations
- Verifies appropriate error conditions are detected

### Flag Conflict Errors (Requirements 8.19-8.23)

**Test Class: `TestFlagConflictErrors`**

✅ **CLI flag conflicts**
- `--use_kpca_tcrdist` with `--no_kpca` (Requirement 8.19)
- `--use_kpca_tcrdist` with `--use_exact_tcrdist_nbrs` (Requirement 8.19)
- `--use_kpca_tcrdist` with encoding flags (Requirement 8.20)
- Exact path flags with encoding flags (Requirement 8.21)
- Encoding flags with unsupported organisms (Requirement 8.22)
- `--use_kpca_tcrdist` with N >= limit (Requirement 8.23)

### Edge Cases and Boundary Conditions

**Test Class: `TestEdgeCases`**

✅ **Additional error scenarios**
- Empty clonotype list handling
- Malformed input structure detection
- None values in input validation

## Test Infrastructure

### Helper Functions

- `get_valid_genes()` - Provides valid V and J gene names for testing
- `VALID_GENES` constant - Cached valid gene names for test consistency

### Mocking Strategy

- Uses `unittest.mock.patch` for simulating missing dependencies
- Mocks filesystem and database access where appropriate
- Preserves real validation logic while controlling external dependencies

### Test Execution

```bash
# Run all error condition tests
mamba run -n conga-dev python -m pytest tests/test_vectorized_errors.py tests/test_representation_selection.py

# Run specific error category
mamba run -n conga-dev python -m pytest tests/test_vectorized_errors.py::TestCDR3ValidationErrors -v

# Run with coverage
mamba run -n conga-dev python -m pytest tests/test_vectorized_errors.py --cov=conga
```

## Key Design Decisions

### Realistic Test Data

- Uses actual gene names from the Gene_Database rather than synthetic names
- Employs realistic CDR3 sequences and lengths
- Tests with multiple organisms to ensure broad coverage

### Error Message Validation

- Validates that error messages contain essential information
- Checks for presence of problematic values (gene names, CDR3 sequences)
- Verifies guidance information (alternative paths, parameter names)

### Logging vs Exception Testing

- Distinguishes between conditions that should raise exceptions vs log warnings
- Tests appropriate log levels (WARNING, INFO, ERROR)
- Captures and validates log message content

### Mock Usage

- Minimal mocking to preserve real validation logic
- Strategic mocking for external dependencies (file system, compiled binaries)
- Mock verification to ensure validation ordering

## Test Results

**Total Tests: 53**
- `test_vectorized_errors.py`: 27 tests
- `test_representation_selection.py`: 26 tests

**Status: ✅ All tests passing**

This comprehensive test suite ensures robust error handling across all specified requirements, providing clear error messages and appropriate error types for all validation failure modes.