# CoNGA Test Suite

This directory contains the comprehensive test suite for the CoNGA vectorized TCRdist feature.

## Test Structure

- **`conftest.py`** - Test fixtures and configuration for all tests
- **`test_fixtures_validation.py`** - Validates that all fixtures work correctly
- **`test_example_usage.py`** - Demonstrates how to use the fixtures
- **`test_vectorized_*.py`** - Core vectorized TCRdist functionality tests (to be added)
- **`test_property_*.py`** - Property-based tests using Hypothesis (to be added)
- **`test_integration_*.py`** - End-to-end integration tests (to be added)

## Test Fixtures

The test fixtures provide deterministic, reproducible data for testing:

### Human TCR Data (Real)
- **`human_tcr_data`** - Complete filtered human TCR database (4000+ clonotypes)
- **`human_clonotypes_small`** - 50 human clonotypes for fast tests
- **`human_clonotypes_medium`** - 300 human clonotypes for regular tests  
- **`human_clonotypes_large`** - 1000 human clonotypes for accuracy tests

### Synthetic Data
- **`mouse_clonotypes_*`** - Synthetic mouse data (small/medium/large)
- **`rhesus_clonotypes_*`** - Synthetic rhesus data (small/medium/large)
- **`edge_case_clonotypes`** - Edge cases for boundary testing

### Configuration and Format Testing
- **`supported_organism`** - Parametrized fixture for all organisms
- **`encoding_configs`** - Various encoding configurations
- **`mixed_format_data`** - Same data in different input formats

## Running Tests

### All tests
```bash
mamba run -n conga-dev pytest tests/
```

### Specific test categories
```bash
# Fixture validation tests
mamba run -n conga-dev pytest tests/test_fixtures_validation.py

# Fast tests only  
mamba run -n conga-dev pytest tests/ -m "not slow"

# Vectorized TCRdist tests
mamba run -n conga-dev pytest tests/ -m vectorized

# Property-based tests
mamba run -n conda-dev pytest tests/ -m property

# Accuracy validation (slow)
mamba run -n conga-dev pytest tests/ -m accuracy
```

### With coverage
```bash
mamba run -n conga-dev pytest tests/ --cov=conga --cov-report=html
```

## Test Data Sources

### Human Data
Real human TCR data from `/conga/data/new_paired_tcr_db_for_matching_nr.tsv`:
- 4124 original paired TCRs from multiple datasets
- Filtered to 4056 valid clonotypes (standard amino acids, reasonable lengths, no duplicates)
- Contains va, vb, cdr3a, cdr3b columns as required

### Synthetic Data
Mouse and rhesus clonotypes generated using:
- V genes sampled from actual gene database (combo_xcr.tsv)
- CDR3 sequences with realistic lengths (8-18 amino acids)  
- Conventional C...F termini patterns
- Seeded random generation for reproducibility

## Reproducibility

All fixtures use a fixed random seed (`TEST_RANDOM_SEED = 42`) to ensure:
- Identical test results across runs
- Reproducible synthetic data generation
- Deterministic sampling from human database
- Cross-process consistency

## Key Features

1. **Deterministic** - All fixtures produce identical results with fixed seeds
2. **Comprehensive** - Covers all supported organisms and edge cases
3. **Flexible** - Multiple data formats (tuples, DataFrames, custom columns)
4. **Scalable** - Small/medium/large datasets for different test needs
5. **Real + Synthetic** - Combines real human data with generated test cases

## Requirements Validated

This test infrastructure validates:
- **Requirement 10.1** - Pytest test infrastructure setup
- **Requirement 10.5** - Small paired clonotype fixture for supported organisms

The fixtures enable comprehensive testing of:
- Encoding accuracy and reproducibility
- Input format flexibility  
- Edge case handling
- Cross-organism compatibility
- Performance at different scales

## Adding New Tests

When adding new tests:

1. Use appropriate fixtures based on test needs:
   - `*_small` for fast unit tests
   - `*_medium` for regular functionality tests
   - `*_large` for accuracy and performance tests

2. Mark tests appropriately:
   ```python
   @pytest.mark.vectorized
   @pytest.mark.slow  # for tests taking >1 second
   @pytest.mark.property  # for property-based tests
   ```

3. Follow the naming conventions:
   - `test_*.py` for test files
   - `test_*` for test functions
   - Use descriptive names that indicate what's being tested

4. Import fixtures from conftest:
   ```python
   def test_my_feature(self, human_clonotypes_small, supported_organism):
       # Test implementation
   ```