# CoNGA Test Suite

This directory contains the test suite for CoNGA, covering the vectorized
TCRdist + FAISS acceleration feature, batch integration, Metaconga matching,
pipeline reproducibility, and general CLI/organism/plotting correctness.

Run everything with:

```bash
mamba run -n conga-dev pytest tests/
```

## Test Structure

Tests are organized by feature area rather than by phase. Supporting docs
(`README_error_tests.md`, `README_error_conditions_I2_2.md`) record the
original requirement-traceability notes for the error-handling test work and
are kept for historical reference; this file is the canonical index.

### Fixtures and test infrastructure

- **`conftest.py`** — Core fixtures: human TCR data sampled from the bundled
  database, synthetic mouse/rhesus clonotype generation, edge cases, encoding
  configs, and mixed input-format fixtures. Also registers `test_config.py`
  as a plugin (see below) and defines the `vectorized` / `accuracy` /
  `property` markers, auto-applying them by filename/test-name pattern.
- **`test_config.py`** — Defines shared tolerances/timeouts (`TEST_CONFIG`),
  dataset scenario specs (`TEST_DATASETS`), and additional fixtures
  (`minimal_adata`, `minimal_clones`, `edge_case_clones`,
  `invalid_gene_clones`, `performance_adata`) backed by files in
  `tests/fixtures/`. It's a plain module, not a `conftest.py`, so it's wired
  in explicitly via `conftest.py`'s `pytest_plugins`.
- **`fixtures/`** — Static fixture data (`minimal_adata.h5ad`,
  `minimal_clones.tsv`, `edge_case_clones.tsv`, `invalid_gene_clones.tsv`) and
  `test_data_generator.py`, a `TestDataGenerator` helper class for building
  synthetic TCR/AnnData test data on demand.
- **`test_fixtures_validation.py`** — Validates that the `conftest.py`
  fixtures themselves produce well-formed, non-empty, correctly-columned
  data.
- **`test_example_usage.py`** — Demonstrates how to consume the fixtures
  (tuple format, dict format, DataFrame format, different sizes); documents
  usage rather than testing production code.

### Vectorized TCRdist core

- **`test_vectorized_b1_1.py`** — Amino-acid dissimilarity matrix and
  MDS-based embedding (`symbol_dissimilarity_matrix`, `aa_embedding`):
  determinism, cross-process reproducibility, gap-penalty and caching
  behavior.
- **`test_vectorized_errors.py`** — Full error-handling table for
  `encode_tcrs` / `germline_code_table`: invalid organism, invalid V gene,
  invalid CDR3, flag conflicts, binary-dependency errors.
- **`test_representation_selection.py`** — Exhaustively tests
  `conga.preprocess.resolve_tcr_representation`'s three-way selection logic
  (vectorized vs. KernelPCA vs. exact TCRdist) against every row of the
  organism/count/override selection table.
- **`test_tier3_validation_warning.py`** — Confirms `encode_tcrs` emits a
  one-time-per-process warning for Tier-3 (synthetic-validated) organisms,
  and never warns for Tier-1 organisms (`human`/`mouse`/`rhesus`).
- **`test_accuracy_report_dedup.py`** — Regression test (AST-based) that
  `conga/tcrdist/vectorized.py` has exactly one top-level `accuracy_report`
  function, guarding against a previously-reintroduced dead duplicate.
- **`test_property_validation_corpus_gene_ids.py`** — Hypothesis property
  test asserting every V/J gene id emitted by the human/mouse/rhesus CDR3
  corpus loaders and the synthetic CDR3 generator resolves to a valid key in
  `all_genes[organism]`.
- **`test_supported_organisms_results_table_agreement.py`** — Parses the
  accuracy-gate results table out of
  `.kiro/specs/tcrdist-db-update/design.md` and asserts exact set equality
  with `vectorized.SUPPORTED_ORGANISMS`, so nothing is marked "supported"
  without recorded PASS evidence (and vice versa).
- **`test_gene_database_organism_coverage.py`** — Confirms `all_genes`
  contains every organism string present in `combo_xcr_2026-08-06.tsv`,
  including newly-supported and chain-completeness-excluded organisms.

### Accuracy and benchmarking infrastructure

- **`test_accuracy_validation.py`** — Tests the `conga.accuracy_validation`
  module itself (`AccuracyValidator`, `EdgeCaseGenerator`,
  `quick_accuracy_check`, `production_validation_suite`), including
  FAISS-conditional skips and an `integration`-marked production suite.
- **`test_comprehensive_validation.py`** — Broad end-to-end suite covering
  vectorized TCRdist + FAISS: encoding determinism, accuracy vs. exact
  TCRdist, performance improvements, backend selection, and workflow
  integration.
- **`test_benchmark.py`** — Validates the `conga.benchmark` module
  (`PerformanceSuite`, `DatasetGenerator`, `quick_gex_benchmark`,
  `quick_tcr_benchmark`, `comprehensive_scaling_benchmark`) used for
  FAISS-vs-sklearn performance comparisons.
- **`test_error_conditions.py`** — Comprehensive error-condition suite
  covering organism/V-gene/CDR3 validation, CLI flag conflicts, FAISS
  backend failures/fallback, binary-dependency errors, and timeouts.

### FAISS neighbor search

- **`test_faiss_error_handling.py`** — Production-grade error handling in
  `conga.neighbors` (`FaissNeighborSearcher` and its `FaissError` /
  `FaissGpuMemoryError` / `FaissCudaError` hierarchy): input validation and
  graceful fallback.
- **`test_faiss_group_exclusion.py`** — Regression test for a bug where
  `FaissNeighborSearcher`'s TCR group-exclusion padded shortfalls with `-1`
  instead of re-querying, which crashed `correlations._make_csr_nbrs` for
  both GEX and TCR neighbor search.

### Batch integration

- **`test_batch_integration.py`** — `conga.preprocess._validate_batch_key`
  (column presence, ≥2 distinct values) plus the Harmony and scVI tracks of
  `batch_integration()`, including a `slow`-marked end-to-end scVI test.
- **`test_run_conga_cli.py`** — `--batch_key` / `--batch_integration_method`
  CLI flags on `scripts/run_conga.py`: flag pairing, mutual exclusion with
  `--force_variable_genes`, unsupported-method validation, and an
  `integration`-marked smoke test reaching `batch_integration()`.
- **`test_fixed_hvg_pathway.py`** — Three-state HVG-selection guard in
  `filter_normalize_and_hvg` (fixed gene list via `force_variable_genes`,
  caller-preset `highly_variable` mask, or default scanpy pathway).
- **`test_counts_layer.py`** — `filter_normalize_and_hvg` unconditionally
  captures a pre-normalization counts matrix into `adata.layers['counts']`,
  correctly narrowed when antibody/protein-capture features are removed.

### Pipeline reproducibility (random-seed wiring)

- **`test_pipeline_reproducibility.py`** — Capstone end-to-end test: runs
  `scripts/run_conga.py` twice with the same `--random_seed` (vectorized
  path and `--no_kpca` exact-TCRdist path) and asserts identical
  `_final.h5ad` output.
- **`test_pipeline_reproducibility_component1.py`** — `random_seed` wiring
  through `cluster_and_tsne_and_umap`'s 6 stochastic call sites (PCA,
  neighbors, UMAP x2, Louvain, Leiden x2).
- **`test_pipeline_reproducibility_component2.py`** — Same pattern for
  `calc_tcrdist_nbrs_umap_clusters_cpp`'s 6 stochastic call sites.
- **`test_pipeline_reproducibility_component3.py`** — Same pattern for
  `reduce_to_single_cell_per_clone`'s single stochastic call site.
- **`test_pipeline_reproducibility_component4.py`** — Same pattern for
  `calc_X_pca_gex_including_protein_features`'s one in-scope seeded call
  site, confirming two other calls are intentionally left unseeded.
- **`test_pipeline_reproducibility_component5.py`** — Same pattern for the
  two `KernelPCA(kernel='precomputed', ...)` call sites in
  `make_tcrdist_kernel_pcs_file_from_clones_file[_V2]`.
- **`test_pipeline_reproducibility_component6.py`** — `scripts/run_conga.py`
  CLI wiring of `args.random_seed` into all 7 call sites of the three seeded
  preprocess functions, checked by exact occurrence count to catch partial
  wiring.

### Metaconga matching

- **`test_metaconga_match_import.py`** — `import conga.metaconga_match`
  succeeds (module-level data loading from `conga/data/metaconga/` works)
  and its four public pipeline functions are resolvable.
- **`test_metaconga_match_data_files.py`** — Parametrized check that all 11
  bundled Metaconga reference TSVs exist under `conga/data/metaconga/`.
- **`test_metaconga_match_tags.py`** — All 5 Metaconga tag constants exist in
  `conga/tags.py` with correct names/values.
- **`test_metaconga_match_cli_flags.py`** — Argparse smoke tests
  (subprocess-based) for `--match_metaconga_aaclusters` /
  `--match_metaconga_clumps` on `scripts/run_conga.py`.
- **`test_metaconga_match_validation.py`** — Subprocess-based tests of the
  early cross-flag validation for metaconga-match flags (e.g. CD4/CD8
  pairing requirements, case-insensitivity).
- **`test_metaconga_match_dispatch.py`** — End-to-end integration tests
  invoking the AACluster and Clump pipeline functions against a synthetic
  human AnnData fixture, asserting no exceptions (not statistical
  correctness).

### CLI / organism validation

- **`test_organism_cli_choices.py`** — Subprocess argparse smoke tests that
  `--organism` accepts newly-supported organisms (`cat`, `dog_gd`,
  `rabbit_ig`, `sheep`) in both `run_conga.py` and `setup_10x_for_conga.py`.
- **`test_get_vdj_type.py`** — `conga.util.get_vdj_type` returns correct VDJ
  type constants for all supported organisms and raises `ValueError` (not
  `KeyError`) for chain-completeness-excluded organisms.
- **`test_get_ab_from_10x_chain.py`** — Same pattern for
  `conga.tcrdist.make_10x_clones_file.get_ab_from_10x_chain`'s 10x-chain to
  Chain_Label mapping.
- **`test_devel_cd4_cd8_split.py`** — Regression test for
  `conga.devel.split_into_cd4_and_cd8_subsets`, which previously called the
  removed `AnnData.concatenate()` (now `anndata.concat()`) and had a
  shadowed-variable bug.

### Plotting

- **`test_plotting_logo_genes.py`** — `conga.plotting.make_logo_plots`'s
  marker-gene lookups degrade gracefully (empty list, no `KeyError`) for
  organisms absent from `default_logo_genes` / `default_gex_header_genes`,
  while preserving existing `human` behavior.

### Portability

- **`test_portable_imports.py`** — Vectorized-TCRdist and FAISS modules
  import cleanly in isolated subprocess environments without filesystem
  assertion errors, with clear messages for missing dependencies.

### Standalone script (not pytest-collected as a suite)

- **`run_all_tests.py`** — A master runner invoked directly:
  `python tests/run_all_tests.py [--smoke] [--categories fixtures validation errors] [--quiet]`.
  It wraps `pytest.main()` over a small category map (`validation` →
  `test_comprehensive_validation.py`, `errors` → `test_error_conditions.py`)
  or, with `--smoke`, runs a quick in-process check (basic `encode_tcrs`
  call, FAISS backend detection, a neighbor search).
  **Known issue:** its `fixtures` category points at
  `test_data_generator.py` in `tests/` directly, but that file actually
  lives in `tests/fixtures/test_data_generator.py`. The script checks for
  existence and only prints a warning, so `--categories fixtures` silently
  runs nothing. Prefer plain `pytest tests/` over this script.

## Pytest Markers

Registered markers (enforced via `--strict-markers` in `pyproject.toml`):

| Marker | Registered in | Meaning |
|---|---|---|
| `slow` | `pyproject.toml` | Long-running test; deselect with `-m "not slow"` |
| `integration` | `pyproject.toml` | End-to-end integration test |
| `vectorized` | `conftest.py` | Part of the vectorized TCRdist feature (auto-applied to files with "vectorized" in the name) |
| `accuracy` | `conftest.py` | Validates accuracy thresholds (auto-applies `slow` too) |
| `property` | `conftest.py` | Property-based test (Hypothesis) |

FAISS-dependent tests don't use a dedicated marker; they use
`pytest.mark.skipif(not FAISS_AVAILABLE, ...)` so they skip cleanly on
machines without FAISS installed.

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
mamba run -n conga-dev pytest tests/ -m property

# Accuracy validation (slow)
mamba run -n conga-dev pytest tests/ -m accuracy

# Integration tests
mamba run -n conga-dev pytest tests/ -m integration

# Metaconga matching only
mamba run -n conga-dev pytest tests/test_metaconga_match_*.py

# Pipeline reproducibility (random seed wiring) only
mamba run -n conga-dev pytest tests/test_pipeline_reproducibility*.py
```

### With coverage
```bash
mamba run -n conga-dev pytest tests/ --cov=conga --cov-report=html
```

## Test Data Sources

### Human Data
Real human TCR data from
`conga/data/new_paired_tcr_db_for_matching_nr.tsv`: ~4124 original paired
TCRs, filtered to valid clonotypes (standard amino acids, reasonable
lengths, no duplicates), providing `va`, `vb`, `cdr3a`, `cdr3b` columns.

### Synthetic Data
Mouse and rhesus clonotypes generated with:
- V genes sampled from the actual gene database (`combo_xcr.tsv`)
- CDR3 sequences with realistic lengths (8-18 amino acids) and conventional
  `C...F` termini
- Seeded random generation for reproducibility

### Static fixtures
`tests/fixtures/` holds small static files (`minimal_adata.h5ad`,
`minimal_clones.tsv`, `edge_case_clones.tsv`, `invalid_gene_clones.tsv`) plus
`test_data_generator.py`, a generator class for building ad hoc synthetic
AnnData/TCR data in tests that need more than the static files provide.

## Reproducibility

Fixtures and seeded pipeline paths use a fixed random seed
(`TEST_RANDOM_SEED = 42` in `conftest.py`, matching
`util.DEFAULT_RANDOM_SEED`) to ensure:
- Identical test results across runs
- Reproducible synthetic data generation
- Deterministic sampling from the human database
- Cross-process consistency (validated directly by
  `test_pipeline_reproducibility*.py` and `test_portable_imports.py`)

## Adding New Tests

1. Use appropriate fixtures based on test needs:
   - `*_small` for fast unit tests
   - `*_medium` for regular functionality tests
   - `*_large` for accuracy and performance tests

2. Mark tests appropriately:
   ```python
   @pytest.mark.vectorized
   @pytest.mark.slow        # for tests taking >1 second
   @pytest.mark.property    # for property-based tests
   @pytest.mark.integration # for end-to-end tests
   ```
   Markers must be one of the registered set above — `--strict-markers` will
   fail the run on typos or new unregistered markers.

3. Follow naming conventions:
   - `test_*.py` for test files
   - `test_*` for test functions
   - Descriptive names indicating what's being tested; files with
     "vectorized" or "accuracy" in the name get those markers auto-applied,
     so don't use those words incidentally in unrelated test file names.

4. Import fixtures from conftest:
   ```python
   def test_my_feature(self, human_clonotypes_small, supported_organism):
       # Test implementation
   ```

5. If adding a new feature area (e.g. a new module under `conga/`), add a
   new subsection to this README under "Test Structure" rather than letting
   new files go undocumented — this file previously fell out of sync with
   the actual suite and listed files "to be added" that had already existed
   for some time.
