# Implementation Plan: Vectorized TCRdist + FAISS Acceleration

## Overview

This implementation delivers two complementary performance optimizations for CoNGA:

1. **Vectorized TCRdist**: Replaces CoNGA's default TCR representation for alpha-beta receptors with a vectorized approach that avoids quadratic memory usage. Encodes each TCR as a fixed-length vector enabling neighbor search without materializing dense N×N matrices.

2. **FAISS Acceleration**: Replaces standard scipy/sklearn neighbor calculations with FAISS for massive performance improvements on GEX data, supporting both CPU and GPU backends with graceful fallback.

Together these optimizations target 10-100x speedup and >50% memory reduction on large datasets while maintaining backward compatibility and accuracy.

## Tasks

### Phase A: Foundation and Infrastructure (Parallel)

- [x] A1. Set up portable module foundation and fix import blockers
  - [x] A1.1 Create `conga/tcrdist/vectorized.py` module with portable initialization
    - Create new module with lazy imports to avoid filesystem assertions at module scope
    - Implement module-level constants and EncodingConfig dataclass
    - Add module docstring documenting accuracy figures and API usage
    - _Requirements: 1.1, 1.2, 9.4_
  
  - [x] A1.2 Fix module-scope import blockers in existing code  
    - Remove filesystem assertion in `conga/util.py` line 26 that blocks installed layouts
    - Ensure `tcrdist_cpp` path constants remain as pure Path arithmetic
    - Update error handling to use `util.tcrdist_cpp_available()` instead of import-time asserts
    - _Requirements: 1.3, 10.9_
  
  - [x] A1.3 Add shared constants to `conga/util.py`
    - Add DEFAULT_RANDOM_SEED = 42, KPCA_REDUCTION_LIMIT = 20000
    - Add obsm keys: OBSM_KEY_VEC_TCR, OBSM_KEY_PCA_TCR, ACTIVE_REP_EXACT
    - Add uns keys: UNS_KEY_ACTIVE_TCR_REP, UNS_KEY_VEC_TCR_CONFIG
    - _Requirements: 8.1, 8.2_

- [x] A2. FAISS Infrastructure Setup (Parallel with A1)
  - [x] A2.1 Create `conga/neighbors.py` module for FAISS integration
    - Implement tiered backend system: faiss-gpu → faiss-cpu → sklearn
    - Add backend detection and capability reporting
    - Create unified neighbor search interface compatible with existing calc_nbrs
    - _Based on dev branch commit 210ce04_
  
  - [x] A2.2 Add FAISS dependencies and compatibility checking
    - Add faiss-cpu>=1.7.4 to pyproject.toml optional dependencies [performance]
    - Add faiss-gpu>=1.7.4 as alternative in [performance-gpu]
    - Update conga/cli.py version checking to report FAISS availability
    - _Integration requirements from branch roadmap_
  
  - [x] A2.3 Update environment.yml and installation documentation
    - Add FAISS to environment.yml with conda-forge channel
    - Update INSTALL.md with FAISS installation instructions
    - Document CPU vs GPU selection criteria
    - _Per python-standards.md requirements_

### Phase B: Core Algorithm Implementation (Parallel tracks)

- [x] B1. Vectorized encoding core algorithm (TCR side)
  - [x] B1.1 Build amino acid dissimilarity matrix and embedding
    - Implement `symbol_dissimilarity_matrix()` using CoNGA's bsd4 table and gap penalty
    - Implement `aa_embedding()` with deterministic SMACOF using classical_mds initialization
    - Add caching with proper fingerprinting for matrix and MDS call parameters
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_
  
  - [ ]* B1.2 Write property test for amino acid embedding determinism
    - **Property 3: Encoding is byte-identical across processes**
    - **Validates: Requirements 2.1, 2.2, 2.3**
  
  - [x] B1.3 Implement germline code table extraction
    - Create `germline_code_table()` to extract V gene CDR loops from Gene_Database
    - Support human, mouse, rhesus organisms with chains A and B
    - Apply invariant column removal to reduce vector length
    - _Requirements: 3.1, 3.2, 3.5, 3.6, 3.7_
  
  - [ ]* B1.4 Write property test for germline encoding accuracy
    - **Property 1: Germline encoding preserves exact TCRdist germline distances**  
    - **Validates: Requirements 3.1, 3.5, 3.6, 3.7**

- [x] B2. FAISS neighbor search implementation (TCR + GEX sides, parallel with B1)
  - [x] B2.1 Implement FAISS-powered GEX neighbor search (calc_nbrs replacement)
    - Create `FaissNeighborSearcher` class with IndexFlatIP and IndexFlatL2 support for GEX data
    - Implement GPU memory management and automatic CPU fallback for high-dimensional GEX matrices
    - Support batched processing for large datasets (>100k cells)
    - Maintain identical API to existing calc_nbrs for drop-in replacement
    - _Based on ~474 lines removed, ~95 lines added from dev branch_
  
  - [x] B2.2 Implement FAISS-powered vectorized TCR neighbor search
    - Extend `FaissNeighborSearcher` to handle vectorized TCR representations (X_vec_tcr)
    - Implement optimized indexing for fixed-length TCR vectors from vectorized encoding
    - Support distance thresholds and neighbor fraction calculations for TCR similarity
    - Integrate with existing TCR neighbor path selection (vectorized vs KernelPCA vs exact)
    - _New capability: FAISS acceleration for vectorized TCR path_
  
  - [x] B2.3 Add comprehensive performance benchmarking infrastructure
    - Create benchmark suite comparing FAISS vs sklearn performance for both TCR and GEX
    - Measure memory usage, query time, and accuracy across dataset sizes (1k-100k+ cells)
    - Support both CPU and GPU FAISS variants for TCR and GEX data types
    - Generate performance scaling curves and backend selection recommendations
    - _Success metric: 5-100x speedup target for both data types_
  
  - [x] B2.4 Implement unified graceful fallback system
    - Auto-detect FAISS availability at runtime for both TCR and GEX (no import failures)
    - Implement tiered fallback: faiss-gpu → faiss-cpu → sklearn for both modalities
    - Log backend selection and performance characteristics for TCR and GEX separately
    - Preserve identical results regardless of backend for both TCR and GEX neighbor search
    - _Backward compatibility requirement for all neighbor search paths_
### Phase C: Data Processing and Validation

- [x] C1. CDR3 processing and input validation (TCR side)
  - [x] C1.1 Create CDR3 trimming and gapping function
    - Implement `trim_and_gap_cdr3()` with configurable trim lengths and fixed output length
    - Apply gap positioning formula from tcr_distances.weighted_cdr3_distance
    - Handle interior residue dropping for sequences longer than num_pos_cdr3
    - _Requirements: 5.7, 4.7_
  
  - [x] C1.2 Implement comprehensive input validation
    - Validate organisms against SUPPORTED_ORGANISMS set
    - Check V gene presence in Gene_Database and report missing genes with counts
    - Validate CDR3 amino acid content and length constraints
    - Perform all validation before any memory allocation
    - _Requirements: 3.3, 3.4, 4.4, 4.5, 4.6, 4.8_
  
  - [ ]* C1.3 Write property test for CDR3 processing
    - **Property 8: CDR3 encoding has fixed length for every input length**
    - **Validates: Requirements 5.7, 4.7**

- [x] C2. FAISS accuracy and correctness validation (GEX side, parallel with C1)
  - [x] C2.1 Create FAISS vs sklearn accuracy validation suite
    - Compare neighbor indices and distances between backends
    - Validate nearest neighbor recall at different k values
    - Test edge cases: duplicate vectors, high dimensionality, batch boundaries
    - Ensure deterministic results with fixed random seeds
    - _Correctness requirement for production use_
  
  - [x] C2.2 Implement comprehensive error handling
    - Handle CUDA out-of-memory gracefully (GPU → CPU fallback)
    - Manage index building failures and corrupted data
    - Add proper logging for backend selection and performance
    - Provide clear error messages for unsupported configurations
    - _Production robustness requirement_

### Phase D: Core Assembly and Integration

- [x] D1. Vectorized assembly pipeline (TCR side)
  - [x] D1.1 Implement vectorized TCR encoding algorithm
    - Create `encode_tcrs()` main function supporting nested tuples and DataFrame input
    - Implement vectorized assembly using numpy fancy indexing for germline and CDR3 blocks  
    - Apply CDR3 weight scaling and ensure float32 C-contiguous output
    - Support custom column names and handle different input formats
    - _Requirements: 4.1, 4.2, 4.3, 5.1, 5.2, 5.4, 5.5, 5.8_
  
  - [ ]* D1.2 Write property test for encoding output contract
    - **Property 6: Output contract holds for every input**
    - **Validates: Requirements 5.2, 5.4**
  
  - [x] D1.3 Implement vector length calculation
    - Create `vector_length()` function to predict output dimensions
    - Ensure length calculation matches actual encoding output
    - _Requirements: 5.3_
  
  - [ ]* D1.4 Write property test for vector length consistency  
    - **Property 5: Predicted vector length equals produced width**
    - **Validates: Requirements 5.1, 5.3**

- [x] D2. Integrate FAISS into neighbor search pipelines (TCR + GEX sides, parallel with D1)
  - [x] D2.1 Update calc_nbrs to use FAISS backend for GEX
    - Modify calc_nbrs() and calc_nbrs_batched() to route GEX queries through neighbors.py
    - Preserve existing obsm_tag_gex behavior and parameter API
    - Add backend selection logging and performance reporting for GEX neighbor search
    - Maintain compatibility with existing analysis pipelines
    - _Integration point from dev branch_
  
  - [x] D2.2 Integrate FAISS backend for vectorized TCR neighbor search
    - Update vectorized TCR path to use FAISS for X_vec_tcr neighbor queries
    - Implement TCR-specific distance thresholds and neighbor fraction handling with FAISS
    - Preserve identical neighbor selection results compared to sklearn baseline
    - Add performance logging for TCR FAISS backend selection
    - _New: FAISS acceleration for vectorized TCR representations_
  
  - [x] D2.3 Update batch processing with unified FAISS optimization
    - Optimize batched neighbor calculation using FAISS index reuse for both TCR and GEX
    - Implement efficient index building for repeated queries on both modalities
    - Handle memory management for GPU indices in batched mode (TCR + GEX)
    - Support mixed backend selection (e.g., FAISS for GEX, sklearn for TCR) based on availability
    - _Large dataset optimization for both data types_

- [x] E1. Accuracy validation framework (TCR side)
  - [x] E1.1 Create accuracy reporting infrastructure
    - Implement `AccuracyReport` dataclass with correlation and recall metrics
    - Create `accuracy_report()` function comparing against TcrDistCalculator
    - Support pair sampling for large datasets and multiple neighbor counts
    - Include both Pearson distance and squared distance correlations
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.7_
  
  - [ ]* E1.2 Write accuracy validation tests
    - Create gated accuracy test asserting Spearman ≥ 0.95 and recall ≥ 0.80
    - Test on bundled human TCR database (1000 clonotype sample)
    - **Validates: Requirements 6.6**
  
  - [ ]* E1.3 Write property test for accuracy reporting determinism
    - **Property 16: Accuracy reporting is deterministic**
    - **Validates: Requirements 6.5**

- [x] E2. FAISS performance validation and optimization (GEX side, parallel with E1)
  - [x] E2.1 Create comprehensive performance test suite
    - Benchmark all backends across multiple dataset sizes (1k, 10k, 100k+ cells)
    - Measure index building time, query time, and memory usage
    - Test both dense and sparse input matrices
    - Generate performance scaling curves
    - _Performance validation requirement_
  
  - [x] E2.2 Optimize FAISS parameters for single-cell data
    - Tune nlist parameters for IVF indices on high-dimensional GEX data
    - Implement adaptive index selection based on data characteristics
    - Add memory-constrained index building for large datasets
    - Document optimal parameter settings
    - _Single-cell specific optimization_

### Phase F: AnnData Integration and Representation Selection

- [ ] F1. AnnData storage and representation management (TCR side)
  - [ ] F1.1 Create AnnData storage functions
    - Implement `store_tcr_vectors_in_adata()` with proper obsm key handling
    - Store EncodingConfig and metadata in uns with proper serialization
    - Handle existing key overwrites with appropriate warnings
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_
  
  - [ ] F1.2 Implement active representation tracking
    - Create `record_active_tcr_representation()` and `get_active_tcr_representation()`
    - Handle the three representation states: X_vec_tcr, X_pca_tcr, exact_tcrdist
    - Ensure exact path records sentinel without creating obsm entry
    - _Requirements: 7.7, 7.8_
  
  - [ ]* F1.3 Write property tests for AnnData persistence
    - **Property 11: AnnData persistence round-trips**
    - **Validates: Requirements 7.9, 7.10**
  
  - [ ]* F1.4 Write property test for row ordering
    - **Property 10: Stored rows follow adata.obs order**
    - **Validates: Requirements 7.1, 7.2, 7.4**

- [ ] F2. Backend selection and configuration management (GEX side, parallel with F1)
  - [ ] F2.1 Implement backend configuration storage in AnnData
    - Store FAISS backend selection and parameters in adata.uns
    - Record performance metrics and index characteristics
    - Enable reproducible analysis with same backend selection
    - Support backend preferences in analysis workflows
    - _Reproducibility and debugging requirement_
  
  - [ ] F2.2 Add backend selection CLI flags and logic
    - Add --use_faiss_gpu, --use_faiss_cpu, --disable_faiss flags to run_conga.py
    - Implement backend conflict detection and error reporting
    - Add performance logging and benchmark reporting options
    - Support backend preferences in batch workflows
    - _User control requirement_

### Phase G: Three-way TCR Selection + FAISS Integration

- [ ] G1. Complete TCR representation selection system
  - [ ] G1.1 Create TcrRepresentation resolver with FAISS awareness
    - Implement `TcrRepresentation` dataclass and `resolve_tcr_representation()` function
    - Implement complete selection table logic for organism support and observation counts
    - Handle all override combinations and conflict detection with FAISS options
    - Generate informative error messages for invalid combinations
    - _Requirements: 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 8.10, 8.11_
  
  - [ ]* G1.2 Write comprehensive selection table tests
    - **Property 12: Path selection is total and matches the table**
    - **Validates: Requirements 7.7, 7.8, 8.3-8.11**
  
  - [ ] G1.3 Implement restart logic for stored representations
    - Handle restart from h5ad files with existing X_pca_tcr and/or X_vec_tcr
    - Apply restart rules from requirements with proper precedence
    - Log warnings for stored representations exceeding current limits
    - Integrate FAISS backend preferences from stored metadata
    - _Requirements: 8.27, 8.28, 8.29, 8.30, 8.31_
  
  - [ ]* G1.4 Write property test for restart behavior
    - **Property 13: Restart reuses stored representations**
    - **Validates: Requirements 8.27, 8.28, 8.29, 8.30, 8.31**

### Phase H: CLI Integration and Validation

- [ ] H1. CLI flag handling and validation
  - [ ] H1.1 Update `run_conga.py` with new flags and integrated logic
    - Add --use_kpca_tcrdist, --kpca_reduction_limit and encoding config flags
    - Add FAISS backend selection flags: --use_faiss_gpu, --use_faiss_cpu, --disable_faiss
    - Implement flag conflict detection and clear error messages
    - Update path selection logic using resolve_tcr_representation with FAISS integration
    - Add run statistics recording for all three TCR paths and FAISS backends
    - _Requirements: 8.15, 8.16, 8.17, 8.18, 8.19, 8.20, 8.21, 8.22, 8.23, 8.24, 8.25, 8.26_
  
  - [ ] H1.2 Update preprocess.py consumer sites
    - Modify calc_nbrs call sites to use resolved obsm_tag_tcr and use_exact flags
    - Update cluster_and_tsne_and_umap to branch on active representation
    - Fix read_dataset warnings for vectorized path
    - Handle shuffle_tcr_kpcs for obsm-less exact path
    - Integrate FAISS backend selection throughout pipeline
    - _Requirements: obsm-less third state table from design_
  
  - [ ]* H1.3 Write property test for flag conflict detection
    - **Property 14: Conflicting flags are rejected naming both members**
    - **Property 18: FAISS backend conflicts are properly detected**
    - **Validates: Requirements 8.19, 8.20, 8.21, 8.22, 8.23**

### Phase I: Setup Integration and Testing Infrastructure

- [ ] I1. Setup_10x_for_conga.py integration
  - [ ] I1.1 Add setup CLI flag support and path selection
    - Add --kpca_reduction_limit and --use_kpca_tcrdist flags
    - Add FAISS backend selection flags to setup workflow
    - Implement behavior changes for supported organisms (skip KernelPCA by default)
    - Add clear messaging about which analysis paths and backends will be used
    - _Requirements: 8.32, 8.33, 8.34, 8.35, 8.36, 8.37, 8.38_

- [ ] I2. Comprehensive test suite infrastructure
  - [ ] I2.1 Create test fixtures and data
    - Set up pytest test infrastructure in tests/ directory
    - Create seeded clonotype fixtures from bundled TCR database
    - Generate synthetic mouse and rhesus test data with seeded random generation
    - Add FAISS backend mocking for CI environments without GPU
    - _Requirements: 10.1, 10.5_
  
  - [ ] I2.2 Create error condition tests
    - Write tests for all ValueError and exit conditions in error handling table
    - Cover organism validation, V gene validation, CDR3 validation, flag conflicts
    - Test exact path binary requirements and logging
    - Test FAISS backend failures and fallback behavior
    - _Requirements: 10.2_
  
  - [ ]* I2.3 Write property test for input format flexibility  
    - **Property 4: Input container form does not affect the result**
    - **Validates: Requirements 4.1, 4.2, 4.3**
  
  - [ ]* I2.4 Write property test for CDR3 weight scaling
    - **Property 7: CDR3 weight scales the CDR3 block quadratically** 
    - **Validates: Requirements 5.5**
  
  - [ ]* I2.5 Write property test for memory efficiency
    - **Property 9: Encoding memory is sub-quadratic in clonotype count**
    - **Property 19: FAISS memory usage scales sub-quadratically**
    - **Validates: Requirements 5.8**
  
  - [ ]* I2.6 Write property test for run statistics completeness
    - **Property 15: Run statistics are complete for whichever path ran**
    - **Property 20: FAISS performance metrics are recorded**
    - **Validates: Requirements 8.24, 8.25, 8.26**

### Phase J: Portability and Documentation

- [ ] J1. Portable import validation and cleanup
  - [ ] J1.1 Create subprocess import test
    - Stage synthetic installed layout excluding repository-specific paths
    - Test import success in scrubbed environment outside repository
    - Verify all vectorized and FAISS modules import without filesystem dependencies
    - Test graceful FAISS import failures in minimal environments
    - _Requirements: 10.9, 1.3_

- [ ] J2. Documentation and cleanup
  - [ ] J2.1 Add comprehensive module documentation  
    - Write NumPy-style docstrings for all public functions
    - Add type hints to all public function signatures
    - Document accuracy approximation and measured correlation figures
    - Document FAISS backend selection and performance characteristics
    - _Requirements: 9.1, 9.2, 9.3_
  
  - [ ]* J2.2 Write property test for documentation completeness
    - **Property 17: Public surface is documented and annotated**
    - **Property 21: FAISS backend API is fully documented**
    - **Validates: Requirements 9.1, 9.2**
  
  - [ ] J2.3 Update README and package metadata
    - Add "TCR representations" section describing all three paths
    - Add "FAISS acceleration" section with backend selection guidance
    - Document default behavior changes for alpha-beta organisms
    - Explain KPCA_REDUCTION_LIMIT and override flags
    - Document tcrdist_cpp binary requirements and FAISS installation
    - _Requirements: 9.7, 9.8_

### Phase K: Final Integration and Performance Validation

- [ ] K1. Remove prototypes and finalize integration
  - [ ] K1.1 Delete prototype file and update references
    - Remove `conga/tcrdist_vectorizing_functions_for_sharing.py`
    - Update steering document references to point to new implementation
    - Verify no remaining imports of prototype module
    - _Requirements: 9.6_
  
  - [ ] K1.2 Update package dependencies
    - Raise scikit-learn requirement to >=1.8 in pyproject.toml
    - Add faiss-cpu>=1.7.4 to [performance] extra
    - Add faiss-gpu>=1.7.4 to [performance-gpu] extra
    - Add hypothesis>=6.100 to dev extra for property testing
    - Update environment.yml if needed for new minimum versions
    - _Requirements: Dependencies section in design_

- [ ] K2. Final integration and validation
  - [ ] K2.1 Run end-to-end pipeline testing with both optimizations
    - Test all three TCR paths with example datasets
    - Test all FAISS backend combinations (GPU, CPU, sklearn fallback)
    - Verify behavior change: alpha-beta defaults to vectorized representation
    - Verify behavior change: large datasets default to FAISS when available
    - Confirm KernelPCA and exact paths work with appropriate overrides
    - Test restart scenarios from existing .h5ad files
    - _Requirements: complete workflow validation_
  
  - [ ] K2.2 Performance and accuracy validation
    - Measure encoding time and memory usage at N=20000 with both optimizations
    - Verify accuracy gates pass for all supported organisms
    - Test vectorized vs exact TCRdist correlation on real data
    - Measure FAISS vs sklearn performance gains across dataset sizes
    - Document combined performance characteristics and accuracy metrics
    - Validate 10-100x speedup and >50% memory reduction targets
    - _Requirements: 6.6, performance considerations from design_

- [ ] K3. Final checkpoint - Complete implementation ready
  - Ensure all tests pass for both vectorized TCRdist and FAISS acceleration
  - Verify backward compatibility maintained
  - Confirm all performance targets achieved
  - Ask the user if questions arise

## Notes

- Tasks marked with `*` are optional property-based and unit tests that can be skipped for faster MVP, but are highly recommended for correctness validation
- Each task references specific requirements for traceability  
- The implementation includes 21 correctness properties (17 original + 4 FAISS-specific) that should be validated through property-based testing
- Special attention needed for accuracy gates (Requirement 6.6): Spearman ≥ 0.95 and mean recall ≥ 0.80 for vectorized TCRdist
- Performance targets: 10-100x speedup and >50% memory reduction with both optimizations
- The feature changes default behavior for alpha-beta organisms from KernelPCA to vectorized representation
- FAISS adds default behavior change: large datasets prefer FAISS when available
- Use `mamba run -n conga-dev python` for all Python execution as specified by the user
- C++ compilation may be needed for full exact TCRdist path functionality (`make` in tcrdist_cpp directory)
- FAISS-GPU requires CUDA-capable hardware; graceful CPU fallback required

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["A1.1", "A1.2", "A1.3", "A2.1", "A2.2", "A2.3"] },
    { "id": 1, "tasks": ["B1.1", "B1.3", "B2.1", "B2.3"] },
    { "id": 2, "tasks": ["B1.2", "B1.4", "C1.1", "C1.2", "B2.2", "C2.1"] },
    { "id": 3, "tasks": ["C1.3", "D1.1", "D1.3", "C2.2", "D2.1"] },
    { "id": 4, "tasks": ["D1.2", "D1.4", "E1.1", "D2.2", "E2.1"] },
    { "id": 5, "tasks": ["E1.2", "E1.3", "F1.1", "F1.2", "E2.2", "F2.1"] },
    { "id": 6, "tasks": ["F1.3", "F1.4", "G1.1", "F2.2"] },
    { "id": 7, "tasks": ["G1.2", "G1.3"] },
    { "id": 8, "tasks": ["G1.4", "H1.1", "H1.2"] },
    { "id": 9, "tasks": ["H1.3", "I1.1"] },
    { "id": 10, "tasks": ["I2.1", "I2.2"] },
    { "id": 11, "tasks": ["I2.3", "I2.4", "I2.5", "I2.6", "J1.1"] },
    { "id": 12, "tasks": ["J2.1", "J2.3"] },
    { "id": 13, "tasks": ["J2.2", "K1.1", "K1.2"] },
    { "id": 14, "tasks": ["K2.1", "K2.2"] }
  ]
}
```

## Success Metrics

### Performance Targets
- **Combined optimization**: 10-100x speedup on large datasets (N≥20k)
- **Memory reduction**: >50% reduction in peak memory usage
- **Vectorized TCRdist**: Sub-quadratic memory scaling for TCR encoding
- **FAISS TCR acceleration**: 5-50x improvement in vectorized TCR neighbor search
- **FAISS GEX acceleration**: 5-50x improvement in GEX neighbor search

### Accuracy Requirements
- **Vectorized TCRdist**: Spearman ≥ 0.95, mean recall@100 ≥ 0.80
- **FAISS neighbors**: Identical results to sklearn (exact neighbor indices)
- **End-to-end**: CoNGA scores maintain correlation >0.98 with baseline

### Compatibility Requirements  
- **Backward compatibility**: All existing workflows continue unchanged
- **Graceful degradation**: Function without FAISS or with CPU-only systems
- **Default behavior**: Automatic selection of best available backends
