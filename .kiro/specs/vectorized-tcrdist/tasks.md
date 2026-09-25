# Implementation Plan: Vectorized TCRdist

## Overview

This implementation replaces CoNGA's default TCR representation for alpha-beta receptors with a new vectorized approach that avoids quadratic memory usage. The vectorized encoder embeds TCRdist amino acid dissimilarity matrices into Euclidean space and encodes each TCR as a fixed-length vector, enabling neighbor search without materializing dense N×N matrices. The feature implements three TCR neighbor paths: vectorized (new default for alpha-beta), KernelPCA (existing, with observation count limits), and exact TCRdist (streaming, no obsm array).

## Tasks

- [ ] 1. Set up portable module foundation and fix import blockers
  - [ ] 1.1 Create `conga/tcrdist/vectorized.py` module with portable initialization
    - Create new module with lazy imports to avoid filesystem assertions at module scope
    - Implement module-level constants and EncodingConfig dataclass
    - Add module docstring documenting accuracy figures and API usage
    - _Requirements: 1.1, 1.2, 9.4_
  
  - [ ] 1.2 Fix module-scope import blockers in existing code  
    - Remove filesystem assertion in `conga/util.py` line 26 that blocks installed layouts
    - Ensure `tcrdist_cpp` path constants remain as pure Path arithmetic
    - Update error handling to use `util.tcrdist_cpp_available()` instead of import-time asserts
    - _Requirements: 1.3, 10.9_
  
  - [ ] 1.3 Add shared constants to `conga/util.py`
    - Add DEFAULT_RANDOM_SEED = 42, KPCA_REDUCTION_LIMIT = 20000
    - Add obsm keys: OBSM_KEY_VEC_TCR, OBSM_KEY_PCA_TCR, ACTIVE_REP_EXACT
    - Add uns keys: UNS_KEY_ACTIVE_TCR_REP, UNS_KEY_VEC_TCR_CONFIG
    - _Requirements: 8.1, 8.2_

- [ ] 2. Implement core vectorized encoding algorithm  
  - [ ] 2.1 Build amino acid dissimilarity matrix and embedding
    - Implement `symbol_dissimilarity_matrix()` using CoNGA's bsd4 table and gap penalty
    - Implement `aa_embedding()` with deterministic SMACOF using classical_mds initialization
    - Add caching with proper fingerprinting for matrix and MDS call parameters
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_
  
  - [ ]* 2.2 Write property test for amino acid embedding determinism
    - **Property 3: Encoding is byte-identical across processes**
    - **Validates: Requirements 2.1, 2.2, 2.3**
  
  - [ ] 2.3 Implement germline code table extraction
    - Create `germline_code_table()` to extract V gene CDR loops from Gene_Database
    - Support human, mouse, rhesus organisms with chains A and B
    - Apply invariant column removal to reduce vector length
    - _Requirements: 3.1, 3.2, 3.5, 3.6, 3.7_
  
  - [ ]* 2.4 Write property test for germline encoding accuracy
    - **Property 1: Germline encoding preserves exact TCRdist germline distances**  
    - **Validates: Requirements 3.1, 3.5, 3.6, 3.7**

- [ ] 3. Implement CDR3 processing and input validation
  - [ ] 3.1 Create CDR3 trimming and gapping function
    - Implement `trim_and_gap_cdr3()` with configurable trim lengths and fixed output length
    - Apply gap positioning formula from tcr_distances.weighted_cdr3_distance
    - Handle interior residue dropping for sequences longer than num_pos_cdr3
    - _Requirements: 5.7, 4.7_
  
  - [ ] 3.2 Implement comprehensive input validation
    - Validate organisms against SUPPORTED_ORGANISMS set
    - Check V gene presence in Gene_Database and report missing genes with counts
    - Validate CDR3 amino acid content and length constraints
    - Perform all validation before any memory allocation
    - _Requirements: 3.3, 3.4, 4.4, 4.5, 4.6, 4.8_
  
  - [ ]* 3.3 Write property test for CDR3 processing
    - **Property 8: CDR3 encoding has fixed length for every input length**
    - **Validates: Requirements 5.7, 4.7**

- [ ] 4. Build vectorized assembly and encoding pipeline
  - [ ] 4.1 Implement vectorized TCR encoding algorithm
    - Create `encode_tcrs()` main function supporting nested tuples and DataFrame input
    - Implement vectorized assembly using numpy fancy indexing for germline and CDR3 blocks  
    - Apply CDR3 weight scaling and ensure float32 C-contiguous output
    - Support custom column names and handle different input formats
    - _Requirements: 4.1, 4.2, 4.3, 5.1, 5.2, 5.4, 5.5, 5.8_
  
  - [ ]* 4.2 Write property test for encoding output contract
    - **Property 6: Output contract holds for every input**
    - **Validates: Requirements 5.2, 5.4**
  
  - [ ] 4.3 Implement vector length calculation
    - Create `vector_length()` function to predict output dimensions
    - Ensure length calculation matches actual encoding output
    - _Requirements: 5.3_
  
  - [ ]* 4.4 Write property test for vector length consistency  
    - **Property 5: Predicted vector length equals produced width**
    - **Validates: Requirements 5.1, 5.3**

- [ ] 5. Checkpoint - Core encoding functionality complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. Implement accuracy validation framework
  - [ ] 6.1 Create accuracy reporting infrastructure
    - Implement `AccuracyReport` dataclass with correlation and recall metrics
    - Create `accuracy_report()` function comparing against TcrDistCalculator
    - Support pair sampling for large datasets and multiple neighbor counts
    - Include both Pearson distance and squared distance correlations
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.7_
  
  - [ ]* 6.2 Write accuracy validation tests
    - Create gated accuracy test asserting Spearman ≥ 0.95 and recall ≥ 0.80
    - Test on bundled human TCR database (1000 clonotype sample)
    - **Validates: Requirements 6.6**
  
  - [ ]* 6.3 Write property test for accuracy reporting determinism
    - **Property 16: Accuracy reporting is deterministic**
    - **Validates: Requirements 6.5**

- [ ] 7. Implement AnnData integration and storage
  - [ ] 7.1 Create AnnData storage functions
    - Implement `store_tcr_vectors_in_adata()` with proper obsm key handling
    - Store EncodingConfig and metadata in uns with proper serialization
    - Handle existing key overwrites with appropriate warnings
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_
  
  - [ ] 7.2 Implement active representation tracking
    - Create `record_active_tcr_representation()` and `get_active_tcr_representation()`
    - Handle the three representation states: X_vec_tcr, X_pca_tcr, exact_tcrdist
    - Ensure exact path records sentinel without creating obsm entry
    - _Requirements: 7.7, 7.8_
  
  - [ ]* 7.3 Write property tests for AnnData persistence
    - **Property 11: AnnData persistence round-trips**
    - **Validates: Requirements 7.9, 7.10**
  
  - [ ]* 7.4 Write property test for row ordering
    - **Property 10: Stored rows follow adata.obs order**
    - **Validates: Requirements 7.1, 7.2, 7.4**

- [ ] 8. Implement three-way TCR representation selection
  - [ ] 8.1 Create TcrRepresentation resolver
    - Implement `TcrRepresentation` dataclass and `resolve_tcr_representation()` function
    - Implement complete selection table logic for organism support and observation counts
    - Handle all override combinations and conflict detection
    - Generate informative error messages for invalid combinations
    - _Requirements: 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 8.10, 8.11_
  
  - [ ]* 8.2 Write comprehensive selection table tests
    - **Property 12: Path selection is total and matches the table**
    - **Validates: Requirements 7.7, 7.8, 8.3-8.11**
  
  - [ ] 8.3 Implement restart logic for stored representations
    - Handle restart from h5ad files with existing X_pca_tcr and/or X_vec_tcr
    - Apply restart rules from requirements with proper precedence
    - Log warnings for stored representations exceeding current limits
    - _Requirements: 8.27, 8.28, 8.29, 8.30, 8.31_
  
  - [ ]* 8.4 Write property test for restart behavior
    - **Property 13: Restart reuses stored representations**
    - **Validates: Requirements 8.27, 8.28, 8.29, 8.30, 8.31**

- [ ] 9. Integrate CLI flag handling and validation
  - [ ] 9.1 Update `run_conga.py` with new flags and logic
    - Add --use_kpca_tcrdist, --kpca_reduction_limit and encoding config flags
    - Implement flag conflict detection and clear error messages
    - Update path selection logic using resolve_tcr_representation
    - Add run statistics recording for all three paths
    - _Requirements: 8.15, 8.16, 8.17, 8.18, 8.19, 8.20, 8.21, 8.22, 8.23, 8.24, 8.25, 8.26_
  
  - [ ] 9.2 Update preprocess.py consumer sites
    - Modify calc_nbrs call sites to use resolved obsm_tag_tcr and use_exact flags
    - Update cluster_and_tsne_and_umap to branch on active representation
    - Fix read_dataset warnings for vectorized path
    - Handle shuffle_tcr_kpcs for obsm-less exact path
    - _Requirements: obsm-less third state table from design_
  
  - [ ]* 9.3 Write property test for flag conflict detection
    - **Property 14: Conflicting flags are rejected naming both members**
    - **Validates: Requirements 8.19, 8.20, 8.21, 8.22, 8.23**

- [ ] 10. Update setup_10x_for_conga.py integration
  - [ ] 10.1 Add setup CLI flag support and path selection
    - Add --kpca_reduction_limit and --use_kpca_tcrdist flags
    - Implement behavior changes for supported organisms (skip KernelPCA by default)
    - Add clear messaging about which analysis path will be used
    - _Requirements: 8.32, 8.33, 8.34, 8.35, 8.36, 8.37, 8.38_

- [ ] 11. Implement comprehensive test suite
  - [ ] 11.1 Create test fixtures and data
    - Set up pytest test infrastructure in tests/ directory
    - Create seeded clonotype fixtures from bundled TCR database
    - Generate synthetic mouse and rhesus test data with seeded random generation
    - _Requirements: 10.1, 10.5_
  
  - [ ] 11.2 Create error condition tests
    - Write tests for all ValueError and exit conditions in error handling table
    - Cover organism validation, V gene validation, CDR3 validation, flag conflicts
    - Test exact path binary requirements and logging
    - _Requirements: 10.2_
  
  - [ ]* 11.3 Write property test for input format flexibility  
    - **Property 4: Input container form does not affect the result**
    - **Validates: Requirements 4.1, 4.2, 4.3**
  
  - [ ]* 11.4 Write property test for CDR3 weight scaling
    - **Property 7: CDR3 weight scales the CDR3 block quadratically** 
    - **Validates: Requirements 5.5**
  
  - [ ]* 11.5 Write property test for memory efficiency
    - **Property 9: Encoding memory is sub-quadratic in clonotype count**
    - **Validates: Requirements 5.8**
  
  - [ ]* 11.6 Write property test for run statistics completeness
    - **Property 15: Run statistics are complete for whichever path ran**
    - **Validates: Requirements 8.24, 8.25, 8.26**

- [ ] 12. Add portable import validation test
  - [ ] 12.1 Create subprocess import test
    - Stage synthetic installed layout excluding repository-specific paths
    - Test import success in scrubbed environment outside repository
    - Verify all vectorized modules import without filesystem dependencies
    - _Requirements: 10.9, 1.3_

- [ ] 13. Documentation and cleanup
  - [ ] 13.1 Add comprehensive module documentation  
    - Write NumPy-style docstrings for all public functions
    - Add type hints to all public function signatures
    - Document accuracy approximation and measured correlation figures
    - _Requirements: 9.1, 9.2, 9.3_
  
  - [ ]* 13.2 Write property test for documentation completeness
    - **Property 17: Public surface is documented and annotated**
    - **Validates: Requirements 9.1, 9.2**
  
  - [ ] 13.3 Update README and package metadata
    - Add "TCR representations" section describing all three paths
    - Document default behavior change for alpha-beta organisms
    - Explain KPCA_REDUCTION_LIMIT and override flags
    - Document tcrdist_cpp binary requirements
    - _Requirements: 9.7, 9.8_

- [ ] 14. Remove prototype and finalize integration
  - [ ] 14.1 Delete prototype file and update references
    - Remove `conga/tcrdist_vectorizing_functions_for_sharing.py`
    - Update steering document references to point to new implementation
    - Verify no remaining imports of prototype module
    - _Requirements: 9.6_
  
  - [ ] 14.2 Update package dependencies
    - Raise scikit-learn requirement to >=1.8 in pyproject.toml
    - Add hypothesis>=6.100 to dev extra for property testing
    - Update environment.yml if needed for new minimum versions
    - _Requirements: Dependencies section in design_

- [ ] 15. Final integration and validation
  - [ ] 15.1 Run end-to-end pipeline testing
    - Test all three TCR paths with example datasets
    - Verify behavior change: alpha-beta defaults to vectorized representation
    - Confirm KernelPCA and exact paths work with appropriate overrides
    - Test restart scenarios from existing .h5ad files
    - _Requirements: complete workflow validation_
  
  - [ ] 15.2 Performance and accuracy validation
    - Measure encoding time and memory usage at N=20000
    - Verify accuracy gates pass for all supported organisms
    - Test vectorized vs exact TCRdist correlation on real data
    - Document performance characteristics and accuracy metrics
    - _Requirements: 6.6, performance considerations from design_

- [ ] 16. Final checkpoint - Complete implementation ready
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional property-based and unit tests that can be skipped for faster MVP, but are highly recommended for correctness validation
- Each task references specific requirements for traceability
- The implementation includes 17 correctness properties that should be validated through property-based testing
- Special attention needed for the accuracy gates (Requirement 6.6): Spearman ≥ 0.95 and mean recall ≥ 0.80
- The feature changes default behavior for alpha-beta organisms from KernelPCA to vectorized representation
- Use `mamba run -n conga-dev python` for all Python execution as specified by the user
- C++ compilation may be needed for full exact TCRdist path functionality (`make` in tcrdist_cpp directory)

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3"] },
    { "id": 1, "tasks": ["2.1", "2.3"] },
    { "id": 2, "tasks": ["2.2", "2.4", "3.1", "3.2"] },
    { "id": 3, "tasks": ["3.3", "4.1", "4.3"] },
    { "id": 4, "tasks": ["4.2", "4.4", "6.1"] },
    { "id": 5, "tasks": ["6.2", "6.3", "7.1", "7.2"] },
    { "id": 6, "tasks": ["7.3", "7.4", "8.1"] },
    { "id": 7, "tasks": ["8.2", "8.3"] },
    { "id": 8, "tasks": ["8.4", "9.1", "9.2"] },
    { "id": 9, "tasks": ["9.3", "10.1"] },
    { "id": 10, "tasks": ["11.1", "11.2"] },
    { "id": 11, "tasks": ["11.3", "11.4", "11.5", "11.6", "12.1"] },
    { "id": 12, "tasks": ["13.1", "13.3"] },
    { "id": 13, "tasks": ["13.2", "14.1", "14.2"] },
    { "id": 14, "tasks": ["15.1", "15.2"] }
  ]
}
```