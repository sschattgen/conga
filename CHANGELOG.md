# Changelog

All notable changes to the CoNGA project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2024-12-19

### Added

#### Major Performance Enhancements
- **Vectorized TCRdist**: New fixed-length vector encoding for α/β TCRs that eliminates quadratic memory scaling
  - Supports human, mouse, and rhesus α/β TCRs
  - Achieves >50% memory reduction on large datasets
  - Sub-quadratic memory usage: 91MB for 20k clonotypes vs 6.4GB for KernelPCA
  - Spearman correlation ≥0.95 with exact TCRdist
  - Deterministic encoding with reproducible random seeds

- **FAISS Acceleration**: Optional GPU/CPU-optimized neighbor search
  - 10-100x speedup on large datasets
  - Tiered backend selection: faiss-gpu → faiss-cpu → sklearn
  - Graceful fallback when FAISS unavailable
  - Automatic backend detection and logging

#### TCR Representation System
- Three-way TCR neighbor path selection system
- Automatic selection based on organism type and dataset size
- CLI flags for manual override: `--use_kpca_tcrdist`, `--no_kpca`, `--kpca_reduction_limit`
- Support for restarting from saved representations in .h5ad files

#### New CLI Options
- `--use_faiss_gpu` / `--use_faiss_cpu` / `--disable_faiss`: Control FAISS backend
- `--aa_mds_dim`, `--num_pos_cdr3`, `--cdr3_weight`: Vectorized encoding parameters
- `--random_seed`: Control reproducibility (default: 42)
- Extended setup script support for new representations

#### Package Infrastructure
- Modern Python packaging with pyproject.toml
- Optional dependency groups: `[performance]`, `[performance-gpu]`, `[batch]`, `[scvi]`, `[dev]`, `[all]`, `[all-gpu]`
- Python 3.12+ requirement
- Updated environment.yml for development
- Comprehensive test suite with property-based testing

### Changed

#### Breaking Changes
- **BREAKING**: Default TCR representation for α/β organisms (human, mouse, rhesus) changed from KernelPCA to vectorized encoding
- **BREAKING**: Minimum Python version raised to 3.12
- **BREAKING**: Updated dependency versions for pandas 3.0+ and numpy 2.0+ compatibility

#### Performance Improvements  
- Default behavior change: large datasets automatically use FAISS when available
- Vectorized TCR encoding eliminates N×N distance matrix materialization
- FAISS backend provides 5-100x improvement in neighbor search
- Memory usage scales sub-quadratically with dataset size

#### Accuracy and Validation
- Comprehensive accuracy validation framework
- Measured performance targets and accuracy gates
- Deterministic behavior with constant random seed (42) by default
- Extensive benchmarking infrastructure

### Fixed
- Resolved quadratic memory scaling issues for large TCR datasets
- Fixed import blockers for portable module initialization
- Improved error handling and validation for all input formats
- Better logging and status reporting throughout pipeline

### Dependencies
- **Required**: scanpy>=1.10.0, anndata>=0.10.0, numpy>=1.26.0, scipy>=1.11.0, pandas>=2.1.0, scikit-learn>=1.8.0
- **Optional**: faiss-cpu>=1.7.4, faiss-gpu>=1.7.4, bbknn>=1.6.0, scvi-tools>=1.1.0
- **Development**: pytest>=7.4.0, hypothesis>=6.100.0, black>=23.0.0, ruff>=0.1.0

## [0.1.2] - 2023-09-21

### Added
- Rhesus alpha beta and gamma delta T cell support
- Extended organism support for rhesus TCRs

### Fixed  
- Gene expression matrix rescaling after reducing to single clone
- Improved GEX UMAP and clustering stability

## [0.1.1] - 2021-09-10

### Fixed
- Rescale adata.X gene expression matrix after reducing to single clone
- Prevents rare cases of wonky GEX UMAPs dominated by individual genes

## [0.1.0] - 2021-01-21

### Added
- Experimental TCR database matching against literature-derived sequences
- TCR clumping analysis for detecting clustered TCR regions  
- C++ implementation of TCRdist for improved performance
- Hotspot autocorrelation algorithm implementation
- Multi-sample merging functionality
- Support for gamma-delta TCRs and B cell receptors
- Docker containerization support

### Changed
- Enhanced statistical testing for TCR neighborhoods
- Improved visualization and plotting capabilities
- Extended organism support: human_gd, mouse_gd, human_ig, mouse_ig

[0.2.0]: https://github.com/phbradley/conga/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/phbradley/conga/compare/v0.1.1...v0.1.2  
[0.1.1]: https://github.com/phbradley/conga/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/phbradley/conga/releases/tag/v0.1.0