# Changelog

All notable changes to the CoNGA project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Species Expansion
- Expanded organism support with chain-completeness-based scoping across three distinct tiers:
  - Gene database coverage: 22 organisms with germline V/J/C gene data (`human`, `human_gd`, `human_ig`, `mouse`, `mouse_gd`, `mouse_ig`, `rhesus`, `rhesus_gd`, `rhesus_ig`, `cat`, `cat_gd`, `cat_ig`, `dog`, `dog_gd`, `dog_ig`, `ferret`, `ferret_gd`, `ferret_ig`, `rabbit`, `rabbit_gd`, `rabbit_ig`, `sheep`)
  - CLI-usable via `--organism` in both `run_conga.py` and `setup_10x_for_conga.py`: 21 organisms (same list minus `mouse_ig`)
  - Vectorized-TCRdist-eligible (`conga.tcrdist.vectorized.SUPPORTED_ORGANISMS`): 18 organisms, excluding `human_gd`, `human_ig`, `mouse_gd`, and `mouse_ig`, which remain usable via the KernelPCA or exact-TCRdist paths
- `sheep_gd`, `sheep_ig`, and `rainbowtrout_ig` exist as rows in the raw gene database TSV but are chain-incomplete and intentionally not exposed as supported organisms

#### Batch Integration
- `conga.preprocess.batch_integration()`: GEX batch correction using Harmony or scVI, writing the corrected representation into `adata.obsm['X_pca_gex']`
- New CLI flags `--batch_key` and `--batch_integration_method` (`harmony` or `scvi`) on `scripts/run_conga.py`, required together and mutually exclusive with `--force_variable_genes`
- This is distinct from the pre-existing `--batch_keys` (plural) annotation-only mechanism, which still only drives visualization (`make_batch_colored_umaps`, `make_clone_batch_clustermaps`) and does not correct GEX

#### Python 3.14 Support
- Full test suite run and passing on Python 3.12 and 3.14 (`environment.yml` now installs 3.14 by default for development); `requires-python` in `pyproject.toml` remains `>=3.12`. Python 3.13 is expected to work based on dependency metadata (all required packages ship 3.13 wheels/conda builds) but has not been directly tested.

#### Converter-free figures
- The TCR/BCR sequence logos, DEG and TCR-feature logos, lit-match logos, and TCRdist tree figures are now drawn directly with matplotlib (new module `conga/tcrdist/mpl_logo_basic.py`), so no external SVG-to-PNG converter (ImageMagick, Inkscape, rsvg-convert, cairosvg) is needed for any figure produced by `run_conga.py`. Glyphs come from the DejaVu Sans Mono font bundled with matplotlib, so output no longer depends on system fonts.
- Affected figures: `graph_vs_graph_logos`, `tcr_clumping_logos` (and other `make_logo_plots` / `make_cluster_logo_plots_figure` figures), `gex_clusters_tcrdist_trees`, `tcr_clusters_tcrdist_trees`, `conga_threshold_tcrdist_tree`, and `scripts/make_tcr_logos.py`.
- The logo and tree layout code (CDR3 alignment, PWMs, junction bars, tree layout, glyph selection) is unchanged and shared by both renderers through a `drawer` argument. Validated on the bundled 5K PBMC dataset: the legacy SVG output is byte-identical before and after the refactor, and tests check that both renderers receive identical geometry. Rasterized output differs from the ImageMagick-converted PNGs only at the pixel level (anti-aliasing, and ImageMagick 6.9's half-pixel offset on thin tree branches).
- New options: `png_scale` on `make_tcrdist_trees` / `make_tcrdist_tree_for_conga_score_threshold` and `run_conga.py --tcrdist_tree_png_scale` (higher-resolution trees, same layout; default 1 = previous 800x1000 pixels per tree); `save_pdf` on the two tree functions and `run_conga.py --save_tcrdist_tree_pdfs` (vector PDF trees); `scripts/make_tcr_logos.py --png_scale`.
- Legacy options `logo_renderer='svg'` (`make_logo_plots`) and `tree_renderer='svg'` (tree functions), and `renderer='svg'` (`make_tcr_logo_for_tcrs`), keep the old SVG route for comparison; they still need a converter and are planned for removal.

### Changed
- New optional dependency extra `batch-integration` (`harmonypy`, `scvi-tools`) backs `batch_integration()`
- `louvain` moved from a core dependency to a new optional `legacy-clustering` extra, backing the deprecated `--clustering_method louvain` option. Only installable on Python 3.12: conda-forge has no build for 3.13+, and building from source there fails against modern compilers (its vendored igraph C core trips `-Werror=uninitialized-const-pointer`). The default `--clustering_method leiden` (backed by the core `leidenalg` dependency) is unaffected. Explicitly requesting `--clustering_method louvain` without the extra installed now raises an actionable `ImportError` instead of a bare scanpy stack trace.
- `bbknn` removed from `environment.yml` entirely: it had the same conda-forge Python-version ceiling as `louvain`, and was never wired into any code path (`conga.preprocess.batch_integration()` only supports `method='harmony'` or `'scvi'` and explicitly rejects `'bbknn'`)
- `run_conga.py` results are now fully reproducible run to run: every remaining random step takes an explicit seed (a `random_seed` argument, default `util.DEFAULT_RANDOM_SEED`, set from `--random_seed`) instead of using the global `random`/`np.random` state. Before, these outputs changed between identical runs: the background TCRs behind the `tcr_clumping`, `tcr_db_match` and metaconga clump-match p-values (`tcr_sampler.resample_shuffled_tcr_chains`), the permutation z-scores in `graph_vs_graph_stats`, which clonotypes are drawn in TCRdist trees (subsampled above 400 TCRs per tree), the `--shuffle_tcr_kpcs`/`--shuffle_gex_nbrs` controls, the point order in the batch-colored UMAPs, the strip-plot jitter in `_qc_violins.png` (seaborn draws it from the global `np.random` state, so it now runs inside the new `util.seeded_global_np_random` context manager), and the column order of `_cluster_gene_scores.tsv` (set-iteration order of gene names, which depends on `PYTHONHASHSEED`; now sorted). Also seeded: the FAISS IVF index-building subsamples in `conga.neighbors` (only reached by direct callers, since `calc_nbrs` uses exact indices), and the random draws in `devel.analyze_proteins`, `devel.calc_nbr_zscore_threshold`, `devel.run_umap_and_clustering_from_indices_distances` and `merge_samples.py`. The affected values change once relative to earlier runs (they were random draws) and are then fixed.
- `--random_seed` now seeds the whole pipeline, and is no longer treated as a vectorized-encoding flag: it is accepted with `--use_kpca_tcrdist`, `--no_kpca`, `--use_exact_tcrdist_nbrs` and organisms the vectorizer doesn't support (where it already seeded clustering, UMAP and kernel PCA, but non-default values were rejected). The vectorized encoding itself is deterministic and doesn't depend on the seed (`init='classical_mds'`). The seed used is written to the run log and to `adata.uns['conga_stats']['random_seed']`.
- `setup_10x_for_conga.py --random_seed` (and `run_conga.py --rerun_kpca`) now pass the seed on to the kernel PCA. The default solver is deterministic, so no output changes. The setup `--random_seed` help text no longer claims it seeds the vectorized encoding.
- `setup_10x_for_conga.py` now exits with an error if given `--aa_mds_dim`, `--num_pos_cdr3` or `--cdr3_weight` (on the command line or in `--config`). Setup never builds the vectorized TCR representation, so these were silently ignored; pass them to `run_conga.py` instead.
- Fixed the `gex_clusters_degs_violinplot` figure (`conga.devel.find_gex_cluster_degs`): the violins were drawn at 1/(number of GEX clusters) of their normal width and shifted off-center, with the strip-plot points left at the column centers. Cause: scanpy's `stacked_violin` colors violins with `hue` set to the groupby categories, and under pandas 3, seaborn 0.13's automatic dodge check counts unobserved categorical combinations and dodges each violin. conga now passes `dodge=False`. The DEG results are unchanged.

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
- Optional dependency groups: `[performance]`, `[performance-gpu]`, `[batch-integration]`, `[dev]`, `[all]`, `[all-gpu]`
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
- **Optional**: faiss-cpu>=1.7.4, faiss-gpu>=1.7.4, harmonypy>=0.0.10, scvi-tools>=1.1.0 (harmonypy and scvi-tools both via the `batch-integration` extra)
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

[Unreleased]: https://github.com/phbradley/conga/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/phbradley/conga/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/phbradley/conga/compare/v0.1.1...v0.1.2  
[0.1.1]: https://github.com/phbradley/conga/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/phbradley/conga/releases/tag/v0.1.0