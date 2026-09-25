# CoNGA Project Overview

## What is CoNGA?

**Clonotype Neighbor Graph Analysis (CoNGA)** is a Python package for detecting correlations between T cell gene expression profiles and TCR (T Cell Receptor) sequences in single-cell datasets. It has been extended to support B cells (BCRs) and gamma-delta TCRs.

**Reference:** Schattgen et al., Nature Biotechnology (2021)
- https://www.nature.com/articles/s41587-021-00989-2

## Supported Data Types

- **Human:** TCRαβ, TCRγδ, and Ig (B cells)
- **Mouse:** TCRαβ, TCRγδ, and Ig
- **Rhesus:** TCRαβ and TCRγδ

## Project Structure

```
conga/
├── __init__.py
├── cd8_scoring.py          # CD8 cell scoring
├── imhc_scoring.py         # iMHC scoring
├── pmhc_scoring.py         # pMHC scoring
├── tcr_scoring.py          # TCR scoring logic
├── correlations.py         # Correlation analysis
├── plotting.py             # Visualization
├── tcr_clumping.py         # TCR clumping analysis
├── preprocess.py           # Data preprocessing
├── data/                   # Reference databases and parameters
└── tcrdist/                # TCRdist distance calculations (includes C++ implementations)

scripts/
├── run_conga.py            # Main analysis pipeline
├── setup_10x_for_conga.py  # 10x data preprocessing
├── merge_samples.py        # Multi-sample merging
└── ...

examples/                   # Example workflows and datasets
tests/                      # Test suite (if present)
```

## Key Technologies

- **Scanpy:** Single-cell analysis framework
- **AnnData:** Annotated data matrix format
- **Scikit-learn:** Machine learning utilities
- **Python-igraph / Louvain:** Graph clustering
- **C++ TCRdist:** Fast distance calculations for TCR sequences

## Main Analysis Workflow

1. **SETUP** (`setup_10x_for_conga.py`): Convert 10x TCR data, compute TCRdist distances, apply KernelPCA
2. **ANALYZE** (`run_conga.py`): Main pipeline with correlation analysis between GEX and TCR
3. **RE-ANALYZE**: Restart from saved .h5ad files for quick iteration

## Output Formats

- `.h5ad` files: AnnData objects with integrated GEX and TCR data
- `.tsv` files: Tab-separated results and feature lists
- `.png` files: SVG-to-PNG converted visualizations
- `_summary.png`: Overview of GEX and TCR landscapes with CoNGA scores
- `_bicluster_logos.png`: Gene expression and TCR V/J usage patterns

## Key Concepts

- **TCRdist:** Distance metric between TCR sequences
- **CoNGA Score:** Measure of overlap between GEX and TCR neighbor graphs
- **Clonotype:** Group of cells with identical TCR sequences
- **Bicluster:** Combination of GEX and TCR cluster assignments
