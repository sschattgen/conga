# CoNGA Architecture and Design Patterns

## Data Model: AnnData

CoNGA uses **AnnData** (Annotated Data Matrix) as its primary data structure. Understanding this is critical for development.

### AnnData Structure

```
adata.X              # Gene expression matrix (cells × genes)
adata.obs            # Cell-level metadata (index: cell barcodes)
adata.var            # Gene-level metadata (index: gene names)
adata.obsm           # Cell embeddings (e.g., 'X_umap', 'X_pca')
adata.varm           # Gene embeddings
adata.obsp           # Pairwise cell distances/similarities
adata.uns            # Unstructured data (parameters, results, plots)
adata.layers         # Alternative representations (e.g., 'raw', 'log1p')
```

### CoNGA-Specific Fields

CoNGA stores TCR and analysis results in `adata` as follows:

```python
# TCR information (in adata.obs)
adata.obs['clonotype_id']           # Clonotype identifier
adata.obs['cdr3_a']                 # Alpha chain CDR3
adata.obs['cdr3_b']                 # Beta chain CDR3
adata.obs['v_a']                    # Alpha chain V gene
adata.obs['v_b']                    # Beta chain V gene
adata.obs['j_a']                    # Alpha chain J gene
adata.obs['j_b']                    # Beta chain J gene

# Analysis results (in adata.uns)
adata.uns['conga_results']          # Main results dictionary
adata.uns['tcr_clumping']           # TCR clumping analysis
adata.uns['hotspot_features']       # Hotspot analysis results
adata.uns['graph_pairs']            # TCR/GEX graph pairings

# Neighbor graphs (in adata.obsp)
adata.obsp['tcr_distances']         # TCR distance matrix
adata.obsp['gex_distances']         # GEX distance matrix (from scanpy)
```

## Module Organization

### Core Modules

**preprocess.py**
- Data validation and cleaning
- Barcode handling and mapping
- Clonotype identification from 10x outputs

**tcr_scoring.py**
- TCR feature extraction
- CDR3 sequence analysis
- V/J gene usage patterns

**cd8_scoring.py, imhc_scoring.py, pmhc_scoring.py**
- Specialized scoring for specific cell types or epitope predictions
- Returns features for correlation analysis

**correlations.py**
- Graph construction (TCR and GEX neighbor graphs)
- CoNGA score calculation (overlap between graphs)
- Statistical significance testing

**plotting.py**
- Visualization of results
- Summary plots and logos
- Requires SVG-to-PNG conversion

**tcr_clumping.py**
- Detection of clustered TCR regions
- Statistical testing against null models
- Uses C++ implementations for efficiency

### TCRdist Module (conga/tcrdist/)

Contains distance metric implementation:
- **tcr_distances.py**: Python TCRdist (slower, reference implementation)
- **tcr_distances_blosum.py**: BLOSUM-based variant
- **tcrdist_cpp/**: C++ implementations for production use
- **db/**: Reference databases (gene sequences, all_genes.py references)

## Key Algorithms

### 1. TCRdist Calculation

TCRdist is a weighted Hamming distance that penalizes mismatches in CDR3 regions more heavily than framework regions. For each alpha-beta TCR pair:

```
distance = weight_cdr3 * hamming(cdr3_a1, cdr3_a2) + 
           weight_cdr3 * hamming(cdr3_b1, cdr3_b2) +
           weight_v * (v_gene_distance) +
           weight_j * (j_gene_distance)
```

### 2. Graph Construction

- **TCR graph:** Cells connected if TCRdist below threshold
- **GEX graph:** Cells connected based on gene expression similarity (from scanpy)
- Both are weighted, undirected graphs

### 3. CoNGA Score

Measures overlap between TCR and GEX neighbor graphs:
- High score = cell's TCR neighbors are also GEX neighbors
- Statistically tested using permutation analysis
- Identifies clonotypes with coordinated TCR and GEX variation

### 4. TCR Clumping

Detects whether TCR sequences cluster more than expected by chance:
- Compares observed neighbors at distance threshold
- Against null model (shuffled V/J pairings)
- Uses background TCRdist distributions

## Data Flow in Main Pipeline

```
10x VDJ/GEX data
    ↓
setup_10x_for_conga.py
    ├─ Parse contigs → clones file
    ├─ Compute TCRdist distances
    ├─ KernelPCA reduction
    └─ Create mapping files
    ↓
run_conga.py --analyze
    ├─ Load GEX data (10x_h5, h5ad, etc.)
    ├─ Load clones file
    ├─ Create AnnData object
    ├─ Run graph_vs_graph analysis
    ├─ Run graph_vs_features analysis
    ├─ Optional: TCR clumping
    ├─ Optional: Hotspot analysis
    └─ Generate visualizations
    ↓
Output files (.h5ad, .tsv, .png)
```

## Organism-Specific Handling

Organism parameter controls:
- Reference gene databases (loaded from `conga/data/`)
- CDR3 region boundaries
- Expected V/J gene segments

Supported values: `'human'`, `'mouse'`, `'rhesus'`, `'human_gd'`, `'mouse_gd'`, `'human_ig'`, `'mouse_ig'`

## Performance Considerations

- **C++ TCRdist:** 10-100x faster than Python implementation; strongly recommended for large datasets
- **KernelPCA:** Dimensionality reduction preserves local distances; good for visualization
- **Graph operations:** Use `python-igraph` for efficiency; avoid large dense matrices
- **Memory:** Large datasets may require chunking or clustering before full analysis

## Common Pitfalls

1. **Barcode mismatch:** GEX and TCR barcodes must align exactly (including suffixes from `cellranger aggr`)
2. **Clonotype definition:** Different thresholds for TCRdist condensing can merge or split clonotypes
3. **Graph connectivity:** Too strict distance thresholds isolate cells; too loose creates spurious correlations
4. **Organism mismatch:** Must specify correct organism to use correct reference databases
