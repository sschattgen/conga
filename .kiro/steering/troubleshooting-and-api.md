# Troubleshooting Guide and API Reference

## Common Issues and Solutions

### Data Loading Issues

#### Problem: "Barcode mismatch between GEX and TCR data"

**Cause:** Cell barcodes in gene expression data don't match TCR clones file. This commonly happens with `cellranger aggr` outputs where barcodes get suffixes.

**Solution:**
```python
# Use make_10x_clone_file_batch to align barcodes
import conga
import pandas as pd

# Create metadata mapping file (CSV with 'file' and 'batch_id' columns)
metadata = pd.DataFrame({
    'file': ['sample_1/filtered_contig_annotations.csv',
             'sample_2/filtered_contig_annotations.csv'],
    'batch_id': [1, 2]
})
metadata.to_csv('metadata.csv', index=False)

# Run batch processing
conga.tcrdist.make_10x_clones_file.make_10x_clones_file_batch(
    'metadata.csv', 'human', 'output_clones.tsv'
)
```

#### Problem: "File not found" when loading h5ad

**Cause:** Path is incorrect or file was moved.

**Solution:**
```python
import os
import anndata as ad

# Verify file exists
data_file = 'path/to/data.h5ad'
if not os.path.exists(data_file):
    raise FileNotFoundError(f"Data file not found: {data_file}")

# Load with error handling
try:
    adata = ad.read_h5ad(data_file)
except Exception as e:
    print(f"Error loading file: {e}")
```

### TCRdist Calculation Issues

#### Problem: "TCRdist computation is very slow"

**Cause:** Using Python implementation instead of C++ version.

**Solution:**
1. Compile C++ code:
   ```bash
   cd conga/tcrdist_cpp
   make
   ```

2. Verify compilation succeeded:
   ```bash
   ls -la bin/find_neighbors bin/calc_distributions
   ```

3. The code will automatically use C++ if available; Python falls back if not.

#### Problem: "Invalid organism: 'human_xx'"

**Cause:** Organism string is misspelled or unsupported.

**Solution:**
```python
# Valid organism strings
valid_organisms = [
    'human', 'mouse', 'rhesus',           # TCRαβ
    'human_gd', 'mouse_gd', 'rhesus_gd',  # TCRγδ
    'human_ig', 'mouse_ig'                # B cells (Ig)
]

# Use one of these; note organism parameter is case-sensitive
adata = conga.load_and_process_data(
    clones_file='clones.tsv',
    gex_data='gex.h5ad',
    organism='human'  # Must be exact
)
```

### Memory and Performance Issues

#### Problem: "MemoryError" when analyzing large dataset

**Cause:** Dataset too large to fit in memory.

**Solutions:**
1. Subset the data first:
   ```python
   import anndata as ad
   adata = ad.read_h5ad('large_data.h5ad')
   adata_subset = adata[adata.obs['cell_type'] == 'CD8T', :].copy()
   # Process subset
   ```

2. Use chunked processing in scripts:
   ```bash
   # Process in batches by clonotype cluster
   python scripts/run_conga.py --clones_file clones.tsv \
       --gex_data gex.h5ad --organism human --outfile_prefix out_batch1
   ```

3. Increase available memory or use a machine with more RAM.

#### Problem: "Analysis step X is taking too long"

**Cause:** Inefficient algorithm choice or data size.

**Solutions:**
- Use C++ implementations (already compiled as described above)
- Subset data to most interesting cells/clonotypes
- Reduce graph connectivity threshold to reduce edge count
- Profile to identify bottleneck:
  ```python
  import cProfile, pstats
  profiler = cProfile.Profile()
  profiler.enable()
  
  # ... analysis code ...
  
  profiler.disable()
  stats = pstats.Stats(profiler)
  stats.sort_stats('cumulative')
  stats.print_stats(30)
  ```

### Visualization Issues

#### Problem: "SVG to PNG conversion fails"

**Cause:** Missing or incorrectly configured SVG conversion tool.

**Supported tools (in priority order):**
- ImageMagick `convert`: `brew install imagemagick`
- Inkscape: `brew install inkscape` (macOS) or conda
- cairosvg: `pip install cairosvg`

**Solution:**
```bash
# macOS
brew install imagemagick

# Or use conda
conda install -c conda-forge imagemagick

# Verify installation
convert --version
```

Edit `conga/convert_svg_to_png.py` if tool is installed but not found:
```python
# Modify PATH_TO_INKSCAPE or other tool paths if needed
PATH_TO_INKSCAPE = "/usr/local/bin/inkscape"  # Update if needed
```

#### Problem: "Fonts look wrong in logos"

**Cause:** Monospace font not available on system.

**Solution:** Edit `conga/convert_svg_to_png.py`:
```python
# Try different monospace fonts
MONOSPACE_FONT_FAMILY = "Courier"  # or "DejaVu Sans Mono", "Liberation Mono"
```

### Statistical Issues

#### Problem: "CoNGA scores are all near zero"

**Possible causes:**
1. TCR and GEX graphs are not correlated (biological - might be correct)
2. Graph parameters not optimized for dataset size
3. Clonotype cluster assignments not meaningful

**Troubleshooting:**
```python
# Check graph properties
import scanpy as sc
import numpy as np

print(f"Number of cells: {adata.n_obs}")
print(f"Number of clonotypes: {adata.obs['clonotype_id'].nunique()}")
print(f"Clonotype sizes: {adata.obs['clonotype_id'].value_counts().describe()}")

# Check graph connectivity
n_neighbors_tcr = adata.obsp['tcr_distances'].nnz
n_neighbors_gex = adata.obsp['gex_distances'].nnz
print(f"TCR graph edges: {n_neighbors_tcr}")
print(f"GEX graph edges: {n_neighbors_gex}")

# If graphs are too sparse, adjust parameters
sc.pp.neighbors(adata, n_neighbors=15, use_rep='X')  # Try different n_neighbors
```

#### Problem: "Statistical tests produce p-value = NaN"

**Cause:** Numerical instability in permutation test (usually too few cells or extreme values).

**Solution:**
```python
# Ensure sufficient clonotype size
min_clonotype_size = 5
adata_filtered = adata[
    adata.obs.groupby('clonotype_id').transform('size') >= min_clonotype_size
].copy()

# Check for extreme values
print(adata.obs.describe())
print(adata.X.min(), adata.X.max())

# Re-run analysis on filtered data
```

## API Quick Reference

### Main Analysis Functions

```python
import conga

# Load example data
adata = conga.preprocess.load_10x_data(
    gex_data='filtered_gene_bc_matrices_h5.h5',
    clones_file='filtered_contig_annotations_tcrdist_clones.tsv',
    organism='human'
)

# Compute TCR-GEX correlation
conga.correlations.get_graph_pairs(adata, organism='human')

# TCR clumping analysis
conga.tcr_clumping.assess_tcr_clumping(adata, organism='human')

# TCR feature extraction
conga.tcr_scoring.make_tcr_features(adata, organism='human')

# Plotting
conga.plotting.plot_summary(adata, outfile_prefix='results/summary')
```

### Common AnnData Operations

```python
import anndata as ad
import pandas as pd

# Load data
adata = ad.read_h5ad('data.h5ad')

# Subset by metadata
adata_cd8 = adata[adata.obs['cell_type'] == 'CD8T'].copy()

# Access data
genes = adata.var_names
cells = adata.obs_names
expression_matrix = adata.X

# Add metadata
adata.obs['new_column'] = pd.Series(
    data=values,
    index=adata.obs_names
)

# Save
adata.write_h5ad('output.h5ad')
```

### TCRdist Distance Calculation

```python
from conga.tcrdist import tcr_distances

# Distance between two sequences
dist = tcr_distances.tcr_distance(
    tcra1='CAVXXXXXX', tcrb1='CASSYXXXX',
    tcra2='CAVXXXXXX', tcrb2='CASSYXXXX',
    organism='human'
)

# Distance matrix for multiple sequences
sequences = [
    ('CAVXXXXXX', 'CASSYXXXX'),
    ('CAVYYYYYY', 'CASSYXXXX'),
    # ...
]
distances = tcr_distances.pairwise_distances(sequences, organism='human')
```

### Logging Configuration

```python
import logging

# Set up logging for development
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress verbose scanpy messages
logging.getLogger('scanpy').setLevel(logging.WARNING)
```

## Performance Benchmarks

Typical runtimes on 10x datasets (macOS, 2019 MacBook Pro):

| Dataset | Cells | Step | Time |
|---------|-------|------|------|
| Human PBMC | 10k | setup_10x_for_conga.py | ~30s |
| Human PBMC | 10k | run_conga.py --all | ~2-3 min |
| Mouse PBMC | 7k | setup_10x_for_conga.py | ~20s |
| Mouse PBMC | 7k | run_conga.py --all | ~1-2 min |

Note: Actual times depend on hardware, clonotype distribution, and specific analysis options.

## File Format Reference

### Clones File (.tsv)

Tab-separated file with TCR information:
```
clonotype_id  cell_barcode  v_a  j_a  cdr3_a  v_b  j_b  cdr3_b  ...
clone_1       AAACCCAGTCTA  TRAV28  TRAJ43  CAVRXXXXX  TRBV20-1  TRBJ1-3  CASSYXXXXXX
clone_1       AAACCCAGTCTG  TRAV28  TRAJ43  CAVRXXXXX  TRBV20-1  TRBJ1-3  CASSYXXXXXX
```

### Results TSV Files

Various output formats depending on analysis:
- `*_conga_scores.tsv`: Per-clonotype CoNGA scores
- `*_graph_vs_features.tsv`: Feature correlations
- `*_tcr_clumping.tsv`: TCR clumping results
- `*_hotspot_features.tsv`: Hotspot analysis results

## Getting Help

### Documentation
- Official paper: https://www.nature.com/articles/s41587-021-00989-2
- BioRxiv preprint: https://www.biorxiv.org/content/10.1101/2020.06.04.134536v1
- GitHub issues: https://github.com/phbradley/conga/issues

### Useful Tools
- **Scanpy:** https://scanpy.readthedocs.io/
- **AnnData:** https://anndata.readthedocs.io/
- **IPython:** For interactive debugging in notebooks

### Debugging Strategy
1. Check input data (dimensions, barcodes, metadata)
2. Verify organism parameter matches data type
3. Run small subset first to verify logic
4. Enable debug logging to trace execution
5. Check intermediate output files (.tsv, .h5ad)
