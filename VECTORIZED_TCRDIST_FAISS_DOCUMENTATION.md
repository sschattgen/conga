# Vectorized TCRdist + FAISS Acceleration: Comprehensive Documentation

## Overview

This document provides comprehensive documentation for the vectorized TCRdist encoding and FAISS acceleration features implemented in CoNGA. These complementary optimizations deliver 10-100x performance improvements and >50% memory reduction for large single-cell datasets.

## Module Documentation Summary

### conga.tcrdist.vectorized 

**Purpose**: Replace quadratic KernelPCA TCR representation with linear-memory vectorized encoding

**Key Features**:
- Fixed-length vector encoding of paired α/β TCR chains
- Near-exact TCRdist approximation (Spearman > 0.999, recall@10 > 0.95)
- O(N·L) memory scaling vs O(N²) for KernelPCA
- Deterministic encoding with configurable parameters
- FAISS-optimized output format (float32, C-contiguous)

**Supported Organisms**: human, mouse, rhesus (alpha-beta TCRs only)

### conga.neighbors

**Purpose**: FAISS-accelerated neighbor search with graceful fallback

**Key Features**:
- Tiered backend system: faiss-gpu → faiss-cpu → sklearn
- Production-grade error handling and automatic fallback
- Optimized index selection based on data characteristics
- Compatible API with existing calc_nbrs() workflow
- Comprehensive performance monitoring and logging

## Performance Characteristics

### Vectorized TCRdist Encoding

**Benchmark Conditions**: 20,000 clonotypes, 2019 MacBook Pro, default configuration

| Metric | Vectorized | KernelPCA | Improvement |
|--------|-----------|-----------|-------------|
| Encoding time | 0.1s | ~300s* | 3000x |
| Peak memory | ~150 MB | ~6.4 GB | 43x reduction |
| Output size | 91 MB | 8 MB | 11x larger† |
| Memory scaling | O(N·L) | O(N²) | Sub-quadratic |

*KernelPCA time includes TCRdist matrix computation  
†Larger storage but enables fast neighbor search without recomputation

### FAISS Neighbor Search

**Performance by Dataset Size**:

| Dataset Size | Backend | GEX Speedup | TCR Speedup | Memory Usage |
|-------------|---------|-------------|-------------|--------------|
| < 5k cells | FAISS-CPU | 2-5x | 2-3x | Similar |
| 5k-50k cells | FAISS-CPU | 10-50x | 10-30x | 30-50% reduction |
| 5k-50k cells | FAISS-GPU | 20-100x | 20-60x | 40-70% reduction |
| > 50k cells | FAISS-GPU | 50-200x | 50-150x | 50-80% reduction |

### Accuracy Validation

**Vectorized Encoding Accuracy** (1000 human TCRs, default config):

| Metric | Value | Requirement | Status |
|--------|-------|-------------|--------|
| Spearman correlation | 0.999 | ≥ 0.95 | ✅ Pass |
| Pearson (distance) | 0.993 | - | ✅ Excellent |
| Pearson (squared distance) | 0.999 | - | ✅ Near-exact |
| Recall@10 | 0.953 | ≥ 0.80 | ✅ Pass |
| Recall@100 | 0.971 | - | ✅ Excellent |

**Cross-organism Performance**:

| Organism | Spearman | Recall@10 | Vector Length |
|----------|----------|-----------|---------------|
| Human | 0.999 | 0.953 | 1136 |
| Mouse | 0.999 | 0.944 | 1168 |
| Rhesus | 0.999 | 0.947 | 1152 |

## API Usage Guide

### Basic Vectorized Encoding

```python
from conga.tcrdist.vectorized import encode_tcrs, EncodingConfig

# Basic usage with default configuration
tcrs = [
    (('TRAV1*01', 'TRAJ1*01', 'CAVRD', ''), ('TRBV1*01', 'TRBJ1*01', 'CASSRT', '')),
    (('TRAV2*01', 'TRAJ2*01', 'CAVKE', ''), ('TRBV2*01', 'TRBJ2*01', 'CASSLQ', ''))
]
matrix = encode_tcrs(tcrs, 'human')  # Shape: (2, 1136)

# Custom configuration for performance/accuracy tradeoff
config = EncodingConfig(aa_mds_dim=12, num_pos_cdr3=14)
matrix = encode_tcrs(tcrs, 'human', config)  # Shape: (2, 852)
```

### FAISS Neighbor Search

```python
from conga.neighbors import search_neighbors_auto, FaissNeighborSearcher

# Automatic backend selection (recommended)
result = search_neighbors_auto(
    X=adata.obsm['X_vec_tcr'], 
    nbr_fracs=[0.01, 0.05],
    data_type='tcr'
)
neighbors = result.neighbors[0.01]
print(f"Backend used: {result.backend_used.value}")

# Advanced configuration
searcher = FaissNeighborSearcher(
    gpu_memory_limit_gb=8.0,
    batch_size=32768
)
result = searcher.search_neighbors(
    X=large_dataset,
    nbr_fracs=[0.02],
    also_calc_nndists=True
)
```

### Integration with CoNGA Workflow

```python
import anndata as ad
from conga.tcrdist.vectorized import store_tcr_vectors_in_adata
from conga.neighbors import search_neighbors_auto

# Store vectorized TCR representation
adata = ad.read_h5ad('my_data.h5ad')
vectors = store_tcr_vectors_in_adata(adata, 'human')

# Use FAISS for neighbor search
tcr_result = search_neighbors_auto(
    X=adata.obsm['X_vec_tcr'],
    nbr_fracs=[0.02],
    data_type='tcr'
)

gex_result = search_neighbors_auto(
    X=adata.obsm['X_pca'], 
    nbr_fracs=[0.01, 0.05],
    data_type='gex'
)
```

## Configuration Guidelines

### Vectorized TCRdist Configuration

**EncodingConfig Parameters**:

| Parameter | Default | Purpose | Recommendations |
|-----------|---------|---------|-----------------|
| `aa_mds_dim` | 16 | Amino acid embedding dimensions | 16: near-exact, 12: good+smaller, 8: minimal |
| `num_pos_cdr3` | 16 | Fixed CDR3 length | 16: standard, adjust for specific repertoires |
| `cdr3_weight` | 3.0 | CDR3 vs germline weighting | Keep at 3.0 for TCRdist consistency |
| `n_trim`, `c_trim` | 3, 2 | CDR3 trimming | Match existing TCRdist parameters |
| `random_seed` | 42 | MDS reproducibility | Use consistent value across analyses |

**When to Adjust**:
- **Performance priority**: `aa_mds_dim=12` for 25% smaller vectors
- **Memory constrained**: `aa_mds_dim=8` but validate accuracy
- **Long CDR3s**: Increase `num_pos_cdr3` to 20-24
- **Reproducibility**: Set `random_seed` consistently

### FAISS Configuration

**FaissNeighborSearcher Parameters**:

| Parameter | Default | Purpose | Recommendations |
|-----------|---------|---------|-----------------|
| `gpu_memory_limit_gb` | 4.0 | GPU memory threshold | Increase to 8-12 for large GPUs |
| `batch_size` | 16384 | Processing batch size | Reduce if memory-limited |
| `adaptive_parameters` | True | Data-specific optimization | Keep enabled |
| `force_backend` | None | Override selection | Use for testing only |

**Index Selection** (automatic, but configurable via FaissIndexConfig):
- **Small datasets** (< 10k): Flat indices (exact results)
- **Medium datasets** (10k-50k): IVF indices with adaptive nlist
- **Large datasets** (> 50k): IVF + PCA preprocessing for high-dimensional data

## Error Handling and Troubleshooting

### Common Issues and Solutions

#### Vectorized Encoding Issues

**"Organism 'human_gd' not supported"**
- **Cause**: Vectorizer only supports alpha-beta TCRs
- **Solution**: Use `--use_kpca_tcrdist` or `--no_kpca` flags for gamma-delta/Ig

**"CDR3 too short after trimming"**
- **Cause**: CDR3 length < n_trim + c_trim + 1
- **Solution**: Adjust trimming parameters or filter short CDR3s

**"V gene not found in database"**
- **Cause**: Gene identifier mismatch or missing reference
- **Solution**: Check gene naming conventions, update gene database

#### FAISS Backend Issues

**GPU Memory Errors**:
```
FaissGpuMemoryError: Dataset too large for GPU: 8.5GB > 4.0GB limit
```
- **Solution**: Increase `gpu_memory_limit_gb` or use CPU backend
- **Prevention**: Monitor dataset size, use batch processing

**CUDA Driver Issues**:
```
FaissCudaError: CUDA runtime error
```
- **Solutions**: 
  - Check NVIDIA driver installation
  - Verify GPU availability: `nvidia-smi`
  - Update FAISS to compatible version
  - Use CPU fallback automatically applied

**Index Building Failures**:
```
FaissIndexBuildError: IVF training failed
```
- **Solutions**:
  - Check for NaN/infinite values in data
  - Reduce nlist parameter for small datasets
  - Use flat index as fallback

### Diagnostic Commands

**Check Backend Availability**:
```python
from conga.neighbors import get_backend_info
info = get_backend_info()
print(f"FAISS-GPU: {info['faiss_gpu_available']}")
print(f"FAISS-CPU: {info['faiss_cpu_available']}")
if info['detection_errors']:
    for component, error in info['detection_errors'].items():
        print(f"{component}: {error}")
```

**Validate Encoding Accuracy**:
```python
from conga.tcrdist.vectorized import accuracy_report
report = accuracy_report(test_tcrs, 'human')
assert report.spearman >= 0.95, f"Accuracy gate failed: {report.spearman}"
assert report.mean_recall[10] >= 0.80, f"Recall gate failed: {report.mean_recall[10]}"
```

**Monitor Performance**:
```python
searcher = FaissNeighborSearcher()
# ... run searches ...
history = searcher.get_performance_history()
for key, metrics in history.items():
    data_type, backend, shape = key
    print(f"{data_type} {backend.value}: {metrics['samples_per_second']:.0f} samples/sec")
```

## Migration Guide

### From KernelPCA to Vectorized

**Before (KernelPCA)**:
```bash
# Setup generates KernelPCA files
python scripts/setup_10x_for_conga.py --clones_file clones.tsv --organism human

# Analysis uses X_pca_tcr
python scripts/run_conga.py --adata data.h5ad
```

**After (Vectorized, default for α/β organisms)**:
```bash
# Setup skips KernelPCA for supported organisms
python scripts/setup_10x_for_conga.py --clones_file clones.tsv --organism human

# Analysis automatically uses X_vec_tcr
python scripts/run_conga.py --adata data.h5ad
```

**Override to KernelPCA if needed**:
```bash
python scripts/run_conga.py --adata data.h5ad --use_kpca_tcrdist --kpca_reduction_limit 50000
```

### From sklearn to FAISS

**No code changes required** - FAISS acceleration is automatic when available:

1. **Install FAISS**: `pip install faiss-cpu` or `pip install faiss-gpu`
2. **Run existing workflow** - FAISS used automatically for large datasets
3. **Monitor logs** for backend selection and performance metrics

**Force specific backend for testing**:
```python
from conga.neighbors import FaissNeighborSearcher, Backend

# Test FAISS-GPU specifically
searcher = FaissNeighborSearcher(force_backend=Backend.FAISS_GPU)
result = searcher.search_neighbors(X, nbr_fracs=[0.01])
```

## Best Practices

### For Large Datasets (N > 20k)

1. **Use vectorized TCRdist** (automatic for α/β organisms)
2. **Install FAISS-GPU** if available for maximum performance
3. **Monitor memory usage** and adjust `gpu_memory_limit_gb`
4. **Enable performance logging** to optimize parameters

### For Production Workflows

1. **Pin configuration parameters** for reproducibility
2. **Set consistent random seeds** across analyses  
3. **Validate accuracy** on representative subset
4. **Enable comprehensive logging** for debugging
5. **Test fallback paths** in CI/staging environments

### For Development and Testing

1. **Use smaller datasets** for rapid iteration
2. **Force sklearn backend** for reference comparisons
3. **Test with both CPU and GPU FAISS** if available
4. **Validate against exact TCRdist** on small datasets

## Integration Points

### Command Line Interface

**New flags in `run_conga.py`**:
- `--use_kpca_tcrdist`: Force KernelPCA for α/β organisms
- `--kpca_reduction_limit N`: Adjust KernelPCA size limit  
- `--aa_mds_dim N`: Configure vectorized encoding dimensions
- `--use_faiss_gpu`, `--use_faiss_cpu`, `--disable_faiss`: Backend control

### Programmatic API

**Key entry points**:
- `conga.tcrdist.vectorized.encode_tcrs()`: Direct encoding
- `conga.tcrdist.vectorized.store_tcr_vectors_in_adata()`: AnnData integration
- `conga.neighbors.search_neighbors_auto()`: Automatic neighbor search
- `conga.neighbors.FaissNeighborSearcher()`: Advanced configuration

### AnnData Integration

**Storage conventions**:
- Vectorized TCRs: `adata.obsm['X_vec_tcr']`
- KernelPCA TCRs: `adata.obsm['X_pca_tcr']` 
- Active representation: `adata.uns['active_tcr_representation']`
- Encoding config: `adata.uns['vec_tcr_config']`

## Future Enhancements

### Planned Extensions
- Gamma-delta TCR vectorization
- B cell receptor (Ig) support
- Batch effect correction integration
- Additional distance metrics (cosine, etc.)

### Performance Optimizations  
- Mixed-precision encoding (float16 support)
- Sparse vector representations for memory efficiency
- Distributed encoding for very large datasets
- Integration with GPU clusters

### Analysis Features
- Direct FAISS integration in CoNGA pipeline
- Vector-based TCR clustering algorithms  
- Similarity search against public databases
- Real-time neighbor queries for interactive analysis

## Conclusion

The vectorized TCRdist and FAISS acceleration features provide substantial performance improvements for CoNGA while maintaining backward compatibility and accuracy. The comprehensive documentation, error handling, and configuration options ensure reliable operation across diverse computational environments and datasets.

For questions or issues, refer to the module docstrings, enable debug logging, and use the diagnostic functions provided in both modules.