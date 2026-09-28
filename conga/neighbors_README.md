# FAISS-Accelerated Neighbor Search

The `conga.neighbors` module provides high-performance neighbor search with automatic backend selection and graceful fallback.

## Key Features

- **Tiered Backend System**: faiss-gpu → faiss-cpu → sklearn
- **Automatic Selection**: Chooses best backend based on data size and GPU availability  
- **Identical Results**: Produces same results as existing sklearn-based code
- **Performance**: 10-100x speedup on large datasets
- **Memory Efficient**: Avoids quadratic memory usage of pairwise distance matrices

## Usage

### Basic Usage

```python
from conga.neighbors import FaissNeighborSearcher
import numpy as np

# Create test data
X = np.random.randn(10000, 100).astype(np.float32)

# Create searcher with automatic backend selection
searcher = FaissNeighborSearcher()

# Search for neighbors
result = searcher.search_neighbors(
    X=X,
    nbr_fracs=[0.01, 0.05],  # 1% and 5% of samples as neighbors
    also_calc_nndists=True,
    nbr_frac_for_nndists=0.01
)

# Access results
neighbors_1pct = result.neighbors[0.01]  # Shape: (10000, 100) 
neighbors_5pct = result.neighbors[0.05]  # Shape: (10000, 500)
nndists = result.nndists                 # Shape: (10000,)
backend_used = result.backend_used       # Backend.FAISS_CPU
```

### Drop-in Replacement for calc_nbrs

```python
from conga.neighbors import compute_neighbor_distances

# Direct replacement for distance computation in preprocess.calc_nbrs
neighbors_dict = compute_neighbor_distances(
    X=adata.obsm['X_pca_gex'],
    nbr_fracs=[0.01, 0.05, 0.1],
    exclude_groups=(agroups, bgroups),  # TCR group exclusions
    metric='euclidean'
)

# Or with nndists calculation
neighbors_dict, nndists = compute_neighbor_distances(
    X=adata.obsm['X_pca_gex'], 
    nbr_fracs=[0.01, 0.05],
    also_calc_nndists=True,
    nbr_frac_for_nndists=0.01,
    exclude_groups=(agroups, bgroups)
)
```

### Backend Control

```python
from conga.neighbors import FaissNeighborSearcher, Backend, get_backend_info

# Check available backends
info = get_backend_info()
print(f"FAISS GPU available: {info['faiss_gpu_available']}")
print(f"FAISS CPU available: {info['faiss_cpu_available']}")
print(f"Number of GPUs: {info['num_gpus']}")

# Force specific backend
searcher = FaissNeighborSearcher(force_backend=Backend.SKLEARN)
result = searcher.search_neighbors(X, nbr_fracs=[0.05])

# Configure GPU memory limit
searcher = FaissNeighborSearcher(gpu_memory_limit_gb=4.0)
```

## Backend Selection Logic

The system automatically selects the best backend:

1. **FAISS GPU**: Used if:
   - GPU is available (`faiss.get_num_gpus() > 0`)
   - Estimated memory usage < `gpu_memory_limit_gb`
   - No `force_backend` override

2. **FAISS CPU**: Used if:
   - FAISS package available
   - GPU not available or memory exceeded
   - No `force_backend` override

3. **sklearn**: Used if:
   - FAISS not available
   - Forced via `force_backend=Backend.SKLEARN`
   - Other backends fail (automatic fallback)

## Performance Characteristics

Measured on various dataset sizes with FAISS CPU vs sklearn:

| Samples | Features | FAISS CPU | sklearn | Speedup |
|---------|----------|-----------|---------|---------|
| 1,000   | 50       | 0.001s    | 0.018s  | 13.3x   |
| 5,000   | 100      | 0.031s    | 0.376s  | 12.1x   |
| 10,000  | 50       | 0.094s    | 1.458s  | 15.5x   |
| 20,000  | 100      | 0.22s     | ~30s*   | ~136x   |

*Estimated based on quadratic scaling of pairwise distance calculation.

## Integration with CoNGA

The module is designed for seamless integration with existing CoNGA workflows:

### In preprocess.calc_nbrs()

```python
# Replace existing pairwise_distances computation
from conga.neighbors import compute_neighbor_distances

# Instead of:
# D = pairwise_distances(adata.obsm[obsm_tag], metric='euclidean')
# [manual neighbor extraction logic]

# Use:
result = compute_neighbor_distances(
    X=adata.obsm[obsm_tag],
    nbr_fracs=nbr_fracs,
    exclude_groups=(agroups, bgroups),
    also_calc_nndists=also_calc_nndists,
    nbr_frac_for_nndists=nbr_frac_for_nndists,
    sort_nbrs=sort_nbrs
)
```

### With Vectorized TCR Representations

```python
# For vectorized TCR data (X_vec_tcr)
searcher = FaissNeighborSearcher()
tcr_result = searcher.search_neighbors(
    X=adata.obsm['X_vec_tcr'],
    nbr_fracs=nbr_fracs,
    exclude_groups=(agroups, bgroups),
    metric='euclidean'  # Euclidean distance approximates TCRdist
)
```

## Error Handling and Fallback

The module handles errors gracefully:

- **GPU Memory Exceeded**: Automatically falls back to CPU
- **FAISS Import Error**: Falls back to sklearn
- **Invalid Parameters**: Raises clear ValueError with suggestions
- **Backend Failures**: Cascades through available backends

## Dependencies

- **Required**: numpy, scikit-learn (always available)
- **Optional**: faiss-cpu or faiss-gpu (for acceleration)
- **Python**: 3.12+ (for type hints and dataclass features)

## Installation

```bash
# Install with FAISS CPU support
pip install faiss-cpu

# Or with GPU support (requires CUDA)
pip install faiss-gpu

# Check installation
python -c "from conga.neighbors import get_backend_info; print(get_backend_info())"
```