"""
Test configuration and utilities for vectorized TCRdist + FAISS testing.
"""

import os
import pytest
from typing import Dict, Any
import anndata as ad
import pandas as pd

# Test data paths
TEST_FIXTURES_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')

# Test configuration
TEST_CONFIG = {
    'random_seed': 42,
    'tolerance': {
        'accuracy_spearman': 0.99,    # Minimum Spearman correlation with exact TCRdist
        'accuracy_recall': 0.90,      # Minimum recall@10 for neighbor accuracy
        'performance_speedup': 2.0,   # Minimum speedup vs sklearn for FAISS
    },
    'timeouts': {
        'small_dataset': 30,          # seconds
        'medium_dataset': 120,        # seconds
        'large_dataset': 300,         # seconds
    },
    'backends_to_test': ['sklearn', 'faiss_cpu'],  # Add 'faiss_gpu' if available
}

# Test data specifications
TEST_DATASETS = {
    'minimal': {
        'n_cells': 100,
        'n_genes': 500,
        'description': 'Minimal dataset for basic functionality testing'
    },
    'small': {
        'n_cells': 500,
        'n_genes': 1000,
        'description': 'Small dataset, should use sklearn backend'
    },
    'medium': {
        'n_cells': 5000,
        'n_genes': 2000,
        'description': 'Medium dataset, should use FAISS CPU'
    },
    'large': {
        'n_cells': 20000,
        'n_genes': 3000,
        'description': 'Large dataset, should prefer FAISS GPU if available'
    }
}


@pytest.fixture(scope="session")
def test_fixtures_dir():
    """Return path to test fixtures directory."""
    return TEST_FIXTURES_DIR


@pytest.fixture(scope="session") 
def minimal_adata():
    """Load minimal test AnnData."""
    return ad.read_h5ad(os.path.join(TEST_FIXTURES_DIR, 'minimal_adata.h5ad'))


@pytest.fixture(scope="session")
def minimal_clones():
    """Load minimal test clones data."""
    return pd.read_csv(os.path.join(TEST_FIXTURES_DIR, 'minimal_clones.tsv'), sep='\t')


@pytest.fixture(scope="session")
def edge_case_clones():
    """Load edge case clones data."""
    return pd.read_csv(os.path.join(TEST_FIXTURES_DIR, 'edge_case_clones.tsv'), sep='\t')


@pytest.fixture(scope="session")
def invalid_gene_clones():
    """Load invalid gene clones data."""
    return pd.read_csv(os.path.join(TEST_FIXTURES_DIR, 'invalid_gene_clones.tsv'), sep='\t')


@pytest.fixture(params=list(TEST_DATASETS.keys()))
def performance_adata(request):
    """Parametrized fixture for different sized datasets."""
    dataset_name = request.param
    filepath = os.path.join(TEST_FIXTURES_DIR, f'performance_{dataset_name}_adata.h5ad')
    return ad.read_h5ad(filepath), TEST_DATASETS[dataset_name]


def check_faiss_availability():
    """Check which FAISS backends are available."""
    available_backends = ['sklearn']  # Always available
    
    try:
        import faiss
        available_backends.append('faiss_cpu')
        
        # Check for GPU
        if hasattr(faiss, 'StandardGpuResources'):
            try:
                gpu_res = faiss.StandardGpuResources()
                available_backends.append('faiss_gpu')
                del gpu_res
            except Exception:
                pass  # GPU not available
    except ImportError:
        pass  # FAISS not available
    
    return available_backends


# Update test configuration with available backends
AVAILABLE_BACKENDS = check_faiss_availability()
TEST_CONFIG['available_backends'] = AVAILABLE_BACKENDS
