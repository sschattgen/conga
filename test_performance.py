#!/usr/bin/env python3
"""
Performance test for FAISS vs sklearn neighbor search.
"""

import numpy as np
import time
import sys
sys.path.insert(0, '/Users/sschattg/conga-dev')

from conga.neighbors import FaissNeighborSearcher, Backend, get_backend_info

def benchmark_backends():
    """Benchmark FAISS vs sklearn performance."""
    print("Performance Benchmark: FAISS vs sklearn")
    print("="*50)
    
    # Test different dataset sizes
    test_sizes = [
        (1000, 50),   # Small
        (5000, 50),   # Medium 
        (10000, 50),  # Large
    ]
    
    nbr_fracs = [0.01, 0.05]
    
    print(f"Backend info: {get_backend_info()}")
    print()
    
    for n_samples, n_features in test_sizes:
        print(f"Testing dataset: {n_samples} samples x {n_features} features")
        
        # Generate test data
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        
        # Test FAISS (CPU)
        if get_backend_info()['faiss_cpu_available']:
            searcher_faiss = FaissNeighborSearcher(force_backend=Backend.FAISS_CPU)
            
            start_time = time.time()
            faiss_result = searcher_faiss.search_neighbors(X, nbr_fracs)
            faiss_time = time.time() - start_time
            
            print(f"  FAISS CPU: {faiss_time:.3f} seconds")
        else:
            faiss_time = float('inf')
            print(f"  FAISS CPU: Not available")
        
        # Test sklearn
        searcher_sklearn = FaissNeighborSearcher(force_backend=Backend.SKLEARN)
        
        start_time = time.time()
        sklearn_result = searcher_sklearn.search_neighbors(X, nbr_fracs)
        sklearn_time = time.time() - start_time
        
        print(f"  sklearn:   {sklearn_time:.3f} seconds")
        
        # Calculate speedup
        if faiss_time < float('inf'):
            speedup = sklearn_time / faiss_time
            print(f"  Speedup:   {speedup:.1f}x")
        
        print()

def test_memory_usage():
    """Test memory efficiency for different backends."""
    import tracemalloc
    
    print("Memory Usage Test")
    print("="*30)
    
    # Medium-sized dataset
    n_samples, n_features = 5000, 50
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    nbr_fracs = [0.05]
    
    backends = [Backend.SKLEARN]
    if get_backend_info()['faiss_cpu_available']:
        backends.append(Backend.FAISS_CPU)
    
    for backend in backends:
        print(f"Testing {backend.value}...")
        
        tracemalloc.start()
        
        searcher = FaissNeighborSearcher(force_backend=backend)
        result = searcher.search_neighbors(X, nbr_fracs)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"  Peak memory: {peak / 1024 / 1024:.1f} MB")
        print(f"  Current:     {current / 1024 / 1024:.1f} MB")
        print()

if __name__ == "__main__":
    benchmark_backends()
    test_memory_usage()