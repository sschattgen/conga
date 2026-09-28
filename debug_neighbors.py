#!/usr/bin/env python3
"""
Debug the neighbor differences between FAISS and sklearn.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/sschattg/conga-dev')

from conga.neighbors import compute_neighbor_distances, FaissNeighborSearcher, Backend
from sklearn.metrics import pairwise_distances

def debug_small_example():
    """Debug with a small example to see what's happening."""
    print("Debug small example...")
    
    # Small test case
    np.random.seed(42)
    X = np.random.randn(10, 5).astype(np.float32)
    nbr_fracs = [0.3]  # 3 neighbors
    
    print(f"Data:\n{X}")
    
    # FAISS results 
    searcher = FaissNeighborSearcher(force_backend=Backend.SKLEARN)
    sklearn_from_faiss = searcher.search_neighbors(X, nbr_fracs)
    
    searcher = FaissNeighborSearcher(force_backend=Backend.FAISS_CPU) 
    faiss_result = searcher.search_neighbors(X, nbr_fracs)
    
    # Direct sklearn
    D = pairwise_distances(X, metric='euclidean')
    num_neighbors = max(1, int(0.3 * 10))
    sklearn_direct = np.argpartition(D, num_neighbors - 1)[:, :num_neighbors]
    
    print(f"\nDistance matrix first row: {D[0]}")
    print(f"Sorted indices for row 0: {np.argsort(D[0])}")
    
    print(f"\nSklearn from FAISS: {sklearn_from_faiss.neighbors[0.3][0]}")
    print(f"FAISS CPU result: {faiss_result.neighbors[0.3][0]}")  
    print(f"Direct sklearn: {sklearn_direct[0]}")
    
    # Check if self is included
    for i in range(10):
        faiss_nbrs = faiss_result.neighbors[0.3][i]
        sklearn_nbrs = sklearn_direct[i]
        
        if i in faiss_nbrs:
            print(f"Row {i}: FAISS includes self!")
        if i in sklearn_nbrs:
            print(f"Row {i}: sklearn includes self!")

def debug_self_exclusion():
    """Check if the problem is self-exclusion."""
    print("\n" + "="*40)
    print("Debugging self-exclusion...")
    
    # Create simple data where distances are clear
    X = np.array([
        [0, 0],
        [1, 0], 
        [2, 0],
        [0, 1],
        [0, 2]
    ], dtype=np.float32)
    
    print(f"Simple data:\n{X}")
    
    D = pairwise_distances(X, metric='euclidean')
    print(f"Distance matrix:\n{D}")
    
    # Test FAISS
    searcher = FaissNeighborSearcher()
    result = searcher.search_neighbors(X, [0.4], metric='euclidean')  # 2 neighbors
    
    print(f"FAISS neighbors: {result.neighbors[0.4]}")
    print(f"Backend used: {result.backend_used}")
    
    # Test sklearn path in FAISS
    searcher_sklearn = FaissNeighborSearcher(force_backend=Backend.SKLEARN)
    result_sklearn = searcher_sklearn.search_neighbors(X, [0.4], metric='euclidean')
    
    print(f"FAISS-sklearn neighbors: {result_sklearn.neighbors[0.4]}")
    
    # Check if distances look right
    for i in range(5):
        nbrs = result.neighbors[0.4][i]
        print(f"Point {i} -> neighbors {nbrs}")
        for j in nbrs:
            print(f"  Distance to {j}: {D[i,j]:.3f}")

if __name__ == "__main__":
    debug_small_example()
    debug_self_exclusion()