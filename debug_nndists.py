#!/usr/bin/env python3
"""
Debug the nndists differences between FAISS and sklearn.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/sschattg/conga-dev')

from conga.neighbors import compute_neighbor_distances, FaissNeighborSearcher, Backend
from sklearn.metrics import pairwise_distances

def debug_nndists():
    """Debug nndists calculation differences."""
    print("Debug nndists calculation...")
    
    # Simple test case where we can trace the calculation
    np.random.seed(42)
    X = np.random.randn(10, 5).astype(np.float32)
    nbr_fracs = [0.3]  # 3 neighbors
    
    print(f"Data shape: {X.shape}")
    
    # Get both results
    faiss_result = compute_neighbor_distances(
        X=X, nbr_fracs=nbr_fracs, also_calc_nndists=True, 
        nbr_frac_for_nndists=0.3, metric='euclidean'
    )
    faiss_neighbors, faiss_nndists = faiss_result
    
    # Sklearn reference
    D = pairwise_distances(X, metric='euclidean')
    num_neighbors = 3
    sklearn_neighbors = np.argpartition(D, num_neighbors - 1)[:, :num_neighbors]
    
    # Manual nndist calculation for sklearn
    def calc_nndists_manual(D, nbrs):
        batch_size, num_nbrs = nbrs.shape
        sample_range = np.arange(batch_size)[:, np.newaxis]
        nbrs_sorted = nbrs[sample_range, np.argsort(D[sample_range, nbrs])]
        D_nbrs_sorted = D[sample_range, nbrs_sorted]
        wts = np.linspace(1.0, 1.0/num_nbrs, num_nbrs)
        wts /= np.sum(wts)
        nndists = np.sum(D_nbrs_sorted * wts[np.newaxis,:], axis=1)
        return nndists, nbrs_sorted, D_nbrs_sorted, wts
    
    sklearn_nndists, sklearn_sorted_nbrs, sklearn_sorted_dists, weights = calc_nndists_manual(D, sklearn_neighbors)
    
    print(f"Weights: {weights}")
    print(f"FAISS nndists[0]: {faiss_nndists[0]:.6f}")
    print(f"Sklearn nndists[0]: {sklearn_nndists[0]:.6f}")
    print(f"Difference: {abs(faiss_nndists[0] - sklearn_nndists[0]):.6f}")
    
    # Detailed analysis for point 0
    print(f"\nPoint 0 analysis:")
    print(f"FAISS neighbors: {faiss_neighbors[0.3][0]}")
    print(f"Sklearn neighbors: {sklearn_neighbors[0]}")
    print(f"Sklearn sorted neighbors: {sklearn_sorted_nbrs[0]}")
    
    print(f"Sklearn sorted distances: {sklearn_sorted_dists[0]}")
    print(f"Weighted calculation: {np.sum(sklearn_sorted_dists[0] * weights):.6f}")
    
    # Check FAISS neighbor distances
    faiss_nbrs = faiss_neighbors[0.3][0]
    faiss_dists = D[0, faiss_nbrs]
    faiss_sorted_indices = np.argsort(faiss_dists)
    faiss_sorted_dists = faiss_dists[faiss_sorted_indices]
    
    print(f"FAISS distances: {faiss_dists}")
    print(f"FAISS sorted distances: {faiss_sorted_dists}")
    print(f"FAISS weighted calc: {np.sum(faiss_sorted_dists * weights):.6f}")

if __name__ == "__main__":
    debug_nndists()