#!/usr/bin/env python3
"""
Test to understand the original calc_nbrs behavior with argpartition.
"""

import numpy as np
from sklearn.metrics import pairwise_distances

def test_original_argpartition():
    """Test the original argpartition behavior."""
    
    # Simple test case
    X = np.array([
        [0, 0],
        [1, 0], 
        [2, 0],
        [0, 1],
        [0, 2]
    ], dtype=np.float32)
    
    print(f"Test data:\n{X}")
    
    D = pairwise_distances(X, metric='euclidean')
    print(f"Distance matrix:\n{D}")
    
    num_neighbors = 2
    print(f"Requesting {num_neighbors} neighbors")
    
    # Original approach: argpartition(D, num_neighbors-1)[:, :num_neighbors]
    original = np.argpartition(D, num_neighbors - 1)[:, :num_neighbors]
    print(f"Original approach result:\n{original}")
    
    # Check if self is included
    for i in range(5):
        if i in original[i]:
            print(f"Row {i}: includes self (index {i})")
        else:
            print(f"Row {i}: does NOT include self")
    
    print("\nChecking distances for row 0:")
    print(f"Row 0 neighbors: {original[0]}")
    for j in original[0]:
        print(f"  Distance to {j}: {D[0, j]:.3f}")
        
    # Let's see what argpartition actually does
    print(f"\nFull argpartition result for row 0: {np.argpartition(D[0], num_neighbors-1)}")
    print(f"All distances for row 0: {D[0]}")
    print(f"Sorted order: {np.argsort(D[0])}")

if __name__ == "__main__":
    test_original_argpartition()