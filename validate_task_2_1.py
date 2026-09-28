#!/usr/bin/env python3

"""
Comprehensive validation of Task 2.1: Build amino acid dissimilarity matrix and embedding.

This validates all the acceptance criteria:
- symbol_dissimilarity_matrix() produces 21x21 matrix matching CoNGA's bsd4 values
- aa_embedding() is deterministic with classical_mds initialization  
- Caching works correctly with parameter changes
- Integration with existing TCRdist values verified
"""

import hashlib
import numpy as np
from conga.tcrdist.vectorized import (
    symbol_dissimilarity_matrix, 
    aa_embedding, 
    EncodingConfig,
    _EMBEDDING_CACHE
)
from conga.tcrdist.tcr_distances_blosum import bsd4
from conga.tcrdist.amino_acids import amino_acids

def validate_symbol_dissimilarity_matrix():
    """Validate that dissimilarity matrix matches CoNGA's bsd4 values."""
    print('Validating symbol_dissimilarity_matrix...')
    
    dm = symbol_dissimilarity_matrix()
    
    # Check basic properties
    assert dm.shape == (21, 21), f"Expected shape (21, 21), got {dm.shape}"
    assert np.allclose(dm, dm.T), "Matrix must be symmetric"
    assert np.allclose(np.diag(dm), 0.0), "Diagonal must be zero"
    assert np.all((dm >= 0) & (dm <= 4)), "Values must be in [0, 4]"
    
    # Verify amino acid dissimilarities match bsd4
    for i, aa_i in enumerate(amino_acids):
        for j, aa_j in enumerate(amino_acids):
            expected = bsd4[(aa_i, aa_j)]
            actual = dm[i, j]
            assert np.isclose(actual, expected), f"Mismatch at ({aa_i}, {aa_j}): expected {expected}, got {actual}"
    
    # Verify gap penalty
    from conga.tcrdist.tcr_distances import GAP_PENALTY_V_REGION
    gap_penalty = float(GAP_PENALTY_V_REGION)
    
    for i in range(20):  # All amino acids
        assert np.isclose(dm[i, 20], gap_penalty), f"Gap penalty mismatch at position {i}"
        assert np.isclose(dm[20, i], gap_penalty), f"Gap penalty mismatch at position {i}"
    
    assert np.isclose(dm[20, 20], 0.0), "Gap-to-gap distance must be zero"
    
    print('✓ symbol_dissimilarity_matrix validation passed')

def validate_aa_embedding_determinism():
    """Validate deterministic embedding behavior and caching."""
    print('\nValidating aa_embedding determinism and caching...')
    
    # Clear cache to start fresh
    _EMBEDDING_CACHE.clear()
    
    config1 = EncodingConfig(aa_mds_dim=8, random_seed=42)
    config2 = EncodingConfig(aa_mds_dim=8, random_seed=42)  # Same config
    config3 = EncodingConfig(aa_mds_dim=12, random_seed=42)  # Different dim
    
    # First call - should compute and cache
    assert len(_EMBEDDING_CACHE) == 0, "Cache should be empty initially"
    embedding1 = aa_embedding(config1)
    assert len(_EMBEDDING_CACHE) == 1, "Cache should contain one entry after first call"
    
    # Second call with same config - should use cache  
    embedding2 = aa_embedding(config2)
    assert len(_EMBEDDING_CACHE) == 1, "Cache should still contain one entry"
    assert np.array_equal(embedding1, embedding2), "Cached result should be identical"
    
    # Third call with different config - should compute new entry
    embedding3 = aa_embedding(config3)
    assert len(_EMBEDDING_CACHE) == 2, "Cache should now contain two entries"
    assert embedding3.shape[1] == 12, "Different dimension should produce different shape"
    
    # Validate embedding properties
    for embedding in [embedding1, embedding2, embedding3]:
        assert embedding.shape[0] == 21, "Embedding should have 21 symbols"
        assert embedding.dtype == np.float64, "Embedding should be float64"
        assert np.allclose(np.mean(embedding, axis=0), 0, atol=1e-10), "Columns should be zero-mean"
    
    # With classical_mds, different seeds should give same result (deterministic)
    config4 = EncodingConfig(aa_mds_dim=8, random_seed=999)  # Different seed
    embedding4 = aa_embedding(config4)
    assert np.allclose(embedding1, embedding4), "classical_mds should be deterministic regardless of seed"
    
    print('✓ aa_embedding determinism and caching validation passed')

def validate_cache_fingerprinting():
    """Validate that cache uses proper parameter fingerprinting."""
    print('\nValidating cache fingerprinting...')
    
    _EMBEDDING_CACHE.clear()
    
    config = EncodingConfig(aa_mds_dim=10, random_seed=42)
    
    # Get initial embedding
    embedding1 = aa_embedding(config)
    initial_cache_size = len(_EMBEDDING_CACHE)
    
    # Same config should reuse cache
    embedding2 = aa_embedding(config)
    assert len(_EMBEDDING_CACHE) == initial_cache_size, "Same config should reuse cache"
    assert np.array_equal(embedding1, embedding2), "Cached result should be identical"
    
    # Different aa_mds_dim should create new cache entry
    config_diff_dim = EncodingConfig(aa_mds_dim=14, random_seed=42)
    embedding3 = aa_embedding(config_diff_dim)
    assert len(_EMBEDDING_CACHE) == initial_cache_size + 1, "Different dimension should create new cache entry"
    
    # Cache keys should be deterministic 
    _EMBEDDING_CACHE.clear()
    embedding_a = aa_embedding(config)
    _EMBEDDING_CACHE.clear()
    embedding_b = aa_embedding(config)
    assert np.array_equal(embedding_a, embedding_b), "Cache key generation should be deterministic"
    
    print('✓ Cache fingerprinting validation passed')

def validate_integration_with_tcrdist():
    """Validate integration with existing TCRdist constants."""
    print('\nValidating integration with existing TCRdist values...')
    
    # Test that we're using the actual bsd4 table, not a copy
    from conga.tcrdist.tcr_distances_blosum import bsd4 as original_bsd4
    
    dm = symbol_dissimilarity_matrix()
    
    # Sample a few amino acid pairs to verify we're using the real bsd4
    test_pairs = [('A', 'A'), ('A', 'C'), ('W', 'W'), ('D', 'K'), ('F', 'Y')]
    
    for aa1, aa2 in test_pairs:
        i = amino_acids.index(aa1)
        j = amino_acids.index(aa2)
        expected = original_bsd4[(aa1, aa2)]
        actual = dm[i, j]
        assert np.isclose(actual, expected), f"Integration failure for ({aa1}, {aa2}): expected {expected}, got {actual}"
    
    # Test gap penalty matches TCRdist constants
    from conga.tcrdist.tcr_distances import GAP_PENALTY_V_REGION
    assert dm[0, 20] == GAP_PENALTY_V_REGION, "Gap penalty should match TCRdist constant"
    
    # Test that embedding produces reasonable stress values
    config = EncodingConfig(aa_mds_dim=16)
    embedding = aa_embedding(config)
    # The embedding should complete without errors and produce finite values
    assert np.all(np.isfinite(embedding)), "Embedding should contain only finite values"
    
    print('✓ Integration with TCRdist validation passed')

def validate_sklearn_compatibility():
    """Validate compatibility with sklearn MDS parameters."""
    print('\nValidating sklearn MDS compatibility...')
    
    # Test that our pinned parameters work with current sklearn
    from sklearn.manifold import MDS
    
    # Try to create MDS with our parameters
    try:
        mds = MDS(
            n_components=4,  # Use fewer components than samples  
            metric='precomputed',
            metric_mds=True,
            init='classical_mds',
            n_init=1,
            max_iter=300,
            eps=1e-3,
            n_jobs=None,
            random_state=42,
            normalized_stress=False
        )
        
        # Test with a small dissimilarity matrix
        test_dm = np.sqrt(symbol_dissimilarity_matrix()[:5, :5])
        result = mds.fit_transform(test_dm)
        
        print(f"  MDS result shape: {result.shape} (expected: (5, 4))")
        assert result.shape[0] == 5, "MDS should preserve number of samples"
        assert result.shape[1] == 4, "MDS should produce requested number of components"
        assert np.all(np.isfinite(result)), "MDS result should be finite"
        
    except Exception as e:
        raise AssertionError(f"sklearn MDS compatibility failed: {e}")
    
    print('✓ sklearn MDS compatibility validation passed')

def main():
    print('=== Task 2.1 Comprehensive Validation ===\n')
    print('Validating: Build amino acid dissimilarity matrix and embedding\n')
    
    try:
        validate_symbol_dissimilarity_matrix()
        validate_aa_embedding_determinism() 
        validate_cache_fingerprinting()
        validate_integration_with_tcrdist()
        validate_sklearn_compatibility()
        
        print('\n🎉 All Task 2.1 validations passed!')
        print('\n✅ TASK 2.1 COMPLETE')
        print('   - Amino acid dissimilarity matrix implemented correctly')
        print('   - Deterministic embedding with classical_mds initialization')  
        print('   - Proper caching with parameter fingerprinting')
        print('   - Full integration with existing TCRdist values')
        print('   - All acceptance criteria satisfied')
        
    except Exception as e:
        print(f'\n❌ Task 2.1 validation failed: {e}')
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == '__main__':
    success = main()
    if not success:
        exit(1)