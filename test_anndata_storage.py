#!/usr/bin/env python3
"""
Test script for the new AnnData storage functions.
"""

import sys
import os
sys.path.insert(0, '/Users/sschattg/conga-dev')

import numpy as np
import pandas as pd
import anndata as ad
from conga.tcrdist.vectorized import (
    store_vectorized_tcr_in_adata, 
    load_vectorized_tcr_from_adata,
    clear_vectorized_tcr_from_adata,
    EncodingConfig
)

def test_basic_functionality():
    """Test basic store/load/clear functionality."""
    print("Testing basic AnnData storage functionality...")
    
    # Create test AnnData with minimal TCR data
    obs = pd.DataFrame({
        'va': ['TRAV1-1*01', 'TRAV1-2*01'],
        'cdr3a': ['CAVRDTIGYKYVF', 'CAVKETIGYKYVF'],
        'vb': ['TRBV1*01', 'TRBV10-1*01'],
        'cdr3b': ['CASSRTGQPQHF', 'CASSLQGQPQHF']
    })
    
    adata = ad.AnnData(
        X=np.random.randn(2, 100),  # Mock gene expression data
        obs=obs
    )
    
    print(f"Created test AnnData: {adata.n_obs} obs x {adata.n_vars} vars")
    
    # Test storage
    print("Testing store_vectorized_tcr_in_adata()...")
    matrix = store_vectorized_tcr_in_adata(adata, 'human')
    print(f"Stored matrix shape: {matrix.shape}")
    
    # Verify storage
    assert 'X_vec_tcr' in adata.obsm, "Vector matrix not stored in obsm"
    assert 'vec_tcr_config' in adata.uns, "Config not stored in uns"
    print("✓ Storage successful")
    
    # Test loading
    print("Testing load_vectorized_tcr_from_adata()...")
    loaded_matrix, loaded_config, loaded_organism = load_vectorized_tcr_from_adata(adata)
    print(f"Loaded matrix shape: {loaded_matrix.shape}")
    print(f"Loaded organism: {loaded_organism}")
    print(f"Loaded config aa_mds_dim: {loaded_config.aa_mds_dim}")
    
    # Verify loading
    assert np.array_equal(matrix, loaded_matrix), "Loaded matrix differs from stored"
    assert loaded_organism == 'human', "Loaded organism incorrect"
    assert loaded_config.aa_mds_dim == 16, "Loaded config incorrect"
    print("✓ Loading successful")
    
    # Test clearing
    print("Testing clear_vectorized_tcr_from_adata()...")
    was_present = clear_vectorized_tcr_from_adata(adata)
    assert was_present == True, "Should report data was present"
    assert 'X_vec_tcr' not in adata.obsm, "Vector matrix still in obsm after clearing"
    assert 'vec_tcr_config' not in adata.uns, "Config still in uns after clearing"
    print("✓ Clearing successful")
    
    # Test clearing when no data present
    was_present_2 = clear_vectorized_tcr_from_adata(adata)
    assert was_present_2 == False, "Should report no data present on second clear"
    print("✓ Double clear handled correctly")
    
    print("All basic functionality tests passed!\n")


def test_custom_config():
    """Test with custom configuration."""
    print("Testing custom configuration...")
    
    obs = pd.DataFrame({
        'va': ['TRAV1-1*01'],
        'cdr3a': ['CAVRDTIGYKYVF'],
        'vb': ['TRBV1*01'], 
        'cdr3b': ['CASSRTGQPQHF']
    })
    
    adata = ad.AnnData(X=np.random.randn(1, 10), obs=obs)
    
    # Custom config with different parameters
    config = EncodingConfig(aa_mds_dim=8, num_pos_cdr3=12, random_seed=123)
    
    matrix = store_vectorized_tcr_in_adata(adata, 'human', config)
    print(f"Custom config matrix shape: {matrix.shape}")
    
    loaded_matrix, loaded_config, loaded_organism = load_vectorized_tcr_from_adata(adata)
    
    # Verify custom config was preserved
    assert loaded_config.aa_mds_dim == 8, "Custom aa_mds_dim not preserved"
    assert loaded_config.num_pos_cdr3 == 12, "Custom num_pos_cdr3 not preserved"
    assert loaded_config.random_seed == 123, "Custom random_seed not preserved"
    print("✓ Custom configuration preserved correctly")
    
    print("Custom configuration test passed!\n")


def test_error_handling():
    """Test error handling for invalid inputs."""
    print("Testing error handling...")
    
    # Create AnnData without vectorized data
    empty_adata = ad.AnnData(X=np.random.randn(2, 10))
    
    # Test loading from empty AnnData
    try:
        load_vectorized_tcr_from_adata(empty_adata)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "No vectorized TCR representation found" in str(e)
        print("✓ Loading from empty AnnData raises appropriate error")
    
    # Test clearing from empty AnnData (should not error)
    was_present = clear_vectorized_tcr_from_adata(empty_adata)
    assert was_present == False
    print("✓ Clearing from empty AnnData handles gracefully")
    
    print("Error handling tests passed!\n")


if __name__ == "__main__":
    try:
        test_basic_functionality()
        test_custom_config()
        test_error_handling()
        print("🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)