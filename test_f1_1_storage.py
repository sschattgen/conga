#!/usr/bin/env python3
"""
Test script for task F1.1: AnnData storage functions

Tests the store_tcr_vectors_in_adata, record_active_tcr_representation,
and get_active_tcr_representation functions.
"""

import numpy as np
import pandas as pd
import anndata as ad
import tempfile
import os

import conga
from conga.tcrdist.vectorized import (
    store_tcr_vectors_in_adata, 
    record_active_tcr_representation,
    get_active_tcr_representation,
    EncodingConfig
)
from conga import util

def test_basic_storage():
    """Test basic storage of vectorized TCR data in AnnData."""
    print("Testing basic TCR vector storage...")
    
    # Create test data with valid V gene names and realistic CDR3 sequences
    n_cells = 5
    obs_data = {
        'va': ['TRAV1-1*01', 'TRAV1-2*01', 'TRAV10*01', 'TRAV11*01', 'TRAV12-1*01'],
        'cdr3a': ['CAVRDSNYQLIW', 'CAVKESNYQLIW', 'CAVSTDSNYQLIW', 'CAVQRSNYQLIW', 'CAVLTDSNYQLIW'],
        'vb': ['TRBV1*01', 'TRBV10-1*01', 'TRBV10-2*01', 'TRBV10-3*01', 'TRBV11-1*01'],
        'cdr3b': ['CASSRTGQGDTQYF', 'CASSLQGQGDTQYF', 'CASSPRGTDTQYF', 'CASSYWGQGDTQYF', 'CASSDEGQGDTQYF']
    }
    
    adata = ad.AnnData(obs=pd.DataFrame(obs_data))
    
    # Store vectorized representation
    config = EncodingConfig(aa_mds_dim=8, num_pos_cdr3=12)
    vector_matrix = store_tcr_vectors_in_adata(adata, 'human', config)
    
    # Verify the matrix was stored correctly
    assert util.OBSM_KEY_VEC_TCR in adata.obsm
    stored_matrix = adata.obsm[util.OBSM_KEY_VEC_TCR]
    assert np.array_equal(vector_matrix, stored_matrix)
    assert vector_matrix.shape[0] == n_cells
    assert vector_matrix.dtype == np.float32
    
    # Verify configuration was stored
    assert util.UNS_KEY_VEC_TCR_CONFIG in adata.uns
    stored_config = adata.uns[util.UNS_KEY_VEC_TCR_CONFIG]
    assert stored_config['organism'] == 'human'
    assert stored_config['aa_mds_dim'] == 8
    assert stored_config['num_pos_cdr3'] == 12
    assert 'vectorizer_version' in stored_config
    
    print("✓ Basic storage works")
    return adata


def test_active_representation_tracking():
    """Test active representation recording and retrieval."""
    print("Testing active representation tracking...")
    
    # Create minimal AnnData
    adata = ad.AnnData(obs=pd.DataFrame({'cell_id': [1, 2, 3]}))
    
    # Initially no active representation
    active_rep = get_active_tcr_representation(adata)
    assert active_rep is None
    print("✓ No initial active representation")
    
    # Record vectorized as active
    record_active_tcr_representation(adata, util.OBSM_KEY_VEC_TCR)
    
    # Retrieve and verify
    active_rep = get_active_tcr_representation(adata)
    print(f"After recording: {active_rep}")
    assert active_rep == util.OBSM_KEY_VEC_TCR
    assert util.UNS_KEY_ACTIVE_TCR_REP in adata.uns
    
    print("✓ Active representation recording works")
    
    # Test other representation types
    record_active_tcr_representation(adata, util.OBSM_KEY_PCA_TCR)
    assert get_active_tcr_representation(adata) == util.OBSM_KEY_PCA_TCR
    
    record_active_tcr_representation(adata, util.ACTIVE_REP_EXACT)
    assert get_active_tcr_representation(adata) == util.ACTIVE_REP_EXACT
    
    print("✓ All representation types work")


def test_overwrite_behavior():
    """Test overwriting existing vectorized representation."""
    print("Testing overwrite behavior...")
    
    adata = test_basic_storage()  # Creates initial representation
    
    # Store again with different config - should overwrite with warning
    new_config = EncodingConfig(aa_mds_dim=16, num_pos_cdr3=14)
    new_matrix = store_tcr_vectors_in_adata(adata, 'human', new_config)
    
    # Verify new config stored
    stored_config2 = adata.uns[util.UNS_KEY_VEC_TCR_CONFIG]
    assert stored_config2['aa_mds_dim'] == 16
    assert stored_config2['num_pos_cdr3'] == 14
    
    # Verify matrix updated
    current_matrix = adata.obsm[util.OBSM_KEY_VEC_TCR]
    assert np.array_equal(current_matrix, new_matrix)
    
    print("✓ Overwrite behavior works")


def test_h5ad_persistence():
    """Test persistence through h5ad save/load cycle."""
    print("Testing h5ad persistence...")
    
    # Create and populate AnnData
    adata = test_basic_storage()
    record_active_tcr_representation(adata, util.OBSM_KEY_VEC_TCR)
    
    # Save and reload
    with tempfile.NamedTemporaryFile(suffix='.h5ad', delete=False) as tmp:
        adata.write(tmp.name)
        adata_loaded = ad.read_h5ad(tmp.name)
        os.unlink(tmp.name)
    
    # Verify everything is preserved
    assert util.OBSM_KEY_VEC_TCR in adata_loaded.obsm
    assert util.UNS_KEY_VEC_TCR_CONFIG in adata_loaded.uns
    assert util.UNS_KEY_ACTIVE_TCR_REP in adata_loaded.uns
    
    # Verify matrices are identical
    assert np.array_equal(adata.obsm[util.OBSM_KEY_VEC_TCR], adata_loaded.obsm[util.OBSM_KEY_VEC_TCR])
    
    # Verify active representation
    active_rep_loaded = get_active_tcr_representation(adata_loaded)
    assert active_rep_loaded == util.OBSM_KEY_VEC_TCR
    
    print("✓ h5ad round-trip successful")


def test_coexistence_with_kpca():
    """Test that vectorized and KernelPCA representations can coexist."""
    print("Testing coexistence with KernelPCA...")
    
    # Create AnnData with fake KernelPCA representation
    adata = test_basic_storage()
    
    # Add fake KernelPCA representation
    n_cells = adata.n_obs
    fake_kpca = np.random.randn(n_cells, 50).astype(np.float32)
    adata.obsm[util.OBSM_KEY_PCA_TCR] = fake_kpca
    
    # Store vectorized - should not affect KernelPCA
    config = EncodingConfig(aa_mds_dim=12)
    store_tcr_vectors_in_adata(adata, 'human', config)
    
    # Verify both representations exist
    assert util.OBSM_KEY_VEC_TCR in adata.obsm
    assert util.OBSM_KEY_PCA_TCR in adata.obsm
    
    # Verify KernelPCA unchanged
    assert np.array_equal(adata.obsm[util.OBSM_KEY_PCA_TCR], fake_kpca)
    
    print("✓ Coexistence works")


def test_error_conditions():
    """Test various error conditions."""
    print("Testing error conditions...")
    
    adata = ad.AnnData(obs=pd.DataFrame({'cell_id': [1]}))
    
    # Test invalid active representation
    try:
        record_active_tcr_representation(adata, 'invalid_rep')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert 'Invalid active_representation' in str(e)
        print("✓ Invalid representation rejected")
    
    print("✓ Error conditions handled")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing F1.1: AnnData Storage Functions")
    print("=" * 60)
    
    test_basic_storage()
    test_active_representation_tracking()
    test_overwrite_behavior()
    test_h5ad_persistence()
    test_coexistence_with_kpca()
    test_error_conditions()
    
    print("=" * 60)
    print("✅ All F1.1 tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    main()