#!/usr/bin/env python3
"""
Test script for Task 7.1 - AnnData storage functions for vectorized TCRdist
"""

import numpy as np
import pandas as pd
import anndata as sc
from anndata import AnnData

# Test the new storage functions
def test_storage_functions():
    print("Testing AnnData storage functions for vectorized TCRdist...")
    
    # Import the functions we implemented
    from conga.preprocess import (
        store_tcr_vectors_in_adata, 
        record_active_tcr_representation, 
        get_active_tcr_representation,
        store_tcrs_in_adata
    )
    from conga.tcrdist.vectorized import EncodingConfig
    from conga import util
    
    # Create a minimal AnnData object with TCR data
    n_cells = 100
    n_genes = 1000
    
    # Create fake gene expression data
    X = np.random.rand(n_cells, n_genes).astype(np.float32)
    obs = pd.DataFrame(index=[f'cell_{i}' for i in range(n_cells)])
    var = pd.DataFrame(index=[f'gene_{i}' for i in range(n_genes)])
    
    adata = AnnData(X=X, obs=obs, var=var)
    adata.uns['organism'] = 'human'
    
    # Create fake TCR data (valid human V genes)
    human_va_genes = ['TRAV1-1*01', 'TRAV1-2*01', 'TRAV2*01', 'TRAV3*01']
    human_vb_genes = ['TRBV2*01', 'TRBV3-1*01', 'TRBV4-1*01', 'TRBV5-1*01']
    
    # Create TCR tuples with proper structure: ((va, ja, cdr3a, cdr3a_nucseq), (vb, jb, cdr3b, cdr3b_nucseq))
    tcrs = []
    for i in range(n_cells):
        va = np.random.choice(human_va_genes)
        vb = np.random.choice(human_vb_genes)
        cdr3a = 'C' + ''.join(np.random.choice(list('ACDEFGHIKLMNPQRSTVWY'), 8)) + 'F'
        cdr3b = 'C' + ''.join(np.random.choice(list('ACDEFGHIKLMNPQRSTVWY'), 10)) + 'F'
        
        tcr = ((va, 'TRAJ43*01', cdr3a, 'atgc'), (vb, 'TRBJ1-1*01', cdr3b, 'atgc'))
        tcrs.append(tcr)
    
    # Store TCRs in AnnData using existing function
    store_tcrs_in_adata(adata, tcrs)
    
    print(f"Created test AnnData with {n_cells} cells and TCR data")
    print(f"TCR keys in obs: {[k for k in adata.obs.keys() if k.startswith(('va', 'vb', 'cdr3'))]}")
    
    # Test 1: Store vectorized TCR representation
    print("\nTest 1: Storing vectorized TCR representation...")
    config = EncodingConfig(aa_mds_dim=8, num_pos_cdr3=12)  # Smaller for testing
    
    vector_matrix = store_tcr_vectors_in_adata(adata, config)
    
    print(f"Stored vector matrix shape: {vector_matrix.shape}")
    print(f"Vector matrix dtype: {vector_matrix.dtype}")
    print(f"Keys in adata.obsm: {list(adata.obsm.keys())}")
    print(f"Keys in adata.uns: {[k for k in adata.uns.keys() if 'tcr' in k.lower()]}")
    
    # Verify the matrix was stored correctly
    assert util.OBSM_KEY_VEC_TCR in adata.obsm
    assert vector_matrix.shape[0] == n_cells
    assert vector_matrix.dtype == np.float32
    assert np.all(np.isfinite(vector_matrix))
    
    # Verify configuration was stored
    assert util.UNS_KEY_VEC_TCR_CONFIG in adata.uns
    stored_config = adata.uns[util.UNS_KEY_VEC_TCR_CONFIG]
    assert stored_config['aa_mds_dim'] == 8
    assert stored_config['num_pos_cdr3'] == 12
    assert 'organism' in stored_config
    
    print("✓ Vectorized representation stored successfully")
    
    # Test 2: Record and retrieve active representation
    print("\nTest 2: Recording and retrieving active representation...")
    
    # Initially no active representation recorded
    active_rep = get_active_tcr_representation(adata)
    print(f"Initial active representation (fallback): {active_rep}")
    
    # Record vectorized as active
    record_active_tcr_representation(adata, util.OBSM_KEY_VEC_TCR)
    
    # Retrieve and verify
    active_rep = get_active_tcr_representation(adata)
    print(f"After recording: {active_rep}")
    assert active_rep == util.OBSM_KEY_VEC_TCR
    assert util.UNS_KEY_ACTIVE_TCR_REP in adata.uns
    
    print("✓ Active representation recording works")
    
    # Test 3: Test overwrite warning (capture it visually)
    print("\nTest 3: Testing overwrite warning...")
    
    # Store again - should produce a warning
    vector_matrix2 = store_tcr_vectors_in_adata(adata, config)
    assert np.array_equal(vector_matrix, vector_matrix2)
    
    print("✓ Overwrite warning test completed")
    
    # Test 4: Test with different config
    print("\nTest 4: Testing with different configuration...")
    
    config2 = EncodingConfig(aa_mds_dim=16, num_pos_cdr3=14, random_seed=123)
    vector_matrix3 = store_tcr_vectors_in_adata(adata, config2)
    
    # Should be different shape due to different config
    assert vector_matrix3.shape != vector_matrix.shape
    print(f"New vector matrix shape: {vector_matrix3.shape}")
    
    # Verify new config stored
    stored_config2 = adata.uns[util.UNS_KEY_VEC_TCR_CONFIG]
    assert stored_config2['aa_mds_dim'] == 16
    assert stored_config2['num_pos_cdr3'] == 14
    assert stored_config2['random_seed'] == 123
    
    print("✓ Different configuration handled correctly")
    
    # Test 5: Test round-trip through h5ad file
    print("\nTest 5: Testing h5ad round-trip...")
    
    # Save and reload
    test_file = '/tmp/test_vectorized_tcrdist.h5ad'
    adata.write_h5ad(test_file)
    
    adata_loaded = sc.read_h5ad(test_file)
    
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
    
    # Clean up
    import os
    os.remove(test_file)
    
    print("\n✅ All tests passed! Task 7.1 implementation is working correctly.")
    
    return adata, vector_matrix3

if __name__ == '__main__':
    try:
        test_storage_functions()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()