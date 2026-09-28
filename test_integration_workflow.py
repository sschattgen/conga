#!/usr/bin/env python3
"""
Integration test showing how the new AnnData storage functions integrate 
with the CoNGA workflow for Task 7.1.
"""

import numpy as np
import pandas as pd
import anndata as sc
from anndata import AnnData

def test_integration_workflow():
    """Test integration with CoNGA workflow patterns."""
    print("Testing integration with CoNGA workflow...")
    
    # Import functions
    from conga.preprocess import (
        store_tcr_vectors_in_adata, 
        record_active_tcr_representation, 
        get_active_tcr_representation,
        resolve_tcr_representation,
        store_tcrs_in_adata
    )
    from conga.tcrdist.vectorized import EncodingConfig
    from conga import util
    
    # Create test data with different organism scenarios
    test_scenarios = [
        {'organism': 'human', 'n_obs': 1000, 'expected_default': util.OBSM_KEY_VEC_TCR},
        {'organism': 'mouse', 'n_obs': 500, 'expected_default': util.OBSM_KEY_VEC_TCR},
        {'organism': 'human_gd', 'n_obs': 15000, 'expected_default': util.OBSM_KEY_PCA_TCR},  # Below limit  
        {'organism': 'human_gd', 'n_obs': 25000, 'expected_default': util.ACTIVE_REP_EXACT},  # Above limit
    ]
    
    for i, scenario in enumerate(test_scenarios):
        print(f"\n--- Scenario {i+1}: {scenario['organism']}, N={scenario['n_obs']} ---")
        
        # Create AnnData for this scenario
        adata = create_test_adata(scenario['n_obs'], scenario['organism'])
        
        # Test the resolver
        rep = resolve_tcr_representation(
            organism=scenario['organism'],
            num_obs=scenario['n_obs'],
            stored_obsm_keys=set()
        )
        
        print(f"Resolved representation: {rep.active}")
        print(f"Reason: {rep.reason}")
        print(f"Build vectorized: {rep.build_vectorized}")
        print(f"Build KPCA: {rep.build_kpca}")
        
        # Verify expected default
        assert rep.active == scenario['expected_default']
        
        # Record the active representation
        record_active_tcr_representation(adata, rep.active)
        
        # If vectorized is selected, test the storage
        if rep.build_vectorized:
            print("Building vectorized representation...")
            config = EncodingConfig(aa_mds_dim=8, num_pos_cdr3=12)  # Small for testing
            vector_matrix = store_tcr_vectors_in_adata(adata, config)
            
            print(f"Vectorized matrix shape: {vector_matrix.shape}")
            assert vector_matrix.shape[0] == scenario['n_obs']
            assert util.OBSM_KEY_VEC_TCR in adata.obsm
            
        # Test restart logic
        print("Testing restart logic...")
        stored_keys = set(adata.obsm.keys()) 
        
        restart_rep = resolve_tcr_representation(
            organism=scenario['organism'],
            num_obs=scenario['n_obs'],
            stored_obsm_keys=stored_keys
        )
        
        print(f"Restart representation: {restart_rep.active}")
        print(f"Restart reason: {restart_rep.reason}")
        
        # Should reuse stored representation
        if rep.build_vectorized:
            assert restart_rep.active == util.OBSM_KEY_VEC_TCR
            assert not restart_rep.build_vectorized  # Should not rebuild
        
        # Verify active representation retrieval  
        retrieved_rep = get_active_tcr_representation(adata)
        assert retrieved_rep == rep.active
    
    print("\n--- Testing Override Scenarios ---")
    
    # Test explicit overrides
    human_adata = create_test_adata(1000, 'human')
    
    # Test KernelPCA override
    kpca_rep = resolve_tcr_representation(
        organism='human',
        num_obs=1000, 
        request_kpca=True
    )
    assert kpca_rep.active == util.OBSM_KEY_PCA_TCR
    assert kpca_rep.build_kpca == True
    print("✓ KernelPCA override works")
    
    # Test exact override
    exact_rep = resolve_tcr_representation(
        organism='human',
        num_obs=1000,
        request_exact_nbrs=True
    )
    assert exact_rep.active == util.ACTIVE_REP_EXACT
    assert exact_rep.obsm_tag_tcr is None
    assert exact_rep.use_exact_tcrdist_nbrs == True
    print("✓ Exact TCRdist override works")
    
    # Test conflicting overrides (should raise error)
    try:
        resolve_tcr_representation(
            organism='human',
            num_obs=1000,
            request_kpca=True,
            request_exact_nbrs=True
        )
        assert False, "Should have raised ValueError for conflicting overrides"
    except ValueError as e:
        print("✓ Conflicting overrides correctly rejected")
    
    # Test KernelPCA above limit (should raise error)
    try:
        resolve_tcr_representation(
            organism='human',
            num_obs=25000,  # Above default limit
            request_kpca=True
        )
        assert False, "Should have raised ValueError for KPCA above limit"
    except ValueError as e:
        print("✓ KernelPCA above limit correctly rejected")
    
    print("\n✅ All integration tests passed!")


def create_test_adata(n_obs, organism):
    """Create minimal test AnnData with TCR data."""
    from conga.preprocess import store_tcrs_in_adata
    
    # Create fake expression data
    X = np.random.rand(n_obs, 100).astype(np.float32)
    obs = pd.DataFrame(index=[f'cell_{i}' for i in range(n_obs)])
    var = pd.DataFrame(index=[f'gene_{i}' for i in range(100)])
    
    adata = AnnData(X=X, obs=obs, var=var)
    adata.uns['organism'] = organism
    
    # Create appropriate TCR data based on organism
    if organism.startswith('human'):
        va_genes = ['TRAV1-1*01', 'TRAV1-2*01', 'TRAV2*01']
        vb_genes = ['TRBV2*01', 'TRBV3-1*01', 'TRBV4-1*01']
    else:  # mouse
        va_genes = ['TRAV1*01', 'TRAV10*01', 'TRAV11*01']
        vb_genes = ['TRBV1*01', 'TRBV10*01', 'TRBV12-1*01']
    
    tcrs = []
    for i in range(n_obs):
        va = np.random.choice(va_genes)
        vb = np.random.choice(vb_genes)
        cdr3a = 'C' + ''.join(np.random.choice(list('ACDEFGHIKLMNPQRSTVWY'), 8)) + 'F'
        cdr3b = 'C' + ''.join(np.random.choice(list('ACDEFGHIKLMNPQRSTVWY'), 10)) + 'F'
        
        tcr = ((va, 'TRAJ43*01', cdr3a, 'atgc'), (vb, 'TRBJ1-1*01', cdr3b, 'atgc'))
        tcrs.append(tcr)
    
    store_tcrs_in_adata(adata, tcrs)
    return adata


if __name__ == '__main__':
    try:
        test_integration_workflow()
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()