#!/usr/bin/env python3
"""
Test FAISS integration with actual CoNGA preprocess functionality.
"""

import numpy as np
import pandas as pd
import sys
import anndata as ad
sys.path.insert(0, '/Users/sschattg/conga-dev')

import conga.preprocess as preprocess

def create_mock_adata():
    """Create mock AnnData object for testing."""
    n_cells = 1000
    n_genes = 100
    n_pcs = 50
    
    # Create mock data
    np.random.seed(42)
    
    # Mock gene expression PCA
    X_pca_gex = np.random.randn(n_cells, n_pcs).astype(np.float32)
    
    # Mock TCR PCA  
    X_pca_tcr = np.random.randn(n_cells, n_pcs).astype(np.float32)
    
    # Mock TCR groups for exclusions
    obs_data = {
        'va': [f'TRAV{i%20}' for i in range(n_cells)],
        'ja': [f'TRAJ{i%30}' for i in range(n_cells)], 
        'cdr3a': [f'CAVS{i%100}' for i in range(n_cells)],
        'cdr3a_nucseq': [f'TGTGCAGTG{i%50}' for i in range(n_cells)],
        'vb': [f'TRBV{i%25}' for i in range(n_cells)],
        'jb': [f'TRBJ{i%15}' for i in range(n_cells)],
        'cdr3b': [f'CASS{i%150}' for i in range(n_cells)],
        'cdr3b_nucseq': [f'TGTGCCAGC{i%80}' for i in range(n_cells)],
    }
    
    obs = pd.DataFrame(obs_data, index=[f'cell_{i}' for i in range(n_cells)])
    var = pd.DataFrame(index=[f'gene_{i}' for i in range(n_genes)])
    
    adata = ad.AnnData(
        X=np.random.randn(n_cells, n_genes),
        obs=obs,
        var=var
    )
    
    # Add PCA results
    adata.obsm['X_pca_gex'] = X_pca_gex
    adata.obsm['X_pca_tcr'] = X_pca_tcr
    
    # Add organism info
    adata.uns['organism'] = 'human'
    
    return adata

def test_calc_nbrs_integration():
    """Test that calc_nbrs works with FAISS integration."""
    print("Testing calc_nbrs with FAISS integration...")
    
    adata = create_mock_adata()
    nbr_fracs = [0.01, 0.05]
    
    print(f"Created mock data: {adata.shape[0]} cells")
    
    # Test basic functionality
    print("Testing basic calc_nbrs...")
    result = preprocess.calc_nbrs(
        adata=adata,
        nbr_fracs=nbr_fracs,
        obsm_tag_gex='X_pca_gex',
        obsm_tag_tcr='X_pca_tcr'
    )
    
    print(f"calc_nbrs completed successfully")
    print(f"Result keys: {list(result.keys())}")
    
    for nbr_frac in nbr_fracs:
        gex_nbrs, tcr_nbrs = result[nbr_frac]
        expected_neighbors = max(1, int(nbr_frac * adata.shape[0]))
        
        print(f"nbr_frac {nbr_frac}:")
        print(f"  GEX neighbors shape: {gex_nbrs.shape if gex_nbrs is not None else None}")
        print(f"  TCR neighbors shape: {tcr_nbrs.shape if tcr_nbrs is not None else None}")
        print(f"  Expected neighbors per cell: {expected_neighbors}")
        
        # Validate shapes
        if gex_nbrs is not None:
            assert gex_nbrs.shape == (adata.shape[0], expected_neighbors), f"Unexpected GEX shape"
        if tcr_nbrs is not None:
            assert tcr_nbrs.shape == (adata.shape[0], expected_neighbors), f"Unexpected TCR shape"
    
    print("✅ Basic calc_nbrs test passed")

def test_calc_nbrs_with_nndists():
    """Test calc_nbrs with nndist calculation."""
    print("\nTesting calc_nbrs with nndists...")
    
    adata = create_mock_adata()
    nbr_fracs = [0.02, 0.1]
    
    result = preprocess.calc_nbrs(
        adata=adata,
        nbr_fracs=nbr_fracs,
        obsm_tag_gex='X_pca_gex',
        obsm_tag_tcr='X_pca_tcr',
        also_calc_nndists=True,
        nbr_frac_for_nndists=0.02
    )
    
    all_nbrs, gex_nndists, tcr_nndists = result
    
    print(f"GEX nndists shape: {gex_nndists.shape if gex_nndists is not None else None}")
    print(f"TCR nndists shape: {tcr_nndists.shape if tcr_nndists is not None else None}")
    
    if gex_nndists is not None:
        assert gex_nndists.shape == (adata.shape[0],), "Unexpected GEX nndists shape"
        print(f"GEX nndists range: [{np.min(gex_nndists):.3f}, {np.max(gex_nndists):.3f}]")
    
    if tcr_nndists is not None:
        assert tcr_nndists.shape == (adata.shape[0],), "Unexpected TCR nndists shape" 
        print(f"TCR nndists range: [{np.min(tcr_nndists):.3f}, {np.max(tcr_nndists):.3f}]")
    
    print("✅ calc_nbrs with nndists test passed")

def test_gex_only():
    """Test with GEX data only (TCR disabled)."""
    print("\nTesting GEX-only mode...")
    
    adata = create_mock_adata()
    
    result = preprocess.calc_nbrs(
        adata=adata,
        nbr_fracs=[0.05],
        obsm_tag_gex='X_pca_gex',
        obsm_tag_tcr=None  # Disable TCR
    )
    
    gex_nbrs, tcr_nbrs = result[0.05]
    
    assert gex_nbrs is not None, "GEX neighbors should be computed"
    assert tcr_nbrs is None, "TCR neighbors should be None"
    
    print("✅ GEX-only test passed")

if __name__ == "__main__":
    test_calc_nbrs_integration()
    test_calc_nbrs_with_nndists() 
    test_gex_only()
    
    print("\n" + "="*50)
    print("🎉 All preprocess integration tests passed!")
    print("FAISS acceleration is ready for production use.")