#!/usr/bin/env python3
"""
Integration test for task F1.1: AnnData storage functions with existing CoNGA workflows

Tests that the new storage functions integrate properly with existing CoNGA preprocessing.
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

def create_realistic_test_data(n_cells=20, organism='human'):
    """Create more realistic test data similar to what CoNGA processes."""
    
    if organism == 'human':
        # Use real V genes from the database (confirmed valid)
        va_genes = ['TRAV1-1*01', 'TRAV1-2*01', 'TRAV10*01', 'TRAV11*01', 'TRAV12-1*01',
                    'TRAV12-2*01', 'TRAV13-1*01', 'TRAV13-2*01', 'TRAV16*01', 'TRAV17*01']
        vb_genes = ['TRBV1*01', 'TRBV10-1*01', 'TRBV10-2*01', 'TRBV10-3*01', 'TRBV11-1*01',
                    'TRBV11-2*01', 'TRBV11-3*01', 'TRBV12-1*01', 'TRBV12-2*01', 'TRBV13*01']
    elif organism == 'mouse':
        # Mouse gene names (different pattern)
        va_genes = ['TRAV1*01', 'TRAV1*02', 'TRAV10*01', 'TRAV11*01', 'TRAV12-1*01',
                    'TRAV12-2*01', 'TRAV13*01', 'TRAV14*01', 'TRAV16*01', 'TRAV17*01']
        vb_genes = ['TRBV1*01', 'TRBV12-1*01', 'TRBV12-2*01', 'TRBV13-1*01', 'TRBV13-2*01',
                    'TRBV13-3*01', 'TRBV14*01', 'TRBV15*01', 'TRBV16*01', 'TRBV17*01']
    elif organism == 'rhesus':
        # Rhesus - use human-like naming but validate separately
        va_genes = ['TRAV1-1*01', 'TRAV1-2*01', 'TRAV10*01', 'TRAV11*01', 'TRAV12-1*01',
                    'TRAV12-2*01', 'TRAV13-1*01', 'TRAV13-2*01', 'TRAV16*01', 'TRAV17*01']
        vb_genes = ['TRBV1*01', 'TRBV10-1*01', 'TRBV10-2*01', 'TRBV10-3*01', 'TRBV11-1*01',
                    'TRBV11-2*01', 'TRBV11-3*01', 'TRBV12-1*01', 'TRBV12-2*01', 'TRBV13*01']
    else:
        raise ValueError(f"Organism {organism} not supported in test data creation")
    
    # Realistic CDR3 sequences of varying lengths
    cdr3a_seqs = [
        'CAVRDSNYQLIW', 'CAVKESNYQLIW', 'CAVSTDSNYQLIW', 'CAVQRSNYQLIW', 'CAVLTDSNYQLIW',
        'CAVRDEFNYQLIW', 'CAVKEFNYQLIW', 'CAVSTFNYQLIW', 'CAVQRFNYQLIW', 'CAVLTFNYQLIW',
        'CAVRDNYQLIW', 'CAVKENYQLIW', 'CAVSTYQLIW', 'CAVQRYQLIW', 'CAVLTYQLIW',
        'CAVRDGNYQLIW', 'CAVKEGNYQLIW', 'CAVSTGNYQLIW', 'CAVQRGNYQLIW', 'CAVLTGNYQLIW'
    ]
    
    cdr3b_seqs = [
        'CASSRTGQGDTQYF', 'CASSLQGQGDTQYF', 'CASSPRGTDTQYF', 'CASSYWGQGDTQYF', 'CASSDEGQGDTQYF',
        'CASSRTGDTQYF', 'CASSLQGDTQYF', 'CASSPRGTQYF', 'CASSYWGDTQYF', 'CASSDEGDTQYF',
        'CASSRTQQYF', 'CASSLQQYF', 'CASSPREQYF', 'CASSYWEQYF', 'CASSDEQYF',
        'CASSRTGGQYF', 'CASSLQGGQYF', 'CASSPRGGQYF', 'CASSYWGGQYF', 'CASSDEGGQYF'
    ]
    
    obs_data = {
        'va': [va_genes[i % len(va_genes)] for i in range(n_cells)],
        'ja': ['TRAJ61*01'] * n_cells,  # J genes (not used by vectorizer but common in data)
        'cdr3a': [cdr3a_seqs[i % len(cdr3a_seqs)] for i in range(n_cells)],
        'vb': [vb_genes[i % len(vb_genes)] for i in range(n_cells)],
        'jb': ['TRBJ2-7*01'] * n_cells,
        'cdr3b': [cdr3b_seqs[i % len(cdr3b_seqs)] for i in range(n_cells)],
        'clonotype_id': [f'clonotype_{i}' for i in range(n_cells)],
        'clone_id': [f'clone_{i}' for i in range(n_cells)],  # Alternative naming
        'subject': 'subject_1',
        'cell_type': 'CD8T'
    }
    
    # Add some dummy gene expression data
    var_data = pd.DataFrame(index=[f'Gene_{i}' for i in range(100)])
    X = np.random.randn(n_cells, 100)
    
    adata = ad.AnnData(X=X, obs=pd.DataFrame(obs_data), var=var_data)
    return adata


def test_integration_with_preprocess():
    """Test integration with CoNGA preprocessing patterns."""
    print("Testing integration with CoNGA preprocessing...")
    
    adata = create_realistic_test_data(30)
    
    # Test different encoding configurations
    configs = [
        EncodingConfig(),  # Default
        EncodingConfig(aa_mds_dim=12, num_pos_cdr3=14),  # Smaller
        EncodingConfig(aa_mds_dim=20, num_pos_cdr3=20, random_seed=123)  # Larger
    ]
    
    for i, config in enumerate(configs):
        print(f"  Testing configuration {i+1}: {config.aa_mds_dim}D, {config.num_pos_cdr3} CDR3 pos")
        
        # Store vectorized representation
        matrix = store_tcr_vectors_in_adata(adata, 'human', config)
        
        # Verify storage
        assert util.OBSM_KEY_VEC_TCR in adata.obsm
        assert adata.obsm[util.OBSM_KEY_VEC_TCR].shape[0] == adata.n_obs
        
        # Verify config stored correctly
        stored_config = adata.uns[util.UNS_KEY_VEC_TCR_CONFIG]
        assert stored_config['aa_mds_dim'] == config.aa_mds_dim
        assert stored_config['num_pos_cdr3'] == config.num_pos_cdr3
        assert stored_config['random_seed'] == config.random_seed
        
        print(f"    ✓ Matrix shape: {matrix.shape}, dtype: {matrix.dtype}")
    
    print("✓ Integration with different configurations works")


def test_column_name_flexibility():
    """Test flexibility in TCR column naming."""
    print("Testing column name flexibility...")
    
    adata = create_realistic_test_data(10)
    
    # Test with alternative column names
    adata.obs['v_alpha'] = adata.obs['va']
    adata.obs['cdr3_alpha'] = adata.obs['cdr3a']
    adata.obs['v_beta'] = adata.obs['vb']  
    adata.obs['cdr3_beta'] = adata.obs['cdr3b']
    
    # Should work with explicit column specification
    matrix = store_tcr_vectors_in_adata(
        adata, 'human', 
        va_column='v_alpha', cdr3a_column='cdr3_alpha',
        vb_column='v_beta', cdr3b_column='cdr3_beta'
    )
    
    assert matrix.shape[0] == adata.n_obs
    print(f"✓ Alternative column names work: {matrix.shape}")
    

def test_multiple_organisms():
    """Test storage with different supported organisms.""" 
    print("Testing multiple organisms...")
    
    organisms = ['human', 'mouse', 'rhesus']
    
    for organism in organisms:
        print(f"  Testing {organism}...")
        adata = create_realistic_test_data(8)
        
        config = EncodingConfig(aa_mds_dim=8)  # Smaller for faster test
        matrix = store_tcr_vectors_in_adata(adata, organism, config)
        
        # Verify organism stored in config
        stored_config = adata.uns[util.UNS_KEY_VEC_TCR_CONFIG]
        assert stored_config['organism'] == organism
        
        print(f"    ✓ {organism}: {matrix.shape}")
    
    print("✓ Multiple organisms work")


def test_workflow_restart_scenarios():
    """Test typical workflow restart scenarios."""
    print("Testing workflow restart scenarios...")
    
    # Scenario 1: Fresh analysis
    adata = create_realistic_test_data(15)
    
    # Store initial vectorized representation
    config1 = EncodingConfig(aa_mds_dim=12)
    matrix1 = store_tcr_vectors_in_adata(adata, 'human', config1)
    record_active_tcr_representation(adata, util.OBSM_KEY_VEC_TCR)
    
    # Verify initial state
    assert get_active_tcr_representation(adata) == util.OBSM_KEY_VEC_TCR
    print("✓ Initial analysis setup")
    
    # Scenario 2: Save and reload (simulates workflow restart)
    with tempfile.NamedTemporaryFile(suffix='.h5ad', delete=False) as tmp:
        adata.write(tmp.name)
        adata_reloaded = ad.read_h5ad(tmp.name)
        os.unlink(tmp.name)
    
    # Verify restart state
    assert get_active_tcr_representation(adata_reloaded) == util.OBSM_KEY_VEC_TCR
    assert util.OBSM_KEY_VEC_TCR in adata_reloaded.obsm
    assert np.array_equal(adata.obsm[util.OBSM_KEY_VEC_TCR], 
                          adata_reloaded.obsm[util.OBSM_KEY_VEC_TCR])
    print("✓ Workflow restart preservation")
    
    # Scenario 3: Add KernelPCA representation (coexistence)
    fake_kpca = np.random.randn(adata_reloaded.n_obs, 50).astype(np.float32) 
    adata_reloaded.obsm[util.OBSM_KEY_PCA_TCR] = fake_kpca
    
    # Switch active representation
    record_active_tcr_representation(adata_reloaded, util.OBSM_KEY_PCA_TCR)
    assert get_active_tcr_representation(adata_reloaded) == util.OBSM_KEY_PCA_TCR
    
    # Both representations should coexist
    assert util.OBSM_KEY_VEC_TCR in adata_reloaded.obsm
    assert util.OBSM_KEY_PCA_TCR in adata_reloaded.obsm
    print("✓ Multiple representation coexistence")
    
    # Scenario 4: Switch to exact path
    record_active_tcr_representation(adata_reloaded, util.ACTIVE_REP_EXACT)
    assert get_active_tcr_representation(adata_reloaded) == util.ACTIVE_REP_EXACT
    
    # Both obsm representations should still exist
    assert util.OBSM_KEY_VEC_TCR in adata_reloaded.obsm
    assert util.OBSM_KEY_PCA_TCR in adata_reloaded.obsm  
    print("✓ Exact path switching")
    

def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("Testing edge cases...")
    
    # Very small dataset
    adata_small = create_realistic_test_data(2) 
    matrix_small = store_tcr_vectors_in_adata(adata_small, 'human')
    assert matrix_small.shape[0] == 2
    print("✓ Small dataset (2 cells)")
    
    # Empty dataset should work but return empty matrix
    obs_empty = pd.DataFrame({'va': [], 'cdr3a': [], 'vb': [], 'cdr3b': []})
    adata_empty = ad.AnnData(obs=obs_empty)
    matrix_empty = store_tcr_vectors_in_adata(adata_empty, 'human')
    assert matrix_empty.shape[0] == 0
    assert matrix_empty.shape[1] > 0  # Should have correct width
    print("✓ Empty dataset")
    
    # Test with minimal valid CDR3 length
    obs_minimal = pd.DataFrame({
        'va': ['TRAV1-1*01'], 
        'cdr3a': ['CAVRDEG'],  # 7 chars: allows 3 n_trim + 2 c_trim + 2 remaining
        'vb': ['TRBV1*01'],
        'cdr3b': ['CASSRTG']   # 7 chars
    })
    adata_minimal = ad.AnnData(obs=obs_minimal)
    matrix_minimal = store_tcr_vectors_in_adata(adata_minimal, 'human')
    assert matrix_minimal.shape[0] == 1
    print("✓ Minimal CDR3 lengths")


def main():
    """Run all integration tests."""
    print("=" * 70)
    print("Testing F1.1 Integration: AnnData Storage with CoNGA Workflows")
    print("=" * 70)
    
    test_integration_with_preprocess()
    test_column_name_flexibility()
    test_multiple_organisms()
    test_workflow_restart_scenarios()
    test_edge_cases()
    
    print("=" * 70)
    print("✅ All F1.1 integration tests passed!")
    print("Integration with existing CoNGA workflows verified.")
    print("=" * 70)


if __name__ == '__main__':
    main()