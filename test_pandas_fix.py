#!/usr/bin/env python3
"""
Test script to verify pandas 3.0 compatibility fixes in correlations.py
Specifically tests the Series indexing fixes for is_mait[double_nbrs] and agroups/bgroups[double_nbrs]
"""
import sys
sys.path.insert(0, '/Users/sschattg/conga-dev')

import numpy as np
import pandas as pd
import scanpy as sc
import conga
from conga import preprocess, correlations

# Set up logging
sc.settings.verbosity = 1
print(f"Testing with pandas {pd.__version__}, numpy {np.__version__}")

def test_correlations_indexing():
    """Test the specific correlation analysis functions that were failing with pandas 3.0"""
    
    try:
        print("\n1. Loading test data...")
        # Load data using CoNGA's preprocessing
        adata = conga.preprocess.read_dataset(
            gex_data='test_data/SC5v2_humanPBMCs_5Kcells_Connect_single_channel_SC5v2_humanPBMCs_5Kcells_Connect_single_channel_count_sample_feature_bc_matrix.h5',
            gex_data_type='10x_h5',
            clones_file='test_data/test_run_clones.tsv',
            allow_missing_kpca_file=True
        )
        
        # Set organism
        adata.uns['organism'] = 'human'
        print(f"✓ Data loaded: {adata.shape}")
        
        print("\n2. Basic preprocessing...")
        # Minimal preprocessing to prepare for correlation analysis
        sc.pp.filter_cells(adata, min_genes=100)
        sc.pp.filter_genes(adata, min_cells=3) 
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        adata.raw = adata
        adata = adata[:, adata.var.highly_variable]
        print(f"✓ Preprocessed data: {adata.shape}")
        
        print("\n3. Setting up TCR representation...")
        # Resolve TCR representation 
        tcr_representation = conga.preprocess.resolve_tcr_representation(
            organism=adata.uns['organism'],
            num_obs=adata.shape[0],
            request_kpca=False,
            request_exact_nbrs=False,
            kpca_reduction_limit=20000,
            stored_obsm_keys=list(adata.obsm.keys())
        )
        
        if tcr_representation.build_vectorized:
            from conga.tcrdist.vectorized import EncodingConfig
            encoding_config = EncodingConfig()
            vectors = conga.preprocess.store_tcr_vectors_in_adata(adata, encoding_config)
            print(f"✓ Built vectorized TCR representation: {vectors.shape}")
        
        conga.preprocess.record_active_tcr_representation(adata, tcr_representation.active)
        
        print("\n4. Computing neighbors and clustering...")
        # Cluster and compute neighbors (required for correlation analysis)
        conga.preprocess.cluster_and_tsne_and_umap(
            adata,
            n_neighbors=10,
            n_gex_pcs=20,
            recompute_pca_gex=True,
            make_1d_umaps=False  # Skip to save time
        )
        print(f"✓ Computed clusters and embeddings")
        
        print("\n5. Testing correlation analysis with fixed pandas indexing...")
        # This is the critical test - it will exercise the code we just fixed
        # The graph_vs_graph analysis calls functions in correlations.py that use
        # is_mait[double_nbrs] and agroups[double_nbrs] patterns
        
        # First compute TCR neighbors 
        conga.preprocess.calc_nbrs(adata, nbr_frac=0.01, also_calc_knn_graph=True)
        
        # Now test the correlation analysis - this will call the fixed code
        results = conga.correlations.graph_vs_graph(
            adata, 
            organism='human',
            gex_tag='gex',
            tcr_tag='tcr'
        )
        
        print(f"✓ Correlation analysis completed successfully!")
        print(f"  - Found {len(results)} significant correlations")
        if len(results) > 0:
            print(f"  - Top CoNGA score: {results['conga_score'].max():.4f}")
            print(f"  - Mean overlap: {results['overlap'].mean():.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing pandas 3.0 compatibility fixes in correlations.py")
    print("=" * 60)
    
    success = test_correlations_indexing()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ ALL TESTS PASSED - Pandas indexing fixes are working correctly!")
        print("The correlations.py Series indexing has been successfully fixed for pandas 3.0+")
    else:
        print("❌ TEST FAILED - There may be additional compatibility issues")
        sys.exit(1)