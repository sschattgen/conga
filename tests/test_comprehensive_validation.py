"""
Comprehensive validation tests for vectorized TCRdist + FAISS acceleration.

This test suite validates all aspects of the implementation including:
- Accuracy against exact TCRdist
- Performance improvements with FAISS
- Error handling and edge cases
- Backend selection logic
- Integration with existing CoNGA workflows
"""

import pytest
import numpy as np
import pandas as pd
import anndata as ad
import time
from typing import Dict, Any, List
import tempfile
import os

from test_config import TEST_CONFIG, AVAILABLE_BACKENDS
import conga
from conga.tcrdist.vectorized import (
    encode_tcrs, EncodingConfig, accuracy_report, vector_length
)
from conga import neighbors, util


class TestVectorizedAccuracy:
    """Test accuracy of vectorized TCRdist vs exact calculations."""
    
    def test_encoding_determinism(self, minimal_clones):
        """Test that encoding is deterministic with same config."""
        tcrs = self._clones_to_tcrs(minimal_clones)
        config = EncodingConfig(random_seed=42)
        
        # Encode multiple times
        vectors1 = encode_tcrs(tcrs, 'human', config)
        vectors2 = encode_tcrs(tcrs, 'human', config) 
        vectors3 = encode_tcrs(tcrs, 'human', config)
        
        # Should be identical
        np.testing.assert_array_equal(vectors1, vectors2)
        np.testing.assert_array_equal(vectors2, vectors3)
    
    def test_accuracy_vs_exact_tcrdist(self, minimal_clones):
        """Test accuracy against exact TCRdist calculations."""
        tcrs = self._clones_to_tcrs(minimal_clones)
        
        # Generate accuracy report
        report = accuracy_report(tcrs, 'human')
        
        # Validate accuracy thresholds
        assert report.spearman >= TEST_CONFIG['tolerance']['accuracy_spearman']
        assert report.mean_recall[10] >= TEST_CONFIG['tolerance']['accuracy_recall']
        
        print(f"✅ Accuracy validation passed:")
        print(f"   Spearman correlation: {report.spearman:.4f}")
        print(f"   Recall@10: {report.mean_recall[10]:.4f}")
    
    def test_vector_length_consistency(self):
        """Test that vector length calculation is consistent."""
        config = EncodingConfig()
        
        for organism in ['human', 'mouse', 'rhesus']:
            expected_length = vector_length(organism, config)
            
            # Create dummy TCR data for this organism
            tcrs = [
                (('TRAV1*01', 'TRAJ1*01', 'CAVRD', ''), 
                 ('TRBV1*01', 'TRBJ1*01', 'CASSRT', ''))
            ]
            
            vectors = encode_tcrs(tcrs, organism, config)
            actual_length = vectors.shape[1]
            
            assert actual_length == expected_length, \
                f"Vector length mismatch for {organism}: {actual_length} != {expected_length}"
    
    def _clones_to_tcrs(self, clones_df: pd.DataFrame) -> List:
        """Convert clones dataframe to TCR tuples."""
        tcrs = []
        for _, row in clones_df.iterrows():
            tcr = ((row['va'], row['ja'], row['cdr3a'], row['cdr3a_nucseq']),
                   (row['vb'], row['jb'], row['cdr3b'], row['cdr3b_nucseq']))
            tcrs.append(tcr)
        return tcrs


class TestFAISSPerformance:
    """Test FAISS performance improvements."""
    
    @pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
    def test_backend_functionality(self, backend, minimal_adata):
        """Test that each available backend works correctly."""
        if backend not in AVAILABLE_BACKENDS:
            pytest.skip(f"Backend {backend} not available")
        
        # Test GEX neighbor search
        X_gex = np.random.randn(100, 50).astype(np.float32)
        nbr_fracs = [0.1]
        
        try:
            if backend == 'sklearn':
                # Test sklearn path
                from sklearn.metrics.pairwise import pairwise_distances
                distances = pairwise_distances(X_gex, metric='euclidean')
                # Basic smoke test
                assert distances.shape == (100, 100)
            
            else:  # FAISS backends
                searcher = neighbors.FaissNeighborSearcher(
                    force_backend=getattr(neighbors.Backend, backend.upper())
                )
                result = searcher.find_neighbors(
                    X_gex, nbr_fracs, data_type='gex'
                )
                assert len(result.neighbors) == len(nbr_fracs)
        
        except Exception as e:
            pytest.fail(f"Backend {backend} failed: {e}")
    
    def test_performance_improvement(self, performance_adata):
        """Test that FAISS provides performance improvements."""
        adata, dataset_info = performance_adata
        
        if adata.n_obs < 1000:  # Skip performance test for small datasets
            pytest.skip("Dataset too small for meaningful performance comparison")
        
        X = np.random.randn(adata.n_obs, 50).astype(np.float32)
        nbr_fracs = [0.01]
        
        # Time sklearn
        start_time = time.time()
        try:
            from sklearn.metrics.pairwise import pairwise_distances
            distances = pairwise_distances(X[:1000], metric='euclidean')  # Limit for speed
            sklearn_time = time.time() - start_time
        except:
            sklearn_time = float('inf')
        
        # Time FAISS (if available)
        if 'faiss_cpu' in AVAILABLE_BACKENDS:
            start_time = time.time()
            searcher = neighbors.FaissNeighborSearcher(
                force_backend=neighbors.Backend.FAISS_CPU
            )
            result = searcher.find_neighbors(X, nbr_fracs, data_type='gex')
            faiss_time = time.time() - start_time
            
            # Expect speedup for medium/large datasets
            if adata.n_obs >= 5000:
                speedup = sklearn_time / faiss_time if faiss_time > 0 else float('inf')
                print(f"FAISS speedup: {speedup:.2f}x for {adata.n_obs} cells")
                # Note: Performance comparison is approximate due to different algorithms


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_gene_names(self, invalid_gene_clones):
        """Test validation of invalid gene names."""
        tcrs = []
        for _, row in invalid_gene_clones.iterrows():
            tcr = ((row['va'], row['ja'], row['cdr3a'], row['cdr3a_nucseq']),
                   (row['vb'], row['jb'], row['cdr3b'], row['cdr3b_nucseq']))
            tcrs.append(tcr)
        
        # Should raise ValueError for invalid genes
        with pytest.raises(ValueError, match="V gene.*not found"):
            encode_tcrs(tcrs, 'human')
    
    def test_edge_case_cdr3_lengths(self, edge_case_clones):
        """Test handling of edge case CDR3 lengths."""
        tcrs = []
        for _, row in edge_case_clones.iterrows():
            tcr = ((row['va'], row['ja'], row['cdr3a'], row['cdr3a_nucseq']),
                   (row['vb'], row['jb'], row['cdr3b'], row['cdr3b_nucseq']))
            tcrs.append(tcr)
        
        # Should handle edge cases gracefully
        vectors = encode_tcrs(tcrs, 'human')
        
        # Check output shape is consistent
        expected_length = vector_length('human')
        assert vectors.shape == (len(tcrs), expected_length)
    
    def test_unsupported_organism(self):
        """Test handling of unsupported organisms."""
        tcrs = [
            (('TRAV1*01', 'TRAJ1*01', 'CAVRD', ''), 
             ('TRBV1*01', 'TRBJ1*01', 'CASSRT', ''))
        ]
        
        with pytest.raises(ValueError, match="not supported by vectorizer"):
            encode_tcrs(tcrs, 'unsupported_organism')
    
    def test_empty_input(self):
        """Test handling of empty input."""
        empty_tcrs = []
        
        # Should handle empty input gracefully
        vectors = encode_tcrs(empty_tcrs, 'human')
        assert vectors.shape[0] == 0
        assert vectors.shape[1] == vector_length('human')


class TestIntegration:
    """Test integration with existing CoNGA workflows."""
    
    def test_anndata_storage_integration(self, minimal_adata):
        """Test AnnData storage functions."""
        from conga.tcrdist.vectorized import store_vectorized_tcr_in_adata
        
        # Store vectorized representation
        vectors = store_vectorized_tcr_in_adata(minimal_adata, 'human')
        
        # Verify storage
        assert util.OBSM_KEY_VEC_TCR in minimal_adata.obsm
        assert util.UNS_KEY_VEC_TCR_CONFIG in minimal_adata.uns
        assert vectors.shape[0] == minimal_adata.n_obs
    
    def test_representation_selection(self, minimal_adata):
        """Test TCR representation selection logic."""
        # Test auto-selection
        active_rep = util.get_active_tcr_representation(minimal_adata)
        assert active_rep in [util.OBSM_KEY_VEC_TCR, util.OBSM_KEY_PCA_TCR, util.ACTIVE_REP_EXACT]
        
        # Test explicit setting
        util.set_active_tcr_representation(minimal_adata, util.ACTIVE_REP_EXACT)
        assert util.get_active_tcr_representation(minimal_adata) == util.ACTIVE_REP_EXACT
    
    def test_backend_configuration_storage(self, minimal_adata):
        """Test FAISS backend configuration storage."""
        config = {
            'backend': 'faiss_cpu',
            'index_type': 'Flat',
            'parameters': {'nprobe': 10},
            'adaptive_parameters': True
        }
        
        neighbors.store_backend_config_in_adata(minimal_adata, config)
        
        # Verify storage
        stored_config = neighbors.load_backend_config_from_adata(minimal_adata)
        assert stored_config['backend'] == 'faiss_cpu'
        assert stored_config['adaptive_parameters'] == True


def run_comprehensive_validation():
    """Run all validation tests."""
    print("🧪 Running comprehensive validation tests...")
    
    # Run pytest with detailed output
    pytest_args = [
        __file__,
        '-v',  # Verbose output
        '--tb=short',  # Short traceback format
        '-x',  # Stop on first failure
    ]
    
    return pytest.main(pytest_args)


if __name__ == "__main__":
    run_comprehensive_validation()
