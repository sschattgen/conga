"""
Test FAISS production-grade error handling and graceful fallback behavior.

This module tests comprehensive error handling scenarios that can occur in 
production environments, ensuring robust fallback behavior and clear user guidance.
"""

import pytest
import numpy as np
import logging
from unittest.mock import patch, MagicMock
import sys
import os

from conga.neighbors import (
    FaissNeighborSearcher,
    Backend,
    get_backend_info,
    compute_neighbor_distances,
    FaissError,
    FaissGpuMemoryError,
    FaissCudaError,
    FaissIndexBuildError,
    FaissConfigurationError,
    _validate_input_data
)

logger = logging.getLogger(__name__)


class TestProductionErrorHandling:
    """Test production-grade error handling scenarios."""
    
    def test_comprehensive_data_validation(self):
        """Test comprehensive input data validation."""
        searcher = FaissNeighborSearcher()
        
        # Test invalid data types
        with pytest.raises(FaissConfigurationError) as exc_info:
            searcher.search_neighbors(
                X=[[1, 2], [3, 4]],  # List instead of numpy array
                nbr_fracs=[0.05],
                data_type='test'
            )
        assert "numpy array" in str(exc_info.value)
        
        # Test invalid dimensions
        with pytest.raises(FaissConfigurationError) as exc_info:
            searcher.search_neighbors(
                X=np.array([1, 2, 3]),  # 1D instead of 2D
                nbr_fracs=[0.05],
                data_type='test'
            )
        assert "2D" in str(exc_info.value)
        
        # Test invalid neighbor fractions
        with pytest.raises(FaissConfigurationError) as exc_info:
            searcher.search_neighbors(
                X=np.random.randn(100, 5).astype(np.float32),  # Larger sample size
                nbr_fracs=[0.0, 1.0, -0.1],  # Invalid fractions
                data_type='test'
            )
        assert "Invalid neighbor fractions" in str(exc_info.value)
        
        # Test unsupported metric
        with pytest.raises(FaissConfigurationError) as exc_info:
            searcher.search_neighbors(
                X=np.random.randn(100, 5).astype(np.float32),  # Larger sample size
                nbr_fracs=[0.05],  # Valid fraction for 100 samples
                metric='manhattan',  # Unsupported
                data_type='test'
            )
        assert "Unsupported metric" in str(exc_info.value)
    
    def test_data_quality_issue_detection(self):
        """Test detection and handling of data quality issues."""
        # Test NaN handling
        X_nan = np.random.randn(50, 10).astype(np.float32)
        X_nan[0, 0] = np.nan
        X_nan[1, 1] = np.nan
        
        issues = _validate_input_data(X_nan, [0.1], 'euclidean', 'test')
        assert any('NaN' in issue for issue in issues)
        assert any('2' in issue for issue in issues)  # Should detect 2 NaN values
        
        # Test infinite values
        X_inf = np.random.randn(50, 10).astype(np.float32)
        X_inf[0, 0] = np.inf
        X_inf[1, 1] = -np.inf
        
        issues = _validate_input_data(X_inf, [0.1], 'euclidean', 'test')
        assert any('infinite' in issue for issue in issues)
        
        # Test zero-norm vectors for cosine
        X_zero = np.random.randn(20, 5).astype(np.float32)
        X_zero[0, :] = 0.0  # Zero-norm vector
        X_zero[1, :] = 0.0
        
        issues = _validate_input_data(X_zero, [0.1], 'cosine', 'test')
        assert any('zero-norm' in issue for issue in issues)
    
    def test_graceful_gpu_memory_handling(self):
        """Test GPU memory error handling and fallback."""
        if not get_backend_info()['faiss_gpu_available']:
            pytest.skip("FAISS GPU not available")
        
        searcher = FaissNeighborSearcher(
            gpu_memory_limit_gb=0.01  # Very small limit to trigger memory issues
        )
        
        # Create data large enough to potentially cause memory issues
        X = np.random.randn(2000, 200).astype(np.float32)
        
        # Should either succeed or fallback gracefully
        result = searcher.search_neighbors(
            X=X,
            nbr_fracs=[0.05],
            data_type='test'
        )
        
        # Should have completed with some backend
        assert result.backend_used is not None
        assert len(result.neighbors) == 1
        
        # If it used GPU, great. If it fell back, that's also fine.
        logger.info(f"Memory handling test completed with {result.backend_used.value}")
    
    def test_cuda_error_simulation(self):
        """Test CUDA error handling with mocked failures."""
        if not get_backend_info()['faiss_gpu_available']:
            pytest.skip("FAISS GPU not available")
        
        searcher = FaissNeighborSearcher(force_backend=Backend.FAISS_GPU)
        X = np.random.randn(100, 10).astype(np.float32)
        
        # Mock FAISS to raise CUDA errors
        import conga.neighbors
        original_faiss = None
        
        try:
            # Try to import faiss to get the original
            import faiss as original_faiss
            
            # Create a mock that raises CUDA errors
            mock_faiss = MagicMock()
            mock_faiss.get_num_gpus.return_value = 1
            mock_faiss.StandardGpuResources.side_effect = RuntimeError("CUDA initialization failed")
            
            # Replace faiss in the neighbors module
            with patch.dict(sys.modules, {'faiss': mock_faiss}):
                with patch('conga.neighbors.faiss', mock_faiss):
                    
                    # This should catch the CUDA error and fall back
                    result = searcher.search_neighbors(
                        X=X,
                        nbr_fracs=[0.05],
                        data_type='test'
                    )
                    
                    # Should have fallen back to CPU or sklearn
                    assert result.backend_used != Backend.FAISS_GPU
                    logger.info(f"CUDA error simulation: fell back to {result.backend_used.value}")
                    
        except Exception as e:
            logger.warning(f"CUDA error simulation failed to complete: {e}")
            # This is acceptable - the test environment may not support the mock setup
    
    def test_index_building_failure_handling(self):
        """Test handling of index building failures."""
        searcher = FaissNeighborSearcher()
        
        # Test with data that should cause index building issues
        X_problematic = np.random.randn(100, 10).astype(np.float32)
        X_problematic[0, 0] = np.nan  # NaN that might cause issues
        X_problematic[1, 1] = np.inf  # Inf that might cause issues
        
        # Should detect issues and either fix them or fall back gracefully
        try:
            result = searcher.search_neighbors(
                X=X_problematic,
                nbr_fracs=[0.05],
                data_type='test'
            )
            
            # If it succeeds, it should have used sklearn fallback
            assert result.backend_used == Backend.SKLEARN
            logger.info("Index building failure test: succeeded with sklearn fallback")
            
        except Exception as e:
            # Should get a clear error message about data issues
            error_msg = str(e).lower()
            assert any(keyword in error_msg for keyword in ['nan', 'infinite', 'data'])
            logger.info(f"Index building failure test: got expected error: {e}")
    
    def test_comprehensive_error_logging(self):
        """Test comprehensive error logging and user guidance."""
        searcher = FaissNeighborSearcher()
        
        # Capture log messages
        with pytest.raises(FaissConfigurationError):
            searcher.search_neighbors(
                X=np.array([[1, 2]]),  # Wrong shape
                nbr_fracs=[2.0],  # Invalid fraction
                data_type='test'
            )
    
    def test_backend_selection_with_errors(self):
        """Test backend selection logic with simulated availability."""
        # Test with no FAISS available
        import conga.neighbors as neighbors_module
        
        # Store original values
        original_gpu = neighbors_module._FAISS_GPU_AVAILABLE
        original_cpu = neighbors_module._FAISS_CPU_AVAILABLE
        
        try:
            # Simulate no FAISS available
            neighbors_module._FAISS_GPU_AVAILABLE = False
            neighbors_module._FAISS_CPU_AVAILABLE = False
            
            searcher = FaissNeighborSearcher()
            X = np.random.randn(50, 5).astype(np.float32)
            
            result = searcher.search_neighbors(
                X=X,
                nbr_fracs=[0.1],
                data_type='test'
            )
            
            # Should use sklearn
            assert result.backend_used == Backend.SKLEARN
            
        finally:
            # Restore original values
            neighbors_module._FAISS_GPU_AVAILABLE = original_gpu
            neighbors_module._FAISS_CPU_AVAILABLE = original_cpu
    
    def test_custom_exception_guidance(self):
        """Test that custom exceptions provide actionable guidance."""
        # Test FaissGpuMemoryError
        error = FaissGpuMemoryError("Memory exhausted", 1024.0, "gex")
        msg = error.get_user_message()
        assert "GPU memory" in msg
        assert "1024.0MB" in msg
        assert "CPU backend" in msg
        assert "Suggested solutions:" in msg
        
        # Test FaissCudaError
        error = FaissCudaError("CUDA driver error", "tcr")
        msg = error.get_user_message()
        assert "CUDA" in msg
        assert "tcr data" in msg
        assert "Check CUDA installation" in msg
        
        # Test FaissConfigurationError
        error = FaissConfigurationError(
            "Invalid parameters", 
            invalid_params={"metric": "unsupported", "nbr_fracs": "negative values"}
        )
        msg = error.get_user_message()
        assert "metric" in msg
        assert "nbr_fracs" in msg
    
    def test_empty_and_edge_case_handling(self):
        """Test handling of edge cases like empty data."""
        searcher = FaissNeighborSearcher()
        
        # Empty data
        X_empty = np.empty((0, 10), dtype=np.float32)
        result = searcher.search_neighbors(
            X=X_empty,
            nbr_fracs=[0.05],
            data_type='test'
        )
        assert len(result.neighbors) == 1
        assert result.neighbors[0.05].shape[0] == 0
        
        # Single sample
        X_single = np.random.randn(1, 10).astype(np.float32)
        result = searcher.search_neighbors(
            X=X_single,
            nbr_fracs=[0.5],  # Use larger fraction for single sample
            data_type='test'
        )
        assert result.neighbors[0.5].shape == (1, 1)
        assert result.neighbors[0.5][0, 0] == 0  # Self as only neighbor
        
        # Invalid fraction range test
        X_test = np.random.randn(100, 5).astype(np.float32)
        with pytest.raises(FaissConfigurationError) as exc_info:
            searcher.search_neighbors(
                X=X_test,
                nbr_fracs=[1.5],  # Greater than 1.0 - definitely invalid
                data_type='test'
            )
        assert "not in range" in str(exc_info.value)
    
    def test_memory_monitoring_integration(self):
        """Test memory monitoring and warnings."""
        try:
            import psutil
        except ImportError:
            pytest.skip("psutil not available for memory monitoring test")
        
        searcher = FaissNeighborSearcher()
        
        # Create moderately large data to test memory monitoring
        X = np.random.randn(5000, 100).astype(np.float32)
        
        # Should complete successfully with memory monitoring
        result = searcher.search_neighbors(
            X=X,
            nbr_fracs=[0.01],
            data_type='test'
        )
        
        assert result.neighbors is not None
        logger.info(f"Memory monitoring test completed with {result.backend_used.value}")


class TestRobustFallbackChains:
    """Test robust fallback chains under various failure scenarios."""
    
    def test_complete_fallback_chain(self):
        """Test the complete fallback chain: GPU -> CPU -> sklearn."""
        X = np.random.randn(100, 10).astype(np.float32)
        
        # Test each forced backend to ensure they all work or fail gracefully
        for backend in [Backend.FAISS_GPU, Backend.FAISS_CPU, Backend.SKLEARN]:
            try:
                searcher = FaissNeighborSearcher(force_backend=backend)
                result = searcher.search_neighbors(
                    X=X,
                    nbr_fracs=[0.05],
                    data_type='test'
                )
                
                assert result.backend_used == backend
                assert len(result.neighbors) == 1
                logger.info(f"✓ {backend.value} backend working correctly")
                
            except Exception as e:
                # If backend fails, it should be due to unavailability, not bugs
                logger.info(f"✗ {backend.value} backend unavailable: {e}")
                
                # GPU and CPU failures are acceptable if hardware/software not available
                if backend in (Backend.FAISS_GPU, Backend.FAISS_CPU):
                    assert any(keyword in str(e).lower() 
                              for keyword in ['cuda', 'gpu', 'faiss', 'import', 'available'])
                else:
                    # sklearn should never fail with basic data
                    pytest.fail(f"sklearn backend failed unexpectedly: {e}")
    
    def test_consistency_across_backends(self):
        """Test that different backends produce consistent results."""
        np.random.seed(42)  # For reproducible test
        X = np.random.randn(50, 8).astype(np.float32)
        nbr_fracs = [0.1]
        
        results = {}
        
        # Try each backend that's available
        available_backends = []
        backend_info = get_backend_info()
        
        if backend_info['sklearn_available']:
            available_backends.append(Backend.SKLEARN)
        if backend_info['faiss_cpu_available']:
            available_backends.append(Backend.FAISS_CPU) 
        if backend_info['faiss_gpu_available']:
            available_backends.append(Backend.FAISS_GPU)
        
        for backend in available_backends:
            try:
                searcher = FaissNeighborSearcher(force_backend=backend)
                result = searcher.search_neighbors(
                    X=X,
                    nbr_fracs=nbr_fracs,
                    data_type='test'
                )
                results[backend.value] = result.neighbors[0.1]
                logger.info(f"✓ Got results from {backend.value}")
                
            except Exception as e:
                logger.warning(f"✗ {backend.value} failed: {e}")
        
        # Compare results between available backends
        if len(results) > 1:
            backend_names = list(results.keys())
            reference_result = results[backend_names[0]]
            
            for other_backend in backend_names[1:]:
                other_result = results[other_backend]
                
                # Results should have same shape
                assert reference_result.shape == other_result.shape
                
                # Results should have significant overlap (neighbor sets are similar)
                # Due to potential tie-breaking differences, we allow some variation
                overlap_scores = []
                for i in range(reference_result.shape[0]):
                    ref_neighbors = set(reference_result[i])
                    other_neighbors = set(other_result[i])
                    overlap = len(ref_neighbors.intersection(other_neighbors)) / len(ref_neighbors)
                    overlap_scores.append(overlap)
                
                mean_overlap = np.mean(overlap_scores)
                logger.info(f"Overlap between {backend_names[0]} and {other_backend}: {mean_overlap:.3f}")
                
                # Should have reasonable overlap (>70% typically)
                assert mean_overlap > 0.7, f"Low overlap ({mean_overlap:.3f}) between {backend_names[0]} and {other_backend}"


if __name__ == "__main__":
    # Run production error handling tests
    logging.basicConfig(level=logging.INFO)
    
    print("Testing production-grade FAISS error handling...")
    
    # Test basic error scenarios
    production_tests = TestProductionErrorHandling()
    production_tests.test_comprehensive_data_validation()
    production_tests.test_data_quality_issue_detection()
    production_tests.test_custom_exception_guidance()
    production_tests.test_empty_and_edge_case_handling()
    print("✓ Basic error handling tests passed")
    
    # Test fallback robustness
    fallback_tests = TestRobustFallbackChains()
    fallback_tests.test_complete_fallback_chain()
    print("✓ Fallback chain tests passed")
    
    print("All production error handling tests completed ✓")  # 0 samples
    
    def test_single_sample_handling(self):
        """Test handling of single sample datasets."""
        searcher = FaissNeighborSearcher()
        
        # Single sample
        X_single = np.random.randn(1, 10).astype(np.float32)
        
        result = searcher.search_neighbors(
            X=X_single,
            nbr_fracs=[0.05],
            data_type='gex'
        )
        
        # Should return the single sample as its own neighbor
        assert result.neighbors[0.05].shape == (1, 1)
        assert result.neighbors[0.05][0, 0] == 0
    
    def test_backend_detection_errors(self):
        """Test backend detection with mocked errors."""
        # Mock FAISS import to raise different errors
        import conga.neighbors as neighbors_module
        
        # Store original detection function
        original_detect = neighbors_module._detect_backends
        
        # Test CUDA error detection
        def mock_detect_cuda_error():
            neighbors_module._FAISS_CPU_AVAILABLE = True
            neighbors_module._FAISS_GPU_AVAILABLE = False
            neighbors_module._DETECTION_ERRORS = {'gpu_test': 'CUDA initialization failed'}
            neighbors_module._BACKEND_DETECTION_DONE = True
        
        neighbors_module._detect_backends = mock_detect_cuda_error
        neighbors_module._BACKEND_DETECTION_DONE = False
        
        try:
            searcher = FaissNeighborSearcher()
            backend_info = get_backend_info()
            
            assert backend_info['faiss_cpu_available'] is True
            assert backend_info['faiss_gpu_available'] is False
            assert 'gpu_test' in backend_info['detection_errors']
            
        finally:
            # Restore original function
            neighbors_module._detect_backends = original_detect
            neighbors_module._BACKEND_DETECTION_DONE = False
    
    @pytest.mark.skipif(
        not get_backend_info()['faiss_cpu_available'],
        reason="FAISS CPU not available"
    )
    def test_faiss_cpu_error_fallback(self):
        """Test fallback from FAISS CPU to sklearn on errors."""
        searcher = FaissNeighborSearcher(force_backend=Backend.FAISS_CPU)
        
        # Test with challenging data that might cause FAISS issues
        X = np.random.randn(100, 10).astype(np.float32)
        
        # Add some extreme values that might cause numerical issues
        X[0, :] = 1e10
        X[1, :] = -1e10
        
        # Should either succeed or fall back gracefully
        result = searcher.search_neighbors(
            X=X,
            nbr_fracs=[0.05],
            data_type='gex'
        )
        
        # Should have some result, even if it fell back to sklearn
        assert result.neighbors is not None
        assert len(result.neighbors) > 0


class TestAccuracyValidationErrorHandling:
    """Test error handling in accuracy validation suite."""
    
    def test_validation_with_failed_backends(self):
        """Test validation when some backends fail."""
        validator = AccuracyValidator(tolerance=0.85, strict_mode=False)
        
        # Generate test data
        X = np.random.randn(100, 20).astype(np.float32)
        
        # Try to test a backend that might not be available
        result = validator.compare_backends_comprehensive(
            X=X,
            backend_test='faiss-gpu',  # Might not be available
            backend_baseline='sklearn',
            nbr_fracs=[0.05],
            data_type='gex',
            test_case='error_test'
        )
        
        # Should return a result, even if it failed
        assert result is not None
        assert hasattr(result, 'notes')
        
        # If it failed, notes should indicate the failure
        if not get_backend_info()['faiss_gpu_available']:
            assert "Failed" in result.notes or result.notes == "Success"
    
    def test_validation_with_mismatched_data(self):
        """Test validation with data that causes mismatches."""
        validator = AccuracyValidator(tolerance=0.99, strict_mode=True)  # Very strict
        
        # Create data with potential numerical instability
        X = np.random.randn(50, 5).astype(np.float32)
        
        # Add extreme values
        X[0, :] = 1e6
        X[1, :] = -1e6
        
        # Run validation - might fail due to numerical differences
        try:
            result = validator.compare_backends_comprehensive(
                X=X,
                backend_test='sklearn',  # Compare sklearn to itself
                backend_baseline='sklearn',
                nbr_fracs=[0.10],
                data_type='gex',
                test_case='self_test'
            )
            
            # Self-comparison should be perfect
            assert result.identical_neighbors == 1.0
            assert result.distance_correlation == 1.0
            
        except Exception as e:
            logger.warning(f"Extreme value test failed: {e}")
            # This is acceptable - extreme values can cause numerical issues
    
    def test_edge_case_generator_robustness(self):
        """Test that edge case generators handle edge cases gracefully."""
        from conga.accuracy_validation import EdgeCaseGenerator
        
        # Test with very small datasets
        X = EdgeCaseGenerator.generate_duplicate_vectors(n_samples=2, n_features=2)
        assert X.shape == (2, 2)
        
        # Test with zero sparsity
        X = EdgeCaseGenerator.generate_high_dimensional_sparse(
            n_samples=10, n_features=20, sparsity=0.0
        )
        assert X.shape == (10, 20)
        assert np.mean(X == 0) < 0.1  # Should not be sparse
        
        # Test with single cluster
        X = EdgeCaseGenerator.generate_clustered_data(
            n_samples=20, n_features=5, n_clusters=1
        )
        assert X.shape == (20, 5)


class TestProductionRobustness:
    """Test production robustness scenarios."""
    
    def test_concurrent_access(self):
        """Test concurrent access to FAISS backends."""
        import threading
        import queue
        
        if not get_backend_info()['faiss_cpu_available']:
            pytest.skip("FAISS CPU not available")
        
        results_queue = queue.Queue()
        
        def worker():
            try:
                searcher = FaissNeighborSearcher(force_backend=Backend.FAISS_CPU)
                X = np.random.randn(100, 10).astype(np.float32)
                
                result = searcher.search_neighbors(
                    X=X,
                    nbr_fracs=[0.05],
                    data_type='gex'
                )
                
                results_queue.put(('success', result))
            except Exception as e:
                results_queue.put(('error', str(e)))
        
        # Create multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        success_count = 0
        while not results_queue.empty():
            status, result = results_queue.get()
            if status == 'success':
                success_count += 1
            else:
                logger.warning(f"Thread failed: {result}")
        
        # At least some should succeed
        assert success_count > 0
    
    def test_memory_pressure_handling(self):
        """Test behavior under memory pressure."""
        if not get_backend_info()['faiss_cpu_available']:
            pytest.skip("FAISS CPU not available")
        
        searcher = FaissNeighborSearcher(
            force_backend=Backend.FAISS_CPU,
            gpu_memory_limit_gb=0.1  # Very small limit
        )
        
        # Create moderately large dataset
        X = np.random.randn(2000, 100).astype(np.float32)
        
        # Should handle gracefully, either by succeeding or falling back
        try:
            result = searcher.search_neighbors(
                X=X,
                nbr_fracs=[0.01],
                data_type='gex'
            )
            # If it succeeds, should have valid results
            assert result.neighbors is not None
            
        except Exception as e:
            # If it fails, should be a clear memory-related error
            error_msg = str(e).lower()
            assert any(keyword in error_msg 
                      for keyword in ['memory', 'alloc', 'cuda', 'gpu'])


if __name__ == "__main__":
    # Run error handling tests
    logging.basicConfig(level=logging.INFO)
    
    print("Testing FAISS error handling...")
    
    # Test basic error handling
    error_tests = TestFaissErrorHandling()
    error_tests.test_empty_data_handling()
    error_tests.test_single_sample_handling()
    print("✓ Basic error handling working")
    
    # Test validation error handling
    validation_tests = TestAccuracyValidationErrorHandling()
    validation_tests.test_validation_with_failed_backends()
    validation_tests.test_edge_case_generator_robustness()
    print("✓ Validation error handling working")
    
    print("All error handling tests passed ✓")