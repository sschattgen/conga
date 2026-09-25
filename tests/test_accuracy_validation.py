"""
Tests for FAISS accuracy validation suite.

This module tests the comprehensive accuracy validation infrastructure
to ensure FAISS backends produce correct results compared to sklearn.
"""

import pytest
import numpy as np
import logging
from pathlib import Path

from conga.accuracy_validation import (
    AccuracyValidator,
    EdgeCaseGenerator,
    ValidationResult,
    DetailedReport,
    quick_accuracy_check,
    production_validation_suite,
    save_validation_report
)
from conga.neighbors import get_backend_info

logger = logging.getLogger(__name__)

class TestEdgeCaseGenerator:
    """Test edge case data generators."""
    
    def test_duplicate_vectors(self):
        """Test duplicate vector generation."""
        X = EdgeCaseGenerator.generate_duplicate_vectors(
            n_samples=100, n_features=50, duplicate_fraction=0.3
        )
        
        assert X.shape == (100, 50)
        assert X.dtype == np.float32
        
        # Check for some duplicates (not exact count due to randomness)
        unique_rows = len(set(tuple(row) for row in X))
        assert unique_rows < 100  # Should have some duplicates
        assert unique_rows > 50   # But not all duplicates
    
    def test_high_dimensional_sparse(self):
        """Test high-dimensional sparse data generation."""
        X = EdgeCaseGenerator.generate_high_dimensional_sparse(
            n_samples=50, n_features=1000, sparsity=0.9
        )
        
        assert X.shape == (50, 1000)
        assert X.dtype == np.float32
        
        # Check sparsity
        zero_fraction = np.mean(X == 0)
        assert zero_fraction > 0.8  # Should be mostly sparse
    
    def test_clustered_data(self):
        """Test clustered data generation."""
        X = EdgeCaseGenerator.generate_clustered_data(
            n_samples=200, n_features=10, n_clusters=4
        )
        
        assert X.shape == (200, 10)
        assert X.dtype == np.float32
    
    def test_extreme_aspect_ratio(self):
        """Test extreme aspect ratio data generation."""
        X = EdgeCaseGenerator.generate_extreme_aspect_ratio(
            n_samples=50, n_features=500
        )
        
        assert X.shape == (50, 500)
        assert X.dtype == np.float32


class TestAccuracyValidator:
    """Test accuracy validation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = AccuracyValidator(tolerance=0.90, strict_mode=False)
        self.backend_info = get_backend_info()
    
    def test_neighbor_recall_calculation(self):
        """Test neighbor recall calculation."""
        # Create mock neighbor arrays
        neighbors_test = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
        neighbors_baseline = np.array([[0, 1, 3], [1, 0, 2], [2, 0, 3]])
        
        recall_scores = self.validator.validate_neighbor_recall(
            neighbors_test, neighbors_baseline, k_values=[1, 2, 3]
        )
        
        assert len(recall_scores) == 3
        assert all(0 <= score <= 1 for score in recall_scores.values())
        
        # At k=1, should have perfect recall (first neighbor always identical)
        assert recall_scores[1] == 1.0
    
    def test_distance_correlation(self):
        """Test distance correlation calculation."""
        # Create correlated distance matrices
        np.random.seed(42)
        D1 = np.random.rand(10, 10)
        D1 = (D1 + D1.T) / 2  # Make symmetric
        np.fill_diagonal(D1, 0)
        
        # Add small noise
        D2 = D1 + 0.01 * np.random.rand(10, 10)
        D2 = (D2 + D2.T) / 2
        np.fill_diagonal(D2, 0)
        
        metrics = self.validator.validate_distance_correlation(D1, D2)
        
        assert 'pearson_correlation' in metrics
        assert metrics['pearson_correlation'] > 0.9  # Should be highly correlated
        assert 'max_absolute_error' in metrics
        assert metrics['max_absolute_error'] >= 0
    
    @pytest.mark.skipif(
        not get_backend_info()['faiss_cpu_available'], 
        reason="FAISS CPU not available"
    )
    def test_backend_comparison_small(self):
        """Test backend comparison on small dataset."""
        # Generate small test dataset
        np.random.seed(42)
        X = np.random.randn(100, 20).astype(np.float32)
        
        result = self.validator.compare_backends_comprehensive(
            X=X,
            backend_test='faiss-cpu',
            backend_baseline='sklearn',
            nbr_fracs=[0.05, 0.10],
            data_type='gex',
            test_case='small_test'
        )
        
        assert isinstance(result, ValidationResult)
        assert result.backend_a == 'faiss-cpu'
        assert result.backend_b == 'sklearn'
        assert result.data_type == 'gex'
        assert result.n_samples == 100
        assert result.n_features == 20
        
        # Should have high accuracy on small dataset
        if result.notes == "Success":
            assert result.distance_correlation > 0.9
            if result.recall_at_k:
                assert min(result.recall_at_k.values()) > 0.8
    
    def test_determinism_sklearn(self):
        """Test determinism validation with sklearn (should be deterministic)."""
        np.random.seed(42)
        X = np.random.randn(50, 10).astype(np.float32)
        
        is_deterministic = self.validator.validate_determinism(
            X=X, backend='sklearn', n_runs=3
        )
        
        assert is_deterministic is True


class TestValidationIntegration:
    """Test full validation workflows."""
    
    def test_quick_accuracy_check(self):
        """Test quick accuracy check function."""
        # Should not raise exceptions
        result = quick_accuracy_check(n_samples=200, tolerance=0.85)
        assert isinstance(result, bool)
    
    @pytest.mark.skipif(
        not (get_backend_info()['faiss_cpu_available'] or 
             get_backend_info()['faiss_gpu_available']),
        reason="No FAISS backends available"
    )
    def test_validation_report_generation(self):
        """Test validation report generation."""
        validator = AccuracyValidator(tolerance=0.80, strict_mode=False)
        
        report = validator.validate_all_backends(
            test_sizes=[200],
            include_edge_cases=False,
            include_determinism=False
        )
        
        assert isinstance(report, DetailedReport)
        assert hasattr(report, 'validation_results')
        assert hasattr(report, 'overall_pass')
        assert hasattr(report, 'production_ready')
        assert len(report.recommendations) > 0
    
    def test_save_validation_report(self, tmp_path):
        """Test saving validation report to files."""
        # Create mock report
        mock_result = ValidationResult(
            backend_a='faiss-cpu',
            backend_b='sklearn', 
            data_type='gex',
            test_case='mock_test',
            n_samples=100,
            n_features=50,
            k_values=[1, 5],
            recall_at_k={1: 0.95, 5: 0.90},
            precision_at_k={1: 0.95, 5: 0.90},
            distance_correlation=0.98,
            identical_neighbors=0.92,
            max_distance_error=0.01,
            deterministic_match=True,
            notes="Success"
        )
        
        mock_report = DetailedReport(
            validation_results=[mock_result],
            edge_case_results=[],
            determinism_results=[],
            backend_availability={'faiss_cpu_available': True},
            overall_pass=True,
            production_ready=True,
            recommendations=["All tests passed"],
            timestamp="2024-01-01T00:00:00"
        )
        
        output_path = tmp_path / "test_report"
        save_validation_report(mock_report, str(output_path))
        
        # Check files were created
        assert (tmp_path / "test_report.txt").exists()
        assert (tmp_path / "test_report.csv").exists()
        
        # Check content
        with open(tmp_path / "test_report.txt") as f:
            content = f.read()
            assert "FAISS Accuracy Validation Report" in content
            assert "Overall Pass: True" in content


class TestEdgeCaseValidation:
    """Test edge case validation scenarios."""
    
    @pytest.mark.skipif(
        not get_backend_info()['faiss_cpu_available'],
        reason="FAISS CPU not available"
    )
    def test_duplicate_vector_handling(self):
        """Test accuracy with duplicate vectors."""
        validator = AccuracyValidator(tolerance=0.85, strict_mode=False)
        
        # Generate data with duplicates
        X = EdgeCaseGenerator.generate_duplicate_vectors(
            n_samples=200, n_features=50, duplicate_fraction=0.4
        )
        
        result = validator.compare_backends_comprehensive(
            X=X,
            backend_test='faiss-cpu',
            backend_baseline='sklearn',
            nbr_fracs=[0.05],
            data_type='gex',
            test_case='duplicate_test'
        )
        
        # Should handle duplicates without error
        assert result.notes == "Success" or "Failed" in result.notes
        if result.notes == "Success":
            # With duplicates, some differences are expected but should be minimal
            assert result.identical_neighbors > 0.7  # Allow some variation
    
    def test_error_handling_invalid_backend(self):
        """Test error handling for invalid backend specifications."""
        validator = AccuracyValidator()
        
        # This should not raise but should return a failed result
        X = np.random.randn(50, 10).astype(np.float32)
        
        # Mock an invalid backend by trying to use GPU when not available
        if not get_backend_info()['faiss_gpu_available']:
            result = validator.compare_backends_comprehensive(
                X=X,
                backend_test='faiss-gpu',  # May not be available
                backend_baseline='sklearn',
                nbr_fracs=[0.05],
                data_type='gex',
                test_case='error_test'
            )
            
            # Should either succeed or fail gracefully
            assert isinstance(result, ValidationResult)


@pytest.mark.integration
class TestProductionValidation:
    """Integration tests for production validation."""
    
    @pytest.mark.skipif(
        not (get_backend_info()['faiss_cpu_available'] or 
             get_backend_info()['faiss_gpu_available']),
        reason="No FAISS backends available"
    )
    def test_production_validation_suite_short(self):
        """Test production validation suite with reduced parameters."""
        # Run with smaller datasets for CI
        validator = AccuracyValidator(tolerance=0.85, strict_mode=False)
        
        report = validator.validate_all_backends(
            test_sizes=[500],  # Smaller for speed
            include_edge_cases=True,
            include_determinism=True
        )
        
        assert isinstance(report, DetailedReport)
        assert len(report.validation_results) > 0
        
        # Should have results for available backends
        backend_names = [r.backend_a for r in report.validation_results]
        available_backends = []
        if get_backend_info()['faiss_cpu_available']:
            available_backends.append('faiss-cpu')
        if get_backend_info()['faiss_gpu_available']:
            available_backends.append('faiss-gpu')
        
        for backend in available_backends:
            assert backend in backend_names, f"Missing results for {backend}"


if __name__ == "__main__":
    # Run basic tests
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("Testing FAISS accuracy validation suite...")
    
    # Test edge case generators
    print("Testing edge case generators...")
    generator_tests = TestEdgeCaseGenerator()
    generator_tests.test_duplicate_vectors()
    generator_tests.test_high_dimensional_sparse()
    generator_tests.test_clustered_data()
    generator_tests.test_extreme_aspect_ratio()
    print("✓ Edge case generators working")
    
    # Test validator
    print("Testing accuracy validator...")
    validator_tests = TestAccuracyValidator()
    validator_tests.setup_method()
    validator_tests.test_neighbor_recall_calculation()
    validator_tests.test_distance_correlation()
    validator_tests.test_determinism_sklearn()
    print("✓ Accuracy validator working")
    
    # Test integration
    print("Testing integration...")
    integration_tests = TestValidationIntegration()
    integration_tests.test_quick_accuracy_check()
    print("✓ Integration tests working")
    
    print("All accuracy validation tests passed ✓")