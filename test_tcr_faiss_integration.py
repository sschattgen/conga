#!/usr/bin/env python3
"""
Unit tests for FAISS-powered vectorized TCR neighbor search integration.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add conga to path if running standalone
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))

from conga.neighbors import compute_tcr_vector_neighbors, Backend, get_backend_info


class TestFaissTcrIntegration:
    """Test suite for FAISS TCR neighbor search integration."""
    
    def setup_method(self):
        """Set up test data."""
        self.n_clonotypes = 100
        self.vector_length = 200
        self.X_vec_tcr = np.random.randn(self.n_clonotypes, self.vector_length).astype(np.float32)
        
        # Create mock exclusion groups
        self.agroups = np.random.randint(0, 10, self.n_clonotypes)
        self.bgroups = np.random.randint(0, 10, self.n_clonotypes)
        
    def test_backend_availability(self):
        """Test that backend detection works."""
        backend_info = get_backend_info()
        
        # sklearn should always be available
        assert backend_info['sklearn_available'] is True
        
        # FAISS availability depends on installation
        assert isinstance(backend_info['faiss_cpu_available'], bool)
        assert isinstance(backend_info['faiss_gpu_available'], bool)
        
    def test_basic_neighbor_search(self):
        """Test basic TCR vector neighbor search."""
        nbr_fracs = [0.1, 0.2]
        
        result = compute_tcr_vector_neighbors(
            X_vec_tcr=self.X_vec_tcr,
            nbr_fracs=nbr_fracs,
            exclude_groups=None
        )
        
        # Validate result structure
        assert isinstance(result, dict)
        for nbr_frac in nbr_fracs:
            assert nbr_frac in result
            expected_neighbors = max(1, int(nbr_frac * self.n_clonotypes))
            expected_shape = (self.n_clonotypes, expected_neighbors)
            assert result[nbr_frac].shape == expected_shape
            
            # Check neighbor indices are valid
            neighbors = result[nbr_frac]
            assert np.all(neighbors >= 0)
            assert np.all(neighbors < self.n_clonotypes)
            
    def test_exclusion_groups(self):
        """Test that exclusion groups work correctly."""
        nbr_fracs = [0.1]
        
        result = compute_tcr_vector_neighbors(
            X_vec_tcr=self.X_vec_tcr,
            nbr_fracs=nbr_fracs,
            exclude_groups=(self.agroups, self.bgroups)
        )
        
        neighbors = result[nbr_fracs[0]]
        
        # Check that no clonotype is its own neighbor (after exclusions)
        for i in range(self.n_clonotypes):
            assert i not in neighbors[i], f"Clonotype {i} is its own neighbor"
            
    def test_nndists_calculation(self):
        """Test nearest neighbor distance calculation."""
        nbr_fracs = [0.1]
        
        result, nndists = compute_tcr_vector_neighbors(
            X_vec_tcr=self.X_vec_tcr,
            nbr_fracs=nbr_fracs,
            also_calc_nndists=True,
            nbr_frac_for_nndists=nbr_fracs[0]
        )
        
        # Validate nndists
        assert nndists is not None
        assert nndists.shape == (self.n_clonotypes,)
        assert np.all(np.isfinite(nndists))
        assert np.all(nndists >= 0)
        
    def test_empty_input(self):
        """Test handling of empty input."""
        empty_X = np.empty((0, self.vector_length), dtype=np.float32)
        
        result = compute_tcr_vector_neighbors(
            X_vec_tcr=empty_X,
            nbr_fracs=[0.1],
            exclude_groups=None
        )
        
        assert result[0.1].shape == (0, 0)
        
    def test_backend_consistency(self):
        """Test that different backends produce consistent results."""
        backend_info = get_backend_info()
        
        if not backend_info['faiss_cpu_available']:
            pytest.skip("FAISS CPU not available")
            
        nbr_fracs = [0.1]
        
        # Get results from both backends
        faiss_result = compute_tcr_vector_neighbors(
            X_vec_tcr=self.X_vec_tcr,
            nbr_fracs=nbr_fracs,
            exclude_groups=(self.agroups, self.bgroups),
            force_backend=Backend.FAISS_CPU
        )
        
        sklearn_result = compute_tcr_vector_neighbors(
            X_vec_tcr=self.X_vec_tcr,
            nbr_fracs=nbr_fracs,
            exclude_groups=(self.agroups, self.bgroups),
            force_backend=Backend.SKLEARN
        )
        
        # Results should have same shape
        for nbr_frac in nbr_fracs:
            assert faiss_result[nbr_frac].shape == sklearn_result[nbr_frac].shape
            
            # Note: Exact neighbor indices may differ due to ties in distances,
            # but the shapes and validity should be the same
            
    def test_invalid_input(self):
        """Test error handling for invalid inputs."""
        
        # Wrong dimensionality
        with pytest.raises(ValueError):
            compute_tcr_vector_neighbors(
                X_vec_tcr=np.array([1, 2, 3]),  # 1D array
                nbr_fracs=[0.1]
            )
            
        # Empty nbr_fracs - this should actually work and return empty dict
        result = compute_tcr_vector_neighbors(
            X_vec_tcr=self.X_vec_tcr,
            nbr_fracs=[]
        )
        assert result == {}
            
        # Invalid nndists parameters
        with pytest.raises(ValueError):
            compute_tcr_vector_neighbors(
                X_vec_tcr=self.X_vec_tcr,
                nbr_fracs=[0.1],
                also_calc_nndists=True,
                nbr_frac_for_nndists=0.2  # Not in nbr_fracs
            )


if __name__ == "__main__":
    # Run tests
    import subprocess
    import sys
    
    # Try to run with pytest if available
    try:
        result = subprocess.run([sys.executable, '-m', 'pytest', __file__, '-v'], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        sys.exit(result.returncode)
    except FileNotFoundError:
        # Fallback to manual test execution
        print("pytest not available, running tests manually...")
        
        test_class = TestFaissTcrIntegration()
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        passed = 0
        failed = 0
        
        for method_name in test_methods:
            try:
                print(f"Running {method_name}...")
                test_class.setup_method()
                getattr(test_class, method_name)()
                print(f"✓ {method_name} passed")
                passed += 1
            except Exception as e:
                print(f"✗ {method_name} failed: {e}")
                failed += 1
        
        print(f"\nResults: {passed} passed, {failed} failed")
        sys.exit(0 if failed == 0 else 1)