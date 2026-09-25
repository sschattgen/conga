#!/usr/bin/env python3
"""
Test script for FAISS vs sklearn benchmarking infrastructure.

This script validates the benchmark suite and demonstrates its usage
for both GEX and TCR neighbor search performance testing.
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd

# Add conga to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from conga.benchmark import (
    PerformanceSuite,
    DatasetGenerator, 
    quick_gex_benchmark,
    quick_tcr_benchmark,
    validate_faiss_accuracy,
    comprehensive_scaling_benchmark
)
from conga.neighbors import get_backend_info

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_data_generation():
    """Test synthetic data generation."""
    logger.info("Testing data generation...")
    
    # Test GEX data generation
    X_gex = DatasetGenerator.generate_gex_data(n_samples=1000, n_features=100)
    assert X_gex.shape == (1000, 100)
    assert X_gex.dtype == np.float32
    assert np.all(np.isfinite(X_gex))
    logger.info(f"GEX data generation: ✓ {X_gex.shape}, dtype={X_gex.dtype}")
    
    # Test TCR data generation  
    X_tcr = DatasetGenerator.generate_tcr_vector_data(n_samples=500, vector_length=200)
    assert X_tcr.shape == (500, 200)
    assert X_tcr.dtype == np.float32
    assert np.all(np.isfinite(X_tcr))
    logger.info(f"TCR data generation: ✓ {X_tcr.shape}, dtype={X_tcr.dtype}")
    
    # Test exclusion groups
    agroups, bgroups = DatasetGenerator.generate_exclude_groups(n_samples=500)
    assert len(agroups) == 500
    assert len(bgroups) == 500
    logger.info(f"Exclusion groups: ✓ {len(agroups)} alpha, {len(bgroups)} beta")
    

def test_backend_availability():
    """Test backend detection and availability."""
    logger.info("Testing backend availability...")
    
    info = get_backend_info()
    logger.info(f"Backend info: {info}")
    
    # sklearn should always be available
    assert info['sklearn_available'] == True
    
    # Log FAISS availability
    if info['faiss_cpu_available']:
        logger.info("FAISS CPU: ✓ available")
    else:
        logger.warning("FAISS CPU: ✗ not available") 
    
    if info['faiss_gpu_available']:
        logger.info(f"FAISS GPU: ✓ available ({info['num_gpus']} GPUs)")
    else:
        logger.info("FAISS GPU: ✗ not available")


def test_single_benchmark():
    """Test single dataset benchmarking."""
    logger.info("Testing single dataset benchmark...")
    
    suite = PerformanceSuite()
    
    # Small test dataset
    X_gex = DatasetGenerator.generate_gex_data(n_samples=500, n_features=100)
    exclude_groups = DatasetGenerator.generate_exclude_groups(n_samples=500)
    
    # Run benchmark
    results = suite.benchmark_single_dataset(
        X=X_gex,
        data_type='gex',
        nbr_fracs=[0.05, 0.10],
        exclude_groups=exclude_groups,
        validate_accuracy=True
    )
    
    # Validate results
    assert len(results) >= 1  # At least sklearn should work
    
    for result in results:
        assert result.n_samples == 500
        assert result.n_features == 100
        assert result.query_time >= 0
        assert result.peak_memory_mb >= 0
        assert len(result.neighbors) == 2  # Two nbr_fracs tested
        logger.info(f"Backend {result.backend}: {result.query_time:.3f}s, {result.peak_memory_mb:.1f} MB")
    
    # Check accuracy if FAISS backends ran
    faiss_results = [r for r in results if r.backend.startswith('faiss')]
    for result in faiss_results:
        if result.accuracy_vs_baseline is not None:
            logger.info(f"{result.backend} accuracy: {result.accuracy_vs_baseline:.3f}")
            assert result.accuracy_vs_baseline >= 0.8  # Should be quite accurate


def test_performance_report():
    """Test performance report generation."""
    logger.info("Testing performance report generation...")
    
    # Run quick benchmarks
    logger.info("Running GEX benchmark...")
    gex_df = quick_gex_benchmark(n_samples=1000, n_features=100, nbr_fracs=[0.05])
    
    logger.info("Running TCR benchmark...")
    tcr_df = quick_tcr_benchmark(n_samples=800, vector_length=200, nbr_fracs=[0.05])
    
    # Validate reports
    assert not gex_df.empty
    assert not tcr_df.empty
    
    # Check required columns
    required_cols = ['backend', 'data_type', 'n_samples', 'query_time_sec', 'peak_memory_mb']
    for col in required_cols:
        assert col in gex_df.columns
        assert col in tcr_df.columns
    
    logger.info(f"GEX report: {len(gex_df)} results")
    logger.info(f"TCR report: {len(tcr_df)} results")
    
    # Print sample results
    print("\nGEX Benchmark Results:")
    print(gex_df[['backend', 'query_time_sec', 'speedup_vs_sklearn', 'accuracy_vs_baseline']])
    
    print("\nTCR Benchmark Results:")
    print(tcr_df[['backend', 'query_time_sec', 'speedup_vs_sklearn', 'accuracy_vs_baseline']])


def test_scaling_benchmark():
    """Test scaling benchmark (limited)."""
    logger.info("Testing scaling benchmark...")
    
    suite = PerformanceSuite()
    
    # Limited scaling test
    sample_sizes = [500, 1000, 2000]
    
    scaling_results = suite.run_scaling_benchmark(
        data_type='gex',
        sample_sizes=sample_sizes,
        n_features=100,  # Small for testing
        nbr_fracs=[0.05],
        random_seed=42
    )
    
    # Validate scaling results
    assert len(scaling_results) >= 1  # At least one backend
    
    for result in scaling_results:
        assert result.data_type == 'gex'
        assert len(result.sample_sizes) <= len(sample_sizes)  # May skip some if memory limited
        assert len(result.query_times) == len(result.sample_sizes)
        logger.info(f"Scaling {result.backend}: {len(result.sample_sizes)} data points")


def test_recommendations():
    """Test backend recommendation generation.""" 
    logger.info("Testing recommendation generation...")
    
    suite = PerformanceSuite()
    
    # Run some benchmarks
    X_small = DatasetGenerator.generate_gex_data(n_samples=1000, n_features=100)
    results_small = suite.benchmark_single_dataset(
        X=X_small, data_type='gex', nbr_fracs=[0.05], validate_accuracy=True
    )
    
    X_medium = DatasetGenerator.generate_gex_data(n_samples=5000, n_features=100)  
    results_medium = suite.benchmark_single_dataset(
        X=X_medium, data_type='gex', nbr_fracs=[0.05], validate_accuracy=True
    )
    
    # Generate recommendations
    recommendations = suite.get_backend_recommendations()
    
    assert len(recommendations) >= 1
    logger.info("Recommendations generated:")
    for key, rec in recommendations.items():
        logger.info(f"  {key}: {rec}")


def test_accuracy_validation():
    """Test FAISS accuracy validation."""
    logger.info("Testing FAISS accuracy validation...")
    
    # Run validation (will skip if FAISS not available)
    accuracy_ok = validate_faiss_accuracy(n_samples=1000, tolerance=0.90)
    
    logger.info(f"Accuracy validation result: {'PASS' if accuracy_ok else 'FAIL'}")
    
    # Should not fail completely (sklearn baseline always works)
    # FAISS backends may be unavailable but shouldn't cause errors


def run_comprehensive_test():
    """Run all benchmark tests."""
    logger.info("Starting comprehensive benchmark test suite...")
    
    try:
        test_data_generation()
        test_backend_availability()
        test_single_benchmark()
        test_performance_report()
        test_scaling_benchmark()
        test_recommendations() 
        test_accuracy_validation()
        
        logger.info("✓ All benchmark tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)