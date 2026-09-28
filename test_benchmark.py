#!/usr/bin/env python3
"""
Quick test script for the FAISS benchmarking infrastructure.

This script validates that the benchmarking system works correctly
by running a small-scale benchmark and checking the results.
"""

import sys
import logging
from pathlib import Path

# Import conga modules
from conga.benchmark import quick_benchmark, BenchmarkSuite, PerformanceBenchmark
from conga.neighbors import get_backend_info

def setup_logging():
    """Setup logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def test_backend_detection():
    """Test backend availability detection."""
    print("=== Testing Backend Detection ===")
    
    info = get_backend_info()
    print(f"Backend availability:")
    print(f"  FAISS GPU: {info['faiss_gpu_available']} ({info['num_gpus']} GPUs)")
    print(f"  FAISS CPU: {info['faiss_cpu_available']}")
    print(f"  sklearn:   True (always available)")
    
    return info

def test_data_generation():
    """Test synthetic data generation."""
    print("\n=== Testing Data Generation ===")
    
    from conga.benchmark import DatasetGenerator
    
    generator = DatasetGenerator(random_seed=42)
    
    # Test GEX data generation
    gex_data = generator.generate_gex_data(n_samples=1000, n_features=500)
    print(f"GEX data shape: {gex_data.shape}, dtype: {gex_data.dtype}")
    print(f"GEX data range: [{gex_data.min():.3f}, {gex_data.max():.3f}]")
    
    # Test TCR data generation
    tcr_data = generator.generate_tcr_data(n_samples=1000, vector_length=1136)
    print(f"TCR data shape: {tcr_data.shape}, dtype: {tcr_data.dtype}")
    print(f"TCR data range: [{tcr_data.min():.3f}, {tcr_data.max():.3f}]")
    
    # Test exclusion groups
    agroups, bgroups = generator.generate_exclude_groups(1000)
    print(f"Exclusion groups: {len(set(agroups))} alpha groups, {len(set(bgroups))} beta groups")
    
    return True

def test_quick_benchmark():
    """Test quick benchmark functionality."""
    print("\n=== Testing Quick Benchmark ===")
    
    try:
        # Run a small quick benchmark
        results = quick_benchmark(
            data_type='gex',
            n_samples=1000,  # Small size for fast testing
            backends=None  # Test all available
        )
        
        print(f"Benchmark completed with {len(results)} backends:")
        
        for backend_name, result in results.items():
            print(f"  {backend_name}: {result.search_time:.3f}s, "
                  f"memory: {result.memory_peak_mb:.1f}MB, "
                  f"success: {result.success}")
            
            if not result.success:
                print(f"    Error: {result.error_message}")
        
        return len(results) > 0
        
    except Exception as e:
        print(f"Quick benchmark failed: {e}")
        return False

def test_single_benchmark_detailed():
    """Test detailed single benchmark."""
    print("\n=== Testing Detailed Single Benchmark ===")
    
    try:
        benchmark = PerformanceBenchmark(verbose=False)
        
        # Test both GEX and TCR if possible
        for data_type in ['gex', 'tcr']:
            print(f"\nTesting {data_type.upper()} benchmark...")
            
            results = benchmark.run_single_benchmark(
                data_type=data_type,
                n_samples=500,  # Very small for fast testing
                backends=benchmark._get_available_backends(),
                nbr_fracs=[0.05],  # Single neighbor fraction
                exclude_groups=(data_type == 'tcr'),
                validate_accuracy=True
            )
            
            print(f"  {data_type} results:")
            for backend, result in results.items():
                accuracy_str = f"{result.neighbor_accuracy:.3f}" if result.neighbor_accuracy else "N/A"
                print(f"    {backend.value}: {result.search_time:.3f}s, accuracy: {accuracy_str}")
        
        return True
        
    except Exception as e:
        print(f"Detailed benchmark failed: {e}")
        return False

def test_scaling_benchmark():
    """Test scaling benchmark with small sizes."""
    print("\n=== Testing Scaling Benchmark ===")
    
    try:
        benchmark = PerformanceBenchmark(verbose=False)
        
        # Very small scaling test
        scaling_results = benchmark.run_scaling_benchmark(
            data_type='gex',
            sample_sizes=[100, 500],  # Very small sizes
            nbr_fracs=[0.05]
        )
        
        print(f"Scaling benchmark results shape: {scaling_results.shape}")
        print("Sample of results:")
        print(scaling_results[['backend', 'n_samples', 'search_time', 'success']].head())
        
        return len(scaling_results) > 0
        
    except Exception as e:
        print(f"Scaling benchmark failed: {e}")
        return False

def main():
    """Run all tests."""
    setup_logging()
    
    print("FAISS Benchmarking Infrastructure Test")
    print("=" * 50)
    
    all_passed = True
    
    # Test 1: Backend detection
    backend_info = test_backend_detection()
    
    # Test 2: Data generation
    data_gen_ok = test_data_generation()
    all_passed = all_passed and data_gen_ok
    
    # Test 3: Quick benchmark
    quick_ok = test_quick_benchmark()
    all_passed = all_passed and quick_ok
    
    # Test 4: Detailed benchmark
    detailed_ok = test_single_benchmark_detailed()
    all_passed = all_passed and detailed_ok
    
    # Test 5: Scaling benchmark (only if other tests pass)
    if all_passed:
        scaling_ok = test_scaling_benchmark()
        all_passed = all_passed and scaling_ok
    
    # Final summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print(f"Backend detection: {'PASS' if backend_info else 'FAIL'}")
    print(f"Data generation:   {'PASS' if data_gen_ok else 'FAIL'}")
    print(f"Quick benchmark:   {'PASS' if quick_ok else 'FAIL'}")
    print(f"Detailed benchmark:{'PASS' if detailed_ok else 'FAIL'}")
    if all_passed:
        print(f"Scaling benchmark: {'PASS' if scaling_ok else 'FAIL'}")
    
    print(f"\nOverall: {'PASS - Benchmarking infrastructure ready!' if all_passed else 'FAIL - Issues detected'}")
    
    # Provide usage instructions if tests pass
    if all_passed:
        print("\nUsage examples:")
        print("  # Quick test:")
        print("  python scripts/benchmark_faiss.py quick --samples 5000")
        print("  # Full suite:")
        print("  python scripts/benchmark_faiss.py full --output-dir ./results")

    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())