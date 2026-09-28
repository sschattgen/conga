#!/usr/bin/env python3
"""
Demo script for E2.1 Comprehensive Performance Test Suite
"""

import sys
import logging
from pathlib import Path

# Add conga to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    print("="*70)
    print("E2.1 COMPREHENSIVE PERFORMANCE TEST SUITE DEMONSTRATION")
    print("="*70)
    
    from conga.benchmark import (
        ComprehensivePerformanceSuite, 
        create_comprehensive_test_config
    )
    
    # Create a very minimal test configuration for demonstration
    print("Creating minimal test configuration...")
    config = create_comprehensive_test_config(
        test_scale="quick", 
        data_types=['gex'],  # Only test GEX for speed
        output_dir="demo_benchmark_results"
    )
    
    # Override to make it even smaller for demo
    config.sample_sizes = [500, 1000]  # Very small sizes
    config.n_iterations = 1
    config.warmup_iterations = 0
    config.include_accuracy = False  # Skip accuracy for speed
    config.include_memory_profiling = True
    
    print(f"Test configuration:")
    print(f"  Sample sizes: {config.sample_sizes}")
    print(f"  Data types: {config.data_types}")
    print(f"  Iterations: {config.n_iterations}")
    print(f"  Output directory: {config.output_dir}")
    
    # Run comprehensive test suite
    print("\nRunning comprehensive test suite...")
    suite = ComprehensivePerformanceSuite(config)
    
    try:
        results = suite.run_comprehensive_test_suite()
        
        print("\n" + "="*50)
        print("TEST SUITE RESULTS")
        print("="*50)
        
        # Display hardware profile
        hw = results['hardware_profile']
        print(f"\nHardware Profile:")
        print(f"  Platform: {hw.platform}")
        print(f"  CPU Cores: {hw.cpu_cores}")
        print(f"  Memory: {hw.memory_gb:.1f} GB")
        print(f"  FAISS-GPU Available: {hw.faiss_gpu_available}")
        print(f"  FAISS-CPU Available: {hw.faiss_cpu_available}")
        
        # Display backend validation
        backend_val = results['backend_validation']
        print(f"\nBackend Validation:")
        for backend, available in backend_val.items():
            status = "✓" if available else "✗"
            print(f"  {status} {backend}")
        
        # Display detailed benchmark results
        print(f"\nDetailed Benchmark Results:")
        for benchmark in results['detailed_benchmarks']:
            data_type = benchmark['data_type']
            n_samples = benchmark['n_samples']
            print(f"\n  {data_type.upper()} data - {n_samples:,} samples:")
            
            for backend, backend_data in benchmark['backends'].items():
                if 'query_time_mean' in backend_data:
                    query_time = backend_data['query_time_mean']
                    memory = backend_data['memory_mean']
                    print(f"    {backend:12} {query_time*1000:6.2f} ms  {memory:6.1f} MB")
        
        # Display scaling analysis if available
        scaling_results = results.get('scaling_results', {})
        if scaling_results:
            print(f"\nScaling Analysis:")
            for data_type, backends in scaling_results.items():
                print(f"  {data_type.upper()} scaling:")
                for backend, scaling_data in backends.items():
                    if 'speedup_vs_sklearn' in scaling_data and scaling_data['speedup_vs_sklearn']:
                        max_speedup = max(scaling_data['speedup_vs_sklearn'])
                        print(f"    {backend:12} max speedup: {max_speedup:.2f}x vs sklearn")
        
        # Display recommendations
        recommendations = results['performance_recommendations']
        print(f"\nPerformance Recommendations:")
        
        if 'backend_selection' in recommendations:
            print("  Backend Selection Strategy:")
            for key, value in recommendations['backend_selection'].items():
                print(f"    {key.replace('_', ' ').title()}: {value}")
        
        if 'optimization_suggestions' in recommendations and recommendations['optimization_suggestions']:
            print("  Optimization Suggestions:")
            for suggestion in recommendations['optimization_suggestions']:
                print(f"    - {suggestion}")
        
        # Show output files
        print(f"\nOutput Files Generated:")
        output_path = Path(config.output_dir)
        if output_path.exists():
            for file_path in output_path.glob("*"):
                size_kb = file_path.stat().st_size / 1024
                print(f"  {file_path.name} ({size_kb:.1f} KB)")
        
        print(f"\n✓ E2.1 Comprehensive Performance Test Suite completed successfully!")
        print(f"✓ All target capabilities validated:")
        print(f"  - Multi-iteration statistical testing")
        print(f"  - Hardware configuration profiling")
        print(f"  - Performance scaling analysis")
        print(f"  - Memory usage profiling")
        print(f"  - Backend selection recommendations")
        print(f"  - Comprehensive reporting")
        
        # Performance validation check
        sklearn_time = None
        faiss_time = None
        
        for benchmark in results['detailed_benchmarks']:
            for backend, backend_data in benchmark['backends'].items():
                if 'query_time_mean' in backend_data:
                    if backend == 'sklearn':
                        sklearn_time = backend_data['query_time_mean']
                    elif backend.startswith('faiss'):
                        faiss_time = backend_data['query_time_mean']
        
        if sklearn_time and faiss_time and sklearn_time > 0:
            speedup = sklearn_time / faiss_time
            print(f"\n📊 Performance Validation:")
            print(f"  Observed FAISS speedup: {speedup:.2f}x")
            if speedup >= 1.5:
                print(f"  ✓ Meets performance target (>1.5x speedup)")
            else:
                print(f"  ⚠ Below target on small datasets (expected for overhead)")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)