#!/usr/bin/env python3
"""
CLI script for running comprehensive FAISS performance benchmarks (E2.1).

This script provides an easy interface to run the comprehensive performance 
test suite implemented in conga/benchmark.py with different test scales and 
configurations.
"""

import argparse
import sys
import logging
from pathlib import Path

# Add conga to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive FAISS performance benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (recommended for first run)
  python run_comprehensive_benchmark.py --scale quick

  # Standard comprehensive test
  python run_comprehensive_benchmark.py --scale standard

  # Test only GEX data with custom output directory
  python run_comprehensive_benchmark.py --scale quick --data-types gex --output-dir my_results

  # Stress test (large datasets, long runtime)
  python run_comprehensive_benchmark.py --scale stress --data-types gex tcr

Available test scales:
  quick        - 3 sample sizes, 2 iterations (~2-5 minutes)
  standard     - 5 sample sizes, 3 iterations (~10-20 minutes) 
  comprehensive - 7 sample sizes, 5 iterations (~30-60 minutes)
  stress       - 8 sample sizes, 5 iterations (1-3 hours)
        """
    )
    
    parser.add_argument(
        '--scale', 
        choices=['quick', 'standard', 'comprehensive', 'stress'],
        default='standard',
        help='Test scale/intensity (default: standard)'
    )
    
    parser.add_argument(
        '--data-types',
        nargs='+',
        choices=['gex', 'tcr'],
        default=['gex', 'tcr'],
        help='Data types to benchmark (default: both gex and tcr)'
    )
    
    parser.add_argument(
        '--backends',
        nargs='+', 
        choices=['sklearn', 'faiss-cpu', 'faiss-gpu'],
        help='Specific backends to test (default: all available)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='comprehensive_benchmark_results',
        help='Output directory for results (default: comprehensive_benchmark_results)'
    )
    
    parser.add_argument(
        '--no-accuracy',
        action='store_true',
        help='Skip accuracy validation (faster)'
    )
    
    parser.add_argument(
        '--no-memory-profiling',
        action='store_true',
        help='Skip memory profiling analysis'
    )
    
    parser.add_argument(
        '--iterations',
        type=int,
        help='Override number of iterations per test'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show configuration and exit without running tests'
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Import after logging setup
    try:
        from conga.benchmark import (
            ComprehensivePerformanceSuite, 
            create_comprehensive_test_config
        )
        from conga.neighbors import get_backend_info
    except ImportError as e:
        print(f"Error importing CoNGA modules: {e}")
        print("Make sure you're in the CoNGA directory and the environment is activated.")
        return 1
    
    # Create test configuration
    logger.info(f"Creating {args.scale} test configuration...")
    config = create_comprehensive_test_config(
        test_scale=args.scale,
        data_types=args.data_types,
        output_dir=args.output_dir
    )
    
    # Apply command line overrides
    if args.iterations:
        config.n_iterations = args.iterations
    
    if args.no_accuracy:
        config.include_accuracy = False
        
    if args.no_memory_profiling:
        config.include_memory_profiling = False
        
    if args.backends:
        config.test_backends = args.backends
    
    # Show configuration
    print("="*70)
    print("COMPREHENSIVE FAISS PERFORMANCE BENCHMARK CONFIGURATION")
    print("="*70)
    print(f"Test Scale: {args.scale}")
    print(f"Sample Sizes: {config.sample_sizes}")
    print(f"Data Types: {config.data_types}")
    print(f"Iterations per test: {config.n_iterations}")
    print(f"Warmup iterations: {config.warmup_iterations}")
    print(f"Max memory limit: {config.max_memory_gb} GB")
    print(f"Include accuracy validation: {config.include_accuracy}")
    print(f"Include memory profiling: {config.include_memory_profiling}")
    if config.test_backends:
        print(f"Specific backends: {config.test_backends}")
    print(f"Output directory: {config.output_dir}")
    
    # Estimate runtime
    n_tests = len(config.sample_sizes) * len(config.data_types)
    estimated_time_min = n_tests * config.n_iterations * 0.5  # 30s per iteration minimum
    estimated_time_max = n_tests * config.n_iterations * 2.0  # 2 min per iteration maximum
    print(f"Estimated runtime: {estimated_time_min:.0f}-{estimated_time_max:.0f} minutes")
    
    # Show available backends
    backend_info = get_backend_info()
    print(f"\nAvailable FAISS backends:")
    print(f"  FAISS-GPU: {'✓' if backend_info['faiss_gpu_available'] else '✗'}")
    print(f"  FAISS-CPU: {'✓' if backend_info['faiss_cpu_available'] else '✗'}")
    print(f"  sklearn: ✓ (always available)")
    
    if args.dry_run:
        print(f"\nDry run complete - configuration validated.")
        return 0
    
    # Confirm with user for long-running tests
    if args.scale in ['comprehensive', 'stress']:
        print(f"\nWarning: {args.scale} test may take {estimated_time_max/60:.1f}+ hours to complete.")
        response = input("Continue? [y/N]: ").strip().lower()
        if response != 'y':
            print("Benchmark cancelled.")
            return 0
    
    # Run comprehensive test suite
    print(f"\n" + "="*70)
    print("RUNNING COMPREHENSIVE PERFORMANCE BENCHMARK")
    print("="*70)
    
    try:
        suite = ComprehensivePerformanceSuite(config)
        results = suite.run_comprehensive_test_suite()
        
        # Display summary results
        print(f"\n" + "="*50)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*50)
        
        # Show hardware
        hw = results['hardware_profile']
        print(f"Hardware: {hw.cpu_cores} cores, {hw.memory_gb:.1f}GB RAM, {hw.platform}")
        print(f"FAISS: GPU={hw.faiss_gpu_available}, CPU={hw.faiss_cpu_available}")
        
        # Show performance highlights
        best_speedups = {}
        for benchmark in results['detailed_benchmarks']:
            data_type = benchmark['data_type']
            sklearn_time = None
            best_faiss_time = float('inf')
            
            for backend, data in benchmark['backends'].items():
                if 'query_time_mean' in data:
                    if backend == 'sklearn':
                        sklearn_time = data['query_time_mean']
                    elif backend.startswith('faiss'):
                        best_faiss_time = min(best_faiss_time, data['query_time_mean'])
            
            if sklearn_time and best_faiss_time < float('inf'):
                speedup = sklearn_time / best_faiss_time
                if data_type not in best_speedups or speedup > best_speedups[data_type]:
                    best_speedups[data_type] = speedup
        
        print(f"\nBest FAISS speedups observed:")
        for data_type, speedup in best_speedups.items():
            print(f"  {data_type.upper()}: {speedup:.2f}x faster than sklearn")
        
        # Show recommendations
        recs = results['performance_recommendations']
        if 'backend_selection' in recs:
            print(f"\nBackend recommendations:")
            for key, value in recs['backend_selection'].items():
                print(f"  {key.replace('_', ' ').title()}: {value}")
        
        # Show output files
        output_path = Path(config.output_dir)
        print(f"\nResults saved to: {output_path.absolute()}")
        if output_path.exists():
            for file_path in sorted(output_path.glob("*")):
                size_kb = file_path.stat().st_size / 1024
                print(f"  {file_path.name} ({size_kb:.1f} KB)")
        
        print(f"\n✓ Comprehensive benchmark completed successfully!")
        
        # Final validation
        target_speedup = 2.0 if args.scale in ['comprehensive', 'stress'] else 1.5
        max_speedup = max(best_speedups.values()) if best_speedups else 0
        
        if max_speedup >= target_speedup:
            print(f"✓ Performance target achieved: {max_speedup:.2f}x ≥ {target_speedup}x")
        else:
            print(f"⚠ Performance target not met: {max_speedup:.2f}x < {target_speedup}x")
            print("  (May be expected for very small datasets due to FAISS overhead)")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\nBenchmark interrupted by user.")
        return 130
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())