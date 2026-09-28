#!/usr/bin/env python3
"""
FAISS Performance Benchmarking CLI

Script to run FAISS vs sklearn performance benchmarks for CoNGA neighbor search.
Supports both GEX and TCR data types with comprehensive performance analysis.

Usage:
    # Quick benchmark
    python benchmark_faiss.py --quick --data-type gex --samples 10000
    
    # Full benchmark suite
    python benchmark_faiss.py --full --output-dir ./results
    
    # Custom scaling test
    python benchmark_faiss.py --scaling --data-type tcr --samples 1000,5000,10000 --backends faiss-gpu,sklearn
"""

import argparse
import sys
import logging
from pathlib import Path

# Add parent directory to path to import conga modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from conga.benchmark import (
    quick_benchmark, 
    full_benchmark_suite,
    BenchmarkSuite, 
    PerformanceBenchmark,
    Backend
)
from conga.neighbors import get_backend_info

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def print_backend_info():
    """Print available backend information."""
    info = get_backend_info()
    print("Available Backends:")
    print(f"  FAISS GPU: {'✓' if info['faiss_gpu_available'] else '✗'} ({info['num_gpus']} GPUs)")
    print(f"  FAISS CPU: {'✓' if info['faiss_cpu_available'] else '✗'}")
    print(f"  sklearn:   ✓ (always available)")
    print()

def run_quick_benchmark(args):
    """Run quick benchmark."""
    print_backend_info()
    
    backends = args.backends.split(',') if args.backends else None
    
    print(f"Running quick benchmark:")
    print(f"  Data type: {args.data_type}")
    print(f"  Samples: {args.samples:,}")
    print(f"  Backends: {backends or 'all available'}")
    print()
    
    results = quick_benchmark(
        data_type=args.data_type,
        n_samples=args.samples,
        backends=backends
    )
    
    # Print results table
    print("Results:")
    print(f"{'Backend':<12} {'Time (s)':<10} {'Memory (MB)':<12} {'Speedup':<10} {'Accuracy':<10}")
    print("-" * 60)
    
    sklearn_time = results.get('sklearn', {}).search_time if 'sklearn' in results else None
    
    for backend_name, result in results.items():
        speedup = ""
        if sklearn_time and result.search_time > 0:
            speedup = f"{sklearn_time / result.search_time:.1f}x"
        
        accuracy = f"{result.neighbor_accuracy:.3f}" if result.neighbor_accuracy else "N/A"
        
        print(f"{backend_name:<12} {result.search_time:<10.3f} {result.memory_peak_mb:<12.1f} "
              f"{speedup:<10} {accuracy:<10}")

def run_scaling_benchmark(args):
    """Run scaling benchmark."""
    print_backend_info()
    
    sample_sizes = [int(s.strip()) for s in args.samples.split(',')]
    backends = None
    if args.backends:
        backend_map = {
            'faiss-gpu': Backend.FAISS_GPU,
            'faiss-cpu': Backend.FAISS_CPU, 
            'sklearn': Backend.SKLEARN
        }
        backends = [backend_map[name.strip()] for name in args.backends.split(',') 
                   if name.strip() in backend_map]
    
    print(f"Running scaling benchmark:")
    print(f"  Data type: {args.data_type}")
    print(f"  Sample sizes: {sample_sizes}")
    print(f"  Backends: {[b.value for b in backends] if backends else 'all available'}")
    print()
    
    benchmark = PerformanceBenchmark(verbose=True)
    results_df = benchmark.run_scaling_benchmark(
        data_type=args.data_type,
        sample_sizes=sample_sizes,
        backends=backends
    )
    
    # Save results
    if args.output_dir:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        csv_file = output_path / f"{args.data_type}_scaling_benchmark.csv"
        results_df.to_csv(csv_file, index=False)
        print(f"\nResults saved to: {csv_file}")

def run_full_benchmark(args):
    """Run full benchmark suite."""
    print_backend_info()
    
    print("Running comprehensive benchmark suite...")
    print("This may take several minutes depending on available backends.")
    print()
    
    output_dir = args.output_dir or "./benchmark_results"
    results = full_benchmark_suite(output_dir)
    
    # Print summary
    print(f"\nBenchmark suite complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Tests run: {list(results.keys())}")

def main():
    parser = argparse.ArgumentParser(
        description="FAISS Performance Benchmarking for CoNGA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 10k GEX samples
  python benchmark_faiss.py --quick --data-type gex --samples 10000
  
  # Scaling test for TCR data 
  python benchmark_faiss.py --scaling --data-type tcr --samples "1000,5000,10000"
  
  # Full benchmark suite
  python benchmark_faiss.py --full --output-dir ./results
  
  # Test specific backends only
  python benchmark_faiss.py --quick --backends "faiss-gpu,sklearn"
        """
    )
    
    # Subcommand selection
    subparsers = parser.add_subparsers(dest='command', help='Benchmark type')
    
    # Quick benchmark
    quick_parser = subparsers.add_parser('quick', help='Quick single benchmark')
    quick_parser.add_argument('--data-type', choices=['gex', 'tcr'], default='gex',
                             help='Data type to test (default: gex)')
    quick_parser.add_argument('--samples', type=int, default=10000,
                             help='Number of samples (default: 10000)')
    quick_parser.add_argument('--backends', type=str, 
                             help='Comma-separated backend list (faiss-gpu,faiss-cpu,sklearn)')
    
    # Scaling benchmark  
    scaling_parser = subparsers.add_parser('scaling', help='Scaling benchmark')
    scaling_parser.add_argument('--data-type', choices=['gex', 'tcr'], default='gex',
                               help='Data type to test (default: gex)')
    scaling_parser.add_argument('--samples', type=str, default='1000,5000,10000',
                               help='Comma-separated sample sizes (default: 1000,5000,10000)')
    scaling_parser.add_argument('--backends', type=str,
                               help='Comma-separated backend list')
    scaling_parser.add_argument('--output-dir', type=str, 
                               help='Output directory for results')
    
    # Full benchmark suite
    full_parser = subparsers.add_parser('full', help='Comprehensive benchmark suite')
    full_parser.add_argument('--output-dir', type=str, default='./benchmark_results',
                            help='Output directory (default: ./benchmark_results)')
    
    # Global options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    # Legacy support: allow old-style flags for backward compatibility
    if len(sys.argv) > 1 and sys.argv[1].startswith('--'):
        # Convert old-style arguments to new subcommand format
        if '--quick' in sys.argv:
            sys.argv = [sys.argv[0], 'quick'] + [arg for arg in sys.argv[1:] if arg != '--quick']
        elif '--scaling' in sys.argv:
            sys.argv = [sys.argv[0], 'scaling'] + [arg for arg in sys.argv[1:] if arg != '--scaling']
        elif '--full' in sys.argv:
            sys.argv = [sys.argv[0], 'full'] + [arg for arg in sys.argv[1:] if arg != '--full']
        
        # Handle argument name changes
        sys.argv = [arg.replace('--data-type', '--data-type') for arg in sys.argv]
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Route to appropriate function
    if args.command == 'quick':
        run_quick_benchmark(args)
    elif args.command == 'scaling':
        run_scaling_benchmark(args)
    elif args.command == 'full':
        run_full_benchmark(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()