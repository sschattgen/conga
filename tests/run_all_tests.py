"""
Master test runner for vectorized TCRdist + FAISS acceleration system.

This script runs the complete test suite including:
- Functionality tests
- Accuracy validation  
- Performance benchmarks
- Error condition tests
- Integration tests
"""

import sys
import pytest
import time
import os
from pathlib import Path

# Add parent directory to path so we can import conga
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_test_suite(test_categories=None, verbose=True):
    """Run the complete test suite.
    
    Parameters
    ----------
    test_categories : list, optional
        Categories to test: ['fixtures', 'validation', 'errors', 'integration']
        If None, runs all categories
    verbose : bool, default=True
        Enable verbose output
    """
    print("🧪 CoNGA Vectorized TCRdist + FAISS Test Suite")
    print("=" * 60)
    
    if test_categories is None:
        test_categories = ['fixtures', 'validation', 'errors']
    
    test_files = []
    
    # Map test categories to files
    category_map = {
        'fixtures': 'test_data_generator.py',
        'validation': 'test_comprehensive_validation.py', 
        'errors': 'test_error_conditions.py',
    }
    
    # Build test file list
    tests_dir = Path(__file__).parent
    for category in test_categories:
        if category in category_map:
            test_file = tests_dir / category_map[category]
            if test_file.exists():
                test_files.append(str(test_file))
            else:
                print(f"⚠️  Test file not found: {test_file}")
    
    if not test_files:
        print("❌ No test files found to run")
        return 1
    
    # Configure pytest arguments
    pytest_args = []
    
    if verbose:
        pytest_args.extend(['-v', '--tb=short'])
    
    # Add coverage if available
    try:
        import pytest_cov
        pytest_args.extend(['--cov=conga', '--cov-report=term-missing'])
    except ImportError:
        pass
    
    # Add test files
    pytest_args.extend(test_files)
    
    # Run tests
    print(f"🚀 Running {len(test_files)} test files...")
    print(f"   Test categories: {', '.join(test_categories)}")
    print(f"   Pytest args: {' '.join(pytest_args)}")
    print("-" * 60)
    
    start_time = time.time()
    result = pytest.main(pytest_args)
    end_time = time.time()
    
    print("-" * 60)
    print(f"⏱️  Tests completed in {end_time - start_time:.2f} seconds")
    
    if result == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ Tests failed (exit code: {result})")
    
    return result


def run_quick_smoke_test():
    """Run a quick smoke test to verify basic functionality."""
    print("🔥 Running quick smoke test...")
    
    try:
        # Test imports
        import conga
        from conga.tcrdist.vectorized import encode_tcrs, EncodingConfig
        from conga import neighbors, util
        print("✅ Imports successful")
        
        # Test basic encoding
        tcrs = [
            (('TRAV1*01', 'TRAJ1*01', 'CAVRD', ''), 
             ('TRBV1*01', 'TRBJ1*01', 'CASSRT', ''))
        ]
        vectors = encode_tcrs(tcrs, 'human')
        print(f"✅ Basic encoding: {vectors.shape}")
        
        # Test FAISS availability
        available_backends = []
        try:
            import faiss
            available_backends.append('faiss_cpu')
            if hasattr(faiss, 'StandardGpuResources'):
                available_backends.append('faiss_gpu')
        except ImportError:
            pass
        available_backends.append('sklearn')
        
        print(f"✅ Available backends: {available_backends}")
        
        # Test backend functionality
        searcher = neighbors.FaissNeighborSearcher()
        import numpy as np
        X = np.random.randn(10, 5).astype(np.float32)
        result = searcher.find_neighbors(X, [0.3], data_type='gex')
        print(f"✅ Neighbor search: {len(result.neighbors[0.3])} neighbors found")
        
        print("🎉 Smoke test passed!")
        return True
        
    except Exception as e:
        print(f"💥 Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point for test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CoNGA Vectorized TCRdist Test Suite")
    parser.add_argument('--smoke', action='store_true',
                       help='Run quick smoke test only')
    parser.add_argument('--categories', nargs='*', 
                       choices=['fixtures', 'validation', 'errors'],
                       help='Test categories to run')
    parser.add_argument('--quiet', action='store_true',
                       help='Quiet mode (less verbose output)')
    
    args = parser.parse_args()
    
    if args.smoke:
        success = run_quick_smoke_test()
        return 0 if success else 1
    else:
        return run_test_suite(
            test_categories=args.categories,
            verbose=not args.quiet
        )


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
