#!/usr/bin/env python3
"""
FAISS accuracy validation script for production deployment.

This script runs comprehensive validation of FAISS backends against sklearn
to ensure accuracy and correctness for production use in CoNGA.

Usage:
    python scripts/validate_faiss_accuracy.py --quick
    python scripts/validate_faiss_accuracy.py --full --output results/
    python scripts/validate_faiss_accuracy.py --edge-cases-only
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from conga.accuracy_validation import (
    quick_accuracy_check,
    production_validation_suite,
    save_validation_report,
    AccuracyValidator
)
from conga.neighbors import get_backend_info

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

def check_prerequisites():
    """Check that required backends are available."""
    backend_info = get_backend_info()
    
    print("Backend Availability:")
    print(f"  FAISS CPU: {'✓' if backend_info['faiss_cpu_available'] else '✗'}")
    print(f"  FAISS GPU: {'✓' if backend_info['faiss_gpu_available'] else '✗'}")
    print(f"  sklearn:   ✓ (always available)")
    
    if backend_info['faiss_gpu_available']:
        print(f"  GPUs:      {backend_info['num_gpus']}")
    
    if backend_info['detection_errors']:
        print("Detection errors:")
        for error_type, error_msg in backend_info['detection_errors'].items():
            print(f"  {error_type}: {error_msg}")
    
    # Check if any FAISS backend is available
    faiss_available = (backend_info['faiss_cpu_available'] or 
                      backend_info['faiss_gpu_available'])
    
    if not faiss_available:
        print("\n⚠️  No FAISS backends available - validation will only test sklearn")
        return False
    
    return True

def run_quick_validation(n_samples: int = 2000, tolerance: float = 0.95):
    """Run quick validation for CI/testing."""
    print(f"\nRunning quick validation (n={n_samples}, tolerance={tolerance:.1%})...")
    
    success = quick_accuracy_check(n_samples=n_samples, tolerance=tolerance)
    
    if success:
        print("✅ Quick validation PASSED - FAISS backends meet accuracy requirements")
        return True
    else:
        print("❌ Quick validation FAILED - FAISS accuracy below threshold")
        return False

def run_full_validation(output_dir: str = None, test_sizes: list = None):
    """Run comprehensive production validation."""
    if test_sizes is None:
        test_sizes = [1000, 5000]
    
    print(f"\nRunning full validation (sizes: {test_sizes})...")
    
    validator = AccuracyValidator(tolerance=0.95, strict_mode=True)
    report = validator.validate_all_backends(
        test_sizes=test_sizes,
        include_edge_cases=True,
        include_determinism=True
    )
    
    # Print summary
    print(f"\nValidation Summary:")
    print(f"  Overall pass:     {'✅' if report.overall_pass else '❌'}")
    print(f"  Production ready: {'✅' if report.production_ready else '❌'}")
    print(f"  Standard tests:   {len(report.validation_results)}")
    print(f"  Edge case tests:  {len(report.edge_case_results)}")
    print(f"  Determinism tests: {len(report.determinism_results)}")
    
    # Show recommendations
    print(f"\nRecommendations ({len(report.recommendations)}):")
    for i, rec in enumerate(report.recommendations[:5], 1):
        print(f"  {i}. {rec}")
    if len(report.recommendations) > 5:
        print(f"  ... and {len(report.recommendations) - 5} more")
    
    # Save report if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_path = output_path / "faiss_validation_report"
        save_validation_report(report, str(report_path))
        
        print(f"\n📄 Detailed report saved to:")
        print(f"  Summary: {report_path}.txt")
        print(f"  Data:    {report_path}.csv")
    
    return report.production_ready

def run_edge_cases_only():
    """Run only edge case validation."""
    print("\nRunning edge case validation...")
    
    validator = AccuracyValidator(tolerance=0.90, strict_mode=False)  # More lenient for edge cases
    edge_results = validator.validate_edge_cases()
    
    print(f"\nEdge Case Results ({len(edge_results)} tests):")
    
    passed = 0
    failed = 0
    
    for result in edge_results:
        status = "✅" if result.notes == "Success" else "❌"
        accuracy = result.identical_neighbors
        
        print(f"  {status} {result.test_case} ({result.backend_a}): {accuracy:.1%} accuracy")
        
        if result.notes == "Success":
            passed += 1
        else:
            failed += 1
    
    print(f"\nEdge case summary: {passed} passed, {failed} failed")
    return failed == 0

def main():
    parser = argparse.ArgumentParser(
        description="Validate FAISS accuracy for CoNGA production use",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --quick                    # Quick validation for CI
  %(prog)s --full --output results/  # Full validation with report
  %(prog)s --edge-cases-only          # Test edge cases only
  %(prog)s --full --sizes 1000 10000  # Custom dataset sizes
        """.strip()
    )
    
    parser.add_argument('--quick', action='store_true',
                       help='Run quick validation (default if no other mode specified)')
    parser.add_argument('--full', action='store_true',
                       help='Run full production validation')
    parser.add_argument('--edge-cases-only', action='store_true',
                       help='Run only edge case tests')
    
    parser.add_argument('--output', '-o', type=str,
                       help='Output directory for reports (full validation only)')
    parser.add_argument('--sizes', type=int, nargs='+', default=[1000, 5000],
                       help='Dataset sizes to test (default: 1000 5000)')
    parser.add_argument('--tolerance', type=float, default=0.95,
                       help='Accuracy tolerance (default: 0.95)')
    parser.add_argument('--samples', type=int, default=2000,
                       help='Sample size for quick validation (default: 2000)')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.verbose)
    
    print("🔬 FAISS Accuracy Validation for CoNGA")
    print("=" * 40)
    
    # Check prerequisites
    faiss_available = check_prerequisites()
    
    # Determine mode
    if args.full:
        mode = 'full'
    elif args.edge_cases_only:
        mode = 'edge'
    else:
        mode = 'quick'  # Default
    
    success = False
    
    try:
        if mode == 'quick':
            success = run_quick_validation(args.samples, args.tolerance)
            
        elif mode == 'full':
            success = run_full_validation(args.output, args.sizes)
            
        elif mode == 'edge':
            success = run_edge_cases_only()
        
        # Final status
        print(f"\n{'='*40}")
        if success:
            print("🎉 Validation PASSED - FAISS is ready for production use")
            exit_code = 0
        else:
            print("⚠️  Validation FAILED - Review results before production use")
            exit_code = 1
            
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user")
        exit_code = 130
        
    except Exception as e:
        print(f"\n❌ Validation error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        exit_code = 1
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()