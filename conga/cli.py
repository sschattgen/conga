"""
Command-line interface for CoNGA.

This module provides a CLI entry point for common CoNGA operations.
"""

import sys
import argparse
from pathlib import Path


def check_faiss_availability():
    """
    Check FAISS availability and return backend type.
    
    Returns:
        tuple: (is_available, backend_type, error_msg)
            - is_available: bool, True if FAISS is importable
            - backend_type: str, 'gpu', 'cpu', or None
            - error_msg: str or None, error message if not available
    """
    try:
        import faiss
        
        # Test GPU availability
        try:
            faiss.StandardGpuResources()
            return True, 'gpu', None
        except (AttributeError, RuntimeError):
            return True, 'cpu', None
            
    except ImportError as e:
        return False, None, str(e)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='conga',
        description='Clonotype Neighbor Graph Analysis (CoNGA) for single-cell TCR/BCR-seq',
        epilog='For detailed usage, see: https://github.com/phbradley/conga'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 0.2.0'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command
    setup_parser = subparsers.add_parser(
        'setup',
        help='Setup 10x data for CoNGA analysis'
    )
    setup_parser.add_argument(
        '--filtered_contig_annotations_csvfile',
        required=True,
        help='Path to 10x filtered_contig_annotations.csv file'
    )
    setup_parser.add_argument(
        '--organism',
        required=True,
        choices=['human', 'mouse', 'rhesus', 'human_gd', 'mouse_gd', 'rhesus_gd', 'human_ig', 'mouse_ig'],
        help='Organism type'
    )
    setup_parser.add_argument(
        '--output_clones_file',
        help='Output clones file (default: auto-generated)'
    )
    
    # Run command
    run_parser = subparsers.add_parser(
        'run',
        help='Run CoNGA analysis pipeline'
    )
    run_parser.add_argument(
        '--clones_file',
        required=True,
        help='Path to clones file (from setup step)'
    )
    run_parser.add_argument(
        '--gex_data',
        required=True,
        help='Path to gene expression data'
    )
    run_parser.add_argument(
        '--gex_data_type',
        default='10x_h5',
        choices=['10x_h5', '10x_mtx', 'h5ad'],
        help='Gene expression data format'
    )
    run_parser.add_argument(
        '--organism',
        required=True,
        help='Organism type'
    )
    run_parser.add_argument(
        '--outfile_prefix',
        required=True,
        help='Prefix for output files'
    )
    run_parser.add_argument(
        '--all',
        action='store_true',
        help='Run all analysis modes'
    )
    
    # Info command
    info_parser = subparsers.add_parser(
        'info',
        help='Show CoNGA installation info'
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    # Execute commands
    if args.command == 'setup':
        return run_setup(args)
    elif args.command == 'run':
        return run_analysis(args)
    elif args.command == 'info':
        return show_info()
    
    return 0


def run_setup(args):
    """Run the setup_10x_for_conga script."""
    import subprocess
    
    script_path = Path(__file__).parent.parent / 'scripts' / 'setup_10x_for_conga.py'
    
    cmd = [
        sys.executable,
        str(script_path),
        '--filtered_contig_annotations_csvfile', args.filtered_contig_annotations_csvfile,
        '--organism', args.organism
    ]
    
    if args.output_clones_file:
        cmd.extend(['--output_clones_file', args.output_clones_file])
    
    try:
        subprocess.run(cmd, check=True)
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Setup failed with error: {e}", file=sys.stderr)
        return 1


def run_analysis(args):
    """Run the run_conga script."""
    import subprocess
    
    script_path = Path(__file__).parent.parent / 'scripts' / 'run_conga.py'
    
    cmd = [
        sys.executable,
        str(script_path),
        '--clones_file', args.clones_file,
        '--gex_data', args.gex_data,
        '--gex_data_type', args.gex_data_type,
        '--organism', args.organism,
        '--outfile_prefix', args.outfile_prefix
    ]
    
    if args.all:
        cmd.append('--all')
    
    try:
        subprocess.run(cmd, check=True)
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Analysis failed with error: {e}", file=sys.stderr)
        return 1


def show_info():
    """Show installation and version information."""
    import conga
    
    print("CoNGA Installation Information")
    print("=" * 50)
    print(f"CoNGA version: 0.2.0")
    print(f"Installation path: {Path(conga.__file__).parent}")
    print()
    
    # Check dependencies
    print("Core Dependencies:")
    try:
        import importlib.metadata
        print(f"  ✓ scanpy: {importlib.metadata.version('scanpy')}")
    except (ImportError, importlib.metadata.PackageNotFoundError):
        try:
            import scanpy
            print(f"  ✓ scanpy: {scanpy.__version__}")
        except ImportError:
            print("  ✗ scanpy: not installed")
    
    try:
        import importlib.metadata
        print(f"  ✓ anndata: {importlib.metadata.version('anndata')}")
    except (ImportError, importlib.metadata.PackageNotFoundError):
        try:
            import anndata
            print(f"  ✓ anndata: {anndata.__version__}")
        except ImportError:
            print("  ✗ anndata: not installed")
    
    try:
        import numpy
        print(f"  ✓ numpy: {numpy.__version__}")
    except ImportError:
        print("  ✗ numpy: not installed")
    
    try:
        import pandas
        print(f"  ✓ pandas: {pandas.__version__}")
    except ImportError:
        print("  ✗ pandas: not installed")
    
    print()
    print("Optional Dependencies:")
    
    # Use the helper function for FAISS checking
    faiss_available, faiss_backend, faiss_error = check_faiss_availability()
    if faiss_available:
        print(f"  ✓ faiss: {faiss_backend.upper()}-enabled")
    else:
        print("  - faiss: not installed (optional, for performance)")
        print("    Install with: pip install 'conga[performance]' (CPU) or 'conga[performance-gpu]' (GPU)")
    
    try:
        import importlib.metadata
        print(f"  ✓ scvi-tools: {importlib.metadata.version('scvi-tools')}")
    except (ImportError, importlib.metadata.PackageNotFoundError):
        print("  - scvi-tools: not installed (optional, experimental)")
    
    print()
    
    # Check C++ components
    tcrdist_cpp = Path(conga.__file__).parent.parent / 'tcrdist_cpp' / 'bin'
    print("C++ Components:")
    
    for binary in ['find_neighbors', 'calc_distributions', 'find_paired_matches']:
        binary_path = tcrdist_cpp / binary
        if binary_path.exists():
            print(f"  ✓ {binary}: compiled")
        else:
            print(f"  ✗ {binary}: not found (compile with: cd tcrdist_cpp && make)")
    
    print()
    print("Performance Backend Information:")
    faiss_available, faiss_backend, faiss_error = check_faiss_availability()
    if faiss_available:
        print(f"  • FAISS: {faiss_backend.upper()}-enabled (high-performance neighbor search)")
        if faiss_backend == 'cpu':
            print("    Note: CPU version detected. For GPU acceleration, install faiss-gpu")
    else:
        print("  • FAISS: Not installed")
        print("    - For CPU: pip install 'conga[performance]'")
        print("    - For GPU: pip install 'conga[performance-gpu]'")
    
    print()
    print("For more information, see:")
    print("  Documentation: https://github.com/phbradley/conga")
    print("  Paper: https://www.nature.com/articles/s41587-021-00989-2")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
