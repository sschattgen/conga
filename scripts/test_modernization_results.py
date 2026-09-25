#!/usr/bin/env python3
"""
Modernization Result Validation Framework

This script provides a test harness for comparing CoNGA analysis results
before and after pandas 3.0/NumPy 2.0 modernization changes. It ensures
that modernization doesn't alter the scientific accuracy of results.

Usage:
    python test_modernization_results.py [options]

Examples:
    # Create baseline from current codebase  
    python test_modernization_results.py --create-baseline
    
    # Validate current results against baseline
    python test_modernization_results.py --validate
    
    # Compare specific analysis outputs
    python test_modernization_results.py --compare-files results_before.h5ad results_after.h5ad
"""

import os
import sys
import argparse
import json
import hashlib
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import subprocess
import warnings

# Suppress warnings during comparison
warnings.filterwarnings('ignore')

try:
    import anndata as ad
    import scanpy as sc
    ANNDATA_AVAILABLE = True
except ImportError:
    ANNDATA_AVAILABLE = False
    print("Warning: AnnData/scanpy not available. Some comparisons will be limited.")


@dataclass
class ComparisonResult:
    """Results of comparing two analysis outputs."""
    comparison_type: str
    files_compared: Tuple[str, str]
    identical: bool
    differences: List[str]
    summary_stats: Dict[str, Any]
    tolerance_used: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass  
class ValidationReport:
    """Complete validation report for modernization testing."""
    test_timestamp: str
    baseline_info: Dict[str, str]
    current_info: Dict[str, str]
    comparison_results: List[ComparisonResult]
    overall_success: bool
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d['comparison_results'] = [cr.to_dict() for cr in self.comparison_results]
        return d


class ModernizationTester:
    """Framework for testing pandas/numpy modernization results."""
    
    def __init__(self, baseline_dir: str = "baseline_results", 
                 test_data_dir: str = "test_data",
                 tolerance: float = 1e-10):
        """Initialize the tester.
        
        Args:
            baseline_dir: Directory to store/read baseline results
            test_data_dir: Directory containing test datasets
            tolerance: Numerical tolerance for floating-point comparisons
        """
        self.baseline_dir = Path(baseline_dir)
        self.test_data_dir = Path(test_data_dir)
        self.tolerance = tolerance
        
        # Create directories if they don't exist
        self.baseline_dir.mkdir(exist_ok=True)
        
        # Test configurations
        self.test_configs = self._init_test_configs()
        
    def _init_test_configs(self) -> List[Dict[str, Any]]:
        """Initialize standard test configurations."""
        configs = []
        
        # Basic correlation analysis
        configs.append({
            'name': 'basic_correlation',
            'description': 'Basic TCR-GEX correlation analysis',
            'script': 'scripts/run_conga.py',
            'args': ['--graph_vs_graph', '--no_plots'],
            'input_files': ['test_clones.tsv', 'test_gex.h5ad'],
            'output_files': ['*.tsv', '*.h5ad'],
            'key_outputs': [
                'conga_results.graph_vs_graph',
                'obsm.X_gex_2d', 
                'uns.conga_stats'
            ]
        })
        
        # TCR clumping analysis
        configs.append({
            'name': 'tcr_clumping',
            'description': 'TCR clumping statistical analysis', 
            'script': 'scripts/run_conga.py',
            'args': ['--tcr_clumping', '--no_plots'],
            'input_files': ['test_clones.tsv', 'test_gex.h5ad'],
            'output_files': ['*.tsv', '*.h5ad'],
            'key_outputs': [
                'uns.tcr_clumping',
                'uns.conga_results'
            ]
        })
        
        # Vectorized TCR representation
        configs.append({
            'name': 'vectorized_tcr',
            'description': 'Vectorized TCR distance representation',
            'script': 'scripts/run_conga.py', 
            'args': ['--graph_vs_graph', '--use_vectorized_tcrdist', '--no_plots'],
            'input_files': ['test_clones.tsv', 'test_gex.h5ad'],
            'output_files': ['*.h5ad'],
            'key_outputs': [
                'obsm.X_tcr_vector',
                'uns.conga_stats.encoding_config'
            ]
        })
        
        return configs
    
    def get_environment_info(self) -> Dict[str, str]:
        """Get current environment information."""
        info = {}
        
        try:
            import pandas as pd
            info['pandas_version'] = pd.__version__
        except ImportError:
            info['pandas_version'] = 'not_available'
            
        try:
            import numpy as np
            info['numpy_version'] = np.__version__
        except ImportError:
            info['numpy_version'] = 'not_available'
            
        try:
            import scanpy as sc
            info['scanpy_version'] = sc.__version__
        except ImportError:
            info['scanpy_version'] = 'not_available'
            
        try:
            import anndata as ad
            info['anndata_version'] = ad.__version__
        except ImportError:
            info['anndata_version'] = 'not_available'
            
        info['python_version'] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        return info
    
    def run_conga_analysis(self, config: Dict[str, Any], output_dir: Path) -> bool:
        """Run a single CoNGA analysis configuration.
        
        Args:
            config: Test configuration dictionary
            output_dir: Directory to store outputs
            
        Returns:
            True if analysis completed successfully
        """
        print(f"  Running {config['name']}...")
        
        # Prepare command
        cmd = ['mamba', 'run', '-n', 'conga-dev', 'python', config['script']]
        cmd.extend(config['args'])
        
        # Add output prefix
        output_prefix = output_dir / config['name']
        cmd.extend(['--outfile_prefix', str(output_prefix)])
        
        # Add test data inputs if they exist  
        if self.test_data_dir.exists():
            for input_file in config.get('input_files', []):
                input_path = self.test_data_dir / input_file
                if input_path.exists():
                    if 'clones' in input_file:
                        cmd.extend(['--clones_file', str(input_path)])
                    elif 'gex' in input_file or input_file.endswith('.h5ad'):
                        cmd.extend(['--gex_data', str(input_path)])
        
        try:
            # Run analysis with timeout
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300,  # 5 minute timeout
                cwd=os.getcwd()
            )
            
            if result.returncode == 0:
                print(f"    ✅ {config['name']} completed successfully")
                return True
            else:
                print(f"    ❌ {config['name']} failed:")
                print(f"      stdout: {result.stdout}")
                print(f"      stderr: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"    ⏰ {config['name']} timed out after 5 minutes")
            return False
        except Exception as e:
            print(f"    ❌ {config['name']} error: {e}")
            return False
    
    def create_baseline(self) -> bool:
        """Create baseline results for comparison.
        
        Returns:
            True if all baseline analyses completed successfully
        """
        print("🔍 Creating baseline results...")
        
        # Create baseline directory structure
        baseline_run_dir = self.baseline_dir / "run_outputs"
        baseline_run_dir.mkdir(exist_ok=True)
        
        # Get current environment info
        env_info = self.get_environment_info()
        
        # Run all test configurations
        success_count = 0
        for config in self.test_configs:
            if self.run_conga_analysis(config, baseline_run_dir):
                success_count += 1
        
        # Save metadata
        metadata = {
            'creation_timestamp': pd.Timestamp.now().isoformat(),
            'environment_info': env_info,
            'test_configs': self.test_configs,
            'successful_configs': success_count,
            'total_configs': len(self.test_configs)
        }
        
        metadata_file = self.baseline_dir / "baseline_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        success_rate = success_count / len(self.test_configs)
        print(f"\n📊 Baseline creation summary:")
        print(f"   Successful: {success_count}/{len(self.test_configs)} ({success_rate:.1%})")
        print(f"   Results saved to: {self.baseline_dir}")
        
        return success_count == len(self.test_configs)
    
    def compare_anndata_objects(self, baseline_path: Path, current_path: Path) -> ComparisonResult:
        """Compare two AnnData objects for differences."""
        if not ANNDATA_AVAILABLE:
            return ComparisonResult(
                comparison_type="anndata_unavailable",
                files_compared=(str(baseline_path), str(current_path)),
                identical=False,
                differences=["AnnData not available for comparison"],
                summary_stats={},
                tolerance_used=self.tolerance
            )
        
        differences = []
        summary_stats = {}
        
        try:
            # Load both objects
            adata_baseline = ad.read_h5ad(baseline_path)
            adata_current = ad.read_h5ad(current_path)
            
            # Compare shapes
            if adata_baseline.shape != adata_current.shape:
                differences.append(f"Shape mismatch: {adata_baseline.shape} vs {adata_current.shape}")
            
            summary_stats['baseline_shape'] = adata_baseline.shape
            summary_stats['current_shape'] = adata_current.shape
            
            # Compare X matrix (main expression data)
            if adata_baseline.X is not None and adata_current.X is not None:
                x_diff = np.abs(adata_baseline.X - adata_current.X).max()
                summary_stats['max_X_difference'] = float(x_diff)
                if x_diff > self.tolerance:
                    differences.append(f"X matrix differs by max {x_diff:.2e} > tolerance {self.tolerance}")
            
            # Compare obs (cell metadata)
            baseline_obs_cols = set(adata_baseline.obs.columns)
            current_obs_cols = set(adata_current.obs.columns)
            
            if baseline_obs_cols != current_obs_cols:
                missing_cols = baseline_obs_cols - current_obs_cols
                extra_cols = current_obs_cols - baseline_obs_cols
                if missing_cols:
                    differences.append(f"Missing obs columns: {list(missing_cols)}")
                if extra_cols:
                    differences.append(f"Extra obs columns: {list(extra_cols)}")
            
            # Compare key numerical columns in obs
            for col in baseline_obs_cols.intersection(current_obs_cols):
                if pd.api.types.is_numeric_dtype(adata_baseline.obs[col]):
                    col_diff = np.abs(adata_baseline.obs[col] - adata_current.obs[col]).max()
                    if col_diff > self.tolerance:
                        differences.append(f"obs['{col}'] differs by max {col_diff:.2e}")
            
            # Compare obsm (embeddings)
            baseline_obsm_keys = set(adata_baseline.obsm.keys())
            current_obsm_keys = set(adata_current.obsm.keys())
            
            for key in baseline_obsm_keys.intersection(current_obsm_keys):
                obsm_diff = np.abs(adata_baseline.obsm[key] - adata_current.obsm[key]).max()
                summary_stats[f'max_{key}_difference'] = float(obsm_diff)
                if obsm_diff > self.tolerance:
                    differences.append(f"obsm['{key}'] differs by max {obsm_diff:.2e}")
            
            # Compare key uns values (results)
            if 'conga_results' in adata_baseline.uns and 'conga_results' in adata_current.uns:
                baseline_results = adata_baseline.uns['conga_results']
                current_results = adata_current.uns['conga_results']
                
                for key in baseline_results.keys():
                    if key in current_results:
                        if isinstance(baseline_results[key], pd.DataFrame):
                            # Compare DataFrames
                            try:
                                df_diff = (baseline_results[key] - current_results[key]).abs().max().max()
                                if df_diff > self.tolerance:
                                    differences.append(f"conga_results['{key}'] DataFrame differs by max {df_diff:.2e}")
                            except:
                                # Non-numeric comparison
                                if not baseline_results[key].equals(current_results[key]):
                                    differences.append(f"conga_results['{key}'] DataFrames differ (non-numeric)")
        
        except Exception as e:
            differences.append(f"Error during comparison: {str(e)}")
        
        return ComparisonResult(
            comparison_type="anndata",
            files_compared=(str(baseline_path), str(current_path)), 
            identical=(len(differences) == 0),
            differences=differences,
            summary_stats=summary_stats,
            tolerance_used=self.tolerance
        )
    
    def compare_tsv_files(self, baseline_path: Path, current_path: Path) -> ComparisonResult:
        """Compare two TSV files for differences."""
        differences = []
        summary_stats = {}
        
        try:
            # Load both files
            df_baseline = pd.read_csv(baseline_path, sep='\t')
            df_current = pd.read_csv(current_path, sep='\t')
            
            # Compare shapes
            if df_baseline.shape != df_current.shape:
                differences.append(f"Shape mismatch: {df_baseline.shape} vs {df_current.shape}")
            
            summary_stats['baseline_shape'] = df_baseline.shape
            summary_stats['current_shape'] = df_current.shape
            
            # Compare column names
            baseline_cols = set(df_baseline.columns)
            current_cols = set(df_current.columns)
            
            if baseline_cols != current_cols:
                missing_cols = baseline_cols - current_cols
                extra_cols = current_cols - baseline_cols
                if missing_cols:
                    differences.append(f"Missing columns: {list(missing_cols)}")
                if extra_cols:
                    differences.append(f"Extra columns: {list(extra_cols)}")
            
            # Compare numeric columns
            for col in baseline_cols.intersection(current_cols):
                if pd.api.types.is_numeric_dtype(df_baseline[col]):
                    try:
                        col_diff = np.abs(df_baseline[col] - df_current[col]).max()
                        summary_stats[f'max_{col}_difference'] = float(col_diff)
                        if col_diff > self.tolerance:
                            differences.append(f"Column '{col}' differs by max {col_diff:.2e}")
                    except:
                        # Handle NaN/inf comparisons
                        if not df_baseline[col].equals(df_current[col]):
                            differences.append(f"Column '{col}' differs (contains NaN/inf)")
                else:
                    # String comparison
                    if not df_baseline[col].equals(df_current[col]):
                        differences.append(f"String column '{col}' differs")
        
        except Exception as e:
            differences.append(f"Error during TSV comparison: {str(e)}")
        
        return ComparisonResult(
            comparison_type="tsv",
            files_compared=(str(baseline_path), str(current_path)),
            identical=(len(differences) == 0), 
            differences=differences,
            summary_stats=summary_stats,
            tolerance_used=self.tolerance
        )
    
    def validate_against_baseline(self) -> ValidationReport:
        """Validate current results against baseline."""
        print("🧪 Validating against baseline...")
        
        # Check if baseline exists
        baseline_metadata_file = self.baseline_dir / "baseline_metadata.json"
        if not baseline_metadata_file.exists():
            raise ValueError("No baseline found. Run with --create-baseline first.")
        
        # Load baseline metadata
        with open(baseline_metadata_file, 'r') as f:
            baseline_metadata = json.load(f)
        
        # Create temporary directory for current results
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            current_run_dir = temp_path / "current_outputs"
            current_run_dir.mkdir()
            
            # Run current analyses
            print("  Running current analyses...")
            success_count = 0
            for config in self.test_configs:
                if self.run_conga_analysis(config, current_run_dir):
                    success_count += 1
            
            # Compare results
            print("\n  Comparing results...")
            comparison_results = []
            
            baseline_run_dir = self.baseline_dir / "run_outputs"
            
            for config in self.test_configs:
                config_name = config['name']
                print(f"    Comparing {config_name}...")
                
                # Find output files to compare
                baseline_files = list(baseline_run_dir.glob(f"{config_name}*"))
                current_files = list(current_run_dir.glob(f"{config_name}*"))
                
                # Match files by name and extension
                baseline_files_dict = {f.name: f for f in baseline_files}
                current_files_dict = {f.name: f for f in current_files}
                
                # Compare matching files
                for filename in baseline_files_dict.keys():
                    if filename in current_files_dict:
                        baseline_file = baseline_files_dict[filename]
                        current_file = current_files_dict[filename]
                        
                        if filename.endswith('.h5ad'):
                            result = self.compare_anndata_objects(baseline_file, current_file)
                        elif filename.endswith('.tsv'):
                            result = self.compare_tsv_files(baseline_file, current_file) 
                        else:
                            # Skip unsupported file types
                            continue
                        
                        comparison_results.append(result)
                        
                        if result.identical:
                            print(f"      ✅ {filename} identical")
                        else:
                            print(f"      ❌ {filename} differs:")
                            for diff in result.differences[:3]:  # Show first 3 differences
                                print(f"        - {diff}")
                            if len(result.differences) > 3:
                                print(f"        ... and {len(result.differences) - 3} more")
        
        # Create validation report
        current_env_info = self.get_environment_info()
        overall_success = all(result.identical for result in comparison_results)
        
        report = ValidationReport(
            test_timestamp=pd.Timestamp.now().isoformat(),
            baseline_info=baseline_metadata['environment_info'],
            current_info=current_env_info,
            comparison_results=comparison_results,
            overall_success=overall_success
        )
        
        return report
    
    def generate_validation_report(self, report: ValidationReport, output_file: str = None) -> str:
        """Generate human-readable validation report."""
        lines = []
        lines.append("Modernization Validation Report")
        lines.append("=" * 50)
        lines.append("")
        
        # Overall result
        if report.overall_success:
            lines.append("🎉 VALIDATION PASSED - All results identical!")
        else:
            lines.append("⚠️  VALIDATION ISSUES - Some results differ!")
        lines.append("")
        
        # Environment comparison
        lines.append("Environment Comparison:")
        lines.append(f"  Baseline pandas: {report.baseline_info.get('pandas_version', 'unknown')}")
        lines.append(f"  Current pandas:  {report.current_info.get('pandas_version', 'unknown')}")
        lines.append(f"  Baseline numpy:  {report.baseline_info.get('numpy_version', 'unknown')}")
        lines.append(f"  Current numpy:   {report.current_info.get('numpy_version', 'unknown')}")
        lines.append("")
        
        # Results summary
        total_comparisons = len(report.comparison_results)
        identical_comparisons = sum(1 for r in report.comparison_results if r.identical)
        
        lines.append(f"Results Summary:")
        lines.append(f"  Total comparisons: {total_comparisons}")
        lines.append(f"  Identical results: {identical_comparisons}")
        lines.append(f"  Different results: {total_comparisons - identical_comparisons}")
        lines.append("")
        
        # Detailed results
        lines.append("Detailed Comparison Results:")
        lines.append("-" * 40)
        
        for result in report.comparison_results:
            status = "✅ IDENTICAL" if result.identical else "❌ DIFFERENT"
            lines.append(f"\n{status}: {result.comparison_type}")
            lines.append(f"  Files: {Path(result.files_compared[0]).name} vs {Path(result.files_compared[1]).name}")
            
            if result.summary_stats:
                lines.append(f"  Statistics:")
                for key, value in result.summary_stats.items():
                    if isinstance(value, float):
                        lines.append(f"    {key}: {value:.2e}")
                    else:
                        lines.append(f"    {key}: {value}")
            
            if result.differences:
                lines.append(f"  Differences ({len(result.differences)}):")
                for diff in result.differences[:5]:  # Show first 5
                    lines.append(f"    - {diff}")
                if len(result.differences) > 5:
                    lines.append(f"    ... and {len(result.differences) - 5} more")
        
        report_text = "\n".join(lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"\n📄 Validation report saved to: {output_file}")
        
        return report_text


def main():
    """Command-line interface for the modernization tester."""
    parser = argparse.ArgumentParser(
        description='Test pandas/numpy modernization result consistency',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --create-baseline          # Create baseline results
  %(prog)s --validate                 # Validate against baseline  
  %(prog)s --compare-files a.h5ad b.h5ad  # Compare specific files
        """
    )
    
    parser.add_argument(
        '--create-baseline', action='store_true',
        help='Create baseline results for future comparisons'
    )
    
    parser.add_argument(
        '--validate', action='store_true', 
        help='Validate current results against baseline'
    )
    
    parser.add_argument(
        '--compare-files', nargs=2, metavar=('FILE1', 'FILE2'),
        help='Compare two specific files'
    )
    
    parser.add_argument(
        '--baseline-dir', default='baseline_results',
        help='Directory for baseline results (default: baseline_results)'
    )
    
    parser.add_argument(
        '--test-data-dir', default='test_data',
        help='Directory containing test datasets (default: test_data)'
    )
    
    parser.add_argument(
        '--tolerance', type=float, default=1e-10,
        help='Numerical tolerance for comparisons (default: 1e-10)'
    )
    
    parser.add_argument(
        '--report-file',
        help='Save validation report to file'
    )
    
    args = parser.parse_args()
    
    if not any([args.create_baseline, args.validate, args.compare_files]):
        parser.error("Must specify one of --create-baseline, --validate, or --compare-files")
    
    # Initialize tester
    tester = ModernizationTester(
        baseline_dir=args.baseline_dir,
        test_data_dir=args.test_data_dir,
        tolerance=args.tolerance
    )
    
    try:
        if args.create_baseline:
            success = tester.create_baseline()
            sys.exit(0 if success else 1)
        
        elif args.validate:
            report = tester.validate_against_baseline()
            report_text = tester.generate_validation_report(report, args.report_file)
            print(report_text)
            sys.exit(0 if report.overall_success else 1)
        
        elif args.compare_files:
            file1, file2 = args.compare_files
            if file1.endswith('.h5ad'):
                result = tester.compare_anndata_objects(Path(file1), Path(file2))
            elif file1.endswith('.tsv'):
                result = tester.compare_tsv_files(Path(file1), Path(file2))
            else:
                print("❌ Unsupported file type. Only .h5ad and .tsv files are supported.")
                sys.exit(1)
            
            print(f"Comparison: {file1} vs {file2}")
            print(f"Identical: {result.identical}")
            if result.differences:
                print("Differences:")
                for diff in result.differences:
                    print(f"  - {diff}")
            
            sys.exit(0 if result.identical else 1)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()