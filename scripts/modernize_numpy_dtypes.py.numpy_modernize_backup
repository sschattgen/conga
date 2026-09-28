#!/usr/bin/env python3
"""
NumPy 2.0 Legacy Dtype Alias Modernization Script

This script systematically finds and replaces NumPy legacy dtype aliases
that were removed in NumPy 2.0. It provides dry-run mode, backup capabilities,
and comprehensive reporting.

Usage:
    python modernize_numpy_dtypes.py [options]

Examples:
    # Dry run to see what would be changed
    python modernize_numpy_dtypes.py --dry-run
    
    # Apply changes with backup
    python modernize_numpy_dtypes.py --backup
    
    # Apply changes to specific directory
    python modernize_numpy_dtypes.py --target-dir conga/tcrdist
"""

import os
import re
import sys
import argparse
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import tempfile
import json


@dataclass
class Replacement:
    """Represents a single replacement operation."""
    file_path: str
    line_number: int
    old_text: str
    new_text: str
    pattern_name: str
    context: str  # Line content for verification


@dataclass
class FileReport:
    """Report for changes in a single file."""
    file_path: str
    replacements: List[Replacement]
    backup_path: Optional[str] = None
    
    @property
    def has_changes(self) -> bool:
        return len(self.replacements) > 0


class NumpyDtypeModernizer:
    """Main modernization engine for NumPy dtype aliases."""
    
    # NumPy 2.0 legacy alias replacement patterns
    DTYPE_PATTERNS = {
        'np.float_': {
            'pattern': r'\bnp\.float_\b',
            'replacement': 'np.float64',
            'description': 'Legacy np.float_ → np.float64'
        },
        'np.int_': {
            'pattern': r'\bnp\.int_\b', 
            'replacement': 'np.int64',
            'description': 'Legacy np.int_ → np.int64'
        },
        'np.bool_': {
            'pattern': r'\bnp\.bool_\b',
            'replacement': 'bool',
            'description': 'Legacy np.bool_ → bool (Python built-in)'
        },
        'np.object_': {
            'pattern': r'\bnp\.object_\b',
            'replacement': 'object', 
            'description': 'Legacy np.object_ → object (Python built-in)'
        },
        'np.complex_': {
            'pattern': r'\bnp\.complex_\b',
            'replacement': 'np.complex128',
            'description': 'Legacy np.complex_ → np.complex128'
        },
        'np.str_': {
            'pattern': r'\bnp\.str_\b',
            'replacement': 'str',
            'description': 'Legacy np.str_ → str (Python built-in)'
        },
        'np.unicode_': {
            'pattern': r'\bnp\.unicode_\b',
            'replacement': 'str',
            'description': 'Legacy np.unicode_ → str (Python built-in)'
        }
    }
    
    # Additional patterns for common variations
    ADDITIONAL_PATTERNS = {
        'numpy.float_': {
            'pattern': r'\bnumpy\.float_\b',
            'replacement': 'numpy.float64',
            'description': 'Legacy numpy.float_ → numpy.float64'
        },
        'numpy.int_': {
            'pattern': r'\bnumpy\.int_\b',
            'replacement': 'numpy.int64', 
            'description': 'Legacy numpy.int_ → numpy.int64'
        },
        'numpy.bool_': {
            'pattern': r'\bnumpy\.bool_\b',
            'replacement': 'bool',
            'description': 'Legacy numpy.bool_ → bool'
        },
        'numpy.object_': {
            'pattern': r'\bnumpy\.object_\b',
            'replacement': 'object',
            'description': 'Legacy numpy.object_ → object'
        },
    }
    
    def __init__(self, target_dirs: List[str] = None, exclude_patterns: List[str] = None):
        """Initialize the modernizer.
        
        Args:
            target_dirs: List of directories to scan (default: ['conga', 'scripts'])
            exclude_patterns: List of file patterns to exclude (default: common excludes)
        """
        self.target_dirs = target_dirs or ['conga', 'scripts']
        self.exclude_patterns = exclude_patterns or [
            '*.pyc', '__pycache__', '.git', '.pytest_cache', 
            '*.egg-info', '.DS_Store', '*.bak', '*.backup'
        ]
        
        # Compile all regex patterns for efficiency
        self.compiled_patterns = {}
        all_patterns = {**self.DTYPE_PATTERNS, **self.ADDITIONAL_PATTERNS}
        for name, pattern_info in all_patterns.items():
            self.compiled_patterns[name] = {
                'regex': re.compile(pattern_info['pattern']),
                'replacement': pattern_info['replacement'],
                'description': pattern_info['description']
            }
    
    def find_python_files(self) -> List[Path]:
        """Find all Python files to process."""
        python_files = []
        
        for target_dir in self.target_dirs:
            target_path = Path(target_dir)
            if not target_path.exists():
                print(f"Warning: Target directory {target_dir} does not exist")
                continue
                
            # Find all .py files recursively
            for py_file in target_path.rglob('*.py'):
                # Skip excluded patterns
                if self._should_exclude_file(py_file):
                    continue
                python_files.append(py_file)
        
        return sorted(python_files)
    
    def _should_exclude_file(self, file_path: Path) -> bool:
        """Check if file should be excluded based on patterns."""
        file_str = str(file_path)
        for pattern in self.exclude_patterns:
            if pattern in file_str:
                return True
        return False
    
    def analyze_file(self, file_path: Path) -> List[Replacement]:
        """Analyze a single file for NumPy dtype patterns."""
        replacements = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except (UnicodeDecodeError, IOError) as e:
            print(f"Warning: Could not read {file_path}: {e}")
            return replacements
        
        for line_num, line in enumerate(lines, 1):
            for pattern_name, pattern_info in self.compiled_patterns.items():
                matches = pattern_info['regex'].findall(line)
                if matches:
                    # Create replacement for each match
                    old_line = line.rstrip()
                    new_line = pattern_info['regex'].sub(
                        pattern_info['replacement'], line
                    ).rstrip()
                    
                    replacement = Replacement(
                        file_path=str(file_path),
                        line_number=line_num,
                        old_text=old_line,
                        new_text=new_line,
                        pattern_name=pattern_name,
                        context=old_line.strip()
                    )
                    replacements.append(replacement)
        
        return replacements
    
    def apply_replacements(self, file_path: Path, replacements: List[Replacement], 
                          backup: bool = True) -> FileReport:
        """Apply replacements to a file."""
        if not replacements:
            return FileReport(str(file_path), [])
        
        # Create backup if requested
        backup_path = None
        if backup:
            backup_path = f"{file_path}.numpy_modernize_backup"
            shutil.copy2(file_path, backup_path)
        
        try:
            # Read original file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Apply all pattern replacements
            for pattern_name, pattern_info in self.compiled_patterns.items():
                content = pattern_info['regex'].sub(
                    pattern_info['replacement'], content
                )
            
            # Write modified content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return FileReport(str(file_path), replacements, backup_path)
            
        except Exception as e:
            # Restore from backup if something went wrong
            if backup and backup_path and os.path.exists(backup_path):
                shutil.copy2(backup_path, file_path)
            raise RuntimeError(f"Failed to apply replacements to {file_path}: {e}")
    
    def scan_all_files(self) -> Dict[str, List[Replacement]]:
        """Scan all target files for NumPy dtype patterns."""
        all_replacements = {}
        python_files = self.find_python_files()
        
        print(f"Scanning {len(python_files)} Python files...")
        
        for file_path in python_files:
            replacements = self.analyze_file(file_path)
            if replacements:
                all_replacements[str(file_path)] = replacements
        
        return all_replacements
    
    def generate_report(self, file_reports: List[FileReport]) -> str:
        """Generate a comprehensive report of all changes."""
        report_lines = []
        report_lines.append("NumPy Dtype Modernization Report")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        total_files = len([r for r in file_reports if r.has_changes])
        total_replacements = sum(len(r.replacements) for r in file_reports)
        
        report_lines.append(f"Summary:")
        report_lines.append(f"  Files modified: {total_files}")
        report_lines.append(f"  Total replacements: {total_replacements}")
        report_lines.append("")
        
        # Count replacements by pattern
        pattern_counts = {}
        for file_report in file_reports:
            for repl in file_report.replacements:
                pattern_counts[repl.pattern_name] = pattern_counts.get(repl.pattern_name, 0) + 1
        
        report_lines.append("Replacements by pattern:")
        for pattern, count in sorted(pattern_counts.items()):
            desc = self.compiled_patterns[pattern]['description']
            report_lines.append(f"  {pattern}: {count} ({desc})")
        report_lines.append("")
        
        # Detailed file-by-file report
        report_lines.append("Detailed Changes:")
        report_lines.append("-" * 30)
        
        for file_report in file_reports:
            if not file_report.has_changes:
                continue
                
            report_lines.append(f"\nFile: {file_report.file_path}")
            if file_report.backup_path:
                report_lines.append(f"Backup: {file_report.backup_path}")
                
            for repl in file_report.replacements:
                report_lines.append(f"  Line {repl.line_number}: {repl.pattern_name}")
                report_lines.append(f"    Old: {repl.old_text}")
                report_lines.append(f"    New: {repl.new_text}")
        
        return "\n".join(report_lines)
    
    def modernize(self, dry_run: bool = False, backup: bool = True, 
                  report_file: str = None) -> None:
        """Execute the modernization process."""
        print("NumPy Dtype Legacy Alias Modernization")
        print("=" * 50)
        print()
        
        if dry_run:
            print("🔍 DRY RUN MODE - No files will be modified")
            print()
        
        # Scan for patterns
        all_replacements = self.scan_all_files()
        
        if not all_replacements:
            print("✅ No NumPy legacy dtype aliases found!")
            print("   All files are already NumPy 2.0+ compatible.")
            return
        
        total_files = len(all_replacements)
        total_replacements = sum(len(repls) for repls in all_replacements.values())
        
        print(f"Found {total_replacements} replacements in {total_files} files")
        print()
        
        if dry_run:
            # Show what would be changed
            for file_path, replacements in all_replacements.items():
                print(f"📁 {file_path}")
                for repl in replacements:
                    print(f"   Line {repl.line_number}: {repl.pattern_name}")
                    print(f"     {repl.context}")
                print()
        else:
            # Apply changes
            file_reports = []
            print("Applying changes...")
            
            for file_path, replacements in all_replacements.items():
                print(f"  Processing {file_path}...")
                file_report = self.apply_replacements(
                    Path(file_path), replacements, backup
                )
                file_reports.append(file_report)
            
            # Generate and save report
            report = self.generate_report(file_reports)
            print("\n" + report)
            
            if report_file:
                with open(report_file, 'w') as f:
                    f.write(report)
                print(f"\nReport saved to: {report_file}")
            
            print(f"\n✅ Successfully modernized {total_files} files!")
            if backup:
                print("   Backups created with .numpy_modernize_backup extension")


def main():
    """Command-line interface for the NumPy dtype modernizer."""
    parser = argparse.ArgumentParser(
        description='Modernize NumPy legacy dtype aliases for NumPy 2.0+ compatibility',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dry-run                    # Preview changes without modifying files
  %(prog)s --backup                     # Apply changes with backup files
  %(prog)s --target-dir conga           # Only process conga/ directory
  %(prog)s --report-file changes.txt    # Save report to file
        """
    )
    
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Preview changes without modifying files'
    )
    
    parser.add_argument(
        '--backup', action='store_true', default=True,
        help='Create backup files before modification (default: True)'
    )
    
    parser.add_argument(
        '--no-backup', dest='backup', action='store_false',
        help='Skip creating backup files'
    )
    
    parser.add_argument(
        '--target-dir', action='append', dest='target_dirs',
        help='Directory to process (can be specified multiple times)'
    )
    
    parser.add_argument(
        '--exclude', action='append', dest='exclude_patterns', 
        help='File pattern to exclude (can be specified multiple times)'
    )
    
    parser.add_argument(
        '--report-file', 
        help='Save detailed report to specified file'
    )
    
    args = parser.parse_args()
    
    # Initialize modernizer
    modernizer = NumpyDtypeModernizer(
        target_dirs=args.target_dirs,
        exclude_patterns=args.exclude_patterns
    )
    
    # Run modernization
    try:
        modernizer.modernize(
            dry_run=args.dry_run,
            backup=args.backup,
            report_file=args.report_file
        )
    except Exception as e:
        print(f"❌ Error during modernization: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()