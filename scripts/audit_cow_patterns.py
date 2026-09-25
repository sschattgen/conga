#!/usr/bin/env python3
"""
Pandas 3.0 Copy-on-Write Pattern Auditing Script

This script detects pandas copy-on-write unsafe patterns that may cause
silent failures in pandas 3.0+ where CoW is mandatory. It reports
potential issues without making changes.

Usage:
    python audit_cow_patterns.py [options]

Examples:
    # Audit all Python files
    python audit_cow_patterns.py
    
    # Audit specific directory with detailed output
    python audit_cow_patterns.py --target-dir conga/preprocess.py --verbose
    
    # Generate JSON report
    python audit_cow_patterns.py --output-format json --report-file cow_audit.json
"""

import os
import re
import sys
import argparse
import ast
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import inspect


class RiskLevel(Enum):
    """Risk levels for copy-on-write issues."""
    HIGH = "high"        # Almost certainly breaks with CoW
    MEDIUM = "medium"    # May break depending on context  
    LOW = "low"          # Potentially problematic
    INFO = "info"        # Worth noting but likely safe


@dataclass 
class COWIssue:
    """Represents a potential copy-on-write issue."""
    file_path: str
    line_number: int
    column: int
    pattern_type: str
    risk_level: RiskLevel
    code_snippet: str
    description: str
    suggestion: str
    context_lines: List[str]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d['risk_level'] = self.risk_level.value
        return d


class COWPatternDetector:
    """Detector for pandas copy-on-write unsafe patterns."""
    
    def __init__(self, target_dirs: List[str] = None, exclude_patterns: List[str] = None):
        """Initialize the detector.
        
        Args:
            target_dirs: Directories to scan (default: ['conga', 'scripts'])  
            exclude_patterns: File patterns to exclude
        """
        self.target_dirs = target_dirs or ['conga', 'scripts']
        self.exclude_patterns = exclude_patterns or [
            '*.pyc', '__pycache__', '.git', '.pytest_cache',
            '*.egg-info', '.DS_Store', '*.bak', '*.backup'  
        ]
        
        # Define pattern detection rules
        self._init_detection_patterns()
    
    def _init_detection_patterns(self):
        """Initialize regex patterns for different CoW issues."""
        
        # Pattern 1: Classic chained assignment (df[col][mask] = value)
        self.chained_assignment_patterns = [
            {
                'regex': re.compile(r'(\w+)\[([^\]]+)\]\[([^\]]+)\]\s*='),
                'description': 'Chained assignment with double indexing',
                'risk_level': RiskLevel.HIGH,
                'suggestion': 'Use df.loc[mask, col] = value instead'
            },
            {
                'regex': re.compile(r'(\w+)\.([a-zA-Z_]\w*)\[([^\]]+)\]\s*='), 
                'description': 'Column selection followed by indexing assignment',
                'risk_level': RiskLevel.MEDIUM,
                'suggestion': 'Use df.loc[mask, "column"] = value instead'
            }
        ]
        
        # Pattern 2: Query chaining (df.query(...)[col] = value)
        self.query_chaining_patterns = [
            {
                'regex': re.compile(r'(\w+)\.query\([^)]+\)\[([^\]]+)\]\s*='),
                'description': 'Assignment after query() operation',
                'risk_level': RiskLevel.HIGH,
                'suggestion': 'Use mask = df.query(...).index; df.loc[mask, col] = value'
            },
            {
                'regex': re.compile(r'(\w+)\.query\([^)]+\)\.([a-zA-Z_]\w*)\s*='),
                'description': 'Column assignment after query() operation', 
                'risk_level': RiskLevel.HIGH,
                'suggestion': 'Use boolean indexing with df.loc[] instead'
            }
        ]
        
        # Pattern 3: Groupby transform assignments
        self.groupby_patterns = [
            {
                'regex': re.compile(r'(\w+)\.groupby\([^)]+\)\.([a-zA-Z_]\w*)\.(transform|apply)\([^)]+\)\s*='),
                'description': 'Assignment to groupby transform result',
                'risk_level': RiskLevel.MEDIUM, 
                'suggestion': 'Assign result to new variable, then use df.loc[]'
            }
        ]
        
        # Pattern 4: Mixed indexing (df[mask].loc[:, col] = value)
        self.mixed_indexing_patterns = [
            {
                'regex': re.compile(r'(\w+)\[([^\]]+)\]\.loc\[([^\]]+),\s*([^\]]+)\]\s*='),
                'description': 'Mixed boolean indexing with loc',
                'risk_level': RiskLevel.MEDIUM,
                'suggestion': 'Combine conditions: df.loc[mask & condition, col] = value'
            }
        ]
        
        # Pattern 5: Inplace operations on selections
        self.inplace_patterns = [
            {
                'regex': re.compile(r'(\w+)\[([^\]]+)\]\.([a-zA-Z_]\w*)\([^)]*inplace\s*=\s*True'),
                'description': 'Inplace operation on DataFrame selection',
                'risk_level': RiskLevel.HIGH,
                'suggestion': 'Use df.loc[] for assignment or avoid inplace on selections'
            }
        ]
        
        # Combine all patterns
        self.all_patterns = {
            'chained_assignment': self.chained_assignment_patterns,
            'query_chaining': self.query_chaining_patterns, 
            'groupby_issues': self.groupby_patterns,
            'mixed_indexing': self.mixed_indexing_patterns,
            'inplace_operations': self.inplace_patterns
        }
    
    def find_python_files(self) -> List[Path]:
        """Find all Python files to analyze."""
        python_files = []
        
        for target_dir in self.target_dirs:
            target_path = Path(target_dir)
            if not target_path.exists():
                print(f"Warning: Target directory {target_dir} does not exist")
                continue
                
            for py_file in target_path.rglob('*.py'):
                if self._should_exclude_file(py_file):
                    continue
                python_files.append(py_file)
        
        return sorted(python_files)
    
    def _should_exclude_file(self, file_path: Path) -> bool:
        """Check if file should be excluded."""
        file_str = str(file_path)
        for pattern in self.exclude_patterns:
            if pattern in file_str:
                return True
        return False
    
    def analyze_file(self, file_path: Path) -> List[COWIssue]:
        """Analyze a single file for CoW issues."""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except (UnicodeDecodeError, IOError) as e:
            print(f"Warning: Could not read {file_path}: {e}")
            return issues
        
        # Analyze each line with regex patterns
        for line_num, line in enumerate(lines, 1):
            line_issues = self._analyze_line(
                file_path, line_num, line, lines
            )
            issues.extend(line_issues)
        
        # Additional AST-based analysis for complex patterns
        try:
            ast_issues = self._analyze_ast(file_path, ''.join(lines))
            issues.extend(ast_issues)
        except SyntaxError:
            # Skip files with syntax errors
            pass
        
        return issues
    
    def _analyze_line(self, file_path: Path, line_num: int, line: str, 
                     all_lines: List[str]) -> List[COWIssue]:
        """Analyze a single line for CoW patterns."""
        issues = []
        
        for pattern_category, patterns in self.all_patterns.items():
            for pattern_info in patterns:
                matches = pattern_info['regex'].finditer(line)
                
                for match in matches:
                    # Get context lines (2 before, 2 after)
                    context_start = max(0, line_num - 3)
                    context_end = min(len(all_lines), line_num + 2)
                    context_lines = [
                        f"{i+1:4d}: {all_lines[i].rstrip()}"
                        for i in range(context_start, context_end)
                    ]
                    
                    issue = COWIssue(
                        file_path=str(file_path),
                        line_number=line_num,
                        column=match.start() + 1,
                        pattern_type=pattern_category,
                        risk_level=pattern_info['risk_level'],
                        code_snippet=line.strip(),
                        description=pattern_info['description'],
                        suggestion=pattern_info['suggestion'],
                        context_lines=context_lines
                    )
                    issues.append(issue)
        
        return issues
    
    def _analyze_ast(self, file_path: Path, content: str) -> List[COWIssue]:
        """Use AST analysis to detect more complex patterns."""
        issues = []
        
        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError:
            return issues
        
        # Look for specific AST patterns
        for node in ast.walk(tree):
            # Detect chained subscript assignments
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if self._is_chained_subscript(target):
                        issue = self._create_ast_issue(
                            file_path, node, target,
                            'AST detected chained subscript assignment',
                            RiskLevel.HIGH,
                            'Use df.loc[row_indexer, col_indexer] = value'
                        )
                        if issue:
                            issues.append(issue)
        
        return issues
    
    def _is_chained_subscript(self, node: ast.AST) -> bool:
        """Check if AST node represents chained subscript access."""
        if isinstance(node, ast.Subscript):
            # Check if the value is also a subscript (chain)
            return isinstance(node.value, ast.Subscript)
        return False
    
    def _create_ast_issue(self, file_path: Path, node: ast.AST, target: ast.AST,
                         description: str, risk_level: RiskLevel, 
                         suggestion: str) -> Optional[COWIssue]:
        """Create COWIssue from AST node."""
        if not hasattr(node, 'lineno'):
            return None
            
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            line_num = node.lineno
            if line_num <= len(lines):
                code_snippet = lines[line_num - 1].strip()
                
                # Context lines
                context_start = max(0, line_num - 3)
                context_end = min(len(lines), line_num + 2) 
                context_lines = [
                    f"{i+1:4d}: {lines[i].rstrip()}"
                    for i in range(context_start, context_end)
                ]
                
                return COWIssue(
                    file_path=str(file_path),
                    line_number=line_num,
                    column=getattr(node, 'col_offset', 0) + 1,
                    pattern_type='ast_analysis',
                    risk_level=risk_level,
                    code_snippet=code_snippet,
                    description=description,
                    suggestion=suggestion,
                    context_lines=context_lines
                )
        except (IOError, IndexError):
            pass
            
        return None
    
    def audit_all_files(self) -> Dict[str, List[COWIssue]]:
        """Audit all target files for CoW issues."""
        all_issues = {}
        python_files = self.find_python_files()
        
        print(f"Auditing {len(python_files)} Python files for CoW patterns...")
        
        for file_path in python_files:
            issues = self.analyze_file(file_path)
            if issues:
                all_issues[str(file_path)] = issues
        
        return all_issues
    
    def generate_report(self, all_issues: Dict[str, List[COWIssue]], 
                       output_format: str = 'text') -> str:
        """Generate audit report in specified format."""
        if output_format == 'json':
            return self._generate_json_report(all_issues)
        else:
            return self._generate_text_report(all_issues)
    
    def _generate_text_report(self, all_issues: Dict[str, List[COWIssue]]) -> str:
        """Generate human-readable text report."""
        lines = []
        lines.append("Pandas 3.0 Copy-on-Write Pattern Audit Report")
        lines.append("=" * 60)
        lines.append("")
        
        total_files = len(all_issues)
        total_issues = sum(len(issues) for issues in all_issues.values())
        
        # Summary
        lines.append(f"Summary:")
        lines.append(f"  Files with issues: {total_files}")
        lines.append(f"  Total issues found: {total_issues}")
        lines.append("")
        
        # Count by risk level
        risk_counts = {level: 0 for level in RiskLevel}
        for issues in all_issues.values():
            for issue in issues:
                risk_counts[issue.risk_level] += 1
        
        lines.append("Issues by risk level:")
        for level in [RiskLevel.HIGH, RiskLevel.MEDIUM, RiskLevel.LOW, RiskLevel.INFO]:
            count = risk_counts[level]
            if count > 0:
                lines.append(f"  {level.value.upper()}: {count}")
        lines.append("")
        
        # Count by pattern type
        pattern_counts = {}
        for issues in all_issues.values():
            for issue in issues:
                pattern_counts[issue.pattern_type] = pattern_counts.get(issue.pattern_type, 0) + 1
        
        lines.append("Issues by pattern type:")
        for pattern_type, count in sorted(pattern_counts.items()):
            lines.append(f"  {pattern_type}: {count}")
        lines.append("")
        
        # Detailed issues
        lines.append("Detailed Issues:")
        lines.append("-" * 40)
        
        for file_path, issues in sorted(all_issues.items()):
            lines.append(f"\n📁 File: {file_path}")
            
            # Sort issues by line number
            issues_sorted = sorted(issues, key=lambda x: x.line_number)
            
            for issue in issues_sorted:
                risk_emoji = {
                    RiskLevel.HIGH: "🔴",
                    RiskLevel.MEDIUM: "🟡", 
                    RiskLevel.LOW: "🟢",
                    RiskLevel.INFO: "ℹ️"
                }
                
                lines.append(f"  {risk_emoji[issue.risk_level]} Line {issue.line_number}:{issue.column} - {issue.risk_level.value.upper()}")
                lines.append(f"     Pattern: {issue.pattern_type}")
                lines.append(f"     Issue: {issue.description}")
                lines.append(f"     Code: {issue.code_snippet}")
                lines.append(f"     Fix: {issue.suggestion}")
                lines.append("")
        
        return "\n".join(lines)
    
    def _generate_json_report(self, all_issues: Dict[str, List[COWIssue]]) -> str:
        """Generate JSON report for machine processing."""
        report_data = {
            'summary': {
                'files_with_issues': len(all_issues),
                'total_issues': sum(len(issues) for issues in all_issues.values()),
                'risk_level_counts': {},
                'pattern_type_counts': {}
            },
            'files': {}
        }
        
        # Calculate summary statistics
        risk_counts = {level.value: 0 for level in RiskLevel}
        pattern_counts = {}
        
        for issues in all_issues.values():
            for issue in issues:
                risk_counts[issue.risk_level.value] += 1
                pattern_counts[issue.pattern_type] = pattern_counts.get(issue.pattern_type, 0) + 1
        
        report_data['summary']['risk_level_counts'] = risk_counts
        report_data['summary']['pattern_type_counts'] = pattern_counts
        
        # Convert issues to serializable format
        for file_path, issues in all_issues.items():
            report_data['files'][file_path] = [issue.to_dict() for issue in issues]
        
        return json.dumps(report_data, indent=2)
    
    def run_audit(self, output_format: str = 'text', report_file: str = None) -> None:
        """Run the complete audit process."""
        print("🔍 Pandas 3.0 Copy-on-Write Pattern Audit")
        print("=" * 50)
        print()
        
        all_issues = self.audit_all_files()
        
        if not all_issues:
            print("✅ No copy-on-write issues found!")
            print("   All files appear to be pandas 3.0 CoW compatible.")
            return
        
        # Generate and display report
        report = self.generate_report(all_issues, output_format)
        print(report)
        
        # Save report if requested
        if report_file:
            with open(report_file, 'w') as f:
                f.write(report)
            print(f"\n📄 Report saved to: {report_file}")
        
        # Summary message
        total_files = len(all_issues)
        total_issues = sum(len(issues) for issues in all_issues.values())
        high_risk_issues = sum(
            1 for issues in all_issues.values() 
            for issue in issues 
            if issue.risk_level == RiskLevel.HIGH
        )
        
        print(f"\n⚠️  Found {total_issues} potential CoW issues in {total_files} files")
        if high_risk_issues > 0:
            print(f"   🔴 {high_risk_issues} high-risk issues require attention")
        print("   Review each issue and apply suggested fixes before pandas 3.0+ deployment")


def main():
    """Command-line interface for the CoW pattern auditor."""
    parser = argparse.ArgumentParser(
        description='Audit pandas copy-on-write unsafe patterns for pandas 3.0+ compatibility',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Audit all Python files  
  %(prog)s --target-dir conga           # Only audit conga/ directory
  %(prog)s --output-format json         # Generate JSON report
  %(prog)s --report-file cow_audit.txt  # Save report to file
        """
    )
    
    parser.add_argument(
        '--target-dir', action='append', dest='target_dirs',
        help='Directory to audit (can be specified multiple times)'
    )
    
    parser.add_argument(
        '--exclude', action='append', dest='exclude_patterns',
        help='File pattern to exclude (can be specified multiple times)' 
    )
    
    parser.add_argument(
        '--output-format', choices=['text', 'json'], default='text',
        help='Report output format (default: text)'
    )
    
    parser.add_argument(
        '--report-file',
        help='Save report to specified file'
    )
    
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = COWPatternDetector(
        target_dirs=args.target_dirs,
        exclude_patterns=args.exclude_patterns
    )
    
    # Run audit
    try:
        detector.run_audit(
            output_format=args.output_format,
            report_file=args.report_file
        )
    except Exception as e:
        print(f"❌ Error during audit: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()