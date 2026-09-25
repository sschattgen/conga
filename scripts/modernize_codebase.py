#!/usr/bin/env python3
"""
CoNGA Pandas 3.0/NumPy 2.0 Modernization Orchestrator

This script orchestrates the complete modernization process for CoNGA,
integrating all three modernization tools into a cohesive workflow.

Usage:
    python modernize_codebase.py [options]

Examples:
    # Full modernization workflow
    python modernize_codebase.py --full-modernization
    
    # Audit-only mode (no changes)
    python modernize_codebase.py --audit-only
    
    # Validation-only mode  
    python modernize_codebase.py --validate-only
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Optional
import tempfile
from datetime import datetime


class ModernizationOrchestrator:
    """Orchestrates the complete pandas/numpy modernization workflow."""
    
    def __init__(self, dry_run: bool = False, backup: bool = True):
        """Initialize the orchestrator.
        
        Args:
            dry_run: If True, only analyze without making changes
            backup: If True, create backup files before modifications
        """
        self.dry_run = dry_run
        self.backup = backup
        self.scripts_dir = Path(__file__).parent
        self.project_root = self.scripts_dir.parent
        
        # Tool paths
        self.numpy_modernizer = self.scripts_dir / "modernize_numpy_dtypes.py"
        self.cow_auditor = self.scripts_dir / "audit_cow_patterns.py"
        self.result_tester = self.scripts_dir / "test_modernization_results.py"
        
        # Results storage
        self.results_dir = self.project_root / "modernization_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def run_tool(self, tool_path: Path, args: List[str], 
                description: str) -> Dict[str, any]:
        """Run a modernization tool and capture results.
        
        Args:
            tool_path: Path to the tool script
            args: Command line arguments for the tool
            description: Human-readable description of what the tool does
            
        Returns:
            Dictionary with tool execution results
        """
        print(f"\n🔧 {description}")
        print("=" * 60)
        
        # Prepare command
        cmd = ['mamba', 'run', '-n', 'conga-dev', 'python', str(tool_path)] + args
        
        print(f"Running: {' '.join(cmd)}")
        print()
        
        try:
            # Run the tool
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
                timeout=600  # 10 minute timeout
            )
            
            # Process results
            success = result.returncode == 0
            
            if success:
                print("✅ Tool completed successfully")
            else:
                print("❌ Tool failed or found issues")
            
            # Show output
            if result.stdout:
                print("\nStdout:")
                print(result.stdout)
            
            if result.stderr and not success:
                print("\nStderr:")  
                print(result.stderr)
            
            return {
                'tool': str(tool_path.name),
                'args': args,
                'success': success,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'description': description
            }
            
        except subprocess.TimeoutExpired:
            print("⏰ Tool timed out after 10 minutes")
            return {
                'tool': str(tool_path.name),
                'args': args, 
                'success': False,
                'returncode': -1,
                'stdout': '',
                'stderr': 'Timeout after 10 minutes',
                'description': description
            }
        except Exception as e:
            print(f"❌ Error running tool: {e}")
            return {
                'tool': str(tool_path.name),
                'args': args,
                'success': False,
                'returncode': -1,
                'stdout': '',
                'stderr': str(e),
                'description': description
            }
    
    def audit_phase(self) -> List[Dict[str, any]]:
        """Run the audit phase to identify issues."""
        print("🔍 PHASE 1: AUDIT AND ANALYSIS")
        print("=" * 80)
        
        results = []
        
        # 1. Audit copy-on-write patterns
        cow_report_file = self.results_dir / f"cow_audit_{self.timestamp}.txt"
        cow_json_file = self.results_dir / f"cow_audit_{self.timestamp}.json"
        
        cow_result = self.run_tool(
            self.cow_auditor,
            [
                '--report-file', str(cow_report_file),
                '--output-format', 'text'
            ],
            "Auditing for pandas 3.0 copy-on-write unsafe patterns"
        )
        results.append(cow_result)
        
        # Also get JSON report for programmatic analysis
        cow_json_result = self.run_tool(
            self.cow_auditor,
            [
                '--report-file', str(cow_json_file),
                '--output-format', 'json'
            ],
            "Generating JSON audit report for copy-on-write patterns"
        )
        results.append(cow_json_result)
        
        # 2. Preview NumPy dtype changes
        numpy_preview_result = self.run_tool(
            self.numpy_modernizer,
            ['--dry-run'],
            "Previewing NumPy legacy dtype alias replacements"
        )
        results.append(numpy_preview_result)
        
        return results
    
    def modernization_phase(self) -> List[Dict[str, any]]:
        """Run the modernization phase to apply changes."""
        if self.dry_run:
            print("⏭️  SKIPPING PHASE 2: MODERNIZATION (dry-run mode)")
            return []
        
        print("\n🛠️  PHASE 2: APPLY MODERNIZATION CHANGES")
        print("=" * 80)
        
        results = []
        
        # 1. Apply NumPy dtype modernization
        numpy_args = []
        if self.backup:
            numpy_args.append('--backup')
        else:
            numpy_args.append('--no-backup')
        
        numpy_report_file = self.results_dir / f"numpy_modernization_{self.timestamp}.txt"
        numpy_args.extend(['--report-file', str(numpy_report_file)])
        
        numpy_result = self.run_tool(
            self.numpy_modernizer,
            numpy_args,
            "Applying NumPy legacy dtype alias modernization"
        )
        results.append(numpy_result)
        
        return results
    
    def validation_phase(self) -> List[Dict[str, any]]:
        """Run the validation phase to verify changes."""
        print("\n🧪 PHASE 3: VALIDATION")
        print("=" * 80)
        
        results = []
        
        if self.dry_run:
            print("ℹ️  In dry-run mode, validation shows what would be tested")
            print("   Run without --dry-run to create baseline and validate changes")
            return results
        
        # 1. Create or update baseline
        baseline_result = self.run_tool(
            self.result_tester,
            ['--create-baseline'],
            "Creating/updating baseline results for validation"
        )
        results.append(baseline_result)
        
        # 2. Validate against baseline (if baseline creation succeeded)
        if baseline_result['success']:
            validation_report_file = self.results_dir / f"validation_report_{self.timestamp}.txt"
            
            validation_result = self.run_tool(
                self.result_tester,
                [
                    '--validate',
                    '--report-file', str(validation_report_file)
                ],
                "Validating modernized results against baseline"
            )
            results.append(validation_result)
        else:
            print("⚠️  Skipping validation due to baseline creation failure")
        
        return results
    
    def generate_summary_report(self, all_results: List[Dict[str, any]]) -> None:
        """Generate a comprehensive summary report."""
        print("\n📊 MODERNIZATION SUMMARY")
        print("=" * 80)
        
        # Count results by phase
        audit_results = [r for r in all_results if 'audit' in r['description'].lower()]
        modernization_results = [r for r in all_results if 'applying' in r['description'].lower()]
        validation_results = [r for r in all_results if 'validat' in r['description'].lower()]
        
        print(f"Phase 1 (Audit): {len(audit_results)} tools run")
        print(f"Phase 2 (Modernization): {len(modernization_results)} tools run") 
        print(f"Phase 3 (Validation): {len(validation_results)} tools run")
        print()
        
        # Success summary
        total_tools = len(all_results)
        successful_tools = sum(1 for r in all_results if r['success'])
        success_rate = successful_tools / total_tools if total_tools > 0 else 0
        
        print(f"Overall Success Rate: {successful_tools}/{total_tools} ({success_rate:.1%})")
        print()
        
        # Detailed results
        print("Detailed Results:")
        print("-" * 40)
        
        for i, result in enumerate(all_results, 1):
            status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
            tool_name = result['tool']
            description = result['description']
            
            print(f"{i:2d}. {status}: {tool_name}")
            print(f"     {description}")
            
            if not result['success'] and result['stderr']:
                # Show first line of error
                error_preview = result['stderr'].split('\n')[0][:100]
                print(f"     Error: {error_preview}...")
            print()
        
        # Save detailed results to JSON
        results_json_file = self.results_dir / f"modernization_summary_{self.timestamp}.json"
        
        summary_data = {
            'timestamp': self.timestamp,
            'dry_run': self.dry_run,
            'backup': self.backup,
            'total_tools_run': total_tools,
            'successful_tools': successful_tools,
            'success_rate': success_rate,
            'tool_results': all_results,
            'files_generated': [
                str(f.relative_to(self.project_root))
                for f in self.results_dir.glob(f"*_{self.timestamp}.*")
            ]
        }
        
        with open(results_json_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"📄 Detailed results saved to: {results_json_file}")
        print(f"📁 All output files in: {self.results_dir}")
    
    def run_full_modernization(self) -> bool:
        """Run the complete modernization workflow.
        
        Returns:
            True if all phases completed successfully
        """
        print("🚀 CoNGA Pandas 3.0/NumPy 2.0 Modernization Workflow")
        print("=" * 80)
        
        if self.dry_run:
            print("🔍 DRY RUN MODE - No files will be modified")
        else:
            print("⚠️  LIVE MODE - Files will be modified")
            if self.backup:
                print("💾 Backup files will be created")
        
        print(f"📁 Results will be saved to: {self.results_dir}")
        print()
        
        all_results = []
        
        # Phase 1: Audit
        audit_results = self.audit_phase()
        all_results.extend(audit_results)
        
        # Phase 2: Modernization
        modernization_results = self.modernization_phase()
        all_results.extend(modernization_results)
        
        # Phase 3: Validation
        validation_results = self.validation_phase()
        all_results.extend(validation_results)
        
        # Generate summary
        self.generate_summary_report(all_results)
        
        # Determine overall success
        overall_success = all(r['success'] for r in all_results)
        
        if overall_success:
            print("🎉 MODERNIZATION COMPLETED SUCCESSFULLY!")
            if not self.dry_run:
                print("   Your codebase is now pandas 3.0+/NumPy 2.0+ compatible!")
        else:
            print("⚠️  MODERNIZATION COMPLETED WITH ISSUES")
            print("   Review the detailed results and address any failures")
        
        return overall_success


def main():
    """Command-line interface for the modernization orchestrator."""
    parser = argparse.ArgumentParser(
        description='Orchestrate complete pandas 3.0/NumPy 2.0 modernization workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --full-modernization         # Complete modernization workflow
  %(prog)s --audit-only                 # Only run audit phase  
  %(prog)s --validate-only              # Only run validation phase
  %(prog)s --dry-run                    # Preview without making changes
        """
    )
    
    parser.add_argument(
        '--full-modernization', action='store_true',
        help='Run complete modernization workflow (audit + modernize + validate)'
    )
    
    parser.add_argument(
        '--audit-only', action='store_true',
        help='Only run audit phase (no modifications)'
    )
    
    parser.add_argument(
        '--validate-only', action='store_true',
        help='Only run validation phase'
    )
    
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Preview changes without modifying files'
    )
    
    parser.add_argument(
        '--no-backup', dest='backup', action='store_false', default=True,
        help='Skip creating backup files (not recommended)'
    )
    
    args = parser.parse_args()
    
    if not any([args.full_modernization, args.audit_only, args.validate_only]):
        parser.error("Must specify one of --full-modernization, --audit-only, or --validate-only")
    
    # Initialize orchestrator
    orchestrator = ModernizationOrchestrator(
        dry_run=args.dry_run,
        backup=args.backup
    )
    
    try:
        if args.full_modernization:
            success = orchestrator.run_full_modernization()
            
        elif args.audit_only:
            print("🔍 Running audit-only mode...")
            audit_results = orchestrator.audit_phase()
            orchestrator.generate_summary_report(audit_results)
            success = all(r['success'] for r in audit_results)
            
        elif args.validate_only:
            print("🧪 Running validation-only mode...")
            validation_results = orchestrator.validation_phase()
            orchestrator.generate_summary_report(validation_results)
            success = all(r['success'] for r in validation_results)
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n❌ Modernization interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Modernization failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()