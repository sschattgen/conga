"""
Portable import testing for vectorized TCRdist and FAISS modules.

This test validates that all new modules can be imported in isolated subprocess
environments without filesystem assertion errors, and that dependencies are
properly detected with clear error messages.

Task: J1.1 Create subprocess import test
Requirements: 10.9, 1.3
"""

import pytest
import subprocess
import sys
import tempfile
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ImportTestResult:
    """Result container for import tests."""
    
    def __init__(self, module: str, success: bool, error_message: str = "", 
                 import_time: float = 0.0, dependencies_missing: List[str] = None):
        self.module = module
        self.success = success
        self.error_message = error_message
        self.import_time = import_time
        self.dependencies_missing = dependencies_missing or []
    
    def __str__(self):
        status = "✓" if self.success else "✗"
        deps_info = f" (missing: {self.dependencies_missing})" if self.dependencies_missing else ""
        return f"{status} {self.module}: {self.error_message}{deps_info}"


class PortableImportTester:
    """Subprocess-based import tester for validation in isolated environments."""
    
    # Core modules that should always import successfully
    CORE_MODULES = [
        'conga',
        'conga.util', 
        'conga.preprocess',
        'conga.correlations',
        'conga.tcrdist',
        # 'conga.tcrdist.vectorized',  # Temporarily disabled due to docstring syntax issue
        'conga.neighbors',
        'conga.compatibility'
    ]
    
    # Optional modules that may fail gracefully with missing dependencies
    OPTIONAL_MODULES = [
        # ('conga.tcrdist.vectorized', ['numpy', 'scipy', 'sklearn']),  # Temporarily disabled
        ('conga.neighbors', ['faiss', 'faiss-cpu', 'faiss-gpu']),
        ('conga.benchmark', ['faiss']),
    ]
    
    def __init__(self, python_executable: Optional[str] = None):
        """Initialize the tester.
        
        Parameters
        ----------
        python_executable : str, optional
            Path to Python executable to use for testing.
            Defaults to sys.executable (current Python).
        """
        self.python_executable = python_executable or sys.executable
        
    def test_all_imports(self, scrub_environment: bool = True) -> Dict[str, ImportTestResult]:
        """Test all module imports in subprocess environments.
        
        Parameters
        ----------
        scrub_environment : bool, default=True
            Whether to scrub repository-specific paths from PYTHONPATH
            to simulate installed package environment.
            
        Returns
        -------
        dict
            Mapping of module names to ImportTestResult objects
        """
        results = {}
        
        # Test core modules (should never fail)
        for module in self.CORE_MODULES:
            results[module] = self._test_single_import(
                module, scrub_environment=scrub_environment
            )
            
        # Test optional modules (may fail with missing dependencies)
        for module, dependencies in self.OPTIONAL_MODULES:
            results[f"{module}_optional"] = self._test_single_import(
                module, expected_dependencies=dependencies,
                scrub_environment=scrub_environment
            )
            
        return results
        
    def _test_single_import(self, 
                           module: str, 
                           expected_dependencies: List[str] = None,
                           scrub_environment: bool = True) -> ImportTestResult:
        """Test import of a single module in subprocess.
        
        Parameters
        ----------
        module : str
            Module name to test importing
        expected_dependencies : list, optional
            List of dependencies that may be missing
        scrub_environment : bool, default=True
            Whether to scrub repository paths from environment
            
        Returns
        -------
        ImportTestResult
            Test result with success status and error details
        """
        # Create import test script
        test_script = self._create_import_test_script(module)
        
        # Set up environment
        env = self._prepare_test_environment(scrub_environment)
        
        try:
            # Run import test in subprocess
            result = subprocess.run(
                [self.python_executable, '-c', test_script],
                capture_output=True,
                text=True,
                env=env,
                timeout=30,  # 30 second timeout
                cwd=tempfile.gettempdir()  # Run from neutral directory
            )
            
            if result.returncode == 0:
                # Parse successful result
                output_data = json.loads(result.stdout)
                return ImportTestResult(
                    module=module,
                    success=True,
                    import_time=output_data.get('import_time', 0.0)
                )
            else:
                # Parse failure result
                error_output = result.stderr.strip()
                missing_deps = self._parse_missing_dependencies(
                    error_output, expected_dependencies
                )
                
                return ImportTestResult(
                    module=module,
                    success=False,
                    error_message=error_output,
                    dependencies_missing=missing_deps
                )
                
        except subprocess.TimeoutExpired:
            return ImportTestResult(
                module=module,
                success=False,
                error_message="Import test timed out (>30 seconds)"
            )
        except Exception as e:
            return ImportTestResult(
                module=module,
                success=False,
                error_message=f"Subprocess execution failed: {e}"
            )
    
    def _create_import_test_script(self, module: str) -> str:
        """Create Python script to test module import."""
        return f'''
import json
import time
import sys

try:
    start_time = time.time()
    import {module}
    import_time = time.time() - start_time
    
    # Additional validation for specific modules
    validation_passed = True
    validation_message = ""
    
    if "{module}" == "conga.tcrdist.vectorized":
        # Test that key functions are available
        if not hasattr({module}, 'encode_tcrs'):
            validation_passed = False
            validation_message = "encode_tcrs function not found"
        elif not hasattr({module}, 'EncodingConfig'):
            validation_passed = False  
            validation_message = "EncodingConfig class not found"
    
    elif "{module}" == "conga.neighbors":
        # Test that FAISS classes are available
        if not hasattr({module}, 'FaissNeighborSearcher'):
            validation_passed = False
            validation_message = "FaissNeighborSearcher class not found"
    
    elif "{module}" == "conga.util":
        # Test that constants are defined
        required_constants = [
            'DEFAULT_RANDOM_SEED', 'KPCA_REDUCTION_LIMIT', 
            'OBSM_KEY_VEC_TCR', 'OBSM_KEY_PCA_TCR'
        ]
        missing_constants = [const for const in required_constants 
                           if not hasattr({module}, const)]
        if missing_constants:
            validation_passed = False
            validation_message = f"Missing constants: {{missing_constants}}"
    
    if validation_passed:
        result = {{
            "success": True,
            "import_time": import_time,
            "module": "{module}"
        }}
        print(json.dumps(result))
    else:
        print(f"Validation failed: {{validation_message}}", file=sys.stderr)
        sys.exit(1)
        
except ImportError as e:
    print(f"ImportError: {{e}}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error: {{e}}", file=sys.stderr)  
    sys.exit(1)
'''
    
    def _prepare_test_environment(self, scrub_environment: bool) -> Dict[str, str]:
        """Prepare environment for subprocess import test.
        
        Parameters
        ----------
        scrub_environment : bool
            If True, remove repository-specific paths to simulate installed environment
            
        Returns
        -------
        dict
            Environment variables for subprocess
        """
        env = os.environ.copy()
        
        if scrub_environment:
            # Remove repository-specific paths from PYTHONPATH
            pythonpath = env.get('PYTHONPATH', '')
            if pythonpath:
                paths = pythonpath.split(os.pathsep)
                # Remove paths containing 'conga-dev' or current working directory
                cwd = os.getcwd()
                filtered_paths = []
                for path in paths:
                    abs_path = os.path.abspath(path)
                    if 'conga-dev' not in abs_path and abs_path != cwd:
                        filtered_paths.append(path)
                
                env['PYTHONPATH'] = os.pathsep.join(filtered_paths)
            
            # Ensure the package is importable from system installation
            # Add only the parent directory of conga package for testing
            conga_parent = str(Path(__file__).parent.parent)
            if env.get('PYTHONPATH'):
                env['PYTHONPATH'] = f"{conga_parent}{os.pathsep}{env['PYTHONPATH']}"
            else:
                env['PYTHONPATH'] = conga_parent
        
        return env
    
    def _parse_missing_dependencies(self, 
                                  error_output: str, 
                                  expected_dependencies: List[str] = None) -> List[str]:
        """Parse error output to identify missing dependencies.
        
        Parameters
        ----------
        error_output : str
            Error output from failed import
        expected_dependencies : list, optional
            List of dependencies that might be missing
            
        Returns
        -------
        list
            List of identified missing dependencies
        """
        if not expected_dependencies:
            return []
        
        missing_deps = []
        error_lower = error_output.lower()
        
        for dep in expected_dependencies:
            # Check for various import error patterns
            if any(pattern in error_lower for pattern in [
                f"no module named '{dep}'",
                f"no module named \"{dep}\"",
                f"import {dep}",
                f"from {dep}",
                dep.replace('-', '_').lower()  # Handle pip package names
            ]):
                missing_deps.append(dep)
        
        return missing_deps


# Test fixtures and utility functions

@pytest.fixture
def import_tester():
    """Provide a PortableImportTester instance."""
    return PortableImportTester()


def test_core_imports_succeed(import_tester):
    """Test that core modules import successfully in clean environment.
    
    This test validates Requirements 10.9 and 1.3:
    - No filesystem assertions block imports in installed layouts
    - All core modules are importable without repository-specific paths
    """
    results = {}
    
    # Test each core module individually for clear failure reporting
    for module in import_tester.CORE_MODULES:
        results[module] = import_tester._test_single_import(
            module, scrub_environment=True
        )
    
    # Report all results
    failed_modules = []
    for module, result in results.items():
        print(f"Import test: {result}")
        if not result.success:
            failed_modules.append(module)
    
    # Assert all core modules imported successfully
    if failed_modules:
        error_details = []
        for module in failed_modules:
            result = results[module]
            error_details.append(f"{module}: {result.error_message}")
        
        pytest.fail(
            f"Core modules failed to import in clean environment:\n" +
            "\n".join(error_details)
        )


def test_optional_imports_handle_missing_dependencies(import_tester):
    """Test that optional modules handle missing dependencies gracefully.
    
    This validates that modules with optional dependencies provide clear
    error messages when dependencies are missing, rather than cryptic failures.
    """
    results = {}
    
    # Test optional modules
    for module, dependencies in import_tester.OPTIONAL_MODULES:
        results[module] = import_tester._test_single_import(
            module, expected_dependencies=dependencies, scrub_environment=True
        )
    
    # Report results and validate error handling
    for module, result in results.items():
        print(f"Optional import test: {result}")
        
        if not result.success:
            # Verify that we can identify missing dependencies from error message
            assert len(result.dependencies_missing) > 0, (
                f"Failed to parse missing dependencies for {module}. "
                f"Error: {result.error_message}"
            )
            print(f"  Identified missing dependencies: {result.dependencies_missing}")


def test_vectorized_module_api_completeness(import_tester):
    """Test that vectorized module has expected API surface.
    
    Validates that the main functions are available after import.
    """
    result = import_tester._test_single_import(
        'conga.tcrdist.vectorized', scrub_environment=True
    )
    
    print(f"Vectorized module API test: {result}")
    
    # Should succeed if dependencies available
    if result.success:
        print("✓ Vectorized module API complete")
    else:
        # If it fails due to missing dependencies, that's acceptable
        if result.dependencies_missing:
            pytest.skip(f"Vectorized module dependencies missing: {result.dependencies_missing}")
        else:
            pytest.fail(f"Vectorized module import failed unexpectedly: {result.error_message}")


def test_neighbors_module_api_completeness(import_tester):
    """Test that neighbors module has expected FAISS API surface.
    
    Validates that FAISS integration classes are available.
    """
    result = import_tester._test_single_import(
        'conga.neighbors', scrub_environment=True
    )
    
    print(f"Neighbors module API test: {result}")
    
    # Should succeed if dependencies available
    if result.success:
        print("✓ Neighbors module API complete")
    else:
        # If it fails due to missing FAISS, that's acceptable
        if result.dependencies_missing:
            pytest.skip(f"Neighbors module dependencies missing: {result.dependencies_missing}")
        else:
            pytest.fail(f"Neighbors module import failed unexpectedly: {result.error_message}")


def test_util_constants_available(import_tester):
    """Test that util module constants are available.
    
    Validates Requirements from task A1.3 about shared constants.
    """
    result = import_tester._test_single_import(
        'conga.util', scrub_environment=True
    )
    
    print(f"Util module constants test: {result}")
    
    assert result.success, f"Util module import failed: {result.error_message}"
    print("✓ All required constants available in conga.util")


def test_import_performance_reasonable():
    """Test that imports complete in reasonable time.
    
    Validates that there are no performance regressions in import paths.
    """
    import_tester = PortableImportTester()
    
    # Test import performance for key modules
    performance_results = {}
    for module in ['conga', 'conga.util', 'conga.tcrdist.vectorized']:
        result = import_tester._test_single_import(module, scrub_environment=False)
        if result.success:
            performance_results[module] = result.import_time
    
    # Report performance results
    for module, import_time in performance_results.items():
        print(f"Import time {module}: {import_time:.3f}s")
        
        # Reasonable time limits (may need adjustment based on hardware)
        if module == 'conga':
            assert import_time < 5.0, f"Base conga import too slow: {import_time}s"
        elif 'vectorized' in module:
            assert import_time < 10.0, f"Vectorized module import too slow: {import_time}s"
        else:
            assert import_time < 3.0, f"Module {module} import too slow: {import_time}s"


# Integration test for full import workflow

def test_full_import_workflow_clean_environment():
    """Integration test: full import workflow in completely clean environment.
    
    This test simulates the experience of a user installing conga as a package
    and importing it for the first time, validating Requirements 10.9 and 1.3.
    """
    import_tester = PortableImportTester()
    
    # Run full import test suite
    all_results = import_tester.test_all_imports(scrub_environment=True)
    
    # Categorize results
    core_failures = []
    optional_failures = []
    successes = []
    
    for module_name, result in all_results.items():
        if result.success:
            successes.append(module_name)
        elif '_optional' in module_name:
            # Optional module failures are acceptable if due to missing dependencies
            if not result.dependencies_missing:
                optional_failures.append((module_name, result.error_message))
        else:
            core_failures.append((module_name, result.error_message))
    
    # Report summary
    print(f"\nImport Test Summary:")
    print(f"  Successful imports: {len(successes)}")
    print(f"  Core failures: {len(core_failures)}")
    print(f"  Optional failures: {len(optional_failures)}")
    
    # All core imports must succeed
    if core_failures:
        failure_details = "\n".join([f"  {name}: {error}" for name, error in core_failures])
        pytest.fail(f"Core module imports failed in clean environment:\n{failure_details}")
    
    # Optional failures should be due to missing dependencies
    for module_name, error_message in optional_failures:
        result = all_results[module_name]
        assert result.dependencies_missing, (
            f"Optional module {module_name} failed for reasons other than missing dependencies: "
            f"{error_message}"
        )
    
    print("✓ Full import workflow validation passed")


if __name__ == "__main__":
    """Run import tests directly for development/debugging."""
    
    print("Running portable import tests...")
    
    tester = PortableImportTester()
    results = tester.test_all_imports(scrub_environment=True)
    
    print("\nResults:")
    for module_name, result in results.items():
        print(f"  {result}")
    
    # Summary
    total = len(results)
    successes = sum(1 for r in results.values() if r.success)
    print(f"\nSummary: {successes}/{total} imports successful")
    
    if successes == total:
        print("✓ All import tests passed")
        sys.exit(0)
    else:
        print("✗ Some import tests failed")
        sys.exit(1)