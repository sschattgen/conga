"""
Compatibility utilities for pandas 3.0+ and NumPy 2.0+ environments.

This module provides version checking, compatibility diagnostics, and utility functions
for safe operation across different versions of pandas, NumPy, and related dependencies.
"""

import warnings
import sys
from packaging import version
from typing import Optional, Dict, Any

# Version requirements
MIN_PANDAS_VERSION = "3.0.0"
MIN_NUMPY_VERSION = "2.0.0"
MIN_PYTHON_VERSION = "3.12.0"
MIN_SCANPY_VERSION = "1.9.0"
MIN_ANNDATA_VERSION = "0.9.0"

# Helper function for getting package versions
def _get_package_version(package_name: str) -> Optional[str]:
    """Get package version using modern importlib.metadata approach."""
    try:
        from importlib import metadata
        return metadata.version(package_name)
    except (ImportError, metadata.PackageNotFoundError):
        # Fallback to package __version__ attribute
        try:
            package = __import__(package_name)
            return getattr(package, '__version__', None)
        except ImportError:
            return None

# Try to import dependencies to check availability
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
    PANDAS_VERSION = _get_package_version('pandas')
except ImportError:
    PANDAS_AVAILABLE = False
    PANDAS_VERSION = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True  
    NUMPY_VERSION = _get_package_version('numpy')
except ImportError:
    NUMPY_AVAILABLE = False
    NUMPY_VERSION = None

try:
    import scanpy as sc
    SCANPY_AVAILABLE = True
    SCANPY_VERSION = _get_package_version('scanpy')
except ImportError:
    SCANPY_AVAILABLE = False
    SCANPY_VERSION = None

try:
    import anndata as ad
    ANNDATA_AVAILABLE = True
    ANNDATA_VERSION = _get_package_version('anndata')
except ImportError:
    ANNDATA_AVAILABLE = False
    ANNDATA_VERSION = None


class CompatibilityError(Exception):
    """Raised when environment has critical compatibility issues."""
    pass


class CompatibilityWarning(UserWarning):
    """Warning for non-critical compatibility issues."""
    pass


def check_version_compatibility(package_name: str, 
                              current_version: str, 
                              minimum_version: str,
                              critical: bool = True) -> bool:
    """
    Check if package version meets minimum requirements.
    
    Parameters
    ----------
    package_name : str
        Name of the package being checked
    current_version : str
        Currently installed version string
    minimum_version : str
        Minimum required version string
    critical : bool, default True
        Whether version mismatch is critical (raises error) or warning
        
    Returns
    -------
    bool
        True if version is compatible, False otherwise
        
    Raises
    ------
    CompatibilityError
        If critical=True and version is incompatible
    """
    if current_version is None:
        msg = f"{package_name} is not installed"
        if critical:
            raise CompatibilityError(msg)
        else:
            warnings.warn(msg, CompatibilityWarning)
        return False
        
    try:
        current = version.parse(current_version)
        minimum = version.parse(minimum_version)
        
        if current < minimum:
            msg = (f"{package_name} version {current_version} is below minimum "
                   f"required {minimum_version}. Please upgrade.")
            if critical:
                raise CompatibilityError(msg)
            else:
                warnings.warn(msg, CompatibilityWarning)
            return False
            
    except version.InvalidVersion as e:
        msg = f"Unable to parse version for {package_name}: {e}"
        warnings.warn(msg, CompatibilityWarning)
        return False
        
    return True


def check_pandas_compatibility() -> Dict[str, Any]:
    """
    Check pandas compatibility and configure for CoNGA usage.
    
    Returns
    -------
    dict
        Compatibility status and configuration info
    """
    status = {
        'compatible': False,
        'version': PANDAS_VERSION,
        'cow_enabled': False,
        'warnings': []
    }
    
    if not PANDAS_AVAILABLE:
        status['warnings'].append("pandas not available")
        return status
        
    # Check version requirement
    compatible = check_version_compatibility(
        'pandas', PANDAS_VERSION, MIN_PANDAS_VERSION, critical=False
    )
    status['compatible'] = compatible
    
    if compatible:
        # Enable copy-on-write if available (pandas 3.0+)
        try:
            if hasattr(pd.options.mode, 'copy_on_write'):
                # Check if we're using pandas 3.x where copy_on_write can still be set
                pandas_ver = version.parse(PANDAS_VERSION)
                if pandas_ver >= version.parse("3.0.0") and pandas_ver < version.parse("4.0.0"):
                    # In pandas 3.x, try to enable CoW but handle deprecation warnings
                    import warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=FutureWarning, message=".*copy_on_write.*")
                        try:
                            pd.options.mode.copy_on_write = True
                            status['cow_enabled'] = True
                        except Exception:
                            pass  # Setting may fail in some pandas versions
                elif pandas_ver >= version.parse("4.0.0"):
                    # In pandas 4.0+, CoW is always enabled
                    status['cow_enabled'] = True
                else:
                    # Pre-3.0 pandas
                    pd.options.mode.copy_on_write = True
                    status['cow_enabled'] = True
                    
                if status['cow_enabled']:
                    status['warnings'].append("Copy-on-write enabled for pandas 3.0+ compatibility")
        except Exception as e:
            status['warnings'].append(f"Could not configure copy-on-write: {e}")
            
        # Check for known pandas 3.0 issues
        pandas_ver = version.parse(PANDAS_VERSION)
        if pandas_ver >= version.parse("3.0.0"):
            status['warnings'].append(
                "Using pandas 3.0+: DataFrame mutations now use copy-on-write semantics"
            )
            
    return status


def check_numpy_compatibility() -> Dict[str, Any]:
    """
    Check NumPy compatibility and validate for CoNGA usage.
    
    Returns
    -------
    dict
        Compatibility status and warnings
    """
    status = {
        'compatible': False,
        'version': NUMPY_VERSION, 
        'legacy_aliases_available': False,
        'warnings': []
    }
    
    if not NUMPY_AVAILABLE:
        status['warnings'].append("NumPy not available")
        return status
        
    # Check version requirement  
    compatible = check_version_compatibility(
        'numpy', NUMPY_VERSION, MIN_NUMPY_VERSION, critical=False
    )
    status['compatible'] = compatible
    
    if compatible:
        # Check for removed legacy aliases (NumPy 2.0+)
        numpy_ver = version.parse(NUMPY_VERSION)
        if numpy_ver >= version.parse("2.0.0"):
            # Test if legacy aliases still exist
            try:
                _ = np.float64
                status['legacy_aliases_available'] = True
                status['warnings'].append(
                    "NumPy 2.0+ detected but legacy aliases still available (unusual)"
                )
            except AttributeError:
                status['legacy_aliases_available'] = False
                status['warnings'].append(
                    "NumPy 2.0+ detected: legacy aliases (np.float64, np.int64) removed"
                )
                
        # Warn about NEP 50 casting changes
        if numpy_ver >= version.parse("2.0.0"):
            status['warnings'].append(
                "NumPy 2.0+ uses NEP 50 casting rules - verify numeric operations"
            )
            
    return status


def check_environment_compatibility(verbose: bool = True) -> Dict[str, Any]:
    """
    Main compatibility check function for CoNGA environment.
    
    Performs comprehensive validation of pandas, NumPy, and related dependencies
    for compatibility with CoNGA workflows.
    
    Parameters
    ----------
    verbose : bool, default True
        Whether to print detailed compatibility information
        
    Returns
    -------
    dict
        Complete compatibility status across all dependencies
        
    Raises
    ------
    CompatibilityError
        If critical compatibility issues are detected
    """
    overall_status = {
        'compatible': True,
        'python': {
            'version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'compatible': False
        },
        'pandas': {},
        'numpy': {},
        'scanpy': {'compatible': False, 'version': SCANPY_VERSION},
        'anndata': {'compatible': False, 'version': ANNDATA_VERSION},
        'warnings': [],
        'errors': []
    }
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    try:
        python_compatible = check_version_compatibility(
            'Python', python_version, MIN_PYTHON_VERSION, critical=True
        )
        overall_status['python']['compatible'] = python_compatible
    except CompatibilityError as e:
        overall_status['errors'].append(str(e))
        overall_status['compatible'] = False
        
    # Check pandas
    pandas_status = check_pandas_compatibility()
    overall_status['pandas'] = pandas_status
    if not pandas_status['compatible']:
        overall_status['compatible'] = False
        
    # Check NumPy
    numpy_status = check_numpy_compatibility() 
    overall_status['numpy'] = numpy_status
    if not numpy_status['compatible']:
        overall_status['compatible'] = False
        
    # Check scanpy (non-critical)
    if SCANPY_AVAILABLE:
        scanpy_compatible = check_version_compatibility(
            'scanpy', SCANPY_VERSION, MIN_SCANPY_VERSION, critical=False
        )
        overall_status['scanpy']['compatible'] = scanpy_compatible
    else:
        overall_status['warnings'].append("scanpy not available")
        
    # Check AnnData (non-critical warning)
    if ANNDATA_AVAILABLE:
        anndata_compatible = check_version_compatibility(
            'anndata', ANNDATA_VERSION, MIN_ANNDATA_VERSION, critical=False
        )
        overall_status['anndata']['compatible'] = anndata_compatible  
    else:
        overall_status['warnings'].append("anndata not available")
        
    # Collect all warnings
    all_warnings = []
    all_warnings.extend(overall_status['warnings'])
    all_warnings.extend(pandas_status.get('warnings', []))
    all_warnings.extend(numpy_status.get('warnings', []))
    overall_status['warnings'] = all_warnings
    
    # Print status if verbose
    if verbose:
        print_compatibility_status(overall_status)
        
    # Raise error if critical issues found
    if overall_status['errors']:
        error_msg = "Critical compatibility issues found:\n" + "\n".join(overall_status['errors'])
        raise CompatibilityError(error_msg)
        
    return overall_status


def print_compatibility_status(status: Dict[str, Any]) -> None:
    """Print formatted compatibility status report."""
    print("CoNGA Environment Compatibility Check")
    print("=" * 40)
    
    # Overall status
    overall_status = "✅ COMPATIBLE" if status['compatible'] else "❌ INCOMPATIBLE"
    print(f"Overall Status: {overall_status}")
    print()
    
    # Core dependencies
    print("Core Dependencies:")
    deps = [
        ('Python', status['python']),
        ('pandas', status['pandas']),
        ('numpy', status['numpy'])
    ]
    
    for name, dep_status in deps:
        compat_icon = "✅" if dep_status['compatible'] else "❌"
        version_str = dep_status.get('version', 'N/A')
        print(f"  {compat_icon} {name}: {version_str}")
        
    print()
    
    # Optional dependencies
    print("Optional Dependencies:")
    opt_deps = [
        ('scanpy', status['scanpy']),
        ('anndata', status['anndata'])
    ]
    
    for name, dep_status in opt_deps:
        compat_icon = "✅" if dep_status['compatible'] else "⚠️"
        version_str = dep_status.get('version', 'N/A')
        print(f"  {compat_icon} {name}: {version_str}")
        
    # Warnings and errors
    if status['warnings']:
        print()
        print("Warnings:")
        for warning in status['warnings']:
            print(f"  ⚠️  {warning}")
            
    if status['errors']:
        print()
        print("Errors:")
        for error in status['errors']:
            print(f"  ❌ {error}")
    
    print()


# Utility functions for safe AnnData method access
def safe_obs_columns(adata) -> Any:
    """
    Get AnnData obs column names in a compatible way.
    
    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object
        
    Returns
    -------
    pandas.Index or list
        Column names from adata.obs
    """
    # Modern approach (AnnData 0.8+)
    if hasattr(adata.obs, 'columns'):
        return adata.obs.columns
    
    # Fallback for older versions
    if hasattr(adata, 'obs_keys'):
        warnings.warn("Using deprecated obs_keys() method", CompatibilityWarning)
        return adata.obs_keys()
    
    # Last resort - direct access
    return list(adata.obs.keys())


def safe_var_columns(adata) -> Any:
    """
    Get AnnData var column names in a compatible way.
    
    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object
        
    Returns
    -------
    pandas.Index or list
        Column names from adata.var
    """
    # Modern approach
    if hasattr(adata.var, 'columns'):
        return adata.var.columns
        
    # Fallback for older versions
    if hasattr(adata, 'var_keys'):
        warnings.warn("Using deprecated var_keys() method", CompatibilityWarning)
        return adata.var_keys()
        
    return list(adata.var.keys())


def safe_uns_keys(adata) -> Any:
    """
    Get AnnData uns keys in a compatible way.
    
    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object
        
    Returns
    -------
    dict_keys or list
        Keys from adata.uns
    """
    # Modern approach
    if hasattr(adata.uns, 'keys'):
        return adata.uns.keys()
        
    # Fallback for older versions  
    if hasattr(adata, 'uns_keys'):
        warnings.warn("Using deprecated uns_keys() method", CompatibilityWarning)
        return adata.uns_keys()
        
    return list(adata.uns.keys())


def safe_obsm_keys(adata) -> Any:
    """
    Get AnnData obsm keys in a compatible way.
    
    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object
        
    Returns
    -------
    dict_keys or list
        Keys from adata.obsm
    """
    # Modern approach
    if hasattr(adata.obsm, 'keys'):
        return adata.obsm.keys()
        
    # Fallback for older versions
    if hasattr(adata, 'obsm_keys'):
        warnings.warn("Using deprecated obsm_keys() method", CompatibilityWarning)
        return adata.obsm_keys()
        
    return list(adata.obsm.keys())


def safe_is_view(adata) -> bool:
    """
    Check if AnnData is a view in a compatible way.
    
    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object
        
    Returns
    -------
    bool
        True if adata is a view, False otherwise
    """
    # Modern approach
    if hasattr(adata, 'is_view'):
        return adata.is_view
        
    # Fallback for older versions
    if hasattr(adata, 'isview'):
        warnings.warn("Using deprecated isview property", CompatibilityWarning)
        return adata.isview
        
    # Conservative fallback
    return False


def validate_string_dtype(series, column_name: str = "column") -> bool:
    """
    Validate that a pandas Series contains string data in a pandas 3.0+ compatible way.
    
    Parameters
    ----------
    series : pandas.Series
        Series to check
    column_name : str, default "column"
        Name of column for error messages
        
    Returns
    -------
    bool
        True if series contains string data
        
    Raises
    ------
    ValueError
        If series doesn't contain string data
    """
    # Use pandas API to check for string data (works with both object and string dtypes)
    if hasattr(pd.api.types, 'is_string_dtype'):
        if not pd.api.types.is_string_dtype(series):
            raise ValueError(f"{column_name} must contain string data, got dtype {series.dtype}")
    else:
        # Fallback for older pandas versions
        if series.dtype.kind not in ['O', 'S', 'U']:  # object, byte string, unicode string
            raise ValueError(f"{column_name} must contain string data, got dtype {series.dtype}")
            
    return True


def ensure_copy_on_write_safe(func):
    """
    Decorator to ensure functions are safe under pandas copy-on-write semantics.
    
    This decorator can be used to mark functions that have been verified
    to work correctly with pandas 3.0+ copy-on-write behavior.
    """
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    wrapper._cow_safe = True
    return wrapper


# Initialize compatibility checking on module import
_compatibility_checked = False

def _auto_check_compatibility():
    """Automatically check compatibility when module is imported."""
    global _compatibility_checked
    if not _compatibility_checked:
        try:
            # Silent check - only raise on critical errors
            check_environment_compatibility(verbose=False)
            _compatibility_checked = True
        except CompatibilityError:
            # Re-raise critical errors
            raise
        except Exception:
            # Suppress non-critical errors during import
            pass


# Run automatic compatibility check
_auto_check_compatibility()