#!/usr/bin/env python3
"""
Fix deprecated AnnData method calls for pandas 3.0/NumPy 2.0 compatibility.

This script systematically replaces:
- adata.obs.columns → 'key' in adata.obs or adata.obs.columns
- adata.uns.keys() → 'key' in adata.uns or adata.uns.keys()
- adata.obsm.keys() → 'key' in adata.obsm or adata.obsm.keys()
- adata.is_view → adata.is_view
"""

import re
import glob
import os

def fix_anndata_methods_in_file(filepath):
    """Apply AnnData method modernization to a Python file."""
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Fix patterns systematically
    
    # Pattern 1: 'key' in adata.obs → 'key' in adata.obs
    content = re.sub(
        r"'([^']+)'\s+in\s+adata\.obs_keys\(\)",
        r"'\1' in adata.obs",
        content
    )
    content = re.sub(
        r'"([^"]+)"\s+in\s+adata\.obs_keys\(\)',
        r'"\1" in adata.obs',
        content
    )
    
    # Pattern 2: 'key' in adata.uns → 'key' in adata.uns
    content = re.sub(
        r"'([^']+)'\s+in\s+adata\.uns_keys\(\)",
        r"'\1' in adata.uns",
        content
    )
    content = re.sub(
        r'"([^"]+)"\s+in\s+adata\.uns_keys\(\)',
        r'"\1" in adata.uns',
        content
    )
    
    # Pattern 3: 'key' in adata.obsm → 'key' in adata.obsm
    content = re.sub(
        r"'([^']+)'\s+in\s+adata\.obsm_keys\(\)",
        r"'\1' in adata.obsm",
        content
    )
    content = re.sub(
        r'"([^"]+)"\s+in\s+adata\.obsm_keys\(\)',
        r'"\1" in adata.obsm',
        content
    )
    
    # Pattern 4: adata.obsm.keys() for iteration → adata.obsm.keys()
    content = re.sub(
        r'adata\.obsm_keys\(\)',
        r'adata.obsm.keys()',
        content
    )
    
    # Pattern 5: adata.uns.keys() not in conditional → adata.uns.keys()
    content = re.sub(
        r'adata\.uns_keys\(\)',
        r'adata.uns.keys()',
        content
    )
    
    # Pattern 6: adata.obs.columns not in conditional → adata.obs.columns
    content = re.sub(
        r'adata\.obs_keys\(\)',
        r'adata.obs.columns',
        content
    )
    
    # Pattern 7: adata.is_view → adata.is_view
    content = re.sub(
        r'adata\.isview',
        r'adata.is_view',
        content
    )
    
    # Write back if changed
    if content != original_content:
        print(f"Updated: {filepath}")
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    """Fix AnnData deprecations in all Python files."""
    
    # Target files in conga package
    python_files = []
    for pattern in ['conga/*.py', 'conga/**/*.py', 'scripts/*.py']:
        python_files.extend(glob.glob(pattern, recursive=True))
    
    updated_files = []
    
    for filepath in python_files:
        if os.path.isfile(filepath):
            if fix_anndata_methods_in_file(filepath):
                updated_files.append(filepath)
    
    print(f"\nUpdated {len(updated_files)} files:")
    for filepath in updated_files:
        print(f"  - {filepath}")
    
    if not updated_files:
        print("No files needed updates.")

if __name__ == "__main__":
    main()
