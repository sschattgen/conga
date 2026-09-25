# Python Standards and Conventions

## General Python Standards

### Code Style

- **Follow PEP 8** with some pragmatism for scientific code
- **Line length:** 120 characters (soft limit, can exceed for long scientific names)
- **Indentation:** 4 spaces
- **Naming conventions:**
  - Functions/variables: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`
  - Private members: `_leading_underscore`

### Type Hints

- Use type hints for function signatures where practical
- Not required for every parameter in scientific computing context, but recommended for public APIs
- Example:
  ```python
  def compute_tcrdist(seq1: str, seq2: str) -> float:
      """Compute TCRdist distance between two sequences."""
      pass
  ```

### Documentation

- Use docstrings for all functions and classes
- Follow NumPy docstring style for scientific functions
- Include parameter descriptions, return types, and example usage for public APIs
- Example:
  ```python
  def analyze_clonotype(adata, organism):
      """
      Analyze clonotype structure in single-cell data.
      
      Parameters
      ----------
      adata : anndata.AnnData
          Annotated data matrix with TCR information
      organism : str
          Organism type ('human', 'mouse', 'rhesus', 'human_gd', etc.)
      
      Returns
      -------
      dict
          Dictionary with analysis results
      """
  ```

### Imports

- Group imports: standard library, third-party, local imports (separated by blank lines)
- Use absolute imports
- Avoid `from module import *` except in special cases
- Use `import numpy as np`, `import pandas as pd`, `import scanpy as sc`, etc. (standard scientific aliases)

## Module Organization

### Data Flow

- Prefer functional approaches for data transformations
- Use AnnData objects as the primary data container for single-cell data
- Functions should be pure where possible (no hidden side effects)
- Modify AnnData in-place sparingly; prefer returning modified copies or updating specific fields

### Error Handling

- Use specific exception types, not bare `except:` statements
- Include informative error messages with context
- Validate input parameters early in functions

### Performance Considerations

- Use NumPy operations for vectorization
- Leverage C++ implementations (in `tcrdist_cpp/`) for computationally intensive tasks
- Profile before optimizing
- Comment performance-critical sections

## Testing

- Write tests for new functionality
- Use pytest as the test framework
- Test both successful cases and error conditions
- Include fixtures for common test data (e.g., mock AnnData objects)

## Dependencies

### Python Version

**Required:** Python 3.12+
- Uses modern Python features
- All dependencies validated for 3.12 compatibility
- Better performance and security

### Current Major Dependencies

```python
python>=3.12                   # Required Python version
scanpy>=1.9.0                  # Single-cell analysis (3.12 compatible)
anndata>=0.9.0                 # Data structure
numpy>=1.23.0                  # Numerical computing
scipy>=1.9.0                   # Scientific computing
scikit-learn>=1.1.0            # Machine learning
pandas>=1.5.0                  # Data manipulation
seaborn>=0.12.0                # Visualization
statsmodels>=0.13.0            # Statistical models
python-igraph>=0.10.0          # Graph algorithms
louvain>=0.8.0                 # Community detection
joblib>=1.3.0                  # Model serialization
```

### Optional Dependencies (Phase 2+)

```python
faiss-cpu>=1.7.4               # Fast neighbor search for GEX (or faiss-gpu)
bbknn>=1.5.1                   # Batch-balanced KNN
scvi-tools>=0.17.0             # Single-cell VAE inference (experimental)
```

- Pin versions in `setup.py` or `requirements.txt`
- Document any optional dependencies
- Avoid breaking changes to the public API
- Test all dependencies with Python 3.12

## Logging

- Use Python's `logging` module for informative messages
- Avoid excessive print statements in library code
- Configure logging level appropriately for scripts vs. libraries

## Example: Good Structure

```python
"""Module for TCR analysis utilities."""

import logging
from typing import Optional, Dict

import numpy as np
import pandas as pd
import scanpy as sc

logger = logging.getLogger(__name__)


def compute_tcr_distances(
    sequences: np.ndarray, 
    metric: str = 'tcrdist'
) -> np.ndarray:
    """
    Compute pairwise distances between TCR sequences.
    
    Parameters
    ----------
    sequences : np.ndarray
        Array of TCR sequences
    metric : str
        Distance metric to use ('tcrdist' or 'hamming')
    
    Returns
    -------
    np.ndarray
        Pairwise distance matrix
    """
    if metric not in ('tcrdist', 'hamming'):
        raise ValueError(f"Unknown metric: {metric}")
    
    logger.info(f"Computing {metric} distances for {len(sequences)} sequences")
    # implementation
    return distances


class TCRAnalyzer:
    """Analyzer for T cell receptor properties."""
    
    def __init__(self, organism: str = 'human'):
        """Initialize TCR analyzer."""
        self.organism = organism
        self._validate_organism()
    
    def _validate_organism(self) -> None:
        """Validate organism parameter."""
        valid = {'human', 'mouse', 'rhesus', 'human_gd', 'mouse_gd'}
        if self.organism not in valid:
            raise ValueError(f"Organism must be one of {valid}")
```
