# Development Workflow and Best Practices

## Active Development Environment

**All development, testing, and Python execution for this project uses the
`conga-dev` mamba environment.** Do not invoke a bare `python`, `pip`, or
`pytest` from the ambient shell — always route through the environment
explicitly:

```bash
mamba run -n conga-dev python ...
mamba run -n conga-dev pytest ...
mamba run -n conga-dev pip install ...
```  

## Environment Setup

### Quick Setup (Recommended)

CoNGA now uses `pyproject.toml` for modern Python packaging and mamba for fast environment creation.

```bash
# Using the automated setup script (easiest)
./setup_dev_env.sh

# Or manually with mamba:
mamba env create -f environment.yml
mamba activate conga-dev
pip install -e .
cd tcrdist_cpp && make && cd ..
```

### Manual Setup (if needed)

```bash
# Using mamba (faster than conda)
mamba create -n conga-dev python=3.12
mamba activate conga-dev

# Install from pyproject.toml with all dependencies
pip install -e ".[all]"

# Or install specific feature sets:
pip install -e ".[performance]"  # FAISS for speed
pip install -e ".[batch]"        # Batch integration
pip install -e ".[dev]"          # Development tools

# Minimal installation (core only)
pip install -e .
```

See `INSTALL.md` for complete installation instructions and troubleshooting.

### Compiling C++ Components

The C++ TCRdist implementation significantly improves performance for large datasets.

```bash
cd conga/tcrdist_cpp
make
# Or manually on systems without make:
g++ -O3 -std=c++11 -Wall -I ./include/ -o ./bin/find_neighbors ./src/find_neighbors.cc
g++ -O3 -std=c++11 -Wall -I ./include/ -o ./bin/calc_distributions ./src/calc_distributions.cc
g++ -O3 -std=c++11 -Wall -I ./include/ -o ./bin/find_paired_matches ./src/find_paired_matches.cc
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_tcr_scoring.py

# Run with coverage
pytest --cov=conga tests/

# Run with verbose output
pytest -v tests/
```

## Development Tasks

Primary Objectives (High Priority)
✅ Clean Up Bugs - COMPLETED (Python 3.12 modernization)

✅ Vectorized TCRdist Implementation - COMPLETED

Status: Production implementation completed in conga/tcrdist/vectorized.py
- Refactored from prototype with portable initialization
- Supports deterministic amino acid embedding with classical MDS
- Implements three-way TCR representation selection
- Includes comprehensive accuracy validation framework
Task 2.2: Integrate into preprocessing pipeline
Task 2.3: Add CLI flags (--use_vectorized_tcrdist)
🔴 FAISS Acceleration - GPU + CPU with graceful fallback

Tiered system: faiss-gpu → faiss-cpu → sklearn
New module: 
neighbors.py
🔴 Batch Integration - Complete function from dev branch

Port batch_integration() with 4 methods: Harmony, scVI, Combat, BBKNN
Add hvg_batch_key parameter
MetaCoNGA TAG support for HVGs
🟡 Containerization - Docker + Singularity

🔴 Unit Tests + Reproducibility

Key change: Constant seed (42) by default, not random
--random_seed 42 default for deterministic results
Secondary/Future Objectives
🟢 MuData Support - muon package integration
🟢 New Species Support - Benjamin's reference builder
Implementation Phases
Phase 1 (Weeks 1-2): Vectorized TCRdist + FAISS ← START HERE
Phase 2 (Weeks 3-4): Batch integration + Tests
Phase 3 (Weeks 5-6): Enhancement + Documentation
Success Metrics
10-100x speedup with vectorized TCRdist + FAISS-GPU
Memory reduction >50% on large datasets
Test coverage >70% with deterministic results
All examples produce identical results with seed=42

### Modifying AnnData Objects

**Good practices:**
```python
# Create a copy if you'll be making major changes
adata_copy = adata.copy()

# Update specific fields
adata.obs['new_feature'] = computed_values
adata.uns['analysis_params'] = {'param1': value1}

# Store large results efficiently
adata.obsm['X_embedding'] = embedding_matrix
```

**Avoid:**
```python
# Don't do this (modifies original unexpectedly):
adata2 = adata  # This is just an alias, not a copy
adata2.obs['feature'] = values  # Modifies adata too!
```

### Working with Large Datasets

For datasets with >100k cells:

1. **Use C++ TCRdist** (`tcrdist_cpp/`) instead of Python implementation
2. **Chunk the data** if memory is limited
3. **Use efficient storage:** `h5ad` format for intermediate files
4. **Profile memory usage:**
   ```python
   import tracemalloc
   tracemalloc.start()
   # ... code ...
   current, peak = tracemalloc.get_traced_memory()
   print(f"Peak memory: {peak / 1024 / 1024:.1f} MB")
   ```

### Debugging Tips

```python
# Add temporary debug output
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug(f"Variable state: {my_var}")

# Inspect AnnData structure
print(adata)
print(adata.obs.columns)
print(adata.obs.head())
print(adata.uns.keys())

# Check shapes and dtypes
print(adata.X.shape)
print(adata.obs.dtypes)
```

### Updating Dependencies

When adding a new dependency:

1. Add to `setup.py` with version constraint
2. Document why it's needed
3. Update this file with installation instructions
4. Test with the new dependency
5. Include in environment setup instructions

## Git Workflow

### Commit Messages

- Use present tense: "Add feature" not "Added feature"
- Be descriptive: explain _why_, not just _what_
- Keep first line under 50 characters for log readability
- Example:
  ```
  Add TCR clumping statistical test
  
  Implement background distribution calculation using
  shuffled V/J pairings. Fixes #42.
  ```

### Branch Naming

- Feature: `feature/description` (e.g., `feature/bcr-support`)
- Bug fix: `bugfix/description` (e.g., `bugfix/barcode-mismatch`)
- Experimental: `exp/description` (e.g., `exp/gpu-acceleration`)

## Documentation Standards

### Jupyter Notebooks

- Clean notebooks before committing (restart kernel, clear outputs)
- Use descriptive cell comments
- Include markdown cells explaining analysis steps
- Store example data in compressed format or provide download links
- Don't commit large data files (>10 MB)

### README and Examples

- Update README when adding new features
- Provide complete, runnable examples
- Include expected output or visualizations
- Document any external dependencies (e.g., system tools for SVG conversion)

### Docstring Examples

Include runnable examples in docstrings:

```python
def compute_clonotype_features(adata, organism):
    """
    Compute features for clonotypes in the dataset.
    
    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data with TCR information
    organism : str
        Organism identifier ('human', 'mouse', etc.)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with clonotype features
    
    Examples
    --------
    >>> import conga
    >>> adata = conga.load_example_data()
    >>> features = conga.compute_clonotype_features(adata, 'human')
    >>> features.head()
    """
```

## Performance Profiling

### Quick profiling with timeit

```python
import timeit

# Time a code snippet
t = timeit.timeit('tcrdist_distance(seq1, seq2)', 
                  globals=globals(), number=1000)
print(f"Average time: {t/1000*1e6:.2f} µs")
```

### Detailed profiling with cProfile

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# ... code to profile ...

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Print top 20 functions
```

## Dealing with C++ Integration Issues

### Rebuilding after changes

```bash
# Clean and rebuild
cd conga/tcrdist_cpp
make clean
make

# Verify the executable exists
ls -la bin/
```

### Troubleshooting

- If C++ code won't compile, check compiler version: `g++ --version`
- Python must find the compiled binaries; check path in `convert_svg_to_png.py` and relevant modules
- Run without C++ implementation as fallback (slower but functional)

## Code Review Checklist

Before submitting a pull request:

- [ ] Code follows PEP 8 and project style guide
- [ ] All functions have docstrings
- [ ] New features have unit tests
- [ ] Tests pass locally (`pytest`)
- [ ] No debugging code or print statements left
- [ ] Commit messages are clear and descriptive
- [ ] Documentation is updated if needed
- [ ] No large files added (check with `git log --oneline --diff-filter=A --name-only` for size)

## Useful Resources

- **Scanpy documentation:** https://scanpy.readthedocs.io/
- **AnnData documentation:** https://anndata.readthedocs.io/
- **NumPy documentation:** https://numpy.org/doc/
- **Pandas documentation:** https://pandas.pydata.org/docs/
- **CoNGA preprint:** https://www.biorxiv.org/content/10.1101/2020.06.04.134536v1
- **Published paper:** https://www.nature.com/articles/s41587-021-00989-2
