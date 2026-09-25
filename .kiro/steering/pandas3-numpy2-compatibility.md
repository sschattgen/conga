# Pandas 3 / NumPy 2 Compatibility

## Status: Not done -- expect real breakage

The dev environment (environment.yml) solves to pandas 3.0.6 and numpy 2.5.3
against the conda-forge/bioconda channels. Earlier steering docs described
Python 3.12 modernization as complete; it is not complete until CoNGA's code
is verified against these two major-version bumps specifically. Version
numbers resolving cleanly is not the same as the code running correctly --
pandas 3 and numpy 2 both change runtime behavior, not just APIs that raise
ImportError.

## Why this breaks quietly rather than loudly

Both changes are opt-in-by-default behavior shifts. Old code that used old
APIs will often still import and run, and produce silently wrong results,
or work on toy data and fail only on real datasets with certain dtypes or
shapes.

### pandas 3.0

- Copy-on-write is now unconditional (no longer an opt-in flag). Chained
  assignment patterns that used to mutate a DataFrame now silently do
  nothing. conga/preprocess.py and conga/tcrdist/*.py were written before
  CoW existed and need a pass for this pattern.
- String columns default to a real string dtype, not object. Code that
  checks dtype == object on V/J gene or CDR3 columns needs checking.
- inplace=True semantics tightened on several methods under CoW.

### NumPy 2.0

- np.float_, np.int_, np.bool_ and similar legacy aliases were removed,
  not just deprecated. Any reference is now a hard AttributeError.
- Default integer casting and promotion rules changed (NEP 50). Mixed
  int/float operations that used to upcast quietly can now produce a
  different dtype than before. This is the category most likely to affect
  distance-matrix and vectorized encoding code.
- numpy.core was privatized to numpy._core; direct imports from numpy.core
  break.

## What done requires

Do not mark Python 3.12 modernization complete based on pyproject.toml
version floors alone. Completion requires:

- Run the existing test suite and the example pipelines against the actual
  pandas 3 / numpy 2 environment, not the versions the code was originally
  written for.
- Grep the codebase for np.float_, np.int_, np.bool_, np.object_, and
  numpy.core usage and fix every hit.
- Audit conga/preprocess.py and conga/tcrdist/*.py for chained assignment
  and in-place mutation patterns that assume pre-CoW pandas.
- Check every place a distance matrix or encoded vector is built for dtype
  promotion changes under NEP 50; confirm float32/float64 casts are
  explicit rather than implicit.
- Re-run this check whenever scanpy, anndata, or scikit-learn versions
  move.

## Relevance to active specs

The vectorized-tcrdist spec design phase assumed scikit-learn 1.3 for its
MDS/SMACOF determinism analysis. The environment that actually solves
gives scikit-learn 1.9.1. Re-verify random_state / normalized_stress
reproducibility behavior against 1.9, not 1.3, before finalizing that
design.
