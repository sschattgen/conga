# scikit-learn 1.9.1 SMACOF/MDS Determinism Re-verification (Task 8)

Date: 2026-10-09
Environment: conga-dev -- scikit-learn 1.9.1 (design phase assumed 1.3 per
`pandas3-numpy2-compatibility` steering doc)
Scope: `conga.tcrdist.vectorized.aa_embedding()`, the only place in the
vectorized-tcrdist feature that calls `sklearn.manifold.MDS`.

## Why this needed re-checking

The vectorized-tcrdist design phase analyzed `random_state`/
`normalized_stress` reproducibility assuming scikit-learn 1.3. The
environment that actually solves (`environment.yml`) installs scikit-learn
1.9.1, which changed several MDS defaults between 1.3 and 1.9 (and is
about to change another in 1.10). The question: does `aa_embedding()`'s
pinned-parameter approach still produce deterministic, accurate output
under 1.9.1, or did something drift?

## What changed in sklearn's MDS between the assumed and actual versions

Inspected `sklearn.manifold.MDS.__init__` and `.fit_transform` source
directly in the installed 1.9.1 package. Confirmed defaults that differ
from what `conga`'s `_MDS_KWARGS` pins (comments in
`conga/tcrdist/vectorized.py` already called these out correctly):

| Parameter | sklearn 1.9.1 default | `_MDS_KWARGS` pin |
|---|---|---|
| `init` | `"warn"` (emits FutureWarning, uses `"random"`; becomes `"classical_mds"` in 1.10) | `"classical_mds"` |
| `eps` | `1e-6` | `1e-3` |
| `n_init` | `1` (changed from 4 in an earlier version per code comment) | `1` (explicit) |
| `normalized_stress` | `"auto"` | `False` |
| `metric`/`dissimilarity` | `metric="euclidean"`, `dissimilarity="deprecated"` | `metric="precomputed"`, `metric_mds=True` |

All of CoNGA's pins are explicit, so none of these default drifts affect
`aa_embedding()` -- confirmed by reading `_MDS_KWARGS` and comparing
against `inspect.signature(MDS.__init__)` on the actual installed version.

## How `init='classical_mds'` interacts with `random_state`

Traced `MDS.fit_transform`'s source in sklearn 1.9.1: when
`init='classical_mds'`, a `ClassicalMDS` instance (itself fully
deterministic -- an eigendecomposition, no RNG) computes a fixed starting
configuration, which is then passed as `init=` into the `smacof()` call
alongside `n_init=1`. Per `smacof`'s own contract, supplying a fixed
`init` with `n_init=1` means SMACOF refines that single deterministic
starting point rather than drawing `n_init` random starts -- so
`random_state` has **no effect on the output** in this configuration. This
is expected, not a bug: it's exactly why `init='classical_mds'` was chosen
over the sklearn default (random init) for reproducibility.

## Empirical verification (ran directly against sklearn 1.9.1)

```python
from conga.tcrdist.vectorized import aa_embedding, EncodingConfig, _EMBEDDING_CACHE

_EMBEDDING_CACHE.clear()
emb1 = aa_embedding(EncodingConfig(aa_mds_dim=16, random_seed=42))
_EMBEDDING_CACHE.clear()
emb2 = aa_embedding(EncodingConfig(aa_mds_dim=16, random_seed=42))
_EMBEDDING_CACHE.clear()
emb3 = aa_embedding(EncodingConfig(aa_mds_dim=16, random_seed=123))
```

Results:
- `np.array_equal(emb1, emb2)` -- **True** (bit-identical across repeated
  calls with cache cleared between them, forcing real recomputation each
  time -- confirms the docstring's "Deterministic: identical inputs
  produce identical outputs" claim under the actual installed sklearn
  1.9.1, not just the originally-assumed 1.3).
- `np.array_equal(emb1, emb3)` -- **True** (different `random_state`
  produces the *same* embedding). Confirmed this is expected given the
  `init='classical_mds'` + `n_init=1` mechanics above, not a determinism
  failure. `random_seed` still matters for the project's broader
  reproducibility story (it's threaded through other stochastic call sites
  per `tests/test_pipeline_reproducibility_component*.py`), just not for
  this specific MDS call once a fixed classical-MDS init is supplied.
- Logged stress at `aa_mds_dim=16`: **0.4526**, matching the docstring's
  claimed "stress 0.45... at dim=16" figure under 1.9.1.
- Ran `tests/test_vectorized_b1_1.py` (21 tests covering determinism,
  caching, cache-key fingerprinting, cross-process reproducibility, and
  edge-case dimensions): **21/21 passed** under sklearn 1.9.1.

## Bug found and fixed: `Pandas4Warning` leak in `conga/compatibility.py`

While capturing warnings during this verification, found that
`check_pandas_compatibility()` (unrelated to MDS itself, but discovered
in the same test run) emitted:

```
conga/compatibility.py:164: Pandas4Warning: The 'mode.copy_on_write' option
is deprecated. Copy-on-Write can no longer be disabled (it is always
enabled with pandas >= 3.0), and setting the option has no impact. This
option will be removed in pandas 4.0.
```

Root cause: the code tried to suppress this exact warning with
`warnings.filterwarnings("ignore", category=FutureWarning, ...)`, but
pandas 3.0.6 raises it as `Pandas4Warning` (a `DeprecationWarning`
subclass), not `FutureWarning` -- so the filter never matched and the
warning leaked through on every call to `check_environment_compatibility()`,
i.e. on every `run_conga.py`/`setup_10x_for_conga.py` invocation after the
task 4 CLI wiring added in this session.

**Fixed** in `conga/compatibility.py`: under pandas 3.0+, CoW is
unconditional and the option is a documented no-op, so the code no longer
attempts to *set* `pd.options.mode.copy_on_write` at all for 3.0+ (it still
sets it for pre-3.0 pandas, where the assignment has a real effect). It
just reports `cow_enabled=True` as a fact. Verified with
`-W error::Warning` style capture that `check_pandas_compatibility()` now
produces zero warnings, and re-ran `tests/test_vectorized_b1_1.py`
(21/21 still pass) to confirm no regression.

## Noted but not fixed: sklearn's own `RuntimeWarning: invalid value
encountered in sqrt`

`tests/test_vectorized_b1_1.py::TestAAEmbedding::test_different_dimensions`
triggers a `RuntimeWarning` from inside sklearn's own
`_classical_mds.py:195` (`self.embedding_ = np.sqrt(w) * U`) when probing
`aa_mds_dim=20` (one short of the 21x21 matrix's full rank). This is
sklearn's `ClassicalMDS` taking `sqrt` of a slightly-negative eigenvalue
that should be ~0 but drifts negative due to floating-point error --
expected behavior at near-maximal embedding dimension, not something
`conga`'s actual default (`aa_mds_dim=16`) hits, and the test file's own
comment already documents this ("Skip 21D test as it can produce NaN
values due to numerical issues... may have numerical issues" for 20D/21D).
Not a pandas3/numpy2 or sklearn-1.9.1-specific regression -- the same
floating-point eigenvalue noise would occur under sklearn 1.3 too. No
action taken.

## Conclusion

**Task 8 is VERIFIED COMPLETE.** `aa_embedding()`'s pinned-parameter
approach to `sklearn.manifold.MDS` produces bit-identical, deterministic
output under the actually-installed scikit-learn 1.9.1 (not just the
originally-assumed 1.3), with accuracy (stress 0.45 at dim=16) matching
the documented figure. One real bug was found and fixed in the process
(`Pandas4Warning` leak in `conga/compatibility.py`'s CoW-reporting code,
unrelated to MDS but surfaced by the same warning-capture investigation)
and is now part of task 6's "zero deprecation warnings" verification --
re-run of the full workflow matrix after this fix would be needed to
confirm it doesn't newly appear elsewhere (tracked in task 10).
