"""
Unit tests for Component 5 of the pipeline-reproducibility feature:
the two `KernelPCA(kernel='precomputed', ...)` constructor call sites,
one in `conga.preprocess.make_tcrdist_kernel_pcs_file_from_clones_file`
and one in
`conga.preprocess.make_tcrdist_kernel_pcs_file_from_clones_file_V2`,
and each function's new `random_seed` parameter:

    pca = KernelPCA(kernel='precomputed', n_components=n_components,
                     random_state=random_seed)

Two complementary styles are used, following
`tests/test_pipeline_reproducibility_component3.py`/`component4.py`'s
established convention:

    - A source-level assertion test: reads `conga/preprocess.py` as text
      and asserts `random_state=random_seed` appears at both `KernelPCA`
      constructor call sites, and that `random_seed=util.DEFAULT_RANDOM_SEED`
      is the last parameter in both function signatures.
    - A runtime mock test confirming a non-default `random_seed` reaches
      the `KernelPCA` constructor for both functions. `KernelPCA` is a
      class (not a function), so `mock.patch(..., wraps=KernelPCA)` is
      used rather than `mock.patch(..., wraps=real_fn)` -- empirically
      confirmed (see design notes below) to behave the same way for a
      class as for a plain function: `wraps` on a `MagicMock` forwards
      the call to the wrapped callable and returns its actual return
      value (a real `KernelPCA` instance), so `.fit_transform(...)`
      downstream of the mocked constructor still works against the real
      object.

Both functions compute a real TCRdist distance matrix before reaching
the `KernelPCA` call. Rather than mocking the distance computation, both
tests exercise the real distance-matrix code path on a tiny 5-clonotype
fixture of real human TCR V/J/CDR3 values (the same
`_VA_JA_CDR3A`/`_VB_JB_CDR3B` fixture values used by
`tests/test_pipeline_reproducibility_component3.py`), which is fast
(confirmed empirically: ~0.1s for the pure-Python `TcrDistCalculator`
path on 5 clonotypes) and does not require mocking any internals of the
seeding logic under test:

    - `make_tcrdist_kernel_pcs_file_from_clones_file` accepts a `tcrs=`
      list directly (bypassing the need for a real `clones_file` on
      disk) and `return_pcs=True` (bypassing the need to write an
      output file), so it is called exactly as a real caller would,
      exercising whichever TCRdist backend (C++ or Python) is available
      in the environment.
    - `make_tcrdist_kernel_pcs_file_from_clones_file_V2` has no such
      bypass -- it always computes TCRdist from `df` -- and has no
      `force_tcrdist_cpp` parameter to select the backend. In this
      `conga-dev` environment the compiled C++ TCRdist binaries are
      available, but `_V2`'s C++ branch has a pre-existing, unrelated
      bug (it references an undefined `outfile` local variable, raising
      `NameError`, since `_V2`'s signature never defines `outfile`) that
      is independent of this feature's seeding change and out of scope
      to fix here. To reach the `KernelPCA` call site without touching
      that unrelated bug, `conga.preprocess.tcrdist_cpp_available` is
      patched to return `False` for `_V2`'s tests only, forcing the
      (working) pure-Python `TcrDistCalculator` path. This is flagged
      here per this task's instructions: it is a mock of an unrelated
      dependency selection, not of anything in the seed-wiring logic
      itself.

Requirements: 1.3
"""

import inspect
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA

from conga.preprocess import (
    make_tcrdist_kernel_pcs_file_from_clones_file,
    make_tcrdist_kernel_pcs_file_from_clones_file_V2,
)
from conga import util


REPO_ROOT = Path(__file__).resolve().parent.parent
PREPROCESS_SOURCE_PATH = REPO_ROOT / 'conga' / 'preprocess.py'
RANDOM_SEED = 42
NON_DEFAULT_RANDOM_SEED = 1234

# Real human V/J alleles + CDR3 sequences, same source list used by
# tests/test_pipeline_reproducibility_component3.py.
_VA_JA_CDR3A = [
    ('TRAV1-1*01', 'TRAJ1*01', 'CALIPGGQKLLF'),
    ('TRAV1-2*01', 'TRAJ10*01', 'CAYRGLGVV'),
    ('TRAV10*01', 'TRAJ11*01', 'CAASKGGSQGNLIF'),
    ('TRAV11*01', 'TRAJ12*01', 'CAVGATGNQF'),
    ('TRAV12-1*01', 'TRAJ13*01', 'CAVTIGFGNVLHQ'),
]
_VB_JB_CDR3B = [
    ('TRBV1*01', 'TRBJ1-1*01', 'CAAGETSGVSYNEQF'),
    ('TRBV10-1*01', 'TRBJ1-2*01', 'CASRPTITVPYSNQPQHF'),
    ('TRBV10-2*01', 'TRBJ1-3*01', 'CASSLVVWDRGGNQPQHF'),
    ('TRBV10-3*01', 'TRBJ1-4*01', 'CASSQDLLSWDEQF'),
    ('TRBV11-1*01', 'TRBJ1-5*01', 'CASSLGNEQF'),
]


def _make_tcrs_list():
    """Build a tiny list of 5 real human alpha/beta TCR tuples in the
    `((va, ja, cdr3a), (vb, jb, cdr3b))` format
    `make_tcrdist_kernel_pcs_file_from_clones_file` expects for its
    `tcrs=` parameter.
    """
    return [
        ((va, ja, cdr3a), (vb, jb, cdr3b))
        for (va, ja, cdr3a), (vb, jb, cdr3b) in zip(_VA_JA_CDR3A, _VB_JB_CDR3B)
    ]


def _make_clones_df():
    """Build a tiny clones DataFrame in the `va`/`ja`/`cdr3a`/`vb`/`jb`/
    `cdr3b` column format `make_tcrdist_kernel_pcs_file_from_clones_file_V2`
    expects for its `df` parameter.
    """
    rows = []
    for i, ((va, ja, cdr3a), (vb, jb, cdr3b)) in enumerate(
            zip(_VA_JA_CDR3A, _VB_JB_CDR3B)):
        rows.append(dict(
            clone_id=f'clone{i}', va=va, ja=ja, cdr3a=cdr3a,
            vb=vb, jb=jb, cdr3b=cdr3b,
        ))
    return pd.DataFrame(rows)


class TestKernelPcsFileV1SourceWiresRandomSeed:
    """Source-level assertion test for
    `make_tcrdist_kernel_pcs_file_from_clones_file`'s `KernelPCA` call
    site and signature.
    """

    def _function_source(self) -> str:
        source = PREPROCESS_SOURCE_PATH.read_text()
        start = source.index(
            'def make_tcrdist_kernel_pcs_file_from_clones_file(')
        # the next top-level `def ` after the function start marks the
        # end of this function's body
        end = source.index(
            '\ndef make_tcrdist_kernel_pcs_file_from_clones_file_V2(', start)
        return source[start:end]

    def test_signature_has_random_seed_as_last_default_param(self):
        source = self._function_source()
        signature_region = source[:source.index('):') + 2]
        assert 'random_seed=util.DEFAULT_RANDOM_SEED' in signature_region

    def test_default_random_seed_is_util_default_random_seed(self):
        sig = inspect.signature(make_tcrdist_kernel_pcs_file_from_clones_file)
        assert sig.parameters['random_seed'].default == util.DEFAULT_RANDOM_SEED

    def test_random_seed_is_last_parameter(self):
        sig = inspect.signature(make_tcrdist_kernel_pcs_file_from_clones_file)
        assert list(sig.parameters.keys())[-1] == 'random_seed'

    def test_kernel_pca_call_site_wires_random_seed(self):
        source = self._function_source()
        assert (
            "pca = KernelPCA(kernel='precomputed', n_components=n_components,\n"
            "                     random_state=random_seed)" in source
        )


class TestKernelPcsFileV2SourceWiresRandomSeed:
    """Source-level assertion test for
    `make_tcrdist_kernel_pcs_file_from_clones_file_V2`'s `KernelPCA` call
    site and signature.
    """

    def _function_source(self) -> str:
        source = PREPROCESS_SOURCE_PATH.read_text()
        start = source.index(
            'def make_tcrdist_kernel_pcs_file_from_clones_file_V2(')
        # the next top-level `def ` after the function start marks the
        # end of this function's body
        end = source.index('\ndef subset_to_CD4_or_CD8_clusters(', start)
        return source[start:end]

    def test_signature_has_random_seed_as_last_default_param(self):
        source = self._function_source()
        signature_region = source[:source.index('):') + 2]
        assert 'random_seed=util.DEFAULT_RANDOM_SEED' in signature_region

    def test_default_random_seed_is_util_default_random_seed(self):
        sig = inspect.signature(
            make_tcrdist_kernel_pcs_file_from_clones_file_V2)
        assert sig.parameters['random_seed'].default == util.DEFAULT_RANDOM_SEED

    def test_random_seed_is_last_parameter(self):
        sig = inspect.signature(
            make_tcrdist_kernel_pcs_file_from_clones_file_V2)
        assert list(sig.parameters.keys())[-1] == 'random_seed'

    def test_kernel_pca_call_site_wires_random_seed(self):
        source = self._function_source()
        assert (
            "pca = KernelPCA(kernel='precomputed', n_components=n_components,\n"
            "                     random_state=random_seed)" in source
        )


class TestKernelPcsFileV1RuntimeSeedWiring:
    """Runtime mock test confirming a non-default `random_seed` reaches
    the `KernelPCA` constructor inside
    `make_tcrdist_kernel_pcs_file_from_clones_file`.
    """

    def test_kernel_pca_receives_random_seed(self):
        tcrs = _make_tcrs_list()
        with mock.patch(
            'conga.preprocess.KernelPCA', wraps=KernelPCA,
        ) as mock_kernel_pca:
            xy = make_tcrdist_kernel_pcs_file_from_clones_file(
                clones_file='unused.tsv', organism='human', tcrs=tcrs,
                n_components_in=3, return_pcs=True,
                random_seed=NON_DEFAULT_RANDOM_SEED,
            )

        mock_kernel_pca.assert_called_once()
        _, kwargs = mock_kernel_pca.call_args
        assert kwargs['random_state'] == NON_DEFAULT_RANDOM_SEED
        # `wraps` forwards to the real class, so the real KernelPCA
        # instance's fit_transform output is still returned untouched.
        assert xy.shape == (len(tcrs), 3)

    def test_same_seed_produces_identical_pcs(self):
        tcrs = _make_tcrs_list()
        xy1 = make_tcrdist_kernel_pcs_file_from_clones_file(
            clones_file='unused.tsv', organism='human', tcrs=tcrs,
            n_components_in=3, return_pcs=True,
            random_seed=NON_DEFAULT_RANDOM_SEED,
        )
        xy2 = make_tcrdist_kernel_pcs_file_from_clones_file(
            clones_file='unused.tsv', organism='human', tcrs=tcrs,
            n_components_in=3, return_pcs=True,
            random_seed=NON_DEFAULT_RANDOM_SEED,
        )
        assert np.array_equal(xy1, xy2)


class TestKernelPcsFileV2RuntimeSeedWiring:
    """Runtime mock test confirming a non-default `random_seed` reaches
    the `KernelPCA` constructor inside
    `make_tcrdist_kernel_pcs_file_from_clones_file_V2`.

    `conga.preprocess.tcrdist_cpp_available` is patched to `False` so
    the test exercises the (working) pure-Python TCRdist path instead
    of `_V2`'s pre-existing, unrelated `NameError` bug in its C++
    branch (see module docstring).
    """

    def test_kernel_pca_receives_random_seed(self):
        df = _make_clones_df()
        with mock.patch(
            'conga.preprocess.tcrdist_cpp_available', return_value=False,
        ), mock.patch(
            'conga.preprocess.KernelPCA', wraps=KernelPCA,
        ) as mock_kernel_pca:
            xy = make_tcrdist_kernel_pcs_file_from_clones_file_V2(
                df, organism='human', n_components_in=3,
                random_seed=NON_DEFAULT_RANDOM_SEED,
            )

        mock_kernel_pca.assert_called_once()
        _, kwargs = mock_kernel_pca.call_args
        assert kwargs['random_state'] == NON_DEFAULT_RANDOM_SEED
        assert xy.shape == (len(df), 3)

    def test_same_seed_produces_identical_pcs(self):
        df = _make_clones_df()
        with mock.patch(
                'conga.preprocess.tcrdist_cpp_available', return_value=False):
            xy1 = make_tcrdist_kernel_pcs_file_from_clones_file_V2(
                df, organism='human', n_components_in=3,
                random_seed=NON_DEFAULT_RANDOM_SEED,
            )
            xy2 = make_tcrdist_kernel_pcs_file_from_clones_file_V2(
                df, organism='human', n_components_in=3,
                random_seed=NON_DEFAULT_RANDOM_SEED,
            )
        assert np.array_equal(xy1, xy2)
