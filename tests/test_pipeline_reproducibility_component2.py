"""
Unit tests for Component 2 of the pipeline-reproducibility feature:
`conga.preprocess.calc_tcrdist_nbrs_umap_clusters_cpp`'s new `random_seed`
parameter and its wiring to that function's 6 internal stochastic call
sites:

    - the placeholder-PCA generation, switched from unseeded
      `np.random.randn` to a seeded `np.random.default_rng(random_seed)`
      `Generator`
    - `sc.tl.umap` (multi-component embedding)
    - `sc.tl.umap` (1D embedding)
    - `sc.tl.louvain` (the primary `clustering_method=='louvain'` branch)
    - `sc.tl.leiden` (both occurrences: the primary `clustering_method==
      'leiden'` branch, and the "try leiden first" branch inside the
      `else: try:` fallback)
    - `sc.tl.louvain` (the fallback branch's `except ImportError` call --
      this branch, unlike Component 1's equivalent, is already correctly
      written and calls `sc.tl.louvain`, not `sc.tl.leiden`; nothing is
      "fixed" here beyond adding the `random_state` kwarg)

Two complementary styles are used, following
`tests/test_pipeline_reproducibility_component1.py`'s established
convention:

    - A source-level assertion test: reads `conga/preprocess.py` as text
      and asserts the seeded `default_rng`/`standard_normal` replacement
      and `random_state=random_seed` all appear at the expected call
      sites.
    - A runtime mock/spy test (following
      `tests/test_batch_integration.py`'s `mock.patch(..., wraps=real_fn)`
      convention): calls `calc_tcrdist_nbrs_umap_clusters_cpp` with a
      non-default `random_seed` and confirms each wrapped scanpy call
      receives it as `random_state`, and that the generated `fake_pca`
      array exactly matches
      `np.random.default_rng(random_seed).standard_normal((n, 10))`
      computed directly in the test.

This function requires the compiled `tcrdist_cpp` `find_neighbors`
executable to build the exact-TCRdist neighbor graph before any of the
stochastic calls under test here run. The runtime tests below are skipped
if that binary is not available in this environment. The fixture uses
real human V/J gene names and CDR3 sequences (following
`tests/test_metaconga_match_dispatch.py`'s `Representative_Human_Fixture`
pattern) so the C++ binary's own TCR parsing succeeds; only `va`, `cdr3a`,
`vb`, `cdr3b` columns are required, confirmed from this function's own
source (`adata.obs['va cdr3a vb cdr3b'.split()]`) -- no nucseq columns are
read by this call path.

Requirements: 2.4, 2.5, 2.6, 3.3
"""

import shutil
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import pytest

from conga.preprocess import calc_tcrdist_nbrs_umap_clusters_cpp
from conga import util


REPO_ROOT = Path(__file__).resolve().parent.parent
PREPROCESS_SOURCE_PATH = REPO_ROOT / 'conga' / 'preprocess.py'
RANDOM_SEED = 42
NON_DEFAULT_RANDOM_SEED = 1234

# Real human V/J alleles + CDR3 sequences, same source list used by
# tests/test_metaconga_match_dispatch.py's Representative_Human_Fixture.
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


def _tcrdist_cpp_available() -> bool:
    return util.tcrdist_cpp_available()


def _make_tcr_adata(n_cells: int = 30, tmp_path: Path = None) -> ad.AnnData:
    """Builds a minimal real AnnData with `va`/`cdr3a`/`vb`/`cdr3b` TCR
    columns (real human gene names/CDR3s so the C++ `find_neighbors`
    binary parses them successfully) and `adata.uns['organism'] =
    'human'`. No GEX matrix is needed since
    `calc_tcrdist_nbrs_umap_clusters_cpp` never reads `adata.X`.
    """
    rows = []
    for i in range(n_cells):
        va, ja, cdr3a = _VA_JA_CDR3A[i % len(_VA_JA_CDR3A)]
        vb, jb, cdr3b = _VB_JB_CDR3B[i % len(_VB_JB_CDR3B)]
        rows.append(dict(va=va, cdr3a=cdr3a, vb=vb, cdr3b=cdr3b))
    obs = pd.DataFrame(rows, index=[f'cell{i}' for i in range(n_cells)])
    adata = ad.AnnData(X=np.zeros((n_cells, 1), dtype=np.float32), obs=obs)
    adata.uns['organism'] = 'human'
    return adata


def _run_calc_tcrdist_nbrs(adata, tmp_path, **kwargs):
    return calc_tcrdist_nbrs_umap_clusters_cpp(
        adata, num_nbrs=5, tmpfile_prefix=str(tmp_path / 'tmp_test'),
        **kwargs)


class TestCalcTcrdistNbrsSourceWiresRandomSeed:
    """Source-level assertion test confirming all 6 internal stochastic
    call sites in `calc_tcrdist_nbrs_umap_clusters_cpp` wire
    `random_seed` through, guarding against the wiring silently
    regressing even if the runtime mock tests below are skipped (e.g. no
    compiled C++ binary available).
    """

    def _function_source(self) -> str:
        source = PREPROCESS_SOURCE_PATH.read_text()
        start = source.index('def calc_tcrdist_nbrs_umap_clusters_cpp(')
        end = source.index('\ndef calc_tcrdist_matrix_cpp(', start)
        return source[start:end]

    def test_signature_has_random_seed_default_param(self):
        source = self._function_source()
        signature_region = source[:source.index('):') + 2]
        assert 'random_seed=util.DEFAULT_RANDOM_SEED' in signature_region

    def test_fake_pca_uses_seeded_default_rng(self):
        source = self._function_source()
        assert 'rng = np.random.default_rng(random_seed)' in source
        assert 'fake_pca = rng.standard_normal((adata.shape[0], 10))' in source
        # the old unseeded call must be gone
        assert 'np.random.randn(adata.shape[0], 10)' not in source

    def test_umap_multi_component_call_site_wires_random_seed(self):
        source = self._function_source()
        assert (
            'sc.tl.umap(adata, n_components=n_components_umap, '
            'random_state=random_seed)' in source
        )

    def test_umap_1d_call_site_wires_random_seed(self):
        source = self._function_source()
        assert (
            'sc.tl.umap(adata, n_components=1, random_state=random_seed)'
            in source
        )

    def test_louvain_call_sites_both_wire_random_seed(self):
        """Two louvain call sites exist in this function: the primary
        `clustering_method=='louvain'` branch and the fallback's
        `except ImportError` branch (which, unlike Component 1's
        equivalent branch, correctly calls `sc.tl.louvain`).
        """
        source = self._function_source()
        louvain_call_starts = [
            i for i in range(len(source))
            if source.startswith(
                'sc.tl.louvain(adata, resolution=resolution, '
                'key_added=cluster_key_added,', i)
        ]
        assert len(louvain_call_starts) == 2
        for start in louvain_call_starts:
            call_text = source[start:start + 200]
            assert 'random_state=random_seed' in call_text

    def test_leiden_call_sites_both_wire_random_seed(self):
        """Two leiden call sites exist: the primary `clustering_method==
        'leiden'` branch and the "try leiden first" branch inside the
        `else: try:` fallback.
        """
        source = self._function_source()
        leiden_call_starts = [
            i for i in range(len(source))
            if source.startswith(
                'sc.tl.leiden(adata, resolution=resolution, '
                'key_added=cluster_key_added,', i)
        ]
        assert len(leiden_call_starts) == 2
        for start in leiden_call_starts:
            call_text = source[start:start + 200]
            assert 'random_state=random_seed' in call_text

    def test_fallback_branch_already_correct_not_modified(self):
        """Confirms this function's fallback branch calls `sc.tl.louvain`
        (not `sc.tl.leiden` again, unlike Component 1's equivalent
        branch) -- nothing to "fix" here, only random_state= is added.
        """
        source = self._function_source()
        except_region_start = source.index('except ImportError')
        except_region = source[except_region_start:except_region_start + 300]
        assert 'sc.tl.louvain(' in except_region


@pytest.mark.skipif(
    not util.tcrdist_cpp_available(),
    reason='tcrdist_cpp find_neighbors binary not available in this '
           'environment')
class TestCalcTcrdistNbrsRuntimeSeedWiring:
    """Runtime mock/spy tests confirming a non-default `random_seed`
    actually reaches each wrapped scanpy call, and that `fake_pca` is
    generated from a `Generator` seeded with that `random_seed`. These
    require the real `find_neighbors` C++ binary to build the neighbor
    graph before the stochastic calls under test run, so a real (small)
    TCR fixture is used rather than mocking the C++ step itself.
    """

    def test_fake_pca_matches_seeded_generator(self, tmp_path):
        adata = _make_tcr_adata(tmp_path=tmp_path)
        captured = {}
        real_umap = sc.tl.umap

        def _capture_pca(*args, **kwargs):
            captured['fake_pca'] = args[0].obsm['X_pca'].copy()
            return real_umap(*args, **kwargs)

        with mock.patch('conga.preprocess.sc.tl.umap', side_effect=_capture_pca):
            _run_calc_tcrdist_nbrs(
                adata, tmp_path, make_1d_umaps=False,
                random_seed=NON_DEFAULT_RANDOM_SEED)

        expected = np.random.default_rng(
            NON_DEFAULT_RANDOM_SEED).standard_normal((adata.shape[0], 10))
        assert 'fake_pca' in captured
        assert np.array_equal(captured['fake_pca'], expected)

    def test_umap_calls_receive_random_seed(self, tmp_path):
        adata = _make_tcr_adata(tmp_path=tmp_path)
        real_umap = sc.tl.umap
        with mock.patch(
            'conga.preprocess.sc.tl.umap', wraps=real_umap,
        ) as mock_umap:
            _run_calc_tcrdist_nbrs(
                adata, tmp_path, make_1d_umaps=True,
                random_seed=NON_DEFAULT_RANDOM_SEED)

        assert mock_umap.call_count == 2
        for call in mock_umap.call_args_list:
            _, kwargs = call
            assert kwargs['random_state'] == NON_DEFAULT_RANDOM_SEED

    def test_louvain_receives_random_seed(self, tmp_path):
        adata = _make_tcr_adata(tmp_path=tmp_path)
        real_louvain = sc.tl.louvain
        with mock.patch(
            'conga.preprocess.sc.tl.louvain', wraps=real_louvain,
        ) as mock_louvain:
            _run_calc_tcrdist_nbrs(
                adata, tmp_path, make_1d_umaps=False,
                clustering_method='louvain',
                random_seed=NON_DEFAULT_RANDOM_SEED)

        mock_louvain.assert_called()
        _, kwargs = mock_louvain.call_args
        assert kwargs['random_state'] == NON_DEFAULT_RANDOM_SEED

    def test_leiden_primary_branch_receives_random_seed(self, tmp_path):
        adata = _make_tcr_adata(tmp_path=tmp_path)
        real_leiden = sc.tl.leiden
        with mock.patch(
            'conga.preprocess.sc.tl.leiden', wraps=real_leiden,
        ) as mock_leiden:
            _run_calc_tcrdist_nbrs(
                adata, tmp_path, make_1d_umaps=False,
                clustering_method='leiden',
                random_seed=NON_DEFAULT_RANDOM_SEED)

        mock_leiden.assert_called()
        _, kwargs = mock_leiden.call_args
        assert kwargs['random_state'] == NON_DEFAULT_RANDOM_SEED

    def test_leiden_fallback_branch_receives_random_seed(self, tmp_path):
        """With `clustering_method=None`, the "try both, prefer modern
        leiden first" branch runs, calling `sc.tl.leiden` (leiden is
        importable in this environment, so the primary try succeeds and
        the `except ImportError` louvain fallback is not exercised).
        """
        adata = _make_tcr_adata(tmp_path=tmp_path)
        real_leiden = sc.tl.leiden
        with mock.patch(
            'conga.preprocess.sc.tl.leiden', wraps=real_leiden,
        ) as mock_leiden:
            _run_calc_tcrdist_nbrs(
                adata, tmp_path, make_1d_umaps=False,
                clustering_method=None,
                random_seed=NON_DEFAULT_RANDOM_SEED)

        mock_leiden.assert_called()
        _, kwargs = mock_leiden.call_args
        assert kwargs['random_state'] == NON_DEFAULT_RANDOM_SEED

    def test_default_random_seed_is_util_default_random_seed(self):
        import inspect
        sig = inspect.signature(calc_tcrdist_nbrs_umap_clusters_cpp)
        assert sig.parameters['random_seed'].default == util.DEFAULT_RANDOM_SEED

    def test_random_seed_is_last_parameter(self):
        import inspect
        sig = inspect.signature(calc_tcrdist_nbrs_umap_clusters_cpp)
        assert list(sig.parameters.keys())[-1] == 'random_seed'
