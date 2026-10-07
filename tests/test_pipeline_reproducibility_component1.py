"""
Unit tests for Component 1 of the pipeline-reproducibility feature:
`conga.preprocess.cluster_and_tsne_and_umap`'s new `random_seed` parameter
and its wiring to that function's 6 internal stochastic call sites:

    - `sc.tl.pca` (X_pca_gex computation)
    - `sc.pp.neighbors`
    - `sc.tl.umap` (multi-component embedding)
    - `sc.tl.umap` (1D embedding)
    - `sc.tl.louvain`
    - `sc.tl.leiden` (both occurrences: the primary 'leiden' branch and the
      "try leiden first" branch inside the `clustering_method=None`
      fallback's `except ImportError` handler)

Two complementary styles are used, following this project's existing
conventions:

    - A source-level assertion test (following
      `tests/test_run_conga_cli.py`'s
      `test_call_site_source_wires_batch_integration_not_filter_and_scale`
      pattern): reads `conga/preprocess.py` as text and asserts
      `random_state=random_seed` appears at each of the 6 call sites.
    - A runtime mock/spy test (following `tests/test_batch_integration.py`'s
      `mock.patch(..., wraps=real_fn)` convention): calls
      `cluster_and_tsne_and_umap` with a non-default `random_seed` and
      confirms each wrapped scanpy call actually receives it as
      `random_state`.

Per design.md's Component 1 note, the pre-existing `except ImportError`
fallback bug (that branch calls `sc.tl.leiden` a second time instead of
`sc.tl.louvain`, despite printing a "ran leiden clustering" message) is
intentionally left untouched by this feature and is not covered or
asserted against here beyond confirming the (unchanged) seeding behavior
of whichever call is actually present.

Requirements: 1.1, 1.4, 2.1, 2.2, 2.3, 3.1, 3.2
"""

import inspect
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy.sparse as sp
import pytest

from conga.preprocess import cluster_and_tsne_and_umap
from conga import util


REPO_ROOT = Path(__file__).resolve().parent.parent
PREPROCESS_SOURCE_PATH = REPO_ROOT / 'conga' / 'preprocess.py'
RANDOM_SEED = 42
NON_DEFAULT_RANDOM_SEED = 1234


def _make_gex_counts(n_cells: int, n_genes: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Build a small, dense-then-sparsified integer counts matrix with no
    all-zero rows or columns, matching the fixture-construction convention
    used in `tests/test_batch_integration.py`/`tests/test_counts_layer.py`.
    """
    gene_means = rng.uniform(low=0.5, high=40.0, size=n_genes)
    counts = rng.poisson(lam=gene_means, size=(n_cells, n_genes)).astype(np.float64)
    noisy_gene_idx = rng.choice(
        n_genes, size=max(1, n_genes // 5), replace=False)
    burst = rng.choice(
        [1.0, 8.0], size=(n_cells, len(noisy_gene_idx)), p=[0.85, 0.15])
    counts[:, noisy_gene_idx] *= burst
    counts += 1.0
    return counts


def _make_adata_with_pca_gex(n_cells: int = 60, n_genes: int = 40,
                              n_pcs: int = 5,
                              random_seed: int = RANDOM_SEED) -> ad.AnnData:
    """Build a minimal AnnData with `X_pca_gex` already populated, so
    `cluster_and_tsne_and_umap(..., skip_tcr=True)` skips straight to the
    neighbors/UMAP/clustering call sites under test without needing a real
    `sc.tl.pca` run first (mirrors the `recompute_pca_gex=True` pattern
    used in `tests/test_batch_integration.py` to force `sc.tl.pca` to run
    when a spy needs to observe it).
    """
    rng = np.random.default_rng(random_seed)
    counts = _make_gex_counts(n_cells, n_genes, rng)
    obs = pd.DataFrame(index=[f'cell{i}' for i in range(n_cells)])
    var = pd.DataFrame(index=[f'GENE{i}' for i in range(n_genes)])
    adata = ad.AnnData(X=sp.csr_matrix(counts), obs=obs, var=var)
    adata.obsm['X_pca_gex'] = rng.standard_normal((n_cells, n_pcs))
    return adata


class TestClusterAndTsneAndUmapSourceWiresRandomSeed:
    """Source-level assertion test confirming the 6 internal stochastic
    call sites in `cluster_and_tsne_and_umap` all pass
    `random_state=random_seed`, guarding against the wiring silently
    regressing even if the runtime mock tests below are weakened or
    skipped.
    """

    def _function_source(self) -> str:
        source = PREPROCESS_SOURCE_PATH.read_text()
        start = source.index('def cluster_and_tsne_and_umap(')
        # the next top-level `def ` after the function start marks the
        # end of this function's body
        end = source.index('\ndef filter_and_scale(', start)
        return source[start:end]

    def test_signature_has_random_seed_default_param(self):
        source = self._function_source()
        signature_region = source[:source.index('):') + 2]
        assert 'random_seed=util.DEFAULT_RANDOM_SEED' in signature_region

    def test_pca_call_site_wires_random_seed(self):
        source = self._function_source()
        assert (
            "sc.tl.pca(adata, svd_solver='arpack', n_comps=n_gex_pcs,\n"
            "                  random_state=random_seed)" in source
        )

    def test_neighbors_call_site_wires_random_seed(self):
        source = self._function_source()
        assert (
            "sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs,\n"
            "                         random_state=random_seed)" in source
        )

    def test_umap_multi_component_call_site_wires_random_seed(self):
        source = self._function_source()
        assert (
            "sc.tl.umap(adata, min_dist=umap_min_dist, spread=umap_spread,\n"
            "                   random_state=random_seed)" in source
        )

    def test_umap_1d_call_site_wires_random_seed(self):
        source = self._function_source()
        assert (
            'sc.tl.umap(adata, n_components=1, random_state=random_seed)'
            in source
        )

    def test_louvain_call_site_wires_random_seed(self):
        source = self._function_source()
        assert source.count(
            'sc.tl.louvain(adata, resolution=resolution, '
            'key_added=cluster_key_added,\n'
            '                          random_state=random_seed)'
        ) == 1

    def test_leiden_call_sites_both_wire_random_seed(self):
        """Both leiden occurrences -- the primary 'leiden' branch and the
        "try leiden first" branch inside the fallback -- must each pass
        `random_state=random_seed`. The fallback's `except ImportError`
        handler also calls `sc.tl.leiden` (the pre-existing, intentionally
        untouched bug), so 3 occurrences total are expected. Call sites
        are indented differently depending on nesting depth (the
        'leiden' branch vs. the two calls nested inside the `else: try:`
        block), so each `sc.tl.leiden(` call is located independently and
        the `random_state=random_seed` keyword is confirmed to appear
        within its own argument list rather than relying on one exact
        whitespace-sensitive string.
        """
        source = self._function_source()
        leiden_call_starts = [
            i for i in range(len(source))
            if source.startswith('sc.tl.leiden(adata, resolution=resolution, '
                                  'key_added=cluster_key_added,', i)
        ]
        assert len(leiden_call_starts) == 3
        for start in leiden_call_starts:
            call_text = source[start:start + 200]
            assert 'random_state=random_seed' in call_text

    def test_fallback_bug_left_untouched(self):
        """Confirms this feature did not fix the pre-existing
        leiden/louvain fallback mislabeling bug: the `except ImportError`
        branch must still call `sc.tl.leiden`, not `sc.tl.louvain`.
        """
        source = self._function_source()
        except_region_start = source.index('except ImportError')
        except_region = source[except_region_start:except_region_start + 300]
        assert 'sc.tl.leiden(' in except_region
        assert 'sc.tl.louvain(' not in except_region


class TestClusterAndTsneAndUmapRuntimeSeedWiring:
    """Runtime mock/spy tests confirming a non-default `random_seed`
    actually reaches each wrapped scanpy call, following
    `tests/test_batch_integration.py`'s `mock.patch(..., wraps=real_fn)`
    convention.
    """

    def test_pca_receives_random_seed_when_recomputed(self):
        adata = _make_adata_with_pca_gex()
        real_pca = sc.tl.pca
        with mock.patch(
            'conga.preprocess.sc.tl.pca', wraps=real_pca,
        ) as mock_pca:
            cluster_and_tsne_and_umap(
                adata, recompute_pca_gex=True, skip_tcr=True,
                n_gex_pcs=4, random_seed=NON_DEFAULT_RANDOM_SEED)

        mock_pca.assert_called()
        _, kwargs = mock_pca.call_args
        assert kwargs['random_state'] == NON_DEFAULT_RANDOM_SEED

    def test_neighbors_receives_random_seed(self):
        adata = _make_adata_with_pca_gex()
        real_neighbors = sc.pp.neighbors
        with mock.patch(
            'conga.preprocess.sc.pp.neighbors', wraps=real_neighbors,
        ) as mock_neighbors:
            cluster_and_tsne_and_umap(
                adata, skip_tcr=True, random_seed=NON_DEFAULT_RANDOM_SEED)

        mock_neighbors.assert_called()
        _, kwargs = mock_neighbors.call_args
        assert kwargs['random_state'] == NON_DEFAULT_RANDOM_SEED

    def test_umap_calls_receive_random_seed(self):
        """Covers both the multi-component and 1D `sc.tl.umap` call
        sites: with `make_1d_umaps=True` (the default), `sc.tl.umap` is
        called twice per tag iteration.
        """
        adata = _make_adata_with_pca_gex()
        real_umap = sc.tl.umap
        with mock.patch(
            'conga.preprocess.sc.tl.umap', wraps=real_umap,
        ) as mock_umap:
            cluster_and_tsne_and_umap(
                adata, skip_tcr=True, make_1d_umaps=True,
                random_seed=NON_DEFAULT_RANDOM_SEED)

        assert mock_umap.call_count == 2
        for call in mock_umap.call_args_list:
            _, kwargs = call
            assert kwargs['random_state'] == NON_DEFAULT_RANDOM_SEED

    def test_louvain_receives_random_seed(self):
        adata = _make_adata_with_pca_gex()
        real_louvain = sc.tl.louvain
        with mock.patch(
            'conga.preprocess.sc.tl.louvain', wraps=real_louvain,
        ) as mock_louvain:
            cluster_and_tsne_and_umap(
                adata, skip_tcr=True, clustering_method='louvain',
                random_seed=NON_DEFAULT_RANDOM_SEED)

        mock_louvain.assert_called()
        _, kwargs = mock_louvain.call_args
        assert kwargs['random_state'] == NON_DEFAULT_RANDOM_SEED

    def test_leiden_primary_branch_receives_random_seed(self):
        adata = _make_adata_with_pca_gex()
        real_leiden = sc.tl.leiden
        with mock.patch(
            'conga.preprocess.sc.tl.leiden', wraps=real_leiden,
        ) as mock_leiden:
            cluster_and_tsne_and_umap(
                adata, skip_tcr=True, clustering_method='leiden',
                random_seed=NON_DEFAULT_RANDOM_SEED)

        mock_leiden.assert_called()
        _, kwargs = mock_leiden.call_args
        assert kwargs['random_state'] == NON_DEFAULT_RANDOM_SEED

    def test_leiden_fallback_branch_receives_random_seed(self):
        """With `clustering_method=None`, the "try both, prefer modern
        leiden first" branch runs, calling `sc.tl.leiden` (the
        pre-existing bug means this is true whether or not the
        `except ImportError` path is actually taken, since leiden is
        importable in this environment and the primary try succeeds).
        """
        adata = _make_adata_with_pca_gex()
        real_leiden = sc.tl.leiden
        with mock.patch(
            'conga.preprocess.sc.tl.leiden', wraps=real_leiden,
        ) as mock_leiden:
            cluster_and_tsne_and_umap(
                adata, skip_tcr=True, clustering_method=None,
                random_seed=NON_DEFAULT_RANDOM_SEED)

        mock_leiden.assert_called()
        _, kwargs = mock_leiden.call_args
        assert kwargs['random_state'] == NON_DEFAULT_RANDOM_SEED

    def test_default_random_seed_is_util_default_random_seed(self):
        """Confirms the parameter default matches
        `util.DEFAULT_RANDOM_SEED` (Requirement 1.4), via introspection
        rather than a call-site string match.
        """
        sig = inspect.signature(cluster_and_tsne_and_umap)
        assert sig.parameters['random_seed'].default == util.DEFAULT_RANDOM_SEED

    def test_random_seed_is_last_parameter(self):
        """Requirement 1.4 / design.md Component 1: `random_seed` must be
        the last parameter, so every existing keyword-only caller is
        unaffected.
        """
        sig = inspect.signature(cluster_and_tsne_and_umap)
        assert list(sig.parameters.keys())[-1] == 'random_seed'
