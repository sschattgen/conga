"""
Unit tests for Component 3 of the pipeline-reproducibility feature:
`conga.preprocess.reduce_to_single_cell_per_clone`'s new `random_seed`
parameter and its wiring to the function's single internal stochastic call
site, the representative-cell `sc.tl.pca` call:

    sc.tl.pca(adata, svd_solver='arpack',
              n_comps=min(adata.shape[0]-1, n_pcs),
              random_state=random_seed)

Two complementary styles are used, following
`tests/test_pipeline_reproducibility_component1.py`/`component2.py`'s
established convention:

    - A source-level assertion test: reads `conga/preprocess.py` as text
      and asserts `random_state=random_seed` appears at this call site,
      and that the new `random_seed` parameter is the last parameter in
      the function signature with the correct default.
    - A runtime mock/spy test (following `tests/test_batch_integration.py`'s
      `mock.patch(..., wraps=real_fn)` convention): calls
      `reduce_to_single_cell_per_clone` with a non-default `random_seed`
      and confirms the wrapped `sc.tl.pca` call receives it as
      `random_state`.

Requirements: 1.2
"""

import inspect
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy.sparse as sp

from conga.preprocess import reduce_to_single_cell_per_clone
from conga import util


REPO_ROOT = Path(__file__).resolve().parent.parent
PREPROCESS_SOURCE_PATH = REPO_ROOT / 'conga' / 'preprocess.py'
RANDOM_SEED = 42
NON_DEFAULT_RANDOM_SEED = 1234

# Real human V/J alleles + CDR3 sequences, same source list used by
# tests/test_pipeline_reproducibility_component2.py / test_metaconga_match_dispatch.py's
# Representative_Human_Fixture.
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


def _make_gex_counts(n_cells: int, n_genes: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Build a small, dense integer counts matrix with no all-zero rows
    or columns, matching the fixture-construction convention used in
    `tests/test_pipeline_reproducibility_component1.py`.
    """
    gene_means = rng.uniform(low=0.5, high=40.0, size=n_genes)
    counts = rng.poisson(lam=gene_means, size=(n_cells, n_genes)).astype(np.float64)
    counts += 1.0
    return counts


def _make_clone_adata(n_cells: int = 20, n_genes: int = 30,
                       n_clones: int = 6,
                       random_seed: int = RANDOM_SEED) -> ad.AnnData:
    """Build a minimal AnnData with TCR columns (so each distinct TCR
    tuple forms a clone of >=1 cells, exercising the representative-cell
    selection logic) and a GEX matrix ready for `sc.tl.pca`.

    `adata.raw` is set and `adata.uns['raw_matrix_is_logged']` is marked
    True so `normalize_and_log_the_raw_matrix` (called unconditionally
    inside `reduce_to_single_cell_per_clone`) returns immediately rather
    than requiring a biologically realistic raw-counts matrix.
    """
    rng = np.random.default_rng(random_seed)
    counts = _make_gex_counts(n_cells, n_genes, rng)

    rows = []
    for i in range(n_cells):
        clone_idx = i % n_clones
        va, ja, cdr3a = _VA_JA_CDR3A[clone_idx % len(_VA_JA_CDR3A)]
        vb, jb, cdr3b = _VB_JB_CDR3B[clone_idx % len(_VB_JB_CDR3B)]
        rows.append(dict(
            va=va, ja=ja, cdr3a=cdr3a, cdr3a_nucseq='ACGT',
            vb=vb, jb=jb, cdr3b=cdr3b, cdr3b_nucseq='ACGT',
        ))
    obs = pd.DataFrame(rows, index=[f'cell{i}' for i in range(n_cells)])
    var = pd.DataFrame(index=[f'GENE{i}' for i in range(n_genes)])

    adata = ad.AnnData(X=sp.csr_matrix(counts), obs=obs, var=var)
    adata.raw = adata.copy()
    adata.uns['raw_matrix_is_logged'] = True
    return adata


class TestReduceToSingleCellPerCloneSourceWiresRandomSeed:
    """Source-level assertion test confirming the single internal
    stochastic call site in `reduce_to_single_cell_per_clone` wires
    `random_seed` through, guarding against the wiring silently
    regressing even if the runtime mock test below is weakened or
    skipped.
    """

    def _function_source(self) -> str:
        source = PREPROCESS_SOURCE_PATH.read_text()
        start = source.index('def reduce_to_single_cell_per_clone(')
        # the next top-level `def ` after the function start marks the
        # end of this function's body
        end = source.index('\ndef write_proj_info(', start)
        return source[start:end]

    def test_signature_has_random_seed_default_param(self):
        source = self._function_source()
        signature_region = source[:source.index('):') + 2]
        assert 'random_seed=util.DEFAULT_RANDOM_SEED' in signature_region

    def test_pca_call_site_wires_random_seed(self):
        source = self._function_source()
        assert (
            "sc.tl.pca(adata, svd_solver='arpack',\n"
            "                  n_comps=min(adata.shape[0]-1, n_pcs),\n"
            "                  random_state=random_seed)" in source
        )

    def test_default_random_seed_is_util_default_random_seed(self):
        """Confirms the parameter default matches
        `util.DEFAULT_RANDOM_SEED` (design.md Component 3), via
        introspection rather than a call-site string match.
        """
        sig = inspect.signature(reduce_to_single_cell_per_clone)
        assert sig.parameters['random_seed'].default == util.DEFAULT_RANDOM_SEED

    def test_random_seed_is_last_parameter(self):
        """design.md Component 3: `random_seed` must be the last
        parameter, so every existing keyword-only caller is unaffected.
        """
        sig = inspect.signature(reduce_to_single_cell_per_clone)
        assert list(sig.parameters.keys())[-1] == 'random_seed'


class TestReduceToSingleCellPerCloneRuntimeSeedWiring:
    """Runtime mock/spy test confirming a non-default `random_seed`
    actually reaches the wrapped `sc.tl.pca` call, following
    `tests/test_batch_integration.py`'s `mock.patch(..., wraps=real_fn)`
    convention.
    """

    def test_pca_receives_random_seed(self):
        adata = _make_clone_adata()
        real_pca = sc.tl.pca
        with mock.patch(
            'conga.preprocess.sc.tl.pca', wraps=real_pca,
        ) as mock_pca:
            reduce_to_single_cell_per_clone(
                adata, random_seed=NON_DEFAULT_RANDOM_SEED)

        mock_pca.assert_called()
        _, kwargs = mock_pca.call_args
        assert kwargs['random_state'] == NON_DEFAULT_RANDOM_SEED

    def test_pca_not_called_when_use_existing_pca_obsm_tag_given(self):
        """When `use_existing_pca_obsm_tag` is supplied, the function
        skips its own `sc.tl.pca` call entirely and reuses the given
        obsm array -- confirms the mock test above is actually exercising
        the intended branch rather than some other, unconditional PCA
        call elsewhere in the function.
        """
        adata = _make_clone_adata()
        rng = np.random.default_rng(RANDOM_SEED)
        adata.obsm['X_existing_pca'] = rng.standard_normal((adata.shape[0], 5))
        real_pca = sc.tl.pca
        with mock.patch(
            'conga.preprocess.sc.tl.pca', wraps=real_pca,
        ) as mock_pca:
            reduce_to_single_cell_per_clone(
                adata, use_existing_pca_obsm_tag='X_existing_pca',
                random_seed=NON_DEFAULT_RANDOM_SEED)

        mock_pca.assert_not_called()
