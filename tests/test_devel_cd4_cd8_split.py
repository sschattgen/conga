"""
Regression test for `conga.devel.split_into_cd4_and_cd8_subsets`.

This function previously called the removed `AnnData.concatenate()`
instance method (dropped in anndata>=0.13), which raised
`AttributeError: 'AnnData' object has no attribute 'concatenate'`.
It was replaced with the module-level `anndata.concat()` function.

A loop variable in the same function was also named `ad`, shadowing
the `import anndata as ad` module alias for the remainder of the
function's scope -- it was renamed to `subset` to avoid the collision.

This test builds a small synthetic fixture with two clearly separable
CD4-high and CD8-high populations and confirms the full split runs
end to end without raising, producing a CD4 subset and a CD8 subset
that match the fixture's known ground truth.
"""

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import scanpy as sc

import conga.devel

def _make_cd4_cd8_fixture(n_cells: int = 200, n_genes: int = 60,
                           seed: int = 42) -> ad.AnnData:
    """Build a synthetic AnnData with a clean CD4-high half and a
    CD8A/CD8B-high half, plus the minimal paired-TCR obs columns
    `conga.preprocess.retrieve_tcrs_from_adata` requires.
    """
    rng = np.random.default_rng(seed)
    genes = ['CD4', 'CD8A', 'CD8B'] + [f'GENE{i}' for i in range(n_genes - 3)]

    is_cd4 = np.arange(n_cells) < n_cells // 2
    X = rng.poisson(2.0, size=(n_cells, n_genes)).astype(float)
    X[is_cd4, 0] += 50       # CD4
    X[~is_cd4, 1] += 50      # CD8A
    X[~is_cd4, 2] += 50      # CD8B

    obs = pd.DataFrame({
        'va': ['TRAV1-1*01'] * n_cells,
        'ja': ['TRAJ1*01'] * n_cells,
        'cdr3a': ['CALIPGGQKLLF'] * n_cells,
        'cdr3a_nucseq': ['acgt'] * n_cells,
        'vb': ['TRBV1*01'] * n_cells,
        'jb': ['TRBJ1-1*01'] * n_cells,
        'cdr3b': ['CAAGETSGVSYNEQF'] * n_cells,
        'cdr3b_nucseq': ['acgt'] * n_cells,
    })
    var = pd.DataFrame(index=genes)
    adata = ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=var)
    adata.uns['organism'] = 'human'
    adata.raw = adata.copy()

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    return adata

def test_split_into_cd4_and_cd8_subsets_runs_without_raising():
    """Confirms the anndata.concat() fix: the function must complete
    without AttributeError and must not raise due to the `ad` loop
    variable shadowing the `anndata` module alias.
    """
    adata = _make_cd4_cd8_fixture()
    adata_cd4, adata_cd8 = conga.devel.split_into_cd4_and_cd8_subsets(
        adata, verbose=False, strict=True)

    assert adata_cd4.n_obs > 0
    assert adata_cd8.n_obs > 0

def test_split_into_cd4_and_cd8_subsets_separates_populations_correctly():
    """The fixture's two populations are unambiguous (CD4 vs. CD8A/CD8B
    expression differs by ~50x), so the split should recover both
    populations at their known sizes with no cross-contamination.
    """
    n_cells = 200
    adata = _make_cd4_cd8_fixture(n_cells=n_cells)
    adata_cd4, adata_cd8 = conga.devel.split_into_cd4_and_cd8_subsets(
        adata, verbose=False, strict=True)

    assert adata_cd4.n_obs == n_cells // 2
    assert adata_cd8.n_obs == n_cells // 2
    assert set(adata_cd4.obs_names).isdisjoint(set(adata_cd8.obs_names))
    assert (set(adata_cd4.obs_names) | set(adata_cd8.obs_names) ==
            set(adata.obs_names))
