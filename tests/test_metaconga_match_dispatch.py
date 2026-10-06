"""End-to-end integration tests for Requirement 8 (Analysis_Dispatch_Section)
and Requirement 9.8/9.9 of the metaconga-match-integration feature.

Builds a small, synthetic, human AnnData fixture with the fields the
AACluster_Pipeline and Clump_Pipeline read, and invokes both pipelines'
public functions directly (not via the CLI), asserting only that no
exception is raised. Per Requirement 9.8/9.9, these tests are explicitly
NOT required to assert on the statistical correctness of the returned
matches/scores -- the synthetic fixture is far too small and arbitrary
for that to be meaningful. They exist to catch integration breakage: bad
imports, signature mismatches with preprocess.py/tcr_clumping.py, missing
data files, or dtype assumptions (e.g. sparse vs. dense adata.raw.X).

Gene/TCR identities are deliberately real (drawn from conga's own gene
database and the bundled metaconga DEG reference files) rather than
arbitrary strings, since several code paths in metaconga_match.py
validate V/J gene names against conga.tcrdist.all_genes and CDR3
nucleotide sequences through the C++ tcrdist backend; arbitrary
placeholder strings fail those checks before the pipelines' own logic is
ever exercised.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import anndata as ad
import scanpy as sc

import conga
import conga.metaconga_match
from conga.tcrdist.genetic_code import genetic_code


# Build an amino-acid -> valid-codon lookup (inverse of the genetic code)
# so synthetic CDR3 nucleotide sequences round-trip-translate correctly
# through the C++ tcrdist backend's own CDR3 validation.
_AA_TO_CODON = {}
for _codon, _aa in genetic_code.items():
    if _aa not in _AA_TO_CODON and len(_codon) == 3 and set(_codon) <= set('acgt'):
        _AA_TO_CODON[_aa] = _codon


def _aa_seq_to_nucseq(aa_seq):
    return ''.join(_AA_TO_CODON[aa] for aa in aa_seq)


# Real human V/J allele names, confirmed present in
# conga.tcrdist.all_genes.all_genes['human'], paired with real CDR3
# sequences drawn from conga/data/human_tcr_db_for_matching.tsv.
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

# A handful of genes confirmed present (via direct inspection this
# session) among the top differentially-expressed genes for CD4
# aacluster 0 in the bundled run105_cd4_deg_results.tsv reference file,
# plus filler genes so scanpy's PCA/neighbors/leiden/rank_genes_groups
# calls have a reasonably-sized gene space to operate on.
_REAL_GENES = ['IKZF2', 'IL7R', 'ANXA1', 'IL32', 'TIGIT', 'FOXP3', 'TCF7',
               'CD27', 'LEF1', 'SELL', 'B2M', 'HLA-A', 'PTPRC', 'VIM']
_FILLER_GENES = [f'FILLERGENE{i}' for i in range(36)]
_GENES = _REAL_GENES + _FILLER_GENES


def _build_representative_human_fixture(n_cells=60, seed=42):
    """Builds a Representative_Human_Fixture (per requirements.md
    Glossary): a synthetic AnnData with organism='human' and the
    GEX/TCR fields the AACluster_Pipeline and Clump_Pipeline read.
    """
    rng = np.random.default_rng(seed)

    rows = []
    for i in range(n_cells):
        va, ja, cdr3a = _VA_JA_CDR3A[i % len(_VA_JA_CDR3A)]
        vb, jb, cdr3b = _VB_JB_CDR3B[i % len(_VB_JB_CDR3B)]
        rows.append(dict(
            va=va, ja=ja, cdr3a=cdr3a, cdr3a_nucseq=_aa_seq_to_nucseq(cdr3a),
            vb=vb, jb=jb, cdr3b=cdr3b, cdr3b_nucseq=_aa_seq_to_nucseq(cdr3b),
            clone_sizes=1,
        ))
    obs = pd.DataFrame(rows, index=[f'cell{i}' for i in range(n_cells)])

    n_genes = len(_GENES)
    X = sp.csr_matrix(
        rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float32))
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=_GENES))
    adata.uns['organism'] = 'human'

    adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    adata.raw = adata

    sc.pp.pca(adata, n_comps=10, random_state=seed)
    adata.obsm['X_pca_gex'] = adata.obsm['X_pca']

    sc.pp.neighbors(adata, n_neighbors=5, random_state=seed)
    sc.tl.leiden(adata, resolution=0.5, key_added='clusters_gex',
                 flavor='igraph', n_iterations=2, directed=False,
                 random_state=seed)
    sc.tl.umap(adata, random_state=seed)
    adata.obsm['X_gex_2d'] = adata.obsm['X_umap']

    return adata


@pytest.fixture(scope='module')
def representative_human_fixture():
    return _build_representative_human_fixture()


class TestAaclusterPipelineEndToEnd:
    """Requirement 9.8: find_aacluster_matches and plot_aacluster_matches
    complete without raising, against a Representative_Human_Fixture.
    Statistical correctness of the returned matches is explicitly out of
    scope for this test.
    """

    @pytest.mark.parametrize('cd48', ['cd4', 'cd8'])
    def test_find_and_plot_aacluster_matches_do_not_raise(
            self, representative_human_fixture, tmp_path, cd48):
        matches = conga.metaconga_match.find_aacluster_matches(
            representative_human_fixture, cd48)
        assert matches is not None
        assert hasattr(matches, 'pvals')
        assert hasattr(matches, 'degs')
        assert hasattr(matches, 'obs')

        outfile_prefix = str(tmp_path / f'aacluster_{cd48}_test')
        conga.metaconga_match.plot_aacluster_matches(
            representative_human_fixture, matches, outfile_prefix)


class TestClumpPipelineEndToEnd:
    """Requirement 9.9: find_clump_matches and plot_clump_matches
    complete without raising, against a Representative_Human_Fixture.
    Statistical correctness of the returned clump matches is explicitly
    out of scope for this test.
    """

    def test_find_and_plot_clump_matches_do_not_raise(
            self, representative_human_fixture, tmp_path):
        # find_clump_matches mutates adata.uns in place and returns None;
        # operate on a copy so this test doesn't interfere with the
        # module-scoped fixture shared with TestAaclusterPipelineEndToEnd.
        adata = representative_human_fixture.copy()

        conga.metaconga_match.find_clump_matches(adata)

        outfile_prefix = str(tmp_path / 'clump_test')
        conga.metaconga_match.plot_clump_matches(adata, outfile_prefix)


class TestBothPipelinesIndependentlyRunnable:
    """Requirement 8.4: the AACluster_Pipeline and Clump_Pipeline must be
    independently runnable in the same invocation; neither reads or sets
    any flag or adata field the other depends on."""

    def test_both_pipelines_run_against_same_adata_without_interference(
            self, representative_human_fixture, tmp_path):
        adata = representative_human_fixture.copy()

        matches = conga.metaconga_match.find_aacluster_matches(adata, 'cd4')
        conga.metaconga_match.plot_aacluster_matches(
            adata, matches, str(tmp_path / 'combo_aacluster_test'))

        conga.metaconga_match.find_clump_matches(adata)
        conga.metaconga_match.plot_clump_matches(
            adata, str(tmp_path / 'combo_clump_test'))
