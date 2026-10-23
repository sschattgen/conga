"""Guard test: the default figure paths must not write SVG files or call an
external SVG-to-PNG converter (svg_free branch).

Every SVG file writer (svg_basic.create_file, tcrdist_svg_basic.create_file)
and every import site of convert_svg_to_png is replaced by a function that
raises. Then the real plotting functions are run with their default arguments
on a small AnnData built from test_data/classic_clones.tsv:
  * make_logo_plots (graph-vs-graph / tcr-clumping logo figure), with DEG,
    TCR-feature and lit-match logos
  * make_tcrdist_trees and make_tcrdist_tree_for_conga_score_threshold,
    including save_pdf
  * make_tcr_logo_for_tcrs (used by scripts/make_tcr_logos.py)
As a negative control, the legacy logo_renderer='svg' route must trip the guard.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from scipy import sparse

import conga
import conga.convert_svg_to_png
from conga import plotting, svg_basic
from conga.tcrdist import basic as tcrdist_basic
from conga.tcrdist import tcrdist_svg_basic
from conga.tcrdist.make_tcr_logo import make_tcr_logo_for_tcrs

REPO_ROOT = Path(__file__).resolve().parent.parent
CLONES_FILE = REPO_ROOT / 'test_data' / 'classic_clones.tsv'

MIN_CLUSTER_SIZE = 5


class SvgRouteUsed(Exception):
    pass


@pytest.fixture
def no_svg(monkeypatch):
    def forbidden(*args, **kwargs):
        raise SvgRouteUsed(f'svg file writer or converter called: {args[:1]}')
    for module in [svg_basic, tcrdist_svg_basic]:
        monkeypatch.setattr(module, 'create_file', forbidden)
    for module in [conga.convert_svg_to_png, svg_basic, tcrdist_basic]:
        monkeypatch.setattr(module, 'convert_svg_to_png', forbidden)
    yield
    plt.close('all')


@pytest.fixture(scope='module')
def adata():
    ''' 60 clonotypes: 30 TRAV1-2 (MAIT-like) + 30 others, with two
    graph-vs-graph "hit" clusters: (0,0) = 20 MAIT-like, (1,1) = 8 others
    '''
    df = pd.read_csv(CLONES_FILE, sep='\t')
    mait = df[df.va_gene.str.startswith('TRAV1-2*')].head(30)
    other = df[~df.va_gene.str.startswith('TRAV1-2*')].head(30)
    df = pd.concat([mait, other]).reset_index(drop=True)
    n = df.shape[0]
    assert n == 60

    rng = np.random.default_rng(0)
    genes = ([g for g in plotting.default_logo_genes['human']] +
             [f'GENE{i}' for i in range(30)])
    X = sparse.csr_matrix(rng.poisson(1.0, size=(n, len(genes))).astype(float))
    adata = sc.AnnData(X=X)
    adata.var_names = genes
    adata.obs_names = [f'clone{i}' for i in range(n)]
    adata.raw = adata.copy()

    for col, src in [('va', 'va_gene'), ('ja', 'ja_gene'), ('cdr3a', 'cdr3a'),
                     ('cdr3a_nucseq', 'cdr3a_nucseq'), ('vb', 'vb_gene'),
                     ('jb', 'jb_gene'), ('cdr3b', 'cdr3b'),
                     ('cdr3b_nucseq', 'cdr3b_nucseq')]:
        adata.obs[col] = list(df[src])
    adata.obs['clone_sizes'] = list(df.clone_size)

    clusters_gex = np.array([0]*30 + [1]*15 + [2]*15)
    clusters_tcr = np.array([0]*30 + [1]*15 + [2]*15)
    conga_scores = np.full(n, 10.0)
    conga_scores[:20] = 1e-4   # hits in cluster pair (0,0)
    conga_scores[30:38] = 1e-2 # hits in cluster pair (1,1)
    adata.obs['clusters_gex'] = clusters_gex
    adata.obs['clusters_tcr'] = clusters_tcr
    adata.obs['conga_scores'] = conga_scores
    adata.obsm['X_gex_2d'] = rng.normal(size=(n, 2))
    adata.obsm['X_tcr_2d'] = rng.normal(size=(n, 2))
    adata.uns['organism'] = 'human'

    # fake DEG results, in the form _parse_bicluster_rank_genes reads
    deg_genes = [f'GENE{i}' for i in range(12)]
    adata.uns['rank_genes_test'] = {
        'names': {c: deg_genes for c in ['clp_0_0', 'clp_1_1']},
        'logfoldchanges': {c: list(np.linspace(3, -1, 12))
                           for c in ['clp_0_0', 'clp_1_1']},
        'pvals_adj': {c: list(np.logspace(-12, -1, 12))
                      for c in ['clp_0_0', 'clp_1_1']},
    }
    return adata


def _nbrs(n, k, seed):
    rng = np.random.default_rng(seed)
    return np.array([rng.choice([j for j in range(n) if j != i], k, replace=False)
                     for i in range(n)])


def _assert_only(dirpath, expected_names):
    found = sorted(p.name for p in Path(dirpath).iterdir())
    assert found == sorted(expected_names), found


def _logo_plot_kwargs():
    return dict(
        rank_genes_uns_tag='rank_genes_test',
        good_bicluster_tcr_scores={
            (0, 0): [('mait', 2.0, 1e-9), ('cdr3len', -1.0, 1e-4)],
            (1, 1): [('cd8', 1.5, 1e-3)]},
        lit_matches=pd.DataFrame(dict(
            clone_index=[0, 1, 2, 31],
            db_mhc=['MR1', 'MR1', np.nan, 'HLA-A*02:01'],
            db_epitope=['5-OP-RU', '5-OP-RU', 'X', 'GILGFVFTL'])),
        make_gex_header=False,
        gex_header_tcr_score_names=[],
    )


def test_logo_figure_needs_no_svg_converter(adata, no_svg, tmp_path):
    pngfile = str(tmp_path / 'logos.png')
    plotting.make_logo_plots(adata.copy(), _nbrs(60, 6, 1), _nbrs(60, 6, 2),
                             MIN_CLUSTER_SIZE, pngfile, **_logo_plot_kwargs())
    _assert_only(tmp_path, ['logos.png']) # no svg or temporary png files
    img = plt.imread(pngfile)
    assert img.shape[0] > 500 and img.shape[1] > 2000


def test_legacy_svg_logo_route_trips_the_guard(adata, no_svg, tmp_path):
    # negative control: shows the guard really catches the converter route
    with pytest.raises(SvgRouteUsed):
        plotting.make_logo_plots(adata.copy(), _nbrs(60, 6, 1), _nbrs(60, 6, 2),
                                 MIN_CLUSTER_SIZE, str(tmp_path / 'logos.png'),
                                 logo_renderer='svg', **_logo_plot_kwargs())


def test_tree_figures_need_no_svg_converter(adata, no_svg, tmp_path):
    prefix = str(tmp_path / 'test')
    a = adata.copy()
    plotting.make_tcrdist_trees(a, prefix, group_by='clusters_gex', save_pdf=True)
    plotting.make_tcrdist_tree_for_conga_score_threshold(
        a, 1.0, prefix, min_clonotypes=10, save_pdf=True)
    figs = ['test_gex_clusters_tcrdist_trees', 'test_conga_threshold_tcrdist_tree']
    _assert_only(tmp_path, [f + ext for f in figs
                            for ext in ['.png', '.pdf', '.png_README.txt']])
    for f in figs:
        assert (tmp_path / f'{f}.pdf').read_bytes()[:5] == b'%PDF-'
    img = plt.imread(str(tmp_path / 'test_conga_threshold_tcrdist_tree.png'))
    assert img.shape[:2] == (1000, 800) # default png_scale=1, as before
    gex = plt.imread(str(tmp_path / 'test_gex_clusters_tcrdist_trees.png'))
    assert gex.shape[:2] == (1000, 3*825 - 25) # 3 clusters side by side


def test_tcr_logo_function_needs_no_svg_converter(adata, no_svg, tmp_path):
    tcrs = conga.preprocess.retrieve_tcrs_from_adata(adata)[:30]
    for chain in 'AB':
        pngfile = str(tmp_path / f'logo_{chain}.png')
        make_tcr_logo_for_tcrs(tcrs, chain, 'human', pngfile, png_scale=2)
        assert plt.imread(pngfile).shape[:2] == (2*158, 2*1250)
    _assert_only(tmp_path, ['logo_A.png', 'logo_B.png'])
