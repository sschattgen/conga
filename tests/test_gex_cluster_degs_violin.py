"""Regression test for the gex_clusters_degs_violinplot figure made by
conga.devel.find_gex_cluster_degs (sc.pl.stacked_violin).

scanpy colors each violin with hue=<the groupby categories>. With pandas 3,
seaborn 0.13's dodge='auto' check (_dodge_needed, which uses value_counts on
categorical columns) counts unobserved category combinations, decides that hue
needs dodging, and draws every violin at 1/num_groups of its slot width, shifted
off-center, while the stripplot points stay at the slot centers. conga passes
dodge=False to avoid this. This test captures the real figure and checks that
every violin is full-width (seaborn's default 0.8 of a slot, since
density_norm='width') and centered on its group.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import scanpy as sc
from scipy import sparse

from conga import devel

N_PER_CLUSTER = [40, 30, 25, 20]
N_MARKERS = 8


def _adata():
    rng = np.random.default_rng(0)
    # genes looked up by score_gex_clusters_naive_vs_memory, plus markers + noise
    fixed_genes = sorted(set(
        'B2M IL32 S100A4 CCL5 PFN1 SH3BGRL3 HLA-C ANXA1 HLA-A CD99 NKG7 CST7 GZMA '
        'GZMH GZMK CTSW SRGN LEF1 CCR7 SELL MAL TCF7 NELL2 BACH2 EEF1A1 EEF1B2 CD74 '
        'PRF1 S100A10 S100A11 CD52 S100A6 ACTB VIM TMSB4X'.split()))
    markers = [f'MARKER{c}_{j}' for c in range(len(N_PER_CLUSTER)) for j in range(N_MARKERS)]
    genes = fixed_genes + markers + [f'NOISE{i}' for i in range(20)]
    clusters = np.concatenate([[c]*n for c, n in enumerate(N_PER_CLUSTER)])
    X = rng.poisson(0.3, size=(len(clusters), len(genes))).astype(float)
    for c in range(len(N_PER_CLUSTER)):
        cols = [genes.index(f'MARKER{c}_{j}') for j in range(N_MARKERS)]
        X[np.ix_(clusters == c, cols)] += rng.poisson(20, size=((clusters == c).sum(), N_MARKERS))
    X = sparse.csr_matrix(np.log1p(X)) # sparse, as in the pipeline
    adata = sc.AnnData(X=X)
    adata.var_names = genes
    adata.obs_names = [f'cell{i}' for i in range(len(clusters))]
    adata.raw = adata.copy()
    adata.obs['clusters_gex'] = clusters
    adata.uns['organism'] = 'human'
    return adata


def test_degs_violins_are_full_width_and_centered(tmp_path, monkeypatch):
    captured = {}
    original_savefig = plt.savefig
    def savefig(fname, *args, **kwargs):
        if str(fname).endswith('_gex_cluster_degs_violin.png'):
            captured['fig'] = plt.gcf()
        return original_savefig(fname, *args, **kwargs)
    monkeypatch.setattr(devel.plt, 'savefig', savefig)

    devel.find_gex_cluster_degs(_adata(), str(tmp_path / 'test'))
    assert (tmp_path / 'test_gex_cluster_degs_violin.png').exists()
    fig = captured['fig']

    widths, centers = [], []
    for ax in fig.axes:
        for coll in ax.collections:
            if not isinstance(coll, PolyCollection):
                continue
            for path in coll.get_paths():
                xs = path.vertices[:, 0]
                widths.append(xs.max() - xs.min())
                centers.append((xs.max() + xs.min()) / 2)
    plt.close('all')

    widths, centers = np.array(widths), np.array(centers)
    assert len(widths) >= len(N_PER_CLUSTER) * 4, len(widths)
    drawn = widths > 1e-6 # a single-valued group can give a zero-width violin
    assert drawn.sum() >= len(N_PER_CLUSTER) * 4
    # the bug: widths of 0.8/num_groups = 0.2 here, centers off by up to 0.3
    np.testing.assert_allclose(widths[drawn], 0.8, atol=1e-6)
    np.testing.assert_allclose(centers[drawn], np.round(centers[drawn]), atol=1e-6)
