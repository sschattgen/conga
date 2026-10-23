"""Cluster labels are numbered by decreasing cluster size (0 = largest).

igraph's leiden implementation (flavor='igraph', used since 0ca6252) returns
labels in arbitrary order; leidenalg, which conga used before, numbered them by
decreasing size. preprocess.relabel_clusters_by_decreasing_size restores that
convention without changing the partition. These tests check the helper, and
that both pipeline clustering paths (cluster_and_tsne_and_umap and the exact
TCRdist path calc_tcrdist_nbrs_umap_clusters_cpp) give exactly the same
partition as without relabeling, now with size-ordered labels.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from conga import preprocess, util
from conga.preprocess import relabel_clusters_by_decreasing_size

REPO_ROOT = Path(__file__).resolve().parent.parent
CLONES_FILE = REPO_ROOT / 'test_data' / 'classic_clones.tsv'


## the helper ##################################################################

def test_relabel_orders_by_size_and_keeps_partition():
    labels = np.array([3]*5 + [0]*2 + [7]*9 + [1]*4)
    new = relabel_clusters_by_decreasing_size(labels)
    assert list(np.bincount(new)) == [9, 5, 4, 2]
    assert list(new[:5]) == [1]*5 and list(new[5:7]) == [3]*2
    assert list(new[7:16]) == [0]*9 and list(new[16:]) == [2]*4


def test_relabel_ties_broken_by_original_label():
    labels = np.array([5, 5, 2, 2, 9, 9, 9])
    assert list(relabel_clusters_by_decreasing_size(labels)) == [2, 2, 1, 1, 0, 0, 0]


def test_relabel_is_noop_when_already_size_ordered():
    labels = np.array([0]*6 + [1]*4 + [2]*4 + [3])
    np.testing.assert_array_equal(relabel_clusters_by_decreasing_size(labels), labels)


def test_relabel_accepts_scanpy_style_string_categories():
    labels = pd.Categorical(['1', '0', '1', '10', '1', '0'],
                            categories=['0', '1', '10'])
    assert list(relabel_clusters_by_decreasing_size(labels)) == [0, 1, 0, 2, 0, 1]


## the pipeline clustering paths ###############################################

def _same_partition_bijection(a, b):
    pairs = set(zip(a, b))
    return len(pairs) == len(set(a)) == len(set(b))


def _run_with_and_without_relabeling(monkeypatch, run):
    ''' run(adata_copy) --> labels; returns (relabeled, raw) '''
    relabeled = run()
    with monkeypatch.context() as m:
        m.setattr(preprocess, 'relabel_clusters_by_decreasing_size',
                  lambda labels: np.array(labels).astype(int))
        raw = run()
    return relabeled, raw


def _blobs_adata(seed=0):
    ''' unequal-sized gaussian blobs, so leiden finds clusters of varied sizes '''
    rng = np.random.default_rng(seed)
    def blobs(sizes, dim):
        centers = rng.normal(scale=8.0, size=(len(sizes), dim))
        return np.vstack([c + rng.normal(size=(n, dim)) for c, n in zip(centers, sizes)])
    gex = blobs([15, 90, 40, 70, 25, 60], 20)
    tcr = blobs([50, 20, 80, 35, 60, 55], 20)
    perm = rng.permutation(gex.shape[0]) # so blob order != row order
    adata = sc.AnnData(X=np.zeros((gex.shape[0], 60)))
    adata.obsm['X_pca_gex'] = gex[perm]
    adata.obsm['X_pca_tcr'] = tcr[perm]
    return adata


def test_cluster_and_tsne_and_umap_labels_size_ordered(monkeypatch):
    base = _blobs_adata()
    results = {}
    def run():
        adata = base.copy()
        preprocess.cluster_and_tsne_and_umap(adata, make_1d_umaps=False)
        results.setdefault('adatas', []).append(adata)
        return {tag: np.array(adata.obs[f'clusters_{tag}']) for tag in ['gex', 'tcr']}
    relabeled, raw = _run_with_and_without_relabeling(monkeypatch, run)
    adata = results['adatas'][0]
    for tag in ['gex', 'tcr']:
        new, old = relabeled[tag], raw[tag]
        counts = np.bincount(new)
        assert len(counts) >= 4
        assert np.all(counts[:-1] >= counts[1:]), counts # size-ordered
        assert _same_partition_bijection(new, old) # same clustering
        # leiden_gex / leiden_tcr kept consistent with clusters_gex / clusters_tcr
        np.testing.assert_array_equal(
            np.array(adata.obs[f'leiden_{tag}']).astype(int), new)


@pytest.mark.skipif(not util.tcrdist_cpp_available(),
                    reason='tcrdist_cpp binaries not compiled')
def test_exact_tcrdist_path_labels_size_ordered(monkeypatch, tmp_path):
    df = pd.read_csv(CLONES_FILE, sep='\t').head(400)
    base = sc.AnnData(X=np.zeros((df.shape[0], 60)))
    base.obs['va'] = list(df.va_gene)
    base.obs['cdr3a'] = list(df.cdr3a)
    base.obs['vb'] = list(df.vb_gene)
    base.obs['cdr3b'] = list(df.cdr3b)
    base.uns['organism'] = 'human'
    def run():
        adata = base.copy()
        preprocess.calc_tcrdist_nbrs_umap_clusters_cpp(
            adata, 10, tmpfile_prefix=str(tmp_path / 'tcrdist'),
            make_1d_umaps=False, clustering_method='leiden')
        return np.array(adata.obs['clusters_tcr'])
    new, old = _run_with_and_without_relabeling(monkeypatch, run)
    counts = np.bincount(new)
    assert len(counts) >= 4
    assert np.all(counts[:-1] >= counts[1:]), counts
    assert _same_partition_bijection(new, old)
