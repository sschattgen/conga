"""
Tests that every random step reachable from scripts/run_conga.py is controlled
by an explicit seed (the --random_seed flag, default util.DEFAULT_RANDOM_SEED).

For each seeded step we check that
  * the same seed gives identical output, even after the global `random` and
    `np.random` states have been advanced (so the step does not secretly use
    them), and that the global states are left untouched;
  * a different seed gives different output (so the randomness is actually
    controlled by the seed, rather than absent).

Plus two CLI checks: --random_seed is accepted on the non-vectorized TCR paths
(it used to be rejected there as an "encoding flag"), while the real encoding
flags still are rejected.
"""

import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conga import util

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_CONGA_SCRIPT = REPO_ROOT / 'scripts' / 'run_conga.py'
CLONES_FILE = REPO_ROOT / 'test_data' / 'classic_clones.tsv'
SEED, OTHER_SEED = util.DEFAULT_RANDOM_SEED, 1234

needs_cpp = pytest.mark.skipif(not util.tcrdist_cpp_available(),
                               reason='needs the compiled tcrdist_cpp binaries')


def _scramble_global_rngs():
    random.random()
    np.random.random()


def _global_rng_states():
    return random.getstate(), np.random.get_state()


def _assert_global_rng_states_equal(a, b):
    assert a[0] == b[0], 'global random state was changed'
    for x, y in zip(a[1], b[1]):
        assert np.array_equal(x, y), 'global np.random state was changed'


def _run_twice_and_with_other_seed(func, check_global_rngs=True):
    """func(seed) -> output. Returns (out_seed, out_seed_again, out_other_seed)
    with the global RNGs scrambled before each call and (by default) checked
    to be untouched by it."""
    outs = []
    for seed in [SEED, SEED, OTHER_SEED]:
        _scramble_global_rngs()
        before = _global_rng_states()
        outs.append(func(seed))
        if check_global_rngs:
            _assert_global_rng_states_equal(before, _global_rng_states())
    return outs


@pytest.fixture(scope='module')
def classic_tcrs():
    df = pd.read_csv(CLONES_FILE, sep='\t').iloc[:300]
    tcrs = [((l.va_gene, l.ja_gene, l.cdr3a, l.cdr3a_nucseq.lower()),
             (l.vb_gene, l.jb_gene, l.cdr3b, l.cdr3b_nucseq.lower()))
            for l in df.itertuples()]
    assert len(tcrs) == 300
    return tcrs


# background TCR sampling (tcr_clumping, tcr_db_match, metaconga clump match)

@pytest.mark.parametrize('chain', ['A', 'B'])
@pytest.mark.parametrize('preserve_vj_pairings', [False, True])
def test_resample_shuffled_tcr_chains_is_seeded(
        classic_tcrs, chain, preserve_vj_pairings):
    from conga.tcrdist import tcr_sampler
    junctions_df = tcr_sampler.parse_tcr_junctions('human', classic_tcrs)

    a, a2, b = _run_twice_and_with_other_seed(
        lambda seed: tcr_sampler.resample_shuffled_tcr_chains(
            'human', 500, chain, junctions_df,
            preserve_vj_pairings=preserve_vj_pairings, random_seed=seed))
    assert len(a) == 500
    assert a == a2
    assert a != b


@needs_cpp
def test_background_tcrdist_distributions_are_seeded(classic_tcrs, tmp_path):
    # the background-probability matrix that the tcr_clumping and tcr_db_match
    # p-values are computed from
    from conga import tcr_clumping
    # it names its temporary files with the global random module, which
    # doesn't affect the results
    a, a2, b = _run_twice_and_with_other_seed(
        lambda seed: tcr_clumping.estimate_background_tcrdist_distributions(
            'human', classic_tcrs, 96, num_random_samples=2000,
            tmpfile_prefix=str(tmp_path/'tmp'), random_seed=seed),
        check_global_rngs=False)
    assert a.shape == (len(classic_tcrs), 97)
    assert np.array_equal(a, a2)
    assert not np.array_equal(a, b)


@needs_cpp
def test_background_alpha_and_beta_chains_use_different_seeds(
        classic_tcrs, monkeypatch, tmp_path):
    from conga import tcr_clumping
    from conga.tcrdist import tcr_sampler
    seeds = {}
    original = tcr_sampler.resample_shuffled_tcr_chains
    def spy(organism, num_samples, chain, junctions_df, **kwargs):
        seeds[chain] = kwargs['random_seed']
        return original(organism, num_samples, chain, junctions_df, **kwargs)
    monkeypatch.setattr(tcr_sampler, 'resample_shuffled_tcr_chains', spy)
    tcr_clumping.estimate_background_tcrdist_distributions(
        'human', classic_tcrs[:50], 48, num_random_samples=500,
        tmpfile_prefix=str(tmp_path/'tmp'), random_seed=SEED)
    assert seeds == {'A': SEED, 'B': SEED+1}


# graph-vs-graph permutation test

def test_graph_overlap_stats_are_seeded():
    from conga.correlations import _compute_graph_overlap_stats
    rng = np.random.default_rng(0)
    N, K = 400, 10
    gex_nbrs = [rng.choice(N, K, replace=False) for _ in range(N)]
    # correlated tcr graph: half of the neighbors are shared
    tcr_nbrs = [np.concatenate([g[:K//2], rng.choice(N, K//2, replace=False)])
                for g in gex_nbrs]
    a, a2, b = _run_twice_and_with_other_seed(
        lambda seed: _compute_graph_overlap_stats(
            gex_nbrs, tcr_nbrs, 50, random_seed=seed))
    assert a['overlap_zscore_source'] == 'shuffling'
    for x in [a, a2, b]:
        del x['calculation_time']
    assert a == a2
    assert a['overlap'] == b['overlap'] # the observed overlap isn't random
    assert a['overlap_zscore'] != b['overlap_zscore']


# TCRdist tree subsampling

def test_tcr_tree_subsampling_is_seeded(classic_tcrs):
    from conga.tcrdist.make_tcr_trees import make_tcr_tree_svg_commands
    from conga.tcrdist.tcr_distances import TcrDistCalculator
    tcrs = classic_tcrs[:80]
    tdist = TcrDistCalculator('human')
    D = np.array([[tdist(x, y) for y in tcrs] for x in tcrs])
    a, a2, b = _run_twice_and_with_other_seed(
        lambda seed: make_tcr_tree_svg_commands(
            tcrs, 'human', [0, 0], [800, 1000], D, tcrdist_calculator=tdist,
            max_tcrs_for_trees=40, random_seed=seed))
    assert len(a) > 0
    assert a == a2
    assert a != b


# FAISS index building (only reached by direct callers of conga.neighbors,
# not by run_conga.py, whose calc_nbrs uses exact indices above 10240 clones)

def test_faiss_ivf_index_building_is_seeded(monkeypatch):
    pytest.importorskip('faiss')
    from conga import neighbors
    rng = np.random.default_rng(0)
    X = rng.standard_normal((15000, 8)).astype(np.float32)

    built = []
    original = neighbors.FaissNeighborSearcher._build_ivf_index
    def spy(self, *args, **kwargs):
        built.append(True)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(neighbors.FaissNeighborSearcher, '_build_ivf_index', spy)

    def run(seed):
        config = neighbors.FaissIndexConfig(train_size_limit=3000,
                                            random_seed=seed)
        searcher = neighbors.FaissNeighborSearcher(
            force_backend=neighbors.Backend.FAISS_CPU, index_config=config)
        result = searcher.search_neighbors(X, [0.001], data_type='tcr')
        return result.neighbors[0.001]
    a, a2, b = _run_twice_and_with_other_seed(run)
    assert len(built) == 3, 'test data did not reach the IVF index path'
    assert np.array_equal(a, a2)
    # IVF search is approximate, so a different training sample gives
    # (slightly) different neighbors
    assert not np.array_equal(a, b)


# devel.py

def test_calc_nbr_zscore_threshold_is_seeded():
    from conga.devel import calc_nbr_zscore_threshold
    rng = np.random.default_rng(0)
    nbrs = np.array([rng.choice(300, 10, replace=False) for _ in range(300)])
    a, a2, b = _run_twice_and_with_other_seed(
        lambda seed: calc_nbr_zscore_threshold(nbrs, random_seed=seed))
    assert a == a2
    assert a != b


# third-party code that only uses the global np.random state

def test_seeded_global_np_random_restores_global_state():
    _scramble_global_rngs()
    before = _global_rng_states()
    draws = []
    for seed in [SEED, SEED, OTHER_SEED]:
        with util.seeded_global_np_random(seed):
            draws.append(np.random.random(5))
        _assert_global_rng_states_equal(before, _global_rng_states())
    assert np.array_equal(draws[0], draws[1])
    assert not np.array_equal(draws[0], draws[2])
    with pytest.raises(ZeroDivisionError):
        with util.seeded_global_np_random(SEED):
            1/0
    _assert_global_rng_states_equal(before, _global_rng_states())


def test_qc_violin_jitter_is_seeded(tmp_path):
    import matplotlib.pyplot as plt
    from conga.preprocess import filter_normalize_and_hvg
    from test_counts_layer import _make_adata
    def run(seed):
        prefix = str(tmp_path/f'qc_{len(list(tmp_path.iterdir()))}')
        filter_normalize_and_hvg(
            _make_adata(), min_genes_per_cell=1, max_genes_per_cell=10000,
            max_percent_mito=1.0, min_cells_per_gene=1,
            outfile_prefix_for_qc_plots=prefix, random_seed=seed)
        return plt.imread(prefix+'_qc_violins.png')
    a, a2, b = _run_twice_and_with_other_seed(run)
    assert np.array_equal(a, a2)
    assert not np.array_equal(a, b)


# the per-gene columns of _cluster_gene_scores.tsv were in set-iteration
# order, which varies between processes (PYTHONHASHSEED)

def test_cluster_gene_scores_columns_are_sorted(tmp_path):
    import anndata as ad
    import scipy.sparse as sps
    from conga.devel import score_gex_clusters_naive_vs_memory
    genes = ('B2M IL32 S100A4 CCL5 PFN1 SH3BGRL3 HLA-C ANXA1 HLA-A CD99 NKG7 '
             'CST7 GZMA GZMH GZMK CTSW SRGN LEF1 CCR7 SELL MAL TCF7 NELL2 '
             'BACH2 EEF1A1 EEF1B2 CD74 PRF1 S100A10 S100A11 CD52 S100A6 ACTB '
             'VIM TMSB4X').split()
    rng = np.random.default_rng(0)
    adata = ad.AnnData(
        X=sps.csr_matrix(rng.poisson(2.0, (40, len(genes))).astype(float)),
        var=pd.DataFrame(index=genes))
    adata.raw = adata
    adata.obs['clusters_gex'] = np.arange(40) % 3
    score_gex_clusters_naive_vs_memory(adata, str(tmp_path/'x'))
    df = pd.read_csv(tmp_path/'x_cluster_gene_scores.tsv', sep='\t')
    gene_cols = [c for c in df.columns if c in genes]
    assert len(gene_cols) == len(genes)
    assert gene_cols == sorted(gene_cols)


# the vectorized encoding doesn't depend on the seed at all; if this fails,
# the seed has started to matter there (it is already passed through, via
# EncodingConfig), so update the --random_seed help text in run_conga.py

def test_aa_embedding_does_not_depend_on_seed():
    from conga.tcrdist.vectorized import EncodingConfig, aa_embedding
    a = aa_embedding(EncodingConfig(random_seed=SEED))
    for seed in [0, OTHER_SEED]:
        assert np.array_equal(a, aa_embedding(EncodingConfig(random_seed=seed)))


# CLI: --random_seed is no longer treated as a vectorized-encoding flag

def _run_cli(args, tmp_path):
    cmd = [sys.executable, str(RUN_CONGA_SCRIPT),
           '--gex_data', str(tmp_path/'missing.h5ad'), '--gex_data_type', 'h5ad',
           '--clones_file', str(tmp_path/'missing.tsv'), '--organism', 'human',
           '--outfile_prefix', str(tmp_path/'out')] + args
    return subprocess.run(cmd, capture_output=True, text=True,
                          cwd=str(REPO_ROOT), timeout=300)


@pytest.mark.parametrize('path_flag', ['--no_kpca', '--use_exact_tcrdist_nbrs',
                                       '--use_kpca_tcrdist'])
def test_cli_accepts_random_seed_on_non_vectorized_paths(path_flag, tmp_path):
    # the input files don't exist, so the run fails later on; we only check
    # that it gets past the flag-conflict checks
    p = _run_cli([path_flag, '--random_seed', '7'], tmp_path)
    output = p.stdout + p.stderr
    assert 'conflicts with encoding flags' not in output, output
    assert 'random_seed: 7' in (tmp_path/'out_log.txt').read_text()


def test_cli_still_rejects_encoding_flags_on_exact_path(tmp_path):
    p = _run_cli(['--no_kpca', '--cdr3_weight', '2.0', '--random_seed', '7'],
                 tmp_path)
    assert p.returncode != 0
    assert ("--no_kpca conflicts with encoding flags: ['--cdr3_weight']"
            in p.stdout), p.stdout + p.stderr
