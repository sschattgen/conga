"""
End-to-end `Reproducibility_Test` for the pipeline-reproducibility feature
(Requirements 6-7).

This is the capstone test proving that the seed wiring added across Tasks
1-8 (every `random_state`/`random_seed` call site threaded through
`conga/preprocess.py` and `scripts/run_conga.py`) actually makes two full
`scripts/run_conga.py` invocations with the same `--random_seed` produce
identical `_final.h5ad` output.

Fixture: a new, module-local, session-scoped fixture (not a reuse of
`tests/fixtures/test_data_generator.py`'s `minimal_adata`/`minimal_clones`,
which lack sufficient per-gene expression variance and crash in
`sc.pp.regress_out` -- see design.md's Testing Strategy section). A
`numpy.random.default_rng(42)`-seeded synthetic GEX count matrix is built
from per-gene lognormal means modulated by a small number of latent
"programs" (so PCA/HVG selection has real structure to find), plus
synthetic `va`/`ja`/`cdr3a`/`cdr3a_nucseq`/`vb`/`jb`/`cdr3b`/`cdr3b_nucseq`
TCR columns written directly into `adata.obs`, satisfying
`conga.preprocess.read_dataset`'s `clones_file=None` early-return branch
(`gex_data_type == 'h5ad'` and the TCR columns already present in
`adata.obs`).

Three Pipeline_Run configurations, each run twice with an identical
`--random_seed 42` and distinct `--outfile_prefix` locations:

    1. Default / vectorized-or-KernelPCA path (Requirement 7.1).
    2. Exact-TCRdist path via `--no_kpca` (Requirement 7.2).
    3. Seeded analyses: `--tcr_clumping --graph_vs_graph_stats
       --match_to_tcr_database --make_tcrdist_trees`, comparing the files
       those analyses write as well as the final h5ad (configurations 1 and
       2 run none of the analyses that take a seed).

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 7.1, 7.2
"""

import subprocess
import sys
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import pytest

from conga import util


REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_CONGA_SCRIPT = REPO_ROOT / 'scripts' / 'run_conga.py'
RANDOM_SEED = 42

N_CELLS = 60
N_GENES = 500
N_PROGRAMS = 4  # small number of latent GEX "programs" so PCA/HVG have
                 # real structure to find, per design.md's feasibility probe
N_CLONES = 40   # distinct TCR identities; kept well below N_CELLS so some
                 # clones have >1 cell (clonal expansion) while still
                 # leaving enough distinct clones post-dedup for the
                 # downstream k-nearest-neighbors steps (num_nbrs=10) to
                 # find enough neighbors

# Real human V/J alleles + CDR3 sequences, same source list used by
# tests/test_pipeline_reproducibility_component2.py / component3.py's
# _VA_JA_CDR3A / _VB_JB_CDR3B fixtures (in turn following
# tests/test_metaconga_match_dispatch.py's Representative_Human_Fixture
# convention).
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

# EXCLUDED_FIELDS (Requirement 6.5): fields found during implementation to
# legitimately vary between two same-seed Pipeline_Run invocations despite
# the Task 1-8 wiring being correctly applied. Per design.md's design-time
# verification, no field was expected to vary. This set is populated only
# if this test's own run against the real fixture finds otherwise, with an
# inline comment naming the field and the observed cause -- never silently
# loosened to an approximate comparison.
EXCLUDED_FIELDS: set[str] = set()


def _run_cli(args, cwd=None, timeout=300):
    """Invoke `scripts/run_conga.py` as a subprocess, following
    `tests/test_run_conga_cli.py`'s `_run_cli` helper exactly (same
    `sys.executable`, `capture_output=True, text=True`, `cwd=REPO_ROOT`),
    with `timeout` raised to 300s since this runs the full pipeline rather
    than only argument-validation paths.
    """
    cmd = [sys.executable, str(RUN_CONGA_SCRIPT)] + args
    return subprocess.run(cmd, capture_output=True, text=True,
                           cwd=cwd or str(REPO_ROOT), timeout=timeout)


def _make_gex_counts(n_cells: int, n_genes: int, n_programs: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Build a synthetic GEX count matrix with real per-gene expression
    structure: each gene is assigned to one of `n_programs` latent
    "programs", each cell has a per-program activity level, and per-gene
    lognormal means are modulated by the activity of the cell in that
    gene's program. This gives `sc.pp.highly_variable_genes`/PCA real
    variance structure to find, unlike a flat `Poisson(lam=constant)`
    matrix (confirmed during design-time feasibility testing to retain
    zero variable genes and crash in `sc.pp.regress_out`).
    """
    # assign each gene to a program, and each cell a per-program activity
    gene_program = rng.integers(0, n_programs, size=n_genes)
    cell_program_activity = rng.lognormal(
        mean=0.0, sigma=0.6, size=(n_cells, n_programs))

    # baseline per-gene mean expression level
    base_gene_mean = rng.lognormal(mean=1.5, sigma=1.0, size=n_genes)

    # modulate each gene's mean by its program's activity in each cell
    program_modulation = cell_program_activity[:, gene_program]  # (cells, genes)
    gene_means = base_gene_mean[np.newaxis, :] * program_modulation

    counts = rng.poisson(lam=gene_means).astype(np.float64)
    return counts


def _make_distinct_clones(n_clones: int) -> list:
    """Build `n_clones` distinct (va, ja, cdr3a, vb, jb, cdr3b) tuples by
    pairing each of the real V/J/CDR3a and V/J/CDR3b base entries from
    `_VA_JA_CDR3A`/`_VB_JB_CDR3B` combinatorially, and -- once the
    `len(_VA_JA_CDR3A) * len(_VB_JB_CDR3B)` combinations are exhausted --
    varying the alpha-chain CDR3 with a trailing amino-acid substitution
    so every clone has a genuinely distinct CDR3 (needed so that
    `reduce_to_single_cell_per_clone`'s dedup-by-TCR-identity step
    actually retains `n_clones` distinct cells rather than collapsing
    cells sharing the same (atcr, btcr) pair into far fewer clones than
    the fixture's cell count, which is too small for the pipeline's
    k-nearest-neighbors steps downstream).
    """
    base_combos = [
        (va, ja, cdr3a, vb, jb, cdr3b)
        for va, ja, cdr3a in _VA_JA_CDR3A
        for vb, jb, cdr3b in _VB_JB_CDR3B
    ]
    # amino acids safe to append without colliding with CDR3-boundary
    # motifs (the base CDR3s already start with C and end with F/W)
    _VARIANTS = 'AGSTNQDEKRHVLIMPWYC'
    clones = []
    variant_idx = 0
    while len(clones) < n_clones:
        for va, ja, cdr3a, vb, jb, cdr3b in base_combos:
            if len(clones) >= n_clones:
                break
            if variant_idx == 0:
                clones.append((va, ja, cdr3a, vb, jb, cdr3b))
            else:
                suffix = _VARIANTS[variant_idx % len(_VARIANTS)]
                clones.append(
                    (va, ja, cdr3a[:-1] + suffix + cdr3a[-1],
                     vb, jb, cdr3b[:-1] + suffix + cdr3b[-1]))
        variant_idx += 1
    return clones[:n_clones]


def _make_tcr_columns(n_cells: int, n_clones: int,
                       rng: np.random.Generator) -> pd.DataFrame:
    """Build synthetic va/ja/cdr3a/cdr3a_nucseq/vb/jb/cdr3b/cdr3b_nucseq
    TCR columns using real human V/J gene names and CDR3 sequences,
    distributing `n_cells` cells across `n_clones` distinct TCRs so each
    clone has multiple cells (clonal expansion), matching the pattern
    established in test_pipeline_reproducibility_component3.py.
    """
    clones = _make_distinct_clones(n_clones)
    clone_idx_for_cell = rng.integers(0, n_clones, size=n_cells)
    rows = []
    for clone_idx in clone_idx_for_cell:
        va, ja, cdr3a, vb, jb, cdr3b = clones[clone_idx]
        rows.append(dict(
            va=va, ja=ja, cdr3a=cdr3a, cdr3a_nucseq='acgt',
            vb=vb, jb=jb, cdr3b=cdr3b, cdr3b_nucseq='acgt',
        ))
    return pd.DataFrame(rows)


def _build_fixture_adata() -> ad.AnnData:
    rng = np.random.default_rng(RANDOM_SEED)
    counts = _make_gex_counts(N_CELLS, N_GENES, N_PROGRAMS, rng)

    barcodes = [f'CELL{i}-1' for i in range(N_CELLS)]
    var = pd.DataFrame(index=[f'GENE{i}' for i in range(N_GENES)])

    tcr_df = _make_tcr_columns(N_CELLS, n_clones=N_CLONES, rng=rng)
    tcr_df.index = barcodes

    adata = ad.AnnData(X=sp.csr_matrix(counts), obs=tcr_df, var=var)
    adata.obs_names = barcodes
    return adata


@pytest.fixture(scope='session')
def fixture_h5ad_path(tmp_path_factory):
    """Session-scoped fixture `.h5ad` file, built once per test session
    and shared read-only across all Pipeline_Run subprocess invocations.
    """
    tmp_dir = tmp_path_factory.mktemp('pipeline_reproducibility_fixture')
    h5ad_path = tmp_dir / 'fixture.h5ad'
    adata = _build_fixture_adata()
    adata.write_h5ad(h5ad_path)
    return h5ad_path


# Analysis-configuration fixture (third Pipeline_Run configuration).
#
# `--tcr_clumping` and `--match_to_tcr_database` only report TCRs that are
# unusually close to other TCRs / to a literature-database TCR, so the
# random-clone fixture above (no database TCRs, no real clumps) yields empty
# result tables, which would make a double-run comparison vacuous. This
# fixture therefore mixes in (a) TCRs copied from the shipped
# conga/data/new_paired_tcr_db_for_matching_nr.tsv (exact database matches)
# and (b) a "clump" of near-identical TCRs, alongside the random clones.
# Both analyses run `tcr_sampler.analyze_junction`, which needs a
# nucleotide CDR3 whose length is 3 * len(cdr3), hence the back-translation.
_DB_TCRS = [  # (va, ja, cdr3a, vb, jb, cdr3b), rows of the shipped database
    ('TRAV35*01', 'TRAJ49*01', 'CAGGTGNQFYF', 'TRBV10-3*01', 'TRBJ1-5*01', 'CAISEVGVGQPQHF'),
    ('TRAV1-2*01', 'TRAJ34*01', 'CAVRDPGNTDKLIF', 'TRBV20-1*01', 'TRBJ2-4*01', 'CSARASGGEAKNIQYF'),
    ('TRAV2*01', 'TRAJ28*01', 'CAVSRAGAGSYQLTF', 'TRBV20-1*01', 'TRBJ2-4*01', 'CSARAGGGEAKNIQYF'),
    ('TRAV12-2*01', 'TRAJ45*01', 'CAVSGGGADGLTF', 'TRBV28*01', 'TRBJ2-3*01', 'CASSFLGTGVEQYF'),
    ('TRAV12-3*01', 'TRAJ12*01', 'CAMRGDSSYKLIF', 'TRBV12-4*01', 'TRBJ2-4*01', 'CASSLWEKLAKNIQYF'),
    ('TRAV26-1*01', 'TRAJ32*01', 'CIVWGGATNKLIF', 'TRBV7-2*01', 'TRBJ2-3*01', 'CASSVRSTDTQYF'),
    ('TRAV26-1*01', 'TRAJ45*01', 'CIVLGGADGLTF', 'TRBV7-2*01', 'TRBJ2-3*01', 'CASSFRFTDTQYF'),
    ('TRAV3*01', 'TRAJ40*01', 'CAVSDITYKYIF', 'TRBV28*01', 'TRBJ2-2*01', 'CATGPRANTGELFF'),
]
# A "clump": six TCRs with the same V/J genes whose CDR3a and CDR3b each
# differ from one another by one or two residues (the first is also a row of
# the shipped database, so it doubles as a database match). CDR3a *and* CDR3b must both
# differ between members: `find_tcr_clumping` does not count TCRs that share
# an identical alpha or beta chain as neighbors.
_CLUMP_TCRS = [
    ('TRAV8-3*01', 'TRAJ21*01', cdr3a, 'TRBV28*01', 'TRBJ2-2*01', cdr3b)
    for cdr3a, cdr3b in [
        ('CAVGDNFNKFYF', 'CASSLTSAAGELFF'),
        ('CAVGDNYNKFYF', 'CASSLTSAAGEQFF'),
        ('CAVGENFNKFYF', 'CASSLTSATGELFF'),
        ('CAVGDNFSKFYF', 'CASSLASAAGELFF'),
        ('CAVADNFNKFYF', 'CASSLTSAAGDLFF'),
        ('CAVGDNFNRFYF', 'CASSVTSAAGELFF'),
    ]
]
N_ANALYSIS_RANDOM_CLONES = 24
N_ANALYSIS_CELLS = 2 * (len(_DB_TCRS) + len(_CLUMP_TCRS)
                        + N_ANALYSIS_RANDOM_CLONES)  # 2 cells per clone

# One fixed codon per amino acid, used only to back-translate the CDR3s.
_BACKTRANSLATE_CODONS = {
    'A': 'gcc', 'C': 'tgc', 'D': 'gac', 'E': 'gag', 'F': 'ttc', 'G': 'ggc',
    'H': 'cac', 'I': 'atc', 'K': 'aag', 'L': 'ctg', 'M': 'atg', 'N': 'aac',
    'P': 'ccc', 'Q': 'cag', 'R': 'cgg', 'S': 'agc', 'T': 'acc', 'V': 'gtg',
    'W': 'tgg', 'Y': 'tac',
}


def _backtranslate(cdr3: str) -> str:
    return ''.join(_BACKTRANSLATE_CODONS[aa] for aa in cdr3)


def _build_analysis_fixture_adata() -> ad.AnnData:
    rng = np.random.default_rng(RANDOM_SEED)
    counts = _make_gex_counts(N_ANALYSIS_CELLS, N_GENES, N_PROGRAMS, rng)
    barcodes = [f'CELL{i}-1' for i in range(N_ANALYSIS_CELLS)]
    var = pd.DataFrame(index=[f'GENE{i}' for i in range(N_GENES)])

    clones = (_DB_TCRS + _CLUMP_TCRS
              + _make_distinct_clones(N_ANALYSIS_RANDOM_CLONES))
    rows = []
    for cell_idx in range(N_ANALYSIS_CELLS):
        va, ja, cdr3a, vb, jb, cdr3b = clones[cell_idx % len(clones)]
        rows.append(dict(
            va=va, ja=ja, cdr3a=cdr3a, cdr3a_nucseq=_backtranslate(cdr3a),
            vb=vb, jb=jb, cdr3b=cdr3b, cdr3b_nucseq=_backtranslate(cdr3b),
        ))
    tcr_df = pd.DataFrame(rows, index=barcodes)

    adata = ad.AnnData(X=sp.csr_matrix(counts), obs=tcr_df, var=var)
    adata.obs_names = barcodes
    return adata


@pytest.fixture(scope='session')
def analysis_fixture_h5ad_path(tmp_path_factory):
    """Session-scoped `.h5ad` fixture for the analysis configuration (see
    the comment above `_DB_TCRS`)."""
    tmp_dir = tmp_path_factory.mktemp('pipeline_reproducibility_analysis_fixture')
    h5ad_path = tmp_dir / 'analysis_fixture.h5ad'
    _build_analysis_fixture_adata().write_h5ad(h5ad_path)
    return h5ad_path


def _base_args(fixture_h5ad_path: Path) -> list:
    return [
        '--gex_data', str(fixture_h5ad_path),
        '--gex_data_type', 'h5ad',
        '--organism', 'human',
        '--min_clones', '5',
        '--min_cells_after_subsetting', '5',
        '--random_seed', str(RANDOM_SEED),
    ]


def _run_pipeline_twice(fixture_h5ad_path: Path, tmp_path: Path,
                         extra_args: list, label: str):
    """Runs the same Pipeline_Run configuration twice with distinct
    `--outfile_prefix` locations and returns both resulting `_final.h5ad`
    paths as loaded AnnData objects. Fails loudly (Requirement 6.3) with
    captured stdout/stderr if either invocation exits nonzero.
    """
    adatas = []
    for run_idx in (1, 2):
        outfile_prefix = tmp_path / f'{label}_run{run_idx}'
        args = (_base_args(fixture_h5ad_path) +
                ['--outfile_prefix', str(outfile_prefix)] +
                extra_args)
        result = _run_cli(args)
        if result.returncode != 0:
            pytest.fail(
                f'{label} run {run_idx} exited with returncode '
                f'{result.returncode}.\n'
                f'--- stdout ---\n{result.stdout}\n'
                f'--- stderr ---\n{result.stderr}'
            )
        final_h5ad = Path(f'{outfile_prefix}_final.h5ad')
        assert final_h5ad.exists(), (
            f'{label} run {run_idx} did not produce {final_h5ad}.\n'
            f'--- stdout ---\n{result.stdout}\n'
            f'--- stderr ---\n{result.stderr}'
        )
        adatas.append(ad.read_h5ad(final_h5ad))
    return adatas[0], adatas[1]


# Comparable_Output fields (Requirement 6.2), confirmed present by direct
# inspection of this test's own fixture output (see
# TestComparableOutputFieldsPresent below) rather than transcribed
# unverified from design.md.
_OBS_COLUMNS = [
    'va', 'ja', 'cdr3a', 'cdr3a_nucseq', 'vb', 'jb', 'cdr3b', 'cdr3b_nucseq',
    'n_genes', 'percent_mito', 'n_counts', 'clone_sizes', 'gex_variation',
    'leiden_gex', 'clusters_gex', 'clusters_tcr', 'is_invariant',
    'nndists_gex', 'nndists_tcr',
]
_OBSM_ARRAYS_COMMON = [
    'X_gex_1d', 'X_gex_2d', 'X_pca_gex', 'X_tcr_1d', 'X_tcr_2d', 'X_umap_gex',
]
_OBSM_ARRAY_DEFAULT_PATH_ONLY = 'X_vec_tcr'
_OBSP_MATRICES = ['distances', 'connectivities']
_UNS_ARRAY_ENTRIES = ['clusters_tcr_names']
_UNS_DICT_ENTRIES_COMMON = ['conga_stats']
_UNS_DICT_ENTRY_DEFAULT_PATH_ONLY = 'vec_tcr_config'


def _assert_comparable_outputs_equal(adata1, adata2, *,
                                      include_vec_tcr_fields: bool):
    failures = []

    for col in _OBS_COLUMNS:
        if col in EXCLUDED_FIELDS:
            continue
        v1 = adata1.obs[col].values
        v2 = adata2.obs[col].values
        if not np.array_equal(v1, v2):
            failures.append(f"adata.obs['{col}'] differs between runs")

    obsm_keys = list(_OBSM_ARRAYS_COMMON)
    if include_vec_tcr_fields:
        obsm_keys.append(_OBSM_ARRAY_DEFAULT_PATH_ONLY)
    for key in obsm_keys:
        if key in EXCLUDED_FIELDS:
            continue
        a1 = np.asarray(adata1.obsm[key])
        a2 = np.asarray(adata2.obsm[key])
        if not np.array_equal(a1, a2):
            failures.append(f"adata.obsm['{key}'] differs between runs")

    for key in _OBSP_MATRICES:
        if key in EXCLUDED_FIELDS:
            continue
        m1 = adata1.obsp[key].toarray()
        m2 = adata2.obsp[key].toarray()
        if not np.array_equal(m1, m2):
            failures.append(f"adata.obsp['{key}'] differs between runs")

    for key in _UNS_ARRAY_ENTRIES:
        if key in EXCLUDED_FIELDS:
            continue
        if not np.array_equal(
                np.asarray(adata1.uns[key]), np.asarray(adata2.uns[key])):
            failures.append(f"adata.uns['{key}'] differs between runs")

    uns_dict_keys = list(_UNS_DICT_ENTRIES_COMMON)
    if include_vec_tcr_fields:
        uns_dict_keys.append(_UNS_DICT_ENTRY_DEFAULT_PATH_ONLY)
    for key in uns_dict_keys:
        if key in EXCLUDED_FIELDS:
            continue
        if adata1.uns[key] != adata2.uns[key]:
            failures.append(f"adata.uns['{key}'] differs between runs")

    assert not failures, 'Reproducibility mismatch(es):\n' + '\n'.join(failures)


class TestDefaultPathReproducibility:
    """Requirement 7.1: the default / vectorized-or-KernelPCA path."""

    def test_same_seed_runs_produce_identical_output(
            self, fixture_h5ad_path, tmp_path):
        adata1, adata2 = _run_pipeline_twice(
            fixture_h5ad_path, tmp_path, extra_args=[], label='default')
        _assert_comparable_outputs_equal(
            adata1, adata2, include_vec_tcr_fields=True)


class TestExactTcrdistPathReproducibility:
    """Requirement 7.2: the exact-TCRdist path via --no_kpca."""

    def test_same_seed_runs_produce_identical_output(
            self, fixture_h5ad_path, tmp_path):
        adata1, adata2 = _run_pipeline_twice(
            fixture_h5ad_path, tmp_path, extra_args=['--no_kpca'],
            label='no_kpca')
        _assert_comparable_outputs_equal(
            adata1, adata2, include_vec_tcr_fields=False)


# ---------------------------------------------------------------------------
# Configuration 3: the seeded analyses
# ---------------------------------------------------------------------------
ANALYSIS_ARGS = [
    '--tcr_clumping',
    '--graph_vs_graph_stats',
    '--match_to_tcr_database',
    '--make_tcrdist_trees',
]

# Known non-deterministic column (review finding #10, deliberately NOT fixed
# here): `_compute_graph_overlap_stats` stores `calculation_time`, the
# wall-clock seconds the computation took (`time.time() - starttime`), in
# the graph_vs_graph_stats table, `adata.uns['conga_results']` and
# `_graph_vs_graph_stats.tsv`. It differs between any two runs. Every other
# column, including `calculation_time_fitted` (a closed-form function of the
# node and edge counts), must match exactly, so nothing else is excluded.
WALL_CLOCK_GRAPH_STATS_COLUMNS = ['calculation_time']

# Files whose bytes depend only on the data and the seed.
_BYTE_COMPARED_SUFFIXES = ('.tsv', '_README.txt')
# Deliberately not byte-compared (their presence is still checked through the
# file listing): `*_log.txt` (hostname, absolute argv paths, wall-clock
# "run_conga took N minutes"), `*_results_summary.html` (absolute paths of the
# run directory plus embedded base64 images) and `*.h5ad` (compared field by
# field above, since h5ad bytes can vary with HDF5 write metadata). PNGs are
# compared as decoded pixel arrays rather than bytes so that metadata chunks
# written by the image library cannot cause a spurious difference.


def _run_analysis_pipeline_twice(h5ad_path: Path, tmp_path: Path):
    """Runs the analysis configuration twice, each run in its own directory
    (used as both cwd, so `tmp_nbrs*` scratch files stay inside `tmp_path`,
    and output directory, with the same file names `out_*` in both so that
    file listings and README texts are comparable). Returns the two run
    directories.
    """
    run_dirs = []
    for run_idx in (1, 2):
        run_dir = tmp_path / f'analysis_run{run_idx}'
        run_dir.mkdir()
        args = (_base_args(h5ad_path) +
                ['--outfile_prefix', str(run_dir / 'out')] + ANALYSIS_ARGS)
        result = _run_cli(args, cwd=str(run_dir))
        if result.returncode != 0:
            pytest.fail(
                f'analysis run {run_idx} exited with returncode '
                f'{result.returncode}.\n'
                f'--- stdout ---\n{result.stdout}\n'
                f'--- stderr ---\n{result.stderr}'
            )
        assert (run_dir / 'out_final.h5ad').exists(), result.stdout
        run_dirs.append(run_dir)
    return run_dirs[0], run_dirs[1]


def _assert_analysis_outputs_equal(adata1, adata2, dir1, dir2):
    """Compares everything the analysis flags produce: the final AnnData
    (including the result tables stored in `uns['conga_results']`), the
    result files in the output directories and the rendered tree/clumping
    figures.
    """
    failures = []

    # --- AnnData: the standard comparable fields plus analysis results -----
    try:
        _assert_comparable_outputs_equal(
            adata1, adata2, include_vec_tcr_fields=True)
    except AssertionError as e:
        failures.append(str(e))
    if not np.array_equal(adata1.obs['tcr_clumping_pvalues'].values,
                          adata2.obs['tcr_clumping_pvalues'].values):
        failures.append("adata.obs['tcr_clumping_pvalues'] differs")
    for key in ('tcr_clumping', 'graph_vs_graph_stats', 'tcr_db_match'):
        df1 = adata1.uns['conga_results'][key].copy()
        df2 = adata2.uns['conga_results'][key].copy()
        if key == 'graph_vs_graph_stats':
            df1 = df1.drop(columns=WALL_CLOCK_GRAPH_STATS_COLUMNS)
            df2 = df2.drop(columns=WALL_CLOCK_GRAPH_STATS_COLUMNS)
        try:
            pd.testing.assert_frame_equal(df1, df2, check_exact=True)
        except AssertionError as e:
            failures.append(f"adata.uns['conga_results']['{key}'] differs: {e}")

    # --- files in the output directories ------------------------------------
    names1 = sorted(p.name for p in dir1.iterdir())
    names2 = sorted(p.name for p in dir2.iterdir())
    if names1 != names2:
        failures.append(f'output file listings differ: {names1} vs {names2}')
    for name in names1:
        if name not in names2:
            continue
        f1, f2 = dir1 / name, dir2 / name
        if name == 'out_graph_vs_graph_stats.tsv':
            t1 = pd.read_csv(f1, sep='\t').drop(
                columns=WALL_CLOCK_GRAPH_STATS_COLUMNS)
            t2 = pd.read_csv(f2, sep='\t').drop(
                columns=WALL_CLOCK_GRAPH_STATS_COLUMNS)
            try:
                pd.testing.assert_frame_equal(t1, t2, check_exact=True)
            except AssertionError as e:
                failures.append(f'{name} differs: {e}')
        elif name.endswith(_BYTE_COMPARED_SUFFIXES):
            if f1.read_bytes() != f2.read_bytes():
                failures.append(f'{name} differs byte for byte')
        elif name.endswith('.png'):
            if not np.array_equal(mpimg.imread(f1), mpimg.imread(f2)):
                failures.append(f'{name} differs pixel for pixel')

    assert not failures, 'Reproducibility mismatch(es):\n' + '\n'.join(failures)


def _assert_analyses_produced_output(adata, run_dir):
    """Guards against the comparison passing vacuously because an analysis
    silently did nothing (e.g. an empty clumping table)."""
    results = adata.uns['conga_results']

    clumping = results['tcr_clumping']
    assert len(clumping) > 0, 'tcr_clumping produced no clumps'
    clump_cdr3bs = {t[5] for t in _CLUMP_TCRS}
    assert set(clumping['cdr3b']) & clump_cdr3bs, (
        'tcr_clumping did not find the planted clump')
    assert (adata.obs['tcr_clumping_pvalues'] < adata.n_obs).any()

    assert len(results['graph_vs_graph_stats']) > 0

    db_match = results['tcr_db_match']
    assert len(db_match) > 0, 'match_to_tcr_database found no matches'
    assert set(db_match['cdr3b']) & {t[5] for t in _DB_TCRS}

    # the .tsv files on disk are non-empty tables too (an empty table is
    # written as a single-byte file)
    for name in ('out_tcr_clumping.tsv', 'out_tcr_db_match.tsv',
                 'out_graph_vs_graph_stats.tsv'):
        assert len(pd.read_csv(run_dir / name, sep='\t')) > 0, name

    # the tcrdist tree figure exists and is not a blank canvas
    trees_png = run_dir / 'out_gex_clusters_tcrdist_trees.png'
    assert trees_png.exists()
    assert (run_dir / 'out_gex_clusters_tcrdist_trees.png_README.txt').exists()
    pixels = mpimg.imread(trees_png)
    assert len(np.unique(pixels.reshape(-1, pixels.shape[-1]), axis=0)) > 2


@pytest.mark.skipif(
    not util.tcrdist_cpp_available(),
    reason='needs compiled tcrdist_cpp (make in conga/tcrdist_cpp)')
class TestSeededAnalysesReproducibility:
    """Third configuration: the analyses that draw random numbers."""

    def test_same_seed_runs_produce_identical_analysis_output(
            self, analysis_fixture_h5ad_path, tmp_path):
        dir1, dir2 = _run_analysis_pipeline_twice(
            analysis_fixture_h5ad_path, tmp_path)
        adata1 = ad.read_h5ad(dir1 / 'out_final.h5ad')
        adata2 = ad.read_h5ad(dir2 / 'out_final.h5ad')
        _assert_analyses_produced_output(adata1, dir1)
        _assert_analyses_produced_output(adata2, dir2)
        _assert_analysis_outputs_equal(adata1, adata2, dir1, dir2)
