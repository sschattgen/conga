"""
Tests for how scripts/setup_10x_for_conga.py handles --random_seed and the
vectorized-encoding flags.

  * --random_seed is passed on to the kernel PCA (the only random step in
    setup), and defaults to util.DEFAULT_RANDOM_SEED.
  * --aa_mds_dim, --num_pos_cdr3 and --cdr3_weight are rejected, because setup
    never builds the vectorized TCR representation and used to silently ignore
    them. --random_seed is not one of them.
  * Pin: the kernel PCs that setup writes don't depend on the seed (sklearn's
    dense solver is deterministic).

The script runs its argparse code at module level, so the seed-threading tests
run it in-process with runpy (so the kernel PCA function can be replaced by a
spy) and the rejection tests run it as a subprocess, as in
tests/test_organism_cli_choices.py.
"""

import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conga import util

REPO_ROOT = Path(__file__).resolve().parent.parent
SETUP_10X_SCRIPT = REPO_ROOT / 'scripts' / 'setup_10x_for_conga.py'
MINIMAL_CLONES = REPO_ROOT / 'tests' / 'fixtures' / 'minimal_clones.tsv'

ENCODING_FLAGS = [['--aa_mds_dim', '8'],
                  ['--num_pos_cdr3', '12'],
                  ['--cdr3_weight', '2.0']]


def _minimal_tcrs(n):
    """First n rows of the fixture, as (alpha, beta) tcr tuples."""
    df = pd.read_csv(MINIMAL_CLONES, sep='\t').iloc[:n]
    return [((l.va, l.ja, l.cdr3a), (l.vb, l.jb, l.cdr3b))
            for l in df.itertuples()]


@pytest.fixture
def clones_file(tmp_path):
    """A 5-row clones file in the format setup reads with --input_clones_file"""
    df = pd.read_csv(MINIMAL_CLONES, sep='\t').iloc[:5]
    clones = pd.DataFrame({
        'clone_id': [f'clone{i}' for i in range(len(df))],
        'va_gene': df.va, 'ja_gene': df.ja, 'cdr3a': df.cdr3a,
        'vb_gene': df.vb, 'jb_gene': df.jb, 'cdr3b': df.cdr3b,
    })
    path = tmp_path / 'clones.tsv'
    clones.to_csv(path, sep='\t', index=False)
    return path


def _run_setup_in_process(args, monkeypatch, tmp_path):
    """Runs the setup script in-process with the kernel PCA function replaced
    by a spy. Returns the list of kwargs it was called with."""
    import conga.preprocess
    calls = []
    def spy(*args, **kwargs):
        calls.append(kwargs)
    monkeypatch.setattr(
        conga.preprocess, 'make_tcrdist_kernel_pcs_file_from_clones_file', spy)
    monkeypatch.chdir(tmp_path) # so nothing lands in the repo
    monkeypatch.setattr(sys, 'argv', [str(SETUP_10X_SCRIPT)] + args)
    runpy.run_path(str(SETUP_10X_SCRIPT), run_name='__main__')
    return calls


def _run_setup(args, tmp_path):
    cmd = [sys.executable, str(SETUP_10X_SCRIPT)] + args
    return subprocess.run(cmd, capture_output=True, text=True,
                          cwd=str(tmp_path), timeout=300)


def test_seed_is_threaded_to_kpca(clones_file, monkeypatch, tmp_path):
    calls = _run_setup_in_process(
        ['--input_clones_file', str(clones_file), '--organism', 'human',
         '--use_kpca_tcrdist', '--random_seed', '7'], monkeypatch, tmp_path)
    assert len(calls) == 1
    assert calls[0]['random_seed'] == 7


def test_default_seed_is_threaded_to_kpca(clones_file, monkeypatch, tmp_path):
    calls = _run_setup_in_process(
        ['--input_clones_file', str(clones_file), '--organism', 'human',
         '--use_kpca_tcrdist'], monkeypatch, tmp_path)
    assert len(calls) == 1
    assert calls[0]['random_seed'] == util.DEFAULT_RANDOM_SEED


@pytest.mark.parametrize('flag', ENCODING_FLAGS, ids=lambda f: f[0])
def test_encoding_flags_rejected(flag, tmp_path):
    p = _run_setup(['--organism', 'human'] + flag, tmp_path)
    assert p.returncode == 1, p.stdout + p.stderr
    assert 'encoding flags' in p.stderr
    assert flag[0] in p.stderr


@pytest.mark.parametrize('key,value', [('aa_mds_dim', 8), ('num_pos_cdr3', 12),
                                       ('cdr3_weight', 2.0)])
def test_encoding_flags_rejected_from_config_yaml(key, value, tmp_path):
    config = tmp_path / 'config.yml'
    config.write_text(f'organism: human\n{key}: {value}\n')
    p = _run_setup(['--config', str(config)], tmp_path)
    assert p.returncode == 1, p.stdout + p.stderr
    assert 'encoding flags' in p.stderr
    assert f'--{key}' in p.stderr


def test_random_seed_is_not_an_encoding_flag(tmp_path):
    # the input file doesn't exist, so the run fails later on; we only check
    # that the seed isn't rejected as an encoding flag
    p = _run_setup(['--organism', 'human', '--random_seed', '7',
                    '--input_clones_file', str(tmp_path / 'missing.tsv')],
                   tmp_path)
    assert 'encoding flags' not in p.stdout + p.stderr


def test_kpca_output_independent_of_seed():
    # Pins the claim in the --random_seed help text. If this fails, the solver
    # or n_components changed (sklearn's 'auto' picks the randomized-start
    # arpack solver for n_components < 10 on more than 200 TCRs) and the seed
    # now matters, so setup and run_conga.py should be checked for a mismatch.
    from conga.preprocess import make_tcrdist_kernel_pcs_file_from_clones_file
    tcrs = _minimal_tcrs(30)
    assert len(tcrs) == 30
    pcs = [make_tcrdist_kernel_pcs_file_from_clones_file(
               None, 'human', tcrs=tcrs, return_pcs=True, random_seed=seed)
           for seed in [util.DEFAULT_RANDOM_SEED, 7]]
    assert pcs[0].shape == (30, 30)
    assert np.array_equal(pcs[0], pcs[1])
