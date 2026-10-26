"""
Tests for how scripts/run_conga.py handles --random_seed.

  * Range validation: np.random.seed (needed by seaborn's jitter, see
    util.seeded_global_np_random) only accepts 0 <= seed < 2**32, so the script
    rejects anything else up front, naming the flag and the valid range,
    instead of failing late inside the QC-violin block. Covers the command
    line and a seed given in a --config yaml file.
  * Batch pathway: the seed resolved from --random_seed has to reach
    conga.preprocess.batch_integration (it used to be dropped, so the batch
    pathway ran under util.DEFAULT_RANDOM_SEED while the log recorded the CLI
    seed). The script runs its code at module level, so this runs it in-process
    with runpy and a spy in place of batch_integration.
"""
import runpy
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from conga import util

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_CONGA_SCRIPT = REPO_ROOT / 'scripts' / 'run_conga.py'

OUT_OF_RANGE_SEEDS = ['-1', '4294967296']
IN_RANGE_SEEDS = [0, 7, 4294967295]


class _StopAtBatchIntegration(Exception):
    """Raised by the spy so the script stops right after the call site."""


def _run_cli(args, cwd):
    cmd = [sys.executable, str(RUN_CONGA_SCRIPT)] + args
    return subprocess.run(cmd, capture_output=True, text=True,
                          cwd=str(cwd), timeout=300)


@pytest.mark.parametrize('seed', OUT_OF_RANGE_SEEDS)
def test_out_of_range_seed_rejected(seed, tmp_path):
    p = _run_cli(['--outfile_prefix', str(tmp_path / 'out'),
                  '--random_seed', seed], tmp_path)
    assert p.returncode == 1, p.stdout + p.stderr
    assert '--random_seed' in p.stderr
    assert '[0, 4294967295]' in p.stderr
    assert f'got {seed}' in p.stderr


@pytest.mark.parametrize('seed', OUT_OF_RANGE_SEEDS + ['7.5', 'True'])
def test_out_of_range_or_non_integer_seed_rejected_from_config_yaml(
        seed, tmp_path):
    config = tmp_path / 'config.yml'
    config.write_text(f'outfile_prefix: {tmp_path / "out"}\n'
                      f'random_seed: {seed}\n')
    p = _run_cli(['--config', str(config)], tmp_path)
    assert p.returncode == 1, p.stdout + p.stderr
    assert '--random_seed' in p.stderr
    assert '[0, 4294967295]' in p.stderr


@pytest.mark.parametrize('seed', IN_RANGE_SEEDS)
def test_in_range_seed_not_rejected(seed, tmp_path):
    # the gex file does not exist, so the run fails later on; we only check
    # that the seed passed validation
    p = _run_cli(['--outfile_prefix', str(tmp_path / 'out'),
                  '--gex_data', str(tmp_path / 'missing.h5ad'),
                  '--random_seed', str(seed)], tmp_path)
    assert 'must be an integer' not in p.stdout + p.stderr


def _multi_batch_adata():
    """Tiny stand-in for what read_dataset returns; only has to get the script
    up to the batch_integration call site."""
    n = 12
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({'cdr3a': ['CAAAF'] * n,
                        'batch': ['a', 'b'] * (n // 2)},
                       index=[f'cell{i}' for i in range(n)])
    return ad.AnnData(X=rng.poisson(1.0, (n, 5)).astype(np.float32), obs=obs)


@pytest.mark.parametrize('seed', IN_RANGE_SEEDS)
def test_batch_integration_receives_cli_seed(seed, monkeypatch, tmp_path):
    import conga.preprocess
    gex_file = tmp_path / 'gex.h5ad'
    gex_file.write_text('') # only has to exist, read_dataset is replaced
    calls = []

    def fake_batch_integration(adata, *args, **kwargs):
        calls.append(kwargs)
        raise _StopAtBatchIntegration()

    monkeypatch.setattr(conga.preprocess, 'read_dataset',
                        lambda *a, **k: _multi_batch_adata())
    monkeypatch.setattr(conga.preprocess, 'batch_integration',
                        fake_batch_integration)
    monkeypatch.chdir(tmp_path) # so nothing lands in the repo
    monkeypatch.setattr(sys, 'argv', [
        str(RUN_CONGA_SCRIPT), '--outfile_prefix', str(tmp_path / 'out'),
        '--gex_data', str(gex_file), '--organism', 'human',
        '--min_clones', '1', '--batch_key', 'batch',
        '--batch_integration_method', 'harmony', '--random_seed', str(seed)])
    with pytest.raises(_StopAtBatchIntegration):
        runpy.run_path(str(RUN_CONGA_SCRIPT), run_name='__main__')
    assert len(calls) == 1
    assert calls[0]['random_seed'] == seed
    assert calls[0]['batch_key'] == 'batch'


def test_batch_integration_receives_default_seed(monkeypatch, tmp_path):
    import conga.preprocess
    gex_file = tmp_path / 'gex.h5ad'
    gex_file.write_text('')
    calls = []

    def fake_batch_integration(adata, *args, **kwargs):
        calls.append(kwargs)
        raise _StopAtBatchIntegration()

    monkeypatch.setattr(conga.preprocess, 'read_dataset',
                        lambda *a, **k: _multi_batch_adata())
    monkeypatch.setattr(conga.preprocess, 'batch_integration',
                        fake_batch_integration)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', [
        str(RUN_CONGA_SCRIPT), '--outfile_prefix', str(tmp_path / 'out'),
        '--gex_data', str(gex_file), '--organism', 'human',
        '--min_clones', '1', '--batch_key', 'batch',
        '--batch_integration_method', 'harmony'])
    with pytest.raises(_StopAtBatchIntegration):
        runpy.run_path(str(RUN_CONGA_SCRIPT), run_name='__main__')
    assert calls[0]['random_seed'] == util.DEFAULT_RANDOM_SEED
class _StopAtCalcNbrs(Exception):
    """Raised by the spy so the script stops right after the call site."""
def _restart_adata():
    """Tiny stand-in for a saved conga h5ad (--restart): already has the TCR
    umap/clusters, so the script goes straight on to calc_nbrs."""
    n = 12
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({'clusters_tcr': ['0'] * n},
                       index=[f'cell{i}' for i in range(n)])
    adata = ad.AnnData(X=rng.poisson(1.0, (n, 5)).astype(np.float32), obs=obs)
    adata.obsm['X_tcr_2d'] = rng.normal(size=(n, 2))
    adata.uns['organism'] = 'human'
    return adata
@pytest.mark.parametrize('seed', [None, 7, 4294967295])
def test_calc_nbrs_receives_cli_seed(seed, monkeypatch, tmp_path):
    import scanpy as sc
    import conga.preprocess
    restart_file = tmp_path / 'restart.h5ad'
    restart_file.write_text('') # only has to exist, read_h5ad is replaced
    calls = []
    def fake_calc_nbrs(adata, *args, **kwargs):
        calls.append(kwargs)
        raise _StopAtCalcNbrs()
    monkeypatch.setattr(sc, 'read_h5ad', lambda *a, **k: _restart_adata())
    monkeypatch.setattr(conga.preprocess, 'calc_nbrs', fake_calc_nbrs)
    monkeypatch.chdir(tmp_path)
    argv = [str(RUN_CONGA_SCRIPT), '--outfile_prefix', str(tmp_path / 'out'),
            '--restart', str(restart_file), '--use_exact_tcrdist_nbrs']
    if seed is not None:
        argv += ['--random_seed', str(seed)]
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(_StopAtCalcNbrs):
        runpy.run_path(str(RUN_CONGA_SCRIPT), run_name='__main__')
    assert len(calls) == 1
    expected = util.DEFAULT_RANDOM_SEED if seed is None else seed
    assert calls[0]['random_seed'] == expected
