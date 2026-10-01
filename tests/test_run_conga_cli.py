"""
Tests for the `--batch_key`/`--batch_integration_method` CLI flags added to
`scripts/run_conga.py` by the batch-integration feature.

Per the batch-integration design (Component 4) and Requirement 6, these
tests cover:

    - Flag-pairing validation: `--batch_integration_method` without
      `--batch_key`, and vice versa, both exit nonzero naming both flags
      (Requirement 6.3).
    - Mutual-exclusion validation: `--force_variable_genes` together with
      either new flag exits nonzero naming the conflicting flags
      (Requirement 6.5).
    - Method-value validation: an unsupported `--batch_integration_method`
      value exits nonzero naming the supplied value and the supported set
      (Requirement 6.6).
    - A smoke-level integration test that the CLI reaches
      `conga.preprocess.batch_integration()` and that call succeeds when
      `--batch_key`/`--batch_integration_method=harmony` are both supplied.

Scope note for the integration test: a fully valid end-to-end
`scripts/run_conga.py` invocation requires a real `clones_file` (TCR V/J/CDR3
columns) plus a kernel-PCA TCR representation (`X_pca_tcr`) built from the
TCRdist pipeline, and runs the complete downstream CoNGA analysis pipeline
(neighbor graphs, clustering, UMAP, etc.) all the way to writing
`<outfile_prefix>_final.h5ad` -- machinery unrelated to batch integration
itself. Per the task's guidance to scope this down when the full pipeline is
disproportionate to the feature under test, `TestBatchIntegrationCliCallSite`
below reproduces the exact `if args.batch_key: ... else: ...` branch added
to `run_conga.py` at the task 8.3 call site against a minimal synthetic
multi-batch GEX `AnnData`, confirming `conga.preprocess.batch_integration()`
is reached and succeeds (producing `util.OBSM_KEY_PCA_GEX_UNINTEGRATED`,
`util.OBSM_KEY_PCA_GEX_INTEGRATED`, and `'X_pca_gex'`), plus a source-level
check that the real branch in `run_conga.py` matches. The CLI-subprocess
tests (flag pairing, mutual exclusion, method validation) do invoke the real
script end to end, since argument parsing and validation exit before any
TCR/GEX data loading is attempted.

Requirements: 6.3, 6.5, 6.6
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import pytest

from conga import util


REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_CONGA_SCRIPT = REPO_ROOT / 'scripts' / 'run_conga.py'
RANDOM_SEED = 42


def _run_cli(args, cwd=None):
    """Invoke `scripts/run_conga.py` as a subprocess via the conga-dev
    interpreter, matching the project's `mamba run -n conga-dev python ...`
    execution convention (the subprocess inherits `sys.executable`, which
    is the conga-dev interpreter when tests themselves are run that way).
    """
    cmd = [sys.executable, str(RUN_CONGA_SCRIPT)] + args
    return subprocess.run(cmd, capture_output=True, text=True,
                           cwd=cwd or str(REPO_ROOT), timeout=120)


def _minimal_required_args(tmp_path, outfile_prefix='tmp_cli_test'):
    """Minimal args every `run_conga.py` invocation needs regardless of
    which preprocessing pathway is requested (see the script's own
    epilog listing --gex_data/--gex_data_type/--clones_file/--organism/
    --outfile_prefix as the minimal command line).
    """
    return [
        '--outfile_prefix', str(tmp_path / outfile_prefix),
    ]


class TestBatchKeyFlagPairing:
    """Requirement 6.3: `--batch_key` and `--batch_integration_method`
    must be supplied together.
    """

    def test_batch_integration_method_without_batch_key_exits_nonzero(
            self, tmp_path):
        result = _run_cli(
            _minimal_required_args(tmp_path) +
            ['--batch_integration_method', 'harmony'])

        assert result.returncode != 0
        assert '--batch_key' in result.stdout + result.stderr
        assert '--batch_integration_method' in result.stdout + result.stderr

    def test_batch_key_without_batch_integration_method_exits_nonzero(
            self, tmp_path):
        result = _run_cli(
            _minimal_required_args(tmp_path) +
            ['--batch_key', 'donor_id'])

        assert result.returncode != 0
        assert '--batch_key' in result.stdout + result.stderr
        assert '--batch_integration_method' in result.stdout + result.stderr


class TestForceVariableGenesMutualExclusion:
    """Requirement 6.5: `--force_variable_genes` is mutually exclusive
    with `--batch_key`/`--batch_integration_method`.
    """

    def test_force_variable_genes_with_batch_key_exits_nonzero(
            self, tmp_path):
        gene_list_file = tmp_path / 'genes.txt'
        gene_list_file.write_text('GENE1\nGENE2\n')

        result = _run_cli(
            _minimal_required_args(tmp_path) +
            ['--force_variable_genes', str(gene_list_file),
             '--batch_key', 'donor_id',
             '--batch_integration_method', 'harmony'])

        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert '--force_variable_genes' in combined
        assert 'mutually exclusive' in combined.lower()

    def test_force_variable_genes_with_batch_integration_method_exits_nonzero(
            self, tmp_path):
        gene_list_file = tmp_path / 'genes.txt'
        gene_list_file.write_text('GENE1\nGENE2\n')

        # --batch_integration_method alone (no --batch_key) would already
        # fail the pairing check; supply both new flags so this test
        # exercises the force_variable_genes conflict specifically rather
        # than the pairing check from TestBatchKeyFlagPairing.
        result = _run_cli(
            _minimal_required_args(tmp_path) +
            ['--force_variable_genes', str(gene_list_file),
             '--batch_key', 'donor_id',
             '--batch_integration_method', 'scvi'])

        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert '--force_variable_genes' in combined
        assert 'mutually exclusive' in combined.lower()


class TestUnsupportedBatchIntegrationMethod:
    """Requirement 6.6: an unsupported `--batch_integration_method` value
    exits nonzero naming the supplied value and the supported set.
    """

    @pytest.mark.parametrize('method', ['scanorama', 'bbknn', 'not_a_method'])
    def test_unsupported_method_exits_nonzero(self, tmp_path, method):
        result = _run_cli(
            _minimal_required_args(tmp_path) +
            ['--batch_key', 'donor_id',
             '--batch_integration_method', method])

        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert method in combined
        assert 'harmony' in combined
        assert 'scvi' in combined


def _make_batch_adata(n_cells=80, n_genes=200, n_batches=2,
                       random_seed=RANDOM_SEED):
    """Build a minimal synthetic multi-batch GEX AnnData, following the
    fixture conventions in `tests/test_batch_integration.py` (sparse X,
    plain 'GENE{i}' names, a 'batch' obs column with a mild per-batch
    multiplicative shift so Harmony has something to correct).
    """
    rng = np.random.default_rng(random_seed)
    gene_means = rng.lognormal(mean=1.0, sigma=1.0, size=n_genes)
    counts = rng.poisson(lam=gene_means, size=(n_cells, n_genes)).astype(float)

    batch_labels = np.array([f'batch_{i % n_batches}' for i in range(n_cells)])
    rng.shuffle(batch_labels)
    shifted_gene_idx = rng.choice(
        n_genes, size=max(1, n_genes // 4), replace=False)
    for i, label in enumerate(batch_labels):
        batch_num = int(label.split('_')[1])
        if batch_num > 0:
            counts[i, shifted_gene_idx] *= (1.0 + 0.3 * batch_num)

    barcodes = [f'CELL{i}-1' for i in range(n_cells)]
    obs = pd.DataFrame({'batch': batch_labels}, index=barcodes)
    var = pd.DataFrame(index=[f'GENE{i}' for i in range(n_genes)])
    adata = ad.AnnData(X=sp.csr_matrix(counts), obs=obs, var=var)
    adata.uns['organism'] = 'human'
    return adata


@pytest.mark.integration
class TestBatchIntegrationCliCallSite:
    """Integration smoke test for the CLI call site added in task 8.3:
    confirms that, given `args.batch_key` set, `run_conga.py` calls
    `conga.preprocess.batch_integration()` (rather than
    `filter_and_scale()`) and that the call succeeds, producing the
    expected `obsm` keys.

    Scoped down per the module docstring: running `scripts/run_conga.py`
    fully end to end requires a real TCR `clones_file` plus a kernel-PCA
    TCR representation and runs the complete downstream CoNGA analysis
    pipeline (neighbor graphs, clustering, UMAP, etc.) -- none of which
    this feature touches or needs to validate. Instead, this test
    reproduces the exact call-site branch from `run_conga.py` (`if
    args.batch_key: adata = conga.preprocess.batch_integration(...) else:
    ...`) against a minimal synthetic multi-batch GEX AnnData, which is
    sufficient to confirm the wiring (Requirement 6.1, 6.2) without
    invoking unrelated TCR/VDJ machinery.
    """

    def test_batch_key_branch_calls_batch_integration(self):
        import conga.preprocess as preprocess

        pytest.importorskip(
            'harmonypy',
            reason='harmonypy not installed; install conga[batch-integration]'
            ' to run this smoke test')

        adata = _make_batch_adata()

        class _Args:
            batch_key = 'batch'
            batch_integration_method = 'harmony'
            max_genes_per_cell = None
            min_genes_per_cell = 1
            max_percent_mito = 1.0

        args = _Args()

        # This mirrors the exact branch wired into run_conga.py at the
        # task 8.3 call site. hvg_min_disp is loosened from the
        # batch_integration() default (0.5) to 0.1, matching the
        # convention established in tests/test_batch_integration.py's
        # `_run_filter_normalize_and_hvg_no_filtering` helper: batch-aware
        # HVG selection intersects per-batch HVG calls across all
        # batches, which is considerably stricter than the single-batch
        # case, and this project's synthetic random-count fixtures
        # otherwise frequently yield too few surviving genes for PCA.
        if args.batch_key:
            adata = preprocess.batch_integration(
                adata, batch_key=args.batch_key,
                method=args.batch_integration_method,
                n_gex_pcs=5,
                hvg_min_disp=0.1,
                max_genes_per_cell=args.max_genes_per_cell,
                min_genes_per_cell=args.min_genes_per_cell,
                max_percent_mito=args.max_percent_mito,
            )
        else:
            raise AssertionError('expected the batch_key branch to run')

        assert util.OBSM_KEY_PCA_GEX_UNINTEGRATED in adata.obsm
        assert util.OBSM_KEY_PCA_GEX_INTEGRATED in adata.obsm
        assert 'X_pca_gex' in adata.obsm
        np.testing.assert_array_equal(
            adata.obsm['X_pca_gex'],
            adata.obsm[util.OBSM_KEY_PCA_GEX_INTEGRATED])

    def test_call_site_source_wires_batch_integration_not_filter_and_scale(
            self):
        """Confirms the actual `run_conga.py` source contains the
        `if args.batch_key:` branch calling `batch_integration`, guarding
        against the wiring silently regressing even when this test
        module's reimplementation above still passes.
        """
        source = RUN_CONGA_SCRIPT.read_text()
        assert 'if args.batch_key:' in source
        assert 'conga.preprocess.batch_integration(' in source
        # the else branch must still call filter_and_scale for the
        # Fixed_HVG_Pathway / Default_Pathway
        branch_start = source.index('if args.batch_key:')
        branch_region = source[branch_start:branch_start + 1200]
        assert 'conga.preprocess.filter_and_scale(' in branch_region
