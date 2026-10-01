"""
Unit tests for `conga.preprocess._validate_batch_key`.

Per the batch-integration design and Requirement 1, `_validate_batch_key`
is the shared guard used by the Full_Integration_Pathway to confirm that
a caller-supplied `batch_key` is usable as the single column that drives
both batch-aware HVG selection and the Integration_Method:

    - Raises `ValueError` naming `batch_key` and the available
      `adata.obs.columns` when the column is absent (Requirement 1.4).
    - Raises `ValueError` naming `batch_key` and the actual distinct
      value count when `adata.obs[batch_key].nunique() < 2`, covering
      both the 0-unique (all-NaN/empty) and 1-unique-value cases
      (Requirement 1.5).
    - Raises no exception when the column has two or more distinct
      values.

This test module also covers the Harmony (task 5.3) and scVI (task 6.2)
Integration_Method tracks of `batch_integration()`. The scVI integration
test that actually trains a model is marked `slow` per this project's
`pyproject.toml` marker configuration; running it (and only it) requires
`scvi-tools` to be importable, which on this project's macOS/arm64
`conga-dev` environment additionally requires the `KMP_DUPLICATE_LIB_OK=TRUE`
environment variable to be set on the *invoking* shell before pytest
starts (see the development-workflow steering doc's environment note) --
this is a local OpenMP-conflict workaround, not something the test or
library code should set itself.

Requirements: 1.4, 1.5, 2.6, 3.4, 3.5, 4.1, 4.2
"""

import sys
from unittest import mock

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy.sparse as sp
import pytest

from conga.preprocess import (
    _validate_batch_key,
    _run_harmony_integration,
    _run_scvi_integration,
    batch_integration,
    cluster_and_tsne_and_umap,
    filter_normalize_and_hvg,
)
from conga import util


RANDOM_SEED = 42


def _make_adata(n_cells: int = 20, n_genes: int = 10,
                 random_seed: int = RANDOM_SEED) -> ad.AnnData:
    """Build a minimal synthetic AnnData for `_validate_batch_key` tests.

    `_validate_batch_key` only inspects `adata.obs`, so the expression
    matrix content is irrelevant; a small sparse `X` is used to match the
    fixture conventions used elsewhere in this test suite (e.g.
    `tests/test_counts_layer.py`, `tests/test_fixed_hvg_pathway.py`),
    with plain 'GENE{i}' names and no TR/IG/MT prefixes.

    Parameters
    ----------
    n_cells : int
        Number of cells (rows) in the fixture.
    n_genes : int
        Number of genes (columns) in the fixture.
    random_seed : int
        Seed for the random count matrix (content is not exercised by
        these tests).

    Returns
    -------
    anndata.AnnData
        A minimal AnnData object with a sparse `X` and an empty `obs`
        index ready for a batch column to be added.
    """
    rng = np.random.default_rng(random_seed)
    counts = rng.poisson(lam=5.0, size=(n_cells, n_genes)).astype(np.float64)

    obs = pd.DataFrame(index=[f'cell{i}' for i in range(n_cells)])
    var = pd.DataFrame(index=[f'GENE{i}' for i in range(n_genes)])

    return ad.AnnData(X=sp.csr_matrix(counts), obs=obs, var=var)


def _make_gex_counts(n_cells: int, n_genes: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Build a small, dense integer counts matrix with no all-zero rows
    or columns, so percent_mito/n_counts computations and gene filtering
    inside `filter_normalize_and_hvg` behave sanely.

    Mirrors the fixture-construction approach in
    `tests/test_counts_layer.py` / `tests/test_fixed_hvg_pathway.py`:
    gene means vary and a noisy subset gets a multiplicative burst so
    that `sc.pp.highly_variable_genes` (run inside
    `filter_normalize_and_hvg`, upstream of batch integration) has a
    real, non-empty, non-universal HVG subset to select.
    """
    gene_means = rng.uniform(low=0.5, high=40.0, size=n_genes)
    counts = rng.poisson(lam=gene_means, size=(n_cells, n_genes)).astype(np.float64)
    noisy_gene_idx = rng.choice(
        n_genes, size=max(1, n_genes // 5), replace=False)
    burst = rng.choice(
        [1.0, 8.0], size=(n_cells, len(noisy_gene_idx)), p=[0.85, 0.15])
    counts[:, noisy_gene_idx] *= burst
    counts += 1.0
    return counts


def _make_batch_adata(n_cells: int = 80, n_genes: int = 60,
                       n_batches: int = 2,
                       random_seed: int = RANDOM_SEED) -> ad.AnnData:
    """Build a small synthetic multi-batch AnnData suitable for driving
    `filter_normalize_and_hvg(hvg_batch_key=...)` and `batch_integration()`
    end to end, following the fixture conventions in
    `tests/test_counts_layer.py` / `tests/test_fixed_hvg_pathway.py`
    (sparse `X`, plain 'GENE{i}' names, no TR/IG/MT prefixes, loose
    enough distributions that no cell/gene gets filtered out by default
    thresholds).

    Cells are split as evenly as possible across `n_batches` distinct
    values of `adata.obs['batch']`, with a mild per-batch multiplicative
    shift applied to gene means so the batches are not numerically
    identical (batch integration on genuinely indistinguishable batches
    would not exercise anything interesting).

    Parameters
    ----------
    n_cells : int
        Total number of cells.
    n_genes : int
        Total number of genes.
    n_batches : int
        Number of distinct batch labels to assign.
    random_seed : int
        Seed for the random count matrix and batch assignment.

    Returns
    -------
    anndata.AnnData
        Synthetic AnnData with `adata.obs['batch']` populated and
        `adata.uns['organism']` set.
    """
    rng = np.random.default_rng(random_seed)

    batch_labels = np.array(
        [f'batch_{i % n_batches}' for i in range(n_cells)])
    rng.shuffle(batch_labels)

    counts = _make_gex_counts(n_cells, n_genes, rng)
    # Apply a mild per-batch multiplicative shift to a subset of genes so
    # the batches are numerically distinguishable (not required for the
    # tests below to pass, but keeps the fixture realistic).
    shifted_gene_idx = rng.choice(
        n_genes, size=max(1, n_genes // 4), replace=False)
    for i, label in enumerate(batch_labels):
        batch_num = int(label.split('_')[1])
        if batch_num > 0:
            counts[i, shifted_gene_idx] *= (1.0 + 0.3 * batch_num)

    gene_names = [f'GENE{i}' for i in range(n_genes)]
    obs = pd.DataFrame(
        {'batch': batch_labels},
        index=[f'cell{i}' for i in range(n_cells)],
    )
    var = pd.DataFrame(index=gene_names)

    adata = ad.AnnData(X=sp.csr_matrix(counts), obs=obs, var=var)
    adata.uns['organism'] = 'human'
    return adata


def _run_filter_normalize_and_hvg_no_filtering(adata, **kwargs):
    """Call filter_normalize_and_hvg with thresholds loose enough that no
    cell or gene is dropped, matching the convention in
    tests/test_counts_layer.py and tests/test_fixed_hvg_pathway.py.

    `hvg_min_disp` defaults to a lower value (0.1) than
    `filter_normalize_and_hvg`'s own default (0.5): batch-aware HVG
    selection (`hvg_batch_key` set) intersects the per-batch HVG calls
    across all batches, which is considerably stricter than the
    single-batch case, and this project's synthetic random-count
    fixtures otherwise frequently yield zero surviving genes under that
    intersection with only ~40-100 genes total. A lower threshold keeps
    the fixtures small (per the task's guidance to avoid unnecessarily
    large synthetic datasets) while still exercising a real, non-empty
    HVG subset for the batch-integration PCA/regress_out/Harmony/scVI
    steps that run after `filter_normalize_and_hvg` returns.
    """
    kwargs.setdefault('min_genes_per_cell', 1)
    kwargs.setdefault('max_genes_per_cell', 10000)
    kwargs.setdefault('max_percent_mito', 1.0)
    kwargs.setdefault('min_cells_per_gene', 1)
    kwargs.setdefault('hvg_min_disp', 0.1)
    return filter_normalize_and_hvg(adata, **kwargs)


class TestValidateBatchKeyColumnAbsent:
    """Requirement 1.4: `batch_key` names a column absent from
    `adata.obs`.
    """

    def test_raises_value_error_when_column_absent(self):
        adata = _make_adata()
        assert 'donor_id' not in adata.obs.columns

        with pytest.raises(ValueError):
            _validate_batch_key(adata, 'donor_id')

    def test_error_message_names_batch_key_and_available_columns(self):
        adata = _make_adata()
        adata.obs['sample_id'] = 'only_one_column'

        with pytest.raises(ValueError) as exc_info:
            _validate_batch_key(adata, 'donor_id')

        message = str(exc_info.value)
        assert 'donor_id' in message
        assert 'sample_id' in message


class TestValidateBatchKeyInsufficientDistinctValues:
    """Requirement 1.5: `batch_key` names a column present in
    `adata.obs` but with fewer than two distinct values (zero, i.e.
    all-NaN, or exactly one).
    """

    def test_raises_value_error_for_all_nan_column(self):
        adata = _make_adata()
        adata.obs['batch'] = np.nan

        with pytest.raises(ValueError):
            _validate_batch_key(adata, 'batch')

    def test_raises_value_error_for_single_distinct_value(self):
        adata = _make_adata()
        adata.obs['batch'] = 'only_batch'

        with pytest.raises(ValueError):
            _validate_batch_key(adata, 'batch')

    def test_error_message_names_batch_key_and_distinct_count_zero(self):
        adata = _make_adata()
        adata.obs['batch'] = np.nan

        with pytest.raises(ValueError) as exc_info:
            _validate_batch_key(adata, 'batch')

        message = str(exc_info.value)
        assert 'batch' in message
        assert '0' in message

    def test_error_message_names_batch_key_and_distinct_count_one(self):
        adata = _make_adata()
        adata.obs['batch'] = 'only_batch'

        with pytest.raises(ValueError) as exc_info:
            _validate_batch_key(adata, 'batch')

        message = str(exc_info.value)
        assert 'batch' in message
        assert '1' in message


class TestValidateBatchKeySufficientDistinctValues:
    """Two or more distinct values in the named column raise no
    exception.
    """

    def test_no_exception_for_two_distinct_values(self):
        adata = _make_adata(n_cells=20)
        adata.obs['batch'] = (
            ['batch_a'] * 10 + ['batch_b'] * 10)

        _validate_batch_key(adata, 'batch')

    def test_no_exception_for_more_than_two_distinct_values(self):
        adata = _make_adata(n_cells=30)
        adata.obs['batch'] = (
            ['batch_a'] * 10 + ['batch_b'] * 10 + ['batch_c'] * 10)

        _validate_batch_key(adata, 'batch')


class TestBatchIntegrationMethodValidation:
    """Requirement 2.2: `batch_integration()` restricts `method` to the
    supported `util.BATCH_INTEGRATION_METHODS` set (`{'harmony', 'scvi'}`).

    An unsupported `method` value must raise `ValueError` naming the
    supplied value and the supported set. This check happens after
    `_validate_batch_key` but before any preprocessing or integration
    branch runs, so a valid `batch_key` (>= 2 distinct values) is enough
    to reach it regardless of whether Harmony/scVI are actually
    implemented yet.
    """

    @pytest.mark.parametrize('method', ['scanorama', 'bbknn', 'not_a_method'])
    def test_unsupported_method_raises_value_error(self, method):
        adata = _make_adata()
        adata.obs['batch'] = ['batch_a'] * 10 + ['batch_b'] * 10

        with pytest.raises(ValueError):
            batch_integration(adata, batch_key='batch', method=method)

    @pytest.mark.parametrize('method', ['scanorama', 'bbknn', 'not_a_method'])
    def test_unsupported_method_error_names_value_and_supported_set(
            self, method):
        adata = _make_adata()
        adata.obs['batch'] = ['batch_a'] * 10 + ['batch_b'] * 10

        with pytest.raises(ValueError) as exc_info:
            batch_integration(adata, batch_key='batch', method=method)

        message = str(exc_info.value)
        assert method in message
        assert 'harmony' in message
        assert 'scvi' in message


class TestBatchIntegrationMutualExclusion:
    """Requirement 5.5: the Fixed_HVG_Pathway (`force_variable_genes`)
    and the Full_Integration_Pathway (`batch_integration`) are mutually
    exclusive. If `adata.uns['force_variable_genes']` is already set when
    `batch_integration()` is called, a `ValueError` must be raised before
    any HVG/integration computation runs -- regardless of which
    supported `method` is requested, since this check happens before the
    `harmony`/`scvi` branches (which currently raise `NotImplementedError`
    and would mask a mutual-exclusion failure if the ordering were
    wrong).
    """

    @pytest.mark.parametrize('method', ['harmony', 'scvi'])
    def test_force_variable_genes_set_raises_value_error(self, method):
        adata = _make_adata()
        adata.obs['batch'] = ['batch_a'] * 10 + ['batch_b'] * 10
        adata.uns['force_variable_genes'] = ['GENE0', 'GENE1']

        with pytest.raises(ValueError):
            batch_integration(adata, batch_key='batch', method=method)

    def test_mutual_exclusion_checked_before_notimplementederror(self):
        # With force_variable_genes set, the ValueError from the mutual
        # exclusion check must fire even though method='harmony' would
        # otherwise reach the (currently NotImplementedError) harmony
        # branch. This confirms the mutual-exclusion check runs first,
        # not that the harmony branch's NotImplementedError is somehow
        # being raised instead.
        adata = _make_adata()
        adata.obs['batch'] = ['batch_a'] * 10 + ['batch_b'] * 10
        adata.uns['force_variable_genes'] = ['GENE0', 'GENE1']

        with pytest.raises(ValueError) as exc_info:
            batch_integration(adata, batch_key='batch', method='harmony')

        assert 'force_variable_genes' in str(exc_info.value)

    def test_error_message_mentions_mutual_exclusion(self):
        adata = _make_adata()
        adata.obs['batch'] = ['batch_a'] * 10 + ['batch_b'] * 10
        adata.uns['force_variable_genes'] = ['GENE0']

        with pytest.raises(ValueError) as exc_info:
            batch_integration(adata, batch_key='batch', method='scvi')

        message = str(exc_info.value).lower()
        assert 'mutually exclusive' in message


class TestScviIntegrationCountsLayerAbsent:
    """Requirement 3.4: `_run_scvi_integration` raises `ValueError` when
    `adata.layers['counts']` is absent.

    Uses a fixture that deliberately skips `filter_normalize_and_hvg` so
    that the `ValueError` for the missing Counts_Layer fires before any
    `scvi` import or PCA computation is attempted. `adata.obs` still
    carries `n_counts`/`percent_mito` so that, if the guard were absent
    (i.e. if this test were exercising a regression), execution would
    proceed into `_regress_out_technical_covariates` rather than failing
    for an unrelated missing-column reason -- keeping this test specific
    to the Counts_Layer check.

    This test does not need `scvi-tools` importable at all: the
    `ValueError` check in `_run_scvi_integration` runs before the
    `import scvi` line, so it is run without `KMP_DUPLICATE_LIB_OK` set,
    for extra confidence the import is never reached.
    """

    def _make_adata_without_counts_layer(self, n_cells=20, n_genes=10,
                                          random_seed=RANDOM_SEED):
        rng = np.random.default_rng(random_seed)
        counts = rng.poisson(lam=5.0, size=(n_cells, n_genes)).astype(np.float64)

        obs = pd.DataFrame(
            {
                'n_counts': counts.sum(axis=1),
                'percent_mito': np.zeros(n_cells),
                'batch': (['batch_a'] * (n_cells // 2) +
                          ['batch_b'] * (n_cells - n_cells // 2)),
            },
            index=[f'cell{i}' for i in range(n_cells)],
        )
        var = pd.DataFrame(index=[f'GENE{i}' for i in range(n_genes)])
        adata = ad.AnnData(X=sp.csr_matrix(counts), obs=obs, var=var)
        adata.uns['organism'] = 'human'
        return adata

    def test_raises_value_error_when_counts_layer_absent(self):
        adata = self._make_adata_without_counts_layer()
        assert 'counts' not in adata.layers

        with pytest.raises(ValueError):
            _run_scvi_integration(adata, batch_key='batch', n_gex_pcs=5)

    def test_error_message_names_filter_normalize_and_hvg(self):
        adata = self._make_adata_without_counts_layer()

        with pytest.raises(ValueError) as exc_info:
            _run_scvi_integration(adata, batch_key='batch', n_gex_pcs=5)

        message = str(exc_info.value)
        assert 'counts' in message
        assert 'filter_normalize_and_hvg' in message

    def test_error_raised_before_pca_is_computed(self):
        # If the ValueError guard were bypassed or misordered, sc.tl.pca
        # would run and populate adata.obsm['X_pca']/OBSM_KEY_PCA_GEX_UNINTEGRATED.
        # Confirm neither happens.
        adata = self._make_adata_without_counts_layer()

        with pytest.raises(ValueError):
            _run_scvi_integration(adata, batch_key='batch', n_gex_pcs=5)

        assert 'X_pca' not in adata.obsm
        assert util.OBSM_KEY_PCA_GEX_UNINTEGRATED not in adata.obsm


class TestScviIntegrationImportErrorWhenUninstalled:
    """Requirement 2.6: with `scvi-tools` uninstalled (simulated via
    `sys.modules` patching), `method='scvi'` raises `ImportError` naming
    `conga[batch-integration]`.

    Uses `unittest.mock.patch.dict(sys.modules, {'scvi': None})`, the
    standard idiom for simulating an uninstalled module when it IS
    actually installed in the environment: setting the `sys.modules`
    entry to `None` forces the next `import scvi` to raise
    `ImportError` (per Python's import system) rather than returning a
    cached module, without needing to uninstall the real package.

    This test does not require the real `scvi-tools` package to be
    importable, so no `KMP_DUPLICATE_LIB_OK` handling is needed to run
    it: the patched `import scvi` inside `_run_scvi_integration` never
    reaches the real module.
    """

    def test_batch_integration_method_scvi_raises_import_error(self):
        # See the docstring on test_run_scvi_integration_directly_raises_import_error
        # below for why a throwaway, unpatched filter_normalize_and_hvg
        # call primes numba's JIT cache before entering the sys.modules
        # patch context -- this avoids an unrelated numba registry
        # collision specific to this environment's numba version, not a
        # workaround for anything in conga's own code.
        _run_filter_normalize_and_hvg_no_filtering(
            _make_batch_adata(n_cells=40, n_genes=30, n_batches=2,
                               random_seed=RANDOM_SEED + 1),
            hvg_batch_key='batch')

        adata = _make_batch_adata(n_cells=40, n_genes=30, n_batches=2)

        with mock.patch.dict(sys.modules, {'scvi': None}):
            with pytest.raises(ImportError) as exc_info:
                batch_integration(
                    adata, batch_key='batch', method='scvi',
                    min_genes_per_cell=1, max_genes_per_cell=10000,
                    max_percent_mito=1.0)

        message = str(exc_info.value)
        assert 'scvi-tools' in message
        assert 'conga[batch-integration]' in message

    def test_run_scvi_integration_directly_raises_import_error(self):
        # NOTE: adata is built and run through filter_normalize_and_hvg
        # *before* entering the sys.modules patch below, deliberately.
        # filter_normalize_and_hvg triggers numba's first-time JIT
        # compilation (via scanpy/fast-array-utils' sparse mean/var
        # helpers); numba's on-disk cache loader re-runs
        # `load_additional_registries()` the first time a numba-jitted
        # function is invoked in a given process, and doing that for the
        # first time while `sys.modules['scvi']` is patched to `None`
        # triggers an unrelated numba registry-collision bug
        # ("duplicate registration for ... PolynomialType") in this
        # environment's numba version. Priming the JIT cache with an
        # unpatched call first (as the other unit tests in this module
        # already do, since they call filter_normalize_and_hvg before
        # any test in this class runs) avoids that collision without
        # masking the actual behavior under test, which is entirely
        # about `_run_scvi_integration`'s own `import scvi` line.
        adata = _make_batch_adata(n_cells=40, n_genes=30, n_batches=2)
        adata = _run_filter_normalize_and_hvg_no_filtering(
            adata, hvg_batch_key='batch')
        assert 'counts' in adata.layers

        with mock.patch.dict(sys.modules, {'scvi': None}):
            with pytest.raises(ImportError) as exc_info:
                _run_scvi_integration(adata, batch_key='batch', n_gex_pcs=5)

        message = str(exc_info.value)
        assert 'scvi-tools' in message
        assert 'conga[batch-integration]' in message


@pytest.mark.slow
class TestScviIntegrationEndToEnd:
    """Requirement 3.5, 4.1, 4.2: integration test for the `scvi`
    Integration_Method, using a small synthetic two-batch `AnnData` that
    has already been through
    `filter_normalize_and_hvg(hvg_batch_key=batch_key, ...)` (matching
    what `batch_integration()` itself does before dispatching to
    `_run_scvi_integration`), so `adata.layers['counts']` and
    `adata.obs['n_counts']`/`adata.obs['percent_mito']` are present.

    Marked `slow` per this project's `pyproject.toml` marker
    configuration, since even a tiny fixture's `SCVI.train()` call is not
    instantaneous. Requires `scvi-tools` to be installed and, on this
    project's macOS/arm64 `conga-dev` environment, requires
    `KMP_DUPLICATE_LIB_OK=TRUE` to be set on the invoking shell before
    pytest starts (a local OpenMP-conflict workaround; see the
    development-workflow steering doc), e.g.:

        KMP_DUPLICATE_LIB_OK=TRUE mamba run -n conga-dev python -m pytest \\
            tests/test_batch_integration.py -v -m slow

    `scvi_max_epochs` is kept very small (2) to keep training fast --
    this test asserts structure (obsm/uns keys, setup_anndata call
    arguments), not trained-model output values.

    Per the design's Notes section, `scvi.model.SCVI.train()` itself is
    NOT mocked away (that would defeat the purpose of an integration
    test); instead, `scvi.model.SCVI.setup_anndata` is wrapped with a
    `mock.patch.object(..., wraps=real_setup_anndata)` spy (the same spy
    idiom used for `sc.pp.highly_variable_genes` in
    `tests/test_fixed_hvg_pathway.py`), so the real implementation still
    runs to completion while the call arguments are also asserted.
    """

    def _make_hvg_processed_batch_adata(self, n_cells=80, n_genes=200,
                                         n_batches=2, random_seed=RANDOM_SEED):
        adata = _make_batch_adata(
            n_cells=n_cells, n_genes=n_genes, n_batches=n_batches,
            random_seed=random_seed)
        adata = _run_filter_normalize_and_hvg_no_filtering(
            adata, hvg_batch_key='batch')
        assert 'counts' in adata.layers
        assert 'n_counts' in adata.obs.columns
        assert 'percent_mito' in adata.obs.columns
        return adata

    def test_setup_anndata_called_with_expected_arguments(self):
        import scvi

        adata = self._make_hvg_processed_batch_adata()
        real_setup_anndata = scvi.model.SCVI.setup_anndata

        with mock.patch.object(
            scvi.model.SCVI, 'setup_anndata',
            wraps=real_setup_anndata,
        ) as mock_setup_anndata:
            _run_scvi_integration(
                adata, batch_key='batch', n_gex_pcs=5,
                scvi_max_epochs=2)

        mock_setup_anndata.assert_called_once()
        _, call_kwargs = mock_setup_anndata.call_args
        assert call_kwargs['layer'] == 'counts'
        assert call_kwargs['batch_key'] == 'batch'
        assert call_kwargs['continuous_covariate_keys'] == ['percent_mito']

    def test_obsm_and_uns_structure_matches_harmony_shape(self):
        adata = self._make_hvg_processed_batch_adata()

        result = _run_scvi_integration(
            adata, batch_key='batch', n_gex_pcs=5, scvi_max_epochs=2)

        assert util.OBSM_KEY_PCA_GEX_UNINTEGRATED in result.obsm
        assert util.OBSM_KEY_PCA_GEX_INTEGRATED in result.obsm
        assert 'X_pca_gex' in result.obsm

        np.testing.assert_array_equal(
            np.asarray(result.obsm['X_pca_gex']),
            np.asarray(result.obsm[util.OBSM_KEY_PCA_GEX_INTEGRATED]))

        unintegrated = np.asarray(result.obsm[util.OBSM_KEY_PCA_GEX_UNINTEGRATED])
        integrated = np.asarray(result.obsm[util.OBSM_KEY_PCA_GEX_INTEGRATED])
        assert unintegrated.shape[0] == integrated.shape[0]
        assert not np.array_equal(unintegrated, integrated)

        config = result.uns[util.UNS_KEY_BATCH_INTEGRATION_CONFIG]
        assert config['method'] == 'scvi'
        assert config['batch_key'] == 'batch'
        assert config['n_batches'] == 2

    def test_via_batch_integration_entry_point(self):
        """Same structural assertions, but through the public
        `batch_integration()` entry point rather than calling
        `_run_scvi_integration` directly, confirming the `scvi` branch is
        wired correctly end to end.
        """
        adata = _make_batch_adata(
            n_cells=80, n_genes=200, n_batches=2, random_seed=RANDOM_SEED)

        result = batch_integration(
            adata, batch_key='batch', method='scvi',
            min_genes_per_cell=1, max_genes_per_cell=10000,
            max_percent_mito=1.0, hvg_min_disp=0.1, scvi_max_epochs=2,
            n_gex_pcs=5)

        assert util.OBSM_KEY_PCA_GEX_UNINTEGRATED in result.obsm
        assert util.OBSM_KEY_PCA_GEX_INTEGRATED in result.obsm
        assert 'X_pca_gex' in result.obsm
        np.testing.assert_array_equal(
            np.asarray(result.obsm['X_pca_gex']),
            np.asarray(result.obsm[util.OBSM_KEY_PCA_GEX_INTEGRATED]))

        config = result.uns[util.UNS_KEY_BATCH_INTEGRATION_CONFIG]
        assert config['method'] == 'scvi'
        assert config['batch_key'] == 'batch'
        assert config['n_batches'] == 2


class TestHarmonyIntegrationEndToEnd:
    """Requirement 4.1, 4.2: integration test for the `harmony`
    Integration_Method, on a small synthetic two-batch `AnnData` run
    through the full `batch_integration()` entry point.

    Unlike the scVI path, Harmony integration has no epoch-based training
    loop and is pure numpy/sklearn plus `harmonypy` (no `torch`
    involved), so this is fast enough on a few-hundred-cell fixture to
    run unmarked (not `@pytest.mark.slow`) and without the
    `KMP_DUPLICATE_LIB_OK`/`OMP_NUM_THREADS`/`MKL_NUM_THREADS`
    workaround the scVI tests in this module require.

    `harmonypy` is expected to already be installed in the `conga-dev`
    environment (it is a listed `conga[batch-integration]` dependency),
    so no `sys.modules` patching or optional-skip guard is used here.
    """

    def _make_hvg_processed_batch_adata(self, n_cells=250, n_genes=200,
                                         n_batches=2, random_seed=RANDOM_SEED):
        """Build a two-batch fixture sized per the task description (a
        few hundred cells) with enough genes surviving HVG selection for
        a modest `n_gex_pcs`, using the same `hvg_min_disp=0.1` override
        the scVI tests in this module use to avoid the batch-aware HVG
        intersection collapsing to an empty mask on small synthetic gene
        counts.
        """
        adata = _make_batch_adata(
            n_cells=n_cells, n_genes=n_genes, n_batches=n_batches,
            random_seed=random_seed)
        adata = _run_filter_normalize_and_hvg_no_filtering(
            adata, hvg_batch_key='batch')
        assert 'n_counts' in adata.obs.columns
        assert 'percent_mito' in adata.obs.columns
        return adata

    def test_via_batch_integration_entry_point_obsm_keys_present(self):
        adata = _make_batch_adata(
            n_cells=250, n_genes=200, n_batches=2, random_seed=RANDOM_SEED)

        result = batch_integration(
            adata, batch_key='batch', method='harmony',
            min_genes_per_cell=1, max_genes_per_cell=10000,
            max_percent_mito=1.0, hvg_min_disp=0.1, n_gex_pcs=5)

        assert util.OBSM_KEY_PCA_GEX_UNINTEGRATED in result.obsm
        assert util.OBSM_KEY_PCA_GEX_INTEGRATED in result.obsm
        assert 'X_pca_gex' in result.obsm

    def test_x_pca_gex_equals_integrated_representation_elementwise(self):
        adata = _make_batch_adata(
            n_cells=250, n_genes=200, n_batches=2, random_seed=RANDOM_SEED)

        result = batch_integration(
            adata, batch_key='batch', method='harmony',
            min_genes_per_cell=1, max_genes_per_cell=10000,
            max_percent_mito=1.0, hvg_min_disp=0.1, n_gex_pcs=5)

        np.testing.assert_array_equal(
            np.asarray(result.obsm['X_pca_gex']),
            np.asarray(result.obsm[util.OBSM_KEY_PCA_GEX_INTEGRATED]))

    def test_unintegrated_and_integrated_representations_differ(self):
        adata = _make_batch_adata(
            n_cells=250, n_genes=200, n_batches=2, random_seed=RANDOM_SEED)

        result = batch_integration(
            adata, batch_key='batch', method='harmony',
            min_genes_per_cell=1, max_genes_per_cell=10000,
            max_percent_mito=1.0, hvg_min_disp=0.1, n_gex_pcs=5)

        unintegrated = np.asarray(result.obsm[util.OBSM_KEY_PCA_GEX_UNINTEGRATED])
        integrated = np.asarray(result.obsm[util.OBSM_KEY_PCA_GEX_INTEGRATED])
        assert unintegrated.shape == integrated.shape
        assert not np.array_equal(unintegrated, integrated)

    def test_via_run_harmony_integration_directly(self):
        """Same structural assertions, but calling
        `_run_harmony_integration` directly on an already
        `filter_normalize_and_hvg`-processed fixture, mirroring the
        scVI `test_obsm_and_uns_structure_matches_harmony_shape` test's
        approach for the other Integration_Method.
        """
        adata = self._make_hvg_processed_batch_adata()

        result = _run_harmony_integration(
            adata, 'batch', 5, random_seed=RANDOM_SEED)

        assert util.OBSM_KEY_PCA_GEX_UNINTEGRATED in result.obsm
        assert util.OBSM_KEY_PCA_GEX_INTEGRATED in result.obsm
        assert 'X_pca_gex' in result.obsm

        np.testing.assert_array_equal(
            np.asarray(result.obsm['X_pca_gex']),
            np.asarray(result.obsm[util.OBSM_KEY_PCA_GEX_INTEGRATED]))

        unintegrated = np.asarray(result.obsm[util.OBSM_KEY_PCA_GEX_UNINTEGRATED])
        integrated = np.asarray(result.obsm[util.OBSM_KEY_PCA_GEX_INTEGRATED])
        assert not np.array_equal(unintegrated, integrated)


class TestHarmonyIntegrationImportErrorWhenUninstalled:
    """Requirement 2.5: with `harmonypy` uninstalled (simulated via
    `sys.modules` patching), `method='harmony'` raises `ImportError`
    naming `conga[batch-integration]`.

    Uses `unittest.mock.patch.dict(sys.modules, {'harmonypy': None})`,
    the same idiom `TestScviIntegrationImportErrorWhenUninstalled` above
    uses for `scvi`: setting the `sys.modules` entry to `None` forces the
    next `import harmonypy` to raise `ImportError` without needing to
    actually uninstall the real package.

    As in the scVI `ImportError` tests, an unpatched
    `filter_normalize_and_hvg` call primes numba's JIT cache before
    entering the `sys.modules` patch context, avoiding the unrelated
    numba registry-collision bug described in
    `TestScviIntegrationImportErrorWhenUninstalled`'s docstring.
    """

    def test_batch_integration_method_harmony_raises_import_error(self):
        _run_filter_normalize_and_hvg_no_filtering(
            _make_batch_adata(n_cells=40, n_genes=30, n_batches=2,
                               random_seed=RANDOM_SEED + 1),
            hvg_batch_key='batch')

        adata = _make_batch_adata(n_cells=40, n_genes=30, n_batches=2)

        with mock.patch.dict(sys.modules, {'harmonypy': None}):
            with pytest.raises(ImportError) as exc_info:
                batch_integration(
                    adata, batch_key='batch', method='harmony',
                    min_genes_per_cell=1, max_genes_per_cell=10000,
                    max_percent_mito=1.0, hvg_min_disp=0.1)

        message = str(exc_info.value)
        assert 'harmonypy' in message
        assert 'conga[batch-integration]' in message

    def test_run_harmony_integration_directly_raises_import_error(self):
        adata = _make_batch_adata(n_cells=40, n_genes=30, n_batches=2)
        adata = _run_filter_normalize_and_hvg_no_filtering(
            adata, hvg_batch_key='batch')

        with mock.patch.dict(sys.modules, {'harmonypy': None}):
            with pytest.raises(ImportError) as exc_info:
                _run_harmony_integration(adata, 'batch', 5)

        message = str(exc_info.value)
        assert 'harmonypy' in message
        assert 'conga[batch-integration]' in message


class TestHarmonyIntegrationConfigUns:
    """Requirement 4.4: `adata.uns[util.UNS_KEY_BATCH_INTEGRATION_CONFIG]`
    contents (`method`, `batch_key`, `n_batches`) match the Harmony run's
    actual inputs.
    """

    def test_config_matches_actual_inputs_two_batches(self):
        adata = _make_batch_adata(
            n_cells=250, n_genes=200, n_batches=2, random_seed=RANDOM_SEED)

        result = batch_integration(
            adata, batch_key='batch', method='harmony',
            min_genes_per_cell=1, max_genes_per_cell=10000,
            max_percent_mito=1.0, hvg_min_disp=0.1, n_gex_pcs=5)

        config = result.uns[util.UNS_KEY_BATCH_INTEGRATION_CONFIG]
        assert config['method'] == 'harmony'
        assert config['batch_key'] == 'batch'
        assert config['n_batches'] == 2
        assert config['n_batches'] == result.obs['batch'].nunique()

    def test_config_matches_actual_inputs_three_batches(self):
        adata = _make_batch_adata(
            n_cells=300, n_genes=200, n_batches=3, random_seed=RANDOM_SEED)

        result = batch_integration(
            adata, batch_key='batch', method='harmony',
            min_genes_per_cell=1, max_genes_per_cell=10000,
            max_percent_mito=1.0, hvg_min_disp=0.1, n_gex_pcs=5)

        config = result.uns[util.UNS_KEY_BATCH_INTEGRATION_CONFIG]
        assert config['method'] == 'harmony'
        assert config['batch_key'] == 'batch'
        assert config['n_batches'] == 3
        assert config['n_batches'] == result.obs['batch'].nunique()


class TestHarmonyIntegrationH5adRoundTrip:
    """Requirement 4.5: after the Full_Integration_Pathway runs,
    `OBSM_KEY_PCA_GEX_UNINTEGRATED`, `OBSM_KEY_PCA_GEX_INTEGRATED`, and
    `UNS_KEY_BATCH_INTEGRATION_CONFIG` must all round-trip through
    `.h5ad` elementwise/value-equal.

    Uses the `harmony` Integration_Method rather than `scvi`: this test
    is checking `.h5ad` serialization mechanics (does anndata faithfully
    persist and restore these specific `obsm`/`uns` entries), not the
    Integration_Method's own numerical behavior, which the
    `TestHarmonyIntegrationEndToEnd`/`TestScviIntegrationEndToEnd` classes
    above already cover. Harmony has no training loop and needs no
    `KMP_DUPLICATE_LIB_OK` workaround, so it keeps this test fast and
    unmarked (not `@pytest.mark.slow`).

    Follows the fixture-sizing conventions of
    `TestHarmonyIntegrationEndToEnd` above (`n_genes=200`,
    `hvg_min_disp=0.1`, `n_gex_pcs=5`) and uses pytest's built-in
    `tmp_path` fixture for the temporary `.h5ad` location, matching the
    convention already used in `tests/test_run_conga_cli.py`.
    """

    def _run_full_integration_pathway(self, n_cells: int = 250,
                                       n_genes: int = 200,
                                       n_batches: int = 2) -> ad.AnnData:
        """Run the Full_Integration_Pathway (`batch_integration` with
        `method='harmony'`) on a small synthetic multi-batch fixture.

        Parameters
        ----------
        n_cells : int
            Total number of cells in the synthetic fixture.
        n_genes : int
            Total number of genes in the synthetic fixture.
        n_batches : int
            Number of distinct batch labels to assign.

        Returns
        -------
        anndata.AnnData
            The AnnData object returned by `batch_integration`, carrying
            `OBSM_KEY_PCA_GEX_UNINTEGRATED`, `OBSM_KEY_PCA_GEX_INTEGRATED`,
            and `UNS_KEY_BATCH_INTEGRATION_CONFIG`.
        """
        adata = _make_batch_adata(
            n_cells=n_cells, n_genes=n_genes, n_batches=n_batches,
            random_seed=RANDOM_SEED)

        return batch_integration(
            adata, batch_key='batch', method='harmony',
            min_genes_per_cell=1, max_genes_per_cell=10000,
            max_percent_mito=1.0, hvg_min_disp=0.1, n_gex_pcs=5)

    def test_obsm_representations_round_trip_elementwise_equal(self, tmp_path):
        """`OBSM_KEY_PCA_GEX_UNINTEGRATED` and `OBSM_KEY_PCA_GEX_INTEGRATED`
        survive a write-then-read `.h5ad` round trip unchanged.
        """
        adata = self._run_full_integration_pathway()
        unintegrated_before = np.asarray(
            adata.obsm[util.OBSM_KEY_PCA_GEX_UNINTEGRATED]).copy()
        integrated_before = np.asarray(
            adata.obsm[util.OBSM_KEY_PCA_GEX_INTEGRATED]).copy()

        h5ad_path = tmp_path / 'batch_integration_roundtrip.h5ad'
        adata.write_h5ad(h5ad_path)
        reloaded = ad.read_h5ad(h5ad_path)

        np.testing.assert_array_equal(
            np.asarray(reloaded.obsm[util.OBSM_KEY_PCA_GEX_UNINTEGRATED]),
            unintegrated_before)
        np.testing.assert_array_equal(
            np.asarray(reloaded.obsm[util.OBSM_KEY_PCA_GEX_INTEGRATED]),
            integrated_before)

    def test_batch_integration_config_uns_round_trips_value_equal(
            self, tmp_path):
        """`UNS_KEY_BATCH_INTEGRATION_CONFIG` (a plain dict with
        `method`/`batch_key`/`n_batches` keys) matches exactly before vs.
        after the round trip.

        `n_batches` is compared via `int(...)` on both sides rather than
        asserting exact type equality: h5ad serialization of `.uns`
        dicts can coerce a Python `int` into a numpy integer scalar
        (e.g. `numpy.int64`) on reload, which is a type-representation
        change, not a value-loss bug, so the comparison here is
        value-robust by design per the task description.
        """
        adata = self._run_full_integration_pathway()
        config_before = dict(adata.uns[util.UNS_KEY_BATCH_INTEGRATION_CONFIG])

        h5ad_path = tmp_path / 'batch_integration_roundtrip.h5ad'
        adata.write_h5ad(h5ad_path)
        reloaded = ad.read_h5ad(h5ad_path)

        config_after = dict(reloaded.uns[util.UNS_KEY_BATCH_INTEGRATION_CONFIG])

        assert set(config_after.keys()) == set(config_before.keys())
        assert str(config_after['method']) == str(config_before['method'])
        assert str(config_after['batch_key']) == str(config_before['batch_key'])
        assert int(config_after['n_batches']) == int(config_before['n_batches'])
        # Confirm the fixture's actual batch count flows through
        # end to end, not just that before/after agree with each other.
        assert int(config_after['n_batches']) == 2

    def test_obsm_arrays_retain_dtype_and_shape_after_round_trip(
            self, tmp_path):
        """Elementwise equality alone would not catch a round trip that
        silently truncates shape or changes dtype in a way that still
        happens to compare equal (e.g. an all-zero slice). Assert shape
        and a floating dtype are preserved as an additional guard.
        """
        adata = self._run_full_integration_pathway()
        unintegrated_before = np.asarray(
            adata.obsm[util.OBSM_KEY_PCA_GEX_UNINTEGRATED])
        integrated_before = np.asarray(
            adata.obsm[util.OBSM_KEY_PCA_GEX_INTEGRATED])

        h5ad_path = tmp_path / 'batch_integration_roundtrip.h5ad'
        adata.write_h5ad(h5ad_path)
        reloaded = ad.read_h5ad(h5ad_path)

        unintegrated_after = np.asarray(
            reloaded.obsm[util.OBSM_KEY_PCA_GEX_UNINTEGRATED])
        integrated_after = np.asarray(
            reloaded.obsm[util.OBSM_KEY_PCA_GEX_INTEGRATED])

        assert unintegrated_after.shape == unintegrated_before.shape
        assert integrated_after.shape == integrated_before.shape
        assert np.issubdtype(unintegrated_after.dtype, np.floating)
        assert np.issubdtype(integrated_after.dtype, np.floating)


class TestClusterAndTsneAndUmapReusesIntegratedPcaGex:
    """Requirement 4.2: once the Full_Integration_Pathway has populated
    `adata.obsm['X_pca_gex']` with the Integrated_Representation,
    `cluster_and_tsne_and_umap` must treat it as already-computed and
    not recompute it via `sc.tl.pca`.

    `batch_integration(..., method='harmony')` is used to populate
    `X_pca_gex` because it has no epoch-based training loop (unlike
    `scvi`) and is therefore fast enough to run unmarked, matching the
    rationale already given in `TestHarmonyIntegrationEndToEnd` above.
    This test is about `cluster_and_tsne_and_umap`'s own guard behavior,
    not about which Integration_Method produced the representation it
    is reusing.

    `conga.preprocess.sc.tl.pca` is patched (not a bare `scanpy.tl.pca`)
    because `conga/preprocess.py` imports scanpy at module level as
    `sc`, so the name must be patched as it is looked up from that
    module -- the same idiom used for
    `conga.preprocess.sc.pp.highly_variable_genes` in
    `tests/test_fixed_hvg_pathway.py`.

    `skip_tcr=True` is used to keep this test scoped to the GEX PCA
    guard: with `skip_tcr=True`, the `tag=='tcr'` iteration of
    `cluster_and_tsne_and_umap`'s per-representation loop is skipped via
    `continue` before any TCR-specific `adata.obsm` lookup is attempted
    (confirmed by reading `conga/preprocess.py`), so no `X_pca_tcr` /
    `X_vec_tcr` representation needs to exist on the fixture. Per the
    function's own source, `sc.tl.pca` is called only inside the
    `'X_pca_gex' not in adata.obsm.keys() or recompute_pca_gex` guard
    block for the GEX representation; the TCR branch never calls
    `sc.tl.pca` at all, so patching `sc.tl.pca` only observes the guard
    this test is about.
    """

    def _make_integrated_batch_adata(self, n_cells=250, n_genes=200,
                                      n_batches=2, random_seed=RANDOM_SEED):
        adata = _make_batch_adata(
            n_cells=n_cells, n_genes=n_genes, n_batches=n_batches,
            random_seed=random_seed)
        return batch_integration(
            adata, batch_key='batch', method='harmony',
            min_genes_per_cell=1, max_genes_per_cell=10000,
            max_percent_mito=1.0, hvg_min_disp=0.1, n_gex_pcs=5)

    def test_pca_not_recomputed_when_x_pca_gex_already_present(self):
        adata = self._make_integrated_batch_adata()
        assert 'X_pca_gex' in adata.obsm

        with mock.patch('conga.preprocess.sc.tl.pca') as mock_pca:
            cluster_and_tsne_and_umap(
                adata, recompute_pca_gex=False, skip_tcr=True)

        mock_pca.assert_not_called()

    def test_pca_recomputed_when_recompute_pca_gex_true(self):
        """Sanity check for the spy itself (mirroring the pattern used by
        `TestForceVariableGenesSkipsAutoHVG.test_highly_variable_genes_called_on_default_pathway`
        in `tests/test_fixed_hvg_pathway.py`): forcing recomputation via
        `recompute_pca_gex=True` must still call `sc.tl.pca` even though
        `X_pca_gex` is already present, confirming the mock above would
        actually catch a regression rather than passing vacuously.

        Uses `mock.patch(..., wraps=real_pca)` (a real spy that still
        delegates to the original implementation) so the function
        continues to run to completion, rather than `side_effect=real_pca`
        on an already-patched attribute (which would recurse into the
        mock itself).
        """
        adata = self._make_integrated_batch_adata()
        assert 'X_pca_gex' in adata.obsm

        real_pca = sc.tl.pca
        with mock.patch(
            'conga.preprocess.sc.tl.pca', wraps=real_pca,
        ) as mock_pca:
            cluster_and_tsne_and_umap(
                adata, recompute_pca_gex=True, skip_tcr=True,
                n_gex_pcs=5)

        mock_pca.assert_called()

    def test_pca_computed_when_x_pca_gex_absent(self):
        """Sanity check: when `X_pca_gex` is absent entirely (the
        Default_Pathway case `cluster_and_tsne_and_umap` already
        handles), `sc.tl.pca` must be called to compute it, confirming
        the guard's "absent" branch is also observed by the spy.
        """
        adata = _make_batch_adata(
            n_cells=250, n_genes=200, n_batches=2, random_seed=RANDOM_SEED)
        adata = _run_filter_normalize_and_hvg_no_filtering(
            adata, hvg_batch_key='batch')
        assert 'X_pca_gex' not in adata.obsm

        real_pca = sc.tl.pca
        with mock.patch(
            'conga.preprocess.sc.tl.pca', wraps=real_pca,
        ) as mock_pca:
            cluster_and_tsne_and_umap(
                adata, recompute_pca_gex=False, skip_tcr=True,
                n_gex_pcs=5)

        mock_pca.assert_called()
