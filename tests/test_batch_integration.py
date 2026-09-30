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

This test module is new; other batch-integration tasks (e.g. 4.4, the
`batch_integration()` method-validation/mutual-exclusion tests) are
expected to add further test classes to this same file as their
implementations land.

Requirements: 1.4, 1.5
"""

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import pytest

from conga.preprocess import _validate_batch_key, batch_integration


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
