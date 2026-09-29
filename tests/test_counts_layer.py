"""
Unit tests for the Counts_Layer capture in
`conga.preprocess.filter_normalize_and_hvg`.

Per the batch-integration design (Component 2) and Requirement 3,
`filter_normalize_and_hvg` unconditionally captures a pre-normalization
count matrix into `adata.layers['counts']`, immediately after the
antibody-feature-removal block and before `sc.pp.normalize_total`/
`sc.pp.log1p` run. When antibody/protein-capture features are present and
removed, the captured layer is narrowed to the same GEX-only columns as
the (also narrowed) `adata` object; when no antibody features are present
(or no feature-type column exists), the layer is simply the full raw
counts matrix at that point.

Requirements: 3.1, 3.2, 3.3
"""

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

from conga.preprocess import filter_normalize_and_hvg
from conga import util


RANDOM_SEED = 42


def _make_gex_counts(n_cells, n_genes, rng):
    """Build a small, dense integer counts matrix with no all-zero rows
    or columns, so percent_mito/n_counts computations and gene filtering
    inside filter_normalize_and_hvg behave sanely.

    Genes are given varying Poisson means (some low-and-noisy, some
    high-and-flat) rather than a single uniform distribution, so that
    `sc.pp.highly_variable_genes` actually selects a non-empty,
    non-universal subset of genes downstream. A uniform-random count
    matrix tends to produce zero (or all) highly-variable genes, which
    would make the elementwise/shape assertions in these tests vacuous.
    """
    gene_means = rng.uniform(low=0.5, high=40.0, size=n_genes)
    counts = rng.poisson(lam=gene_means, size=(n_cells, n_genes)).astype(np.float64)
    # Make a subset of genes much noisier (higher dispersion relative to
    # their mean) than the rest by adding a strong per-cell multiplicative
    # burst to ~20% of genes, so sc.pp.highly_variable_genes has a real,
    # non-empty, non-universal subset to select.
    noisy_gene_idx = rng.choice(
        n_genes, size=max(1, n_genes // 5), replace=False)
    burst = rng.choice(
        [1.0, 8.0], size=(n_cells, len(noisy_gene_idx)), p=[0.85, 0.15])
    counts[:, noisy_gene_idx] *= burst
    # Guarantee no all-zero row/column (would break percent_mito/n_counts
    # division and gene filtering) by adding a baseline count of 1.
    counts += 1.0
    return counts


def _make_adata(n_cells=60, n_genes=200, with_antibody=False, n_antibody=10,
                 random_seed=RANDOM_SEED):
    """Build a synthetic AnnData that survives filter_normalize_and_hvg's
    filtering steps unmodified (no cells/genes dropped), optionally with
    antibody-capture features appended.

    Gene names deliberately avoid any TR/IG/MT prefix so that
    `util.is_vdj_gene` and the mitochondrial-gene detection in
    `filter_normalize_and_hvg` do not treat any of them specially.
    """
    rng = np.random.default_rng(random_seed)

    gex_counts = _make_gex_counts(n_cells, n_genes, rng)
    gex_gene_names = [f'GENE{i}' for i in range(n_genes)]

    if with_antibody:
        antibody_counts = rng.integers(
            low=1, high=20, size=(n_cells, n_antibody)).astype(np.float64)
        antibody_names = [f'AB{i}' for i in range(n_antibody)]

        X = np.hstack([gex_counts, antibody_counts])
        var_names = gex_gene_names + antibody_names
        feature_types = (
            [util.GENE_EXPRESSION_FEATURE_TYPE] * n_genes +
            [util.ANTIBODY_CAPTURE_FEATURE_TYPE] * n_antibody
        )
    else:
        X = gex_counts
        var_names = gex_gene_names
        feature_types = None

    obs = pd.DataFrame(index=[f'cell{i}' for i in range(n_cells)])
    var = pd.DataFrame(index=var_names)
    if feature_types is not None:
        # get_feature_types_varname requires a column whose name *starts
        # with* 'feature_types' (e.g. 'feature_types-0'), matching the
        # naming convention produced by cellranger aggr outputs.
        var['feature_types-0'] = feature_types

    # filter_normalize_and_hvg relies on scipy-sparse-only accessors (e.g.
    # `.A1` on the result of `adata[:, mask].X.sum(...)`), so build the
    # fixture with a sparse X, matching real 10x-derived AnnData objects.
    adata = ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=var)
    adata.uns['organism'] = 'human'
    return adata


def _run_filter_normalize_and_hvg_no_filtering(adata):
    """Call filter_normalize_and_hvg with thresholds loose enough that no
    cell or gene is dropped, given the fixtures built by _make_adata.
    """
    return filter_normalize_and_hvg(
        adata,
        min_genes_per_cell=1,
        max_genes_per_cell=10000,
        max_percent_mito=1.0,
        min_cells_per_gene=1,
    )


class TestCountsLayerNoAntibodyFeatures:
    """Requirement 3.1, 3.2, 3.3: Counts_Layer capture when no antibody
    features are present.
    """

    def test_counts_layer_present(self):
        adata = _make_adata(with_antibody=False)
        result = _run_filter_normalize_and_hvg_no_filtering(adata)
        assert 'counts' in result.layers

    def test_counts_layer_matches_pre_normalization_raw_x(self):
        adata = _make_adata(with_antibody=False)
        # Capture the pre-normalization raw counts ourselves, using a
        # fixture built so that no cells/genes get filtered out, so what
        # survives is exactly the input matrix.
        expected_counts = np.asarray(adata.X.todense())

        result = _run_filter_normalize_and_hvg_no_filtering(adata)

        assert 'counts' in result.layers
        captured = result.layers['counts']
        if hasattr(captured, 'toarray'):
            captured = captured.toarray()
        captured = np.asarray(captured)

        # No antibody features and no filtering, so the captured counts
        # layer should equal the original input matrix elementwise
        # (restricted to whatever genes survive the HVG selection step,
        # since the returned adata is the HVG-filtered view). Match
        # columns by var_names against the original fixture.
        original_var_names = list(adata.var_names)
        col_indices = [original_var_names.index(g) for g in result.var_names]
        expected_restricted = expected_counts[:, col_indices]

        np.testing.assert_array_equal(captured, expected_restricted)


class TestCountsLayerWithAntibodyFeatures:
    """Requirement 3.1, 3.2, 3.3: Counts_Layer capture when antibody
    features are present and removed.
    """

    def test_counts_layer_present_and_no_error(self):
        adata = _make_adata(with_antibody=True)
        result = _run_filter_normalize_and_hvg_no_filtering(adata)
        assert 'counts' in result.layers

    def test_counts_layer_width_matches_narrowed_adata(self):
        adata = _make_adata(with_antibody=True)
        result = _run_filter_normalize_and_hvg_no_filtering(adata)

        # adata is narrowed to GEX-only (antibody-excluded) columns, then
        # further narrowed by HVG selection; the counts layer must match
        # that final narrowed width, not the original wider input.
        assert result.layers['counts'].shape[1] == result.shape[1]
        assert result.layers['counts'].shape[0] == result.shape[0]

    def test_counts_layer_excludes_antibody_columns_by_value(self):
        adata = _make_adata(with_antibody=True)
        # Pre-normalization GEX-only raw counts, indexed by gene name, so
        # we can confirm antibody columns were actually excluded by value
        # rather than merely by coincidental shape.
        original_var_names = list(adata.var_names)
        expected_counts = np.asarray(adata.X.todense())

        result = _run_filter_normalize_and_hvg_no_filtering(adata)

        assert 'counts' in result.layers
        captured = result.layers['counts']
        if hasattr(captured, 'toarray'):
            captured = captured.toarray()
        captured = np.asarray(captured)

        # None of the surviving var_names should be antibody features.
        assert all(not name.startswith('AB') for name in result.var_names)

        col_indices = [original_var_names.index(g) for g in result.var_names]
        expected_restricted = expected_counts[:, col_indices]

        np.testing.assert_array_equal(captured, expected_restricted)


class TestCountsLayerDefaultPathway:
    """Requirement 3.2: the Counts_Layer is present when
    filter_normalize_and_hvg is called with no hvg_batch_key and no
    force_variable_genes set (the Default_Pathway), documented explicitly
    rather than relying on it being an incidental side effect of another
    test's fixture choice.
    """

    def test_counts_layer_present_on_default_pathway(self):
        adata = _make_adata(with_antibody=False)
        assert 'force_variable_genes' not in adata.uns
        assert 'highly_variable' not in adata.var.columns

        result = filter_normalize_and_hvg(
            adata,
            min_genes_per_cell=1,
            max_genes_per_cell=10000,
            max_percent_mito=1.0,
            min_cells_per_gene=1,
            hvg_batch_key=None,
        )

        assert 'counts' in result.layers
