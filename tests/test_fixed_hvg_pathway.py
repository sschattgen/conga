"""
Unit tests for the Fixed_HVG_Pathway three-state guard in
`conga.preprocess.filter_normalize_and_hvg`.

Per the batch-integration design (Component 1's HVG-selection block) and
Requirement 5, `filter_normalize_and_hvg` supports three mutually
exclusive states when selecting highly-variable genes:

    (A) `adata.uns['force_variable_genes']` is set -- `hvg_mask` is built
        directly from that gene list, `sc.pp.highly_variable_genes` is
        never called, gene symbols absent from `adata.var_names` are
        excluded and logged at `logging.WARNING`, and
        `conga_stats['fixed_hvg_list_size']` /
        `conga_stats['fixed_hvg_mask_size']` are recorded.
    (B) No `force_variable_genes`, but the caller pre-set
        `adata.var['highly_variable']` before calling
        `filter_normalize_and_hvg` -- that mask is used verbatim and
        `sc.pp.highly_variable_genes` is never called (so it cannot
        silently overwrite the caller's column).
    (C) Neither of the above -- `sc.pp.highly_variable_genes` runs as it
        always has (the Default_Pathway, not exercised in depth here;
        see other existing test modules).

States A and B both still fall through to the same unconditional TR/IG
and sex-linked gene exclusion code that runs after the three-way branch.

Requirements: 5.2, 5.3, 5.6, 5.7, 5.8
"""

import logging
from typing import List
from unittest import mock

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

from conga.preprocess import filter_normalize_and_hvg
from conga import util


RANDOM_SEED = 42


def _make_gex_counts(n_cells: int, n_genes: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Build a small, dense integer counts matrix with no all-zero rows
    or columns, so percent_mito/n_counts computations and gene filtering
    inside `filter_normalize_and_hvg` behave sanely.

    Mirrors the fixture-construction approach in `tests/test_counts_layer.py`:
    gene means vary and a noisy subset gets a multiplicative burst so that
    a real (non-empty, non-universal) HVG subset would exist if automatic
    detection were to run -- not strictly required for these tests (which
    mostly bypass automatic HVG detection), but keeps fixtures realistic
    and reusable if a Default_Pathway comparison is ever added here.
    """
    gene_means = rng.uniform(low=0.5, high=40.0, size=n_genes)
    counts = rng.poisson(lam=gene_means, size=(n_cells, n_genes)).astype(np.float64)
    noisy_gene_idx = rng.choice(
        n_genes, size=max(1, n_genes // 5), replace=False)
    burst = rng.choice(
        [1.0, 8.0], size=(n_cells, len(noisy_gene_idx)), p=[0.85, 0.15])
    counts[:, noisy_gene_idx] *= burst
    # Guarantee no all-zero row/column.
    counts += 1.0
    return counts


def _make_adata(n_cells: int = 60, n_genes: int = 40,
                 extra_gene_names: List[str] = None,
                 random_seed: int = RANDOM_SEED) -> ad.AnnData:
    """Build a synthetic AnnData that survives `filter_normalize_and_hvg`'s
    filtering steps unmodified (no cells/genes dropped), with plain
    'GENE{i}' names plus any caller-specified extra gene names appended
    (e.g. TR/IG or sex-linked names for exclusion tests).

    Uses a sparse `adata.X` (scipy.sparse.csr_matrix), matching real
    10x-derived AnnData objects -- `filter_normalize_and_hvg` relies on
    sparse-only accessors (e.g. `.A1`) internally.
    """
    rng = np.random.default_rng(random_seed)

    extra_gene_names = extra_gene_names or []
    n_extra = len(extra_gene_names)

    gex_counts = _make_gex_counts(n_cells, n_genes, rng)
    gex_gene_names = [f'GENE{i}' for i in range(n_genes)]

    if n_extra:
        extra_counts = _make_gex_counts(n_cells, n_extra, rng)
        X = np.hstack([gex_counts, extra_counts])
        var_names = gex_gene_names + list(extra_gene_names)
    else:
        X = gex_counts
        var_names = gex_gene_names

    obs = pd.DataFrame(index=[f'cell{i}' for i in range(n_cells)])
    var = pd.DataFrame(index=var_names)

    adata = ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=var)
    adata.uns['organism'] = 'human'
    return adata


def _run_no_filtering(adata, **kwargs):
    """Call filter_normalize_and_hvg with thresholds loose enough that no
    cell or gene is dropped, matching the convention in
    tests/test_counts_layer.py.
    """
    kwargs.setdefault('min_genes_per_cell', 1)
    kwargs.setdefault('max_genes_per_cell', 10000)
    kwargs.setdefault('max_percent_mito', 1.0)
    kwargs.setdefault('min_cells_per_gene', 1)
    return filter_normalize_and_hvg(adata, **kwargs)


class TestForceVariableGenesExcludedSymbolWarning:
    """Requirement 5.6: a Fixed_Gene_List symbol absent from
    adata.var_names is excluded from hvg_mask and logged as a
    logging.WARNING with the correct count.
    """

    def test_missing_symbol_excluded_and_warned(self, caplog):
        adata = _make_adata(n_genes=40)
        present_genes = ['GENE0', 'GENE1', 'GENE2']
        missing_genes = ['NOT_A_GENE_A', 'NOT_A_GENE_B']
        adata.uns['force_variable_genes'] = present_genes + missing_genes

        with caplog.at_level(logging.WARNING, logger='conga.preprocess'):
            result = _run_no_filtering(adata)

        # The two missing symbols must not appear in the resulting mask,
        # i.e. must not be among the surviving (HVG-selected) var_names.
        for gene in missing_genes:
            assert gene not in set(result.var_names)
        for gene in present_genes:
            assert gene in set(result.var_names)

        warning_records = [
            r for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert len(warning_records) >= 1
        combined_message = ' '.join(r.getMessage() for r in warning_records)
        assert str(len(missing_genes)) in combined_message
        for gene in missing_genes:
            assert gene in combined_message

    def test_no_warning_when_all_symbols_present(self, caplog):
        adata = _make_adata(n_genes=40)
        present_genes = ['GENE0', 'GENE1', 'GENE2']
        adata.uns['force_variable_genes'] = present_genes

        with caplog.at_level(logging.WARNING, logger='conga.preprocess'):
            _run_no_filtering(adata)

        warning_records = [
            r for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert len(warning_records) == 0


class TestForceVariableGenesSkipsAutoHVG:
    """Requirement 5.3: sc.pp.highly_variable_genes is not called when
    force_variable_genes is set, verified via mock/spy assertion.
    """

    def test_highly_variable_genes_not_called(self):
        adata = _make_adata(n_genes=40)
        adata.uns['force_variable_genes'] = ['GENE0', 'GENE1', 'GENE2']

        with mock.patch(
            'conga.preprocess.sc.pp.highly_variable_genes'
        ) as mock_hvg:
            _run_no_filtering(adata)

        mock_hvg.assert_not_called()

    def test_highly_variable_genes_called_on_default_pathway(self):
        """Sanity check for the spy itself: with neither force_variable_genes
        nor a pre-set highly_variable column, sc.pp.highly_variable_genes
        must still be called (state C), confirming the mock above is
        actually exercising the guard rather than passing vacuously.

        Uses `mock.patch(..., wraps=real_fn)` (a real spy that still
        delegates to the original implementation) rather than
        `side_effect=real_fn`, since assigning the *unpatched* attribute
        as a side_effect after patching it in the same target recurses
        into the mock itself.
        """
        import scanpy as sc
        adata = _make_adata(n_genes=40)
        assert 'force_variable_genes' not in adata.uns
        assert 'highly_variable' not in adata.var.columns

        real_hvg = sc.pp.highly_variable_genes
        with mock.patch(
            'conga.preprocess.sc.pp.highly_variable_genes',
            side_effect=real_hvg,
        ) as mock_hvg:
            _run_no_filtering(adata)

        mock_hvg.assert_called()


class TestCallerSuppliedHighlyVariableSurvivesUnmodified:
    """Requirement 5.2: a direct adata.var['highly_variable'] assignment
    survives filter_normalize_and_hvg unmodified when neither
    force_variable_genes nor hvg_batch_key is set.
    """

    def test_caller_mask_used_verbatim(self):
        adata = _make_adata(n_genes=40)
        assert 'force_variable_genes' not in adata.uns

        caller_mask = np.zeros(adata.shape[1], dtype=bool)
        chosen_genes = ['GENE0', 'GENE5', 'GENE10']
        for gene in chosen_genes:
            caller_mask[list(adata.var_names).index(gene)] = True
        adata.var['highly_variable'] = caller_mask

        result = _run_no_filtering(adata, hvg_batch_key=None)

        assert set(result.var_names) == set(chosen_genes)

    def test_highly_variable_genes_not_called_for_caller_mask(self):
        adata = _make_adata(n_genes=40)
        caller_mask = np.zeros(adata.shape[1], dtype=bool)
        caller_mask[:3] = True
        adata.var['highly_variable'] = caller_mask

        with mock.patch(
            'conga.preprocess.sc.pp.highly_variable_genes'
        ) as mock_hvg:
            _run_no_filtering(adata, hvg_batch_key=None)

        mock_hvg.assert_not_called()


class TestTRIGAndSexLinkedExclusionStillApplies:
    """Requirement 5.7: TR/IG and sex-linked exclusion still reduce the
    caller-supplied mask under the Fixed_HVG_Pathway (both state A via
    force_variable_genes and state B via a direct highly_variable
    assignment).
    """

    # Real TR gene name prefixes recognized by util.is_vdj_gene for
    # organism='human' (TCR_AB_VDJ_TYPE): trav/trbv/traj/trbj/trbd.
    TR_GENE_NAMES = ['TRAV1-1', 'TRBV2']
    # Recognized sex-linked gene, from preprocess.all_sexlinked_genes.
    SEXLINKED_GENE_NAME = 'XIST'

    def test_force_variable_genes_excludes_tr_genes(self):
        adata = _make_adata(
            n_genes=40, extra_gene_names=self.TR_GENE_NAMES)
        adata.uns['force_variable_genes'] = (
            ['GENE0', 'GENE1'] + self.TR_GENE_NAMES)

        result = _run_no_filtering(adata)

        for tr_gene in self.TR_GENE_NAMES:
            assert tr_gene not in set(result.var_names)
        assert 'GENE0' in set(result.var_names)
        assert 'GENE1' in set(result.var_names)

    def test_force_variable_genes_excludes_sexlinked_genes_when_requested(self):
        adata = _make_adata(
            n_genes=40, extra_gene_names=[self.SEXLINKED_GENE_NAME])
        adata.uns['force_variable_genes'] = (
            ['GENE0', 'GENE1', self.SEXLINKED_GENE_NAME])

        result = _run_no_filtering(adata, exclude_sexlinked=True)

        assert self.SEXLINKED_GENE_NAME not in set(result.var_names)
        assert 'GENE0' in set(result.var_names)
        assert 'GENE1' in set(result.var_names)

    def test_force_variable_genes_keeps_sexlinked_gene_when_not_requested(self):
        adata = _make_adata(
            n_genes=40, extra_gene_names=[self.SEXLINKED_GENE_NAME])
        adata.uns['force_variable_genes'] = (
            ['GENE0', 'GENE1', self.SEXLINKED_GENE_NAME])

        result = _run_no_filtering(adata, exclude_sexlinked=False)

        assert self.SEXLINKED_GENE_NAME in set(result.var_names)

    def test_caller_supplied_highly_variable_excludes_tr_genes(self):
        adata = _make_adata(
            n_genes=40, extra_gene_names=self.TR_GENE_NAMES)
        caller_mask = np.zeros(adata.shape[1], dtype=bool)
        for gene in ['GENE0', 'GENE1'] + self.TR_GENE_NAMES:
            caller_mask[list(adata.var_names).index(gene)] = True
        adata.var['highly_variable'] = caller_mask

        result = _run_no_filtering(adata, hvg_batch_key=None)

        for tr_gene in self.TR_GENE_NAMES:
            assert tr_gene not in set(result.var_names)
        assert 'GENE0' in set(result.var_names)
        assert 'GENE1' in set(result.var_names)


class TestFixedHvgStatsRecorded:
    """Requirement 5.8: conga_stats['fixed_hvg_list_size'] and
    conga_stats['fixed_hvg_mask_size'] are recorded with correct values.
    """

    def test_force_variable_genes_stats_no_exclusions(self):
        adata = _make_adata(n_genes=40)
        fixed_gene_list = ['GENE0', 'GENE1', 'GENE2', 'GENE3']
        adata.uns['force_variable_genes'] = fixed_gene_list

        result = _run_no_filtering(adata)

        assert result.uns['conga_stats']['fixed_hvg_list_size'] == \
            len(fixed_gene_list)
        # No TR/IG/sex-linked genes among the fixed list and no missing
        # symbols, so the mask size should equal the list size exactly.
        assert result.uns['conga_stats']['fixed_hvg_mask_size'] == \
            len(fixed_gene_list)
        assert result.shape[1] == len(fixed_gene_list)

    def test_force_variable_genes_stats_with_missing_and_excluded_symbols(self):
        adata = _make_adata(
            n_genes=40, extra_gene_names=self.__class__._tr_genes())
        present_genes = ['GENE0', 'GENE1', 'GENE2']
        missing_genes = ['NOT_A_REAL_GENE']
        tr_genes = self.__class__._tr_genes()
        fixed_gene_list = present_genes + missing_genes + tr_genes
        adata.uns['force_variable_genes'] = fixed_gene_list

        result = _run_no_filtering(adata)

        # fixed_hvg_list_size reflects the caller-supplied list length,
        # including symbols later excluded as missing or TR genes.
        assert result.uns['conga_stats']['fixed_hvg_list_size'] == \
            len(fixed_gene_list)
        # fixed_hvg_mask_size reflects the mask *after* missing-symbol
        # exclusion and TR/IG/sex-linked exclusion: only present_genes
        # survive (missing_genes were never in var_names, tr_genes are
        # excluded by the TR-gene guard).
        assert result.uns['conga_stats']['fixed_hvg_mask_size'] == \
            len(present_genes)
        assert set(result.var_names) == set(present_genes)

    @staticmethod
    def _tr_genes():
        return ['TRAV1-1', 'TRBV2']

    def test_caller_supplied_highly_variable_stats(self):
        adata = _make_adata(n_genes=40)
        caller_mask = np.zeros(adata.shape[1], dtype=bool)
        chosen_genes = ['GENE0', 'GENE1', 'GENE2', 'GENE3', 'GENE4']
        for gene in chosen_genes:
            caller_mask[list(adata.var_names).index(gene)] = True
        adata.var['highly_variable'] = caller_mask

        result = _run_no_filtering(adata, hvg_batch_key=None)

        assert result.uns['conga_stats']['fixed_hvg_list_size'] == \
            len(chosen_genes)
        assert result.uns['conga_stats']['fixed_hvg_mask_size'] == \
            len(chosen_genes)

    def test_default_pathway_does_not_record_fixed_hvg_stats(self):
        """Sanity check: the fixed_hvg_* stats keys are specific to the
        Fixed_HVG_Pathway (states A/B) and should not appear when neither
        force_variable_genes nor a pre-set highly_variable column is
        present (state C, Default_Pathway).
        """
        adata = _make_adata(n_genes=40)
        assert 'force_variable_genes' not in adata.uns
        assert 'highly_variable' not in adata.var.columns

        result = _run_no_filtering(adata)

        assert 'fixed_hvg_list_size' not in result.uns['conga_stats']
        assert 'fixed_hvg_mask_size' not in result.uns['conga_stats']
