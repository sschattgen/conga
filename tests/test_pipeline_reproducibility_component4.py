"""
Unit tests for Component 4 of the pipeline-reproducibility feature:
`conga.preprocess.calc_X_pca_gex_including_protein_features`'s new
`random_seed` parameter and its wiring to the function's single
in-scope internal stochastic call site, the diagnostic sampling inside
the `compare_distance_distributions` branch:

    rng = np.random.default_rng(random_seed)
    inds = rng.permutation(adata.shape[0])[:nrandom]

Two intentionally-unseeded calls remain untouched by this feature, per
design.md's Component 4 scope note and requirements.md's Requirement 1
acceptance criteria (which name only specific `sc.tl.pca` call sites and
the two `KernelPCA` constructors, not this function's own `sc.tl.pca`
call or its standalone `sklearn.decomposition.PCA()` call):

    - `sc.tl.pca(adata, svd_solver='arpack', n_comps=n_components_gex)`
      (the GEX-PC computation near the top of the function)
    - `pca = PCA()` (the protein-feature PCA, unseeded sklearn PCA, no
      `random_state` kwarg at all)

Two complementary styles are used, following
`tests/test_pipeline_reproducibility_component3.py`'s established
convention:

    - A source-level assertion test: reads `conga/preprocess.py` as text
      and asserts the seeded `default_rng`/`permutation` replacement is
      present in the `compare_distance_distributions` branch, and that
      the two intentionally-unseeded calls remain unseeded (guarding
      against someone "fixing" them later without updating the scope
      decision in design.md).
    - A runtime test confirming that calling with
      `compare_distance_distributions=True` and a non-default
      `random_seed` produces the same `inds` as
      `np.random.default_rng(random_seed).permutation(adata.shape[0])[:nrandom]`
      computed directly in the test.

Requirements: 4.1
"""

import inspect
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

from conga.preprocess import calc_X_pca_gex_including_protein_features
from conga import util


REPO_ROOT = Path(__file__).resolve().parent.parent
PREPROCESS_SOURCE_PATH = REPO_ROOT / 'conga' / 'preprocess.py'
RANDOM_SEED = 42
NON_DEFAULT_RANDOM_SEED = 1234

# Protein (antibody-capture) feature names using prefixes NOT in the
# function's default `exclude_protein_prefixes` list (['HTO', 'Hashtag',
# 'Va7.2', 'TCRV', 'TCRv']), so `prot_mask` is non-empty and the
# `assert np.sum(prot_mask) > 0` guard inside the function passes.
_PROTEIN_FEATURES = ['CD3_PROT', 'CD4_PROT', 'CD8_PROT', 'CD19_PROT', 'CD56_PROT']


def _function_source() -> str:
    source = PREPROCESS_SOURCE_PATH.read_text()
    start = source.index('def calc_X_pca_gex_including_protein_features(')
    # the next top-level `def ` after the function start marks the end
    # of this function's body
    end = source.index('\ndef cluster_and_tsne_and_umap(', start)
    return source[start:end]


def _make_gex_counts(n_cells: int, n_genes: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Build a small, dense integer counts matrix with no all-zero rows
    or columns, matching the fixture-construction convention used in
    `tests/test_pipeline_reproducibility_component1.py`/`component3.py`.
    """
    gene_means = rng.uniform(low=0.5, high=40.0, size=n_genes)
    counts = rng.poisson(lam=gene_means, size=(n_cells, n_genes)).astype(np.float64)
    counts += 1.0
    return counts


def _make_adata_with_protein_features(
        n_cells: int = 20, n_genes: int = 30,
        random_seed: int = RANDOM_SEED) -> ad.AnnData:
    """Build a minimal AnnData whose `adata.raw` contains both
    gene-expression and antibody-capture (protein) features, with a
    `feature_types` var column so `util.get_feature_types_varname` and
    the function's own `prot_mask` logic succeed.

    `adata.uns['raw_matrix_is_logged']` is set True so
    `normalize_and_log_the_raw_matrix` (called unconditionally inside
    `calc_X_pca_gex_including_protein_features`) returns immediately
    rather than requiring a biologically realistic raw-counts matrix.
    """
    rng = np.random.default_rng(random_seed)
    gex_counts = _make_gex_counts(n_cells, n_genes, rng)
    prot_counts = rng.poisson(
        lam=rng.uniform(low=5.0, high=100.0, size=len(_PROTEIN_FEATURES)),
        size=(n_cells, len(_PROTEIN_FEATURES)),
    ).astype(np.float64) + 1.0

    obs = pd.DataFrame(index=[f'cell{i}' for i in range(n_cells)])

    # adata.X / adata.var: GEX features only (post-raw-split shape, as
    # the real pipeline has it by the time this function runs). The
    # function looks up `feature_types` varname via
    # `util.get_feature_types_varname(adata)` (i.e. on `adata.var`, not
    # `adata.raw.var`) and then indexes `adata.raw.var` with that same
    # column name, so `adata.var` needs a `feature_types` column too.
    gex_var = pd.DataFrame(
        {'feature_types': ['Gene Expression'] * n_genes},
        index=[f'GENE{i}' for i in range(n_genes)],
    )
    adata = ad.AnnData(X=sp.csr_matrix(gex_counts), obs=obs, var=gex_var)

    # adata.raw: GEX + protein features combined, with a feature_types
    # column distinguishing them, matching the real 10x-derived layout
    # that `get_feature_types_varname`/`prot_mask` expect.
    raw_var = pd.DataFrame(
        {
            'feature_types': (
                ['Gene Expression'] * n_genes
                + ['Antibody Capture'] * len(_PROTEIN_FEATURES)
            ),
        },
        index=[f'GENE{i}' for i in range(n_genes)] + _PROTEIN_FEATURES,
    )
    raw_X = np.hstack([gex_counts, prot_counts])
    raw_adata = ad.AnnData(X=sp.csr_matrix(raw_X), obs=obs.copy(), var=raw_var)
    adata.raw = raw_adata
    adata.uns['raw_matrix_is_logged'] = True
    return adata


class TestCalcXPcaGexIncludingProteinFeaturesSourceWiresRandomSeed:
    """Source-level assertion test confirming the diagnostic-sampling
    call site wires `random_seed` through, and that the two
    intentionally-unseeded calls (`sc.tl.pca` and `PCA()`) remain
    unseeded.
    """

    def test_signature_has_random_seed_as_last_default_param(self):
        source = _function_source()
        signature_region = source[:source.index('):') + 2]
        assert 'random_seed=util.DEFAULT_RANDOM_SEED' in signature_region

    def test_default_random_seed_is_util_default_random_seed(self):
        sig = inspect.signature(calc_X_pca_gex_including_protein_features)
        assert sig.parameters['random_seed'].default == util.DEFAULT_RANDOM_SEED

    def test_random_seed_is_last_parameter(self):
        sig = inspect.signature(calc_X_pca_gex_including_protein_features)
        assert list(sig.parameters.keys())[-1] == 'random_seed'

    def test_diagnostic_branch_wires_seeded_default_rng(self):
        source = _function_source()
        assert (
            "nrandom = min(1000, num_clones)\n"
            "        rng = np.random.default_rng(random_seed)\n"
            "        inds = rng.permutation(adata.shape[0])[:nrandom]"
            in source
        )
        # the old unseeded call must be fully gone, not merely shadowed
        assert 'np.random.permutation(adata.shape[0])' not in source

    def test_gex_pca_call_remains_unseeded(self):
        """design.md Component 4 scope note: this call is not named by
        any acceptance criterion in Requirement 1 and must stay
        unseeded. This guards against someone "fixing" it later without
        updating the scope decision.
        """
        source = _function_source()
        assert (
            "sc.tl.pca(adata, svd_solver='arpack', n_comps=n_components_gex)"
            in source
        )
        assert (
            "sc.tl.pca(adata, svd_solver='arpack', n_comps=n_components_gex,\n"
            "              random_state=random_seed)"
            not in source
        )
        assert (
            "sc.tl.pca(adata, svd_solver='arpack', n_comps=n_components_gex, random_state=random_seed)"
            not in source
        )

    def test_protein_pca_call_remains_unseeded(self):
        """design.md Component 4 scope note: the standalone sklearn
        `PCA()` call for protein-feature PCA is likewise not named by
        any acceptance criterion in Requirement 1 and must stay
        unseeded (no `random_state` kwarg at all).
        """
        source = _function_source()
        assert 'pca = PCA()' in source
        assert 'random_state' not in source.split('pca = PCA()')[1].split('\n')[0]


class TestCalcXPcaGexIncludingProteinFeaturesRuntimeSeedWiring:
    """Runtime test confirming that calling with
    `compare_distance_distributions=True` and a non-default
    `random_seed` produces the same `inds` as
    `np.random.default_rng(random_seed).permutation(adata.shape[0])[:nrandom]`
    computed directly in the test.
    """

    def test_same_seed_produces_identical_diagnostic_output(self, capsys):
        """Two calls with the same non-default seed must sample the
        same `inds`, and therefore print identical diagnostic distance
        statistics -- the externally observable proxy for `inds`
        equality, since the function has no return value.
        """
        adata1 = _make_adata_with_protein_features()
        adata2 = _make_adata_with_protein_features()

        calc_X_pca_gex_including_protein_features(
            adata1, n_components_gex=5, n_components_prot=3,
            compare_distance_distributions=True,
            random_seed=NON_DEFAULT_RANDOM_SEED,
        )
        out1 = capsys.readouterr().out

        calc_X_pca_gex_including_protein_features(
            adata2, n_components_gex=5, n_components_prot=3,
            compare_distance_distributions=True,
            random_seed=NON_DEFAULT_RANDOM_SEED,
        )
        out2 = capsys.readouterr().out

        def _dist_lines(text):
            return [
                line for line in text.splitlines()
                if line.startswith('gex_dists:')
                or line.startswith('prot_dists:')
                or line.startswith('dist_ratio:')
            ]

        assert _dist_lines(out1) == _dist_lines(out2)

    def test_different_seeds_select_different_inds(self):
        """A direct equivalence check: `np.random.default_rng(seed)`
        with two different seeds produces different permutations for a
        large-enough `n_cells`, confirming the seed actually controls
        which `inds` are drawn (and thus that `random_seed` is a
        meaningful input to the diagnostic branch, not a dead
        parameter).
        """
        n_cells = 20
        nrandom = min(1000, n_cells)
        inds_a = np.random.default_rng(RANDOM_SEED).permutation(n_cells)[:nrandom]
        inds_b = np.random.default_rng(
            NON_DEFAULT_RANDOM_SEED).permutation(n_cells)[:nrandom]
        assert not np.array_equal(inds_a, inds_b)

    def test_inds_match_default_rng_permutation_directly(self):
        """Directly confirms the function's internal `inds` equals
        `np.random.default_rng(random_seed).permutation(adata.shape[0])[:nrandom]`
        by patching `conga.preprocess.np.random.default_rng` with a
        wrapper that only intercepts the call made with the exact
        `random_seed` passed to the function under test, delegating any
        other seed value (e.g. integer seeds sklearn's PCA/ARPACK code
        may separately pass to its own internal `default_rng` calls) to
        the real implementation untouched. This avoids the module-wide
        hijack that would otherwise break unrelated random draws inside
        `sc.tl.pca`/`PCA().fit_transform`, since `conga.preprocess`'s
        `np` is the same module object those libraries use internally.
        """
        import conga.preprocess as preprocess_module

        captured = {}
        real_default_rng = np.random.default_rng

        def spying_default_rng(seed):
            gen = real_default_rng(seed)
            if seed == NON_DEFAULT_RANDOM_SEED:
                real_permutation = gen.permutation

                class _Proxy:
                    def permutation(self, *args, **kwargs):
                        result = real_permutation(*args, **kwargs)
                        captured['permutation_result'] = result
                        return result

                    def __getattr__(self, name):
                        return getattr(gen, name)

                return _Proxy()
            return gen

        adata = _make_adata_with_protein_features()
        n_cells = adata.shape[0]
        nrandom = min(1000, n_cells)

        with mock.patch.object(
                preprocess_module.np.random, 'default_rng', spying_default_rng):
            calc_X_pca_gex_including_protein_features(
                adata, n_components_gex=5, n_components_prot=3,
                compare_distance_distributions=True,
                random_seed=NON_DEFAULT_RANDOM_SEED,
            )

        assert 'permutation_result' in captured
        actual_inds = captured['permutation_result'][:nrandom]
        expected_inds = real_default_rng(
            NON_DEFAULT_RANDOM_SEED).permutation(n_cells)[:nrandom]
        assert np.array_equal(actual_inds, expected_inds)
