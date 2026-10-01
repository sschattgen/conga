"""
Correctness tests for FaissNeighborSearcher's TCR group exclusion logic.

Regression coverage for the bug where `_search_faiss` queried FAISS for a
FIXED-SIZE candidate pool (`max_neighbors + 1`) and, when `exclude_groups`
was provided, filtered out same-alpha-group / same-beta-group candidates
from that fixed pool without re-querying for more -- padding any shortfall
with -1 instead. Those -1 values then propagated into `neighbors_dict` and
crashed `correlations.py::_make_csr_nbrs` with
`ValueError: negative axis 1 index: -1`.

Since `conga.preprocess.calc_nbrs()` passes `(agroups, bgroups)` to the
neighbor search for BOTH the GEX and TCR branches unconditionally, this bug
affected GEX neighbor search as well, whenever some clonotype's TCR group
(shared alpha or beta chain) was large enough to deplete the fixed top-k
pool returned by FAISS.

See test_data/e2e_batch_integration/diagnose_faiss_minus1.py for the
original diagnostic reproduction.
"""

import numpy as np
import pytest

from conga.neighbors import Backend, FaissNeighborSearcher

try:
    import faiss  # noqa: F401
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

requires_faiss = pytest.mark.skipif(
    not FAISS_AVAILABLE, reason="faiss package not installed"
)


def _make_realistic_group_structure(n_samples, n_agroups, rng):
    """Few alpha-groups (shared TCR alpha chain) spread over many cells,
    all-unique beta-groups -- mirrors a repertoire with some expanded
    clones sharing an alpha chain but distinct beta chains."""
    agroups = rng.integers(0, n_agroups, size=n_samples)
    bgroups = np.arange(n_samples)
    return agroups, bgroups


@requires_faiss
class TestFaissGroupExclusionNoLeakage:
    """FAISS-backed search must never return -1 'placeholder' neighbors
    when genuine non-excluded candidates are available."""

    def test_no_minus1_with_realistic_group_structure(self):
        rng = np.random.default_rng(42)
        n_samples, n_features = 1390, 40
        X = rng.normal(size=(n_samples, n_features)).astype(np.float32)
        agroups, bgroups = _make_realistic_group_structure(n_samples, 40, rng)

        nbr_fracs = [0.01, 0.1]
        searcher = FaissNeighborSearcher(force_backend=Backend.FAISS_CPU)
        result = searcher.search_neighbors(
            X=X,
            nbr_fracs=nbr_fracs,
            exclude_groups=(agroups, bgroups),
            also_calc_nndists=False,
            nbr_frac_for_nndists=None,
            sort_nbrs=False,
            metric='euclidean',
            data_type='gex',
        )

        for frac in nbr_fracs:
            arr = result.neighbors[frac]
            assert np.all(arr != -1), (
                f"Found -1 placeholder neighbors for frac={frac}; "
                "FAISS candidate pool was not widened enough to "
                "avoid padding."
            )
            expected_num_neighbors = min(
                max(1, int(frac * n_samples)), n_samples - 1
            )
            assert arr.shape == (n_samples, expected_num_neighbors)

    def test_no_minus1_with_large_single_group(self):
        """40 alpha-groups is realistic; also check a case where a
        handful of cells share one large alpha-group (bigger than
        max_neighbors) while the rest are singletons."""
        rng = np.random.default_rng(7)
        n_samples, n_features = 500, 20
        X = rng.normal(size=(n_samples, n_features)).astype(np.float32)

        agroups = np.arange(n_samples)
        # Make a group of 120 cells (> max_neighbors for frac=0.1 -> 50)
        # share one alpha-group value.
        agroups[:120] = -1
        bgroups = np.arange(n_samples)

        nbr_fracs = [0.1]
        searcher = FaissNeighborSearcher(force_backend=Backend.FAISS_CPU)
        result = searcher.search_neighbors(
            X=X,
            nbr_fracs=nbr_fracs,
            exclude_groups=(agroups, bgroups),
            also_calc_nndists=False,
            nbr_frac_for_nndists=None,
            sort_nbrs=False,
            metric='euclidean',
            data_type='tcr',
        )
        arr = result.neighbors[0.1]
        assert np.all(arr != -1)

    def test_matches_sklearn_neighbor_sets(self):
        """For a realistic TCR group structure, the FAISS-returned
        neighbor SET per row should match the sklearn (reference)
        neighbor SET exactly (order may legitimately differ due to
        tie-breaking, so compare as sets)."""
        rng = np.random.default_rng(123)
        n_samples, n_features = 600, 30
        X = rng.normal(size=(n_samples, n_features)).astype(np.float32)
        agroups, bgroups = _make_realistic_group_structure(n_samples, 20, rng)

        nbr_fracs = [0.05]
        common_kwargs = dict(
            X=X,
            nbr_fracs=nbr_fracs,
            exclude_groups=(agroups, bgroups),
            also_calc_nndists=False,
            nbr_frac_for_nndists=None,
            sort_nbrs=False,
            metric='euclidean',
            data_type='tcr',
        )

        faiss_searcher = FaissNeighborSearcher(force_backend=Backend.FAISS_CPU)
        faiss_result = faiss_searcher.search_neighbors(**common_kwargs)

        sklearn_searcher = FaissNeighborSearcher(force_backend=Backend.SKLEARN)
        sklearn_result = sklearn_searcher.search_neighbors(**common_kwargs)

        faiss_arr = faiss_result.neighbors[0.05]
        sklearn_arr = sklearn_result.neighbors[0.05]

        assert faiss_arr.shape == sklearn_arr.shape
        assert np.all(faiss_arr != -1)

        for i in range(n_samples):
            faiss_set = set(faiss_arr[i].tolist())
            sklearn_set = set(sklearn_arr[i].tolist())
            assert faiss_set == sklearn_set, (
                f"Row {i}: FAISS neighbor set {faiss_set} != "
                f"sklearn neighbor set {sklearn_set}"
            )

    def test_sklearn_backend_unaffected(self):
        """Explicitly confirm the sklearn reference path still behaves as
        before (no change expected there)."""
        rng = np.random.default_rng(99)
        n_samples, n_features = 300, 15
        X = rng.normal(size=(n_samples, n_features)).astype(np.float32)
        agroups, bgroups = _make_realistic_group_structure(n_samples, 10, rng)

        nbr_fracs = [0.02, 0.1]
        searcher = FaissNeighborSearcher(force_backend=Backend.SKLEARN)
        result = searcher.search_neighbors(
            X=X,
            nbr_fracs=nbr_fracs,
            exclude_groups=(agroups, bgroups),
            also_calc_nndists=False,
            nbr_frac_for_nndists=None,
            sort_nbrs=False,
            metric='euclidean',
            data_type='tcr',
        )
        for frac in nbr_fracs:
            assert np.all(result.neighbors[frac] != -1)
