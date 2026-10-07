"""
Unit tests for Component 6 of the pipeline-reproducibility feature: the
`scripts/run_conga.py` CLI wiring that threads the already-resolved
`args.random_seed` into every call site of the three
`conga/preprocess.py` functions seeded by Tasks 1-3
(`cluster_and_tsne_and_umap`, `calc_tcrdist_nbrs_umap_clusters_cpp`,
`reduce_to_single_cell_per_clone`).

Following `tests/test_run_conga_cli.py`'s existing
`test_call_site_source_wires_batch_integration_not_filter_and_scale`
pattern, this module reads the actual `scripts/run_conga.py` source as text
and asserts `random_seed=args.random_seed` is present at all 7 call sites:

    - `cluster_and_tsne_and_umap` -- 5 call sites:
        1. the `--make_clone_plots` branch
        2. the main post-`reduce_to_single_cell_per_clone` call
        3. the post-`--exclude_mait_and_inkt_cells` subsetting call
        4. the post-`--exclude_gex_clusters` subsetting call
        5. the post-`--subset_to_CD4`/`--subset_to_CD8` call
    - `calc_tcrdist_nbrs_umap_clusters_cpp` -- 1 call site
    - `reduce_to_single_cell_per_clone` -- 1 call site

Three of the five `cluster_and_tsne_and_umap` call sites (#3, #4, #5 above)
share textually identical surrounding code
(`clustering_method=args.clustering_method,\n    clustering_resolution=
args.clustering_resolution,\n    random_seed=args.random_seed)`, modulo
indentation). A simple `in source` substring check would pass even if only
one of the three were actually wired, so this module counts exact
occurrences of each call-site pattern instead of just checking membership,
per this task's explicit instruction.

Requirements: 5.1, 5.2, 5.3
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_CONGA_SCRIPT = REPO_ROOT / 'scripts' / 'run_conga.py'


def _source() -> str:
    return RUN_CONGA_SCRIPT.read_text()


class TestClusterAndTsneAndUmapCallSitesWireRandomSeed:
    """Requirement 5.1: all 5 `cluster_and_tsne_and_umap(...)` call sites
    in `scripts/run_conga.py` must pass `random_seed=args.random_seed`.
    """

    def test_make_clone_plots_call_site_wires_random_seed(self):
        source = _source()
        assert (
            "adata = conga.preprocess.cluster_and_tsne_and_umap(\n"
            "            adata, skip_tcr=True, random_seed=args.random_seed)"
            in source
        )

    def test_main_post_reduce_call_site_wires_random_seed(self):
        source = _source()
        assert (
            "adata = conga.preprocess.cluster_and_tsne_and_umap(\n"
            "        adata, clustering_resolution = clustering_resolution,\n"
            "        clustering_method=args.clustering_method,\n"
            "        random_seed=args.random_seed)"
            in source
        )

    def test_exact_count_of_three_identical_subsetting_call_sites(self):
        """Call sites #3 (--exclude_mait_and_inkt_cells), #4
        (--exclude_gex_clusters), and #5 (--subset_to_CD4/--subset_to_CD8)
        share identical surrounding code apart from indentation. This
        asserts the exact count of each indentation variant so that fixing
        only 1 or 2 of the 3 occurrences cannot pass silently.
        """
        source = _source()

        # call site #3 is nested one level deeper (inside an `if mask...`
        # block), so its continuation lines carry extra leading
        # whitespace relative to call sites #4 and #5.
        indented_pattern = (
            "adata = conga.preprocess.cluster_and_tsne_and_umap(\n"
            "            adata, clustering_method=args.clustering_method,\n"
            "            clustering_resolution=args.clustering_resolution,\n"
            "            random_seed=args.random_seed)"
        )
        unindented_pattern = (
            "adata = conga.preprocess.cluster_and_tsne_and_umap(\n"
            "        adata, clustering_method=args.clustering_method,\n"
            "        clustering_resolution=args.clustering_resolution,\n"
            "        random_seed=args.random_seed)"
        )

        assert source.count(indented_pattern) == 1, (
            'expected exactly 1 occurrence of the indented '
            '(--exclude_mait_and_inkt_cells) call-site pattern wired with '
            'random_seed=args.random_seed')
        assert source.count(unindented_pattern) == 2, (
            'expected exactly 2 occurrences of the unindented '
            '(--exclude_gex_clusters and --subset_to_CD4/--subset_to_CD8) '
            'call-site pattern wired with random_seed=args.random_seed')

    def test_no_unwired_cluster_and_tsne_and_umap_call_sites_remain(self):
        """Guards against a regression where only some occurrences of the
        shared `clustering_resolution=args.clustering_resolution)` closing
        pattern were updated: every occurrence of that argument in a
        `cluster_and_tsne_and_umap` call must be immediately followed (on
        the next call-site line) by `random_seed=args.random_seed)`, never
        by a bare closing `)` on its own.
        """
        source = _source()
        unwired = 'clustering_resolution=args.clustering_resolution)'
        assert unwired not in source, (
            'found a cluster_and_tsne_and_umap call site still closing '
            'without random_seed=args.random_seed')

    def test_total_random_seed_wired_cluster_and_tsne_and_umap_sites(self):
        """Sanity count across all 5 call sites combined: 1 (make_clone_plots)
        + 1 (main) + 3 (identical-pattern subsetting sites) = 5 total
        `cluster_and_tsne_and_umap(` invocations, each paired with
        `random_seed=args.random_seed` somewhere in its own call.
        """
        source = _source()
        total_calls = source.count(
            'conga.preprocess.cluster_and_tsne_and_umap(')
        assert total_calls == 5
        # every one of those 5 calls contributes exactly one
        # `random_seed=args.random_seed)` closing line among the patterns
        # already asserted individually above; cross-check the aggregate.
        total_wired_closings = (
            source.count('skip_tcr=True, random_seed=args.random_seed)')
            + source.count(
                'clustering_method=args.clustering_method,\n'
                '        random_seed=args.random_seed)')
            + source.count(
                'clustering_resolution=args.clustering_resolution,\n'
                '            random_seed=args.random_seed)')
            + source.count(
                'clustering_resolution=args.clustering_resolution,\n'
                '        random_seed=args.random_seed)')
        )
        assert total_wired_closings == 5


class TestCalcTcrdistNbrsUmapClustersCppCallSiteWiresRandomSeed:
    """Requirement 5.2: the single
    `calc_tcrdist_nbrs_umap_clusters_cpp(...)` call site must pass
    `random_seed=args.random_seed`.
    """

    def test_call_site_wires_random_seed(self):
        source = _source()
        assert source.count(
            'conga.preprocess.calc_tcrdist_nbrs_umap_clusters_cpp(') == 1
        assert (
            "conga.preprocess.calc_tcrdist_nbrs_umap_clusters_cpp(\n"
            "        adata, num_nbrs,\n"
            "        tmpfile_prefix=args.outfile_prefix,\n"
            "        umap_key_added=umap_key_added,\n"
            "        cluster_key_added=cluster_key_added,\n"
            "        random_seed=args.random_seed)"
            in source
        )


class TestReduceToSingleCellPerCloneCallSiteWiresRandomSeed:
    """The single `reduce_to_single_cell_per_clone(...)` call site must
    pass `random_seed=args.random_seed`, so the CLI's resolved
    `args.random_seed` reaches Component 3's new parameter.
    """

    def test_call_site_wires_random_seed(self):
        source = _source()
        assert source.count(
            'conga.preprocess.reduce_to_single_cell_per_clone(') == 1
        assert (
            "adata = conga.preprocess.reduce_to_single_cell_per_clone(\n"
            "        adata, average_clone_gex=args.average_clone_gex,\n"
            "        random_seed=args.random_seed )"
            in source
        )


class TestRandomSeedDefaultResolutionLogicUnchanged:
    """Requirement 5.3 is already satisfied by pre-existing code that this
    task must not modify. This guards against an accidental edit to that
    resolution block while making the 7 call-site edits above.
    """

    def test_default_resolution_block_present_and_unmodified(self):
        source = _source()
        assert (
            'if args.random_seed is None:\n'
            '    args.random_seed = util.DEFAULT_RANDOM_SEED' in source
        )
