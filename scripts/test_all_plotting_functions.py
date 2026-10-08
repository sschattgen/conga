#!/usr/bin/env python
"""
Smoke-test harness: runs every plotting.py make_*/plot_* function against
a single processed h5ad file and reports which ones raise exceptions.

This mirrors the real call sequence in scripts/run_conga.py so that each
function receives correctly-shaped prerequisite data (nbrs, conga_scores,
tcr_clumping_pvalues, etc). It does NOT validate plot correctness/content --
only that each function runs to completion without raising. Treat any
exception reported here as a real bug to investigate, the same way the
clustermap bugs were found.

Usage:
    mamba run -n conga-dev python scripts/test_all_plotting_functions.py \
        --adata processed.h5ad --organism human --outfile_prefix /tmp/plot_test

The input h5ad should already be "processed enough" to exercise most
functions meaningfully:
  - adata.uns['organism'] set
  - one row per clonotype (post reduce_to_single_cell_per_clone), OR a raw
    per-cell adata if you also pass --per_cell_adata for make_clone_gex_umap_plots
  - ideally has gone through cluster_and_tsne_and_umap already (clusters_gex,
    clusters_tcr, X_gex_2d, X_tcr_2d present) so most functions don't no-op
  - ideally has gone through graph_vs_graph (conga_scores present) and
    tcr_clumping (tcr_clumping_pvalues present) for full coverage; functions
    that need these will be skipped (not failed) if missing, with a note.
  - IMPORTANT: --nbr_fracs should match (or be a superset of) whatever
    nbr_fracs the h5ad's stored correlation results (graph_vs_features,
    etc.) were originally computed with. run_conga.py's own default is
    [0.01, 0.1] -- if your h5ad came from a real run_conga.py --all run
    with default settings, use that same default here too, otherwise
    make_graph_vs_features_plots will raise KeyError on a missing nbr_frac
    key that the stored results reference.
"""
import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import scanpy as sc

import conga
import conga.plotting as plotting
import conga.preprocess as preprocess
import conga.correlations as correlations
import conga.tcr_clumping as tcr_clumping

results = []  # list of (name, status, detail)

def run(name, fn, *args, **kwargs):
    """Call fn(*args, **kwargs), catch and record any exception."""
    print(f"\n{'='*70}\nRUNNING: {name}\n{'='*70}")
    try:
        ret = fn(*args, **kwargs)
        results.append((name, "OK", ""))
        print(f"OK: {name}")
        return ret
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"
        results.append((name, "FAIL", detail))
        print(f"FAIL: {name}\n{detail}")
        traceback.print_exc()
        return None

def skip(name, reason):
    results.append((name, "SKIP", reason))
    print(f"\nSKIP: {name} ({reason})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adata', required=True,
                         help='Path to processed h5ad file (one row per '
                              'clonotype, ideally post clustering/'
                              'graph_vs_graph/tcr_clumping)')
    parser.add_argument('--organism', default=None,
                         help='Override adata.uns["organism"] if not set')
    parser.add_argument('--outfile_prefix', default='/tmp/conga_plot_test',
                         help='Prefix for all output files written during '
                              'this test run')
    parser.add_argument('--nbr_fracs', type=float, nargs='*',
                         default=[0.01, 0.1],
                         help='Neighbor fractions used to build nbrs via '
                              'conga.preprocess.calc_nbrs. Should match '
                              'whatever nbr_fracs produced any correlation '
                              'results already stored in the h5ad (real '
                              'run_conga.py default: 0.01 0.1)')
    parser.add_argument('--min_cluster_size', type=int, default=None,
                         help='min_cluster_size for make_graph_vs_graph_logos '
                              '/ make_tcr_clumping_plots. Default: mirrors '
                              'run_conga.py min_cluster_size_fraction logic '
                              '(0.1%% of num_clones, floor 3)')
    args = parser.parse_args()

    print(f"Loading {args.adata} ...")
    adata = sc.read_h5ad(args.adata)

    if args.organism is not None:
        adata.uns['organism'] = args.organism
    if 'organism' not in adata.uns:
        print("ERROR: adata.uns['organism'] not set; pass --organism")
        sys.exit(1)

    organism = adata.uns['organism']
    outfile_prefix = args.outfile_prefix
    Path(outfile_prefix).parent.mkdir(parents=True, exist_ok=True)

    nbr_fracs = sorted(args.nbr_fracs)
    min_cluster_size = args.min_cluster_size
    if min_cluster_size is None:
        min_cluster_size = max(3, int(0.5 + 0.001 * adata.shape[0]))

    run("preprocess.setup_uns_dicts", preprocess.util.setup_uns_dicts, adata)

    has_clusters = ('clusters_gex' in adata.obs and 'clusters_tcr' in adata.obs
                     and 'X_gex_2d' in adata.obsm and 'X_tcr_2d' in adata.obsm)
    has_conga_scores = 'conga_scores' in adata.obs
    has_batch_keys = 'batch_keys' in adata.uns

    print(f"\nFixture summary: has_clusters={has_clusters} "
          f"has_conga_scores={has_conga_scores} "
          f"has_tcr_clumping={'tcr_clumping_pvalues' in adata.obs} "
          f"has_batch_keys={has_batch_keys} "
          f"nbr_fracs={nbr_fracs} min_cluster_size={min_cluster_size}")

    # ------------------------------------------------------------------
    # make_clone_gex_umap_plots -- note: real usage calls this on a PER-CELL
    # adata before clonotype reduction. If this adata is already reduced,
    # this still runs structurally the same for the purposes of this smoke
    # test, just not representative of the real intended input.
    # ------------------------------------------------------------------
    run("make_clone_gex_umap_plots", plotting.make_clone_gex_umap_plots,
        adata, outfile_prefix)

    # ------------------------------------------------------------------
    # nbrs_gex / nbrs_tcr / all_nbrs -- required by several functions below
    # ------------------------------------------------------------------
    all_nbrs = None
    nbrs_gex = nbrs_tcr = None
    if has_clusters:
        try:
            print(f"\nBuilding neighbor graphs via calc_nbrs(nbr_fracs="
                  f"{nbr_fracs}) ...")
            all_nbrs, nndists_gex, nndists_tcr = preprocess.calc_nbrs(
                adata, nbr_fracs, also_calc_nndists=True,
                nbr_frac_for_nndists=min(nbr_fracs))
            nbrs_gex, nbrs_tcr = all_nbrs[max(nbr_fracs)]
            results.append(("calc_nbrs", "OK", ""))
        except Exception as e:
            detail = f"{type(e).__name__}: {e}"
            results.append(("calc_nbrs", "FAIL", detail))
            print(f"FAIL: calc_nbrs\n{detail}")
            traceback.print_exc()
    else:
        skip("calc_nbrs", "adata missing clusters_gex/clusters_tcr/X_gex_2d/"
                           "X_tcr_2d; run cluster_and_tsne_and_umap first")

    # ------------------------------------------------------------------
    # make_tcr_db_match_plot -- needs TCR_DB_MATCH results in adata.uns;
    # run the matcher first if not already present (human only, built-in db)
    # ------------------------------------------------------------------
    if organism == 'human':
        if conga.tags.TCR_DB_MATCH not in adata.uns.get('conga_results', {}):
            run("tcr_clumping.match_adata_tcrs_to_db_tcrs (setup)",
                tcr_clumping.match_adata_tcrs_to_db_tcrs, adata)
        run("make_tcr_db_match_plot", plotting.make_tcr_db_match_plot,
            adata, outfile_prefix)
    else:
        skip("make_tcr_db_match_plot",
             f"organism={organism} has no built-in literature db")

    # ------------------------------------------------------------------
    # make_tcr_clumping_plots -- needs tcr_clumping results + nbrs_gex.
    # Signature: (adata, nbrs_gex, nbrs_tcr, outfile_prefix,
    #             min_cluster_size_for_logos=3, ...)
    # ------------------------------------------------------------------
    if nbrs_gex is not None:
        if conga.tags.TCR_CLUMPING not in adata.uns.get('conga_results', {}):
            run("tcr_clumping.assess_tcr_clumping (setup)",
                tcr_clumping.assess_tcr_clumping, adata)
        run("make_tcr_clumping_plots", plotting.make_tcr_clumping_plots,
            adata, nbrs_gex, nbrs_tcr, outfile_prefix,
            min_cluster_size_for_logos=min_cluster_size)
    else:
        skip("make_tcr_clumping_plots", "nbrs_gex unavailable (see calc_nbrs)")

    # ------------------------------------------------------------------
    # make_graph_vs_graph_logos / make_summary_figure / make_logo_plots
    # (make_logo_plots is called internally by make_graph_vs_graph_logos)
    # Signature: (adata, outfile_prefix, min_cluster_size, nbrs_gex,
    #             nbrs_tcr, ...)
    # ------------------------------------------------------------------
    did_graph_vs_graph = False
    if nbrs_gex is not None:
        if not has_conga_scores:
            run("correlations.run_graph_vs_graph (setup)",
                correlations.run_graph_vs_graph, adata, all_nbrs)
            has_conga_scores = 'conga_scores' in adata.obs
        if has_conga_scores:
            run("make_graph_vs_graph_logos", plotting.make_graph_vs_graph_logos,
                adata, outfile_prefix, min_cluster_size, nbrs_gex, nbrs_tcr)
            did_graph_vs_graph = True
        else:
            skip("make_graph_vs_graph_logos", "conga_scores still missing "
                                               "after run_graph_vs_graph")
    else:
        skip("make_graph_vs_graph_logos", "nbrs_gex unavailable")

    # ------------------------------------------------------------------
    # make_graph_vs_features_plots -- needs all_nbrs keyed by whatever
    # nbr_fracs the stored correlation results in adata.uns reference.
    # ------------------------------------------------------------------
    did_graph_vs_features = False
    if all_nbrs is not None:
        run("make_graph_vs_features_plots", plotting.make_graph_vs_features_plots,
            adata, all_nbrs, outfile_prefix, clustermap_max_type_features=25)
        did_graph_vs_features = True
    else:
        skip("make_graph_vs_features_plots", "all_nbrs unavailable")

    # ------------------------------------------------------------------
    # make_summary_figure -- real usage only calls this when BOTH
    # graph_vs_graph and graph_vs_features were run
    # ------------------------------------------------------------------
    if did_graph_vs_graph and did_graph_vs_features:
        run("make_summary_figure", plotting.make_summary_figure,
            adata, outfile_prefix)
    else:
        skip("make_summary_figure",
             "requires both graph_vs_graph and graph_vs_features to have run")

    # ------------------------------------------------------------------
    # make_tcrdist_trees / make_tcrdist_tree_for_conga_score_threshold
    # ------------------------------------------------------------------
    if 'clusters_gex' in adata.obs:
        run("make_tcrdist_trees", plotting.make_tcrdist_trees,
            adata, outfile_prefix, group_by='clusters_gex')
    else:
        skip("make_tcrdist_trees", "clusters_gex missing")

    if has_conga_scores:
        run("make_tcrdist_tree_for_conga_score_threshold",
            plotting.make_tcrdist_tree_for_conga_score_threshold,
            adata, 10., outfile_prefix)
    else:
        skip("make_tcrdist_tree_for_conga_score_threshold",
             "conga_scores missing")

    # ------------------------------------------------------------------
    # make_hotspot_plots
    # ------------------------------------------------------------------
    if all_nbrs is not None:
        run("make_hotspot_plots", plotting.make_hotspot_plots,
            adata, all_nbrs, outfile_prefix, make_raw_feature_plots=False)
    else:
        skip("make_hotspot_plots", "all_nbrs unavailable")

    # ------------------------------------------------------------------
    # make_batch_colored_umaps / make_clone_batch_clustermaps
    # ------------------------------------------------------------------
    if has_batch_keys:
        run("make_batch_colored_umaps", plotting.make_batch_colored_umaps,
            adata, outfile_prefix)

        conga_scores_arr = (adata.obs['conga_scores'].to_numpy()
                             if has_conga_scores else None)
        tcr_clumping_arr = (adata.obs['tcr_clumping_pvalues'].to_numpy()
                             if 'tcr_clumping_pvalues' in adata.obs else None)
        run("make_clone_batch_clustermaps", plotting.make_clone_batch_clustermaps,
            adata, outfile_prefix, adata.uns['batch_keys'],
            conga_scores=conga_scores_arr,
            tcr_clumping_pvalues=tcr_clumping_arr,
            batch_bias_results=None)
    else:
        skip("make_batch_colored_umaps", "adata.uns['batch_keys'] not set")
        skip("make_clone_batch_clustermaps", "adata.uns['batch_keys'] not set")

    # ------------------------------------------------------------------
    # make_html_summary -- safe to call last regardless of what ran
    # ------------------------------------------------------------------
    run("make_html_summary", plotting.make_html_summary,
        adata, outfile_prefix + '_results_summary.html',
        command_string='test_all_plotting_functions.py', title='plot_test')

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------
    print(f"\n\n{'='*70}\nSUMMARY\n{'='*70}")
    n_ok = sum(1 for _, s, _ in results if s == "OK")
    n_fail = sum(1 for _, s, _ in results if s == "FAIL")
    n_skip = sum(1 for _, s, _ in results if s == "SKIP")
    for name, status, detail in results:
        line = f"{status:5s}  {name}"
        if detail:
            line += f"  -- {detail}"
        print(line)
    print(f"\n{n_ok} OK, {n_fail} FAIL, {n_skip} SKIP "
          f"(total {len(results)})")

    if n_fail:
        sys.exit(1)

if __name__ == '__main__':
    main()
