# CoNGA Examples

Shell scripts for running `conga` on three publicly available 10X
genomics datasets can be found in the `examples/` directory:
[`examples/setup_all.bash`](../examples/setup_all.bash) which preprocesses the
clonotype data, and [`examples/run_all.bash`](../examples/run_all.bash) which calls
`scripts/run_conga.py` on the three datasets. Here we discuss some of the
key outputs. For details on the algorithm and additional details on the plots,
refer to our [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2020.06.04.134536v1).
Here we focus on images, but the results of the different analysis
modes are also saved in a variety of tab-separated-values (`*.tsv`) files.
You can download a [zip file](https://www.dropbox.com/s/r7rpsftbtxl89y5/conga_example_datasets_v1.zip?dl=0)
containing all three datasets. Or access them individually from the 10x website
at the locations given below.

## Human PBMC dataset

```
# DOWNLOAD FROM:
# https://support.10xgenomics.com/single-cell-vdj/datasets/3.1.0/vdj_v1_hs_pbmc3

# SETUP
/home/pbradley/anaconda2/envs/scanpy_new/bin/python /home/pbradley/gitrepos/conga/scripts/setup_10x_for_conga.py --organism human --filtered_contig_annotations_csvfile ./conga_example_datasets/vdj_v1_hs_pbmc3_t_filtered_contig_annotations.csv

# RUN
/home/pbradley/anaconda2/envs/scanpy_new/bin/python /home/pbradley/gitrepos/conga/scripts/run_conga.py --all --organism human --clones_file ./conga_example_datasets/vdj_v1_hs_pbmc3_t_filtered_contig_annotations_tcrdist_clones.tsv --gex_data ./conga_example_datasets/vdj_v1_hs_pbmc3_5gex_filtered_gene_bc_matrices_h5.h5 --gex_data_type 10x_h5 --outfile_prefix tcr_hs_pbmc

```

NOTE: In the `RUN` command above the `--all` argument to `run_conga.py` is a shorthand
for running all the current major modes of analysis, and is equivalent (right now)
to `--graph_vs_graph --graph_vs_gex_features --graph_vs_tcr_features --cluster_vs_cluster --find_hotspot_features --find_gex_cluster_degs --make_tcrdist_trees`.

In looking at the CoNGA results a good place to start is with the summary image,
which will be named something
like `OUTFILE_PREFIX_summary.png` where `OUTFILE_PREFIX` was the `--outfile_prefix`
command line argument passed to `run_conga.py`. This image shows 2D projections of the
clonotypes in the dataset, based on distances in gene expression (GEX) space in
the top row and in TCR/BCR space in the bottom row. The left panels are colored
by clonotype clusters (GEX clusters on top and TCR clusters on the bottom);
the middle panels are colored by CoNGA score, which reflects overlap
between the GEX and TCR neighbor graphs on a per-clonotype basis; and the right
panels are overlaid with the top graph-vs-feature hits: TCR features that are
enriched in GEX graph neighborhoods on top, and genes that are differentially
expressed in TCR graph neighborhoods on the bottom.

![summary](../_images/tcr_hs_pbmc_summary.png)

To gain greater insight into the clonotypes with significant CoNGA scores,
the software groups them by their joint GEX and TCR cluster assignments and
displays logos that summarize gene expression and TCR V/J gene usage and CDR3
sequence features for groups with size above a threshold (by default, at least
5 clonotypes and .1% of the overall dataset size). These logos are shown in the
plot `OUTFILE_PREFIX_bicluster_logos.png`. In this image the top panels are
show the clusters, conga scores, and conga hits (clonotypes with conga score<1)
in the GEX (left 3 panels) and TCR/BCR (right 3 panels) 2D lanscapes. Below
them are shown a sequence of thumbnail GEX projection scatter plots colored by
selected (and user configurable) marker genes; the top set are colored by raw
(normalized for cell reads and log+1 transformed) expression and in the bottom
set those values are Z-standardized and averaged over GEX graph neighborhoods,
to smooth and highlight overall trends. Below the snapshots are the rows of
cluster-pair (or 'bicluster') logos.

![clusters](../_images/tcr_hs_pbmc_bicluster_logos.png)

The results of the graph-vs-features analysis are summarized in a number of graphs
with names that include text like `graph_vs_XXX_features`. For example, the
top few TCR/GEX feature hits are shown projected onto the other landscape
(ie, TCR features onto the GEX landscape and GEX features onto the TCR landscape)
in plots that end in `_panels.png`, like the
`OUTFILE_PREFIX_gex_nbr_graph_vs_tcr_features_panels.png` shown below, where we can
see MAIT-cell features as well as features that differ between CD4 and CD8 T cells
(charge, TRBV20 and TRAJ30 usage). You can identify the CD4/CD8 cells by looking
at the CD4/CD8a/CD8b gene thumbnails in the cluster logos plot above.  

![gex_graph_vs_tcr_features](../_images/tcr_hs_pbmc_gex_nbr_graph_vs_tcr_features_panels.png)

In addition to the graph-vs-feature analysis described in the preprint,
we recently implemented a first, experimental version of the
[HotSpot](https://github.com/YosefLab/Hotspot) method
developed by the Yosef lab
([bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2020.02.06.937805v1)).
Hotspot identifies numerical features that respect a given neighbor graph
structure, so we can use it to identify GEX features that correlate with
the TCR similarity graph and TCR/BCR features that correlate with the the
GEX similarity graph. These features are saved to `tsv` files and visualized
in a number of images, including `seaborn clustermap` plots if you have the
`seaborn` package installed. For example the image with the rather long
name
`OUTFILE_PREFIX_0.100_nbrs_combo_hotspot_features_vs_gex_clustermap_lessredundant.png`
shows the top combined GEX and TCR features (rows) versus clonotypes (columns), where the
clonotypes (columns) are ordered based on a hierarchical clustering dendrogram
built using GEX similarity (`_vs_gex_clustermap` in filename) and the scores
are averaged over GEX neighborhoods
(so that visible trends need to be consistent the GEX graph structure, and to smooth
noise). Features that are highly correlated with another, more significant
feature are filtered out and listed in the tiny blue text on the left (`_lessredundant` in
the filename). In this plot we can see MAIT, CD4, and CD8 groupings and many of the
same features as in the graph-vs-features analysis above. Since we have only
implemented the simplest model right now, the P-values may not be completely
trustworthy, and we suggest focusing on the results from larger neighbor
graphs (here we are showing the results for the graph with 0.1 neighbor fraction,
`_0.100_nbrs` in the filename,
ie where each clonotype is connected to the nearest 10 percent of the dataset).
The colors along the top of the matrix show the GEX cluster assignments of
each clonotype. The two columns of colors along the left-hand side of the matrix
show (left column) the P-value and (right column) the feature type (GEX vs TCR)
of the feature corresponding to that row (P-values and feature types are also
given in the row names along the right-hand side of the matrix). '[+N]' in the
row name means that N additional highly-correlated features were filtered out;
their names will be listed in the tiny blue text along the left-hand side of the
figure, listed below the name of the representative feature. (Some of these images
are big and very detailed-- downloading or opening in a separate tab and zooming
in may be helpful).

![clustermap](../_images/tcr_hs_pbmc_0.100_nbrs_combo_hotspot_features_vs_gex_clustermap_lessredundant.png)

## Mouse PBMC dataset

```
# DOWNLOAD FROM:
# https://support.10xgenomics.com/single-cell-vdj/datasets/3.0.0/vdj_v1_mm_balbc_pbmc_5gex

# SETUP
/home/pbradley/anaconda2/envs/scanpy_new/bin/python /home/pbradley/gitrepos/conga/scripts/setup_10x_for_conga.py --organism mouse --filtered_contig_annotations_csvfile ./conga_example_datasets/vdj_v1_mm_balbc_pbmc_t_filtered_contig_annotations.csv

# RUN
/home/pbradley/anaconda2/envs/scanpy_new/bin/python /home/pbradley/gitrepos/conga/scripts/run_conga.py --all --organism mouse --clones_file ./conga_example_datasets/vdj_v1_mm_balbc_pbmc_t_filtered_contig_annotations_tcrdist_clones.tsv --gex_data ./conga_example_datasets/vdj_v1_mm_balbc_pbmc_5gex_filtered_gene_bc_matrices_h5.h5 --gex_data_type 10x_h5 --outfile_prefix tcr_mm_pbmc
```

The summary image, where we can see a tight cluster of conga hits that turn out
to be iNKT cells (inkt TCR feature enriched in top right panel), some CD8-enriched
genes (TRAV16N and TRBV29), and the EphB6 gene which turns out to be correlated
with usage of the TRBV31 gene:
![summary](../_images/tcr_mm_pbmc_summary.png)

Here the cluster logo image shows iNKT cells and some CD8-positive clusters that
likely reflect TCR sequence features shared by CD8s.

![biclusters](../_images/tcr_mm_pbmc_bicluster_logos.png)

The top few GEX features that are enriched in TCR neighborhoods: some iNKT genes
and the EphB6 gene showing localized expression in the TCR landscape projection
(in the TRBV31 expressing clonotypes).

![gex_features](../_images/tcr_mm_pbmc_tcr_nbr_graph_vs_gex_features_panels.png)

In the hotspot clustermap showing features versus TCR-arranged clonotypes we
can see the correlation between EphB6 and TRBV31 nicely. The color rows along the
top of the matrix show (from top to bottom) the TCR cluster assignment and the
TRAV, TRAJ, TRBV, and TRBJ gene segment usage patterns (color key for the top
few genes is shown at the top of the column dendrogram).

![hotspot](../_images/tcr_mm_pbmc_0.100_nbrs_combo_hotspot_features_vs_tcr_clustermap_lessredundant.png)


## Human Melanoma B cell dataset

```
# DOWNLOAD FROM:
# https://support.10xgenomics.com/single-cell-vdj/datasets/4.0.0/sc5p_v1p1_hs_melanoma_10k

# SETUP
/home/pbradley/anaconda2/envs/scanpy_new/bin/python /home/pbradley/gitrepos/conga/scripts/setup_10x_for_conga.py --organism human_ig --filtered_contig_annotations_csvfile ./conga_example_datasets/sc5p_v1p1_hs_melanoma_10k_b_filtered_contig_annotations.csv --condense_clonotypes_by_tcrdist --tcrdist_threshold_for_condensing 50

# RUN
/home/pbradley/anaconda2/envs/scanpy_new/bin/python /home/pbradley/gitrepos/conga/scripts/run_conga.py --all --organism human_ig --clones_file ./conga_example_datasets/sc5p_v1p1_hs_melanoma_10k_b_filtered_contig_annotations_tcrdist_clones_condensed.tsv --gex_data ./conga_example_datasets/sc5p_v1p1_hs_melanoma_10k_filtered_feature_bc_matrix.h5 --gex_data_type 10x_h5 --outfile_prefix bcr_hs_melanoma
```

Two features to note in the commands above: (1) we are passing `--organism human_ig`
to let conga know we are working with BCR data -- note that `human_ig` is CLI-usable
but is not in the vectorized-TCRdist-eligible set, so this dataset takes the KernelPCA
path (see the Organism support and TCR Representations sections in the main [README](../README.md)); (2) in the
setup command we added the flags `--condense_clonotypes_by_tcrdist --tcrdist_threshold_for_condensing 50`,
which trigger merging of 10X clonotypes whose TCRdist (actually BCR dist) is
less than or equal to 50 (ie, we do single-linkage clustering and cut the tree
at a distance threshold of 50). Here the goal is to merge clonally related
families so that GEX/BCR covariation that we detect reflects the correlation
across independent rearrangements. The user could instead apply a more
sophisticated clone identification procedure and modify the `raw_clonotype_id`
column in the contigs csv file, which is where conga gets the clonotype info.

The summary image, where we can see that most of the conga hits are in GEX cluster
2; that there are differences in CDR3 length across the landscape; and some
specific genes that are differentially expressed in TCR graph neighborhoods
(TNFRSF13B, B2M, CRIP1).
![summary](../_images/bcr_hs_melanoma_summary.png)

Logos for the clusters of conga hits, with one cluster of 'naive' clonotypes
with long CDRH3s and high TCL1A expression.

![biclusters](../_images/bcr_hs_melanoma_bicluster_logos.png)

In the hotspot clustermap showing features versus TCR-arranged clonotypes we
can see a breakdown between naive and non-naive features, where IGHJ6 and CDR3 length
are correlated with genes like TCL1A and class II HLA genes, and IGHJ4 is enriched
in the non-naives.

![hotspot](../_images/bcr_hs_melanoma_0.100_nbrs_combo_hotspot_features_vs_tcr_clustermap_lessredundant.png)

CoNGA also makes a tree of all the clonotypes with a conga score less than a
loose threshold value of 10, where we can look for sequence clustering. The branches
are colored by a transformed version of the conga score that maps the threshold
value of 10 to zero (blue) with more significant scores trending toward red at
a value of 1e-8:

![tcrdist_tree](../_images/bcr_hs_melanoma_conga_score_lt_10.0_tcrdist_tree.png)

