# Clonotype Neighbor Graph Analysis (CoNGA) -- developmental

This repository contains the developmental `conga` python package and associated scripts
and workflows that may or may not be stable. Please see the main repo at phbradley/conga for the latest stable version.
Questions and requests can be directed to `pbradley` at `fredhutch` dot `org`.

# Running

Running `conga` on a single-cell dataset is a two- (or more) step process, as outlined below.
Python scripts are provided in the `scripts/` directory but analysis steps can also be accessed interactively
in jupyter notebooks (for example, [a simple pipeline](simple_conga_pipeline.ipynb) and
[Seurat to conga](Seurat_to_Conga.ipynb) in the top directory of this repo)
or in your own python scripts through the interface in the `conga` python package.

1. **SETUP**: The TCR data is converted to a form that can be read by `conga` and then
a matrix of `TCRdist` distances is computed. KernelPCA is applied to this distance
matrix to generate a PC matrix that can be used in clustering and dimensionality reduction. This
is accomplished with the python script `scripts/setup_10x_for_conga.py` for 10x datasets. For example:

```
python conga/scripts/setup_10x_for_conga.py --filtered_contig_annotations_csvfile vdj_v1_hs_pbmc_t_filtered_contig_annotations.csv --organism human
```

2. **ANALYZE**: The `scripts/run_conga.py` script has an implementation of the main pipeline and can be run
as follows:

```
python conga/scripts/run_conga.py --graph_vs_graph --gex_data data/vdj_v1_hs_pbmc_5gex_filtered_gene_bc_matrices_h5.h5 --gex_data_type 10x_h5 --clones_file vdj_v1_hs_pbmc_t_filtered_contig_annotations_tcrdist_clones.tsv --organism human --outfile_prefix tmp_hs_pbmc
```

3. **RE-ANALYZE**: Step 2 will generate a processed `.h5ad` file that contains all the gene expression
and TCR sequence information along with the results of clustering and dimensionality reduction. It can then
be much faster to perform subsequent re-analysis or downstream analysis by "restarting" from those files.

```
python conga/scripts/run_conga.py --restart tmp_hs_pbmc_final.h5ad --graph_vs_tcr_features --graph_vs_gex_features --outfile_prefix tmp_hs_pbmc_restart
```


# Installation
Create conga environment. We prefer using Anaconda.
Stefan's conga_env
```
conda create -n conga_env python=3.6
conda activate conga_env
conda install seaborn scikit-learn statsmodels numba pytables ipython 
conda install -c conda-forge python-igraph leidenalg notebook louvain spyder
pip install scanpy loompy anndata2ri
```
anndata2ri and loompy dependencies are only needed if converting from Seurat to h5ad or loom formats as CoNGA inputs. Anndata2ri requires R on the PATH or assigned to R_HOME for installation.

# svg to png
The `conga` image-making pipeline requires an svg to png conversion. There seem to be a variety of
options for doing this, with the best choice being somewhat platform dependent. We've had good luck with
ImageMagick `convert` (on linux) and Inkscape (on mac). The conversion is handled in the file
`conga/convert_svg_to_png.py`, so you can modify that file if things are not working and you have
a tool installed; `conga` may not be looking in the right place. 
