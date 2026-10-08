# Clonotype Neighbor Graph Analysis (CoNGA) -- version 0.2.0

This repository contains the `conga` python package and associated scripts
and workflows. `conga` was developed to detect correlation between
T cell gene expression profile and TCR sequence in single-cell datasets.
We have since added support for gamma delta TCRs and for B cells, too.

Further details on `conga` can be found in the Nature Biotechnology manuscript
**"Integrating T cell receptor sequences and transcriptional profiles by clonotype neighbor graph analysis (CoNGA)"**
by Stefan A. Schattgen, Kate Guion, Jeremy Chase Crawford, Aisha Souquette, Alvaro Martinez Barrio, Michael J.T. Stubbington,
Paul G. Thomas, and Philip Bradley, accessible here:
https://www.nature.com/articles/s41587-021-00989-2
(original BioRxiv preprint
[here](https://www.biorxiv.org/content/10.1101/2020.06.04.134536v1)).

`conga` is in active development right now so the interface may change in
the next few months. Questions and requests can be directed to `pbradley` at `fredhutch` dot `org` and/or
`sschattg` at `fredhutch` dot `org`.

## Organism support

"Which organisms does CoNGA support?" doesn't have one answer -- there are three different, overlapping lists depending on what you're asking about. Mixing these up is a common source of confusion, so they're kept separate here rather than collapsed into a single number.

* **Gene database coverage** (species with germline V/J/C gene data in `conga/tcrdist/db/combo_xcr_2026-08-06.tsv`), 22 entries:
  `human, human_gd, human_ig, mouse, mouse_gd, mouse_ig, rhesus, rhesus_gd, rhesus_ig, cat, cat_gd, cat_ig, dog, dog_gd, dog_ig, ferret, ferret_gd, ferret_ig, rabbit, rabbit_gd, rabbit_ig, sheep`.
  The raw reference file also contains rows for `sheep_gd`, `sheep_ig`, and `rainbowtrout_ig`, but these are chain-incomplete and have no usable analysis pathway anywhere in the codebase -- they are deliberately not exposed as supported organisms, so don't be surprised if you find them while grepping the TSV.
* **CLI-usable** (the `--organism` choices accepted by both `scripts/run_conga.py` and `scripts/setup_10x_for_conga.py`), 21 entries:
  `mouse, human, mouse_gd, human_gd, human_ig, rhesus, rhesus_gd, rhesus_ig, cat, cat_gd, cat_ig, dog, dog_gd, dog_ig, ferret, ferret_gd, ferret_ig, rabbit, rabbit_gd, rabbit_ig, sheep`.
* **Vectorized-TCRdist-eligible** (`conga.tcrdist.vectorized.SUPPORTED_ORGANISMS`, the fast fixed-length-encoding path described below), 18 entries:
  `human, mouse, rhesus, rhesus_gd, rhesus_ig, cat, cat_gd, cat_ig, dog, dog_gd, dog_ig, ferret, ferret_gd, ferret_ig, rabbit, rabbit_gd, rabbit_ig, sheep`.
  Notably absent: `human_gd, human_ig, mouse_gd, mouse_ig`. These four organisms are still fully usable end-to-end through the CLI and the python package -- they just take the KernelPCA or exact-TCRdist path instead of the vectorized one. This is a real, user-relevant gap (not an oversight), so if you're working with human/mouse gamma-delta or Ig data, expect KernelPCA/exact-neighbor performance characteristics rather than vectorized ones.

# Table of Contents

* [TCR Representations](#tcr-representations)
* [Batch Integration](#batch-integration)
* [FAISS Acceleration](#faiss-acceleration)
* [Running](#running)
* [Installation](#installation)
* [Migrating Seurat data to CoNGA](#migrating-seurat-data-to-conga)
* [Merging multiple datasets for CoNGA analysis](#merging-multiple-datasets-into-a-single-object-for-conga-analysis)
* [Updates](#updates)
* [SVG to PNG](#svg-to-png)
* [Testing CoNGA without going through the pain of installing it](#testing-conga-without-going-through-the-pain-of-installing-it)
    - [Docker](#docker)
    - [Google colab](#google-colab)
* [Examples](#examples)
* [The CoNGA data model: where stuff is stored](#conga-data-model-where-stuff-is-stored)
* [Frequently Asked Questions](#frequently-asked-questions)

# TCR Representations

CoNGA supports three distinct TCR neighbor paths, automatically selected based on organism type and dataset size.

## 1. Vectorized TCRdist (default for eligible organisms)

**When used:** Any organism in `conga.tcrdist.vectorized.SUPPORTED_ORGANISMS` (18 organisms -- see the organism support section above for the full list; notably this excludes `human_gd`, `human_ig`, `mouse_gd`, and `mouse_ig`).

**How it works:** Each paired TCR is encoded as a fixed-length vector by:
- Embedding the TCRdist amino acid substitution matrix into Euclidean space using multidimensional scaling
- Concatenating per-position amino acid vectors for germline CDR1/CDR2/CDR2.5 loops
- Adding a trimmed-and-gapped CDR3 representation
- Scaling CDR3 positions to preserve TCRdist weighting

**Benefits:**
- **Sub-quadratic memory**: No N×N distance matrices
- **Fast neighbor search**: Compatible with FAISS acceleration
- **High accuracy**: Spearman correlation ≥0.95 with exact TCRdist
- **Deterministic**: Reproducible encodings with fixed random seed

**Performance:** Encoding 20,000 clonotypes takes ~0.1s and produces a 91MB array, vs. ~6.4GB transient memory for the KernelPCA approach.

## 2. KernelPCA Representation (fallback)

**When used:**
- Organisms not in the vectorized-eligible set, including `human_gd`, `mouse_gd`, `human_ig`, and `mouse_ig`
- Small datasets when explicitly requested
- Datasets with fewer observations than the KernelPCA reduction limit where vectorized encoding isn't applicable

**How it works:** Traditional approach computing the full TCRdist distance matrix, then applying KernelPCA dimensionality reduction.

**Limitations:** Quadratic memory scaling makes this impractical for large datasets (tens of thousands of clonotypes or more).

## 3. Exact TCRdist Neighbors (large datasets)

**When used:**
- Datasets large enough that KernelPCA would exceed memory limits
- When exact distances are required (via the `--no_kpca` flag)
- Organism/dataset-size combinations where neither of the other two paths applies

**How it works:** Computes TCR neighbors on-demand using exact TCRdist without storing distance matrices. Uses the C++ implementation when available, falls back to Python otherwise.

**Benefits:** Handles datasets of any size with constant memory overhead.

## Controlling TCR Representation Selection

You can override automatic selection with command-line flags:

```bash
# Use the default (automatic) representation selection
python scripts/run_conga.py --organism human --gex_data data.h5 --clones_file clones.tsv

# Force KernelPCA (small datasets only)
python scripts/run_conga.py --use_kpca_tcrdist --organism human --gex_data data.h5 --clones_file clones.tsv

# Force exact neighbors (any size)
python scripts/run_conga.py --no_kpca --organism human --gex_data data.h5 --clones_file clones.tsv

# Adjust KernelPCA size limit (default: 20,000)
python scripts/run_conga.py --kpca_reduction_limit 50000 --organism human --gex_data data.h5 --clones_file clones.tsv
```

# Batch Integration

CoNGA has two separate, unrelated mechanisms that both get called "batch" handling. They solve different problems and are easy to confuse, so they're documented separately here.

## GEX batch correction: `conga.preprocess.batch_integration()`

This is the mechanism that actually corrects gene expression for batch effects. It runs batch-aware highly-variable-gene selection and then corrects the GEX PCA representation using either [Harmony](https://github.com/immunogenomics/harmony) (`method='harmony'`, requires `harmonypy`) or [scVI](https://scvi-tools.org/) (`method='scvi'`, requires `scvi-tools`). The corrected representation is written back into `adata.obsm['X_pca_gex']`, so downstream clustering and neighbor-finding code needs no changes to consume it.

Install the dependencies with:

```bash
pip install "conga[batch-integration]"
```

Use it from `run_conga.py` with the paired flags `--batch_key` and `--batch_integration_method`:

```bash
python scripts/run_conga.py \
--organism human \
--gex_data data.h5 \
--gex_data_type 10x_h5 \
--clones_file clones.tsv \
--batch_key donor \
--batch_integration_method harmony \
--outfile_prefix tmp_batch_corrected
```

`--batch_key` names the single `adata.obs` column driving both the HVG selection and the integration method; `--batch_integration_method` must be `harmony` or `scvi`. The two flags must be supplied together. `batch_integration()` is mutually exclusive with `--force_variable_genes` -- supplying both raises a `ValueError`, since they represent two different ways of picking the HVG set for the same pipeline step.

From the python package directly:

```python
import conga
adata = conga.preprocess.batch_integration(
    adata, batch_key='donor', method='harmony',
)
```

## Batch annotation for visualization: `--batch_keys`

This is the older, separate mechanism, and it does **not** perform any GEX correction. It lets you attach existing categorical metadata (donor, timepoint, outcome, etc.) to clonotypes purely so it can be displayed in plots and clustermaps -- colored UMAPs (`conga.plotting.make_batch_colored_umaps`) and batch-aware clustermaps (`conga.plotting.make_clone_batch_clustermaps`).

Each batch category must be represented as an integer-valued column in `adata.obs`, with the column names listed in `adata.uns['batch_keys']` (plural -- note this is a different field from the `batch_key` singular used by `batch_integration()` above). Pass the column names on the command line:

```bash
python scripts/run_conga.py --organism human --gex_data data.h5ad --gex_data_type h5ad \
    --clones_file clones.tsv --batch_keys donor timepoint --outfile_prefix tmp_annotated
```

See the FAQ entry below for a worked example of adding batch columns to an `AnnData` object by hand.

# FAISS Acceleration

FAISS (Facebook AI Similarity Search) provides GPU and CPU-optimized vector similarity search for performance improvements on large datasets.

## Installation Options

FAISS is an optional dependency with tiered backend selection:

```bash
# CPU-only performance boost (recommended)
pip install "conga[performance]"

# GPU acceleration (requires CUDA-capable hardware)
pip install "conga[performance-gpu]"

# All features including FAISS CPU
pip install "conga[all]"

# All features including FAISS GPU
pip install "conga[all-gpu]"
```

## Automatic Backend Selection

CoNGA automatically selects the best available backend:

1. **faiss-gpu** (if installed and CUDA available): Maximum performance
2. **faiss-cpu** (if installed): Major speedup over sklearn
3. **sklearn** (always available): Reliable fallback

## Performance Benefits

| Dataset Size | Backend | GEX Search | TCR Search | Memory Usage |
|--------------|---------|------------|------------|---------------|
| 10,000 cells | sklearn | 45s | 12s | 2.1GB |
| 10,000 cells | faiss-cpu | 8s | 3s | 1.8GB |
| 10,000 cells | faiss-gpu | 2s | 1s | 1.5GB |
| 100,000 cells | sklearn | >30min | >10min | >20GB |
| 100,000 cells | faiss-cpu | 4min | 2min | 8GB |
| 100,000 cells | faiss-gpu | 45s | 30s | 6GB |

These numbers are point-in-time measurements from a specific test environment, not guarantees -- see the caveat in the Performance Benchmarks section below.

## Controlling FAISS Usage

```bash
# Force GPU backend (fails if unavailable)
python scripts/run_conga.py --use_faiss_gpu --organism human --gex_data data.h5 --clones_file clones.tsv

# Force CPU backend
python scripts/run_conga.py --use_faiss_cpu --organism human --gex_data data.h5 --clones_file clones.tsv

# Disable FAISS entirely
python scripts/run_conga.py --disable_faiss --organism human --gex_data data.h5 --clones_file clones.tsv
```

FAISS backend selection is logged and recorded in analysis outputs for reproducibility.

# Running

Running `conga` on a single-cell dataset is a two- (or more) step process, as outlined below.
Python scripts are provided in the `scripts/` directory but analysis steps can also be accessed interactively
in jupyter notebooks (for example, [a simple pipeline](simple_conga_pipeline.ipynb) in the top directory of this repo)
or in your own python scripts through the interface in the `conga` python package.
There's also a [google colab notebook](colab_conga_pipeline.ipynb) which you can
[![open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phbradley/conga/blob/master/colab_conga_pipeline.ipynb) and run. If you want to
experiment before installing CoNGA locally you can save a copy of that notebook
to your google drive, edit and run the pipeline, either on the provided examples or on
data that you upload to the colab instance.
The examples in the `examples/` folder described below and in the jupyter notebooks feature publicly available data from 10x Genomics,
which can be downloaded in a single
[zip file](https://www.dropbox.com/s/r7rpsftbtxl89y5/conga_example_datasets_v1.zip?dl=0) or at the
[10x genomics datasets webpage](https://support.10xgenomics.com/single-cell-vdj/datasets/).

1. **SETUP**: The TCR data is converted to a form that can be read by `conga` and then
a matrix of `TCRdist` distances is computed. KernelPCA is applied to this distance
matrix to generate a PC matrix that can be used in clustering and dimensionality reduction. This
is accomplished with the python script `scripts/setup_10x_for_conga.py` for 10x datasets. For example:

```
python conga/scripts/setup_10x_for_conga.py --filtered_contig_annotations_csvfile vdj_v1_hs_pbmc3_t_filtered_contig_annotations.csv --organism human
```

2. **ANALYZE**: The `scripts/run_conga.py` script has an implementation of the main pipeline and can be run
as follows:

```
python conga/scripts/run_conga.py --graph_vs_graph --gex_data data/vdj_v1_hs_pbmc3_5gex_filtered_gene_bc_matrices_h5.h5 --gex_data_type 10x_h5 --clones_file vdj_v1_hs_pbmc3_t_filtered_contig_annotations_tcrdist_clones.tsv --organism human --outfile_prefix tcr_hs_pbmc3
```

3. **RE-ANALYZE**: Step 2 will generate a processed `.h5ad` file that contains all the gene expression
and TCR sequence information along with the results of clustering and dimensionality reduction. It can then
be much faster to perform subsequent re-analysis or downstream analysis by "restarting" from those files.
Here we are using the `--all` command line flag which requests all the major analysis modes:

```
python conga/scripts/run_conga.py --restart tcr_hs_pbmc3_final.h5ad --all --outfile_prefix tcr_hs_pbmc3_restart
```

See the examples section below for more details.

# Installation

We *highly* recommend installing CoNGA in a virtual environment, for example using the
`mamba` or `conda`. Linux folks can check out the
[Dockerfile](Dockerfile) for a minimal set of installation commands. At the
top of the [google colab jupyter notebook](colab_conga_pipeline.ipynb)
([link to the notebook on colab](https://colab.research.google.com/github/phbradley/conga/blob/master/colab_conga_pipeline.ipynb))
are the necessary installation commands from within a notebook environment.

## Quick Installation (Recommended)

CoNGA uses modern Python packaging with optional dependencies for performance features. CoNGA requires **Python 3.12+**.

```bash
# Create environment (Python 3.12+ required)
mamba create -n conga_env python=3.12
mamba activate conga_env

# Basic installation
pip install conga

# Performance-optimized (with FAISS CPU)
pip install "conga[performance]"

# Full installation (all optional features)
pip install "conga[all]"

# GPU-accelerated (requires CUDA)
pip install "conga[all-gpu]"
```

## Development Installation

For the latest features or to contribute:

```bash
# Clone repository
git clone https://github.com/phbradley/conga.git
cd conga

# Create development environment
mamba env create -f environment.yml
mamba activate conga-dev

# Install in development mode
pip install -e .

# Compile C++ components (recommended)
cd conga/tcrdist_cpp && make && cd ../..
```

## Optional Dependencies

CoNGA provides several optional feature sets, defined in `pyproject.toml`:

### Performance Optimization
- `conga[performance]`: Adds `faiss-cpu` and `fastcluster` for faster neighbor search
- `conga[performance-gpu]`: Adds `faiss-gpu` and `fastcluster` for maximum performance (requires CUDA)

### Batch Integration
- `conga[batch-integration]`: Adds `harmonypy` and `scvi-tools`, needed for `conga.preprocess.batch_integration()` (Harmony or scVI-based GEX correction -- see the Batch Integration section above)

### Development
- `conga[dev]`: Testing, linting, and development tools
- `conga[all]`: performance + batch-integration + dev (CPU performance)
- `conga[all-gpu]`: performance-gpu + batch-integration + dev (GPU performance)

### FAISS Installation Notes

FAISS provides performance improvements but requires specific installation:

**CPU-only (recommended for most users):**
```bash
# Via pip (included in conga[performance])
pip install faiss-cpu>=1.7.4

# Via mamba (alternative)
mamba install -c conda-forge faiss-cpu
```

**GPU acceleration (Linux/Windows with CUDA):**
```bash
# Via pip (included in conga[performance-gpu])
pip install faiss-gpu>=1.7.4

# Via mamba (alternative)
mamba install -c conda-forge faiss-gpu
```

**macOS users:** Only faiss-cpu is supported. GPU acceleration is not available on macOS.

## Even more details

The calculations in the
`conga` manuscript were conducted with the following package versions:

```
scanpy==1.4.3 anndata==0.6.18 umap-learn==0.3.9 numpy==1.16.2 scipy==1.2.1 pandas==0.24.1 scikit-learn==0.20.2 statsmodels==0.9.0 python-igraph==0.7.1 louvain==0.6.1
```

This historical environment predates the Python 3.12+ requirement and will not work with the current `conga` package; it's included only for reference if you need to reproduce the exact numbers in the original manuscript. For current installations, use the Quick Installation or Development Installation instructions above.

5. Ensure you have a tool for SVG to PNG conversion available.

See the section below on SVG to PNG conversion for more details.

# Migrating Seurat data to CoNGA
We recommend using the write10XCounts function from the DropletUtils package for
converting Seurat objects into 10x format for importing into CoNGA/scanpy.
```
require(Seurat)
require(DropletUtils)
hs1 <- readRDS('~/vdj_v1_hs_V1_sc_5gex.rds')
```
If the object contains only gene expression:
```
write10xCounts(x = hs1@assays$RNA@counts, path = './hs1_mtx/')
# import the hs1_mtx directory into CoNGA using the '10x_mtx' option
```
If the object contains both gene expression and antibody labeling:
```
# Concatenate the GEX and antibody labeling count matrices
# Here, ADT is the antibody labeling assay slot.

count_matrix <- rbind(hs1@assays$RNA@counts, hs1@assays$ADT@counts)

# create vector of feature type labels
features <- c(
  rep("Gene Expression", nrow(hs1@assays$RNA@counts)), 
  rep("Antibody Capture", nrow(hs1@assays$ADT@counts))
  )
              
# write out              
write10xCounts( count_matrix, 
                path = './hs1_mtx/',
                gene.id = rownames(count_matrix),
                gene.symbol = rownames(count_matrix),
                barcodes = colnames(count_matrix),
                gene.type = features,
                version = "3")
# import the hs1_mtx directory into CoNGA using the '10x_mtx' option
```

# Merging multiple datasets into a single object for CoNGA analysis

This can be done in two easy steps using the `setup_10x_clones.py` and `merge_samples.py` scripts in `conga`. 

1. **SETUP**: The TCR data for each sample being merge must be converted to a form that can be read by `conga`. 
This can be done using the python script `scripts/setup_10x_for_conga.py` for 10x datasets. 
By default the matrix of `TCRdist` distances calculated and reduced in dimensionality by KernelPCA, however, 
since these will need to be recalculated after merging we can skip this step with the `--no_kpca` flag.

```
python ~/conga/scripts/setup_10x_for_conga.py \
--filtered_contig_annotations_csvfile vdj_v1_hs_pbmc3_t_filtered_contig_annotations.csv \
--output_clones_file vdj_v1_hs_pbmc3_clones.tsv \
--organism human \
--no_kpca 

python ~/conga/scripts/setup_10x_for_conga.py \
--filtered_contig_annotations_csvfile sc5p_v2_hs_PBMC_10k_t_filtered_contig_annotations.csv \
--output_clones_file sc5p_v2_hs_PBMC_10k_clones.tsv \
--organism human \
--no_kpca

```

2. **MERGE SAMPLES**: The `scripts/merge_samples.py` script uses a tab-delimted file 
with three columns: "clones_file", "gex_data", "gex_data_type" specifying the paths 
to the clones file from step 1, it’s companion gex data, and the gex data type (e.g 10x_h5)
for each sample:

| clones_file | gex_data | gex_data_type |
| --- | --- | --- |
| vdj_v1_hs_pbmc3_clones.tsv | vdj_v1_hs_pbmc_5gex_filtered_gene_bc_matrices_h5.h5 | 10x_h5 |
| sc5p_v2_hs_PBMC_10k_clones.tsv | sc5p_v2_hs_PBMC_10k_filtered_feature_bc_matrix.h5 | 10x_h5 |

```
python ~/conga/scripts/merge_samples.py \
--samples pbmc_samples.txt \
--output_clones_file merged_pbmc_clones.tsv \
--output_gex_data merged_pbmc_gex.h5ad \
--organism human 
```
The `TCRdist` distances are calculated and KernelPCA is applied to the matrix here.

3. **ANALYZE**: The merged `AnnData` object containing the gene expression and the merged clones file can 
now be analyzed using the `scripts/run_conga.py` script:

```
python ~/conga/scripts/run_conga.py \
--gex_data merged_pbmc_gex.h5ad \
--gex_data_type h5ad \
--clones_file merged_pbmc_clones.tsv \
--organism human \
--graph_vs_graph \
--outfile_prefix ../merged_pbmc_outs/merged_pbmc
```
# Merging multiple TCR files for alignment with a single aggregate GEX matrix

Often times, multiple GEX count matrices are aggregated together using `cellranger aggr`
while the outputs of the TCR libraries remain as individual folders. The barcode suffix of 
each GEX library is changed during this process while those of the samples' TCR
filtered_contig_annotations.csv file remains the default "1". We will need to adjust the barcode 
suffixes of the TCR outputs in order to merge the information with the aggregate GEX matrix. 
For this we can use `conga.tcrdist.make_10x_clone_file.make_10x_clone_file_batch`. 
Here, the barcode suffix of the TCR library can be updated to match with samples' corresponding 
GEX library suffix, and all the resulting TCR clones tables will merged into a single output run
with the GEX file through `run_conga.py`, or interactively. 

The function uses a csv-format metadata table to guide the merging. The file should have tow columns,
'file' with the path to the filtered_contig_annotations.csv file, and 'batch_id' which should contain the value
of the samples' barcode suffix in the GEX matrix we are trying to merge with.

| file | batch_id |
| --- | --- |
| sample_1_filtered_contig_annotations.csv | 1 |
| sample_2_filtered_contig_annotations.csv | 2 |
| sample_3_filtered_contig_annotations.csv | 3 |

```
organism = 'human'
conga.tcrdist.make_10x_clone_file.make_10x_clone_file_batch('path/to/metadata.csv', organism, 'path/to/clones_file_output.tsv )
```

# Updates

See [`docs/UPDATES.md`](docs/UPDATES.md) for the release and change history.

# svg to png
The `conga` image-making pipeline requires an svg to png conversion. There seem to be a variety of
options for doing this, with the best choice being somewhat platform dependent. We've had good luck with
ImageMagick `convert` (on Linux, MacOS, and Windows) and Inkscape (on mac).

On Mac, we recommend installing Inkscape (https://inkscape.org/ or via conda
`conda install -c conda-forge inkscape`)

or

ImageMagick (using Homebrew with `brew install imagemagick` or
via conda with `conda install -c conda-forge imagemagick`).

On Windows, we recommend the self-installing executable available from ImageMagick:
(https://imagemagick.org/script/download.php)

Another possibility is `pip install cairosvg` from within the relevant
environment.

The conversion is handled in the file `conga/convert_svg_to_png.py`, so you can modify that file if things are
not working and you have a tool installed; `conga` may not be looking in the right place. For example, the Inkscape install location on Mac seems to
switch around; it may be necessary to fiddle with the variable
`PATH_TO_INKSCAPE` in `conga/convert_svg_to_png.py`. Also if the fonts
in the TCR/BCR logos look bad you could try switching the MONOSPACE_FONT_FAMILY
variable in that python file (see comments at the top of the file).

# Testing CoNGA without going through the pain of installing it
If you want to test CoNGA without taking the time to install it, here are some options.
## Docker
There is a [Dockerfile](Dockerfile) in the repository root that builds a
self-contained image with CoNGA (including FAISS-CPU acceleration), the
compiled `tcrdist_cpp` executables, and ImageMagick for SVG-to-PNG
conversion. Build it from the repository root:
```
docker build -t conga .
```
Erick Matsen has a nice [mini intro to docker](http://erick.matsen.org/2018/04/19/docker.html)
that describes, among other things, how to run an image and make folders visible
inside the image (so you can run the conga scripts on your data). For example,
if you have your data in the folder `/path/to/datasets/` you could type these
commands at the command prompt (aka terminal window on mac)
```
docker run -v /path/to/datasets:/data -it conga /bin/bash
```
and then within the new docker shell that opens:
```
root@d0fa5d83e40d:/opt/conga# python scripts/setup_10x_for_conga.py --filtered_contig_annotations_csvfile /data/filtered_contig_annotations.csv --organism human
root@d0fa5d83e40d:/opt/conga# mkdir /data/output/
root@d0fa5d83e40d:/opt/conga# python scripts/run_conga.py --all --organism human --clones_file /data/filtered_contig_annotations_tcrdist_clones.tsv --gex_data /data/filtered_gene_bc_matrices_h5.h5 --gex_data_type 10x_h5 --outfile_prefix /data/output/conga_test1
root@d0fa5d83e40d:/opt/conga# exit
```
You can also skip the interactive shell and run `conga info` directly to
sanity-check the installation (confirms FAISS, the compiled C++ TCRdist
binaries, and core dependency versions):
```
docker run --rm conga conga info
```
(changing the filenames and `--outfile_prefix` as needed). This would put the output
into a folder `output` in the `/path/to/datasets/` folder (so you can see it
outside the docker image),

## Google colab
Another quick-start option is to use the free computing environment available
through google colab.
[This link](https://colab.research.google.com/github/phbradley/conga/blob/master/colab_conga_pipeline.ipynb)
will open an example notebook. If you click on `Connect` near the top right
it will connect to a
cloud-hosted machine somewhere and you will be able to run the commands in the
notebook (the code in some cells may start out hidden but you can click on the cells to
make it visible). You can even upload your own datasets with the file explorer
button on the left-hand side. If you want to modify the document save a copy
with the `Copy to drive` button.

# Examples
Shell scripts for running `conga` on three publicly available 10X
genomics datasets can be found in the `examples/` directory:
[`examples/setup_all.bash`](examples/setup_all.bash) which preprocesses the
clonotype data, and [`examples/run_all.bash`](examples/run_all.bash) which calls
`scripts/run_conga.py` on the three datasets.

See [`docs/EXAMPLES.md`](docs/EXAMPLES.md) for a walkthrough of the key
outputs on the human PBMC, mouse PBMC, and human melanoma B cell datasets,
along with performance benchmarks for vectorized TCRdist and FAISS
acceleration.

# Performance Benchmarks

CoNGA delivers performance improvements on large datasets through vectorized TCRdist and FAISS acceleration. The numbers below are point-in-time measurements from one specific hardware/software configuration (noted at the bottom of each table) -- they are not guarantees, and actual results on your hardware, dataset, and library versions may differ meaningfully. Treat them as a general indication of scaling behavior, not a benchmark you should expect to reproduce exactly.

### Memory Usage Comparison

| Dataset Size | Traditional KernelPCA | Vectorized TCRdist | Memory Reduction |
|--------------|----------------------|-------------------|------------------|
| 5,000 clones | 400 MB | 225 MB | 44% |
| 10,000 clones | 1.6 GB | 450 MB | 72% |
| 20,000 clones | 6.4 GB | 900 MB | 86% |
| 50,000 clones | Not feasible | 2.2 GB | N/A |

### Processing Time Benchmarks

| Operation | Dataset Size | sklearn | FAISS-CPU | FAISS-GPU | Speedup |
|-----------|--------------|---------|-----------|-----------|---------|
| TCR encoding | 20k clones | KernelPCA: 45s | Vectorized: 0.1s | Vectorized: 0.1s | 450x |
| GEX neighbors | 10k cells | 45s | 8s | 2s | 5-22x |
| GEX neighbors | 100k cells | >30min | 4min | 45s | >40x |
| TCR neighbors | 10k clones | 12s | 3s | 1s | 4-12x |
| TCR neighbors | 50k clones | >20min | 2min | 30s | >40x |

### Accuracy Validation

Vectorized TCRdist maintains high accuracy compared to exact TCRdist on the datasets we've tested:

| Metric | Requirement | Achieved |
|---------|-------------|----------|
| Spearman correlation | ≥0.95 | 0.999 |
| Neighbor recall@10 | ≥0.80 | 0.953 |
| Neighbor recall@100 | ≥0.80 | 0.971 |


# CoNGA data model: where stuff is stored

After setup, the conga package stores data in various locations
in the `scanpy` `AnnData` object. Below we assume that `adata` is
the name of the AnnData object where the GEX and
TCR data is stored (this is the naming
convention followed in CoNGA).

## The core stuff
CoNGA functionality like graph-vs-graph and graph-vs-feature analyses will
generally expect these to be set once the setup phase has completed. The
CoNGA routines that fill these arrays are:
* `conga.preprocess.read_data`: loads the GEX data into an `AnnData` object
(here called `adata`); puts the TCR information into the `adata.obs` arrays;
reads the TCRdist kernel principal components and stores them in `adata.obsm` under
the key `X_pca_tcr`. Eliminates cells without paired TCR information.
* `conga.preprocess.filter_and_scale`: sets up the `adata.raw` object, does
some typical single-cell filtering and preprocessing.
* `conga.preprocess.reduce_to_single_cell_per_clone`: reduces to a single
cell per clonotype; fills the `adata.obs['clone_sizes']` array and
potentially `adata.obsm[<batch_key>]` for one or more `<batch_key>`s if
there is batch structure defined in the input data.
* `conga.preprocess.cluster_and_tsne_and_umap`: Fills `adata.obsm['X_pca_gex']`,
and the `adata.obs` arrays `X_gex_2d`, `X_tcr_2d`, `clusters_gex`,
and `clusters_tcr`.
* `conga.preprocess.batch_integration`: Alternative to the normalization/HVG
steps above when GEX batch correction is requested. Overwrites
`adata.obsm['X_pca_gex']` with a Harmony- or scVI-corrected representation,
so everything downstream (clustering, UMAP, neighbor-finding) consumes the
corrected PCs transparently. See the Batch Integration section above.

### `adata.obs`
The following 1-D arrays are stored in the `obs` array and can be accessed
with expressions like `adata.obs['va']`

* `va`: V gene names, alpha chain
* `ja`: J gene names, alpha chain
* `cdr3a`: CDR3 amino acid sequences, alpha chain
* `cdr3a_nucseq`: CDR3 nucleotide sequences, alpha chain
* `vb`: V gene names, beta chain
* `jb`: J gene names, beta chain
* `cdr3b`: CDR3 amino acid sequences, beta chain
* `cdr3b_nucseq`: CDR3 nucleotide sequences, beta chain
* `clusters_gex`: GEX cluster assignments, integers, range `[0, num_clusters)`
* `clusters_tcr`: TCR cluster assignments, integers, range `[0, num_clusters)`
* `clone_sizes`: The number of cells in each clonotype.


### `adata.obsm`
The following multidimensional arrays are stored in the `obsm` array after
setup.

* `X_pca_gex`: The GEX principal components. Used for neighbor-finding,
UMAP projections, etc. When `conga.preprocess.batch_integration()` is used,
this array holds the Harmony- or scVI-corrected representation instead of
the raw PCA, with no change needed in how downstream code reads it.
* `X_pca_tcr`: The TCRdist kernel principal components. Present when using
the KernelPCA representation path (see TCR Representations above); may be
missing if we are using the vectorized or exact-TCRdist-neighbors paths.
* `X_vec_tcr`: The vectorized TCR representation. Fixed-length vector
encodings of paired TCRs used for sub-quadratic neighbor search. Present
when using the vectorized TCRdist path (see TCR Representations above for
which organisms are eligible).
* `X_gex_2d`: The 2D landscape projection based on GEX (UMAP by default).
* `X_tcr_2d`: The 2D landscape projection based on TCR (UMAP by default).

### `adata.uns`
These miscellaneous data are stashed in the `adata.uns` dictionary:

* `organism`: A string indicating what type of TCR/BCR data is being analyzed.
See the Organism support section near the top of this README for the full,
current breakdown of supported organisms (gene-database coverage, CLI-usable,
and vectorized-TCRdist-eligible are three different lists).
* `active_tcr_representation`: The TCR representation used in the analysis.
One of: `X_vec_tcr` (vectorized), `X_pca_tcr` (KernelPCA), or `exact_tcrdist` (no obsm array).
* `vec_tcr_config`: Configuration parameters for vectorized TCR encoding.
Contains `aa_mds_dim`, `num_pos_cdr3`, `cdr3_weight`, etc.
* `faiss_backend_info`: FAISS backend selection and performance metrics.
Records which backend was used and performance characteristics.

### `adata.raw`
This is where the raw data on gene expression is expected to live.

* `adata.raw.X` Sparse matrix with the gene expression values for each gene.
These will have been normalized to sum to 10,000 and then `np.log1p`'ed (had the natural logarithm taken after adding 1).
* `adata.raw.var_names` The gene names; should match the number of columns in
`adata.raw.X`.

### GEX and TCR neighbors
Currently the neighborhood information is stored independently of the
`adata` object, in a dictionary called `all_nbrs`. The keys of this
dictionary are the neighborhood fractions aka `nbr_fracs`, floats that
represent the size of the neighborhood as a fraction of the total number
of clonotypes. The default `nbr_fracs` are `[0.01, 0.1]`. For each `nbr_frac`,
`all_nbrs[nbr_frac] = [gex_nbrs, tcr_nbrs]` where `gex_nbrs` and `tcr_nbrs`
are `numpy` arrays of shape `(num_clonotypes,num_nbrs)`, and
`num_nbrs = int(nbr_frac*num_clonotypes)`. Note that a clonotype is not
included in its own set of neighbors. Also note that the CoNGA neighbor
information is distinct from neighbor information
that `scanpy` uses for UMAP projection and clustering.
CoNGA neighborhoods are larger than the neighborhoods typically used
in clustering and dimensionality reduction.

## Extras
This might be results of calculations that are stored for easier access,
or optional data that is present in certain circumstances
(for example when there are batches present).
It should be OK if any of these are missing.

### `adata.obs`
The following 1-D arrays are stored in the `obs` array and can be accessed
with expressions like `adata.obs['va']`

* `is_invariant`: Boolean array recording the presence of
canonical invariant (MAIT or iNKT) TCR chains.
* `nndists_tcr`: Nearest-neighbor distances based on TCR sequence.
Gives an approximate measure of (inverse) TCR density.
* `nndists_gex`: Nearest-neighbor distances based on GEX.
Gives an approximate measure of (inverse) GEX density.
* `conga_scores`: CoNGA scores for each clonotype.
Filled after the graph-vs-graph analysis has been run.
* `<batch_key>`: When there are multiple batches present in a dataset
these can be tracked and visualized in many of the analysis
and plotting routines (the annotation-only mechanism described in the
Batch Integration section above, not GEX correction).
Here `<batch_key>` is the name of the batch/category (for example `'outcome'` or `'subject'` or `'timepoint'`).
The entry in `adata.obs` for each batch key should contain integers in the range `[0,num_batch_classes)`.
This information is stored in the `adata.obs` array *prior* to condensing to a single
cell per clonotype, and in the `adata.obsm` array *after* condensing to a single cell per clonotype
(since expanded clonotypes can span multiple batch assignments).
See the Batch Integration section above and the FAQ entry on batches in CoNGA.

### `adata.obsm`
The following multidimensional arrays are stored in the `obsm` array and can be accessed
with expressions like `adata.obsm[<tag>]`

* `<batch_key>`: When there are multiple batches present in a dataset
these can be tracked and visualized in many of the analysis
and plotting routines.
Here `<batch_key>` is the name of the batch/category (for example `'outcome'` or `'subject'` or `'timepoint'`).
For each `<batch_key>`, the array stored in `adata.obsm` should have shape
`(num_clonotypes,num_categories)` where `num_categories` is the number of possible
batch assignments. For example if there are three timepoints then `num_categories` for the `'timepoint'` batch key would be 3.
The `(i,j)` entry in the array will give the number of cells in clonotype `i` that were
assigned to the batch assignment `j`.
This array is filled automatically when we reduce to a single cell per clonotype,
based on the information in the array `adata.obs[<batch_key>]` (see above).
See the Batch Integration section above and the FAQ entry on batches in CoNGA.

### `adata.uns`
* `batch_keys`: A list of strings that gives the names of the different
batch types/categories, if present (for example something like `['subject', 'outcome', 'timepoint']`.
For each name in `adata.uns['batch_keys']` there should be
an entry in the `adata.obs` array with that name (see above for description of that data).
When we condense to a single cell per clonotype,
we add an entry in the `adata.obsm` array with the same name, which contains the counts
for each batch assignment summed over
all the cells in each clonotype (so it's a 2D array and hence has to be stored in `obsm` not `obs`.
Note this is the plural annotation-only mechanism, distinct from the singular
`batch_key` argument to `conga.preprocess.batch_integration()` -- see the
Batch Integration section above.

### `adata.var`
* `feature_types`: This array is used to detect and exclude
antibody (site-seq) or other non-gene-expression
features. If it's missing, then during setup CoNGA will assume that
all the counts in the `adata.X` array are gene-expression features.
The expected value for gene expression features in this
array is the string `'Gene Expression'`. CoNGA will look for and use any column
in the `adata.var` array whose name starts with
`feature_types` (since sometimes they get renamed during concatenation).

# Frequently asked questions

1. My CoNGA docker process mysteriously stopped with the cryptic error message
'killed'
   * You may need to increase the resources allocated to docker processes, in
	 particular the memory. You could do this on the Resources tab of settings in
	 the Docker desktop app. Or try a quick google search.
1. I get an error when I type `import conga`
   * Note that this won't work automatically unless you used the `pip install -e .`
	 installation method.
	 * If you didn't, or that's not working for some reason, you can just manually
	 add the `conga` github repository directory to your path before importing conga.
	 For example:
   ```python
   import sys
   path_to_conga = '/path/to/gitrepos/conga/' # contains README.md, scripts, conga
   sys.path.append(path_to_conga)
   import conga
   ```
1. How do I know which TCR representation CoNGA is using?
   * Check the log output during analysis - CoNGA will report which path was selected
   * Look at `adata.uns['active_tcr_representation']` in your results
   * For organisms in `conga.tcrdist.vectorized.SUPPORTED_ORGANISMS`, CoNGA defaults to the vectorized representation
   * For other organisms (including `human_gd`, `human_ig`, `mouse_gd`, `mouse_ig`) or small datasets, it uses KernelPCA or exact neighbors
   * You can override with `--use_kpca_tcrdist` or `--no_kpca` flags

1. How do I enable FAISS acceleration?
   * Install with `pip install "conga[performance]"` for CPU or `pip install "conga[performance-gpu]"` for GPU
   * CoNGA will automatically detect and use the best available backend
   * Check the log output to see which backend was selected
   * Force a specific backend with `--use_faiss_gpu`, `--use_faiss_cpu`, or `--disable_faiss`

1. My dataset is taking too much memory, what can I do?
   * For organisms eligible for vectorized TCRdist, CoNGA automatically uses vectorized encoding which reduces memory substantially
   * For large datasets, CoNGA automatically uses exact neighbor calculation to avoid memory issues
   * Install FAISS for additional memory efficiency: `pip install "conga[performance]"`
   * Use `--kpca_reduction_limit` to control when KernelPCA is skipped (default: 20,000)

1. How can I visualize the different batches in my data? Or other discrete/categorical
features assigned to individual cells?
   * This is the annotation-only mechanism described in the Batch Integration section
	 above (`--batch_keys`, plural) -- it does not correct GEX for batch effects, it
	 only lets you color plots and clustermaps by existing categorical metadata.
	 If you need actual GEX batch correction, use `conga.preprocess.batch_integration()`
	 (`--batch_key`/`--batch_integration_method`, singular) instead; see the Batch
	 Integration section above for both.

	 For the annotation-only mechanism: each "batch" flavor (donor, tissue, disease,
	 etc.) must be represented as an integer-valued column in
	 `adata.obs`, and the names of the different batch columns should be given
	 as a list in `adata.uns['batch_keys']`.

	 One easy way to add these columns is through the `scripts/merge_samples.py`
	 script, which accepts a `--batch_keys` argument. That argument should be a list
	 of column names where those columns are present in the samples tsvfile provided
	 with the `--samples` argument. So if you are merging data and the batch structure
	 you want to visualize breaks down by the input files, that should work.

	 For more complicated data, the best thing is to read the GEX data into a
	 `scanpy` `AnnData` object and manually add the new batch columns to the
	 `adata.obs` array. Then save the new `AnnData` for analyzing with
	 `scripts/run_conga.py` or a jupyter notebook.

	 For example, something like this:
	 ```python
	 import scanpy as sc
	 import pandas as pd
	 
	 old_gex_filename = 'filtered_gene_bc_matrices.h5'
	 new_gex_filename = 'filtered_gene_bc_matrices_w_batches.h5ad'

	 # has columns: barcode donor sample tissue
	 batch_info_tsvfile = 'cell_batch_info.tsv'

	 adata = sc.read_10x_h5(old_gex_filename)
	 batch_info = pd.read_csv(batch_info_tsvfile, sep='\t')
	 batch_info.set_index('barcode', drop=True, inplace=True)

	 df = adata.obs.join(batch_info) # funny behavior if I try to reassign adata.obs
	 for col in batch_info:
	   adata.obs[col] = df[col]
	 adata.uns['batch_keys'] = list(batch_info.columns)

   adata.write_h5ad(new_gex_filename)
	 ```

	 When using `run_conga.py` to analyze data with batches, provide
	 the `adata.obs` batch column names with the argument `--batch_keys`.

   This functionality is still under development. Let us know if you run into any
	 trouble.
1. I have a question that isn't addressed here. What should I do?
   * You could open an issue on github, or email Phil and Stefan (emails at the
	 top of this README) and we will try to help.
</content>
