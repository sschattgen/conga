# Updates

## Since v0.1 (2021): what's changed

CoNGA has gone through a major performance and packaging overhaul since the
original 0.1 release. The short version:

* **Much faster on large datasets.** Vectorized TCRdist replaces the old
  KernelPCA approach for most organisms, cutting memory use from gigabytes
  to tens of megabytes. Optional FAISS acceleration speeds up neighbor
  search by 10-100x, with automatic fallback to sklearn if it's not
  installed.
* **Modernized under the hood.** CoNGA now requires Python 3.12+ and has
  been updated for pandas 3.0+ and numpy 2.0+. Packaging moved to
  `pyproject.toml`, so installation and optional feature sets
  (`[performance]`, `[batch-integration]`, etc.) work the normal pip way.
* **Many more organisms.** Support grew from a handful of species to over
  20, including gamma-delta T cells and B cell receptors across human,
  mouse, rhesus, and others.
* **New analysis capabilities.** TCR clumping, TCR database matching,
  Hotspot-based feature discovery, multi-sample merging, and GEX batch
  correction (Harmony/scVI) were all added after 0.1.
* **Easier to run.** Docker support and a dedicated CLI (`conga` command)
  were added alongside the packaging modernization.

See the full version-by-version breakdown in [CHANGELOG.md](../CHANGELOG.md)
for exact details on any of the above.

---

* **Latest: Python 3.14 support**
  * The full test suite has been run and passes on Python 3.12 and 3.14 (`environment.yml` installs 3.14 by default for development). `requires-python` in `pyproject.toml` remains `>=3.12`. Python 3.13 is expected to work based on dependency metadata but has not been directly tested.
  * The `louvain` package (backing the deprecated `--clustering_method louvain` option) is now an optional `conga[legacy-clustering]` extra rather than a core dependency, and is only installable on Python 3.12 -- conda-forge has no build for 3.13+, and building it from source fails there against modern compilers (its vendored igraph C core trips `-Werror=uninitialized-const-pointer`). The default `--clustering_method leiden` (backed by `leidenalg`, unaffected) has no such restriction, so most users will not notice this change.
  * `bbknn` was removed from `environment.yml` entirely. It had the same conda-forge Python-version ceiling as `louvain`, and was never actually wired to any code path: `conga.preprocess.batch_integration()` only supports `method='harmony'` or `method='scvi'` and explicitly rejects `'bbknn'`.

* **Species expansion and GEX batch integration**
  * Organism support now breaks down into three tiers: 22 organisms with germline gene database coverage, 21 usable through the `--organism` CLI flag, and 18 eligible for the vectorized TCRdist path (see the Organism support section above). Scoping across these tiers is based on chain completeness in the gene database, not an arbitrary cutoff.
  * Added `conga.preprocess.batch_integration()` for GEX batch correction via Harmony or scVI, exposed through `scripts/run_conga.py --batch_key`/`--batch_integration_method` and installed with `pip install -e ".[batch-integration]"` (from a local clone -- CoNGA is not published on PyPI). This is distinct from the existing `--batch_keys` annotation-only mechanism, which still only drives visualization and does not correct GEX.

* **2024-12-19: Version 0.2.0 - Major Performance Release**
  CoNGA 0.2.0 introduced vectorized TCRdist, a fixed-length-vector encoding for TCRs that replaces the quadratic-memory KernelPCA approach for eligible organisms, along with optional FAISS acceleration for neighbor search (tiered backend selection: faiss-gpu → faiss-cpu → sklearn). This release also raised the minimum Python version to 3.12 and updated dependencies for pandas 3.0+/numpy 2.0+ compatibility. The default TCR representation for vectorized-eligible organisms changed from KernelPCA to the vectorized encoding; see the TCR Representations section above for how representation selection now works.

* 2023-09-21: Rhesus alpha beta and gamma delta T cells are now supported.
* 2021-09-10: Rescale the adata.X gene expression matrix after reducing to a
single clone. In very rare cases, not doing this was leading to wonky GEX UMAPs and
clusters, seemingly due to GEX PC components dominated by individual genes.

* 2021-01-21: (EXPERIMENTAL) New mode of analysis in which the paired TCR
sequences in the analyzed dataset are matched to paired sequences in a literature-derived
database compiled from VDJdb, McPAS, the large 10x dextramer dataset,
the protein structure databank, and a few other studies. See the database
[README](conga/data/new_paired_tcr_db_for_matching_nr_README.txt) for citation
details and the
[database itself](conga/data/new_paired_tcr_db_for_matching_nr.tsv).
This mode of analysis is included in `scripts/run_conga.py --all` or with
the `--match_to_tcr_database` flag. Or from within the `conga` package
using the `conga.tcr_clumping.match_adata_tcrs_to_db_tcrs` function.
You can also pass in a user-provided paired TCR sequence database for matching
against using the
`run_conga.py` command line flag `--tcr_database_tsvfile` or the second
argument to the `conga.tcr_clumping.match_adata_tcrs_to_db_tcrs`
function. The statistical significance of matches is evaluated using the
same background `tcrdist` calculations that go into the 'TCR clumping'
analysis described below (which incidentally means that this mode
requires compilation of the C++ tcrdist executable as described in
the installation section above).

* 2020-12-31: (EXPERIMENTAL) New mode of analysis, TCR clumping, to detect
clustered regions of TCR space. This mode will identify TCR clonotypes that have
more TCRdist neighbors at specified distance thresholds in the analyzed dataset
than would be expected by chance under
a simple null model based on shuffling the observed alpha-beta and V-J pairings
while (mostly) preserving V(D)J rearrangement statistics. Accessed through the
`conga/scripts/run_conga.py` script with the flag `--tcr_clumping` or when
using `--all` to run all major analyses. Also accessible in the python package
via the `conga.tcr_clumping.assess_tcr_clumping` routine. Note that this
analysis does not use the GEX information at all. Inspired by ALICE from Walczak
and Mora and TCRnet from the VDJtools folks.

* 2020-12-31: C++ implementation of TCRdist distance calculations for speed and
to power the 'TCR clumping' analysis. Requires C++ compiler. Code is stored in
`conga/tcrdist_cpp` and compiled as described above in the Installation section.

* 2020-09-16: (EXPERIMENTAL) Added a preliminary implementation of the
Hotspot autocorrelation algorithm developed by the Yosef lab, for finding informative features
in multi-modal data (check out the [github repo](https://github.com/YosefLab/Hotspot)
and the [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2020.02.06.937805v1)).
Hotspot finds numerical features whose pattern of variation across a single-cell
dataset respects a user-supplied notion of cell-cell similarity (a neighbor graph
with edge weights). We are using Hotspot to identify genes that respect the TCR
neighbor graph, and TCR features that respect the gene expression neighbor
graph (see examples in the Examples section below).

* 2020-09-16: Added a simple script for merging multiple datasets
(`scripts/merge_samples.py`). More functionality to
come in the future; for the time being this will merge multiple datasets that each could
be run through conga individually (ie clones files have already been generated with
associated .barcode_mapping.tsv and kernel PCA files, and the barcodes in the barcode
mapping files match those in the GEX data files. These conditions will be satisfied if
each was generated by `scripts/setup_10x_for_conga.py`). The input to the script
is a tab-separated values `.tsv` file with three columns (corresponding to the three
input arguments for `scripts/run_conga.py`: `clones_file` `gex_data` and `gex_data_type`)
which give the locations of the clonotype and gene expression data files.
If these datasets are from the same individual and/or could contain cells from the
same expanded clonotypes it might be worth using the arguments `--condense_clonotypes_by_tcrdist
--tcrdist_threshold_for_condensing 0.01` which will merge clonotypes containing identical
TCR sequences (for BCRs a larger tcrdist threshold value of 50ish might make sense).

* 2020-09-04: (EXPERIMENTAL) Added support for bcrs and for gamma-delta TCRs. Right now `conga` uses the
`'organism'` specifier to communicate the data type: `human` and `mouse` mean alpha-beta TCRs;
`human_gd` and `mouse_gd` mean gamma-deltas; `human_ig` means B cell data (not setup for mouse
yet but let us know if that's of interest to you, for example by opening an Issue on github).
Hacking the organism field like this allows us to put all the gene sequence information into a single,
enlarged database file (`conga/tcrdist/db/combo_xcrs.tsv`). We still haven't updated all the
image labels and filenames, so even though you are analyzing BCR data your plots will probably
still say TCR in a few places...
</content>