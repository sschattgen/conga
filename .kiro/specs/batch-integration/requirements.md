# Requirements Document

## Introduction

CoNGA's GEX preprocessing pipeline (`conga.preprocess.filter_normalize_and_hvg` and `conga.preprocess.filter_and_scale`) already exposes an `hvg_batch_key` parameter that makes highly-variable-gene (HVG) selection batch-aware, and a `force_variable_genes` mechanism (already wired to the `--force_variable_genes` CLI flag in `scripts/run_conga.py`) that lets a caller override automatic HVG detection with a fixed gene list read from `adata.uns`. Neither mechanism is connected to any batch *integration* step: there is no `batch_integration()` function anywhere in `conga/preprocess.py` today, and nothing in the current codebase computes a batch-corrected GEX representation.

The project's `dev` branch contains a `batch_integration(adata, method, key, basis='X_pca', adjusted_basis='X_batch', **kwargs)` function supporting Harmony, scVI, and Scanorama, called by `scripts/run_conga.py` independently of `filter_and_scale`'s `hvg_batch_key` — a caller must manually keep a `--hvg_batch_key` value (which does not even exist as a dev CLI flag) and a separate batch-integration key in sync, and dev's own CLI script never does so. This feature closes that gap and narrows the ported functionality:

1. **Full batch integration.** A batch-aware GEX preprocessing pathway that ports `batch_integration()` restricted to exactly two methods, `harmony` and `scvi` (Scanorama and BBKNN, which `dev` also supports, are explicitly excluded from this feature). The pathway is driven by exactly one batch key: HVG selection and the integration method read the same `adata.obs` column, so the two steps cannot silently disagree about which batches exist.
2. **Fixed HVG list, no integration.** An alternative pathway, mutually exclusive with pathway 1, in which the caller supplies a pre-defined gene list to use as the highly-variable set, bypassing automatic HVG detection entirely and performing no batch integration step. This pathway already has a working foundation in the existing `force_variable_genes` mechanism; this feature specifies how a caller populates `adata.var['highly_variable']` or `adata.uns['force_variable_genes']` from an external list and confirms that no batch-integration code runs downstream of it.

A caller picks at most one pathway per preprocessing run. Today's default — automatic, non-batch-aware HVG selection with no integration — remains available and is unaffected by either pathway.

### Why scVI needs a new preprocessing step

`conga.preprocess.filter_normalize_and_hvg` normalizes and log-transforms `adata.raw.X` in place (`sc.pp.normalize_total`, `sc.pp.log1p`) before any HVG or integration step runs, and the only layer CoNGA currently stashes (`adata.layers['scaled']`, added later in `filter_and_scale`) holds mean-centered, unit-variance-scaled values, not raw counts. `dev`'s scVI branch calls `scvi.model.SCVI.setup_anndata(adata, layer="counts", ...)`, which requires an `adata.layers['counts']` holding pre-normalization integer counts. No such layer exists anywhere in CoNGA's current pipeline. This feature must introduce the capture of that layer before normalization runs, or the scVI method is not reachable at all — this is called out explicitly in Requirement 3 rather than assumed away.

## Glossary

- **CoNGA_Preprocessor**: The existing preprocessing code in `conga/preprocess.py`, specifically `filter_normalize_and_hvg` and `filter_and_scale`, which this feature extends.
- **Batch_Key**: The single `adata.obs` column name, supplied once per preprocessing run, that identifies which batch each cell belongs to. The Batch_Key drives both HVG selection and the batch integration method in the Full_Integration_Pathway; it is never specified as two independent values.
- **Full_Integration_Pathway**: The preprocessing pathway in which HVG selection is made batch-aware using the Batch_Key and a batch integration method (Harmony or scVI) is subsequently run using the same Batch_Key, producing a batch-corrected GEX representation.
- **Fixed_HVG_Pathway**: The preprocessing pathway in which the caller supplies a pre-defined gene list that becomes the highly-variable gene set, automatic HVG detection is skipped, and no batch integration method runs.
- **Default_Pathway**: The pre-existing, unmodified CoNGA behavior of automatic, non-batch-aware HVG detection with no batch integration. Neither this feature's pathway is active unless the caller requests one.
- **Integration_Method**: One of the two supported values, `harmony` or `scvi`, naming which algorithm the Full_Integration_Pathway uses to compute a batch-corrected representation.
- **Fixed_Gene_List**: The caller-supplied collection of gene symbols that becomes the highly-variable gene set under the Fixed_HVG_Pathway.
- **Counts_Layer**: The `adata.layers` entry named `counts`, holding the per-cell, per-gene count matrix exactly as it existed before `sc.pp.normalize_total` and `sc.pp.log1p` are applied. Required by the `scvi` Integration_Method and not present anywhere in CoNGA today.
- **Unintegrated_Representation**: The GEX PCA representation computed before batch integration, preserved under a dedicated `adata.obsm` key so that a caller can compare corrected and uncorrected embeddings after the Full_Integration_Pathway runs.
- **Integrated_Representation**: The batch-corrected GEX representation produced by the Integration_Method, stored under a dedicated `adata.obsm` key and subsequently treated by `conga.preprocess.cluster_and_tsne_and_umap` as the active `X_pca_gex` representation.
- **Harmony_Integration**: The `harmony` Integration_Method, implemented by calling `scanpy.external.pp.harmony_integrate`, which itself depends on the third-party `harmonypy` package.
- **ScVI_Integration**: The `scvi` Integration_Method, implemented by calling `scvi.model.SCVI.setup_anndata` and training an `scvi.model.SCVI` model, which depends on the third-party `scvi-tools` package.
- **Analysis_CLI**: The command-line entry point `scripts/run_conga.py`.
- **CoNGA_Preprocessor** function names referenced below (`filter_normalize_and_hvg`, `filter_and_scale`) refer to their current signatures on this repository's master branch, not to the `dev` branch's versions of the same names.

## Requirements

### Requirement 1: Single shared Batch_Key drives both HVG selection and integration

**User Story:** As a CoNGA user with multi-batch data, I want one batch key to control both HVG selection and batch integration, so that the two steps cannot disagree about which column defines a batch.

#### Acceptance Criteria

1. THE CoNGA_Preprocessor SHALL accept a single `batch_key` parameter for the Full_Integration_Pathway that names the `adata.obs` column used both for batch-aware HVG selection and for the Integration_Method.
2. WHEN the Full_Integration_Pathway runs, THE CoNGA_Preprocessor SHALL pass `batch_key` to the existing `hvg_batch_key` parameter of `filter_normalize_and_hvg` without requiring the caller to separately specify `hvg_batch_key`.
3. WHEN the Full_Integration_Pathway runs, THE CoNGA_Preprocessor SHALL pass the same `batch_key` value to the Integration_Method as the batch column.
4. IF the Full_Integration_Pathway is requested and `batch_key` names a column absent from `adata.obs`, THEN THE CoNGA_Preprocessor SHALL raise a `ValueError` naming the missing column before any HVG or integration computation runs.
5. IF the Full_Integration_Pathway is requested and the `adata.obs` column named by `batch_key` contains fewer than two distinct values, THEN THE CoNGA_Preprocessor SHALL raise a `ValueError` naming the column and stating that batch integration requires at least two batches.
6. THE CoNGA_Preprocessor SHALL NOT expose an independent parameter that sets a batch-integration key separately from `hvg_batch_key`, so that the single-key requirement cannot be bypassed by supplying two different values.

> Verification note for criterion 6: this is the gap identified on the `dev` branch, where `filter_and_scale`'s `hvg_batch_key` and `batch_integration`'s `key` argument are independent parameters that `scripts/run_conga.py` never reconciles. This requirement is satisfied by construction if the Full_Integration_Pathway's public entry point takes exactly one `batch_key` parameter and threads it to both steps internally.

### Requirement 2: Integration method selection restricted to Harmony and scVI

**User Story:** As a CoNGA maintainer, I want the ported batch integration function to support only the two methods this project has agreed to maintain, so that Scanorama and BBKNN are not silently available through a code path this feature adds.

#### Acceptance Criteria

1. THE CoNGA_Preprocessor SHALL accept an `Integration_Method` parameter restricted to the values `harmony` and `scvi`.
2. IF a value other than `harmony` or `scvi` is supplied as the Integration_Method, including `scanorama` and `bbknn`, THEN THE CoNGA_Preprocessor SHALL raise a `ValueError` naming the supplied value and listing `harmony` and `scvi` as the supported values.
3. WHERE the Integration_Method is `harmony`, THE CoNGA_Preprocessor SHALL compute the Integrated_Representation by calling `scanpy.external.pp.harmony_integrate` with the Batch_Key column cast to a pandas categorical dtype.
4. WHERE the Integration_Method is `scvi`, THE CoNGA_Preprocessor SHALL compute the Integrated_Representation by calling `scvi.model.SCVI.setup_anndata` followed by training an `scvi.model.SCVI` model and reading `model.get_latent_representation()`.
5. IF the Integration_Method is `harmony` and the `harmonypy` package is not installed, THEN THE CoNGA_Preprocessor SHALL raise an `ImportError` naming `harmonypy` and the optional-dependency extra that installs it.
6. IF the Integration_Method is `scvi` and the `scvi-tools` package is not installed, THEN THE CoNGA_Preprocessor SHALL raise an `ImportError` naming `scvi-tools` and the optional-dependency extra that installs it.
7. THE CoNGA_Preprocessor SHALL import `harmonypy` and `scvi-tools` only inside the code path for the Integration_Method that needs them, so that neither optional dependency is required to import `conga.preprocess`.

### Requirement 3: Counts_Layer capture for scVI

**User Story:** As a CoNGA user selecting the scVI integration method, I want the raw count matrix scVI requires to be captured automatically, so that I do not have to restructure my own preprocessing call to make scVI reachable.

#### Acceptance Criteria

1. WHEN `filter_normalize_and_hvg` runs, THE CoNGA_Preprocessor SHALL capture the per-cell, per-gene count matrix into the Counts_Layer before `sc.pp.normalize_total` and `sc.pp.log1p` are applied to `adata.raw.X`.
2. THE CoNGA_Preprocessor SHALL capture the Counts_Layer regardless of which pathway is subsequently requested, so that a caller can select the Full_Integration_Pathway with the `scvi` Integration_Method after preprocessing has already run.
3. WHEN the Counts_Layer is captured, THE CoNGA_Preprocessor SHALL store a matrix with the same cell and gene ordering as `adata.raw.X` at the time of capture.
4. IF the Integration_Method is `scvi` and no Counts_Layer is present on the AnnData object at the time batch integration is requested, THEN THE CoNGA_Preprocessor SHALL raise a `ValueError` stating that the `scvi` Integration_Method requires the Counts_Layer and naming `filter_normalize_and_hvg` as the step that produces it.
5. WHEN `scvi.model.SCVI.setup_anndata` is called, THE CoNGA_Preprocessor SHALL pass the Counts_Layer as the `layer` argument, the Batch_Key as a categorical covariate key, and `percent_mito` as a continuous covariate key, so that the covariate structure matches the columns CoNGA's preprocessing already populates on `adata.obs`.

> Scope note for criterion 5: `percent_mito` is already computed unconditionally by `filter_normalize_and_hvg` (see the existing `adata.obs['percent_mito']` assignment), so no new covariate-producing code is required beyond wiring the existing column name into the scVI call.

### Requirement 4: Preserving both corrected and uncorrected GEX representations

**User Story:** As a CoNGA user running batch integration, I want both the batch-corrected and the original GEX representations retained on my AnnData object, so that I can compare them or fall back to the uncorrected embedding.

#### Acceptance Criteria

1. WHEN the Full_Integration_Pathway completes, THE CoNGA_Preprocessor SHALL store the GEX PCA representation computed prior to integration as the Unintegrated_Representation under a dedicated `adata.obsm` key distinct from `X_pca_gex`.
2. WHEN the Full_Integration_Pathway completes, THE CoNGA_Preprocessor SHALL store the Integrated_Representation under a dedicated `adata.obsm` key and SHALL set `adata.obsm['X_pca_gex']` to that same Integrated_Representation, so that `conga.preprocess.cluster_and_tsne_and_umap` and `conga.preprocess.calc_nbrs` consume the corrected representation without further changes.
3. THE CoNGA_Preprocessor SHALL define the Unintegrated_Representation and Integrated_Representation `adata.obsm` key names as named, documented module-level constants in `conga/util.py`, following the existing `OBSM_KEY_*` naming convention used for `OBSM_KEY_VEC_TCR` and `OBSM_KEY_PCA_TCR`.
4. WHEN the Full_Integration_Pathway completes, THE CoNGA_Preprocessor SHALL record the Integration_Method, the Batch_Key, and the number of distinct batches under an `adata.uns` key, following the existing `UNS_KEY_*` naming convention used for `UNS_KEY_ACTIVE_TCR_REP`.
5. WHEN an AnnData object carrying an Unintegrated_Representation and an Integrated_Representation is written to `.h5ad` and read back, THE CoNGA_Preprocessor SHALL recover both arrays elementwise equal to the ones written.

### Requirement 5: Fixed_HVG_Pathway from a caller-supplied gene list

**User Story:** As a CoNGA user with a pre-defined gene panel of interest, I want to supply that panel directly as the highly-variable gene set, so that automatic HVG detection and batch integration are both skipped for this run.

#### Acceptance Criteria

1. THE CoNGA_Preprocessor SHALL accept a Fixed_Gene_List as a collection of gene symbols and SHALL support populating the highly-variable gene set from it by way of the existing `adata.uns['force_variable_genes']` mechanism already read by `filter_normalize_and_hvg`.
2. THE CoNGA_Preprocessor SHALL also support a Fixed_Gene_List supplied by the caller setting `adata.var['highly_variable']` directly to a boolean mask before `filter_and_scale` or `filter_normalize_and_hvg` is called, consistent with the pattern of mapping `adata.var_names` against a gene list read from an external file.
3. WHEN the Fixed_HVG_Pathway is active, THE CoNGA_Preprocessor SHALL NOT invoke `sc.pp.highly_variable_genes`, so that no automatic HVG detection overwrites the caller-supplied mask.
4. WHEN the Fixed_HVG_Pathway is active, THE CoNGA_Preprocessor SHALL NOT invoke the Integration_Method, so that no batch-corrected representation is computed for this run.
5. IF the Fixed_HVG_Pathway and the Full_Integration_Pathway are both requested for the same preprocessing call, THEN THE CoNGA_Preprocessor SHALL raise a `ValueError` stating that the two pathways are mutually exclusive.
6. IF a Fixed_Gene_List contains a gene symbol absent from `adata.var_names`, THEN THE CoNGA_Preprocessor SHALL exclude that symbol from the resulting highly-variable mask and SHALL log the excluded symbols and their count at `logging.WARNING` level, rather than raising an error.
7. WHEN the Fixed_HVG_Pathway is active, THE CoNGA_Preprocessor SHALL still apply TR/IG gene exclusion and, where requested, sex-linked gene exclusion to the resulting highly-variable mask, consistent with the existing behavior of the `force_variable_genes` mechanism.
8. WHEN the Fixed_HVG_Pathway is active, THE CoNGA_Preprocessor SHALL record the size of the caller-supplied Fixed_Gene_List and the size of the resulting highly-variable mask after gene exclusion under the existing `adata.uns['conga_stats']` dictionary.

> Verification note for criterion 1: `filter_normalize_and_hvg` on this repository's master branch already reads `adata.uns['force_variable_genes']` (see the existing code building `hvg_mask` from `force_variable_genes` before TR/IG and sex-linked exclusion), and `scripts/run_conga.py` already wires a `--force_variable_genes <path>` flag that reads a plain gene-symbol-per-line file into that key. This requirement's Fixed_HVG_Pathway is this existing, already-functional mechanism; this feature's obligation is criteria 3 through 8, which state behavior the current code does not yet guarantee explicitly (no existing test or requirement currently states that HVG auto-detection and batch integration are skipped when `force_variable_genes` is set).

> Verification note for criterion 2: the user's own example sets `adata.var['highly_variable']` directly via `xdata.var_names.map(...)` rather than populating `adata.uns['force_variable_genes']`. Both mechanisms must be supported because `sc.pp.highly_variable_genes` also writes its automatic result to `adata.var['highly_variable']`, so a caller who sets that column directly and skips the `force_variable_genes` uns key is relying on `filter_normalize_and_hvg` not overwriting a pre-existing `highly_variable` column with its own automatic detection. This is a distinct code path from criterion 1's uns-based override and both must be honored.

### Requirement 6: Analysis_CLI flags for both pathways

**User Story:** As a CoNGA user running the standard analysis script, I want command-line flags for both pathways, so that I do not have to call preprocessing functions programmatically to use batch integration or a fixed gene list.

#### Acceptance Criteria

1. THE Analysis_CLI SHALL accept a `--batch_key` flag naming the `adata.obs` column shared by HVG selection and the Integration_Method.
2. THE Analysis_CLI SHALL accept a `--batch_integration_method` flag restricted to the values `harmony` and `scvi`.
3. IF `--batch_integration_method` is supplied without `--batch_key`, or `--batch_key` is supplied without `--batch_integration_method`, THEN THE Analysis_CLI SHALL exit with a nonzero status and a message naming both flags and stating that the Full_Integration_Pathway requires both.
4. THE Analysis_CLI SHALL retain its existing `--force_variable_genes` flag as the Fixed_HVG_Pathway entry point, with its current behavior of reading a plain gene-symbol-per-line file into `adata.uns['force_variable_genes']`.
5. IF `--force_variable_genes` is supplied together with `--batch_key` or `--batch_integration_method`, THEN THE Analysis_CLI SHALL exit with a nonzero status and a message naming the conflicting flags and stating that the two pathways are mutually exclusive.
6. IF `--batch_integration_method` names a value outside `harmony` and `scvi`, THEN THE Analysis_CLI SHALL exit with a nonzero status and a message naming the supplied value and the two supported values.

### Requirement 7: Dependency declaration

**User Story:** As a person installing CoNGA, I want the new optional dependencies declared, so that I can install exactly the support I need without guessing package names.

#### Acceptance Criteria

1. THE CoNGA_Preprocessor's packaging metadata SHALL declare `harmonypy` as an optional dependency associated with the Harmony_Integration method.
2. THE CoNGA_Preprocessor's packaging metadata SHALL continue to declare `scvi-tools` as an optional dependency associated with the ScVI_Integration method, consistent with the existing `scvi` extras group.
3. THE CoNGA_Preprocessor's packaging metadata SHALL NOT declare `scanorama` or `bbknn` as dependencies of the extras group introduced by this feature, so that installing batch-integration support for this feature does not pull in the two excluded methods.

> Scope note for criterion 3: `bbknn` already exists as a separate, pre-existing optional-dependency group (`batch`) in `pyproject.toml`, unrelated to this feature. This feature does not remove that group; it only avoids adding `bbknn` or `scanorama` to whichever extras group it introduces for Harmony and scVI.

## Out of Scope

- **Scanorama and BBKNN integration methods.** `dev`'s `batch_integration()` supports `scanorama` in addition to `harmony` and `scvi`; BBKNN is a separate code path in `dev` entirely. Neither is ported by this feature. See Requirement 2.
- **`hvg_batch_key` as an independently callable parameter for the Full_Integration_Pathway's public entry point.** It remains an internal parameter of `filter_normalize_and_hvg` that the Full_Integration_Pathway sets from `batch_key`; callers of the new pathway do not set it separately. See Requirement 1.
- **BBKNN-based GEX neighbor graph construction** (`use_bbknn`, `bbknn_batch_key` on `conga.correlations`), described in the branch-integration-roadmap steering document as a `dev`/`bcr`-branch feature distinct from `batch_integration()`. Not addressed here.
- **Cell type prediction models, TCR QC module, Meta-CoNGA matching, and other `dev`/`bcr`/`metaconga_match` branch features** unrelated to batch-aware GEX preprocessing.
- **Automatic detection or discovery of a batch column.** The caller must name the Batch_Key explicitly; this feature does not infer batch structure from metadata.
- **GPU-specific configuration for scVI training** (for example, device selection or mixed precision). The `scvi.model.SCVI` defaults apply; performance tuning of scVI training is not addressed.
- **Changing the Default_Pathway.** Automatic, non-batch-aware HVG selection with no integration remains the default when neither pathway is requested; this feature adds two opt-in alternatives.

## Resolved Decisions

- **Two methods only, not three or four:** `harmony` and `scvi` are supported; `scanorama` and `bbknn` are explicitly excluded, per the user's stated scope. See Requirement 2.
- **One shared key, not two:** the Full_Integration_Pathway's public entry point takes a single `batch_key` parameter that drives both HVG selection and the Integration_Method internally, closing the gap identified on `dev` where `hvg_batch_key` and the integration `key` argument could diverge. See Requirement 1.
- **Two pathways are mutually exclusive alternatives to each other and to the Default_Pathway:** a caller selects at most one of the Full_Integration_Pathway or the Fixed_HVG_Pathway per run; both are opt-in relative to today's unmodified default behavior. See Requirement 5.5 and Requirement 6.5.
- **The Fixed_HVG_Pathway builds on an already-working mechanism:** `force_variable_genes` and the `--force_variable_genes` CLI flag already exist and are already wired end to end on this repository's master branch. This feature's obligation for that pathway is to state and verify the previously-implicit guarantee that automatic HVG detection and batch integration are both skipped when it is active, and to additionally support the direct `adata.var['highly_variable']` assignment pattern from the user's own example. See Requirement 5.
- **A Counts_Layer must be introduced for scVI to be reachable at all:** no code path in CoNGA today stashes raw counts before normalization, and `dev`'s scVI branch requires exactly that. This feature adds the capture step as part of `filter_normalize_and_hvg` rather than asking the scVI Integration_Method to re-derive counts from already-normalized data. See Requirement 3.
- **New `obsm`/`uns` keys follow the existing `OBSM_KEY_*`/`UNS_KEY_*` convention in `conga/util.py`**, the same convention used for the vectorized-tcrdist feature's `OBSM_KEY_VEC_TCR`, `OBSM_KEY_PCA_TCR`, and `UNS_KEY_ACTIVE_TCR_REP` constants, rather than introducing an independent naming scheme. See Requirement 4.3 and 4.4.

## Open Questions

None outstanding.
