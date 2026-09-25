# Requirements Document

## Introduction

CoNGA currently derives its TCR representation by computing a full pairwise TCRdist matrix and reducing it with KernelPCA (`conga.preprocess.make_tcrdist_kernel_pcs_file_from_clones_file`). That function materializes a dense N×N distance array `D` and then a second dense N×N Gram array before calling `KernelPCA.fit_transform`. At N=20000 each of those is roughly 3.2 GB in float64. The quadratic memory cost lives in the *reduction*, not in TCRdist itself.

A prototype vectorized encoder already exists in the repository at `conga/tcrdist_vectorizing_functions_for_sharing.py`. It embeds the TCRdist amino acid substitution matrix into Euclidean space with MDS, then encodes each TCR chain as a fixed-length real vector by concatenating per-position amino acid vectors for the germline CDR1/CDR2/CDR2.5 loops and a trimmed-and-gapped CDR3. Euclidean distance between two such vectors approximates TCRdist, so neighbor search can be done in vector space without materializing a distance matrix.

The prototype is not usable as shipped: it hardcodes an absolute path belonging to another developer's machine and asserts that path exists at import time, it reads the gene database from the wrong location, it only accepts `human` and `mouse`, it expects AIRR-style column names rather than CoNGA's, and it hardcodes an MDS random seed that differs from the project standard.

This feature refactors the prototype into a supported CoNGA module, `conga/tcrdist/vectorized.py`, that produces reproducible fixed-length vector encodings of paired receptor chains, stores them in AnnData, and validates their accuracy against the existing exact TCRdist implementation.

### Three TCR neighbor paths, not two

CoNGA already contains a third path that neither computes KernelPCA nor stores an `adata.obsm` array: `conga.preprocess.calculate_tcrdist_nbrs`, reached today via `run_conga.py --no_kpca` / `--use_exact_tcrdist_nbrs`. It computes Exact_TCRdist distances on demand and emits neighbor index arrays directly. Both of its implementations stream:

- `calculate_tcrdist_nbrs_python` loops over clonotypes, building one length-N distance vector at a time and discarding it after `argpartition`. Peak additional memory is O(N).
- `calculate_tcrdist_nbrs_cpp` writes the clonotype table to a temporary TSV, shells out to the `find_neighbors` binary, and reads back `knn_indices` and `knn_distances` of shape (N, num_nbrs). `find_neighbors.cc` holds a single length-N row buffer and streams results to disk.

Neither route ever holds a dense N×N array, so exact TCRdist neighbor search is viable at observation counts where KernelPCA is not. Its output — a mapping from `nbr_frac` to an int32 (N, num_nbrs) index array — is the same shape the `obsm`-based path produces, so downstream analysis consumes it unchanged. It does, however, store nothing in `adata.obsm`; `calc_nbrs` signals this path by setting `obsm_tag_tcr` to `None`.

This feature therefore specifies **three** TCR neighbor paths and the rules for choosing among them:

1. **Vectorized** (`adata.obsm['X_vec_tcr']`) — alpha-beta organisms only, and the default for those organisms at any observation count.
2. **Exact TCRdist reduced by KernelPCA** (`adata.obsm['X_pca_tcr']`) — available only below a documented observation-count limit, because the reduction is what carries the quadratic memory cost. It remains the default for organisms the vectorizer does not support, below that limit.
3. **Exact TCRdist used directly, no KernelPCA** — available for every organism at every observation count, selectable explicitly, and selected automatically for unsupported organisms at or above the limit. It stores no `adata.obsm` array.

Making the vectorized encoding the default for alpha-beta data is an intentional behavior change, not an opt-in addition. Escape hatches are preserved: a user can request the KernelPCA path on a small dataset to reproduce an older analysis, and can request the raw exact path at any size to obtain reference-quality distances for comparison against the vectorized approximation.

Vectorized *encoding* covers alpha-beta organisms only. Gamma-delta and Ig receptors reach path 2 below the limit and path 3 at or above it. No dataset is rejected on size grounds. Extending the vectorizer to those receptor types is deferred.

FAISS/GPU neighbor search, batch integration, containerization, MuData support, and new-species reference building are explicitly out of scope; they are separate roadmap objectives. This feature only has to emit dense arrays that a future FAISS backend can consume directly.

## Glossary

- **TCR_Vectorizer**: The new module `conga/tcrdist/vectorized.py`, containing the public encoding API and its helper functions.
- **Encoding_Config**: The set of tunable parameters that determine an encoding: MDS dimensionality (`aa_mds_dim`), fixed CDR3 position count (`num_pos_cdr3`), CDR3 weight (`cdr3_weight`), N-terminal trim length (`n_trim`), C-terminal trim length (`c_trim`), and random seed (`random_seed`).
- **AA_Embedding**: The mapping from each of the 20 amino acids plus the gap character to a real vector of length `aa_mds_dim`, produced by multidimensional scaling of the square root of the TCRdist amino acid distance matrix.
- **Chain_Vector**: The fixed-length real vector encoding a single receptor chain (germline CDR loops plus CDR3) for one clonotype.
- **TCR_Vector**: The concatenation of the two Chain_Vectors of a paired clonotype, i.e. one row of the output matrix.
- **Vector_Matrix**: The two-dimensional array of TCR_Vectors for a set of clonotypes, one row per clonotype.
- **Exact_TCRdist**: The existing reference TCRdist implementation in CoNGA, either `conga.tcrdist.tcr_distances.TcrDistCalculator` (Python) or the compiled `tcrdist_cpp` binaries.
- **Gene_Database**: The V/J gene reference data loaded by `conga.tcrdist.all_genes`, which supplies per-gene CDR loop strings keyed by organism and chain.
- **CoNGA_Preprocessor**: The existing preprocessing code in `conga/preprocess.py` that builds AnnData objects and computes neighbor graphs.
- **Analysis_CLI**: The command-line entry point `scripts/run_conga.py`.
- **Setup_CLI**: The command-line entry point `scripts/setup_10x_for_conga.py`.
- **Accuracy_Report**: The structured result produced when vectorized distances are compared against Exact_TCRdist on a set of clonotypes, containing distance correlation and neighbor-set recall figures.
- **Supported_Organism**: An organism string accepted by the TCR_Vectorizer, namely `human`, `mouse`, or `rhesus`. Gamma-delta and Ig organism strings are not Supported_Organisms.
- **Chain_Label**: The single-character chain identifier used by the Gene_Database, `A` for the alpha chain and `B` for the beta chain.
- **Vectorized_Representation**: The Vector_Matrix produced by the TCR_Vectorizer and stored under the `adata.obsm` key `X_vec_tcr`.
- **KPCA_Representation**: The TCR representation produced by reducing an Exact_TCRdist matrix with KernelPCA and stored under the `adata.obsm` key `X_pca_tcr`.
- **Exact_Nbr_Path**: The TCR neighbor path implemented by `conga.preprocess.calculate_tcrdist_nbrs`, which computes Exact_TCRdist distances on demand and returns neighbor index arrays directly, storing no array in `adata.obsm`.
- **KPCA_Reduction_Limit**: The observation count at or above which KernelPCA reduction of an Exact_TCRdist matrix is not applied, because that reduction requires dense N×N arrays. Its default value is 20000 and it is settable by caller parameter and by CLI flag. It does not gate whether CoNGA can analyze a dataset; it gates only whether the KernelPCA reduction is performed.
- **Active_Representation**: The TCR neighbor path a given run used, recorded as one of three values: the `adata.obsm` key `X_vec_tcr`, the `adata.obsm` key `X_pca_tcr`, or the sentinel string `exact_tcrdist` denoting the Exact_Nbr_Path, which has no `adata.obsm` key.
- **Default_Seed**: The integer 42, the project-standard deterministic random seed.

## Requirements

### Requirement 1: Portable module initialization

**User Story:** As a CoNGA user installing the package on any machine, I want the vectorized TCRdist code to import cleanly, so that I can use it without editing source paths.

#### Acceptance Criteria

1. THE TCR_Vectorizer SHALL resolve all reference data locations from `conga.util` path helpers or from the Gene_Database rather than from string literals containing absolute filesystem paths.
2. WHEN the TCR_Vectorizer module is imported, THE TCR_Vectorizer SHALL complete the import without reading any file from disk and without evaluating any filesystem existence assertion at module scope.
3. WHEN any module of the conga package is imported on a machine holding only the installed package and its declared dependencies, THE conga package SHALL complete the import, so that no module depends on a filesystem path specific to one developer's machine.

### Requirement 2: Deterministic amino acid embedding

**User Story:** As a researcher publishing CoNGA results, I want vector encodings to be reproducible, so that a re-run of my analysis yields the same numbers.

#### Acceptance Criteria

1. THE TCR_Vectorizer SHALL accept a `random_seed` parameter that controls the multidimensional scaling used to build the AA_Embedding, defaulting to Default_Seed.
2. WHEN the AA_Embedding is computed twice within one process with identical Encoding_Config values, THE TCR_Vectorizer SHALL produce two arrays that are elementwise identical.
3. WHEN the same clonotype set is encoded in two separate processes with identical Encoding_Config values, THE TCR_Vectorizer SHALL produce two Vector_Matrix values that are elementwise identical.
4. THE TCR_Vectorizer SHALL record the stress value reported by the multidimensional scaling fit at `logging.INFO` level.
5. WHEN the AA_Embedding is computed, THE TCR_Vectorizer SHALL include the gap character as a 21st symbol whose distance to every amino acid equals the TCRdist gap penalty per position.

### Requirement 3: Shared gene reference data, organism, and chain coverage

**User Story:** As a CoNGA user working with alpha-beta, gamma-delta, or B cell data, I want the vectorized encoder to draw its germline reference data from the same gene database the exact TCRdist path reads, and to state plainly which receptor types it supports, so that I do not silently get wrong distances.

#### Acceptance Criteria

1. THE TCR_Vectorizer SHALL obtain germline CDR loop strings for a given organism and Chain_Label from the Gene_Database, so that the vectorized encoding and Exact_TCRdist read the same reference gene records.
2. THE TCR_Vectorizer SHALL support the alpha-beta organisms `human`, `mouse`, and `rhesus`, reading Chain_Label `A` records for the alpha chain and Chain_Label `B` records for the beta chain.
3. IF an organism string outside the supported set is supplied, including the gamma-delta strings `human_gd`, `mouse_gd`, and `rhesus_gd` and the Ig strings `human_ig` and `mouse_ig`, THEN THE TCR_Vectorizer SHALL raise a `ValueError` listing the supported organism strings and naming the KPCA_Representation and the Exact_Nbr_Path as the alternatives available for the rejected organism.
4. IF a requested organism and Chain_Label combination has no V gene records at all in the Gene_Database, THEN THE TCR_Vectorizer SHALL raise a `ValueError` naming the organism and the Chain_Label.
5. WHEN germline CDR loop strings are assembled for a Supported_Organism and Chain_Label, THE TCR_Vectorizer SHALL concatenate every CDR loop record of the V gene except the final record, which holds the CDR3 N-terminal residues, so that the germline string is free of CDR3 residues.
6. WHEN germline CDR loop strings are assembled for a Supported_Organism and Chain_Label, THE TCR_Vectorizer SHALL produce strings of equal length for every V gene of that organism and Chain_Label.
7. THE TCR_Vectorizer SHALL remove germline CDR string positions that carry the same character across every V gene of the organism and Chain_Label being encoded.

> Verification note for criterion 1: `conga/tcrdist/tcr_distances.py` imports `all_genes` from `conga.tcrdist.all_genes` and builds its germline V-loop strings as `' '.join(g.cdrs[:-1])`. The Gene_Database is therefore already the single source the Exact_TCRdist path reads, so sourcing the vectorized germline strings from it gives both paths identical reference gene records rather than two independently maintained copies.

> Scope note for criterion 4: this criterion covers the case where the Gene_Database holds no V gene record whatsoever for the requested organism and Chain_Label, which makes encoding that chain impossible. Requirement 4.4 covers the different case where the combination is populated but one V gene identifier supplied in the input is not among its records. The two failures have distinct causes and distinct messages.

> Verification note for criterion 5: every V gene record in the active gene database (`conga/tcrdist/db/combo_xcr_2023-12-30.tsv`, selected by `conga.tcrdist.basic.db_file`) carries exactly four CDR loop entries, and no V record lacks CDR annotation. The record layout is therefore uniform across `human`, `mouse`, and `rhesus`, and dropping the final record yields the three germline loops. This matches the `g.cdrs[:-1]` convention already used in `conga/tcrdist/all_genes.py`.

### Requirement 4: CoNGA-native input handling

**User Story:** As a CoNGA developer, I want the encoder to read the column names CoNGA already uses, so that I can call it on a clones file or on `adata.obs` without renaming columns.

#### Acceptance Criteria

1. THE TCR_Vectorizer SHALL accept clonotype input as the list-of-nested-tuples structure returned by `conga.preprocess.retrieve_tcrs_from_adata`.
2. THE TCR_Vectorizer SHALL accept clonotype input as a `pandas.DataFrame` whose V gene and CDR3 columns are named `va`, `cdr3a`, `vb`, and `cdr3b`.
3. WHERE the caller supplies explicit column names, THE TCR_Vectorizer SHALL read V gene and CDR3 values from the named columns instead of the defaults.
4. IF a V gene identifier in the input is absent from the Gene_Database for the requested organism, THEN THE TCR_Vectorizer SHALL raise a `ValueError` naming the offending identifier and the count of affected clonotypes.
5. IF a CDR3 string in the input contains a character outside the 20 standard amino acid single-letter codes, THEN THE TCR_Vectorizer SHALL raise a `ValueError` naming the offending string.
6. IF a CDR3 string in the input has fewer residues than the sum of `n_trim` and `c_trim` plus one, THEN THE TCR_Vectorizer SHALL raise a `ValueError` naming the offending string and the trim parameters.
7. WHEN a CDR3 string is longer than the encodable length implied by `num_pos_cdr3`, `n_trim`, and `c_trim`, THE TCR_Vectorizer SHALL encode the CDR3 with interior residues dropped and SHALL log the count of such clonotypes at `logging.WARNING` level.
8. THE TCR_Vectorizer SHALL validate all input before allocating the Vector_Matrix.

### Requirement 5: Fixed-length vector output

**User Story:** As a developer of downstream neighbor search, I want a predictable dense numeric matrix, so that I can hand it to a vector index without reshaping or copying.

#### Acceptance Criteria

1. WHEN a set of N paired clonotypes is encoded with a given Encoding_Config, THE TCR_Vectorizer SHALL return a `numpy.ndarray` of shape (N, L) where L is determined solely by the Encoding_Config, the organism, and the Gene_Database.
2. THE TCR_Vectorizer SHALL return an array of dtype `numpy.float32` in C-contiguous memory order.
3. THE TCR_Vectorizer SHALL expose a function that returns the vector length L for a given organism and Encoding_Config without encoding any clonotype.
4. WHEN the Vector_Matrix is returned, THE TCR_Vectorizer SHALL return an array containing only finite values.
5. WHEN a single Chain_Vector is built, THE TCR_Vectorizer SHALL scale the CDR3 portion by the square root of `cdr3_weight`, so that squared Euclidean distance reproduces the TCRdist CDR3 weighting.
6. THE TCR_Vectorizer SHALL accept `aa_mds_dim`, `num_pos_cdr3`, `cdr3_weight`, `n_trim`, `c_trim`, and `random_seed` as parameters with documented defaults.
7. WHEN the encoded CDR3 of a clonotype is built, THE TCR_Vectorizer SHALL produce exactly `num_pos_cdr3` encoded positions regardless of the input CDR3 length.
8. WHEN a Vector_Matrix is built for N clonotypes, THE TCR_Vectorizer SHALL complete the encoding without allocating any array whose element count is proportional to the square of N.

### Requirement 6: Accuracy against exact TCRdist

**User Story:** As a researcher deciding whether to use the vectorized path, I want a measured statement of how closely it reproduces TCRdist, so that I can judge whether my conclusions would change.

#### Acceptance Criteria

1. THE TCR_Vectorizer SHALL provide a function that, given a clonotype set and an organism, returns an Accuracy_Report comparing vectorized Euclidean distances against Exact_TCRdist distances.
2. WHEN an Accuracy_Report is produced, THE TCR_Vectorizer SHALL include the Pearson correlation and the Spearman correlation between vectorized distance and Exact_TCRdist distance over the sampled clonotype pairs.
3. WHEN an Accuracy_Report is produced, THE TCR_Vectorizer SHALL include, for each requested neighbor count k, the mean fraction of each clonotype's k nearest Exact_TCRdist neighbors that also appear among its k nearest vectorized neighbors.
4. WHEN an Accuracy_Report is produced, THE TCR_Vectorizer SHALL record the neighbor counts k at which recall was measured, chosen to match the neighbor counts CoNGA derives from its `nbr_frac` settings.
5. WHERE the clonotype set exceeds a caller-supplied pair-sampling limit, THE TCR_Vectorizer SHALL compute correlations on a seeded random sample of clonotype pairs and SHALL record the sample size in the Accuracy_Report.
6. WHEN the accuracy test suite runs on the bundled test clonotype sets, THE Accuracy_Report SHALL show a Spearman correlation of at least 0.95 and a mean recall of at least 0.80 at every measured neighbor count.
7. THE TCR_Vectorizer SHALL compare against Exact_TCRdist values produced by the Python `TcrDistCalculator`, so that accuracy testing does not require compiled binaries.

### Requirement 7: Storage in AnnData

**User Story:** As a CoNGA user, I want the vector encoding stored on my AnnData object, so that downstream steps and saved `.h5ad` files carry it.

#### Acceptance Criteria

1. THE CoNGA_Preprocessor SHALL provide a function that encodes the clonotypes held in an AnnData object and stores the resulting Vector_Matrix in `adata.obsm`.
2. THE CoNGA_Preprocessor SHALL use the `adata.obsm` key `X_vec_tcr` for the Vectorized_Representation, following the existing `X_<representation>_<modality>` naming of `X_pca_tcr` and `X_pca_gex`.
3. WHEN a Vector_Matrix is stored in `adata.obsm`, THE CoNGA_Preprocessor SHALL store the Encoding_Config values used, the organism, and the TCR_Vectorizer version tag under an `adata.uns` key.
4. WHEN a Vector_Matrix is stored in `adata.obsm`, THE CoNGA_Preprocessor SHALL store rows in the same order as `adata.obs`.
5. IF the `adata.obsm` key `X_vec_tcr` already holds an array, THEN THE CoNGA_Preprocessor SHALL overwrite the array and SHALL log the overwrite at `logging.WARNING` level.
6. WHEN a Vector_Matrix is stored on an AnnData object that already holds a KPCA_Representation, THE CoNGA_Preprocessor SHALL leave the `X_pca_tcr` array unchanged.
7. WHEN a TCR neighbor path is selected, THE CoNGA_Preprocessor SHALL record the Active_Representation under an `adata.uns` key as one of the three values `X_vec_tcr`, `X_pca_tcr`, or `exact_tcrdist`, so that a reader can tell which path the run used.
8. WHERE the Active_Representation is `exact_tcrdist`, THE CoNGA_Preprocessor SHALL record that value without creating any `adata.obsm` entry for it, because the Exact_Nbr_Path produces neighbor index arrays rather than a per-observation representation.
9. WHEN an AnnData object carrying a stored Vector_Matrix is written to `.h5ad` and read back, THE CoNGA_Preprocessor SHALL recover a Vector_Matrix elementwise equal to the one written and Encoding_Config values equal to those written.
10. WHEN an AnnData object carrying a recorded Active_Representation is written to `.h5ad` and read back, THE CoNGA_Preprocessor SHALL recover the same Active_Representation value.

### Requirement 8: Three-way TCR neighbor path selection

**User Story:** As a CoNGA user analyzing any receptor type at any scale, I want CoNGA to pick a TCR neighbor path that fits my dataset, and to let me override that choice, so that large runs complete, unsupported receptor types still work, and small runs remain reproducible.

#### Selection table

Every combination of organism support, observation count, and override resolves to exactly one outcome:

| Organism | Observation count | Override requested | Outcome |
|---|---|---|---|
| Supported_Organism | any | none | Vectorized_Representation, Active_Representation `X_vec_tcr` |
| Supported_Organism | any | Exact_Nbr_Path | Exact_Nbr_Path, Active_Representation `exact_tcrdist` |
| Supported_Organism | below KPCA_Reduction_Limit | KernelPCA | KPCA_Representation, Active_Representation `X_pca_tcr` |
| Supported_Organism | at or above KPCA_Reduction_Limit | KernelPCA | Error naming the limit |
| not a Supported_Organism | below KPCA_Reduction_Limit | none | KPCA_Representation, Active_Representation `X_pca_tcr` |
| not a Supported_Organism | at or above KPCA_Reduction_Limit | none | Exact_Nbr_Path, Active_Representation `exact_tcrdist` |
| not a Supported_Organism | any | Exact_Nbr_Path | Exact_Nbr_Path, Active_Representation `exact_tcrdist` |
| not a Supported_Organism | below KPCA_Reduction_Limit | KernelPCA | KPCA_Representation, Active_Representation `X_pca_tcr` |
| not a Supported_Organism | at or above KPCA_Reduction_Limit | KernelPCA | Error naming the limit |

#### Acceptance Criteria

**Limit definition**

1. THE CoNGA_Preprocessor SHALL accept a `kpca_reduction_limit` parameter, defaulting to 20000, that sets the KPCA_Reduction_Limit.
2. THE CoNGA_Preprocessor SHALL define the KPCA_Reduction_Limit default as a single named, documented module-level constant that the Analysis_CLI, the Setup_CLI, and `scripts/merge_samples.py` all read for their flag defaults.
3. WHILE the observation count of the dataset is at or above KPCA_Reduction_Limit, THE CoNGA_Preprocessor SHALL leave the KernelPCA computation unperformed.
4. THE CoNGA_Preprocessor SHALL apply KPCA_Reduction_Limit only to the decision of whether to perform the KernelPCA reduction, and SHALL analyze datasets at or above that limit by way of the Vectorized_Representation or the Exact_Nbr_Path.

**Automatic selection**

5. WHERE the organism is a Supported_Organism and no override is requested, THE CoNGA_Preprocessor SHALL build the Vectorized_Representation and SHALL set the Active_Representation to `X_vec_tcr`, regardless of observation count.
6. WHERE the organism is not a Supported_Organism and no override is requested, WHILE the observation count is below KPCA_Reduction_Limit, THE CoNGA_Preprocessor SHALL build the KPCA_Representation and SHALL set the Active_Representation to `X_pca_tcr`.
7. WHERE the organism is not a Supported_Organism and no override is requested, WHILE the observation count is at or above KPCA_Reduction_Limit, THE CoNGA_Preprocessor SHALL select the Exact_Nbr_Path, SHALL set the Active_Representation to `exact_tcrdist`, and SHALL log at `logging.INFO` level that the observation count reached KPCA_Reduction_Limit and that exact TCRdist neighbors will be computed directly.

**Explicit overrides**

8. WHERE the Exact_Nbr_Path override is requested, THE CoNGA_Preprocessor SHALL select the Exact_Nbr_Path and SHALL set the Active_Representation to `exact_tcrdist`, for every organism and at every observation count.
9. WHERE the KernelPCA override is requested, WHILE the observation count is below KPCA_Reduction_Limit, THE CoNGA_Preprocessor SHALL build the KPCA_Representation from Exact_TCRdist and SHALL set the Active_Representation to `X_pca_tcr`.
10. IF the KernelPCA override is requested WHILE the observation count is at or above KPCA_Reduction_Limit, THEN THE CoNGA_Preprocessor SHALL raise a `ValueError` naming the observation count, the KPCA_Reduction_Limit value, and the parameter that raises the limit.
11. IF the KernelPCA override and the Exact_Nbr_Path override are both requested, THEN THE CoNGA_Preprocessor SHALL raise a `ValueError` naming both overrides.

**Exact_Nbr_Path prerequisites**

12. WHERE the Active_Representation is `exact_tcrdist`, THE CoNGA_Preprocessor SHALL compute TCR neighbors without allocating any array whose element count is proportional to the square of the observation count.
13. WHERE the Active_Representation is `exact_tcrdist` and the compiled `tcrdist_cpp` binaries are unavailable, THE CoNGA_Preprocessor SHALL compute neighbors using the Python `TcrDistCalculator` route and SHALL log at `logging.WARNING` level that the run is using the Python route, naming the observation count and the compilation step that enables the faster route.
14. IF the Active_Representation is `exact_tcrdist` and the TCR two-dimensional projection or the TCR clusters must be computed and the compiled `tcrdist_cpp` binaries are unavailable, THEN THE Analysis_CLI SHALL exit with a nonzero status and a message naming the missing binary and the compilation step, because `conga.preprocess.calc_tcrdist_nbrs_umap_clusters_cpp` has no Python route.

**Analysis_CLI flags**

15. THE Analysis_CLI SHALL treat its existing `--no_kpca` flag as the request for the Exact_Nbr_Path override, retaining its current behavior of also selecting exact TCRdist neighbors, two-dimensional projection, and clusters.
16. THE Analysis_CLI SHALL accept a `--use_kpca_tcrdist` flag that requests the KernelPCA override and a `--kpca_reduction_limit` flag that sets the KPCA_Reduction_Limit.
17. THE Analysis_CLI SHALL retain its existing `--use_exact_tcrdist_nbrs`, `--use_tcrdist_umap`, `--use_tcrdist_clusters`, `--kpca_file`, `--rerun_kpca`, `--kpca_kernel`, `--kpca_gaussian_kernel_sdev`, and `--kpca_default_kernel_Dmax` flags with their current meanings, where the `--kpca_*` flags govern how a KPCA_Representation is obtained rather than whether one is used.
18. THE Analysis_CLI SHALL accept flags that set `aa_mds_dim`, `num_pos_cdr3`, `cdr3_weight`, and `random_seed`.
19. IF `--use_kpca_tcrdist` is supplied together with `--no_kpca` or with `--use_exact_tcrdist_nbrs`, THEN THE Analysis_CLI SHALL exit with a nonzero status and a message naming both conflicting flags.
20. IF `--use_kpca_tcrdist` is supplied together with any flag from criterion 18, THEN THE Analysis_CLI SHALL exit with a nonzero status and a message naming both conflicting flags, so that a vectorized Encoding_Config is never silently discarded.
21. IF `--no_kpca` or `--use_exact_tcrdist_nbrs` is supplied together with any flag from criterion 18, THEN THE Analysis_CLI SHALL exit with a nonzero status and a message naming both conflicting flags.
22. IF a flag from criterion 18 is supplied for an organism that is not a Supported_Organism, THEN THE Analysis_CLI SHALL exit with a nonzero status and a message naming the organism and the supplied flag.
23. IF `--use_kpca_tcrdist` is supplied WHILE the observation count is at or above KPCA_Reduction_Limit, THEN THE Analysis_CLI SHALL exit with a nonzero status and a message naming the observation count, the KPCA_Reduction_Limit value, the `--kpca_reduction_limit` flag that raises the limit, and `--no_kpca` as the path that needs no reduction.

**Recorded run statistics**

24. WHERE the Active_Representation is `X_vec_tcr`, THE Analysis_CLI SHALL record the Encoding_Config values and the KPCA_Reduction_Limit in the run's `conga_stats` entries.
25. WHERE the Active_Representation is `X_pca_tcr`, THE Analysis_CLI SHALL record the observation count and the KPCA_Reduction_Limit in the run's `conga_stats` entries.
26. WHERE the Active_Representation is `exact_tcrdist`, THE Analysis_CLI SHALL record the observation count, the KPCA_Reduction_Limit, whether the compiled or the Python route computed the distances, and whether the path was selected automatically or by override, in the run's `conga_stats` entries.

**Restart from a saved `.h5ad`**

27. WHEN the Analysis_CLI restarts from an `.h5ad` file that holds a KPCA_Representation and no Vectorized_Representation, and no override is requested, THE CoNGA_Preprocessor SHALL set the Active_Representation to `X_pca_tcr` and SHALL leave the KernelPCA computation unperformed, so that an older analysis reproduces without recomputation.
28. IF the Analysis_CLI restarts from an `.h5ad` file that holds a KPCA_Representation whose row count is at or above KPCA_Reduction_Limit, THEN THE CoNGA_Preprocessor SHALL reuse the stored array and SHALL log at `logging.WARNING` level that the stored representation exceeds the current KPCA_Reduction_Limit.
29. WHERE no override is requested, WHEN the Analysis_CLI restarts from an `.h5ad` file that holds both a Vectorized_Representation and a KPCA_Representation, THE CoNGA_Preprocessor SHALL set the Active_Representation to `X_vec_tcr`.
30. WHEN the Analysis_CLI restarts from an `.h5ad` file that holds neither a Vectorized_Representation nor a KPCA_Representation, THE CoNGA_Preprocessor SHALL apply the selection table to the restored observation count and organism.
31. WHERE the Exact_Nbr_Path override is requested, WHEN the Analysis_CLI restarts from an `.h5ad` file holding either stored representation, THE CoNGA_Preprocessor SHALL set the Active_Representation to `exact_tcrdist` and SHALL leave both stored arrays unchanged.

**Setup_CLI**

32. WHERE the organism is a Supported_Organism and the KernelPCA override is not requested, THE Setup_CLI SHALL leave the Exact_TCRdist matrix and the KernelPCA files uncomputed at every clonotype count, and SHALL emit a message naming the Vectorized_Representation as the representation the Analysis_CLI will build, so that setup does not pay for an O(N²) matrix that the default analysis path discards.
33. WHERE the organism is a Supported_Organism and the KernelPCA override is requested, WHILE the clonotype count is below KPCA_Reduction_Limit, THE Setup_CLI SHALL compute the Exact_TCRdist matrix and the KernelPCA files.
34. IF the KernelPCA override is requested for the Setup_CLI WHILE the clonotype count is at or above KPCA_Reduction_Limit, THEN THE Setup_CLI SHALL exit with a nonzero status and a message naming the clonotype count, the KPCA_Reduction_Limit value, and the `--kpca_reduction_limit` flag that raises the limit.
35. WHERE the organism is not a Supported_Organism, WHILE the clonotype count is below KPCA_Reduction_Limit, THE Setup_CLI SHALL retain its existing behavior of computing the Exact_TCRdist matrix and the KernelPCA files.
36. WHERE the organism is not a Supported_Organism, WHILE the clonotype count is at or above KPCA_Reduction_Limit, THE Setup_CLI SHALL leave the Exact_TCRdist matrix and the KernelPCA files uncomputed and SHALL emit a message naming the Exact_Nbr_Path as the path the Analysis_CLI will use.
37. THE Setup_CLI SHALL accept the same `--kpca_reduction_limit` flag as the Analysis_CLI, with the same default, and SHALL accept a `--use_kpca_tcrdist` flag that requests the KernelPCA override.
38. THE Setup_CLI SHALL retain its existing `--no_kpca` and `--no_tcrdists` flags, which already skip the Exact_TCRdist matrix and the KernelPCA files.

### Requirement 9: Documentation and code standards

**User Story:** As a contributor, I want the new module to match project conventions, so that it is reviewable and maintainable.

#### Acceptance Criteria

1. THE TCR_Vectorizer SHALL provide a NumPy-style docstring for every public function, stating parameters, return values, and units or ranges where applicable.
2. THE TCR_Vectorizer SHALL provide type hints on every public function signature.
3. THE TCR_Vectorizer SHALL emit progress and diagnostic messages through a module-level `logging` logger.
4. THE TCR_Vectorizer SHALL document in its module docstring that Euclidean distance in the encoded space approximates TCRdist rather than reproducing TCRdist exactly, and SHALL cite the measured accuracy figures from Requirement 6.
5. THE conga package SHALL expose the TCR_Vectorizer public API through an import path documented in the README.
6. WHEN the refactored TCR_Vectorizer passes its accuracy tests, THE repository SHALL no longer contain `conga/tcrdist_vectorizing_functions_for_sharing.py`.
7. THE README SHALL describe all three TCR neighbor paths, SHALL state that the Vectorized_Representation is the default for Supported_Organisms, SHALL state that the KernelPCA reduction is not applied at or above KPCA_Reduction_Limit, and SHALL state how to request the KernelPCA override and the Exact_Nbr_Path override.
8. THE README SHALL state that the Exact_Nbr_Path needs the compiled `tcrdist_cpp` binaries for practical runtimes and for TCR projection and clustering.

### Requirement 10: Test coverage

**User Story:** As a maintainer, I want automated tests for the encoder, so that refactors do not silently change distances.

#### Acceptance Criteria

1. THE repository SHALL contain pytest tests for the CDR3 trimming and gapping function, the sequence encoding function, and the paired-chain encoding function.
2. THE repository SHALL contain pytest tests covering each error condition named in Requirement 3, Requirement 4, and Requirement 8.
3. THE repository SHALL contain a pytest test that encodes a fixed clonotype set twice with the same Encoding_Config and asserts elementwise equality.
4. THE repository SHALL contain a pytest test that produces an Accuracy_Report on a bundled clonotype set and asserts the thresholds named in Requirement 6.
5. THE repository SHALL contain a pytest fixture supplying a small paired clonotype set with valid V genes and CDR3s for at least one Supported_Organism.
6. THE repository SHALL contain a pytest test for every row of the Requirement 8 selection table, asserting the resulting Active_Representation value or the raised error.
7. THE repository SHALL contain pytest tests for each restart case named in Requirement 8 criteria 27 through 31.
8. THE repository SHALL contain a pytest test asserting that the Exact_Nbr_Path records the Active_Representation value `exact_tcrdist` and creates no `adata.obsm` entry for it.
9. THE repository SHALL contain a pytest test that imports the conga package in a subprocess whose working directory lies outside the repository and whose environment is scrubbed to the installed package and its declared dependencies, and asserts that the import succeeds, so that Requirement 1 criteria 1 through 3 are checked in a process that cannot rely on any path or file present only on the developer's own machine.

## Out of Scope

The following are separate roadmap objectives and are deliberately excluded from this feature:

- Vectorized encoding for gamma-delta organisms (`human_gd`, `mouse_gd`, `rhesus_gd`) and Ig organisms (`human_ig`, `mouse_ig`). These receptor types use the KPCA_Representation below KPCA_Reduction_Limit and the Exact_Nbr_Path at or above it, so they remain analyzable at every dataset size. Extending the vectorizer to them is deferred to a later feature.
- A Python route for TCR two-dimensional projection and clustering from exact TCRdist neighbors. `conga.preprocess.calc_tcrdist_nbrs_umap_clusters_cpp` remains C++ only; this feature documents that prerequisite rather than removing it.
- FAISS CPU/GPU neighbor search backends and the tiered `faiss-gpu` → `faiss-cpu` → `sklearn` fallback. This feature only guarantees that the Vector_Matrix is a dense `float32` C-contiguous array suitable for such a backend.
- Batch integration (Harmony, scVI, Combat, BBKNN) and the `hvg_batch_key` parameter.
- Containerization, MuData support, and the new-species reference builder.
- Removing the Exact_TCRdist code path. Exact TCRdist remains in use for the Exact_Nbr_Path, for the KernelPCA override below KPCA_Reduction_Limit, and for accuracy validation.

## Resolved Decisions

- **Three paths, not two**: Raw Exact_TCRdist without KernelPCA is a first-class path. It does not materialize a dense N×N array in either its Python or its C++ implementation, so it is viable at every observation count. See the Introduction and Requirement 8.
- **Organism coverage**: Vectorized encoding covers alpha-beta only (`human`, `mouse`, `rhesus`). Gamma-delta and Ig use the KPCA_Representation below the limit and the Exact_Nbr_Path at or above it. See Requirement 3 and Requirement 8.
- **No size-based rejection**: The earlier hard error for unsupported organisms above the limit is removed. Every organism has a path at every size. See Requirement 8.7.
- **Accuracy gates**: Spearman ≥ 0.95 and mean recall@k ≥ 0.80, both gating. See Requirement 6.6.
- **Limit semantics and name**: `KPCA_Size_Limit` is renamed `KPCA_Reduction_Limit`, defined as the observation count at or above which the KernelPCA reduction is not applied. It no longer gates whether CoNGA can run. See Requirement 8.1 through 8.4.
- **Raw-exact selectable for alpha-beta**: Confirmed with the user. `--no_kpca` explicitly selects the Exact_Nbr_Path for any organism at any size, giving a reference-quality comparison against the vectorized encoding. See Requirement 8.8 and 8.15.
- **Flag reuse over duplication**: `--no_kpca` already selects the raw-exact path in `run_conga.py`, `setup_10x_for_conga.py`, and `merge_samples.py`, and is reused rather than duplicated. `--use_exact_tcrdist_nbrs`, `--use_tcrdist_umap`, `--use_tcrdist_clusters`, `--kpca_file`, `--rerun_kpca`, and the `--kpca_kernel` family keep their current meanings. The only new flags are `--use_kpca_tcrdist`, `--kpca_reduction_limit`, and the four Encoding_Config flags. No flag is deprecated. See Requirement 8.15 through 8.18.
- **Setup_CLI no longer computes discarded artifacts**: For Supported_Organisms without the KernelPCA override, the Setup_CLI skips the Exact_TCRdist matrix and the KernelPCA files at every size, since the default analysis path would ignore them. See Requirement 8.32.
- **Active_Representation has three states**: `X_vec_tcr`, `X_pca_tcr`, or the sentinel `exact_tcrdist`. The Exact_Nbr_Path stores no `adata.obsm` array, and no placeholder entry is fabricated for it. See Requirement 7.7 and 7.8.
- **AnnData key**: `X_vec_tcr`. See Requirement 7.2.
- **Prototype file**: `conga/tcrdist_vectorizing_functions_for_sharing.py` is deleted once the refactored module passes its accuracy tests. See Requirement 9.6.
- **Requirement 1 scope**: Requirement 1 covers portable module initialization only. Shared Gene_Database sourcing and the missing-records error moved to Requirement 3, and the module-level logger convention moved to Requirement 9, so that each requirement title names what its criteria actually constrain. No obligation was dropped. See Requirement 1, Requirement 3.1, Requirement 3.4, and Requirement 9.3.

## Open Questions

None outstanding.
