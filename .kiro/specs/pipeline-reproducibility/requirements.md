# Requirements Document

## Introduction

CoNGA's main analysis pipeline (`scripts/run_conga.py`, via `conga/preprocess.py`)
contains multiple stochastic computation steps -- PCA, UMAP, graph-based
clustering (Leiden/Louvain), and KernelPCA -- that currently do not
consistently receive a seed. `scripts/run_conga.py` already exposes a
`--random_seed` CLI flag (default resolved to `util.DEFAULT_RANDOM_SEED = 42`),
but today that seed is only threaded through to the vectorized TCRdist
amino-acid embedding step. The rest of the stochastic surface in
`conga/preprocess.py` -- `sc.tl.pca`, `sc.pp.neighbors`, `sc.tl.umap`,
`sc.tl.leiden`/`sc.tl.louvain`, `KernelPCA`, and two unseeded raw NumPy
random calls (`np.random.permutation`, `np.random.randn`) -- runs without
any seed wired to `--random_seed`, so two runs of the same pipeline over the
same input can produce different UMAP coordinates and cluster assignments.
There is also no automated test that runs the real `scripts/run_conga.py`
pipeline twice and confirms identical output, which existing tests
(`tests/test_run_conga_cli.py`) explicitly scope out as being out of their
concern.

This feature wires `--random_seed` through every stochastic call in the
pipeline that accepts a `random_state`/seed parameter, and adds an automated
test that runs the real pipeline twice end to end with the same seed over a
small fixture dataset, asserting that the resulting outputs are identical.

## Glossary

- **Pipeline**: The analysis sequence invoked by `scripts/run_conga.py`,
  encompassing preprocessing, dimensionality reduction, neighbor-graph
  construction, UMAP projection, and clustering, as implemented in
  `conga/preprocess.py`.
- **Random_Seed_Flag**: The `--random_seed` command-line argument of
  `scripts/run_conga.py`, resolved to `util.DEFAULT_RANDOM_SEED` (42) when
  not explicitly supplied.
- **Stochastic_Call**: A call to a scanpy, scikit-learn, or NumPy function
  within `conga/preprocess.py` whose output depends on pseudo-random number
  generation, including but not limited to `sc.tl.pca`, `sc.pp.neighbors`,
  `sc.tl.umap`, `sc.tl.leiden`, `sc.tl.louvain`, `sklearn.decomposition.KernelPCA`,
  `np.random.permutation`, and `np.random.randn`.
- **Pipeline_Run**: A single invocation of `scripts/run_conga.py` that
  completes and writes a `<outfile_prefix>_final.h5ad` output file.
- **Reproducibility_Test**: An automated test that invokes the Pipeline
  twice with the same Random_Seed_Flag value over the same input data and
  compares the resulting outputs.
- **Comparable_Output**: A field written into the final AnnData object
  (an `adata.obs` column, `adata.obsm` array, or `adata.uns` entry) that is
  deterministic given a fixed Random_Seed_Flag value and is asserted for
  exact equality by the Reproducibility_Test.

## Requirements

### Requirement 1: Seed propagation to PCA calls

**User Story:** As a CoNGA developer, I want every PCA computation in the
pipeline to use the configured random seed, so that principal component
results are reproducible across runs.

#### Acceptance Criteria

1. WHEN `conga.preprocess.cluster_and_tsne_and_umap` computes `X_pca_gex` via
   `sc.tl.pca`, THE Pipeline SHALL pass the configured Random_Seed_Flag value
   as the `random_state` argument to that call.
2. WHEN `conga.preprocess` computes a representative-cell PCA via `sc.tl.pca`
   (the call near the clone-representative-selection step), THE Pipeline
   SHALL pass the configured Random_Seed_Flag value as the `random_state`
   argument to that call.
3. WHEN `conga.preprocess` constructs a `KernelPCA` instance with a
   precomputed kernel, THE Pipeline SHALL pass the configured Random_Seed_Flag
   value as the `random_state` argument to that `KernelPCA` constructor call.
4. THE `conga.preprocess.cluster_and_tsne_and_umap` function SHALL accept a
   `random_seed` parameter that defaults to `util.DEFAULT_RANDOM_SEED`.

### Requirement 2: Seed propagation to neighbor graph and embedding calls

**User Story:** As a CoNGA developer, I want neighbor-graph construction and
UMAP embedding to use the configured random seed, so that embeddings and
graph structure are reproducible across runs.

#### Acceptance Criteria

1. WHEN `conga.preprocess.cluster_and_tsne_and_umap` calls `sc.pp.neighbors`,
   THE Pipeline SHALL pass the configured Random_Seed_Flag value as the
   `random_state` argument to that call.
2. WHEN `conga.preprocess.cluster_and_tsne_and_umap` calls `sc.tl.umap` for
   the multi-component embedding, THE Pipeline SHALL pass the configured
   Random_Seed_Flag value as the `random_state` argument to that call.
3. WHEN `conga.preprocess.cluster_and_tsne_and_umap` calls `sc.tl.umap` for
   the 1-dimensional embedding, THE Pipeline SHALL pass the configured
   Random_Seed_Flag value as the `random_state` argument to that call.
4. WHEN `conga.preprocess.calc_tcrdist_nbrs_umap_clusters_cpp` calls
   `sc.tl.umap` for either the multi-component or 1-dimensional embedding,
   THE Pipeline SHALL pass the configured Random_Seed_Flag value as the
   `random_state` argument to that call.
5. WHEN `conga.preprocess.calc_tcrdist_nbrs_umap_clusters_cpp` generates the
   placeholder PCA array via `np.random.randn`, THE Pipeline SHALL generate
   that array using a seeded random number generator initialized with the
   configured Random_Seed_Flag value instead of the global NumPy random
   state.
6. THE `conga.preprocess.calc_tcrdist_nbrs_umap_clusters_cpp` function SHALL
   accept a `random_seed` parameter that defaults to `util.DEFAULT_RANDOM_SEED`.

### Requirement 3: Seed propagation to clustering calls

**User Story:** As a CoNGA developer, I want Leiden and Louvain clustering
to use the configured random seed, so that cluster assignments are
reproducible across runs.

#### Acceptance Criteria

1. WHEN `conga.preprocess.cluster_and_tsne_and_umap` calls `sc.tl.leiden`,
   THE Pipeline SHALL pass the configured Random_Seed_Flag value as the
   `random_state` argument to that call.
2. WHEN `conga.preprocess.cluster_and_tsne_and_umap` calls `sc.tl.louvain`,
   THE Pipeline SHALL pass the configured Random_Seed_Flag value as the
   `random_state` argument to that call.
3. WHEN `conga.preprocess.calc_tcrdist_nbrs_umap_clusters_cpp` calls
   `sc.tl.leiden` or `sc.tl.louvain`, THE Pipeline SHALL pass the configured
   Random_Seed_Flag value as the `random_state` argument to that call.

### Requirement 4: Seed propagation to diagnostic sampling

**User Story:** As a CoNGA developer, I want optional diagnostic sampling
within preprocessing to use the configured random seed, so that enabling a
diagnostic comparison does not introduce unseeded randomness into an
otherwise reproducible run.

#### Acceptance Criteria

1. WHEN `conga.preprocess` samples cell indices via `np.random.permutation`
   for the `compare_distance_distributions` diagnostic, THE Pipeline SHALL
   generate that sample using a seeded random number generator initialized
   with the configured Random_Seed_Flag value instead of the global NumPy
   random state.

### Requirement 5: CLI wiring of the random seed into preprocessing

**User Story:** As a CoNGA user, I want the `--random_seed` flag I pass on
the command line to control every stochastic step of the pipeline, so that
re-running the same command with the same seed reproduces my results.

#### Acceptance Criteria

1. WHEN `scripts/run_conga.py` calls `conga.preprocess.cluster_and_tsne_and_umap`,
   THE Pipeline SHALL pass `args.random_seed` as that call's `random_seed`
   argument.
2. WHEN `scripts/run_conga.py` calls
   `conga.preprocess.calc_tcrdist_nbrs_umap_clusters_cpp`, THE Pipeline SHALL
   pass `args.random_seed` as that call's `random_seed` argument.
3. WHERE `args.random_seed` is not supplied on the command line, THE Pipeline
   SHALL resolve it to `util.DEFAULT_RANDOM_SEED` before it reaches any
   Stochastic_Call, consistent with the existing resolution behavior of the
   Random_Seed_Flag.

### Requirement 6: End-to-end pipeline reproducibility test

**User Story:** As a CoNGA developer, I want an automated test that proves
two full pipeline runs with the same seed produce identical output, so that
regressions in seed propagation are caught automatically.

#### Acceptance Criteria

1. THE Reproducibility_Test SHALL invoke `scripts/run_conga.py` as a
   subprocess twice, with identical arguments including the same
   `--random_seed` value and the same input fixture data, writing to two
   distinct `--outfile_prefix` locations.
2. WHEN both Pipeline_Run invocations of the Reproducibility_Test complete,
   THE Reproducibility_Test SHALL load both resulting `_final.h5ad` files
   and assert that every identified Comparable_Output field is exactly equal
   between the two runs.
3. IF either Pipeline_Run invocation of the Reproducibility_Test exits with a
   nonzero return code, THEN THE Reproducibility_Test SHALL fail and SHALL
   report the captured stdout and stderr of the failing invocation.
4. THE Reproducibility_Test SHALL use a fixture dataset small enough for two
   full Pipeline_Run invocations to complete within the test suite's normal
   runtime budget.
5. WHERE a candidate Comparable_Output field is found during implementation
   to vary between two Pipeline_Run invocations using the same
   Random_Seed_Flag value despite the Requirement 1-5 wiring being applied,
   THE Reproducibility_Test SHALL exclude that specific field from its exact
   equality assertions and the implementation SHALL report the field and the
   observed source of nondeterminism rather than weakening the assertion to
   an approximate comparison.

### Requirement 7: Reproducibility across the exact-TCRdist and vectorized paths

**User Story:** As a CoNGA developer, I want pipeline reproducibility
verified for both supported TCR-representation code paths, so that seed
propagation is confirmed regardless of which representation a user selects.

#### Acceptance Criteria

1. THE Reproducibility_Test SHALL cover a Pipeline_Run configuration that
   exercises `conga.preprocess.cluster_and_tsne_and_umap` using a kernel-PCA
   or vectorized TCR representation.
2. THE Reproducibility_Test SHALL cover a Pipeline_Run configuration that
   exercises `conga.preprocess.calc_tcrdist_nbrs_umap_clusters_cpp` via the
   `--use_exact_tcrdist_nbrs` or `--no_kpca` flag.
