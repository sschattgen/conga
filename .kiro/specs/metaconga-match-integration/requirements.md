# Requirements Document

## Introduction

`conga-dev` lacks the metaconga-match analysis capability that exists in an external branch, cloned locally at `/Users/sschattg/conga_mc_match` (not reachable via git from this repository). That branch's `conga/metaconga_match.py` (1331 lines) implements two independent, opt-in analysis pipelines that match a user's clonotypes against pretrained reference signatures and a curated literature database:

1. **CDR3aa-bias-cluster matching** (`find_aacluster_matches` / `plot_aacluster_matches`): scores clonotypes' GEX and TCR features against pretrained cluster-enrichment signatures loaded at module import time from bundled TSV files, using hypergeometric tests to assess GEX/TCR overlap significance and DEG (differentially expressed gene) analysis to characterize matched clusters.
2. **TCR clump matching** (`find_clump_matches` / `plot_clump_matches`): uses exact TCRdist (via the existing C++ backend, through `conga.tcr_clumping.find_significant_tcrdist_matches`) to match query TCRs against a curated database of literature-derived TCR specificity clusters annotated with CMV/HLA associations.

This feature ports `conga/metaconga_match.py` and its 11 reference data files from the external branch into `conga-dev`, close to verbatim, and wires two new standalone CLI flags into `scripts/run_conga.py`. It is not a reimplementation: every non-CLI dependency the module calls (`preprocess.calc_nbrs`, `preprocess.add_mait_info_to_adata_obs`, `tcr_clumping.find_significant_tcrdist_matches`, `util.path_to_data`) is already present in `conga-dev` with a compatible signature, confirmed by direct inspection of both codebases this session.

**Why this is not a from-scratch build.** The module's imports (`from .tags import *`, `from . import preprocess`, `from . import tcr_scoring`, `from . import util`, `from . import correlations`, `from . import plotting`, `from . import tcr_clumping`) and its calls into those modules — `preprocess.calc_nbrs(adata, [nbr_frac], obsm_tag_gex='X_pca_gex', obsm_tag_tcr=None)` inside `find_aacluster_matches`, `add_mait_info_to_adata_obs(adata)` (imported locally inside `find_aacluster_matches`), and `tcr_clumping.find_significant_tcrdist_matches(...)` inside `find_clump_matches` — already match `conga-dev`'s current function signatures. The adaptation work is therefore limited to CLI wiring in `scripts/run_conga.py`, not to the ported module's internals.

**Divergence from source-branch CLI behavior.** The source branch wires `--match_metaconga_aaclusters` and `--match_metaconga_clumps` into its own `run_conga.py` with two behaviors this feature deliberately tightens for `conga-dev`:
- The source branch only prints a `WARNING!!!`-banner (repeated 50 times) when `--match_metaconga_aaclusters` is used without the matching `--subset_to_CD4_cells`/`--subset_to_CD8_cells` flag, but does not block execution. This feature makes that pairing a hard, fail-fast requirement instead.
- The source branch's existing (unrelated) `--match_to_tcr_database` organism gate is a silent conditional skip deep in the execution flow (`if (args.match_to_tcr_database and (args.tcr_database_tsvfile or adata.uns['organism'] == 'human')):`), evaluated only after `adata` is constructed. This feature's two new flags instead use an explicit, fail-fast `sys.exit` in the early argument-validation block, checking `args.organism` directly (available immediately after `argparse`, before `adata` exists). This is a confirmed, deliberate deviation from the `--match_to_tcr_database` precedent, not an inconsistency to resolve toward matching it.

Neither new flag is added to `conga-dev`'s `--all` bundle (`all_modes` list in `scripts/run_conga.py`); the source branch never did this either, and both remain standalone opt-in flags.

**Scope boundaries.** Two optimization ideas were discussed and explicitly scoped out of hard requirements for this feature:
- Vectorizing the per-row Python loop in `_encode_tcr_seqs` (`for ii, l in enumerate(tcr_df.itertuples()):`) into native NumPy operations is a real, low-risk opportunity but is captured only as an optional task, not a requirement.
- Replacing the TCR clump-matching path's exact-TCRdist database search with an approximate vectorized-TCRdist + FAISS nearest-neighbor search is a bigger design tradeoff and is entirely out of scope for this feature — not even an optional task — and is a candidate for a future, separate spec.

**Testing emphasis.** Per explicit user direction, this feature's testing requirements emphasize integration-level correctness — CLI flag parsing and `choices` validation, the new `sys.exit` validation paths, data-file-presence checks, successful import of the ported module, and end-to-end invocability of both analysis functions on a representative synthetic human `AnnData` fixture without crashing — over deep correctness testing of the underlying statistical algorithms (hypergeometric tests, DEG scoring, TCRdist background sampling), which is explicitly deferred to future work.

## Glossary

- **Metaconga_Module**: The ported file `conga/metaconga_match.py`, containing the AACluster_Pipeline and the Clump_Pipeline, plus their supporting helpers (`get_categorical_colors`, `_encode_tcr_seqs`, `get_cdr3aa_bias_tcr_scores`, `get_cdr3aa_bias_deg_scores`, `_find_degs_for_subset`, `reduce_to_single_aacluster_match_per_clonotype`, `_process_clump_matches`).
- **Source_Branch**: The external, locally-cloned branch at `/Users/sschattg/conga_mc_match`, not reachable via git from `conga-dev`. The origin of the Metaconga_Module, its 11 Metaconga_Data_Files, the 5 Metaconga_Tags, and the reference CLI wiring this feature adapts.
- **Metaconga_Data_Files**: The 11 TSV files copied verbatim from `/Users/sschattg/conga_mc_match/conga/data/metaconga/` into `conga/data/metaconga/`: `big_combo_tcrs_2024-02-02a_gp4_MCC10_NG200_groups_info_MCC10_extras.tsv`, `big_combo_tcrs_2024-02-02a_gp4_MCC10_NG200_groups_info_MCC10_extras_new_matches.tsv`, `big_combo_tcrs_2024-02-02a_gp4_ten_tcrs.tsv`, `cdr3aa_bias_cluster_names.tsv`, `good_clumps_v1.tsv`, `hsgenes_1000_plus_cdr3aa_bias_top30_degs.tsv`, `hsgenes_200_plus_cdr3aa_bias_top30_degs.tsv`, `round8_v5cd4_run84_xribo_200_process_v1_leiden2_cdr3aa_sig_1e-06_cluster_feature_enrichments.tsv`, `round8_v5cd8_run84_xribo_200_process_v1_leiden2_cdr3aa_sig_1e-06_cluster_feature_enrichments.tsv`, `run105_cd4_deg_results.tsv`, `run106_cd8_deg_results.tsv`.
- **Metaconga_Tags**: The 5 string constants ported verbatim into `conga/tags.py`: `METACONGA_MATCH_CLUMPS = 'metaconga_match_clumps'` and `METACONGA_MATCH_AACLUSTERS = 'metaconga_match_aaclusters'` (table tags, grouped with the existing table-tag constants); `METACONGA_MATCH_AACLUSTERS_BARS = 'metaconga_match_aaclusters_bars'`, `METACONGA_MATCH_AACLUSTERS_UMAPS = 'metaconga_match_aaclusters_umaps'`, and `METACONGA_MATCH_CLUMPS_UMAPS = 'metaconga_match_clumps_umaps'` (figure tags, grouped with the existing figure-tag constants).
- **AACluster_Pipeline**: The CDR3aa-bias-cluster matching analysis, consisting of `conga.metaconga_match.find_aacluster_matches(adata, cd48)` followed by `conga.metaconga_match.plot_aacluster_matches(adata, matches, outfile_prefix)`, where `cd48` is the literal string `'cd4'` or `'cd8'`.
- **Clump_Pipeline**: The TCR clump-matching analysis, consisting of `conga.metaconga_match.find_clump_matches(adata)` followed by `conga.metaconga_match.plot_clump_matches(adata, outfile_prefix)`.
- **Analysis_CLI**: The command-line entry point `scripts/run_conga.py`, as in the `tcrdist-db-update` spec.
- **AACluster_Flag**: The new CLI argument `--match_metaconga_aaclusters`, with `choices=['cd4', 'cd8', 'CD4', 'CD8', None]` and `default=None`, lowercased to `'cd4'` or `'cd8'` immediately after parsing when not `None` (matching Source_Branch's `args.match_metaconga_aaclusters = args.match_metaconga_aaclusters.lower()`).
- **Clump_Flag**: The new CLI argument `--match_metaconga_clumps`, declared with `action='store_true'` (default `False`).
- **Fixed_HVG_Pathway**: The existing `conga-dev` code path triggered by `--force_variable_genes`, as named in the mutual-exclusion error already present in `scripts/run_conga.py` (`if args.force_variable_genes and (args.batch_key or args.batch_integration_method): sys.exit('ERROR: --force_variable_genes (Fixed_HVG_Pathway) and --batch_key/--batch_integration_method (Full_Integration_Pathway) are mutually exclusive')`).
- **Full_Integration_Pathway**: The existing `conga-dev` code path triggered by supplying both `--batch_key` and `--batch_integration_method` together, as named in that same existing mutual-exclusion error.
- **Bundled_AACluster_HVG_File**: The specific Metaconga_Data_File `conga/data/metaconga/hsgenes_1000_plus_cdr3aa_bias_top30_degs.tsv`, used as the auto-injected value of `--force_variable_genes` when the AACluster_Flag is set and the user did not already supply `--force_variable_genes`.
- **Auto_Injection_Behavior**: The confirmed ("option A") behavior that when the AACluster_Flag is set (not `None`) and the user did not supply `--force_variable_genes` themselves, the Analysis_CLI sets `args.force_variable_genes` to the Bundled_AACluster_HVG_File and prints a `WARNING:`-prefixed message naming the auto-injected file, matching Source_Branch's `print('WARNING: --match_metaconga_aaclusters', 'adding --force_variable_genes', variable_genes_file)` behavior.
- **CD_Subset_Pairing_Rule**: The hard requirement that `--match_metaconga_aaclusters cd4` (case-insensitive) must be accompanied by `--subset_to_CD4_cells`, and `--match_metaconga_aaclusters cd8` (case-insensitive) must be accompanied by `--subset_to_CD8_cells`. Unlike Source_Branch's softer behavior (a scary-but-non-blocking warning banner), this feature enforces the rule via `sys.exit`.
- **Metaconga_Organism_Gate**: The fail-fast validation, applied independently to the AACluster_Flag and the Clump_Flag, that each requires `args.organism == 'human'`, implemented as an explicit `sys.exit('ERROR: ...')` in the early cross-flag validation block (immediately after the existing `--force_variable_genes`/batch-integration mutual-exclusion check), evaluated directly against `args.organism` before `adata` is constructed. This is a deliberate deviation from the pre-existing `--match_to_tcr_database` organism gate (a silent conditional skip evaluated against `adata.uns['organism']` deep in the execution flow), confirmed by the user as intentional and not to be reconciled with that precedent.
- **Analysis_Dispatch_Section**: The section of `scripts/run_conga.py` (consistent in location and style with the existing `--match_to_tcr_database` and `--tcr_clumping` blocks) where each analysis mode's flag is checked and its corresponding function(s) are invoked against the constructed `adata`.
- **Representative_Human_Fixture**: A synthetic or minimal `AnnData` object, constructed for integration testing, with `adata.uns['organism'] == 'human'` and the GEX/TCR fields (e.g. `X_pca_gex`, clonotype and CDR3 columns) that the AACluster_Pipeline and Clump_Pipeline read, sufficient to exercise both pipelines end-to-end without requiring statistically meaningful output.

## Requirements

### Requirement 1: Port the metaconga_match module

**User Story:** As a CoNGA maintainer, I want `conga/metaconga_match.py` available in `conga-dev`, so that the CDR3aa-bias-cluster matching and TCR clump-matching analyses can be run against this codebase's data model.

#### Acceptance Criteria

1. THE conga package SHALL contain a file `conga/metaconga_match.py`, ported from `/Users/sschattg/conga_mc_match/conga/metaconga_match.py`, preserving its public functions `find_aacluster_matches`, `plot_aacluster_matches`, `find_clump_matches`, and `plot_clump_matches`, and its supporting helpers, with no behavioral changes beyond what is required for compatibility with `conga-dev`'s current `preprocess.py`, `util.py`, and `tcr_clumping.py` APIs.
2. THE Metaconga_Module SHALL import `preprocess`, `tcr_scoring`, `util`, `correlations`, `plotting`, and `tcr_clumping` from the `conga` package using the same relative-import style already used elsewhere in `conga-dev`'s modules.
3. THE Metaconga_Module's calls to `preprocess.calc_nbrs`, `preprocess.add_mait_info_to_adata_obs`, and `tcr_clumping.find_significant_tcrdist_matches` SHALL use these functions' existing `conga-dev` signatures without requiring changes to `preprocess.py` or `tcr_clumping.py`.
4. WHEN `import conga.metaconga_match` is executed in the `conga-dev` codebase, THE import SHALL succeed without raising an exception, given that the Metaconga_Data_Files (Requirement 2) are present at the paths the module expects under `util.path_to_data / 'metaconga'`.
5. THE conga package SHALL NOT add `conga.metaconga_match` to any automatic import bundle (e.g. `conga/__init__.py`'s top-level imports) beyond what is needed for `scripts/run_conga.py` to call it directly, consistent with the Source_Branch's own treatment of the module as a standalone, opt-in component.

### Requirement 2: Bundle the metaconga reference data files

**User Story:** As a CoNGA maintainer, I want the 11 reference data files the Metaconga_Module depends on bundled into `conga-dev`, so that the ported module's module-level data loading succeeds.

#### Acceptance Criteria

1. THE conga package SHALL contain a directory `conga/data/metaconga/` populated with all 11 Metaconga_Data_Files, copied verbatim (byte-identical) from `/Users/sschattg/conga_mc_match/conga/data/metaconga/`.
2. THE conga package SHALL verify the presence of each of the 11 Metaconga_Data_Files by filename under `conga/data/metaconga/`; file-presence verification alone is sufficient acceptance evidence for this requirement, and no checksum or row-count verification is required.
3. THE conga package SHALL NOT rename, reformat, or otherwise modify the contents of any Metaconga_Data_File during the port.

### Requirement 3: Port the metaconga tags

**User Story:** As a CoNGA maintainer, I want the metaconga-specific result and figure tags available in `conga/tags.py`, so that the ported module's `from .tags import *` import resolves and results are stored under the same keys the Source_Branch uses.

#### Acceptance Criteria

1. THE conga package SHALL contain all 5 Metaconga_Tags in `conga/tags.py`, with identical names and string values to the Source_Branch's `conga/tags.py`: `METACONGA_MATCH_CLUMPS = 'metaconga_match_clumps'`, `METACONGA_MATCH_AACLUSTERS = 'metaconga_match_aaclusters'`, `METACONGA_MATCH_AACLUSTERS_BARS = 'metaconga_match_aaclusters_bars'`, `METACONGA_MATCH_AACLUSTERS_UMAPS = 'metaconga_match_aaclusters_umaps'`, and `METACONGA_MATCH_CLUMPS_UMAPS = 'metaconga_match_clumps_umaps'`.
2. THE conga package SHALL place `METACONGA_MATCH_CLUMPS` and `METACONGA_MATCH_AACLUSTERS` among the existing table-tag constants in `conga/tags.py`, and `METACONGA_MATCH_AACLUSTERS_BARS`, `METACONGA_MATCH_AACLUSTERS_UMAPS`, and `METACONGA_MATCH_CLUMPS_UMAPS` among the existing figure-tag constants, matching the grouping convention already used in that file.
3. THE conga package SHALL NOT port any tag from the Source_Branch's `tags.py` other than the 5 Metaconga_Tags as part of this feature.

### Requirement 4: New CLI flags

**User Story:** As a CoNGA user, I want to invoke the AACluster_Pipeline or the Clump_Pipeline from `scripts/run_conga.py` via command-line flags, so that I can run these analyses without writing a custom script.

#### Acceptance Criteria

1. THE Analysis_CLI SHALL define the AACluster_Flag as `parser.add_argument('--match_metaconga_aaclusters', choices=['cd4', 'cd8', 'CD4', 'CD8', None], default=None)`.
2. THE Analysis_CLI SHALL define the Clump_Flag as `parser.add_argument('--match_metaconga_clumps', action='store_true')`.
3. WHEN the AACluster_Flag is supplied with a value other than one of `'cd4'`, `'cd8'`, `'CD4'`, `'CD8'`, THEN THE Analysis_CLI's argument parser SHALL reject the invocation before any analysis code runs, consistent with standard `argparse` `choices` behavior.
4. WHEN the AACluster_Flag is supplied with any of `'CD4'`, `'CD8'`, THEN THE Analysis_CLI SHALL lowercase `args.match_metaconga_aaclusters` to `'cd4'` or `'cd8'` respectively immediately after parsing, before any downstream validation or dispatch logic reads it.
5. THE Analysis_CLI SHALL NOT add either the AACluster_Flag or the Clump_Flag to the `all_modes` list consulted when `--all` is supplied, consistent with the Source_Branch never having wired these flags into its own `--all` bundle.

### Requirement 5: Auto-injection of the bundled HVG file, and its mutual-exclusion error with batch integration

**User Story:** As a CoNGA user running the AACluster_Pipeline, I want the correct bundled gene list used as my highly-variable-gene set by default, so that I don't have to know about or locate that file myself, while still being blocked clearly if I've also asked for batch integration.

#### Acceptance Criteria

1. WHEN the AACluster_Flag is set (not `None`) AND the user did not supply `--force_variable_genes`, THEN THE Analysis_CLI SHALL set `args.force_variable_genes` to the path of the Bundled_AACluster_HVG_File (`conga/data/metaconga/hsgenes_1000_plus_cdr3aa_bias_top30_degs.tsv`, resolved via `util.path_to_data`), matching Source_Branch's behavior.
2. WHEN the Auto_Injection_Behavior in criterion 1 occurs, THE Analysis_CLI SHALL print a message prefixed with `WARNING:` naming the auto-injected file path, matching Source_Branch's `print('WARNING: --match_metaconga_aaclusters', 'adding --force_variable_genes', variable_genes_file)` behavior.
3. WHEN the AACluster_Flag is set (not `None`) AND the user explicitly supplied `--force_variable_genes` with a value, THEN THE Analysis_CLI SHALL NOT override the user-supplied value, and SHALL NOT print the Auto_Injection_Behavior warning.
4. IF the AACluster_Flag is set (not `None`) AND the user supplied `--batch_key` or `--batch_integration_method`, THEN THE Analysis_CLI SHALL raise the same mutual-exclusion error already raised for `--force_variable_genes` and batch integration (Fixed_HVG_Pathway vs. Full_Integration_Pathway) via `sys.exit`, with the error message additionally naming `--match_metaconga_aaclusters` as a trigger of the Fixed_HVG_Pathway, so the user understands why the AACluster_Flag alone (without `--force_variable_genes` ever appearing on their command line) produced this error.
5. THE check in criterion 4 SHALL be evaluated in the same early cross-flag validation block as the existing `--force_variable_genes`/batch-integration mutual-exclusion check in `scripts/run_conga.py`, so that the Auto_Injection_Behavior (criteria 1-2) never silently proceeds to construct `adata` when batch integration was also requested.
6. THE Analysis_CLI SHALL perform the check in criterion 4 regardless of whether the user supplied `--force_variable_genes` directly or relied on the Auto_Injection_Behavior, so that the mutual-exclusion outcome is identical in both cases.

### Requirement 6: Hard requirement pairing AACluster_Flag with the matching CD-subset flag

**User Story:** As a CoNGA user running the AACluster_Pipeline, I want to be stopped with a clear error if I forget to subset my data to the matching CD4 or CD8 population, so that I don't get a silently misleading result from running the wrong pretrained signature against the wrong cell population.

#### Acceptance Criteria

1. IF `args.match_metaconga_aaclusters == 'cd4'` AND `args.subset_to_CD4_cells` is not set, THEN THE Analysis_CLI SHALL raise a clear `sys.exit('ERROR: ...')` naming both `--match_metaconga_aaclusters cd4` and the required `--subset_to_CD4_cells` flag, and SHALL NOT proceed to construct `adata` or run any analysis.
2. IF `args.match_metaconga_aaclusters == 'cd8'` AND `args.subset_to_CD8_cells` is not set, THEN THE Analysis_CLI SHALL raise a clear `sys.exit('ERROR: ...')` naming both `--match_metaconga_aaclusters cd8` and the required `--subset_to_CD8_cells` flag, and SHALL NOT proceed to construct `adata` or run any analysis.
3. IF `args.match_metaconga_aaclusters == 'cd4'` AND `args.subset_to_CD4_cells` is set, THEN THE Analysis_CLI SHALL proceed without raising an error under this requirement (independent of other validation requirements).
4. IF `args.match_metaconga_aaclusters == 'cd8'` AND `args.subset_to_CD8_cells` is set, THEN THE Analysis_CLI SHALL proceed without raising an error under this requirement (independent of other validation requirements).
5. THE CD_Subset_Pairing_Rule enforcement SHALL replace Source_Branch's softer behavior entirely; THE Analysis_CLI SHALL NOT merely print a warning banner and continue when the pairing is missing or mismatched.
6. THE CD_Subset_Pairing_Rule check SHALL be evaluated after AACluster_Flag lowercasing (Requirement 4, criterion 4) and in the same early cross-flag validation phase as the checks in Requirements 5 and 7, before `adata` is constructed.

### Requirement 7: Organism restriction for both new flags

**User Story:** As a CoNGA user running CoNGA on a non-human organism, I want to be stopped immediately with a clear error if I request a metaconga analysis, so that I don't waste time running a pipeline that only has human reference data.

#### Acceptance Criteria

1. IF the AACluster_Flag is set (not `None`) AND `args.organism != 'human'`, THEN THE Analysis_CLI SHALL raise a clear `sys.exit('ERROR: ...')` stating that `--match_metaconga_aaclusters` requires `--organism human`, and SHALL NOT proceed to construct `adata` or run any analysis.
2. IF the Clump_Flag is set AND `args.organism != 'human'`, THEN THE Analysis_CLI SHALL raise a clear `sys.exit('ERROR: ...')` stating that `--match_metaconga_clumps` requires `--organism human`, and SHALL NOT proceed to construct `adata` or run any analysis.
3. THE Metaconga_Organism_Gate SHALL check `args.organism` directly, evaluated immediately after `argparse` parsing completes and before `adata` is constructed; it SHALL NOT wait for or depend on `adata.uns['organism']`.
4. THE Metaconga_Organism_Gate SHALL be placed in `scripts/run_conga.py`'s early cross-flag validation block, immediately after the existing `--force_variable_genes`/batch-integration mutual-exclusion check (Requirement 5).
5. THE Metaconga_Organism_Gate's fail-fast `sys.exit` behavior SHALL be implemented independently of the pre-existing `--match_to_tcr_database` organism conditional (which silently skips deep in the execution flow by checking `adata.uns['organism']`); THE conga package SHALL NOT modify the `--match_to_tcr_database` gate's existing behavior as part of this feature.

### Requirement 8: Analysis dispatch

**User Story:** As a CoNGA user who has set the AACluster_Flag or the Clump_Flag and passed all validation, I want the corresponding analysis functions actually invoked against my data, so that I get the matching results and plots as output.

#### Acceptance Criteria

1. WHEN `args.match_metaconga_aaclusters is not None` and execution reaches the Analysis_Dispatch_Section, THE Analysis_CLI SHALL call `matches = conga.metaconga_match.find_aacluster_matches(adata, cd48)` where `cd48` is `args.match_metaconga_aaclusters` (guaranteed to be `'cd4'` or `'cd8'` by this point), followed by `conga.metaconga_match.plot_aacluster_matches(adata, matches, args.outfile_prefix)`.
2. WHEN `args.match_metaconga_clumps` is `True` and execution reaches the Analysis_Dispatch_Section, THE Analysis_CLI SHALL call `conga.metaconga_match.find_clump_matches(adata)` followed by `conga.metaconga_match.plot_clump_matches(adata, args.outfile_prefix)`.
3. THE Analysis_Dispatch_Section's AACluster_Pipeline and Clump_Pipeline blocks SHALL be placed in the same part of `scripts/run_conga.py` where other analysis-mode dispatch blocks already live (consistent in location and style with the existing `--match_to_tcr_database` and `--tcr_clumping` blocks), i.e. after `adata` has been fully constructed and preprocessed.
4. THE Analysis_CLI SHALL allow both the AACluster_Flag and the Clump_Flag to be set simultaneously in a single invocation, running both pipelines independently, since neither pipeline's validation or dispatch logic depends on the other being unset.

### Requirement 9: Integration-level test coverage

**User Story:** As a CoNGA maintainer, I want automated tests that confirm the new CLI flags, validation errors, and dispatch wiring work correctly end-to-end, so that regressions in the integration surface are caught, while deep statistical correctness of the underlying algorithms is left to future, more targeted test work.

#### Acceptance Criteria

1. THE conga test suite SHALL include a test confirming that `import conga.metaconga_match` succeeds without raising an exception.
2. THE conga test suite SHALL include a test confirming that each of the 11 Metaconga_Data_Files is present on disk under `conga/data/metaconga/` by filename.
3. THE conga test suite SHALL include tests confirming the Analysis_CLI's `--match_metaconga_aaclusters` argument accepts each of `'cd4'`, `'cd8'`, `'CD4'`, `'CD8'`, and `None` (i.e. the flag omitted), and rejects any other value via `argparse`'s standard `choices` rejection.
4. THE conga test suite SHALL include a test confirming the Metaconga_Organism_Gate's `sys.exit` behavior from Requirement 7 is triggered when the AACluster_Flag or the Clump_Flag is set together with an `--organism` value other than `'human'`.
5. THE conga test suite SHALL include a test confirming the CD_Subset_Pairing_Rule's `sys.exit` behavior from Requirement 6 is triggered when `--match_metaconga_aaclusters cd4` is set without `--subset_to_CD4_cells`, and separately when `--match_metaconga_aaclusters cd8` is set without `--subset_to_CD8_cells`.
6. THE conga test suite SHALL include a test confirming the mutual-exclusion `sys.exit` behavior from Requirement 5, criterion 4 is triggered when the AACluster_Flag is set together with `--batch_key` or `--batch_integration_method`.
7. THE conga test suite SHALL include a test confirming the Auto_Injection_Behavior from Requirement 5, criteria 1-3: that `args.force_variable_genes` is set to the Bundled_AACluster_HVG_File path when the AACluster_Flag is set and `--force_variable_genes` was not supplied, and that a user-supplied `--force_variable_genes` value is left unmodified.
8. THE conga test suite SHALL include a test that invokes `conga.metaconga_match.find_aacluster_matches` and `conga.metaconga_match.plot_aacluster_matches` end-to-end against a Representative_Human_Fixture, asserting that both calls complete without raising an exception; this test is not required to assert on the statistical correctness of the returned matches or scores.
9. THE conga test suite SHALL include a test that invokes `conga.metaconga_match.find_clump_matches` and `conga.metaconga_match.plot_clump_matches` end-to-end against a Representative_Human_Fixture, asserting that both calls complete without raising an exception; this test is not required to assert on the statistical correctness of the returned clump matches.
10. THE conga test suite SHALL NOT be required, as part of this feature, to add tests validating the correctness of the hypergeometric tests, DEG scoring, or TCRdist background-sampling statistics internal to the Metaconga_Module; such tests are explicitly deferred to future work.

### Requirement 10: Deferred and out-of-scope optimizations

**User Story:** As a CoNGA maintainer reviewing this feature's scope, I want the two performance ideas discussed during design clearly separated by priority, so that neither is mistaken for a requirement of this feature.

#### Acceptance Criteria

1. THE task breakdown for this feature SHALL list vectorizing `_encode_tcr_seqs`'s per-row Python loop (`for ii, l in enumerate(tcr_df.itertuples()):`) into native NumPy vectorized operations as an OPTIONAL task, not a required one.
2. THE task breakdown for this feature SHALL NOT include, in any form (required or optional), replacing the Clump_Pipeline's exact-TCRdist-against-fixed-database search with an approximate vectorized-TCRdist + FAISS nearest-neighbor search; this idea is out of scope for this feature entirely and is noted as a candidate for a future, separate spec.
3. THE conga package SHALL port and run the Clump_Pipeline's exact-TCRdist matching exactly as implemented in the Source_Branch (via `tcr_clumping.find_significant_tcrdist_matches`), with no behavioral change to its matching algorithm as part of this feature.
