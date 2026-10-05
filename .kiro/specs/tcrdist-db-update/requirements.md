# Requirements Document

## Introduction

CoNGA's active tcrdist reference database is stale. `conga/tcrdist/basic.py` hardcodes `db_file = 'combo_xcr_2023-12-30.tsv'`, which the module loads via `conga.tcrdist.all_genes`. A newer file, `conga/tcrdist/db/combo_xcr_2026-08-06.tsv`, already exists in the repository and adds real germline V/J gene data for several new species, plus previously-absent `rhesus_ig` data for an existing species. (`conga/tcrdist/db/combo_xcr.tsv`, with no date suffix, is a separate stale and unused file — it is missing `mouse_ig`/`rhesus_ig` entirely and is not the active database; it is not touched by this feature.)

This feature updates the active database and wires up usable support for a scoped subset of the new species, extends the vectorized TCRdist encoder to attempt coverage of those species subject to an accuracy-validation gate, and fixes two pre-existing hard-crash bugs in organism-string handling that would otherwise block the new species even after the file swap.

**Database contents.** `combo_xcr_2023-12-30.tsv` (the currently active file) contains exactly 8 organism-string values: `human`, `human_gd`, `human_ig`, `mouse`, `mouse_gd`, `mouse_ig`, `rhesus`, `rhesus_gd`. Notably, `rhesus_ig` is absent from this file. `combo_xcr_2026-08-06.tsv` adds 16 new organism-string values across 6 new base species (`cat`, `dog`, `ferret`, `rabbit`, `rainbowtrout`, `sheep`, each with base/`_gd`/`_ig` suffix variants), and newly includes `rhesus_ig` (781 rows) for the already-existing rhesus species.

**Chain-completeness scope decision.** An organism string is includable in this feature only if the database holds at least one V-region row and one J-region row for both chain `A` and chain `B` (four non-zero cells total). An exhaustive scan of `combo_xcr_2026-08-06.tsv` found: `cat`, `cat_gd`, `cat_ig`, `dog`, `dog_gd`, `dog_ig`, `ferret`, `ferret_gd`, `ferret_ig`, `rabbit`, `rabbit_gd`, `rabbit_ig`, and `sheep` (alpha-beta only) are complete; `rainbowtrout`, `rainbowtrout_ig`, `sheep_gd`, and `sheep_ig` are each missing an entire chain; `rainbowtrout_gd` does not occur in the file at all. `rhesus_ig` is independently confirmed complete (781 rows: 300 V-A, 392 V-B, 14 J-A, 11 J-B). Per this rule, `rainbowtrout` is excluded in every receptor-type variant, and `sheep` is included only as an alpha-beta organism.

**Blocking code locations.** Three locations validate or map organism strings in ways that would reject the new species even after the file swap: `conga/tcrdist/make_10x_clones_file.py`'s `get_ab_from_10x_chain`, which calls `sys.exit()` for any organism outside three hardcoded lists; `conga/util.py`'s `organism2vdj_type` dict, which has no entry for most of the newly-supported organisms and whose lookup sites currently raise a bare `KeyError`; and the `--organism choices=` argparse declarations in `scripts/run_conga.py` and `scripts/setup_10x_for_conga.py`, which reject unlisted organism strings before any Python code runs. Other CLI scripts (`run_conga_fixed.py`, `run_conga_updated.py`, `run_conga_with_backend.py`, `run_conga_with_validation.py`, `scripts/merge_samples.py`, `scripts/make_tcr_logos.py`) are out of scope for this feature.

**Plotting gap.** `conga/plotting.py`'s `default_logo_genes` and `default_gex_header_genes` dicts are keyed by exact organism string with no fallback; a lookup for an organism absent from either dict raises `KeyError`. This feature does not curate new marker-gene lists for the new species — gene curation depends on which GEX reference the user applied and cannot be assumed generically — but does change the lookup mechanism to omit any gene absent from the dict or absent from the user's actual GEX reference, rather than raising an error.

**Vectorized TCRdist extension.** `conga/tcrdist/vectorized.py` restricts `SUPPORTED_ORGANISMS` to `{human, mouse, rhesus}`. This is a documented scope decision, not an algorithmic limitation: the encoder's amino-acid embedding, germline-loop extraction, and CDR3 trim/gap math already read organism and chain-label strings generically, and gamma-delta/Ig records in `combo_xcr` use the same `chain='A'`/`chain='B'` convention as alpha-beta records. The one place real risk exists is the CDR3 gap-position formula in `trim_and_gap_cdr3`, whose shape was validated only against human/mouse/rhesus alpha-beta CDR3 length distributions; gamma-delta and Ig CDR3s are known to run longer and more length-variable, which is exactly outside the regime that formula was shaped for. This feature attempts to extend `SUPPORTED_ORGANISMS` to every newly-scoped species and receptor type, but gates each one individually through a reusable accuracy-validation harness; a combination that fails validation is excluded outright, with no partial or warned inclusion.

Real paired junctional CDR3 sequence data exists in this repository for `human`, `mouse`, and `rhesus`: `combo_xcr` holds only germline V/J gene segments, but the existing bundled matching-db files supply real paired CDR3 sequences for human and mouse, and the newly-added `conga/data/rhesus_clones.tsv` supplies 450 real, species-matched rhesus alpha-beta clonotypes with complete paired CDR3 sequence data. Validation therefore uses real sequence data for human, mouse, and rhesus, and synthetic sequences modeled on real human CDR3 content for the five fully-new species. This is a genuine limitation of the validation for those five species, not merely a formality, and this feature requires it to be documented plainly rather than glossed over.

## Glossary

- **Active_Database**: The tcrdist reference database file selected by `conga.tcrdist.basic.db_file` and loaded by `conga.tcrdist.all_genes`. Before this feature, `combo_xcr_2023-12-30.tsv`; after this feature, `combo_xcr_2026-08-06.tsv`.
- **Legacy_Database**: `combo_xcr_2023-12-30.tsv`, retained on disk for reference but no longer the Active_Database once this feature is implemented.
- **Stale_Database**: `combo_xcr.tsv` (no date suffix), an older, unused file missing `mouse_ig`/`rhesus_ig` entirely. Out of scope; not read, written, or referenced by this feature's implementation.
- **New_Species**: The five base organism strings `cat`, `dog`, `ferret`, `rabbit`, and `sheep`, which pass the Chain_Completeness_Rule for at least their alpha-beta receptor type and have no prior entry in the Legacy_Database.
- **Chain_Completeness_Rule**: The scope rule that an organism string is includable only if the Active_Database holds at least one V-region row and one J-region row for both Chain_Label `A` and Chain_Label `B`. Per this rule, `rainbowtrout` (all three receptor-type variants) is excluded entirely, and `sheep` is included only for its alpha-beta receptor type, excluding `sheep_gd` and `sheep_ig`.
- **Newly_Supported_Organism**: Any organism string this feature adds end-to-end support for: `cat`, `cat_gd`, `cat_ig`, `dog`, `dog_gd`, `dog_ig`, `ferret`, `ferret_gd`, `ferret_ig`, `rabbit`, `rabbit_gd`, `rabbit_ig`, `sheep`, and `rhesus_ig`.
- **Chain_Label**: The single-character chain identifier used by the Gene_Database, `A` for the alpha/gamma/light chain and `B` for the beta/delta/heavy chain, per the existing convention in `conga/tcrdist/all_genes.py`.
- **Gene_Database**: The in-memory structure built by `conga.tcrdist.all_genes` from the Active_Database, mapping organism string and gene identifier to germline gene records.
- **VDJ_Type**: One of the three values `TCR_AB_VDJ_TYPE`, `TCR_GD_VDJ_TYPE`, or `IG_VDJ_TYPE` defined in `conga/util.py`, used to classify an organism string's receptor type for downstream gene-prefix filtering logic.
- **Chain_Mapper**: The function `conga.tcrdist.make_10x_clones_file.get_ab_from_10x_chain`, which maps a 10x contig chain label (e.g. `TRA`, `TRG`, `IGH`) and an organism string to a Chain_Label.
- **Setup_CLI**: The command-line entry point `scripts/setup_10x_for_conga.py`.
- **Analysis_CLI**: The command-line entry point `scripts/run_conga.py`.
- **Marker_Gene_Lookup**: The organism-keyed gene-list lookup in `conga/plotting.py` (`default_logo_genes`, `default_gex_header_genes`) used to select which genes appear in logo plots and GEX header panels.
- **TCR_Vectorizer**: The module `conga/tcrdist/vectorized.py`, as defined in the `vectorized-tcrdist` spec.
- **Supported_Organism**: An organism string accepted by the TCR_Vectorizer for vectorized encoding, i.e. present in `SUPPORTED_ORGANISMS`. Before this feature, exactly `{human, mouse, rhesus}`; after this feature, that set plus every Newly_Supported_Organism (and, for `rhesus`, the pre-existing alpha-beta entry) that passes the Accuracy_Gate.
- **Accuracy_Gate**: The pass/fail criterion applied to a candidate organism-and-receptor-type combination before it is added to `SUPPORTED_ORGANISMS`: Spearman correlation of at least 0.90 and mean recall@10 of at least 0.70 between vectorized distance and exact TCRdist distance, matching the "Good" tier already documented in `conga.tcrdist.vectorized.accuracy_report`'s docstring. A combination that does not meet both thresholds fails the Accuracy_Gate and is excluded from `SUPPORTED_ORGANISMS` outright; there is no partial or warned-but-included outcome.
- **Validation_Harness**: A reusable, repeatable tool (not a one-off script) that computes an `AccuracyReport` for a given organism and records whether it passes the Accuracy_Gate, able to be re-run against a future Active_Database update.
- **Validation_Tier**: One of two provenance classifications assigned to an organism's Accuracy_Gate evidence: Tier_1 for `human`, `mouse`, and `rhesus` (real, species-matched paired CDR3 sequences from Human_CDR3_Corpus, Mouse_CDR3_Corpus, and Rhesus_CDR3_Corpus respectively); Tier_3 for `cat`, `dog`, `ferret`, `rabbit`, and `sheep` (fully synthetic CDR3 sequences with no real junction sequence of any kind). The Tier_3 label is retained as-is even though only two tiers exist, since Tier_3 is referenced by that name elsewhere in this document.
- **Human_CDR3_Corpus**: The real, paired human CDR3 sequences in the existing bundled file `conga/data/new_paired_tcr_db_for_matching_nr.tsv`.
- **Mouse_CDR3_Corpus**: The real, paired mouse CDR3 sequences in the existing bundled file `conga/data/mouse_tcr_db_for_matching.tsv`, restricted to rows where both `cdr3a` and `cdr3b` are populated.
- **Rhesus_CDR3_Corpus**: The real, paired rhesus alpha-beta CDR3 sequences in the newly-added bundled file `conga/data/rhesus_clones.tsv` (450 clonotypes; columns include `va_gene`, `ja_gene`, `vb_gene`, `jb_gene`, `cdr3a`, `cdr3b`; every row has complete alpha and beta chain data).
- **Synthetic_CDR3_Generator**: The Tier_3 validation method for `cat`, `dog`, `ferret`, `rabbit`, and `sheep`: an empirical per-position amino-acid frequency model derived from Human_CDR3_Corpus, sampled to produce synthetic CDR3 sequences of a length drawn from the target organism's own germline-implied CDR3 length distribution, paired with V/J genes sampled from that organism's own Gene_Database records.
- **Validation_Warning**: The non-silent notice (at minimum, a `logging.WARNING`) surfaced at the point of use when a Tier_3 organism is selected for vectorized encoding, naming the organism, stating that its Accuracy_Gate evidence is not species-matched real repertoire data, and naming the KernelPCA representation (`X_pca_tcr`) and the exact TCRdist path as more conservative alternatives.

## Requirements

### Requirement 1: Active database update

**User Story:** As a CoNGA maintainer, I want the package to load the current tcrdist reference database by default, so that users get the latest germline gene annotations without manual configuration.

#### Acceptance Criteria

1. THE conga package SHALL set `conga.tcrdist.basic.db_file` to `combo_xcr_2026-08-06.tsv`, making it the Active_Database.
2. THE conga package SHALL leave the Legacy_Database and the Stale_Database present on disk, unmodified, and unreferenced by any code path.
3. WHEN `conga.tcrdist.all_genes` is imported after this change, THE Gene_Database SHALL contain records for every organism string present in `combo_xcr_2026-08-06.tsv`, including every Newly_Supported_Organism and every organism excluded by the Chain_Completeness_Rule (the Gene_Database itself is not filtered; only the downstream code paths in Requirements 2-4 are scoped to Newly_Supported_Organism).

> Verification note: `conga/tcrdist/all_genes.py` builds `all_genes[organism][id]` generically from whatever organism strings occur in the file selected by `basic.db_file`, with no organism allowlist of its own. Changing the one `db_file` string is sufficient to make every organism in the new file available to the Gene_Database; the Chain_Completeness_Rule and the New_Species scope decision apply to the downstream consumer code paths covered by Requirements 2-4, not to this loading step.

### Requirement 2: Organism-string validation and error handling

**User Story:** As a CoNGA user who supplies an unsupported or misspelled organism string, I want a clear error naming the problem, so that I do not encounter a bare crash or silent misclassification.

#### Acceptance Criteria

1. THE Chain_Mapper SHALL accept every Newly_Supported_Organism, correctly mapping its 10x contig chain labels to Chain_Label values following the same per-receptor-type convention already applied to `human`/`mouse`/`rhesus` (alpha-beta), `human_gd`/`mouse_gd`/`rhesus_gd` (gamma-delta), and `human_ig`/`mouse_ig` (Ig).
2. IF the Chain_Mapper is called with an organism string it does not recognize, THEN THE Chain_Mapper SHALL raise a `ValueError` naming the unrecognized organism and listing the supported organism strings, replacing its prior `sys.exit()` behavior.
3. THE VDJ_Type lookup in `conga/util.py` SHALL contain an entry for every Newly_Supported_Organism, correctly classified as `TCR_AB_VDJ_TYPE`, `TCR_GD_VDJ_TYPE`, or `IG_VDJ_TYPE` according to its organism-string suffix.
4. IF a VDJ_Type lookup is performed for an organism string absent from the lookup, THEN THE conga package SHALL raise a `ValueError` naming the organism and listing the supported organism strings at the lookup site, replacing any bare `KeyError`.
5. THE conga package SHALL apply criteria 2 and 4 consistently, so that an unsupported organism string produces a `ValueError` with a clear message at both the Chain_Mapper and the VDJ_Type lookup, rather than a crash at one and a clear error at the other.

### Requirement 3: CLI organism choices

**User Story:** As a CoNGA user running the setup or analysis pipeline from the command line, I want to pass a new species' organism string without the argument parser rejecting it before my job starts.

#### Acceptance Criteria

1. THE Analysis_CLI's `--organism` argument SHALL accept every Newly_Supported_Organism in addition to its existing accepted values.
2. THE Setup_CLI's `--organism` argument SHALL accept every Newly_Supported_Organism in addition to its existing accepted values.
3. THE conga package SHALL NOT modify the `--organism` argument of any script other than the Analysis_CLI and the Setup_CLI.

### Requirement 4: Graceful marker-gene omission in plotting

**User Story:** As a CoNGA user analyzing a species without a curated marker-gene list, or whose GEX reference uses different gene symbols than the curated list expects, I want plots to render without the missing genes rather than crash.

#### Acceptance Criteria

1. IF the Marker_Gene_Lookup is performed for an organism string absent from `default_logo_genes` or `default_gex_header_genes`, THEN THE conga package SHALL omit that organism's logo genes or header genes from the affected plot rather than raising an error.
2. IF a gene named in `default_logo_genes` or `default_gex_header_genes` for a given organism is absent from the `adata`/GEX reference gene names at plot time, THEN THE conga package SHALL omit that specific gene from the affected plot rather than raising an error, regardless of whether the organism is a Newly_Supported_Organism or a pre-existing one.
3. WHEN one or more genes are omitted under criterion 1 or criterion 2, THE conga package SHALL continue to render the plot using the remaining available genes.
4. THE conga package SHALL NOT add new entries to `default_logo_genes` or `default_gex_header_genes` for any Newly_Supported_Organism as part of this feature.

### Requirement 5: Reusable accuracy-validation harness

**User Story:** As a CoNGA maintainer deciding whether to enable vectorized TCRdist for a species, and as a future maintainer updating the reference database again, I want a repeatable tool that measures encoding accuracy per organism, so that the decision is based on recorded evidence and can be re-checked later.

#### Acceptance Criteria

1. THE Validation_Harness SHALL be implemented as a standalone, re-runnable tool (for example, a script under `scripts/`), not a one-off interactive session.
2. THE Validation_Harness SHALL accept an organism string and produce an `AccuracyReport` via `conga.tcrdist.vectorized.accuracy_report`, using the validation data source specified for that organism in Requirement 6.
3. THE Validation_Harness SHALL evaluate the Accuracy_Gate for the produced `AccuracyReport` and report a pass/fail outcome alongside the measured Spearman correlation and mean recall@10 value.
4. THE Validation_Harness SHALL be runnable against any future Active_Database update without modification to the harness itself, reading organism and chain data solely through the Gene_Database.
5. WHEN the Validation_Harness is run for every Newly_Supported_Organism and `rhesus`, at every receptor type permitted by the Chain_Completeness_Rule, THE conga package SHALL record the measured Spearman correlation, mean recall@10, Validation_Tier, and pass/fail outcome for each combination in the design documentation for this feature.

### Requirement 6: Validation data sources

**User Story:** As a CoNGA maintainer, I want each organism's accuracy evidence to come from the most realistic data source available, so that the Accuracy_Gate outcome is as meaningful as the available data allows.

#### Acceptance Criteria

1. THE Validation_Harness SHALL validate `human` using paired CDR3 sequences drawn from the Human_CDR3_Corpus.
2. THE Validation_Harness SHALL validate `mouse` using paired CDR3 sequences drawn from the Mouse_CDR3_Corpus.
3. THE Validation_Harness SHALL validate `rhesus` using paired CDR3 sequences drawn from the Rhesus_CDR3_Corpus.
4. THE Validation_Harness SHALL validate each of `cat`, `dog`, `ferret`, `rabbit`, and `sheep` using the Synthetic_CDR3_Generator, deriving CDR3 amino-acid content from Human_CDR3_Corpus and CDR3 length and V/J gene identity from that organism's own Gene_Database records.
5. THE conga package SHALL classify `human`, `mouse`, and `rhesus` as Validation_Tier Tier_1, and each of `cat`, `dog`, `ferret`, `rabbit`, and `sheep` as Validation_Tier Tier_3.
6. THE design documentation for this feature SHALL state, for each Validation_Tier, which part of the evidence is real sequence data and which part (if any) is modeled, so that the limitation described in Requirement 9 is traceable to its source.

### Requirement 7: Accuracy-gated inclusion in SUPPORTED_ORGANISMS

**User Story:** As a CoNGA user selecting vectorized TCRdist for a species, I want that path to be available only where it has been shown to approximate exact TCRdist reasonably well, so that I am not silently given a poor approximation.

#### Acceptance Criteria

1. IF a candidate organism-and-receptor-type combination's `AccuracyReport` meets or exceeds both Accuracy_Gate thresholds (Spearman correlation at least 0.90 and mean recall@10 at least 0.70), THEN THE conga package SHALL add that combination's organism string to `SUPPORTED_ORGANISMS`.
2. IF a candidate organism-and-receptor-type combination's `AccuracyReport` does not meet both Accuracy_Gate thresholds, THEN THE conga package SHALL NOT add that combination's organism string to `SUPPORTED_ORGANISMS`, and SHALL leave that organism string handled exactly as any other unsupported organism is handled today: the TCR_Vectorizer's `_validate_organism` SHALL raise a `ValueError` directing the caller to the KernelPCA representation or the exact TCRdist path.
3. THE conga package SHALL NOT introduce a partial-inclusion or included-with-warning outcome for an organism that fails the Accuracy_Gate; the outcome is binary ("if the accuracy gate fails then vectorized should not be enabled as an option for the species/receptor-type").
4. THE conga package SHALL apply the Accuracy_Gate independently to each receptor type of a given species permitted by the Chain_Completeness_Rule (for example, `cat` passing does not imply `cat_gd` or `cat_ig` pass), since each receptor type's CDR3 length and content distribution differs.
5. THE conga package SHALL re-evaluate `rhesus`'s existing alpha-beta inclusion in `SUPPORTED_ORGANISMS` against the Accuracy_Gate using the Rhesus_CDR3_Corpus data source (confirmed by the user: rhesus is to be re-validated, not grandfathered in), and SHALL retain its inclusion only if it continues to meet both thresholds under this feature's validation. If rhesus fails under this re-validation, it SHALL be removed from `SUPPORTED_ORGANISMS` and handled exactly as any other organism that fails the Accuracy_Gate.

### Requirement 8: Tier 3 validation-provenance warning

**User Story:** As a CoNGA user selecting vectorized TCRdist for a species whose accuracy evidence is not species-matched real data, I want to be told so at the point I make that choice, so that I can decide whether the approximation is appropriate for my analysis.

#### Acceptance Criteria

1. WHEN vectorized encoding is performed for an organism classified as Validation_Tier Tier_3, THE TCR_Vectorizer SHALL emit a Validation_Warning.
2. THE Validation_Warning SHALL name the organism, state that its Accuracy_Gate evidence is not species-matched real repertoire data, and name the KernelPCA representation (`X_pca_tcr`) and the exact TCRdist path as alternatives.
3. WHEN vectorized encoding is performed for an organism classified as Validation_Tier Tier_1 (`human`, `mouse`, or `rhesus`), THE TCR_Vectorizer SHALL NOT emit a Validation_Warning, preserving today's silent behavior for those organisms.

### Requirement 9: Documented validation limitation

**User Story:** As a CoNGA maintainer or user reviewing this feature's design, I want the limits of synthetic-data accuracy validation stated plainly, so that a passing Accuracy_Gate is not mistaken for a guarantee of real-world performance.

#### Acceptance Criteria

1. THE design documentation for this feature SHALL state that `accuracy_report` measures agreement between vectorized-encoding distance and exact-TCRdist distance computed on the same input sequences, and that this comparison cannot detect a case where the Synthetic_CDR3_Generator's modeled sequence distribution diverges from a species' real-world CDR3 biology.
2. THE design documentation for this feature SHALL state this limitation as a property of the validation method itself, not as a deficiency specific to any one Newly_Supported_Organism.
3. THE design documentation for this feature SHALL record, for every evaluated combination, whether it passed or failed the Accuracy_Gate and its measured Spearman correlation and mean recall@10, regardless of outcome, so that a future tightening of the Accuracy_Gate thresholds has real recorded numbers to act on.
