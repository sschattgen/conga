# Implementation Plan: tcrdist-db-update

## Overview

This plan repoints CoNGA's active tcrdist reference database to `combo_xcr_2026-08-06.tsv`, fixes two hard-crash organism-handling bugs, extends three organism allowlists, makes marker-gene plotting lookups degrade gracefully, and extends `conga.tcrdist.vectorized.SUPPORTED_ORGANISMS` to newly-reachable organisms subject to an accuracy-validation gate. Work proceeds in dependency order: the database swap lands first (everything else depends on the new organisms being loadable), then the three allowlist/crash fixes, then CLI and plotting fixes, then the vectorized-TCRdist validation pipeline (dead-code removal → data-source loaders → harness → harness run with recorded numbers → `SUPPORTED_ORGANISMS` hand-edit → Tier 3 warning, strictly in that order since each step consumes the previous one's output).

All Python execution uses `mamba run -n conga-dev python ...` / `mamba run -n conga-dev pytest ...` per the project's development workflow. Property-based tests use `hypothesis` at a minimum of 100 iterations, matching the existing convention in this repo.

## Tasks

- [x] 1. Swap the active database and verify the Gene_Database loads every organism
  - In `conga/tcrdist/basic.py`, change `db_file = 'combo_xcr_2023-12-30.tsv' # now including mouse_ig'` to `db_file = 'combo_xcr_2026-08-06.tsv' # adds cat/dog/ferret/rabbit/sheep + rhesus_ig`
  - Leave `combo_xcr.tsv` (Stale_Database) and `combo_xcr_2023-12-30.tsv` (Legacy_Database) untouched on disk and unreferenced by any code path
  - _Requirements: 1.1, 1.2, 1.3_

  - [x]* 1.1 Write unit test confirming Gene_Database coverage after the swap
    - Import `conga.tcrdist.all_genes` fresh and assert `all_genes` contains every organism string present in `combo_xcr_2026-08-06.tsv`, including every Newly_Supported_Organism (`cat`, `cat_gd`, `cat_ig`, `dog`, `dog_gd`, `dog_ig`, `ferret`, `ferret_gd`, `ferret_ig`, `rabbit`, `rabbit_gd`, `rabbit_ig`, `sheep`, `rhesus_ig`) and every Chain_Completeness_Rule-excluded string (`rainbowtrout`, `rainbowtrout_ig`, `sheep_gd`, `sheep_ig`)
    - _Requirements: 1.3_

- [x] 2. Fix the `get_ab_from_10x_chain` crash and extend its organism lists
  - In `conga/tcrdist/make_10x_clones_file.py::get_ab_from_10x_chain`, append each Newly_Supported_Organism to the organism tuple matching its receptor type: `cat`, `dog`, `ferret`, `rabbit`, `sheep` to the alpha-beta list; `cat_gd`, `dog_gd`, `ferret_gd`, `rabbit_gd` to the gamma-delta list; `cat_ig`, `dog_ig`, `ferret_ig`, `rabbit_ig`, `rhesus_ig` to the Ig list
  - Replace the trailing `else: print('unrecognized organism in get_ab_from_10x_chain:', organism); sys.exit()` with `raise ValueError(...)` naming the unrecognized organism and listing every supported organism string, exactly as specified in design Component 1
  - Deliberately omit `sheep_gd`, `sheep_ig`, and every `rainbowtrout` variant from all three lists so they fall through to the new `ValueError`
  - _Requirements: 2.1, 2.2_

  - [x]* 2.1 Write unit tests for `get_ab_from_10x_chain`
    - For every Newly_Supported_Organism, assert the correct Chain_Label is returned for each valid 10x chain label (e.g. `TRA`/`TRB` for `cat`, `TRG`/`TRD` for `dog_gd`, `IGH`/`IGK`/`IGL` for `rabbit_ig`)
    - For `rainbowtrout`, `rainbowtrout_gd`, `rainbowtrout_ig`, `sheep_gd`, and `sheep_ig`, assert a `ValueError` is raised whose message contains the organism string
    - _Requirements: 2.1, 2.2_

- [x] 3. Add `get_vdj_type` helper to `conga/util.py` and extend `organism2vdj_type`, fixing the three bare-`KeyError` call sites
  - Add one dict entry per Newly_Supported_Organism to `organism2vdj_type`, classified per design Component 1's literal (`cat`/`dog`/`ferret`/`rabbit`/`sheep` → `TCR_AB_VDJ_TYPE`; `cat_gd`/`dog_gd`/`ferret_gd`/`rabbit_gd` → `TCR_GD_VDJ_TYPE`; `cat_ig`/`dog_ig`/`ferret_ig`/`rabbit_ig`/`rhesus_ig` → `IG_VDJ_TYPE`)
  - Add the `get_vdj_type(organism: str) -> str` helper function next to the dict, raising `ValueError` naming the organism and `sorted(organism2vdj_type)` on `KeyError`
  - Replace the bare `organism2vdj_type[organism]` subscript at `conga/util.py::is_vdj_gene` (~line 199) with `get_vdj_type(organism)`
  - Replace the bare `organism2vdj_type[organism]` subscript at `conga/tcrdist/make_10x_clones_file.py::read_tcr_data` (~line 104) with `get_vdj_type(organism)`
  - Replace the bare `organism2vdj_type[organism]` subscript at `conga/tcrdist/make_10x_clones_file.py::read_tcr_data_batch` (~line 296) with `get_vdj_type(organism)`
  - _Requirements: 2.3, 2.4, 2.5_

  - [x]* 3.1 Write unit tests for `get_vdj_type`
    - For every Newly_Supported_Organism, assert the correct VDJ_Type constant is returned
    - For `rainbowtrout` (and the other Chain_Completeness_Rule-excluded strings), assert `ValueError` is raised with the organism name in the message
    - _Requirements: 2.3, 2.4, 2.5_

- [x] 4. Checkpoint - Ensure all tests pass
  - Run `mamba run -n conga-dev pytest tests/ -v` covering tasks 1-3
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Update CLI `--organism` choices in both scripts
  - In `scripts/run_conga.py`, extend the `--organism` `choices=` list to add `rhesus_ig`, `cat`, `cat_gd`, `cat_ig`, `dog`, `dog_gd`, `dog_ig`, `ferret`, `ferret_gd`, `ferret_ig`, `rabbit`, `rabbit_gd`, `rabbit_ig`, `sheep`, per design Component 2
  - In `scripts/setup_10x_for_conga.py`, extend the `--organism` `choices=` list with the same set
  - Do not modify `--organism` in any other script (`run_conga_fixed.py`, `run_conga_updated.py`, `run_conga_with_backend.py`, `run_conga_with_validation.py`, `scripts/merge_samples.py`, `scripts/make_tcr_logos.py`)
  - _Requirements: 3.1, 3.2, 3.3_

  - [x]* 5.1 Write CLI argparse smoke tests
    - For each of `run_conga.py` and `setup_10x_for_conga.py`, construct the script's `argparse.ArgumentParser` (or call `parse_args` with a minimal required-argument stub) with `--organism` set to a representative sample of Newly_Supported_Organism values (e.g. `cat`, `dog_gd`, `rabbit_ig`, `sheep`) and assert no `SystemExit` is raised
    - _Requirements: 3.1, 3.2_

- [x] 6. Make marker-gene lookups in `conga/plotting.py` degrade gracefully
  - Add a module-level `logger = logging.getLogger(__name__)` to `conga/plotting.py` (adding the `logging` import if not already present; the rest of the file keeps using `print(...)`, unchanged)
  - In `make_logo_plots`, replace `logo_genes = default_logo_genes[organism]` with `default_logo_genes.get(organism, [])`, logging a `logger.warning(...)` when the result is empty
  - Replace `header2_genes = default_gex_header_genes[organism]` with `default_gex_header_genes.get(organism, [])`, logging a `logger.warning(...)` when the result is empty
  - Make the `gene_width`/`assert len(logo_genes) == 3*gene_width - 2` block conditional: when `logo_genes` is non-empty, keep `gene_width = gene_logo_width` and the existing assert; when `logo_genes` is empty, set `gene_width = 1` and skip the assert, exactly as specified in design Component 3
  - Leave the existing `X_igex_genes = sorted(set(x for x in logo_genes+header2_genes if x in raw_var_names))` filter at ~line 520 untouched
  - Do not add any new entries to `default_logo_genes` or `default_gex_header_genes`
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [x]* 6.1 Write unit test for the missing-organism graceful fallback
    - Call `make_logo_plots` (or the extracted gene-selection logic) with an organism string absent from both `default_logo_genes` and `default_gex_header_genes`, asserting no exception is raised and the resulting logo/header gene lists are empty
    - **Property 5: Logo/header gene lookup never raises**
    - **Validates: Requirements 4.1, 4.3**

  - [x]* 6.2 Write regression test for existing curated organisms
    - Confirm `default_logo_genes['human']` still resolves to exactly 16 genes and still satisfies `assert len(logo_genes) == 3*gene_width - 2` for the default `gene_logo_width=6`, confirming the conditional `gene_width` branch did not alter existing behavior
    - _Requirements: 4.1_

- [x] 7. Checkpoint - Ensure all tests pass
  - Run `mamba run -n conga-dev pytest tests/ -v` covering tasks 5-6
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Remove the dead duplicate `accuracy_report` definition in `conga/tcrdist/vectorized.py`
  - Delete the first `accuracy_report` definition (roughly lines 1656-1881), which calls the nonexistent `TcrDistCalculator.tcr_distance` method and is unreachable dead code
  - Retain the second definition (roughly lines 1991-2203) unmodified; this is the one every docstring example in the module documents and the one later tasks' Validation_Harness will call
  - _Requirements: 5.2 (prerequisite: ensures the harness built in task 10 calls the working implementation)_

  - [x]* 8.1 Write unit test confirming exactly one `accuracy_report` symbol remains
    - Assert `conga.tcrdist.vectorized` exposes exactly one `accuracy_report` symbol at module scope, confirming the dead first definition was actually deleted and not merely shadowed
    - _Requirements: 5.2_

- [x] 9. Implement the four validation data-source functions in `conga/tcrdist/vectorized.py`
  - Implement `_load_human_cdr3_corpus() -> pd.DataFrame`: load `conga/data/new_paired_tcr_db_for_matching_nr.tsv` via `conga.util.path_to_data`, filtered to rows where both `cdr3a` and `cdr3b` are non-empty, using columns `cdr3a`, `cdr3b`, `va`, `vb`, `ja`, `jb`
  - Implement `_load_mouse_cdr3_corpus() -> pd.DataFrame`: load `conga/data/mouse_tcr_db_for_matching.tsv` via `conga.util.path_to_data`, same paired-row filter, accounting for the file's extra leading unnamed index column
  - Implement `_load_rhesus_cdr3_corpus() -> pd.DataFrame`: load `conga/data/rhesus_clones.tsv` via `conga.util.path_to_data`, using columns `cdr3a`, `cdr3b`, `va_gene`, `ja_gene`, `vb_gene`, `jb_gene` (the clones-file naming convention, distinct from the two loaders above); no paired-row filtering needed since all 450 rows are already complete
  - Implement `_build_synthetic_cdr3_corpus(organism: str, n: int, random_seed: int) -> list[tuple[tuple, tuple]]` (the Synthetic_CDR3_Generator): build a pooled per-residue amino-acid frequency table from `_load_human_cdr3_corpus()`'s `cdr3a`/`cdr3b` columns (20 amino acids, normalized counts, pooled across all positions and lengths); sample CDR3 length per sequence from the empirical length distribution observed in `_load_human_cdr3_corpus()`; sample V/J gene identifiers uniformly and independently from `all_genes[organism]` filtered to `chain='A'/'B'`, `region='V'/'J'` respectively, so every sampled gene id is guaranteed to be a valid key in `germline_code_table`'s gene list for that organism
  - Document in each docstring the exact column-naming distinction and path-resolution convention (via `conga.util.path_to_data`, not a hardcoded string) called out in design Component 4
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [x]* 9.1 Write property test for validation-corpus gene-id resolution
    - Parameterize over every organism in the harness dispatch table (task 10) and assert that for every gene id produced by that organism's data-source function (real or synthetic), the gene id is a valid key in that organism's `germline_code_table` gene list
    - **Property 3: Validation corpus gene ids always resolve**
    - **Validates: Requirements 5.2, 6.4**

- [x] 10. Build the Validation_Harness script
  - Create `scripts/validate_vectorized_tcrdist_accuracy.py`, structurally modeled on `scripts/validate_faiss_accuracy.py` (argparse CLI, `logging.basicConfig`, printed pass/fail with measured numbers), but organism-indexed
  - Add CLI arguments: `--organism` (one of the 18 evaluated strings, or `all`), `--n-synthetic` (default 1000, used only for Tier 3 organisms), `--random-seed` (default `conga.util.DEFAULT_RANDOM_SEED`), `--verbose`
  - For each requested organism, dispatch to the correct data-source function per the Validation_Tier table in design Component 4: `_load_human_cdr3_corpus` for `human`; `_load_mouse_cdr3_corpus` for `mouse`; `_load_rhesus_cdr3_corpus` for `rhesus`; `_build_synthetic_cdr3_corpus` for `rhesus_gd`, `rhesus_ig`, `cat`, `cat_gd`, `cat_ig`, `dog`, `dog_gd`, `dog_ig`, `ferret`, `ferret_gd`, `ferret_ig`, `rabbit`, `rabbit_gd`, `rabbit_ig`, `sheep`
  - Project each `DataFrame`-returning loader's output into the common tuple shape `list[tuple[tuple[str, str, str], tuple[str, str, str]]]` (per the Data Models normalization contract) before calling `accuracy_report`; pass `_build_synthetic_cdr3_corpus`'s output straight through unchanged since it already returns this shape
  - Call `report = conga.tcrdist.vectorized.accuracy_report(tcrs, organism)`, compute `passed = report.spearman >= 0.90 and report.mean_recall[10] >= 0.70`, and print organism, Validation_Tier, spearman, mean recall@10, and PASS/FAIL
  - Exit code 0 if every requested organism passed, 1 otherwise
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [x] 11. Run the Validation_Harness against all 18 combinations and record results in design.md
  - Run `mamba run -n conga-dev python scripts/validate_vectorized_tcrdist_accuracy.py --organism all --verbose` covering `human`, `mouse`, `rhesus`, `rhesus_gd`, `rhesus_ig`, `cat`, `cat_gd`, `cat_ig`, `dog`, `dog_gd`, `dog_ig`, `ferret`, `ferret_gd`, `ferret_ig`, `rabbit`, `rabbit_gd`, `rabbit_ig`, `sheep`
  - Replace every `*pending*` cell in the Results table in `.kiro/specs/tcrdist-db-update/design.md` (Component 4) with the measured Spearman correlation, mean recall@10, Accuracy_Gate pass/fail outcome, and whether the organism was added to `SUPPORTED_ORGANISMS`, for every row regardless of outcome
  - _Requirements: 5.5, 9.3_

- [x] 12. Hand-edit `SUPPORTED_ORGANISMS` based on the recorded harness results
  - In `conga/tcrdist/vectorized.py`, update the `SUPPORTED_ORGANISMS` frozenset literal to add exactly the organisms whose Results-table row (task 11) reads pass
  - Apply Requirement 7.5: treat `rhesus`'s pre-existing inclusion as subject to the same re-validation as every new candidate; if its row fails, remove `rhesus` from the frozenset
  - Do not introduce any partial-inclusion or included-with-warning outcome; a failing combination is excluded outright and continues to raise `ValueError` from `_validate_organism` directing the caller to `X_pca_tcr` or the exact TCRdist path
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [x]* 12.1 Write unit test for SUPPORTED_ORGANISMS/Results-table agreement
    - For every organism string in `SUPPORTED_ORGANISMS` after this edit, assert there is a corresponding passing row in the Results table recorded in task 11 (read from design.md or a committed copy of the harness output used as a fixture)
    - **Property 2: No unsupported organism in SUPPORTED_ORGANISMS without recorded evidence**
    - **Validates: Requirements 7.1, 7.2, 7.3**

- [x] 13. Implement the Tier 3 Validation_Warning in `encode_tcrs`
  - Add the module-level `_TIER_3_ORGANISMS: frozenset[str]` to `conga/tcrdist/vectorized.py`, populated from the final Results table (task 11): every organism that both passed the Accuracy_Gate and is Tier_3 per the Validation_Tier table (candidates: `rhesus_gd`, `rhesus_ig`, `cat`, `cat_gd`, `cat_ig`, `dog`, `dog_gd`, `dog_ig`, `ferret`, `ferret_gd`, `ferret_ig`, `rabbit`, `rabbit_gd`, `rabbit_ig`, `sheep`, filtered to those that actually passed)
  - Add the module-level `_already_warned_organisms: set[str] = set()`
  - Inside `encode_tcrs`, immediately after the existing `_validate_organism(organism)` call, add the one-time-per-process `logger.warning(...)` for any organism in `_TIER_3_ORGANISMS` not yet in `_already_warned_organisms`, naming the organism, stating its Accuracy_Gate evidence is not species-matched real repertoire data, and naming `X_pca_tcr` and the exact TCRdist path as alternatives, exactly as specified in design Component 4
  - Ensure `human`, `mouse`, and `rhesus` are never added to `_TIER_3_ORGANISMS` regardless of validation outcome, since Tier_1 classification is independent of pass/fail
  - _Requirements: 8.1, 8.2, 8.3_

  - [x]* 13.1 Write unit test for once-per-process Validation_Warning behavior
    - Call `encode_tcrs` twice in the same test for the same Tier_3 organism (via `caplog` or an equivalent log-capture fixture) and assert the warning is logged on the first call only
    - Call `encode_tcrs` for `human`, `mouse`, and `rhesus` and assert the warning never fires for any of them, regardless of call count
    - **Property 4: Validation_Warning fires if and only if Tier_3**
    - **Validates: Requirements 8.1, 8.2, 8.3**

- [x] 14. Final checkpoint - Ensure all tests pass
  - Run `mamba run -n conga-dev pytest tests/ -v` covering the full suite (tasks 1-13)
  - Confirm `conga/tcrdist/vectorized.py` exposes exactly one `accuracy_report` and that `SUPPORTED_ORGANISMS` matches the Results table recorded in design.md
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional property-based and unit tests and are not implemented as part of this plan's execution; core implementation tasks (unmarked) are always implemented
- Task 8 (dead-code removal) must land before task 9 (data-source loaders) and task 10 (harness), since the harness calls `accuracy_report` and must get the sole working implementation
- Task 9 (loaders) must land before task 10 (harness), since the harness dispatches to these functions
- Task 10 (harness) must land before task 11 (run + record), task 11 before task 12 (`SUPPORTED_ORGANISMS` edit), and task 12 before task 13 (Tier 3 warning), since each step consumes the previous step's output
- Task 1 (database swap) must land before any task that depends on new organisms being loadable via `all_genes` (tasks 2, 3, 9, 10, 11, 12)
- Per design Resolved Decision 5, `rhesus` is re-validated rather than grandfathered; its Tier_1 classification does not guarantee inclusion in `SUPPORTED_ORGANISMS`

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["2.1", "3.1", "5.1"] },
    { "id": 2, "tasks": ["6.1", "6.2"] },
    { "id": 3, "tasks": ["8.1"] },
    { "id": 4, "tasks": ["9.1"] },
    { "id": 5, "tasks": ["12.1"] },
    { "id": 6, "tasks": ["13.1"] }
  ]
}
```
