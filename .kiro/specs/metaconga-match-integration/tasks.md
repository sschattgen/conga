# Implementation Plan: metaconga-match-integration

## Overview

Tasks are ordered so each one leaves the repo in a working, test-passing state. Data
files and tags land first since the module's own module-level code depends on them at
import time. The module port and `__init__.py` registration follow, then a checkpoint.
CLI argument declarations come before the validation block (which reads those args),
which comes before dispatch (which depends on validation having already run). A final
checkpoint closes out the feature. The optional `_encode_tcr_seqs` vectorization task
is isolated at the end of the module-port step so it can be skipped without blocking
anything downstream. No task performs the FAISS/vectorized-TCRdist rewrite of the
Clump_Pipeline — that is out of scope for this spec per Requirement 10.2.

1. **Task 1: Bundle the 11 metaconga reference data files**
   - Objective: Create `conga/data/metaconga/` and copy all 11 files verbatim from
     `/Users/sschattg/conga_mc_match/conga/data/metaconga/`.
   - Guidance: Byte-identical copy, no renaming or reformatting (Requirement 2).
   - Test: Add a test asserting all 11 filenames exist under `conga/data/metaconga/`
     (Requirement 9.2).
   - Demo: `ls conga/data/metaconga/` shows all 11 files; the new presence test passes
     in isolation.

2. **Task 2: Port the 5 metaconga tags into `conga/tags.py`**
   - Objective: Add `METACONGA_MATCH_CLUMPS`, `METACONGA_MATCH_AACLUSTERS` to the
     table-tag section, and `METACONGA_MATCH_AACLUSTERS_BARS`,
     `METACONGA_MATCH_AACLUSTERS_UMAPS`, `METACONGA_MATCH_CLUMPS_UMAPS` to the
     figure-tag section, exactly as placed in design.md Component 3.
   - Guidance: No other tag in `tags.py` is touched (Requirement 3.3).
   - Test: Add a test importing `conga.tags` and asserting each of the 5 new constants
     equals its expected string value.
   - Demo: `python -c "import conga.tags as t; print(t.METACONGA_MATCH_CLUMPS)"` prints
     `metaconga_match_clumps`; tag test passes.

3. **Task 3: Port `conga/metaconga_match.py` verbatim and register it**
   - Objective: Copy the module from the source branch unchanged, then add
     `from . import metaconga_match` to `conga/__init__.py` immediately after
     `tcr_clumping` (Component 4).
   - Guidance: No internal edits to the module's logic. This step depends on Tasks 1
     and 2 already being complete, since the module's own module-level code loads the
     data files and uses the tags at import time.
   - Test: Add a test that `import conga.metaconga_match` succeeds without raising
     (Requirement 9.1), and that `conga.metaconga_match.find_aacluster_matches` /
     `plot_aacluster_matches` / `find_clump_matches` / `plot_clump_matches` are all
     resolvable as attributes.
   - Demo: `mamba run -n conga-dev python -c "import conga; conga.metaconga_match.find_aacluster_matches"`
     succeeds; import test passes.

   - **Task 3.2 (OPTIONAL): Vectorize `_encode_tcr_seqs`'s per-row Python loop**
     - Objective: Replace the `for ii, l in enumerate(tcr_df.itertuples()):` loop with
       NumPy-vectorized gene lookup and per-residue AA counting, preserving identical
       output.
     - Guidance: This is a performance optimization only, not required for
       correctness or for any requirement in requirements.md. Skip if time-constrained.
     - Test: If implemented, add a test comparing vectorized output against the
       original loop-based output on a small fixture, asserting exact equality.
     - Demo: Side-by-side timing comparison on a synthetic TCR dataframe showing a
       speedup, with identical returned arrays.

4. **Checkpoint: Run full test suite**
   - Objective: Confirm Tasks 1-3 integrate cleanly before touching CLI code.
   - Guidance: Run `mamba run -n conga-dev pytest tests/ -v`. All existing tests plus
     the new ones from Tasks 1-3 must pass.
   - Demo: Clean pytest run, zero failures.

5. **Task 4: Add the two new CLI argument declarations**
   - Objective: Add `--match_metaconga_aaclusters` and `--match_metaconga_clumps` to
     `scripts/run_conga.py`'s argparse setup, placed between `--tcr_clumping` and
     `--find_hotspot_features` (Component 5).
   - Guidance: Do not add either flag to `all_modes`.
   - Test: Add tests confirming `--match_metaconga_aaclusters` accepts `cd4`, `cd8`,
     `CD4`, `CD8`, and omission (`None`), and rejects any other value via argparse's
     `choices` mechanism (Requirement 9.3).
   - Demo: `python scripts/run_conga.py --help` lists both new flags with their help
     text; `--match_metaconga_aaclusters bogus` fails with argparse's standard usage
     error.

6. **Task 5: Add the early cross-flag validation block**
   - Objective: Implement checks (a)-(d) from design.md Component 6, in the exact
     order specified, inserted into the existing gap after the
     `--force_variable_genes`/batch mutual-exclusion check.
   - Guidance: This depends on Task 4 (the flags must exist to be validated). Order
     matters: (a) extended mutual-exclusion, (b) CD_Subset_Pairing_Rule, (c)
     Metaconga_Organism_Gate, (d) Auto_Injection_Behavior — in that sequence, so (d)
     never runs when (a)-(c) would have exited first.
   - Test: Add tests for each `sys.exit` path:
     - mutual-exclusion fires when `--match_metaconga_aaclusters` + `--batch_key`/
       `--batch_integration_method` (Requirement 9.6)
     - CD_Subset_Pairing_Rule fires for `cd4` without `--subset_to_CD4_cells` and for
       `cd8` without `--subset_to_CD8_cells` (Requirement 9.5)
     - Metaconga_Organism_Gate fires for either flag with `--organism` != `human`
       (Requirement 9.4)
     - Auto_Injection_Behavior sets `args.force_variable_genes` to the bundled file
       when not user-supplied, and leaves a user-supplied value untouched
       (Requirement 9.7)
   - Demo: Running `run_conga.py --match_metaconga_aaclusters cd4 --organism mouse ...`
     exits immediately with a clear organism error; running it with
     `--subset_to_CD4_cells` and no `--force_variable_genes` prints the
     `WARNING: ... adding --force_variable_genes ...` message and proceeds.

7. **Checkpoint: Run full test suite**
   - Objective: Confirm CLI parsing and validation logic is correct and fully covered
     before wiring dispatch.
   - Guidance: Run `mamba run -n conga-dev pytest tests/ -v`.
   - Demo: Clean pytest run, zero failures, including all new CLI validation tests.

8. **Task 6: Wire the Analysis_Dispatch_Section**
   - Objective: Add the two dispatch blocks (AACluster_Pipeline and Clump_Pipeline)
     from design.md Component 7, inserted after the existing `--tcr_clumping` block
     and before `--graph_vs_graph_stats`.
   - Guidance: Depends on Tasks 3 (module importable) and 5 (validation already
     guarantees `cd48` is `'cd4'`/`'cd8'` and organism is `'human'` by the time this
     code runs). Both blocks must be able to run in the same invocation independently.
   - Test: Add end-to-end tests invoking `find_aacluster_matches`/
     `plot_aacluster_matches` and `find_clump_matches`/`plot_clump_matches` against a
     Representative_Human_Fixture, asserting no exception is raised (Requirements 9.8,
     9.9).
   - Demo: Running `run_conga.py --match_metaconga_aaclusters cd4
     --subset_to_CD4_cells --match_metaconga_clumps --organism human ...` against a
     small real or synthetic dataset completes and produces the expected `.tsv`/`.png`
     outputs under both new table tags and all three new figure tags.

9. **Checkpoint: Final full test suite run**
   - Objective: Confirm the complete feature (data files, tags, module, CLI flags,
     validation, dispatch) works end-to-end with no regressions to existing
     functionality.
   - Guidance: Run `mamba run -n conga-dev pytest tests/ -v`.
   - Demo: Clean pytest run, zero failures, full new test coverage from Requirement 9
     present and passing.

## Notes

- Tasks 1 and 2 have no dependency on each other and could be done in either order;
  they are listed data-files-first only because that's the larger, more mechanical
  step.
- Task 3.2 (vectorization) is optional and can be deferred indefinitely without
  blocking any other task — nothing downstream depends on `_encode_tcr_seqs`'s
  internal implementation, only on its external behavior staying the same.
- No task implements the FAISS/vectorized-TCRdist rewrite of the Clump_Pipeline's
  exact-TCRdist search; that idea is explicitly out of scope for this spec
  (Requirement 10.2) and is not represented here even as a placeholder.

## Task Dependency Graph

{
  "waves": [
    ["1", "2"],
    ["3"],
    ["3.2 (optional)"],
    ["checkpoint-1"],
    ["4"],
    ["5"],
    ["checkpoint-2"],
    ["6"],
    ["checkpoint-3"]
  ]
}
