# `inplace=True` Audit (Task 4.4)

Date: 2026-10-09
Environment: conga-dev (pandas 3.0.6, numpy 2.5.3, Python 3.14.8)
Scope: all `inplace=True` call sites in `conga/` (29 sites across 9 files,
found via `grep -rn "inplace\s*=\s*True" conga/`)

## Why this matters under pandas 3.0

Pandas 3.0 makes copy-on-write (CoW) unconditional. Under CoW, `inplace=True`
is only safe when the DataFrame/Series you're calling it on is the object you
actually own (a fresh local variable, a dict value you just built, or an
AnnData `.obs`/`.var` container accessed directly) rather than something that
might be a CoW-tracked view of another DataFrame (e.g. the result of
`df[mask]`, `df.loc[...]`, or a column slice). Calling `inplace=True` on a
view can silently no-op instead of raising, which is the dangerous failure
mode this task is checking for.

## Method

Reviewed every call site's surrounding code (not just the `inplace=True`
line) to determine the provenance of the DataFrame/Series being mutated:
is it a transient local object the function constructed/owns outright, or
could it be a slice/view of a DataFrame passed in or stored elsewhere?

## Findings

All 29 call sites fall into one of two safe categories:

### Category 1: Freshly constructed local DataFrames (24 sites)

The object being mutated was constructed in the same function via
`pd.DataFrame(...)`, `pd.concat(...)`, `pd.read_csv(...)`, or returned from
another function and reassigned to a plain local variable -- never derived
from `df[mask]`/`df.loc[...]` subscripting of something else in the same
scope. Examples:

- `conga/correlations.py:647,1388,1431,1480,1768-1770` -- `results_df =
  pd.concat(...)` or `results_df = pd.DataFrame(...)` immediately followed by
  `.sort_values(..., inplace=True)` / `.drop(..., inplace=True)` /
  `.reset_index(inplace=True)` on that same local, freshly-built object.
- `conga/tcr_clumping.py:433,502,719,806,809` -- same pattern:
  `results_df = pd.DataFrame(dfl)` then `.sort_values(..., inplace=True)`.
- `conga/metaconga_match.py:551,659,1060,1063,1068,1091,1110,1204` -- all
  operate on `results`/`dfdegs`/`info` local DataFrames built or copied
  earlier in the same function.
- `conga/devel.py:1833,2139` -- `results = pd.DataFrame(dfl)` then
  `.sort_values(inplace=True)`.
- `conga/plotting.py:1617,2751` -- `df = pd.DataFrame(dfl)` then
  `.sort_values(inplace=True)`.
- Module-level setup in `conga/tcr_scoring.py:28` and
  `conga/imhc_scoring.py:14`: `aa_props_df = pd.read_csv(...)` /
  `imhc_model_df = pd.read_csv(...)` followed by `.set_index(inplace=True)`
  on the DataFrame that was just read -- not a slice of anything.
- `conga/preprocess.py:347,2875` -- `clones_df.rename(...)` on the
  DataFrame just loaded via `pd.read_csv`; `all_barcodes.set_index(...)` on
  a DataFrame just loaded via `pd.read_csv`.

None of these are views/slices of another DataFrame; they are the sole
reference to data the function just created. CoW does not affect this
pattern -- it is equivalent to `df2 = df; df2.method(inplace=True)` where
`df2 is df` was never a subscript result.

### Category 2: Direct AnnData `.obs` container mutation (5 sites)

- `conga/correlations.py:754` -- `adata.obs.drop(columns=[rg_tag],
  inplace=True)`, dropping a column added two lines earlier via
  `adata.obs[rg_tag] = vals` in the same function. `adata.obs` is accessed
  directly as an attribute, not as a subscript/slice of another frame, so
  this is the CoW-safe "I own this object" pattern. Verified by reading
  lines 735-759 of `correlations.py` directly.

(The other `adata.obs[...]`/`adata.uns[...]` assignments flagged by
`audit_cow_patterns.py` elsewhere in the codebase are column/key assignment,
not `inplace=True` calls, and are covered in
`modernization_results/cow_audit_2026-10-09.txt`'s manual-review section.)

## Conclusion

**Task 4.4 is VERIFIED COMPLETE.** All 29 `inplace=True` call sites in
`conga/` were individually reviewed for CoW safety. Every site operates on a
DataFrame/Series the calling function owns outright (freshly constructed,
freshly loaded, or an AnnData container accessed by attribute) rather than a
subscript-derived view/slice that CoW could silently detach. No code changes
were required.

This complements the `audit_cow_patterns.py` run documented in
`cow_audit_2026-10-09.txt`, which separately flagged (as false positives) a
large number of plain dict/`.uns` subscript assignments that are unrelated
to `inplace=True` semantics.

No further action needed on this task unless new `inplace=True` call sites
are added in the future -- any new site should be checked against the same
"do I own this object outright" question before merging.
