# Suggestions for the CoNGA dev team

A running log of errors, inconsistencies, and points of confusion found in the
code and docs while setting up and using CoNGA. Each entry says where the
problem is, what goes wrong, and a suggested fix. Line numbers refer to the
commit current when the entry was written (started 2026-10-16 at `ebe0732`).
Entries 1, 2 and 8 were updated 2026-10-18 after merging master `e5644fd`.

Severity: **[bug]** = wrong behavior or instructions that fail;
**[risk]** = can silently give wrong or degraded results;
**[confusing]** = inconsistent or stale, but not harmful by itself.

---

## Installation and environment

### 1. [bug] Pip-only install fails on older-glibc Linux (h5py builds from source)
- **Where:** README "Quick Installation", INSTALL.md (all `pip install -e .` routes)
- **What:** On hosts with glibc < 2.28 (e.g. Ubuntu 18.04, glibc 2.27, which
  includes the Fred Hutch `rhino` nodes), the newest h5py release only ships
  `manylinux_2_28` wheels. pip falls back to the sdist and fails with
  `error: libhdf5.so: cannot open shared object file`.
- **Fix that works (verified on rhino03, Python 3.12):**
  `pip install --only-binary :all: -e ".[performance,dev]"`. pip backtracks
  to h5py 3.15.1, which has a `manylinux_2_17` wheel. Every other package
  still resolves to its newest version (numpy 2.5.4, pandas 3.0.6, scanpy
  1.12.4, anndata 0.13.4).
  Alternative: `mamba install -c conda-forge h5py` before the pip step.
- **Suggestion:** add a short "Troubleshooting: older Linux" note to the
  README/INSTALL.md.
- **Update (master `e5644fd`):** `environment.yml` now pins `h5py=3.15.1`,
  which fixes the `mamba env create -f environment.yml` route. The pip routes
  are unchanged: `pyproject.toml` has no h5py constraint (h5py comes in
  through anndata/scanpy/tables), so `pip install -e .` on an older-glibc
  host still tries to build h5py 3.16 and fails. Options: add the same
  workaround note to the docs, or add `h5py<3.16` (or `h5py!=3.16.*`) to
  `pyproject.toml` dependencies, keeping in mind that this also holds back
  hosts that could use 3.16 wheels.

### 2. [bug] Wrong path to `tcrdist_cpp` in the dev-install instructions
- **Where:** INSTALL.md:90: `cd conga/tcrdist_cpp && make && cd ../..`, and
  README.md:318: `cd tcrdist_cpp && make && cd ../..`
- **What:** `tcrdist_cpp/` is at the repo root, not inside the `conga/`
  package. Master `e5644fd` fixed the first `cd` in the README, but its
  trailing `cd ../..` now goes one level *above* the repository, so the
  next command in the instructions runs in the wrong directory. INSTALL.md:90
  still has the original wrong path. Elsewhere INSTALL.md (lines 53, 72, 436),
  `setup_dev_env.sh`, and the Dockerfile are correct.
- **Fix:** `cd tcrdist_cpp && make && cd ..` in both places.

### 3. [risk] The README's Quick Installation skips the C++ build step
- **Where:** README.md ~lines 288-310
- **What:** The Quick Installation block never runs `make` in `tcrdist_cpp/`.
  Without the binaries conga falls back to the much slower Python TCRdist.
  The fallback warns, but it's easy to miss. INSTALL.md's quick install
  includes the step; the README's does not.
- **Fix:** add `(cd tcrdist_cpp && make)` to the README quick install.

### 4. [risk] The documented pandas 3 / NumPy 2 requirement is not enforced
- **Where:** README.md:398 ("CoNGA requires **pandas 3.0+** and **NumPy
  2.0+** (pulled in automatically by `pip install -e .` / `mamba env create
  -f environment.yml`)"), INSTALL.md:9,29 vs. pyproject.toml:37,39
  (`numpy>=1.26.0`, `pandas>=2.1.0`) and both `environment*.yml`
  (`numpy>=1.26`, `pandas>=2.1`).
- **What:** A user who installs into an existing env with pandas 2.x / numpy
  1.x keeps those versions; nothing upgrades them. The README itself warns
  that the pandas 2→3 copy-on-write change can make code *silently do
  nothing*, so running on unvalidated versions is a real correctness risk.
- **Fix:** raise the floors to `numpy>=2.0`, `pandas>=3.0` in pyproject.toml
  and both environment files. Or, if older versions really are supported,
  correct the docs.

### 5. [risk] Non-editable installs silently lose the C++ binaries
- **Where:** conga/util.py:22 (`path_to_tcrdist_cpp = path_to_conga.parents[0] / 'tcrdist_cpp'`),
  pyproject.toml:128-130 (`package-data` globs `tcrdist_cpp/bin/*` etc.
  under the `conga` package)
- **What:** The binaries are looked up *next to* the package directory, i.e.
  in the repo checkout. The `package-data` globs point at
  `conga/tcrdist_cpp/...`, which doesn't exist, so they match nothing.
  A regular (non-`-e`) `pip install .` therefore installs no binaries and
  conga quietly uses the slow Python path.
- **Fix:** either say clearly that an editable install is *required*, or
  move `tcrdist_cpp/` inside the package (or build it at install time) and
  fix the globs.

### 6. [bug] `pip install conga[legacy-clustering]` would install an unrelated package
- **Where:** INSTALL.md:265
- **What:** As the README notes, `conga` is not on PyPI and the name belongs
  to an unrelated package (`CONGA`). This command would fetch that package.
- **Fix:** `pip install -e ".[legacy-clustering]"`

### 7. [confusing] Environment names differ across docs
- README Quick Install: `conga_env`; INSTALL.md: `conga` (lines 42, 61) and
  `conga-dev`; both `environment.yml` and `environment-minimal.yml`:
  `conga-dev`; tests/README.md: `mamba run -n conga-dev pytest ...`
  throughout. Pick one name.

### 8. [confusing] Python version differs across install routes
- `environment.yml`: 3.13 (changed from 3.14 in master `db1148d`, "due to
  issue with numba build with 3.14.8"); `environment-minimal.yml`: 3.12;
  README quick install (README.md:271): `mamba create -n conga_env
  python=3.14`; INSTALL.md: 3.12; Dockerfile: 3.12; `setup_dev_env.sh:17`
  sets `PYTHON_VERSION="3.12"` but never uses it.
- The docs now contradict the default dev environment: README.md:262 and
  INSTALL.md:9,139-140 say the suite "passes on 3.12 and 3.14" and that 3.13
  "has not been directly tested", yet 3.13 is now what `environment.yml`
  installs, and the README quick install still recommends 3.14, which hit
  the numba build problem. Suggest: pick one recommended version (3.12 or
  3.13), use it in every route, and record which versions the suite was
  actually run on.
- CHANGELOG.md ("Python 3.14 Support") also still says "`environment.yml`
  now installs 3.14 by default for development".
- `environment-minimal.yml:29` includes `louvain`, while `environment.yml`
  has a long comment explaining that louvain is "intentionally NOT
  included". (That works only because the minimal file pins 3.12, but the
  two files contradict each other.)

### 9. [confusing] Stale `setup.py` next to `pyproject.toml`
- **Where:** setup.py
- **What:** It declares `version='0.1.2'` and `install_requires=["scanpy", "leidenalg"]`,
  while pyproject.toml declares 0.2.0 and the real dependency list.
  pyproject.toml wins, but readers can't tell which file is authoritative.
- **Fix:** delete `setup.py`.

### 10. [confusing] `conga.egg-info/` is tracked in git, and is stale
- **What:** Build output is committed. The committed copy still lists
  `louvain` as a core dependency and has `[batch]`/`bbknn` and `[scvi]`
  extras that no longer exist. Every `pip install -e .` rewrites it, so the
  working tree shows as modified right after install.
- **Fix:** `git rm -r --cached conga.egg-info` and add `*.egg-info/` to
  `.gitignore` (it's already in `.dockerignore`).

### 11. [confusing] Stale leftovers from the louvain/bbknn removal
- Dockerfile:13,20 installs `cmake`/`flex`/`bison` "for louvain", but the
  image installs `.[performance]`, which no longer pulls in louvain.
- setup_dev_env.sh:174 checks for `bbknn`, which is no longer an extra.
- setup_dev_env.sh:139 checks only 3 of the 7 binaries the Makefile builds.

---

## Code

### 12. [risk] `convert_svg_to_png` "errors" exit with status 0, and failures are non-fatal by default
- **Where:** conga/convert_svg_to_png.py:155-192
- **What:**
  - On a missing SVG (line 169) and on conversion failure with
    `allow_failure=False` (line 192), it calls bare `exit()`, which exits
    with status **0** (success). Scripts and pipelines therefore can't
    detect the failure.
  - The default is `allow_failure=True`, so a failed conversion only prints
    a message and continues, with no PNG produced.
- **Fix:** `sys.exit(1)`, or better, `raise RuntimeError(...)`. Consider
  making failure fatal by default.
- Minor: the override env var is named `CONGA_PNG_TO_SVG_UTILITY` (line
  172) but controls SVG→PNG conversion. The name is backwards.
- Related: plotting.py (e.g. lines 1330-1333, 1363-1365, 1419-1424) writes a
  logo SVG, converts it to `<prefix>.tmp_*.png`, then reads it back with
  `mpimg.imread`. With the default `allow_failure=True`, a failed conversion
  leads either to a `FileNotFoundError` far from the cause, or, if an earlier
  crashed run left that tmp PNG behind, to a *stale logo from the previous
  run* being silently embedded in the figure.

### 13. [confusing] `import conga` takes ~10 s; half of it is parsing the gene database
- **Where:** `conga.tcrdist.all_genes` (pulled in at import time via
  `conga.preprocess` → `conga.tcr_scoring` → `conga.tcrdist`)
- **What:** `python -X importtime -c "import conga"` on rhino03 (NFS home,
  1 core) shows `conga.tcrdist.all_genes` at ~5.5 s self time (eagerly
  parsing the combined XCR database for every organism) and scanpy at ~4.2 s.
  This also makes `tests/test_portable_imports.py::test_import_performance_reasonable`
  fail (asserts < 5 s).
- **Suggestion:** load the gene tables lazily, per organism, on first use.
  Also make the test threshold environment-tolerant or mark it `slow`.

### 14. [risk] `plt.figure` without `plt.close` (figure leak)
- **Where:** plotting.py:2445 (matplotlib warns "More than 20 figures have
  been opened" during a `--all` run)
- **What:** Memory grows over a long run; mostly harmless at 5K cells.

### 15. [risk] TCR-clumping and graph-vs-graph null-model statistics are not seeded (results change run to run)
- **Where:** conga/correlations.py:330 (`p = np.random.permutation(N)`, numpy's
  global RNG, for the shuffled graph-overlap null model behind
  `_graph_vs_graph_stats.tsv`); conga/tcrdist/tcr_sampler.py:608-631
  (Python's unseeded `random.choice`/`random.shuffle`, used to generate the
  background TCRs that TCR clumping p-values are computed against).
- **What:** Two identical `run_conga.py --all` runs on the bundled 5K PBMC
  data (2026-10-16) gave an identical `_graph_vs_graph.tsv` (CoNGA scores),
  but `_tcr_clumping.tsv` differed: same hit clonotypes and neighbor counts,
  but `expected_num_nbrs`, `pvalue_adj`, and `clonotype_fdr_value` changed by
  up to ~2x (e.g. 0.0056 vs 0.0027). One hit cluster also changed size (7 vs 8
  clonotypes in `_tcr_clumping_logos.png`). The overlap z-scores in
  `_graph_vs_graph_stats.tsv` also changed (14.99 vs 12.04). Commit `0ca6252`
  ("Make pipeline stochastic steps actually seeded") says the pipeline is now
  deterministic, but its reproducibility tests evidently don't cover these
  outputs.
- **Fix:** route `--random_seed` into a `np.random.default_rng(seed)` passed
  to these functions, and use a seeded `random.Random(seed)` instance in
  tcr_sampler instead of the module-level `random` functions. Extend the
  reproducibility tests to compare `_tcr_clumping.tsv` and
  `_graph_vs_graph_stats.tsv`. Borderline TCR-clumping hits near the FDR
  threshold can currently appear or disappear between runs.
- **Same issue in the TCRdist tree figures:** plotting.py calls
  `make_tcr_tree_svg_commands` without a `random_seed`, so any cluster with
  more than 400 TCRs is subsampled with `random.Random(None)` (OS entropy)
  for its tree (make_tcr_trees.py, `max_tcrs_for_trees`). In the 5K PBMC run,
  GEX cluster 0 (408 clonotypes) draws a different tree every run.
  `make_tcrdist_trees` / `make_tcrdist_tree_for_conga_score_threshold`
  also use the unseeded module-level `random.shuffle` when a cluster has more
  than 5000 TCRs.

### 16. [confusing] Inconsistent font metrics tables in the SVG code
- **Where:** conga/svg_basic.py and conga/tcrdist/tcrdist_svg_basic.py
  (`all_text_width = {20:12.5, 100:60.}`, `all_text_height = {20:15.5, 100:75.}`)
- **What:** The size-20 entries imply a character 0.625 em wide and 0.775 em
  tall; the size-100 entries imply 0.60 x 0.75. Only the size-100 values are
  used by the logos (`text_in_box`); the size-20 values are used only by
  `color_stack`. Not a bug in current figures, but confusing. (These two
  files are also near-duplicates of each other.)

---

## Test suite

Run on 2026-10-16, rhino03, Python 3.12.15, `conga_env` built as in entry 1
(`pip install --only-binary :all: -e ".[performance,dev]"`), C++ binaries
built: `pytest tests/ -m "not slow"` → **480 passed, 13 failed, 3 errors,
7 skipped, 22 deselected** (15.5 min). None of the failures indicate a broken
install; each is explained below. After `pip install harmonypy`,
`test_batch_integration.py` + `test_run_conga_cli.py` → 46 passed.

### 17. [bug] Harmony tests fail instead of skipping when `harmonypy` is absent
- **Where:** tests/test_batch_integration.py (11 tests in
  `TestHarmonyIntegrationEndToEnd`, `TestHarmonyIntegrationConfigUns`,
  `TestHarmonyIntegrationH5adRoundTrip`, `TestClusterAndTsneAndUmapReusesIntegratedPcaGex`)
- **What:** harmonypy is an *optional* extra (`[batch-integration]`), but these
  tests assume it's installed (docstring ~line 646: "expected to already be
  installed in the `conga-dev` env"). `test_run_conga_cli.py:220` does this
  correctly with a skip.
- **Fix:** `pytest.importorskip("harmonypy")` in those classes.

### 18. [bug] Cross-process reproducibility test hardcodes a developer's Mac path
- **Where:** tests/test_vectorized_b1_1.py:370-377
- **What:** It runs `mamba run -n conga-dev python /tmp/test_embedding_reproducibility.py`
  with `cwd='/Users/sschattg/conga-dev'`, so it fails with
  `FileNotFoundError` on any other machine. (It's meant to `skip` on
  subprocess failure, but the bad `cwd` raises before that check.) It also
  assumes an env named `conga-dev` and writes a fixed path in `/tmp`.
- **Fix:** use `[sys.executable, script_path]` with pytest's `tmp_path`, no `cwd`.
- **Verified by hand:** the property it checks *does* hold. Running the same
  embedding script in 3 separate processes with `conga_env` python gave an
  identical SHA-256 each time (`755f8e73…`).

### 19. [bug] Performance fixtures used by tests are not in the repo
- **Where:** tests/test_config.py:88-92 (`performance_adata` fixture),
  used by tests/test_comprehensive_validation.py:134 → 3 errors
  (`performance_{small,medium,large}_adata.h5ad` not found).
- **What:** These files were never committed. The only way to make them,
  `python tests/fixtures/test_data_generator.py`, *also overwrites the
  tracked fixtures* (`minimal_adata.h5ad`, `minimal_clones.tsv`,
  `edge_case_clones.tsv`, `invalid_gene_clones.tsv`), so it's unsafe to run.
- **Fix:** generate them in a session-scoped fixture under `tmp_path_factory`
  (deterministic seed), or skip with a clear message when absent.

### 20. [risk] `scripts/test_modernization_results.py` references inputs that don't exist, and silently drops them
- **Where:** scripts/test_modernization_results.py:114, 129, 144
  (`'input_files': ['test_clones.tsv', 'test_gex.h5ad']`) and lines 206-213
- **What:** Neither file exists in `test_data/`. The loop only adds
  `--clones_file`/`--gex_data` `if input_path.exists()`, so missing inputs
  are skipped without any error and `run_conga.py` gets invoked without its
  input data. Any "baseline vs. current" comparison built on this can't be
  testing what it claims.
- **Fix:** point at real files (e.g. `classic_clones.tsv` + the
  `SC5v2_humanPBMCs_*` `.h5`, gex_data_type `10x_h5`), and `assert` that
  every input exists.

### 21. [confusing] Test and example docs assume specific personal environments
- tests/README.md uses `mamba run -n conga-dev ...` throughout (see entry 7);
  tests/README_error_conditions_I2_2.md:129 uses `cd /Users/sschattg/conga-dev/tests`.
- examples/setup_all.bash and examples/run_all.bash hardcode `PYTHON=` paths
  in old personal conda envs (`/home/pbradley/anaconda2/envs/scanpy_new`,
  `/home/pbradley/miniconda3/envs/scanpy_2022`) and
  `CONGA_REPO=/home/pbradley/gitrepos/conga/`. These are flagged "change
  this", but they could default to `python` on PATH and a path relative to
  the script.

---

## Docs and notebooks

### 22. [confusing] Which GitHub repository is canonical? (Colab notebook, README links)
- **Where:** colab_conga_pipeline.ipynb (cell 7: `!git clone
  https://github.com/phbradley/conga.git`; cells 4-5: "Install a tool for
  converting from SVG to PNG files" / `!pip install cairosvg`); README.md
  "Open in Colab" badge (points at phbradley/conga) vs. the Development
  Installation instructions (now `git clone https://github.com/sschattgen/conga.git`).
- **What:** The notebook installs and runs whatever is on `phbradley/conga`
  master, so changes merged into `sschattgen/conga` (including the
  converter-free figures from the `svg_free` branch) don't reach Colab users.
  The cairosvg install cell was deliberately left in place on `svg_free` for
  that reason: removing it while the notebook still clones phbradley/conga
  would break the notebook.
- **Suggestion:** decide which repository is canonical, point the README,
  badge and notebook at it, and once the svg_free changes are there, drop
  notebook cells 4-5.

---

## Clustering

### 23. [risk] Leiden clustering changed in `0ca6252`: labels no longer ordered by size, and partitions shifted
- **Where:** conga/preprocess.py `cluster_and_tsne_and_umap` (~line 1312) and
  `calc_tcrdist_nbrs_umap_clusters_cpp` (~line 3191); also devel.py.
- **What:** Commit `0ca6252` ("Make pipeline stochastic steps actually
  seeded") switched every `sc.tl.leiden` call from scanpy's default
  (`leidenalg`, run to convergence on a directed graph) to `flavor='igraph',
  n_iterations=2, directed=False`. Two consequences, measured on the 5K PBMC
  run by rebuilding the exact neighbor graphs (the current call reproduces
  the stored clusters exactly):
  - **Labels are no longer ordered by cluster size.** `leidenalg` renumbers
    clusters "in decreasing size"; igraph does not, and neither scanpy nor
    conga re-sorts. GEX cluster sizes in label order are now 408, **6**, 391,
    24, 16, 174, 246, 72, 29, 24, versus 407, 407, 255, 167, 72, 28, 24, 16, 7,
    **7** with the old default. The 6-clonotype cluster (myeloid signature:
    S100A8/9, LYZ, CST3, FCN1, so likely doublets or contamination) is found
    by both methods; it used to be labeled last.
  - **The partitions also changed.** GEX: adjusted Rand index (ARI) 0.96
    versus the old default. TCR: ARI 0.82 (13 vs 12 clusters, with several
    clusters split or recombined). Modularity is about the same for both
    (0.6375 vs 0.6380 GEX; 0.7129 vs 0.7139 TCR), so neither is wrong, but
    cluster-dependent outputs (hit grouping in the logo figures, GEX-cluster
    DEGs, per-cluster trees, cluster names and colors) no longer match runs
    made before 2026-10-07. This isn't mentioned in the CHANGELOG or docs.
- **Fix:** relabel clusters by decreasing size after clustering (branch
  `sort_clusters_by_size`, separate PR), and decide whether to keep the
  igraph method (faster; scanpy's announced future default) or return to
  `leidenalg`. Either way, note the change in the CHANGELOG.
