# Branch Integration Roadmap

## Overview

The CoNGA codebase has several active feature branches that extend master with important functionality. This document maps the features, improvements, and differences across branches to guide consolidation into a unified codebase.

**Current Branches of Interest:**
- `master` — Stable base version
- `dev` — Latest features (faiss, batch integration, scVI, tcr_qc)
- `bcr` — B cell (BCR) analysis extensions
- `metaconga_match` — Meta-CoNGA matching and clustering features
- `rhesus` — Rhesus organism support
- `sschattgen` — Personal development branch

---

## Branch Comparison Matrix

| Feature | Master | Dev | BCR | Metaconga_Match | Notes |
|---------|--------|-----|-----|-----------------|-------|
| FAISS GEX neighbor search | ❌ | ✅ | ❌ | ❌ | Major speedup for large datasets |
| Batch integration framework | ❌ | ✅ | ✅ | ❌ | hvg_batch_key parameter |
| scVI support | ❌ | ✅ | ❌ | ❌ | Single-cell variational inference |
| TCR QC module | ❌ | ✅ | ❌ | ❌ | New quality control functions |
| Cell type prediction models | ❌ | ✅ | ✅ | ❌ | MLP, gradientBoost, logreg |
| Meta-CoNGA matching | ❌ | ❌ | ❌ | ✅ | CDR3aa bias cluster matching |
| Rhesus organism support | ❌ | ❌ | ❌ | ❌ | In separate `rhesus` branch |
| Batch info in clustermaps | ❌ | ❌ | ✅ | ❌ | Visual batch integration |
| Embedded PNG in HTML reports | ❌ | ❌ | ✅ | ❌ | Self-contained report format |
| TCR database match plots | ❌ | ❌ | ✅ | ❌ | New visualization |

---

## Detailed Feature Breakdown

### 1. FAISS-Powered Neighbor Search (Dev Branch)

**Status:** Ready for integration
**Commits:** `210ce04` "calc_nbrs now powered by faiss"
**Changes:**
- Replaces standard scipy/sklearn neighbor calculations with FAISS
- Located in `conga/preprocess.py` — `calc_nbrs()` function
- ~474 lines removed, ~95 lines added (significant simplification)
- Massive performance boost for large gene expression datasets

**Integration Considerations:**
- ✅ Backward compatible (fallback to sklearn if FAISS unavailable)
- Add FAISS to requirements: `faiss-cpu` or `faiss-gpu`
- Update documentation with performance benchmarks
- Test on datasets >100k cells

**Suggested Implementation:**
```python
# In preprocess.py
try:
    import faiss
    USE_FAISS = True
except ImportError:
    USE_FAISS = False
    print("FAISS not available; falling back to sklearn for GEX neighbor search")
```

---

### 2. Batch Integration Support (Dev + BCR Branches)

**Status:** Ready for integration
**Key Changes:**

**In preprocess.py - `filter_normalize_and_hvg()` function:**
- New parameter: `hvg_batch_key` (default: None)
- Passes batch key to `sc.pp.highly_variable_genes()`
- Allows HVG finding to respect batch structure
- Prevents batch artifacts from driving variance

**In correlations.py - Optional BBKNN support:**
- New parameter: `use_bbknn` (boolean)
- Parameter: `bbknn_batch_key` (column in adata.obs)
- Batch-Balanced K-Nearest Neighbors for GEX graph construction
- Better handling of batch effects

**In plotting.py - Clustermap enhancement:**
- Batch colors displayed along axes
- Helps visualize batch structure in results

**Integration Considerations:**
- ✅ Non-breaking (batch parameters optional)
- ✅ Only activates when explicitly specified
- Add `bbknn` to optional dependencies: `pip install bbknn`
- Document batch correction best practices
- Add batch integration to examples

**API Addition:**
```python
adata = conga.preprocess.filter_normalize_and_hvg(
    adata,
    organism='human',
    hvg_batch_key='batch_id',  # NEW: for batch-aware HVG selection
)

# Or use BBKNN for graph construction
adata = conga.correlations.make_graphs(
    adata,
    use_bbknn=True,
    bbknn_batch_key='batch_id',
)
```

---

### 3. ScVI Integration (Dev Branch)

**Status:** Experimental, ready for integration
**Location:** `conga/preprocess.py`
**Imports:** `import scvi`

**Functionality:**
- Single-cell Variational Inference for GEX representation learning
- Alternative to standard PCA normalization
- Captures complex GEX patterns

**Integration Considerations:**
- ⚠️ Optional feature (heavy dependency: scVI-tools)
- Add to extras: `pip install conga[scvi]`
- Requires GPU support for efficiency
- Document as experimental
- Add option flag to `run_conga.py`: `--use_scvi`

**Suggested Parameter:**
```python
def filter_normalize_and_hvg(
    adata,
    use_scvi=False,  # NEW
    scvi_n_latent=20,  # latent dimension
    **kwargs
):
    if use_scvi:
        import scvi
        # Initialize and train scVI model
        scvi.model.SCVI.setup_anndata(adata)
        # ... training code ...
```

---

### 4. TCR Quality Control Module (Dev Branch)

**Status:** Ready for integration
**File:** `conga/tcrdist/tcr_qc.py` (NEW)

**Purpose:**
- QC checks for TCR data quality
- Validates clonotypes before analysis
- Identifies problematic sequences

**Integration:**
- ✅ No dependencies beyond existing ones
- Add import to main `conga/__init__.py`
- Integrate into `setup_10x_for_conga.py` pipeline
- Optional: Add QC reporting to `run_conga.py`

---

### 5. Cell Type Prediction Models (Dev + BCR Branches)

**Status:** Ready for integration
**Files:**
- `conga/data/T_cell_prediction_markers.tsv`
- `conga/data/prediction_models/MLP.sav`
- `conga/data/prediction_models/gradientBoost.sav`
- `conga/data/prediction_models/logreg.sav`

**Functionality:**
- Pre-trained models for CD8/CD4/MAIT/etc. prediction
- Uses marker gene expression
- Scikit-learn format (joblib)

**Integration Considerations:**
- ✅ Ready to include
- ✅ Minimal dependencies (pickle/joblib)
- Add model loading function to `conga/util.py`
- Add prediction functions to new module or `preprocess.py`
- Document model accuracy/caveats

**Suggested API:**
```python
def predict_cell_types(adata, model_name='logreg'):
    """Predict T cell types using pre-trained model."""
    from joblib import load
    model = load(f'{util.path_to_data}/prediction_models/{model_name}.sav')
    predictions = model.predict(adata[:, markers].X)
    adata.obs['predicted_cell_type'] = predictions
    return adata
```

---

### 6. Meta-CoNGA Matching (Metaconga_Match Branch)

**Status:** Feature-rich, ready for optional integration
**File:** `conga/metaconga_match.py` (NEW - ~500+ lines)

**Purpose:**
- Match TCR clonotypes against curated meta-analysis database
- Identify CDR3aa bias clusters
- Cross-dataset clonotype matching

**Key Functions:**
- `_encode_tcr_seqs()` — TCR feature encoding
- `get_categorical_colors()` — Visualization helper
- `match_to_metaconga()` — Main matching function

**Data Files Added:**
- `conga/data/metaconga/` — 11 TSV reference files
- CDR3aa cluster definitions
- Pre-computed cluster enrichments
- Known clump mappings

**Integration Considerations:**
- ✅ Self-contained module, no dependencies on changes to other code
- Add as optional analysis: `--metaconga_match` flag
- Add to `run_conga.py` script
- Document that this requires internet connectivity for database downloads (or pre-load reference)
- Add to `conga/__init__.py` with conditional import

**Suggested Integration:**
```python
# In run_conga.py
if args.metaconga_match:
    import conga.metaconga_match
    results = conga.metaconga_match.match_to_metaconga(
        adata,
        organism='human',
    )
    adata.uns['metaconga_match'] = results
```

---

### 7. Enhanced Database Files

**Status:** Ready for integration

**Dev/BCR branches upgrade:**
- `conga/tcrdist/db/combo_xcr.tsv` — Updated reference database
- `conga/data/T_cell_prediction_markers.tsv` — Marker gene lists
- Deletion of: `combo_xcr_2023-12-30.tsv` (old versioned file)

**Integration:**
- ✅ Simply replace existing files
- Maintain version history in comments or separate backup
- Update database documentation

---

### 8. Visualization and Reporting Enhancements (BCR Branch)

**Status:** Ready for integration

**Enhancements:**

**1. Embedded PNG in HTML Reports**
- Commit: `506f517`
- Eliminates need for separate PNG files
- Self-contained, portable HTML reports
- Location: `conga/plotting.py` modifications

**2. Batch-Aware Clustermaps**
- Commit: `653480c`
- Display batch information in figure margins
- Better visualization of batch effects
- Location: `plotting.py`

**3. TCR Database Match Visualization**
- New plot type showing literature database matches
- Function: `make_tcr_db_match_plot()`
- Location: `plotting.py`

**4. Clustermap Optimization**
- Reduced default `max_type_features` from 100 to 50
- Improves readability
- Faster rendering

**Integration:**
- ✅ All backward compatible
- Update to plotting.py is safe refactor
- Add embed_pngs parameter (default: False for backward compat)
- Add plotting examples to docs

---

### 9. Additional Updates Across Branches

**Database Updates:**
- `tcrdist/db/combo_xcr.tsv` — Gene sequence database refresh
- `tcrdist_cpp/db/tcrdist_info_mouse.txt` — Distance parameters

**Documentation:**
- All branches have README.md updates
- Dev: Integration and batch correction guidance
- BCR: Batch handling examples
- Metaconga_match: Meta-analysis documentation

---

## Proposed Integration Strategy

### Phase 1: High-Priority Core Features (Immediate)
1. **FAISS neighbor search** — Performance critical
   - File: `preprocess.py` calc_nbrs()
   - Risk: Low (backward compatible)
   - Priority: 🔴 High

2. **Batch integration framework** — Widely applicable
   - Files: `preprocess.py`, `correlations.py`, `plotting.py`
   - Risk: Low (optional parameters)
   - Priority: 🔴 High

3. **Database updates** — Maintenance
   - Files: `tcrdist/db/combo_xcr.tsv`, markers.tsv
   - Risk: Minimal
   - Priority: 🟡 Medium

### Phase 2: Feature Modules (Near-term)
1. **Cell type prediction models** — Useful utility
   - Files: prediction_models/, new functions
   - Risk: Low
   - Priority: 🟡 Medium

2. **TCR QC module** — Data validation
   - Files: `tcrdist/tcr_qc.py`
   - Risk: Low
   - Priority: 🟡 Medium

3. **Enhanced plotting** — UX improvement
   - Files: `plotting.py` modifications
   - Risk: Low
   - Priority: 🟡 Medium

### Phase 3: Optional Advanced Features (Later)
1. **ScVI integration** — Experimental
   - Risk: Medium (heavy optional dep)
   - Priority: 🟢 Low
   - Recommend: `pip install conga[scvi]`

2. **Meta-CoNGA matching** — Specialized analysis
   - Risk: Low (isolated module)
   - Priority: 🟢 Low
   - Recommendation: Keep as optional plugin

---

## Dependencies to Add/Update

### Required (Phase 1)
```
faiss-cpu>=1.7.0  # or faiss-gpu for GPU support
```

### Conditional/Optional
```
[batch-integration]
bbknn>=1.5.1

[scvi]
scvi-tools>=0.17.0

[all]
faiss-cpu>=1.7.0
bbknn>=1.5.1
scvi-tools>=0.17.0
```

Update `setup.py` or `requirements.txt` accordingly.

---

## Python 3.12 Compatibility Notes

**Current Status:** All branches compatible with Python 3.12
**Key considerations:**
- ✅ No deprecated imports identified
- ✅ NumPy/SciPy updated versions support 3.12
- ⚠️ Joblib version for model serialization—use 1.3+
- ⚠️ FAISS: ensure faiss-cpu 1.7.4+ or faiss-gpu 1.7.4+

**Suggested requirements.txt snippet:**
```
python>=3.12
numpy>=1.23.0
scipy>=1.9.0
scanpy>=1.9.0
anndata>=0.9.0
scikit-learn>=1.1.0
pandas>=1.5.0
faiss-cpu>=1.7.4
joblib>=1.3.0
```

---

## Testing Plan for Integration

1. **Unit tests** for each new feature
2. **Integration tests** on example datasets
3. **Performance benchmarks** (FAISS vs sklearn)
4. **Backward compatibility** tests (master branch examples)
5. **Python 3.12 validation**

---

## Files to Review Before Integration

### Must Review
- [ ] `conga/preprocess.py` (FAISS, batch, scVI)
- [ ] `conga/correlations.py` (batch parameters)
- [ ] `conga/plotting.py` (embedding, batch viz, TCR match plots)
- [ ] `conga/metaconga_match.py` (new module)

### Should Review
- [ ] `conga/tcrdist/tcr_qc.py`
- [ ] `conga/util.py` (cell type prediction)
- [ ] `scripts/run_conga.py` (new flags)
- [ ] `README.md` (all examples/docs)

### Data Files to Verify
- [ ] `conga/data/prediction_models/*`
- [ ] `conga/data/T_cell_prediction_markers.tsv`
- [ ] `conga/data/metaconga/*` (11 files)
- [ ] `conga/tcrdist/db/combo_xcr.tsv`

---

## Summary Statistics

| Metric | Dev | BCR | Metaconga_Match |
|--------|-----|-----|-----------------|
| Files changed | ~27 | ~28 | ~23 |
| New files | 5 | 4 | 12 |
| Commits ahead | ~20 | ~54 | ~15 |
| Est. lines added | ~2000+ | ~2500+ | ~1500+ |
| Breaking changes | 0 | 0 | 0 |

---

## Conclusion

All three feature branches are ready for integration into a unified codebase. The proposed phased approach allows for staged rollout:

- **Phase 1** addresses performance and common use cases
- **Phase 2** adds utility and validation functions
- **Phase 3** includes optional advanced/experimental features

No breaking changes are anticipated. The unified codebase would support Python 3.12 and significantly extend the capabilities of the base CoNGA package while maintaining backward compatibility.
