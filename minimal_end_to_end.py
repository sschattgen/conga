# Minimal end-to-end test to verify vectorized TCR representation
import sys
sys.path.insert(0, '/Users/sschattg/conga-dev')
import conga
import scanpy as sc
sc.settings.verbosity = 1  # Reduce scanpy output

print("Running minimal end-to-end test...")

# 1. Load data
adata = conga.preprocess.read_dataset(
    gex_data='test_data/SC5v2_humanPBMCs_5Kcells_Connect_single_channel_SC5v2_humanPBMCs_5Kcells_Connect_single_channel_count_sample_feature_bc_matrix.h5',
    gex_data_type='10x_h5',
    clones_file='test_data/test_run_clones.tsv',
    allow_missing_kpca_file=True
)

# 2. Set organism (this is what run_conga.py should do)
adata.uns['organism'] = 'human'
print(f"Data loaded: {adata.shape}, organism: {adata.uns['organism']}")

# 3. Resolve representation
tcr_representation = conga.preprocess.resolve_tcr_representation(
    organism=adata.uns['organism'],
    num_obs=adata.shape[0],
    request_kpca=False,
    request_exact_nbrs=False,
    kpca_reduction_limit=20000,
    stored_obsm_keys=list(adata.obsm.keys())
)
print(f"Representation: {tcr_representation.active} ({tcr_representation.reason})")

# 4. Build vectorized if needed
if tcr_representation.build_vectorized:
    from conga.tcrdist.vectorized import EncodingConfig
    encoding_config = EncodingConfig()
    vectors = conga.preprocess.store_tcr_vectors_in_adata(adata, encoding_config)
    print(f"Built vectorized: {vectors.shape}")

# 5. Record active representation
conga.preprocess.record_active_tcr_representation(adata, tcr_representation.active)

# 6. Test that get_active_tcr_representation works
active = conga.preprocess.get_active_tcr_representation(adata)
print(f"Active representation: {active}")
print(f"Available obsm keys: {list(adata.obsm.keys())}")

# 7. Try to call the function that was failing in the full test
try:
    print("\nTesting cluster_and_tsne_and_umap...")
    # Do minimal preprocessing first
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
    
    # Now try the clustering function
    conga.preprocess.cluster_and_tsne_and_umap(
        adata,
        n_neighbors=10,
        n_gex_pcs=20,
        recompute_pca_gex=True,
        make_1d_umaps=True
    )
    print("✓ cluster_and_tsne_and_umap succeeded!")
    print(f"Final obsm keys: {list(adata.obsm.keys())}")
    
except Exception as e:
    print(f"✗ cluster_and_tsne_and_umap failed: {e}")
    import traceback
    traceback.print_exc()

print("Test complete.")
