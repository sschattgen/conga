# Test TCR clumping analysis with vectorized representation
import sys
sys.path.insert(0, '/Users/sschattg/conga-dev')
import conga
import scanpy as sc
sc.settings.verbosity = 1

print("Testing TCR clumping analysis with vectorized representation...")

# Load and prepare data (same as before)
adata = conga.preprocess.read_dataset(
    gex_data='test_data/SC5v2_humanPBMCs_5Kcells_Connect_single_channel_SC5v2_humanPBMCs_5Kcells_Connect_single_channel_count_sample_feature_bc_matrix.h5',
    gex_data_type='10x_h5',
    clones_file='test_data/test_run_clones.tsv',
    allow_missing_kpca_file=True
)

# Set organism and build vectorized representation
adata.uns['organism'] = 'human'
print(f"Data loaded: {adata.shape}, organism: {adata.uns['organism']}")

# Resolve and build vectorized representation
tcr_representation = conga.preprocess.resolve_tcr_representation(
    organism=adata.uns['organism'],
    num_obs=adata.shape[0],
    request_kpca=False,
    request_exact_nbrs=False,
    kpca_reduction_limit=20000,
    stored_obsm_keys=list(adata.obsm.keys())
)

if tcr_representation.build_vectorized:
    from conga.tcrdist.vectorized import EncodingConfig
    encoding_config = EncodingConfig()
    vectors = conga.preprocess.store_tcr_vectors_in_adata(adata, encoding_config)
    print(f"Built vectorized representation: {vectors.shape}")

conga.preprocess.record_active_tcr_representation(adata, tcr_representation.active)
active = conga.preprocess.get_active_tcr_representation(adata)
print(f"Active TCR representation: {active}")

# Do basic preprocessing
print("\nPreprocessing data...")
sc.pp.filter_cells(adata, min_genes=100)
sc.pp.filter_genes(adata, min_cells=3)  
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata.raw = adata
adata = adata[:, adata.var.highly_variable]
print(f"After preprocessing: {adata.shape}")

# Run clustering and UMAP
print("\nRunning clustering and UMAP...")
conga.preprocess.cluster_and_tsne_and_umap(
    adata,
    n_neighbors=10,
    n_gex_pcs=20,
    recompute_pca_gex=True,
    make_1d_umaps=True
)
print(f"Available obsm keys: {list(adata.obsm.keys())}")

# Now test TCR clumping analysis
print("\nRunning TCR clumping analysis...")
try:
    # Run TCR clumping analysis (organism comes from adata.uns)
    results_df = conga.tcr_clumping.assess_tcr_clumping(
        adata,
        pvalue_threshold=0.01,
        num_random_samples=1000,  # Reduced for faster testing  
        verbose=True
    )
    
    print("✓ TCR clumping analysis completed successfully!")
    print(f"Results DataFrame shape: {results_df.shape}")
    print(f"Results columns: {list(results_df.columns)}")
    
    if len(results_df) > 0:
        significant = len(results_df[results_df['pvalue_adj'] < 0.01])
        print(f"Found {significant} significant clumps out of {len(results_df)} tested")
        
        if significant > 0:
            print("Top significant clumps:")
            top_clumps = results_df[results_df['pvalue_adj'] < 0.01].head(3)
            for _, row in top_clumps.iterrows():
                print(f"  Clone {row['clone_index']}: radius={row['nbr_radius']}, "
                      f"pval={row['pvalue_adj']:.3e}, neighbors={row['num_nbrs']}")
    
    # Check what was stored in adata
    if 'tcr_clumping' in adata.uns:
        clumping_results = adata.uns['tcr_clumping']
        print(f"Stored TCR clumping results keys: {list(clumping_results.keys())}")
    
    # Check for TCR clumping annotations in obs
    tcr_clump_cols = [col for col in adata.obs.columns if 'clump' in col.lower()]
    if tcr_clump_cols:
        print(f"TCR clumping columns in obs: {tcr_clump_cols}")
        for col in tcr_clump_cols[:2]:  # Show first 2
            n_clumped = sum(adata.obs[col] != '')
            print(f"  {col}: {n_clumped} cells in clumps")
    else:
        print("No TCR clumping columns found in obs")
    
except Exception as e:
    print(f"✗ TCR clumping analysis failed: {e}")
    import traceback
    traceback.print_exc()

print("\nTCR clumping test with vectorized representation complete.")
