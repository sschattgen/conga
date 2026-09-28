# Test the integration step by step
import sys
sys.path.insert(0, '/Users/sschattg/conga-dev')
import conga
import conga.preprocess
import scanpy as sc
import pandas as pd

print("=" * 60)
print("TESTING VECTORIZED TCR INTEGRATION")
print("=" * 60)

# Step 1: Load the data like run_conga.py does
print("\n1. Loading GEX data...")
gex_data = 'test_data/SC5v2_humanPBMCs_5Kcells_Connect_single_channel_SC5v2_humanPBMCs_5Kcells_Connect_single_channel_count_sample_feature_bc_matrix.h5'
clones_file = 'test_data/test_run_clones.tsv'

try:
    adata = conga.preprocess.read_dataset(
        gex_data=gex_data,
        gex_data_type='10x_h5',
        clones_file=clones_file,
        allow_missing_kpca_file=True,  # Allow missing for vectorized path
        gex_only=False
    )
    print(f"Data loaded: {adata.shape}")
    print(f"Organism: {adata.uns.get('organism', 'not set')}")
    print(f"Initial obsm keys: {list(adata.obsm.keys())}")
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit(1)

# Step 2: Resolve TCR representation
print("\n2. Resolving TCR representation...")
tcr_representation = conga.preprocess.resolve_tcr_representation(
    organism="human",
    num_obs=adata.n_obs,
    request_kpca=False,
    request_exact_nbrs=False,
    kpca_reduction_limit=20000,
    stored_obsm_keys=set(adata.obsm.keys())
)
print(f"Resolved representation: {tcr_representation.active}")
print(f"Build vectorized: {tcr_representation.build_vectorized}")
print(f"Build kpca: {tcr_representation.build_kpca}")
print(f"Reason: {tcr_representation.reason}")

# Step 3: Build vectorized representation if needed
print("\n3. Building vectorized representation...")
if tcr_representation.build_vectorized:
    from conga.tcrdist.vectorized import EncodingConfig
    
    encoding_config = EncodingConfig()  # Default config
    print(f"Using encoding config: {encoding_config}")
    
    try:
        print("Calling store_tcr_vectors_in_adata...")
        vectors = conga.preprocess.store_tcr_vectors_in_adata(adata, encoding_config)
        print(f"Vectorized representation built: {vectors.shape}")
        print(f"Updated obsm keys: {list(adata.obsm.keys())}")
    except Exception as e:
        print(f"Error building vectorized representation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
else:
    print("Not building vectorized representation")

# Step 4: Record active representation
print("\n4. Recording active representation...")
try:
    conga.preprocess.record_active_tcr_representation(adata, tcr_representation.active)
    print(f"Recorded active representation: {tcr_representation.active}")
    print(f"UNS keys now: {list(adata.uns.keys())}")
except Exception as e:
    print(f"Error recording active representation: {e}")
    import traceback
    traceback.print_exc()

# Step 5: Test get_active_tcr_representation
print("\n5. Testing get_active_tcr_representation...")
try:
    active_rep = conga.preprocess.get_active_tcr_representation(adata)
    print(f"get_active_tcr_representation returned: {active_rep}")
    print(f"Expected: {tcr_representation.active}")
    print(f"Match: {active_rep == tcr_representation.active}")
except Exception as e:
    print(f"Error getting active representation: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("INTEGRATION TEST COMPLETE")
print("=" * 60)

