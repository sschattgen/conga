# Debug script to understand the TCR representation issue
import sys
sys.path.insert(0, '/Users/sschattg/conga-dev')
import conga
import pandas as pd

print("Loading clones file...")
clones_df = pd.read_csv('test_data/test_run_clones.tsv', sep='\t')
print(f"Clones file loaded: {len(clones_df)} clones")
print(f"Columns: {list(clones_df.columns)}")

print("\nTesting representation resolution...")
from conga.preprocess import resolve_tcr_representation
rep = resolve_tcr_representation(
    organism='human',
    num_obs=len(clones_df),
    request_kpca=False,
    request_exact_nbrs=False,
    kpca_reduction_limit=20000,
    stored_obsm_keys=set()
)
print(f'  Active: {rep.active}')
print(f'  Build vectorized: {rep.build_vectorized}')
print(f'  Build kpca: {rep.build_kpca}')
print(f'  Reason: {rep.reason}')

# Test vectorized encoding on sample data
print("\nTesting vectorized encoding on sample...")
sample_clones = clones_df.head(10)
print("Sample TCR data:")
for i, row in sample_clones.iterrows():
    print(f"  {row['va_gene']} | {row['cdr3a']} | {row['vb_gene']} | {row['cdr3b']}")
    break  # Just show one

print("\nTesting encoding function...")
try:
    from conga.tcrdist.vectorized import encode_tcrs, EncodingConfig
    config = EncodingConfig()
    
    # Convert to tuple format
    tcrs = []
    for _, row in sample_clones.iterrows():
        alpha = (row['va_gene'], row.get('ja_gene', ''), row['cdr3a'], '')
        beta = (row['vb_gene'], row.get('jb_gene', ''), row['cdr3b'], '')
        tcrs.append((alpha, beta))
    
    print(f"Encoding {len(tcrs)} clones...")
    vectors = encode_tcrs(tcrs, 'human', config)
    print(f"Success! Encoded to shape: {vectors.shape}")
    print(f"Vector dtype: {vectors.dtype}")
    print(f"First vector (first 10 values): {vectors[0][:10]}")
    
except Exception as e:
    print(f"Error in encoding: {e}")
    import traceback
    traceback.print_exc()

