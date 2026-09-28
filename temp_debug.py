import re

# Add debug output to understand what's happening with the representation
with open('scripts/run_conga.py.debug', 'r') as f:
    content = f.read()

# Add debug output after the representation is resolved
debug_insert = '''
# DEBUG: Check what representation was resolved
print(f"DEBUG: TCR representation resolved: {tcr_representation.active}")
print(f"DEBUG: Build vectorized: {tcr_representation.build_vectorized}")
print(f"DEBUG: Build kpca: {tcr_representation.build_kpca}")
print(f"DEBUG: Reason: {tcr_representation.reason}")
'''

# Insert after the representation is resolved
content = re.sub(
    r"(tcr_representation = conga\.preprocess\.resolve_tcr_representation.*?\n)",
    r"\1" + debug_insert,
    content, 
    flags=re.DOTALL
)

# Add debug output after vectorized representation should be built
debug_insert2 = '''
    
    # DEBUG: Check if vectorized representation was built
    print(f"DEBUG: After building vectorized, obsm keys: {list(adata.obsm.keys())}")
    print(f"DEBUG: UNS active rep key present: {util.UNS_KEY_ACTIVE_TCR_REP in adata.uns}")
    if util.UNS_KEY_ACTIVE_TCR_REP in adata.uns:
        print(f"DEBUG: Active rep from uns: {adata.uns[util.UNS_KEY_ACTIVE_TCR_REP]}")
    active_rep = conga.preprocess.get_active_tcr_representation(adata)
    print(f"DEBUG: get_active_tcr_representation returned: {active_rep}")
'''

content = re.sub(
    r"(conga\.preprocess\.record_active_tcr_representation.*?\n)",
    r"\1" + debug_insert2,
    content,
    flags=re.DOTALL
)

with open('scripts/run_conga.py.debug', 'w') as f:
    f.write(content)

print("Added debug output")
