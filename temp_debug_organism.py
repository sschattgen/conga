import re

with open('scripts/run_conga.py.original', 'r') as f:
    content = f.read()

# Add debug output before and after organism setting
debug_before = '''
    print(f"DEBUG: About to set organism: args.organism = {args.organism}")
    print(f"DEBUG: adata.uns keys before: {list(adata.uns.keys())}")
'''

debug_after = '''
    print(f"DEBUG: Set organism in adata.uns: {adata.uns['organism']}")  
    print(f"DEBUG: adata.uns keys after: {list(adata.uns.keys())}")
'''

# Insert debug output
content = re.sub(
    r'(    assert args\.organism\n)',
    debug_before + r'\1' + debug_after,
    content
)

with open('scripts/run_conga.py', 'w') as f:
    f.write(content)

print("Added debug output for organism setting")
