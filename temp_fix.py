import re

# Read the file
with open('scripts/run_conga.py', 'r') as f:
    content = f.read()

# Find and replace the allow_missing_kpca_file logic
old_pattern = r'''    allow_missing_kpca_file = \(
        args\.use_exact_tcrdist_nbrs and
        args\.use_tcrdist_umap and
        args\.use_tcrdist_clusters
        \)'''

new_pattern = '''    allow_missing_kpca_file = (
        # Allow missing KernelPCA file for vectorized path (supported organisms)
        (args.organism in {"human", "mouse", "rhesus"} and not args.use_kpca_tcrdist) or
        # Original logic for exact path
        (args.use_exact_tcrdist_nbrs and
         args.use_tcrdist_umap and
         args.use_tcrdist_clusters)
        )'''

content = re.sub(old_pattern, new_pattern, content, flags=re.MULTILINE)

# Write back
with open('scripts/run_conga.py', 'w') as f:
    f.write(content)

print("Fixed allow_missing_kpca_file logic")
