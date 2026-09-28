"""
Test fixtures and data generation for vectorized TCRdist + FAISS testing.

This module provides comprehensive test data generation for all aspects of the
vectorized TCRdist and FAISS acceleration system, including edge cases and
performance scenarios.
"""

import numpy as np
import pandas as pd
import anndata as ad
import tempfile
import os
from typing import Tuple, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TestDataGenerator:
    """Generate test data for vectorized TCRdist + FAISS testing."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize with deterministic random seed."""
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def create_minimal_tcr_data(self, n_cells: int = 100) -> Tuple[pd.DataFrame, List]:
        """Create minimal but valid TCR data for basic testing.
        
        Returns
        -------
        clones_df : pd.DataFrame
            Clones file format with minimal required columns
        tcrs : list
            TCR tuples in CoNGA format
        """
        # Use realistic human TCR V/J genes
        va_genes = ['TRAV1-1*01', 'TRAV1-2*01', 'TRAV2*01', 'TRAV3*01', 'TRAV4*01']
        ja_genes = ['TRAJ1*01', 'TRAJ2*01', 'TRAJ3*01', 'TRAJ4*01', 'TRAJ5*01']
        vb_genes = ['TRBV1*01', 'TRBV2*01', 'TRBV3-1*01', 'TRBV4-1*01', 'TRBV5-1*01']
        jb_genes = ['TRBJ1-1*01', 'TRBJ1-2*01', 'TRBJ2-1*01', 'TRBJ2-2*01', 'TRBJ2-3*01']
        
        # Generate realistic CDR3 sequences
        aa_chars = 'ACDEFGHIKLMNPQRSTVWY'  # Standard amino acids
        
        clones_data = []
        tcrs = []
        
        for i in range(n_cells):
            # Create cell barcode
            barcode = f"CELL{i:06d}"
            
            # Random V/J selection
            va = np.random.choice(va_genes)
            ja = np.random.choice(ja_genes)
            vb = np.random.choice(vb_genes)
            jb = np.random.choice(jb_genes)
            
            # Generate realistic CDR3 sequences (8-20 amino acids)
            cdr3a_len = np.random.randint(8, 21)
            cdr3b_len = np.random.randint(8, 21)
            
            cdr3a = 'C' + ''.join(np.random.choice(list(aa_chars), cdr3a_len-2)) + 'F'
            cdr3b = 'C' + ''.join(np.random.choice(list(aa_chars), cdr3b_len-2)) + 'F'
            
            # Create clones entry
            clones_data.append({
                'cell_barcode': barcode,
                'clonotype_id': f'clonotype_{i}',  # Each cell is its own clonotype for simplicity
                'va': va,
                'ja': ja,
                'cdr3a': cdr3a,
                'cdr3a_nucseq': 'N' * (cdr3a_len * 3),  # Dummy nucleotide
                'vb': vb,
                'jb': jb,
                'cdr3b': cdr3b,
                'cdr3b_nucseq': 'N' * (cdr3b_len * 3),  # Dummy nucleotide
            })
            
            # Create TCR tuple
            tcr = ((va, ja, cdr3a, 'N' * (cdr3a_len * 3)), 
                   (vb, jb, cdr3b, 'N' * (cdr3b_len * 3)))
            tcrs.append(tcr)
        
        clones_df = pd.DataFrame(clones_data)
        return clones_df, tcrs
    
    def create_adata_with_tcrs(self, n_cells: int = 100, n_genes: int = 2000) -> ad.AnnData:
        """Create AnnData object with both GEX and TCR data."""
        # Create gene expression data
        X = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes)).astype(np.float32)
        
        # Create gene names
        gene_names = [f"GENE{i:04d}" for i in range(n_genes)]
        
        # Create cell barcodes
        cell_barcodes = [f"CELL{i:06d}" for i in range(n_cells)]
        
        # Create AnnData
        adata = ad.AnnData(X=X)
        adata.obs_names = cell_barcodes
        adata.var_names = gene_names
        
        # Add TCR data
        clones_df, tcrs = self.create_minimal_tcr_data(n_cells)
        
        # Add TCR info to obs
        for col in ['clonotype_id', 'va', 'ja', 'cdr3a', 'vb', 'jb', 'cdr3b']:
            adata.obs[col] = clones_df[col].values
        
        # Add organism info
        adata.uns['organism'] = 'human'
        
        return adata
    
    def create_large_dataset(self, n_cells: int = 10000) -> ad.AnnData:
        """Create large dataset for performance testing."""
        return self.create_adata_with_tcrs(n_cells, n_genes=5000)
    
    def create_edge_case_tcrs(self) -> Tuple[pd.DataFrame, List]:
        """Create TCR data with edge cases for testing validation."""
        edge_cases = [
            # Very short CDR3s
            {
                'cell_barcode': 'EDGE001',
                'clonotype_id': 'edge_1',
                'va': 'TRAV1-1*01', 'ja': 'TRAJ1*01', 'cdr3a': 'CAF',  # Very short
                'vb': 'TRBV1*01', 'jb': 'TRBJ1-1*01', 'cdr3b': 'CASF'
            },
            # Very long CDR3s  
            {
                'cell_barcode': 'EDGE002',
                'clonotype_id': 'edge_2',
                'va': 'TRAV2*01', 'ja': 'TRAJ2*01', 
                'cdr3a': 'C' + 'A' * 30 + 'F',  # Very long
                'vb': 'TRBV2*01', 'jb': 'TRBJ1-2*01',
                'cdr3b': 'C' + 'S' * 25 + 'F'
            },
            # Normal cases for comparison
            {
                'cell_barcode': 'EDGE003',
                'clonotype_id': 'edge_3', 
                'va': 'TRAV3*01', 'ja': 'TRAJ3*01', 'cdr3a': 'CAVRDSSYKLIF',
                'vb': 'TRBV3-1*01', 'jb': 'TRBJ2-1*01', 'cdr3b': 'CASSQETQYF'
            }
        ]
        
        # Add nucleotide sequences
        for case in edge_cases:
            case['cdr3a_nucseq'] = 'N' * (len(case['cdr3a']) * 3)
            case['cdr3b_nucseq'] = 'N' * (len(case['cdr3b']) * 3)
        
        clones_df = pd.DataFrame(edge_cases)
        
        # Convert to TCR tuples
        tcrs = []
        for _, row in clones_df.iterrows():
            tcr = ((row['va'], row['ja'], row['cdr3a'], row['cdr3a_nucseq']),
                   (row['vb'], row['jb'], row['cdr3b'], row['cdr3b_nucseq']))
            tcrs.append(tcr)
        
        return clones_df, tcrs
    
    def create_invalid_gene_data(self) -> pd.DataFrame:
        """Create data with invalid gene names for validation testing."""
        invalid_cases = [
            {
                'cell_barcode': 'INVALID001',
                'clonotype_id': 'invalid_1',
                'va': 'INVALID_GENE_A', 'ja': 'TRAJ1*01', 'cdr3a': 'CAVRDSSYKLIF',
                'vb': 'TRBV1*01', 'jb': 'INVALID_GENE_B', 'cdr3b': 'CASSQETQYF',
                'cdr3a_nucseq': 'N' * 36, 'cdr3b_nucseq': 'N' * 30
            },
            {
                'cell_barcode': 'INVALID002', 
                'clonotype_id': 'invalid_2',
                'va': 'TRAV1-1*01', 'ja': 'TRAJ1*01', 'cdr3a': 'CAVRDSSYKLIF',
                'vb': 'NONEXISTENT_VB', 'jb': 'TRBJ1-1*01', 'cdr3b': 'CASSQETQYF',
                'cdr3a_nucseq': 'N' * 36, 'cdr3b_nucseq': 'N' * 30
            }
        ]
        
        return pd.DataFrame(invalid_cases)
    
    def create_performance_test_scenarios(self) -> Dict[str, ad.AnnData]:
        """Create different sized datasets for performance testing."""
        scenarios = {}
        
        # Small dataset (should use sklearn)
        scenarios['small'] = self.create_adata_with_tcrs(500, 1000)
        
        # Medium dataset (should use FAISS CPU)
        scenarios['medium'] = self.create_adata_with_tcrs(5000, 2000)
        
        # Large dataset (should prefer FAISS GPU if available)
        scenarios['large'] = self.create_adata_with_tcrs(20000, 3000)
        
        return scenarios
    
    def save_test_fixtures(self, output_dir: str = "tests/fixtures"):
        """Save all test fixtures to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Minimal test data
        minimal_clones, minimal_tcrs = self.create_minimal_tcr_data(100)
        minimal_clones.to_csv(f"{output_dir}/minimal_clones.tsv", sep='\t', index=False)
        
        minimal_adata = self.create_adata_with_tcrs(100, 500)
        minimal_adata.write(f"{output_dir}/minimal_adata.h5ad")
        
        # Edge case data
        edge_clones, _ = self.create_edge_case_tcrs()
        edge_clones.to_csv(f"{output_dir}/edge_case_clones.tsv", sep='\t', index=False)
        
        # Invalid gene data
        invalid_clones = self.create_invalid_gene_data()
        invalid_clones.to_csv(f"{output_dir}/invalid_gene_clones.tsv", sep='\t', index=False)
        
        # Performance test scenarios
        scenarios = self.create_performance_test_scenarios()
        for name, adata in scenarios.items():
            adata.write(f"{output_dir}/performance_{name}_adata.h5ad")
        
        logger.info(f"Test fixtures saved to {output_dir}")


def generate_all_test_fixtures():
    """Generate all test fixtures for the test suite."""
    generator = TestDataGenerator(random_seed=42)
    generator.save_test_fixtures()
    
    print("✅ Generated comprehensive test fixtures:")
    print("   - minimal_clones.tsv: Basic TCR data (100 cells)")
    print("   - minimal_adata.h5ad: Basic AnnData with GEX+TCR")
    print("   - edge_case_clones.tsv: Edge cases for validation")
    print("   - invalid_gene_clones.tsv: Invalid genes for error testing")
    print("   - performance_*_adata.h5ad: Different sized datasets")
    print("   - All fixtures saved to tests/fixtures/")


if __name__ == "__main__":
    generate_all_test_fixtures()
