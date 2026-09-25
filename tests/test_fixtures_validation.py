"""
Test fixture validation for CoNGA vectorized TCRdist feature.

This module validates that all test fixtures are working correctly and 
provide the expected data formats and quality.

Requirements validated: 10.5
"""

import pytest
import pandas as pd
import numpy as np
from typing import List

from .conftest import (
    CloneData, 
    CloneFixtureGenerator,
    load_human_tcr_database,
    filter_valid_human_clonotypes,
    TEST_RANDOM_SEED,
    AMINO_ACIDS
)


class TestFixtureValidation:
    """Test that all fixtures provide valid, reproducible data."""
    
    def test_human_tcr_database_loads(self, human_tcr_data):
        """Test that human TCR database loads and is filtered correctly."""
        assert isinstance(human_tcr_data, pd.DataFrame)
        assert len(human_tcr_data) > 0, "Human TCR database should not be empty"
        
        # Check required columns are present
        required_cols = ['va', 'vb', 'cdr3a', 'cdr3b']
        for col in required_cols:
            assert col in human_tcr_data.columns, f"Missing required column: {col}"
        
        # Check no missing values in required columns
        for col in required_cols:
            assert not human_tcr_data[col].isna().any(), f"Column {col} has missing values"
        
        # Check CDR3 sequences are valid (amino acids only)
        for cdr3_col in ['cdr3a', 'cdr3b']:
            for cdr3 in human_tcr_data[cdr3_col]:
                assert all(aa in AMINO_ACIDS for aa in cdr3), f"Invalid amino acid in {cdr3_col}: {cdr3}"
                assert len(cdr3) >= 6, f"CDR3 too short in {cdr3_col}: {cdr3}"
    
    def test_human_clonotypes_small(self, human_clonotypes_small):
        """Test small human clonotype fixture."""
        assert isinstance(human_clonotypes_small, list)
        assert len(human_clonotypes_small) <= 50
        assert len(human_clonotypes_small) > 0
        
        # Check first clonotype structure
        clone = human_clonotypes_small[0]
        assert isinstance(clone, CloneData)
        self._validate_clonotype(clone)
    
    def test_human_clonotypes_medium(self, human_clonotypes_medium):
        """Test medium human clonotype fixture.""" 
        assert isinstance(human_clonotypes_medium, list)
        assert len(human_clonotypes_medium) <= 300
        assert len(human_clonotypes_medium) > 0
        
        # Check all clonotypes are valid
        for clone in human_clonotypes_medium[:10]:  # Check first 10
            self._validate_clonotype(clone)
    
    def test_human_clonotypes_large(self, human_clonotypes_large):
        """Test large human clonotype fixture."""
        assert isinstance(human_clonotypes_large, list)
        assert len(human_clonotypes_large) <= 1000
        assert len(human_clonotypes_large) > 0
        
        # Check sample of clonotypes
        for i in range(0, min(len(human_clonotypes_large), 100), 10):
            self._validate_clonotype(human_clonotypes_large[i])
    
    def test_mouse_clonotypes_small(self, mouse_clonotypes_small):
        """Test synthetic mouse clonotype fixture."""
        assert isinstance(mouse_clonotypes_small, list)
        assert len(mouse_clonotypes_small) == 50
        
        for clone in mouse_clonotypes_small[:10]:
            self._validate_clonotype(clone)
    
    def test_mouse_clonotypes_medium(self, mouse_clonotypes_medium):
        """Test synthetic mouse clonotype fixture."""
        assert isinstance(mouse_clonotypes_medium, list) 
        assert len(mouse_clonotypes_medium) == 300
        
        for clone in mouse_clonotypes_medium[:10]:
            self._validate_clonotype(clone)
    
    def test_rhesus_clonotypes_small(self, rhesus_clonotypes_small):
        """Test synthetic rhesus clonotype fixture."""
        assert isinstance(rhesus_clonotypes_small, list)
        assert len(rhesus_clonotypes_small) == 50
        
        for clone in rhesus_clonotypes_small[:10]:
            self._validate_clonotype(clone)
    
    def test_edge_case_clonotypes(self, edge_case_clonotypes):
        """Test edge case clonotypes cover boundary conditions."""
        assert isinstance(edge_case_clonotypes, list)
        assert len(edge_case_clonotypes) > 0
        
        # Should have clonotypes with different CDR3 lengths
        cdr3a_lengths = [len(clone.cdr3a) for clone in edge_case_clonotypes]
        cdr3b_lengths = [len(clone.cdr3b) for clone in edge_case_clonotypes]
        
        assert min(cdr3a_lengths) <= 6, "Should have short CDR3a sequences"
        assert max(cdr3a_lengths) >= 18, "Should have long CDR3a sequences" 
        assert min(cdr3b_lengths) <= 6, "Should have short CDR3b sequences"
        assert max(cdr3b_lengths) >= 18, "Should have long CDR3b sequences"
    
    def test_mixed_format_data(self, mixed_format_data):
        """Test mixed format data provides same clonotypes in different formats."""
        data = mixed_format_data
        
        # Check all formats are present
        assert 'tuples' in data
        assert 'dataframe_default' in data
        assert 'dataframe_custom' in data
        assert 'custom_columns' in data
        
        # Check tuples format
        tuples = data['tuples']
        assert isinstance(tuples, list)
        assert len(tuples) > 0
        assert isinstance(tuples[0], tuple)
        assert len(tuples[0]) == 2  # (alpha_chain, beta_chain)
        
        # Check DataFrame formats
        df_default = data['dataframe_default']
        assert isinstance(df_default, pd.DataFrame)
        assert 'va' in df_default.columns
        assert 'cdr3a' in df_default.columns
        
        df_custom = data['dataframe_custom']
        assert isinstance(df_custom, pd.DataFrame)
        custom_cols = data['custom_columns']
        assert custom_cols['va_column'] in df_custom.columns
        assert custom_cols['cdr3a_column'] in df_custom.columns
    
    def test_supported_organism_fixture(self, supported_organism):
        """Test parametrized organism fixture."""
        assert supported_organism in ['human', 'mouse', 'rhesus']
    
    def test_encoding_configs(self, encoding_configs):
        """Test encoding configuration fixtures."""
        assert isinstance(encoding_configs, list)
        assert len(encoding_configs) >= 2  # At least default + one variant
        
        for config in encoding_configs:
            assert isinstance(config, dict)
            required_keys = ['aa_mds_dim', 'num_pos_cdr3', 'cdr3_weight', 'n_trim', 'c_trim', 'random_seed']
            for key in required_keys:
                assert key in config, f"Missing config key: {key}"
            
            # Validate ranges
            assert 1 <= config['aa_mds_dim'] <= 21
            assert config['num_pos_cdr3'] >= 1
            assert config['cdr3_weight'] > 0
            assert config['n_trim'] >= 0
            assert config['c_trim'] >= 0
    
    def test_reproducibility_across_sessions(self):
        """Test that fixtures produce identical results across different sessions."""
        # Generate same data twice with same seed
        gen1 = CloneFixtureGenerator(TEST_RANDOM_SEED)
        gen2 = CloneFixtureGenerator(TEST_RANDOM_SEED)
        
        clones1 = gen1.generate_clonotypes('mouse', 10)
        clones2 = gen2.generate_clonotypes('mouse', 10)
        
        assert len(clones1) == len(clones2)
        for c1, c2 in zip(clones1, clones2):
            assert c1.va == c2.va
            assert c1.vb == c2.vb
            assert c1.cdr3a == c2.cdr3a
            assert c1.cdr3b == c2.cdr3b
    
    def _validate_clonotype(self, clone: CloneData):
        """Helper to validate a single clonotype structure."""
        assert isinstance(clone, CloneData)
        
        # Check all required fields are present and non-empty
        assert clone.va and isinstance(clone.va, str)
        assert clone.ja and isinstance(clone.ja, str) 
        assert clone.vb and isinstance(clone.vb, str)
        assert clone.jb and isinstance(clone.jb, str)
        assert clone.cdr3a and isinstance(clone.cdr3a, str)
        assert clone.cdr3b and isinstance(clone.cdr3b, str)
        
        # Check CDR3 sequences are valid amino acids
        for cdr3 in [clone.cdr3a, clone.cdr3b]:
            assert len(cdr3) >= 4, f"CDR3 too short: {cdr3}"
            assert all(aa in AMINO_ACIDS for aa in cdr3), f"Invalid amino acid in CDR3: {cdr3}"
        
        # Test conversion methods
        tuple_form = clone.as_tuple()
        assert isinstance(tuple_form, tuple)
        assert len(tuple_form) == 2
        
        dict_form = clone.as_dict() 
        assert isinstance(dict_form, dict)
        required_keys = ['va', 'ja', 'cdr3a', 'vb', 'jb', 'cdr3b']
        for key in required_keys:
            assert key in dict_form


class TestCloneFixtureGenerator:
    """Test the synthetic clonotype generator."""
    
    def test_generator_initialization(self):
        """Test generator initializes with proper random seed."""
        gen = CloneFixtureGenerator(42)
        assert gen.rng is not None
    
    def test_cdr3_generation(self):
        """Test CDR3 generation produces valid sequences."""
        gen = CloneFixtureGenerator(42)
        
        for _ in range(20):
            cdr3 = gen._generate_cdr3(8, 15)
            assert len(cdr3) >= 8
            assert len(cdr3) <= 15 
            assert cdr3.startswith('C')
            assert cdr3.endswith('F')
            assert all(aa in AMINO_ACIDS for aa in cdr3)
    
    def test_gene_loading_fallback(self):
        """Test that gene loading works with fallback for missing database."""
        gen = CloneFixtureGenerator(42)
        
        # Should not raise exception even if database is missing
        genes_a = gen._load_genes_for_organism_chain('human', 'A')
        genes_b = gen._load_genes_for_organism_chain('human', 'B')
        
        assert isinstance(genes_a, list)
        assert isinstance(genes_b, list)
        assert len(genes_a) > 0
        assert len(genes_b) > 0
        
        # All should be valid gene names (strings)
        for gene in genes_a[:5]:
            assert isinstance(gene, str)
            assert len(gene) > 0
    
    @pytest.mark.parametrize("organism", ["human", "mouse", "rhesus"])
    def test_clonotype_generation_all_organisms(self, organism):
        """Test clonotype generation for all supported organisms."""
        gen = CloneFixtureGenerator(42)
        clones = gen.generate_clonotypes(organism, 10)
        
        assert len(clones) == 10
        for clone in clones:
            assert isinstance(clone, CloneData)
            # Basic validation
            assert len(clone.cdr3a) >= 6
            assert len(clone.cdr3b) >= 6
            assert clone.va.startswith('TR')  # Should be TCR gene names
            assert clone.vb.startswith('TR')


if __name__ == "__main__":
    pytest.main([__file__])