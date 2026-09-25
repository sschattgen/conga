"""
Example usage of test fixtures for vectorized TCRdist testing.

This demonstrates how to use the various fixtures for testing the 
vectorized TCRdist implementation.

Requirements validated: Example usage of 10.1, 10.5
"""

import pytest
import pandas as pd
import numpy as np


class TestFixtureExamples:
    """Demonstrate how to use the test fixtures."""
    
    def test_human_data_format_examples(self, human_clonotypes_small):
        """Show different ways to access human clonotype data."""
        
        # Access as objects
        first_clone = human_clonotypes_small[0]
        print(f"First clone: {first_clone.va} | {first_clone.cdr3a}")
        
        # Convert to nested tuples (format expected by vectorized encoder)
        tuples_format = [clone.as_tuple() for clone in human_clonotypes_small]
        print(f"Tuple format: {tuples_format[0]}")
        
        # Convert to DataFrame
        df = pd.DataFrame([clone.as_dict() for clone in human_clonotypes_small])
        print(f"DataFrame shape: {df.shape}")
        assert 'va' in df.columns
        assert 'cdr3a' in df.columns
    
    def test_synthetic_data_examples(self, mouse_clonotypes_small, rhesus_clonotypes_small):
        """Show synthetic data from mouse and rhesus."""
        
        # Mouse data
        assert len(mouse_clonotypes_small) == 50
        mouse_clone = mouse_clonotypes_small[0]
        print(f"Mouse clone: {mouse_clone.va} | {mouse_clone.cdr3a}")
        
        # Rhesus data  
        assert len(rhesus_clonotypes_small) == 50
        rhesus_clone = rhesus_clonotypes_small[0]
        print(f"Rhesus clone: {rhesus_clone.va} | {rhesus_clone.cdr3a}")
        
        # Both should have valid gene names and CDR3 sequences
        for organism, clones in [("mouse", mouse_clonotypes_small), ("rhesus", rhesus_clonotypes_small)]:
            clone = clones[0]
            assert clone.va.startswith('TR')
            assert clone.vb.startswith('TR')
            assert len(clone.cdr3a) >= 6
            assert len(clone.cdr3b) >= 6
    
    def test_edge_cases_examples(self, edge_case_clonotypes):
        """Show how to use edge case data."""
        
        assert len(edge_case_clonotypes) >= 4
        
        # Should have different CDR3 lengths 
        cdr3_lengths = [(len(clone.cdr3a), len(clone.cdr3b)) for clone in edge_case_clonotypes]
        print(f"CDR3 length ranges: {cdr3_lengths}")
        
        # At least one short and one long CDR3
        all_lengths = [length for pair in cdr3_lengths for length in pair]
        assert min(all_lengths) <= 6  # Short CDR3s
        assert max(all_lengths) >= 18  # Long CDR3s
    
    def test_mixed_format_examples(self, mixed_format_data):
        """Show how to use mixed format data for format flexibility testing."""
        
        # Get the different formats
        tuples = mixed_format_data['tuples']
        df_default = mixed_format_data['dataframe_default']
        df_custom = mixed_format_data['dataframe_custom']
        custom_cols = mixed_format_data['custom_columns']
        
        # Should all represent the same clonotypes
        assert len(tuples) == len(df_default) == len(df_custom)
        
        # Show accessing the data in different ways
        print(f"Tuple format: {tuples[0]}")
        print(f"Default DataFrame: va={df_default.iloc[0]['va']}, cdr3a={df_default.iloc[0]['cdr3a']}")
        print(f"Custom DataFrame: va_gene={df_custom.iloc[0]['va_gene']}, cdr3a_seq={df_custom.iloc[0]['cdr3a_seq']}")
    
    @pytest.mark.parametrize("size", ["small", "medium", "large"])
    def test_different_sizes(self, size, request):
        """Demonstrate using different sample sizes."""
        
        # Dynamically get the fixture based on size
        fixture_name = f"human_clonotypes_{size}"
        clonotypes = request.getfixturevalue(fixture_name)
        
        expected_sizes = {"small": 50, "medium": 300, "large": 1000}
        expected_max = expected_sizes[size]
        
        assert len(clonotypes) <= expected_max
        assert len(clonotypes) > 0
        print(f"Got {len(clonotypes)} clonotypes for size '{size}'")
    
    def test_encoding_config_examples(self, encoding_configs):
        """Show how to use different encoding configurations."""
        
        # Should have multiple configurations
        assert len(encoding_configs) >= 2
        
        for i, config in enumerate(encoding_configs):
            print(f"Config {i}: aa_mds_dim={config['aa_mds_dim']}, "
                  f"num_pos_cdr3={config['num_pos_cdr3']}, "
                  f"cdr3_weight={config['cdr3_weight']}")
            
            # All configs should be valid
            assert config['aa_mds_dim'] >= 1
            assert config['num_pos_cdr3'] >= 1
            assert config['cdr3_weight'] > 0
    
    def test_organism_parametrization_example(self, supported_organism):
        """Show parametrized testing across organisms."""
        
        assert supported_organism in ['human', 'mouse', 'rhesus']
        print(f"Testing organism: {supported_organism}")
        
        # Could use this to test organism-specific logic
        if supported_organism == 'human':
            assert True  # Human-specific tests
        elif supported_organism == 'mouse':
            assert True  # Mouse-specific tests  
        else:  # rhesus
            assert True  # Rhesus-specific tests
    
    def test_reproducibility_example(self):
        """Show that fixtures are reproducible."""
        from .conftest import CloneFixtureGenerator, TEST_RANDOM_SEED
        
        # Generate data twice with same seed
        gen1 = CloneFixtureGenerator(TEST_RANDOM_SEED)
        gen2 = CloneFixtureGenerator(TEST_RANDOM_SEED)
        
        clones1 = gen1.generate_clonotypes('human', 5)
        clones2 = gen2.generate_clonotypes('human', 5)
        
        # Should be identical
        for c1, c2 in zip(clones1, clones2):
            assert c1.va == c2.va
            assert c1.cdr3a == c2.cdr3a
            assert c1.vb == c2.vb
            assert c1.cdr3b == c2.cdr3b
        
        print("Reproducibility confirmed!")


if __name__ == "__main__":
    pytest.main([__file__])