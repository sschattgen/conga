#!/usr/bin/env python3

"""
Test deterministic behavior of the vectorized encoder.
Validates that identical configurations produce identical results.
"""

import numpy as np
from conga.tcrdist.vectorized import aa_embedding, encode_tcrs, EncodingConfig

def test_embedding_determinism():
    """Test that aa_embedding is deterministic."""
    print('Testing aa_embedding determinism...')
    
    config1 = EncodingConfig(aa_mds_dim=12, random_seed=42)
    config2 = EncodingConfig(aa_mds_dim=12, random_seed=42)
    config3 = EncodingConfig(aa_mds_dim=12, random_seed=123)
    
    # Same config should give same result
    embedding1 = aa_embedding(config1)
    embedding2 = aa_embedding(config2)
    
    # Different seed should give different result (if random initialization is used)
    # Note: with classical_mds initialization, the seed is ignored
    embedding3 = aa_embedding(config3)
    
    assert np.allclose(embedding1, embedding2), "Same config should give identical embeddings"
    
    # With classical_mds init, different seeds may give identical results (deterministic algorithm)
    if np.allclose(embedding1, embedding3):
        print('  Note: classical_mds initialization is deterministic (seed ignored)')
    else:
        print('  Different seeds produced different embeddings')
    
    print('✓ Embedding determinism verified')

def test_encoding_determinism():
    """Test that encode_tcrs is deterministic."""
    print('\nTesting encode_tcrs determinism...')
    
    tcrs = [
        (('TRAV1-2*01', None, 'CAVRDSNYQLIW', None), 
         ('TRBV20-1*01', None, 'CSARDQETQYF', None)),
        (('TRAV12-1*01', None, 'CAVSLGGSQGNLIF', None), 
         ('TRBV5-1*01', None, 'CASSQDAGGYTF', None)),
    ]
    
    config1 = EncodingConfig(aa_mds_dim=8, num_pos_cdr3=10, random_seed=42)
    config2 = EncodingConfig(aa_mds_dim=8, num_pos_cdr3=10, random_seed=42)
    config3 = EncodingConfig(aa_mds_dim=8, num_pos_cdr3=10, random_seed=999)
    
    # Same config should give same result
    matrix1 = encode_tcrs(tcrs, 'human', config1)
    matrix2 = encode_tcrs(tcrs, 'human', config2)
    
    # Different seed might give same result with classical_mds
    matrix3 = encode_tcrs(tcrs, 'human', config3)
    
    assert np.allclose(matrix1, matrix2), "Same config should give identical encodings"
    
    # With classical_mds init, different seeds may give identical results
    if np.allclose(matrix1, matrix3):
        print('  Note: classical_mds gives identical results regardless of seed')
    else:
        print('  Different seeds produced different encodings')
    
    print('✓ Encoding determinism verified')

def test_config_parameters():
    """Test that config parameters affect output as expected."""
    print('\nTesting config parameter effects...')
    
    tcrs = [
        (('TRAV1-2*01', None, 'CAVRDSNYQLIW', None), 
         ('TRBV20-1*01', None, 'CSARDQETQYF', None)),
    ]
    
    # Test different aa_mds_dim
    config_dim8 = EncodingConfig(aa_mds_dim=8, random_seed=42)
    config_dim12 = EncodingConfig(aa_mds_dim=12, random_seed=42)
    
    matrix_dim8 = encode_tcrs(tcrs, 'human', config_dim8)
    matrix_dim12 = encode_tcrs(tcrs, 'human', config_dim12)
    
    # Different dimensions should give different vector lengths
    assert matrix_dim8.shape[1] != matrix_dim12.shape[1], "Different aa_mds_dim should change vector length"
    
    # Test different cdr3_weight
    config_weight1 = EncodingConfig(cdr3_weight=1.0, random_seed=42)
    config_weight3 = EncodingConfig(cdr3_weight=3.0, random_seed=42)
    
    matrix_weight1 = encode_tcrs(tcrs, 'human', config_weight1)
    matrix_weight3 = encode_tcrs(tcrs, 'human', config_weight3)
    
    # Different CDR3 weights should give different encodings
    assert not np.allclose(matrix_weight1, matrix_weight3), "Different cdr3_weight should change encoding"
    
    print('✓ Config parameter effects verified')

def main():
    print('Testing deterministic behavior of vectorized encoder...\n')
    
    test_embedding_determinism()
    test_encoding_determinism()
    test_config_parameters()
    
    print('\n🎉 All determinism tests passed!')

if __name__ == '__main__':
    main()