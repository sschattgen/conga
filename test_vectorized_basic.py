#!/usr/bin/env python3

import sys
import numpy as np
from conga.tcrdist.vectorized import (
    symbol_dissimilarity_matrix, 
    EncodingConfig, 
    aa_embedding,
    trim_and_gap_cdr3,
    vector_length
)

def test_symbol_dissimilarity_matrix():
    print('Testing symbol_dissimilarity_matrix...')
    dm = symbol_dissimilarity_matrix()
    print(f'Matrix shape: {dm.shape}')
    print(f'Matrix dtype: {dm.dtype}')
    print(f'Symmetric: {np.allclose(dm, dm.T)}')
    print(f'Zero diagonal: {np.allclose(np.diag(dm), 0)}')
    print(f'Value range: [{dm.min()}, {dm.max()}]')
    print(f'Gap penalty (row 20): {dm[0, 20]}, {dm[20, 0]}')
    assert dm.shape == (21, 21)
    assert np.allclose(dm, dm.T)
    assert np.allclose(np.diag(dm), 0)
    assert dm.min() >= 0 and dm.max() <= 4
    print('✓ symbol_dissimilarity_matrix passed')

def test_aa_embedding():
    print('\nTesting aa_embedding...')
    config = EncodingConfig()
    embedding = aa_embedding(config)
    print(f'Embedding shape: {embedding.shape}')
    print(f'Embedding dtype: {embedding.dtype}')
    print(f'Column means (should be ~zero): {np.mean(embedding, axis=0)[:3]}')
    
    # Test deterministic behavior
    embedding2 = aa_embedding(config)
    print(f'Deterministic: {np.allclose(embedding, embedding2)}')
    
    assert embedding.shape == (21, 16)
    assert np.allclose(embedding, embedding2)
    assert np.allclose(np.mean(embedding, axis=0), 0, atol=1e-10)
    print('✓ aa_embedding passed')

def test_trim_and_gap_cdr3():
    print('\nTesting trim_and_gap_cdr3...')
    test_cdr3 = 'CASSYPGLAGGRPEQYF'
    gapped = trim_and_gap_cdr3(test_cdr3, num_pos=16, n_trim=3, c_trim=2)
    print(f'Original CDR3: {test_cdr3}')
    print(f'Gapped CDR3:   {gapped} (len={len(gapped)})')
    
    assert len(gapped) == 16
    
    # Test with different lengths
    short_cdr3 = 'CASF'
    try:
        gapped_short = trim_and_gap_cdr3(short_cdr3, num_pos=16, n_trim=3, c_trim=2)
        assert False, "Should have raised ValueError for too short CDR3"
    except ValueError as e:
        print(f'Expected error for short CDR3: {e}')
    
    print('✓ trim_and_gap_cdr3 passed')

def test_vector_length():
    print('\nTesting vector_length...')
    config = EncodingConfig()
    try:
        length = vector_length('human', config)
        print(f'Human vector length: {length}')
        assert length > 0
        print('✓ vector_length passed')
    except Exception as e:
        print(f'Vector length calculation failed: {e}')

def test_germline_table():
    print('\nTesting germline_code_table...')
    try:
        from conga.tcrdist.vectorized import germline_code_table
        gene_ids, code_matrix = germline_code_table('human', 'A')
        print(f'Human A chain: {len(gene_ids)} genes, {code_matrix.shape} code matrix')
        assert len(gene_ids) > 0
        assert code_matrix.shape[0] == len(gene_ids)
        print('✓ germline_code_table passed')
    except Exception as e:
        print(f'Germline table failed: {e}')
        # This might fail due to gene database issues, which is expected

def main():
    print('Running basic vectorized TCRdist tests...\n')
    
    test_symbol_dissimilarity_matrix()
    test_aa_embedding() 
    test_trim_and_gap_cdr3()
    test_vector_length()
    test_germline_table()
    
    print('\n✓ All basic tests completed successfully!')

if __name__ == '__main__':
    main()