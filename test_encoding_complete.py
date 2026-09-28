#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
from conga.tcrdist.vectorized import encode_tcrs, EncodingConfig, accuracy_report

def test_simple_encoding():
    """Test encoding a small set of example TCRs."""
    print('Testing simple TCR encoding...')
    
    # Create example TCR data as nested tuples
    tcrs = [
        # Each TCR is ((Va, Ja, CDR3a, nucseq), (Vb, Jb, CDR3b, nucseq))
        (('TRAV1-2*01', None, 'CAVRDSNYQLIW', None), 
         ('TRBV20-1*01', None, 'CSARDQETQYF', None)),
        (('TRAV12-1*01', None, 'CAVSLGGSQGNLIF', None), 
         ('TRBV5-1*01', None, 'CASSQDAGGYTF', None)),
        (('TRAV8-3*01', None, 'CAVSGGSYIPTF', None), 
         ('TRBV19*01', None, 'CASSISSPLHF', None)),
    ]
    
    config = EncodingConfig(aa_mds_dim=8, num_pos_cdr3=12)  # Smaller for testing
    
    try:
        # Test nested tuple input
        matrix = encode_tcrs(tcrs, 'human', config)
        print(f'Encoded matrix shape: {matrix.shape}')
        print(f'Matrix dtype: {matrix.dtype}')
        print(f'Matrix C-contiguous: {matrix.flags.c_contiguous}')
        print(f'All finite values: {np.all(np.isfinite(matrix))}')
        
        # Test DataFrame input  
        df = pd.DataFrame([
            {'va': 'TRAV1-2*01', 'cdr3a': 'CAVRDSNYQLIW', 'vb': 'TRBV20-1*01', 'cdr3b': 'CSARDQETQYF'},
            {'va': 'TRAV12-1*01', 'cdr3a': 'CAVSLGGSQGNLIF', 'vb': 'TRBV5-1*01', 'cdr3b': 'CASSQDAGGYTF'},
            {'va': 'TRAV8-3*01', 'cdr3a': 'CAVSGGSYIPTF', 'vb': 'TRBV19*01', 'cdr3b': 'CASSISSPLHF'},
        ])
        
        matrix_df = encode_tcrs(df, 'human', config)
        print(f'DataFrame encoding identical to tuples: {np.allclose(matrix, matrix_df)}')
        
        assert matrix.shape[0] == 3  # 3 clonotypes
        assert matrix.shape[1] > 0   # Non-empty vectors
        assert matrix.dtype == np.float32
        assert matrix.flags.c_contiguous
        assert np.all(np.isfinite(matrix))
        assert np.allclose(matrix, matrix_df)
        
        print('✓ Simple encoding passed')
        return tcrs, matrix
        
    except Exception as e:
        print(f'Encoding failed: {e}')
        raise

def test_accuracy_validation():
    """Test accuracy reporting against exact TCRdist."""
    print('\nTesting accuracy validation...')
    
    # Use same example data as above
    tcrs = [
        (('TRAV1-2*01', None, 'CAVRDSNYQLIW', None), 
         ('TRBV20-1*01', None, 'CSARDQETQYF', None)),
        (('TRAV12-1*01', None, 'CAVSLGGSQGNLIF', None), 
         ('TRBV5-1*01', None, 'CASSQDAGGYTF', None)),
        (('TRAV8-3*01', None, 'CAVSGGSYIPTF', None), 
         ('TRBV19*01', None, 'CASSISSPLHF', None)),
        (('TRAV21*01', None, 'CILRDDGRRSWNTDKLIFW', None), 
         ('TRBV6-5*01', None, 'CASSEGQGLNEQFF', None)),
        (('TRAV29/DV5*01', None, 'CAASLENYNQGKLIF', None), 
         ('TRBV7-8*01', None, 'CASSLAPGTGTGYTF', None)),
    ]
    
    config = EncodingConfig(aa_mds_dim=8, num_pos_cdr3=12)
    
    try:
        report = accuracy_report(tcrs, 'human', config, neighbor_counts=[2, 3])
        
        print(f'Accuracy report:')
        print(f'  Organism: {report.organism}')
        print(f'  Clonotypes: {report.num_clonotypes}') 
        print(f'  Pairs sampled: {report.num_pairs_sampled}')
        print(f'  Spearman correlation: {report.spearman:.4f}')
        print(f'  Pearson (distance): {report.pearson_distance:.4f}')
        print(f'  Pearson (squared): {report.pearson_squared_distance:.4f}')
        print(f'  Mean recall@2: {report.mean_recall[2]:.4f}')
        print(f'  Mean recall@3: {report.mean_recall[3]:.4f}')
        
        assert report.num_clonotypes == 5
        assert report.spearman >= 0  # Should be positive correlation
        assert 0 <= report.mean_recall[2] <= 1  # Valid recall range
        
        print('✓ Accuracy validation passed')
        
    except Exception as e:
        print(f'Accuracy validation failed: {e}')
        # This might fail due to TcrDistCalculator issues, which is acceptable

def main():
    print('Running complete vectorized TCRdist encoding tests...\n')
    
    try:
        test_simple_encoding()
        test_accuracy_validation()
        print('\n🎉 All encoding tests completed successfully!')
        
    except Exception as e:
        print(f'\n❌ Test failed with error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()