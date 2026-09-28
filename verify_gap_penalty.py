#!/usr/bin/env python3

"""
Verify gap penalty consistency between vectorized encoder and TCRdist.
This validates the design assumption that GAP_PENALTY_V_REGION = 4 is correct
for both germline and CDR3 regions when combined with CDR3 weight scaling.
"""

from conga.tcrdist.tcr_distances import (
    GAP_PENALTY_V_REGION, 
    GAP_PENALTY_CDR3_REGION, 
    WEIGHT_CDR3_REGION,
    WEIGHT_V_REGION
)
from conga.tcrdist.vectorized import DEFAULT_CDR3_WEIGHT

def main():
    print('Verifying gap penalty consistency...\n')
    
    print(f'TCRdist constants:')
    print(f'  GAP_PENALTY_V_REGION = {GAP_PENALTY_V_REGION}')
    print(f'  GAP_PENALTY_CDR3_REGION = {GAP_PENALTY_CDR3_REGION}')
    print(f'  WEIGHT_V_REGION = {WEIGHT_V_REGION}')
    print(f'  WEIGHT_CDR3_REGION = {WEIGHT_CDR3_REGION}')
    print(f'  DEFAULT_CDR3_WEIGHT = {DEFAULT_CDR3_WEIGHT}')
    
    # Verify the relationship described in the design
    v_region_contribution = WEIGHT_V_REGION * GAP_PENALTY_V_REGION
    cdr3_region_contribution = WEIGHT_CDR3_REGION * GAP_PENALTY_V_REGION
    expected_cdr3_penalty = GAP_PENALTY_CDR3_REGION
    
    print(f'\nGap penalty contributions:')
    print(f'  V-region: WEIGHT_V_REGION * GAP_PENALTY_V_REGION = {v_region_contribution}')
    print(f'  CDR3 with scaling: WEIGHT_CDR3_REGION * GAP_PENALTY_V_REGION = {cdr3_region_contribution}')
    print(f'  Expected CDR3 penalty: GAP_PENALTY_CDR3_REGION = {expected_cdr3_penalty}')
    
    # Check if the design assumption holds
    cdr3_weight_matches = (DEFAULT_CDR3_WEIGHT == WEIGHT_CDR3_REGION)
    penalty_relationship_holds = (cdr3_region_contribution == expected_cdr3_penalty)
    
    print(f'\nVerification:')
    print(f'  DEFAULT_CDR3_WEIGHT matches WEIGHT_CDR3_REGION: {cdr3_weight_matches}')
    print(f'  Gap penalty relationship holds: {penalty_relationship_holds}')
    
    if cdr3_weight_matches and penalty_relationship_holds:
        print(f'\n✅ Gap penalty consistency verified!')
        print(f'   Using GAP_PENALTY_V_REGION = {GAP_PENALTY_V_REGION} for both regions is correct')
        print(f'   CDR3 weight scaling ({DEFAULT_CDR3_WEIGHT}) preserves TCRdist gap penalties')
    else:
        print(f'\n❌ Gap penalty consistency check failed!')
        if not cdr3_weight_matches:
            print(f'   DEFAULT_CDR3_WEIGHT ({DEFAULT_CDR3_WEIGHT}) != WEIGHT_CDR3_REGION ({WEIGHT_CDR3_REGION})')
        if not penalty_relationship_holds:
            print(f'   Gap penalty relationship broken: {cdr3_region_contribution} != {expected_cdr3_penalty}')

if __name__ == '__main__':
    main()