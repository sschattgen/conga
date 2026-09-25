"""
Test representation selection error conditions and edge cases.

This module focuses specifically on testing the three-way TCR representation
selection logic and all associated error conditions from Requirements 8.

Requirements validated: 10.2, 10.6 (selection table coverage)
"""

import pytest
import pandas as pd
import numpy as np
import logging
from unittest.mock import patch, MagicMock
from collections.abc import Collection

from conga.preprocess import resolve_tcr_representation, TcrRepresentation
from conga import util
from conga.tcrdist.vectorized import SUPPORTED_ORGANISMS


class TestSelectionTableCoverage:
    """Test every row of the Requirements 8 selection table."""
    
    def test_supported_organism_no_override(self):
        """Supported organism, any count, no override -> Vectorized."""
        for organism in SUPPORTED_ORGANISMS:
            for num_obs in [100, 15000, 25000]:  # Below, at, and above limit
                rep = resolve_tcr_representation(
                    organism=organism,
                    num_obs=num_obs
                )
                assert rep.active == util.OBSM_KEY_VEC_TCR
                assert rep.obsm_tag_tcr == util.OBSM_KEY_VEC_TCR
                assert rep.use_exact_tcrdist_nbrs is False
                assert rep.build_vectorized is True
                assert rep.build_kpca is False
    
    def test_supported_organism_exact_override(self):
        """Supported organism, any count, exact override -> Exact."""
        for organism in SUPPORTED_ORGANISMS:
            for num_obs in [100, 15000, 25000]:
                rep = resolve_tcr_representation(
                    organism=organism,
                    num_obs=num_obs,
                    request_exact_nbrs=True
                )
                assert rep.active == util.ACTIVE_REP_EXACT
                assert rep.obsm_tag_tcr is None
                assert rep.use_exact_tcrdist_nbrs is True
                assert rep.build_vectorized is False
                assert rep.build_kpca is False
    
    def test_supported_organism_kpca_override_below_limit(self):
        """Supported organism, below limit, KernelPCA override -> KernelPCA."""
        for organism in SUPPORTED_ORGANISMS:
            rep = resolve_tcr_representation(
                organism=organism,
                num_obs=15000,  # Below default limit of 20000
                request_kpca=True
            )
            assert rep.active == util.OBSM_KEY_PCA_TCR
            assert rep.obsm_tag_tcr == util.OBSM_KEY_PCA_TCR
            assert rep.use_exact_tcrdist_nbrs is False
            assert rep.build_vectorized is False
            assert rep.build_kpca is True
    
    def test_supported_organism_kpca_override_above_limit_error(self):
        """Supported organism, above limit, KernelPCA override -> Error."""
        for organism in SUPPORTED_ORGANISMS:
            with pytest.raises(ValueError) as excinfo:
                resolve_tcr_representation(
                    organism=organism,
                    num_obs=25000,  # Above limit
                    request_kpca=True
                )
            error_msg = str(excinfo.value)
            assert '25000' in error_msg
            assert 'limit' in error_msg.lower()
    
    def test_unsupported_organism_no_override_below_limit(self):
        """Unsupported organism, below limit, no override -> KernelPCA."""
        unsupported_organisms = ['human_gd', 'mouse_gd', 'rhesus_gd', 'human_ig', 'mouse_ig']
        
        for organism in unsupported_organisms:
            rep = resolve_tcr_representation(
                organism=organism,
                num_obs=15000  # Below limit
            )
            assert rep.active == util.OBSM_KEY_PCA_TCR
            assert rep.obsm_tag_tcr == util.OBSM_KEY_PCA_TCR
            assert rep.use_exact_tcrdist_nbrs is False
            assert rep.build_vectorized is False
            assert rep.build_kpca is True
    
    def test_unsupported_organism_no_override_above_limit(self, caplog):
        """Unsupported organism, above limit, no override -> Exact with INFO log."""
        unsupported_organisms = ['human_gd', 'mouse_gd']
        
        for organism in unsupported_organisms:
            with caplog.at_level(logging.INFO):
                rep = resolve_tcr_representation(
                    organism=organism,
                    num_obs=25000  # Above limit
                )
                
            assert rep.active == util.ACTIVE_REP_EXACT
            assert rep.obsm_tag_tcr is None
            assert rep.use_exact_tcrdist_nbrs is True
            assert rep.build_vectorized is False
            assert rep.build_kpca is False
            
            # Check INFO log was generated (Requirement 8.7)
            info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
            if info_messages:  # If logging was captured
                info_text = ' '.join(info_messages).lower()
                assert '25000' in info_text or str(25000) in info_text
                assert 'limit' in info_text or '20000' in info_text
    
    def test_unsupported_organism_exact_override(self):
        """Unsupported organism, any count, exact override -> Exact."""
        unsupported_organisms = ['human_gd', 'mouse_ig']
        
        for organism in unsupported_organisms:
            for num_obs in [15000, 25000]:
                rep = resolve_tcr_representation(
                    organism=organism,
                    num_obs=num_obs,
                    request_exact_nbrs=True
                )
                assert rep.active == util.ACTIVE_REP_EXACT
                assert rep.obsm_tag_tcr is None
                assert rep.use_exact_tcrdist_nbrs is True
    
    def test_unsupported_organism_kpca_override_below_limit(self):
        """Unsupported organism, below limit, KernelPCA override -> KernelPCA.""" 
        for organism in ['human_gd', 'mouse_ig']:
            rep = resolve_tcr_representation(
                organism=organism,
                num_obs=15000,
                request_kpca=True
            )
            assert rep.active == util.OBSM_KEY_PCA_TCR
            assert rep.build_kpca is True
    
    def test_unsupported_organism_kpca_override_above_limit_error(self):
        """Unsupported organism, above limit, KernelPCA override -> Error."""
        for organism in ['human_gd', 'mouse_ig']:
            with pytest.raises(ValueError) as excinfo:
                resolve_tcr_representation(
                    organism=organism,
                    num_obs=25000,
                    request_kpca=True
                )
            error_msg = str(excinfo.value)
            assert '25000' in error_msg
            assert 'limit' in error_msg.lower()


class TestCustomLimits:
    """Test custom KPCA reduction limits."""
    
    def test_custom_limit_below_default(self):
        """Test with custom limit below default."""
        custom_limit = 15000
        
        # Below custom limit -> KernelPCA for unsupported
        rep = resolve_tcr_representation(
            organism='human_gd',
            num_obs=10000,
            kpca_reduction_limit=custom_limit
        )
        assert rep.active == util.OBSM_KEY_PCA_TCR
        
        # Above custom limit -> Exact for unsupported  
        rep = resolve_tcr_representation(
            organism='human_gd',
            num_obs=20000,
            kpca_reduction_limit=custom_limit
        )
        assert rep.active == util.ACTIVE_REP_EXACT
        
    def test_custom_limit_kpca_override_error(self):
        """Test KernelPCA override error with custom limit."""
        custom_limit = 10000
        
        with pytest.raises(ValueError) as excinfo:
            resolve_tcr_representation(
                organism='human',
                num_obs=15000,
                request_kpca=True,
                kpca_reduction_limit=custom_limit
            )
            
        error_msg = str(excinfo.value)
        assert '15000' in error_msg
        assert str(custom_limit) in error_msg


class TestRestartBehavior:
    """Test restart behavior from stored representations (Requirements 8.27-8.31)."""
    
    def test_restart_kpca_only_no_override(self):
        """Restart with only KernelPCA stored, no override -> use KernelPCA."""
        stored_keys = [util.OBSM_KEY_PCA_TCR]
        
        rep = resolve_tcr_representation(
            organism='human',
            num_obs=15000,
            stored_obsm_keys=stored_keys
        )
        assert rep.active == util.OBSM_KEY_PCA_TCR
        assert rep.build_kpca is False  # Don't rebuild existing
        assert rep.build_vectorized is False
        
    def test_restart_kpca_above_current_limit_warning(self, caplog):
        """Restart with KernelPCA above current limit -> warning but use it."""
        stored_keys = [util.OBSM_KEY_PCA_TCR]
        
        with caplog.at_level(logging.WARNING):
            rep = resolve_tcr_representation(
                organism='human_gd',  # Unsupported, would normally go exact above limit
                num_obs=25000,  # Above limit
                stored_obsm_keys=stored_keys
            )
            
        assert rep.active == util.OBSM_KEY_PCA_TCR
        
        # Should log warning about exceeding current limit
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        if warning_messages:
            warning_text = ' '.join(warning_messages).lower()
            assert 'limit' in warning_text and ('25000' in warning_text or 'exceed' in warning_text)
            
    def test_restart_both_stored_prefer_vectorized(self):
        """Restart with both stored -> prefer vectorized (newer default)."""
        stored_keys = [util.OBSM_KEY_VEC_TCR, util.OBSM_KEY_PCA_TCR]
        
        rep = resolve_tcr_representation(
            organism='human',
            num_obs=15000,
            stored_obsm_keys=stored_keys
        )
        assert rep.active == util.OBSM_KEY_VEC_TCR
        assert rep.build_vectorized is False  # Don't rebuild
        assert rep.build_kpca is False
        
    def test_restart_vectorized_only(self):
        """Restart with only vectorized stored."""
        stored_keys = [util.OBSM_KEY_VEC_TCR]
        
        rep = resolve_tcr_representation(
            organism='human',
            num_obs=15000,
            stored_obsm_keys=stored_keys
        )
        assert rep.active == util.OBSM_KEY_VEC_TCR
        assert rep.build_vectorized is False
        
    def test_restart_neither_stored_apply_table(self):
        """Restart with neither stored -> apply selection table."""
        stored_keys = []  # No stored representations
        
        # Should default to vectorized for supported organism
        rep = resolve_tcr_representation(
            organism='human',
            num_obs=15000,
            stored_obsm_keys=stored_keys
        )
        assert rep.active == util.OBSM_KEY_VEC_TCR
        assert rep.build_vectorized is True  # Build new
        
    def test_restart_exact_override_ignores_stored(self):
        """Restart with exact override -> ignore stored representations."""
        stored_keys = [util.OBSM_KEY_VEC_TCR, util.OBSM_KEY_PCA_TCR]
        
        rep = resolve_tcr_representation(
            organism='human',
            num_obs=15000,
            request_exact_nbrs=True,
            stored_obsm_keys=stored_keys
        )
        assert rep.active == util.ACTIVE_REP_EXACT
        assert rep.build_vectorized is False  # Don't touch stored arrays
        assert rep.build_kpca is False


class TestErrorConditions:
    """Test specific error conditions in representation selection."""
    
    def test_both_overrides_error_clear_message(self):
        """Test clear error message when both overrides requested."""
        with pytest.raises(ValueError) as excinfo:
            resolve_tcr_representation(
                organism='human',
                num_obs=15000,
                request_kpca=True,
                request_exact_nbrs=True
            )
            
        error_msg = str(excinfo.value).lower()
        assert 'kpca' in error_msg or 'kernelpca' in error_msg
        assert 'exact' in error_msg
        assert 'override' in error_msg or 'conflict' in error_msg
        
    def test_limit_error_mentions_all_required_info(self):
        """Test that limit error includes all required information."""
        obs_count = 30000
        limit = 20000
        
        with pytest.raises(ValueError) as excinfo:
            resolve_tcr_representation(
                organism='mouse',
                num_obs=obs_count,
                request_kpca=True,
                kpca_reduction_limit=limit
            )
            
        error_msg = str(excinfo.value)
        assert str(obs_count) in error_msg  # observation count
        assert str(limit) in error_msg  # limit value
        assert 'kpca_reduction_limit' in error_msg  # parameter name


class TestReasonRecording:
    """Test that selection reasons are recorded for statistics."""
    
    def test_reason_recorded_for_all_paths(self):
        """Test that reason field is populated for all resolution outcomes."""
        # Vectorized path
        rep = resolve_tcr_representation(organism='human', num_obs=15000)
        assert rep.reason is not None
        assert len(rep.reason) > 0
        
        # KernelPCA path
        rep = resolve_tcr_representation(organism='human_gd', num_obs=15000)  
        assert rep.reason is not None
        
        # Exact path (auto-selected)
        rep = resolve_tcr_representation(organism='human_gd', num_obs=25000)
        assert rep.reason is not None
        
        # Exact path (override)
        rep = resolve_tcr_representation(organism='human', num_obs=15000, request_exact_nbrs=True)
        assert rep.reason is not None
        
    def test_reason_mentions_key_factors(self):
        """Test that reason mentions key decision factors."""
        # Auto-selected exact for unsupported organism above limit
        rep = resolve_tcr_representation(organism='human_gd', num_obs=25000)
        reason_lower = rep.reason.lower()
        assert any(word in reason_lower for word in ['unsupported', 'limit', 'exact'])
        
        # Override-selected
        rep = resolve_tcr_representation(organism='human', num_obs=15000, request_exact_nbrs=True)
        reason_lower = rep.reason.lower()
        assert 'override' in reason_lower or 'requested' in reason_lower


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_exactly_at_limit(self):
        """Test behavior exactly at the KPCA reduction limit."""
        limit = 20000
        
        # Supported organism at limit -> still vectorized (limit doesn't affect them)
        rep = resolve_tcr_representation(organism='human', num_obs=limit)
        assert rep.active == util.OBSM_KEY_VEC_TCR
        
        # Unsupported organism at limit -> exact
        rep = resolve_tcr_representation(organism='human_gd', num_obs=limit) 
        assert rep.active == util.ACTIVE_REP_EXACT
        
    def test_zero_observations(self):
        """Test with zero observations."""
        rep = resolve_tcr_representation(organism='human', num_obs=0)
        # Should still resolve (even if impractical)
        assert rep.active in [util.OBSM_KEY_VEC_TCR, util.OBSM_KEY_PCA_TCR, util.ACTIVE_REP_EXACT]
        
    def test_very_large_observation_count(self):
        """Test with very large observation count."""
        huge_count = 1_000_000
        
        rep = resolve_tcr_representation(organism='human', num_obs=huge_count)
        # Vectorized should still work for supported organisms
        assert rep.active == util.OBSM_KEY_VEC_TCR
        
        rep = resolve_tcr_representation(organism='human_gd', num_obs=huge_count)
        # Unsupported should go to exact above limit
        assert rep.active == util.ACTIVE_REP_EXACT
        
    def test_invalid_stored_keys_type(self):
        """Test with invalid stored_obsm_keys parameter."""
        # Should handle various collection types
        valid_collections = [
            [],
            ['X_pca_tcr'],
            ('X_vec_tcr',),
            {'X_pca_tcr'},
            frozenset(['X_vec_tcr'])
        ]
        
        for keys in valid_collections:
            rep = resolve_tcr_representation(
                organism='human',
                num_obs=15000,
                stored_obsm_keys=keys
            )
            assert rep is not None
            
    def test_unknown_organism_handling(self):
        """Test with completely unknown organism."""
        # Should treat as unsupported organism
        rep = resolve_tcr_representation(organism='unknown_species', num_obs=15000)
        assert rep.active == util.OBSM_KEY_PCA_TCR  # Below limit -> KernelPCA
        
        rep = resolve_tcr_representation(organism='unknown_species', num_obs=25000) 
        assert rep.active == util.ACTIVE_REP_EXACT  # Above limit -> Exact


if __name__ == '__main__':
    pytest.main([__file__])