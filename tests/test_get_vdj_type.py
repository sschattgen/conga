"""Unit tests for Requirements 2.3, 2.4, and 2.5 of the tcrdist-db-update feature.

Confirms that `conga.util.get_vdj_type`:
- returns the correct VDJ_Type constant for every Newly_Supported_Organism
  and every pre-existing organism, following the classification set out
  in design Component 1, and
- raises a `ValueError` naming the organism (rather than a bare
  `KeyError`) for any organism string excluded by the
  Chain_Completeness_Rule (`rainbowtrout` and its variants, `sheep_gd`,
  `sheep_ig`) or otherwise absent from `organism2vdj_type`.
"""

import pytest

from conga.util import get_vdj_type, TCR_AB_VDJ_TYPE, TCR_GD_VDJ_TYPE, IG_VDJ_TYPE


TCR_AB_ORGANISMS = ['cat', 'dog', 'ferret', 'rabbit', 'sheep']
TCR_GD_ORGANISMS = ['cat_gd', 'dog_gd', 'ferret_gd', 'rabbit_gd']
IG_ORGANISMS = ['cat_ig', 'dog_ig', 'ferret_ig', 'rabbit_ig', 'rhesus_ig']

CHAIN_COMPLETENESS_EXCLUDED_ORGANISMS = [
    'rainbowtrout',
    'rainbowtrout_gd',
    'rainbowtrout_ig',
    'sheep_gd',
    'sheep_ig',
]


@pytest.mark.parametrize("organism", TCR_AB_ORGANISMS)
def test_tcr_ab_newly_supported_organism(organism):
    assert get_vdj_type(organism) == TCR_AB_VDJ_TYPE


@pytest.mark.parametrize("organism", TCR_GD_ORGANISMS)
def test_tcr_gd_newly_supported_organism(organism):
    assert get_vdj_type(organism) == TCR_GD_VDJ_TYPE


@pytest.mark.parametrize("organism", IG_ORGANISMS)
def test_ig_newly_supported_organism(organism):
    assert get_vdj_type(organism) == IG_VDJ_TYPE


@pytest.mark.parametrize("organism", CHAIN_COMPLETENESS_EXCLUDED_ORGANISMS)
def test_chain_completeness_excluded_organism_raises_value_error(organism):
    with pytest.raises(ValueError, match=organism):
        get_vdj_type(organism)


def test_unrecognized_organism_message_lists_supported_organisms():
    with pytest.raises(ValueError) as exc_info:
        get_vdj_type('not_a_real_organism')
    message = str(exc_info.value)
    assert 'not_a_real_organism' in message
    # Spot-check a representative sample of supported organisms are named.
    for organism in ['human', 'cat', 'rabbit_gd', 'rhesus_ig']:
        assert organism in message


def test_preexisting_organisms_still_work():
    """Regression check: pre-existing organism handling is unchanged."""
    assert get_vdj_type('human') == TCR_AB_VDJ_TYPE
    assert get_vdj_type('mouse') == TCR_AB_VDJ_TYPE
    assert get_vdj_type('rhesus') == TCR_AB_VDJ_TYPE
    assert get_vdj_type('human_gd') == TCR_GD_VDJ_TYPE
    assert get_vdj_type('mouse_gd') == TCR_GD_VDJ_TYPE
    assert get_vdj_type('rhesus_gd') == TCR_GD_VDJ_TYPE
    assert get_vdj_type('human_ig') == IG_VDJ_TYPE
    assert get_vdj_type('mouse_ig') == IG_VDJ_TYPE
