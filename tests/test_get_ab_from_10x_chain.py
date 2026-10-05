"""Unit tests for Requirements 2.1 and 2.2 of the tcrdist-db-update feature.

Confirms that `conga.tcrdist.make_10x_clones_file.get_ab_from_10x_chain`:
- correctly maps 10x contig chain labels to Chain_Label values for every
  Newly_Supported_Organism, following the same per-receptor-type
  convention already applied to the pre-existing organisms, and
- raises a `ValueError` naming the organism (rather than calling
  `sys.exit()`) for any organism string excluded by the
  Chain_Completeness_Rule (`rainbowtrout` and its variants, `sheep_gd`,
  `sheep_ig`).
"""

import pytest

from conga.tcrdist.make_10x_clones_file import get_ab_from_10x_chain


# organism -> {chain_label: expected_return_value}
ALPHA_BETA_ORGANISMS = ['cat', 'dog', 'ferret', 'rabbit', 'sheep']
GAMMA_DELTA_ORGANISMS = ['cat_gd', 'dog_gd', 'ferret_gd', 'rabbit_gd']
IG_ORGANISMS = ['cat_ig', 'dog_ig', 'ferret_ig', 'rabbit_ig', 'rhesus_ig']

CHAIN_COMPLETENESS_EXCLUDED_ORGANISMS = [
    'rainbowtrout',
    'rainbowtrout_gd',
    'rainbowtrout_ig',
    'sheep_gd',
    'sheep_ig',
]


@pytest.mark.parametrize("organism", ALPHA_BETA_ORGANISMS)
def test_alpha_beta_newly_supported_organism(organism):
    assert get_ab_from_10x_chain('TRA', organism) == 'A'
    assert get_ab_from_10x_chain('TRB', organism) == 'B'
    # Chains invalid for this receptor type return None, not raise.
    assert get_ab_from_10x_chain('TRG', organism) is None
    assert get_ab_from_10x_chain('TRD', organism) is None
    assert get_ab_from_10x_chain('IGH', organism) is None


@pytest.mark.parametrize("organism", GAMMA_DELTA_ORGANISMS)
def test_gamma_delta_newly_supported_organism(organism):
    assert get_ab_from_10x_chain('TRG', organism) == 'A'
    assert get_ab_from_10x_chain('TRD', organism) == 'B'
    # TRA is accepted by the gamma-delta branch's membership check, and
    # maps to 'B' since it is not 'TRG' -- matching the existing
    # human_gd/mouse_gd/rhesus_gd convention exactly.
    assert get_ab_from_10x_chain('TRA', organism) == 'B'
    assert get_ab_from_10x_chain('TRB', organism) is None


@pytest.mark.parametrize("organism", IG_ORGANISMS)
def test_ig_newly_supported_organism(organism):
    assert get_ab_from_10x_chain('IGH', organism) == 'B'
    assert get_ab_from_10x_chain('IGK', organism) == 'A'
    assert get_ab_from_10x_chain('IGL', organism) == 'A'
    assert get_ab_from_10x_chain('TRA', organism) is None


@pytest.mark.parametrize("organism", CHAIN_COMPLETENESS_EXCLUDED_ORGANISMS)
def test_chain_completeness_excluded_organism_raises_value_error(organism):
    with pytest.raises(ValueError, match=organism):
        get_ab_from_10x_chain('TRA', organism)


def test_unrecognized_organism_message_lists_supported_organisms():
    with pytest.raises(ValueError) as exc_info:
        get_ab_from_10x_chain('TRA', 'not_a_real_organism')
    message = str(exc_info.value)
    assert 'not_a_real_organism' in message
    # Spot-check a representative sample of supported organisms are named.
    for organism in ['human', 'cat', 'rabbit_gd', 'rhesus_ig']:
        assert organism in message


def test_preexisting_organisms_still_work():
    """Regression check: pre-existing organism handling is unchanged."""
    assert get_ab_from_10x_chain('TRA', 'human') == 'A'
    assert get_ab_from_10x_chain('TRB', 'mouse') == 'B'
    assert get_ab_from_10x_chain('TRG', 'rhesus_gd') == 'A'
    assert get_ab_from_10x_chain('TRD', 'human_gd') == 'B'
    assert get_ab_from_10x_chain('IGH', 'mouse_ig') == 'B'
    assert get_ab_from_10x_chain('IGK', 'human_ig') == 'A'
