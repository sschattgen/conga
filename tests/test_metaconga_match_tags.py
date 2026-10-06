"""Unit test for Requirement 3 of the metaconga-match-integration feature.

Confirms that all 5 Metaconga_Tags were ported verbatim into
`conga/tags.py`, with identical names and string values to the
Source_Branch's `conga/tags.py`.
"""

from conga import tags


EXPECTED_METACONGA_TAGS = {
    # table tags
    'METACONGA_MATCH_CLUMPS': 'metaconga_match_clumps',
    'METACONGA_MATCH_AACLUSTERS': 'metaconga_match_aaclusters',
    # figure tags
    'METACONGA_MATCH_AACLUSTERS_BARS': 'metaconga_match_aaclusters_bars',
    'METACONGA_MATCH_AACLUSTERS_UMAPS': 'metaconga_match_aaclusters_umaps',
    'METACONGA_MATCH_CLUMPS_UMAPS': 'metaconga_match_clumps_umaps',
}


def test_all_metaconga_tags_defined_with_expected_values():
    """Each of the 5 Metaconga_Tags exists as a module-level attribute of
    conga.tags with its expected string value."""
    for name, expected_value in EXPECTED_METACONGA_TAGS.items():
        assert hasattr(tags, name), f"conga.tags is missing expected tag: {name}"
        assert getattr(tags, name) == expected_value, (
            f"conga.tags.{name} has unexpected value: {getattr(tags, name)!r}"
        )


def test_metaconga_tags_are_unique_strings():
    """The 5 Metaconga_Tags do not collide with each other or with any
    other existing tag value in conga.tags."""
    all_tag_values = [
        value for key, value in vars(tags).items()
        if key.isupper() and isinstance(value, str)
    ]
    # No duplicate values anywhere in tags.py, including the 5 new ones.
    assert len(all_tag_values) == len(set(all_tag_values)), (
        "Duplicate tag values found in conga.tags"
    )
