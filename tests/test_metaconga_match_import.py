"""Unit test for Requirement 1 (Requirement 9.1) of the
metaconga-match-integration feature.

Confirms that `conga/metaconga_match.py` was ported successfully and
registered in `conga/__init__.py`: `import conga.metaconga_match` succeeds
without raising (its module-level data loading from
`conga/data/metaconga/` must succeed), and all four public pipeline
functions are resolvable as attributes.
"""

import importlib
import sys

import pytest


def test_import_conga_metaconga_match_succeeds():
    """import conga.metaconga_match succeeds without raising an exception.

    Reimports fresh so this test does not depend on module caching from
    an earlier import elsewhere in the test session.
    """
    for mod_name in list(sys.modules):
        if mod_name == 'conga.metaconga_match':
            del sys.modules[mod_name]

    module = importlib.import_module('conga.metaconga_match')
    assert module is not None


def test_metaconga_match_registered_on_conga_package():
    """conga.metaconga_match is reachable via plain attribute access on the
    top-level conga package, i.e. `from . import metaconga_match` was
    added to conga/__init__.py."""
    import conga
    assert hasattr(conga, 'metaconga_match')


@pytest.mark.parametrize('func_name', [
    'find_aacluster_matches',
    'plot_aacluster_matches',
    'find_clump_matches',
    'plot_clump_matches',
])
def test_metaconga_match_public_functions_resolvable(func_name):
    """Each of the 4 public pipeline functions is resolvable as an
    attribute of conga.metaconga_match and is callable."""
    import conga.metaconga_match as metaconga_match
    assert hasattr(metaconga_match, func_name), (
        f"conga.metaconga_match is missing expected function: {func_name}"
    )
    assert callable(getattr(metaconga_match, func_name))
