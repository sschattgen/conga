"""
Unit test for agreement between `SUPPORTED_ORGANISMS` and the Accuracy_Gate
Results table recorded in `.kiro/specs/tcrdist-db-update/design.md`
(Component 4).

Covers task 12.1 of the tcrdist-db-update spec: for every organism string
in `SUPPORTED_ORGANISMS`, there must be a corresponding row in the Results
table whose `Accuracy_Gate` column reads PASS, and conversely no organism
with a PASS row may be missing from `SUPPORTED_ORGANISMS` -- Requirement
7.3 specifies a binary, all-or-nothing inclusion rule, so this test checks
exact set equality rather than a one-directional subset check.

This test parses the Results table directly out of design.md (rather than
hardcoding a duplicate fixture dict) so that re-running the Validation_Harness
and updating design.md's table is the single source of truth that keeps
this test in sync, with no second place requiring a manual edit.

**Property 2: No unsupported organism in SUPPORTED_ORGANISMS without
recorded evidence**
**Validates: Requirements 7.1, 7.2, 7.3**
"""

import re
from pathlib import Path

import pytest

from conga.tcrdist.vectorized import SUPPORTED_ORGANISMS

DESIGN_MD_PATH = (
    Path(__file__).resolve().parent.parent
    / '.kiro' / 'specs' / 'tcrdist-db-update' / 'design.md'
)

# Matches a Results table data row, e.g.:
# | `human` | Tier_1 | 0.9988 | 0.9400 | PASS | Yes |
_RESULTS_ROW_RE = re.compile(
    r"^\|\s*`([a-z_]+)`\s*\|\s*(Tier_\d)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*(PASS|FAIL)\s*\|\s*(Yes|No)\s*\|\s*$",
    re.MULTILINE,
)


def _parse_results_table(design_md_text: str) -> dict[str, str]:
    """Parse the Results table in design.md into {organism: 'PASS'/'FAIL'}.

    Looks for rows matching the `| `organism` | Tier_N | spearman | recall
    | PASS/FAIL | Yes/No |` schema introduced in Component 4's Results
    table. Returns a dict mapping each organism string to its recorded
    Accuracy_Gate outcome.
    """
    results = {}
    for match in _RESULTS_ROW_RE.finditer(design_md_text):
        organism, tier, spearman, recall, gate, added = match.groups()
        results[organism] = gate
    return results


@pytest.fixture(scope='module')
def results_table():
    assert DESIGN_MD_PATH.exists(), f"design.md not found at {DESIGN_MD_PATH}"
    text = DESIGN_MD_PATH.read_text()
    table = _parse_results_table(text)
    # Sanity check: the harness was run against 18 organism combinations
    # per task 11; fail loudly if parsing picked up fewer rows than that,
    # since a silent partial-parse would make this test vacuously weak.
    assert len(table) == 18, (
        f"Expected 18 parsed Results table rows, got {len(table)}: {sorted(table)}"
    )
    return table


def test_supported_organisms_match_passing_rows(results_table):
    """SUPPORTED_ORGANISMS must equal exactly the set of PASS-rated organisms."""
    passing_organisms = {org for org, gate in results_table.items() if gate == 'PASS'}
    failing_organisms = {org for org, gate in results_table.items() if gate == 'FAIL'}

    # Every organism in SUPPORTED_ORGANISMS has recorded, passing evidence.
    unsupported_evidence = SUPPORTED_ORGANISMS - passing_organisms
    assert not unsupported_evidence, (
        f"SUPPORTED_ORGANISMS contains organisms with no recorded PASS row: "
        f"{sorted(unsupported_evidence)}"
    )

    # No organism with a FAIL row is included.
    wrongly_included = SUPPORTED_ORGANISMS & failing_organisms
    assert not wrongly_included, (
        f"SUPPORTED_ORGANISMS includes organisms with a recorded FAIL row: "
        f"{sorted(wrongly_included)}"
    )

    # Exact equality: every PASS organism is included too (binary rule,
    # Requirement 7.3), not just a subset relationship.
    assert SUPPORTED_ORGANISMS == passing_organisms, (
        f"SUPPORTED_ORGANISMS does not exactly match the PASS-rated rows.\n"
        f"Missing from SUPPORTED_ORGANISMS: {sorted(passing_organisms - SUPPORTED_ORGANISMS)}\n"
        f"Extra in SUPPORTED_ORGANISMS: {sorted(SUPPORTED_ORGANISMS - passing_organisms)}"
    )


def test_rhesus_subject_to_same_rule_as_new_candidates(results_table):
    """`rhesus`'s pre-existing inclusion is governed by its own row, not
    grandfathered in -- Requirement 7.5."""
    assert 'rhesus' in results_table, "rhesus must have a recorded Results row"
    rhesus_gate = results_table['rhesus']
    if rhesus_gate == 'PASS':
        assert 'rhesus' in SUPPORTED_ORGANISMS
    else:
        assert 'rhesus' not in SUPPORTED_ORGANISMS
