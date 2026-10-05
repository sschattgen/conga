"""
Unit test for the once-per-process Tier 3 Validation_Warning in `encode_tcrs`.

Covers task 13.1 of the tcrdist-db-update spec: `encode_tcrs` must emit a
`logger.warning(...)` the first time it is called for a Tier_3 organism
(synthetic-data-validated), and must not emit it again for that same
organism within the same process. `human`, `mouse`, and `rhesus` are
Tier_1 and must never emit the warning, regardless of call count, since
tier classification is independent of Accuracy_Gate pass/fail
(Resolved Decision 5).

**Property 4: Validation_Warning fires if and only if Tier_3**
**Validates: Requirements 8.1, 8.2, 8.3**
"""

import logging

import pytest

import conga.tcrdist.vectorized as vectorized
from conga.tcrdist.vectorized import (
    encode_tcrs,
    _build_synthetic_cdr3_corpus,
    _load_human_cdr3_corpus,
    _load_mouse_cdr3_corpus,
    _load_rhesus_cdr3_corpus,
)


@pytest.fixture(autouse=True)
def _reset_already_warned_organisms():
    """`_already_warned_organisms` is module-level global state; save/restore
    it around every test in this file so test order and repeated pytest
    runs within the same process don't cause false negatives from a
    previous test having already warned for the same organism.
    """
    saved = set(vectorized._already_warned_organisms)
    vectorized._already_warned_organisms.clear()
    yield
    vectorized._already_warned_organisms.clear()
    vectorized._already_warned_organisms.update(saved)


def _tier3_tcrs(organism, n=3, random_seed=42):
    """Build a small list of synthetic nested-tuple TCRs for a Tier_3 organism."""
    return _build_synthetic_cdr3_corpus(organism, n=n, random_seed=random_seed)


def _real_corpus_tcrs(organism, n=3):
    """Build a small list of nested-tuple TCRs from the real Tier_1 corpora."""
    if organism == 'human':
        df = _load_human_cdr3_corpus().head(n)
        va_col, ja_col, vb_col, jb_col = 'va', 'ja', 'vb', 'jb'
    elif organism == 'mouse':
        df = _load_mouse_cdr3_corpus().head(n)
        va_col, ja_col, vb_col, jb_col = 'va', 'ja', 'vb', 'jb'
    elif organism == 'rhesus':
        df = _load_rhesus_cdr3_corpus().head(n)
        va_col, ja_col, vb_col, jb_col = 'va_gene', 'ja_gene', 'vb_gene', 'jb_gene'
    else:
        raise ValueError(organism)

    tcrs = []
    for _, row in df.iterrows():
        atcr = (row[va_col], row[ja_col], row['cdr3a'], '')
        btcr = (row[vb_col], row[jb_col], row['cdr3b'], '')
        tcrs.append((atcr, btcr))
    return tcrs


@pytest.mark.parametrize('organism', sorted(vectorized._TIER_3_ORGANISMS))
def test_tier3_warning_fires_once_per_process(organism, caplog):
    """Calling encode_tcrs twice for the same Tier_3 organism warns only once."""
    tcrs = _tier3_tcrs(organism)

    with caplog.at_level(logging.WARNING, logger=vectorized.logger.name):
        caplog.clear()
        encode_tcrs(tcrs, organism)
        first_call_warnings = [
            r for r in caplog.records if organism in r.getMessage() and r.levelno == logging.WARNING
        ]
        assert len(first_call_warnings) == 1, (
            f"Expected exactly one warning on first call for {organism!r}, "
            f"got {len(first_call_warnings)}"
        )

        caplog.clear()
        encode_tcrs(tcrs, organism)
        second_call_warnings = [
            r for r in caplog.records if organism in r.getMessage() and r.levelno == logging.WARNING
        ]
        assert len(second_call_warnings) == 0, (
            f"Expected no warning on second call for {organism!r} (already warned), "
            f"got {len(second_call_warnings)}"
        )


@pytest.mark.parametrize('organism', ['human', 'mouse', 'rhesus'])
def test_tier1_organisms_never_warn(organism, caplog):
    """human/mouse/rhesus must never emit the Tier 3 Validation_Warning,
    regardless of call count."""
    tcrs = _real_corpus_tcrs(organism)

    with caplog.at_level(logging.WARNING, logger=vectorized.logger.name):
        for _ in range(2):
            caplog.clear()
            encode_tcrs(tcrs, organism)
            tier3_warnings = [
                r for r in caplog.records
                if 'validated using synthetic sequence data' in r.getMessage()
            ]
            assert tier3_warnings == [], (
                f"Tier_1 organism {organism!r} must never emit the Tier 3 "
                f"Validation_Warning, got: {[r.getMessage() for r in tier3_warnings]}"
            )
        assert organism not in vectorized._already_warned_organisms
