"""
Property-based test for validation-corpus gene-id resolution.

Covers task 9.1 of the tcrdist-db-update spec: for every validation
data-source function in `conga.tcrdist.vectorized` (the three real-data
loaders and the Synthetic_CDR3_Generator), every V/J gene id it produces
must be a valid key in `all_genes[organism]` for that organism's V/J
gene segments -- the exact invariant `encode_tcrs`'s bare
`gene_to_row[v_gene]` dict lookup depends on.

**Property 3: Validation corpus gene ids always resolve**
**Validates: Requirements 5.2, 6.4**
"""

import pytest
from hypothesis import given, settings, strategies as st

from conga.tcrdist.all_genes import all_genes
from conga.tcrdist.vectorized import (
    _load_human_cdr3_corpus,
    _load_mouse_cdr3_corpus,
    _load_rhesus_cdr3_corpus,
    _build_synthetic_cdr3_corpus,
)


def _gene_id_sets(organism):
    """Return (va_ids, ja_ids, vb_ids, jb_ids) sets for an organism."""
    genes_dict = all_genes[organism]
    va_ids = {g.id for g in genes_dict.values() if g.chain == 'A' and g.region == 'V'}
    ja_ids = {g.id for g in genes_dict.values() if g.chain == 'A' and g.region == 'J'}
    vb_ids = {g.id for g in genes_dict.values() if g.chain == 'B' and g.region == 'V'}
    jb_ids = {g.id for g in genes_dict.values() if g.chain == 'B' and g.region == 'J'}
    return va_ids, ja_ids, vb_ids, jb_ids


# Synthetic Tier 3 organisms (new species + rhesus's gd/ig receptor types).
# Not every receptor-type suffix necessarily exists in the gene database for
# every species; SYNTHETIC_ORGANISMS is filtered to those actually present.
_SYNTHETIC_ORGANISM_CANDIDATES = [
    'cat', 'cat_gd', 'cat_ig',
    'dog', 'dog_gd', 'dog_ig',
    'ferret', 'ferret_gd', 'ferret_ig',
    'rabbit', 'rabbit_gd', 'rabbit_ig',
    'sheep',
    'rhesus_gd', 'rhesus_ig',
]
SYNTHETIC_ORGANISMS = [o for o in _SYNTHETIC_ORGANISM_CANDIDATES if o in all_genes]


@pytest.mark.parametrize('organism', SYNTHETIC_ORGANISMS)
def test_synthetic_corpus_gene_ids_resolve(organism):
    """Every sampled gene id for synthetic organisms is a valid gene key."""
    va_ids, ja_ids, vb_ids, jb_ids = _gene_id_sets(organism)

    corpus = _build_synthetic_cdr3_corpus(organism, n=50, random_seed=42)
    assert len(corpus) == 50

    for atcr, btcr in corpus:
        va_gene, ja_gene, cdr3a, nucseq_a = atcr
        vb_gene, jb_gene, cdr3b, nucseq_b = btcr

        assert va_gene in va_ids, f"{va_gene!r} not a valid V-alpha gene for {organism!r}"
        assert ja_gene in ja_ids, f"{ja_gene!r} not a valid J-alpha gene for {organism!r}"
        assert vb_gene in vb_ids, f"{vb_gene!r} not a valid V-beta gene for {organism!r}"
        assert jb_gene in jb_ids, f"{jb_gene!r} not a valid J-beta gene for {organism!r}"

        assert cdr3a and isinstance(cdr3a, str)
        assert cdr3b and isinstance(cdr3b, str)
        assert nucseq_a == ''
        assert nucseq_b == ''


@settings(max_examples=100, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_synthetic_corpus_gene_ids_resolve_property(seed):
    """Hypothesis-driven check across many seeds for a representative organism."""
    organism = 'cat'
    assert organism in all_genes, "cat must be present in the gene database for this test"
    va_ids, ja_ids, vb_ids, jb_ids = _gene_id_sets(organism)

    corpus = _build_synthetic_cdr3_corpus(organism, n=10, random_seed=seed)

    for atcr, btcr in corpus:
        va_gene, ja_gene, _, _ = atcr
        vb_gene, jb_gene, _, _ = btcr
        assert va_gene in va_ids
        assert ja_gene in ja_ids
        assert vb_gene in vb_ids
        assert jb_gene in jb_ids


def test_human_corpus_gene_ids_resolve():
    """Every va/ja/vb/jb gene id in the real human corpus resolves."""
    va_ids, ja_ids, vb_ids, jb_ids = _gene_id_sets('human')
    df = _load_human_cdr3_corpus()

    assert len(df) > 0
    assert set(df['va'].unique()) <= va_ids
    assert set(df['ja'].unique()) <= ja_ids
    assert set(df['vb'].unique()) <= vb_ids
    assert set(df['jb'].unique()) <= jb_ids


def test_mouse_corpus_gene_ids_resolve():
    """Every va/ja/vb/jb gene id in the real mouse corpus resolves."""
    va_ids, ja_ids, vb_ids, jb_ids = _gene_id_sets('mouse')
    df = _load_mouse_cdr3_corpus()

    assert len(df) > 0
    assert set(df['va'].unique()) <= va_ids
    assert set(df['ja'].unique()) <= ja_ids
    assert set(df['vb'].unique()) <= vb_ids
    assert set(df['jb'].unique()) <= jb_ids


def test_rhesus_corpus_gene_ids_resolve():
    """Every va_gene/ja_gene/vb_gene/jb_gene id in the real rhesus corpus resolves."""
    va_ids, ja_ids, vb_ids, jb_ids = _gene_id_sets('rhesus')
    df = _load_rhesus_cdr3_corpus()

    assert len(df) == 450
    assert set(df['va_gene'].unique()) <= va_ids
    assert set(df['ja_gene'].unique()) <= ja_ids
    assert set(df['vb_gene'].unique()) <= vb_ids
    assert set(df['jb_gene'].unique()) <= jb_ids
