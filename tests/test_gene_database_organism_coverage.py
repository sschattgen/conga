"""Unit test for Requirement 1.3 of the tcrdist-db-update feature.

Confirms that after repointing `conga.tcrdist.basic.db_file` to
`combo_xcr_2026-08-06.tsv`, the Gene_Database (`conga.tcrdist.all_genes.all_genes`)
contains every organism string present in that file, including every
Newly_Supported_Organism and every organism excluded by the
Chain_Completeness_Rule (the Gene_Database itself is not filtered; only
downstream consumer code paths are scoped to Newly_Supported_Organism).
"""

import importlib
import os
import sys

import pandas as pd
import pytest

import conga.tcrdist


NEWLY_SUPPORTED_ORGANISMS = [
    'cat', 'cat_gd', 'cat_ig',
    'dog', 'dog_gd', 'dog_ig',
    'ferret', 'ferret_gd', 'ferret_ig',
    'rabbit', 'rabbit_gd', 'rabbit_ig',
    'sheep',
    'rhesus_ig',
]

CHAIN_COMPLETENESS_EXCLUDED_ORGANISMS = [
    'rainbowtrout',
    'rainbowtrout_ig',
    'sheep_gd',
    'sheep_ig',
]


@pytest.fixture(scope="module")
def fresh_all_genes():
    """Import conga.tcrdist.all_genes fresh so it rebuilds from the
    current conga.tcrdist.basic.db_file setting, rather than reusing a
    module possibly imported (and cached) before this change.
    """
    for mod_name in list(sys.modules):
        if mod_name == 'conga.tcrdist.all_genes' or mod_name == 'conga.tcrdist.basic':
            del sys.modules[mod_name]

    all_genes_module = importlib.import_module('conga.tcrdist.all_genes')
    return all_genes_module.all_genes


@pytest.fixture(scope="module")
def db_file_organisms():
    """Independently read the organism column of combo_xcr_2026-08-06.tsv
    (the Active_Database after this feature) to avoid relying on the
    same loading code path we are testing.
    """
    tcrdist_pkg_dir = os.path.dirname(os.path.realpath(conga.tcrdist.__file__))
    db_path = os.path.join(tcrdist_pkg_dir, 'db', 'combo_xcr_2026-08-06.tsv')
    df = pd.read_csv(db_path, sep='\t')
    return sorted(df['organism'].unique())


def test_active_database_is_the_2026_file():
    from conga.tcrdist import basic
    assert basic.db_file == 'combo_xcr_2026-08-06.tsv'


def test_gene_database_contains_every_organism_in_active_database(fresh_all_genes, db_file_organisms):
    for organism in db_file_organisms:
        assert organism in fresh_all_genes, (
            f"Gene_Database (all_genes) is missing organism {organism!r} "
            f"present in combo_xcr_2026-08-06.tsv"
        )


@pytest.mark.parametrize("organism", NEWLY_SUPPORTED_ORGANISMS)
def test_gene_database_contains_newly_supported_organism(fresh_all_genes, organism):
    assert organism in fresh_all_genes
    assert len(fresh_all_genes[organism]) > 0


@pytest.mark.parametrize("organism", CHAIN_COMPLETENESS_EXCLUDED_ORGANISMS)
def test_gene_database_contains_chain_completeness_excluded_organism(fresh_all_genes, organism):
    # The Gene_Database itself is not filtered by the Chain_Completeness_Rule;
    # only downstream consumer code paths are scoped to Newly_Supported_Organism.
    assert organism in fresh_all_genes
    assert len(fresh_all_genes[organism]) > 0
