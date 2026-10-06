"""Unit test for Requirement 2 (Requirement 9.2) of the
metaconga-match-integration feature.

Confirms that all 11 Metaconga_Data_Files were bundled verbatim into
`conga/data/metaconga/`, as required for `conga.metaconga_match`'s
module-level data loading to succeed at import time. File-presence
verification alone is sufficient acceptance evidence per Requirement 2.2;
no checksum or row-count verification is required here.
"""

import pytest

from conga import util


METACONGA_DATA_FILES = [
    'big_combo_tcrs_2024-02-02a_gp4_MCC10_NG200_groups_info_MCC10_extras.tsv',
    'big_combo_tcrs_2024-02-02a_gp4_MCC10_NG200_groups_info_MCC10_extras_new_matches.tsv',
    'big_combo_tcrs_2024-02-02a_gp4_ten_tcrs.tsv',
    'cdr3aa_bias_cluster_names.tsv',
    'good_clumps_v1.tsv',
    'hsgenes_1000_plus_cdr3aa_bias_top30_degs.tsv',
    'hsgenes_200_plus_cdr3aa_bias_top30_degs.tsv',
    'round8_v5cd4_run84_xribo_200_process_v1_leiden2_cdr3aa_sig_1e-06_cluster_feature_enrichments.tsv',
    'round8_v5cd8_run84_xribo_200_process_v1_leiden2_cdr3aa_sig_1e-06_cluster_feature_enrichments.tsv',
    'run105_cd4_deg_results.tsv',
    'run106_cd8_deg_results.tsv',
]


@pytest.mark.parametrize('filename', METACONGA_DATA_FILES)
def test_metaconga_data_file_present(filename):
    """Each of the 11 Metaconga_Data_Files exists under conga/data/metaconga/."""
    filepath = util.path_to_data / 'metaconga' / filename
    assert filepath.is_file(), f"Missing metaconga data file: {filepath}"


def test_metaconga_data_dir_has_exactly_eleven_files():
    """conga/data/metaconga/ contains exactly the 11 expected files (no
    extras, nothing missing), confirming a clean verbatim bundle."""
    metaconga_dir = util.path_to_data / 'metaconga'
    actual_files = {p.name for p in metaconga_dir.glob('*.tsv')}
    assert actual_files == set(METACONGA_DATA_FILES)
