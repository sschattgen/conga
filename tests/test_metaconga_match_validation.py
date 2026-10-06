"""Integration tests for the early cross-flag validation block added to
`scripts/run_conga.py` by the metaconga-match-integration feature
(design Component 6, Requirements 5, 6, 7).

`scripts/run_conga.py` builds its parser and runs its validation logic
at module level, so each case is exercised as a subprocess, matching the
convention in tests/test_organism_cli_choices.py and
tests/test_metaconga_match_cli_flags.py.

Each test supplies the minimal set of flags needed to reach the specific
check under test; later pipeline stages (missing GEX/clones data) are
expected to fail afterward for unrelated reasons once a case is NOT
supposed to be blocked by validation -- those cases only assert that the
specific `sys.exit` under test did NOT fire, not that the whole pipeline
succeeds.
"""

import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_CONGA_SCRIPT = REPO_ROOT / 'scripts' / 'run_conga.py'


def _run_script(args, cwd=None, timeout=120):
    cmd = [sys.executable, str(RUN_CONGA_SCRIPT)] + args
    return subprocess.run(cmd, capture_output=True, text=True,
                           cwd=cwd or str(REPO_ROOT), timeout=timeout)


class TestMutualExclusionWithBatchIntegration:
    """Requirement 5.4-5.6, design Component 6 check (a): setting
    --match_metaconga_aaclusters together with --batch_key or
    --batch_integration_method is a hard sys.exit, naming the AACluster
    flag as a Fixed_HVG_Pathway trigger even though --force_variable_genes
    was never typed by the user.
    """

    def test_aaclusters_with_batch_key_and_method_exits(self, tmp_path):
        result = _run_script([
            '--match_metaconga_aaclusters', 'cd4',
            '--subset_to_CD4_cells',
            '--organism', 'human',
            '--batch_key', 'batch',
            '--batch_integration_method', 'harmony',
            '--outfile_prefix', str(tmp_path / 'tmp_cli_test'),
        ])
        assert result.returncode != 0
        assert 'mutually exclusive' in result.stdout + result.stderr
        assert '--match_metaconga_aaclusters' in result.stdout + result.stderr

    def test_aaclusters_without_batch_flags_does_not_hit_this_check(self, tmp_path):
        """Negative control: without batch flags, this specific check
        must not fire (other, later checks/failures are fine)."""
        result = _run_script([
            '--match_metaconga_aaclusters', 'cd4',
            '--subset_to_CD4_cells',
            '--organism', 'human',
            '--outfile_prefix', str(tmp_path / 'tmp_cli_test'),
        ])
        combined = result.stdout + result.stderr
        assert 'Full_Integration_Pathway' not in combined or 'mutually exclusive' not in combined


class TestCdSubsetPairingRule:
    """Requirement 6: cd4/cd8 AACluster_Flag values require the matching
    --subset_to_CD4_cells/--subset_to_CD8_cells flag, enforced as a hard
    sys.exit rather than Source_Branch's non-blocking warning banner.
    """

    def test_cd4_without_subset_flag_exits(self, tmp_path):
        result = _run_script([
            '--match_metaconga_aaclusters', 'cd4',
            '--organism', 'human',
            '--outfile_prefix', str(tmp_path / 'tmp_cli_test'),
        ])
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert '--subset_to_CD4_cells' in combined
        assert 'ERROR' in combined

    def test_cd8_without_subset_flag_exits(self, tmp_path):
        result = _run_script([
            '--match_metaconga_aaclusters', 'cd8',
            '--organism', 'human',
            '--outfile_prefix', str(tmp_path / 'tmp_cli_test'),
        ])
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert '--subset_to_CD8_cells' in combined
        assert 'ERROR' in combined

    def test_cd4_with_subset_flag_does_not_hit_this_check(self, tmp_path):
        result = _run_script([
            '--match_metaconga_aaclusters', 'cd4',
            '--subset_to_CD4_cells',
            '--organism', 'human',
            '--outfile_prefix', str(tmp_path / 'tmp_cli_test'),
        ])
        combined = result.stdout + result.stderr
        assert '--subset_to_CD4_cells requires' not in combined
        assert 'requires --subset_to_CD4_cells' not in combined

    def test_cd8_with_subset_flag_does_not_hit_this_check(self, tmp_path):
        result = _run_script([
            '--match_metaconga_aaclusters', 'cd8',
            '--subset_to_CD8_cells',
            '--organism', 'human',
            '--outfile_prefix', str(tmp_path / 'tmp_cli_test'),
        ])
        combined = result.stdout + result.stderr
        assert 'requires --subset_to_CD8_cells' not in combined

    @pytest.mark.parametrize('value', ['CD4', 'CD8'])
    def test_uppercase_value_is_lowercased_before_pairing_check(self, tmp_path, value):
        """Requirement 4.4: CD4/CD8 (uppercase) must be lowercased before
        this check runs, so passing the matching subset flag (also
        case-correct) must not trigger the pairing error."""
        subset_flag = f'--subset_to_{value.upper()}_cells'
        result = _run_script([
            '--match_metaconga_aaclusters', value,
            subset_flag,
            '--organism', 'human',
            '--outfile_prefix', str(tmp_path / 'tmp_cli_test'),
        ])
        combined = result.stdout + result.stderr
        assert 'requires --subset_to_CD4_cells' not in combined
        assert 'requires --subset_to_CD8_cells' not in combined


class TestMetacongaOrganismGate:
    """Requirement 7: both new flags require --organism human, enforced
    as a fail-fast sys.exit checked directly against args.organism,
    before adata is constructed."""

    def test_aaclusters_with_non_human_organism_exits(self, tmp_path):
        result = _run_script([
            '--match_metaconga_aaclusters', 'cd4',
            '--subset_to_CD4_cells',
            '--organism', 'mouse',
            '--outfile_prefix', str(tmp_path / 'tmp_cli_test'),
        ])
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert '--match_metaconga_aaclusters' in combined
        assert '--organism human' in combined

    def test_clumps_with_non_human_organism_exits(self, tmp_path):
        result = _run_script([
            '--match_metaconga_clumps',
            '--organism', 'mouse',
            '--outfile_prefix', str(tmp_path / 'tmp_cli_test'),
        ])
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert '--match_metaconga_clumps' in combined
        assert '--organism human' in combined

    def test_aaclusters_with_human_organism_does_not_hit_this_check(self, tmp_path):
        result = _run_script([
            '--match_metaconga_aaclusters', 'cd4',
            '--subset_to_CD4_cells',
            '--organism', 'human',
            '--outfile_prefix', str(tmp_path / 'tmp_cli_test'),
        ])
        combined = result.stdout + result.stderr
        assert 'requires --organism human' not in combined

    def test_clumps_with_human_organism_does_not_hit_this_check(self, tmp_path):
        result = _run_script([
            '--match_metaconga_clumps',
            '--organism', 'human',
            '--outfile_prefix', str(tmp_path / 'tmp_cli_test'),
        ])
        combined = result.stdout + result.stderr
        assert 'requires --organism human' not in combined

    def test_neither_flag_set_does_not_enforce_organism_gate(self, tmp_path):
        """Negative control: without either new flag, no organism
        restriction from this feature applies."""
        result = _run_script([
            '--organism', 'mouse',
            '--outfile_prefix', str(tmp_path / 'tmp_cli_test'),
        ])
        combined = result.stdout + result.stderr
        assert '--match_metaconga_aaclusters requires --organism human' not in combined
        assert '--match_metaconga_clumps requires --organism human' not in combined


class TestAutoInjectionBehavior:
    """Requirement 5.1-5.3: when the AACluster_Flag is set and
    --force_variable_genes was not supplied, args.force_variable_genes is
    auto-set to the Bundled_AACluster_HVG_File and a WARNING is printed;
    a user-supplied --force_variable_genes value is left untouched and no
    warning is printed.
    """

    def test_auto_injection_warning_printed_when_not_supplied(self, tmp_path):
        result = _run_script([
            '--match_metaconga_aaclusters', 'cd4',
            '--subset_to_CD4_cells',
            '--organism', 'human',
            '--outfile_prefix', str(tmp_path / 'tmp_cli_test'),
        ])
        combined = result.stdout + result.stderr
        assert 'WARNING' in combined
        assert '--match_metaconga_aaclusters' in combined
        assert 'force_variable_genes' in combined
        assert 'hsgenes_1000_plus_cdr3aa_bias_top30_degs.tsv' in combined

    def test_user_supplied_force_variable_genes_left_untouched(self, tmp_path):
        """When the user supplies --force_variable_genes explicitly, the
        auto-injection warning must not be printed and the user's value
        must not be overridden."""
        custom_file = tmp_path / 'my_custom_genes.txt'
        custom_file.write_text('GENE1\nGENE2\n')

        result = _run_script([
            '--match_metaconga_aaclusters', 'cd4',
            '--subset_to_CD4_cells',
            '--organism', 'human',
            '--force_variable_genes', str(custom_file),
            '--outfile_prefix', str(tmp_path / 'tmp_cli_test'),
        ])
        combined = result.stdout + result.stderr
        assert 'WARNING: --match_metaconga_aaclusters' not in combined
        # the bundled file name should not appear as an injected value
        assert 'adding --force_variable_genes' not in combined

    def test_no_auto_injection_when_aaclusters_flag_unset(self, tmp_path):
        result = _run_script([
            '--organism', 'human',
            '--outfile_prefix', str(tmp_path / 'tmp_cli_test'),
        ])
        combined = result.stdout + result.stderr
        assert 'WARNING: --match_metaconga_aaclusters' not in combined
