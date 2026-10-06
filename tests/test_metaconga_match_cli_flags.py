"""CLI argparse smoke tests for the two new metaconga-match flags added
to `scripts/run_conga.py` by the metaconga-match-integration feature
(design Component 5, Requirement 4).

`scripts/run_conga.py` builds its `argparse.ArgumentParser` and calls
`parser.parse_args()` at module level, so each invocation is run as a
subprocess, matching the convention already established in
`tests/test_organism_cli_choices.py` and `tests/test_run_conga_cli.py`.

These tests exercise argparse's own `choices=` validation for
`--match_metaconga_aaclusters` and confirm `--match_metaconga_clumps` is
accepted as a plain store_true flag. They do not exercise the early
cross-flag validation block (Requirement 5-7, covered separately in
tests/test_metaconga_match_validation.py) or analysis dispatch
(Requirement 8); invocations here are expected to fail later for
unrelated reasons (missing GEX/clones data) once past argument parsing.
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


def _argparse_rejected(result, flag):
    """True if argparse itself rejected `flag` (its own 'invalid choice'
    error on stderr), as opposed to the script failing later for an
    unrelated reason (missing data files, validation sys.exit, etc.).
    """
    return 'invalid choice' in result.stderr and flag in result.stderr


class TestMatchMetacongaAaclustersChoices:
    """Requirement 4.1, 4.3: --match_metaconga_aaclusters accepts
    'cd4'/'cd8'/'CD4'/'CD8' (and omission), and rejects any other value
    via argparse's standard choices mechanism, before any analysis code
    runs.
    """

    @pytest.mark.parametrize('value', ['cd4', 'cd8', 'CD4', 'CD8'])
    def test_accepted_values_pass_argparse(self, tmp_path, value):
        result = _run_script([
            '--match_metaconga_aaclusters', value,
            '--subset_to_CD4_cells',
            '--subset_to_CD8_cells',
            '--organism', 'human',
            '--outfile_prefix', str(tmp_path / 'tmp_cli_test'),
        ])
        assert not _argparse_rejected(result, '--match_metaconga_aaclusters'), (
            f"--match_metaconga_aaclusters {value!r} was rejected by "
            f"argparse choices=:\n{result.stderr}"
        )

    def test_omission_is_accepted(self, tmp_path):
        """Omitting the flag entirely (default=None) must not be
        rejected by argparse."""
        result = _run_script([
            '--outfile_prefix', str(tmp_path / 'tmp_cli_test'),
        ])
        assert not _argparse_rejected(result, '--match_metaconga_aaclusters')

    @pytest.mark.parametrize('bad_value', ['cd5', 'CD9', 'nk', 'bogus'])
    def test_invalid_value_rejected_by_argparse(self, tmp_path, bad_value):
        result = _run_script([
            '--match_metaconga_aaclusters', bad_value,
            '--outfile_prefix', str(tmp_path / 'tmp_cli_test'),
        ])
        assert result.returncode == 2, (
            f"Expected argparse to reject --match_metaconga_aaclusters "
            f"{bad_value!r} with exit code 2, got {result.returncode}:\n"
            f"{result.stderr}"
        )
        assert 'invalid choice' in result.stderr


class TestMatchMetacongaClumpsFlag:
    """Requirement 4.2: --match_metaconga_clumps is a plain store_true
    flag, accepted by argparse with no value, defaulting to False."""

    def test_flag_accepted_by_argparse(self, tmp_path):
        result = _run_script([
            '--match_metaconga_clumps',
            '--organism', 'human',
            '--outfile_prefix', str(tmp_path / 'tmp_cli_test'),
        ])
        assert 'unrecognized argument' not in result.stderr
        assert 'invalid choice' not in result.stderr

    def test_omission_is_accepted(self, tmp_path):
        result = _run_script([
            '--outfile_prefix', str(tmp_path / 'tmp_cli_test'),
        ])
        assert 'unrecognized argument' not in result.stderr


class TestNewFlagsNotInAllModes:
    """Requirement 4.5: neither new flag is added to the `all_modes` list
    consulted when --all is supplied."""

    def test_all_modes_list_excludes_new_flags(self):
        source = RUN_CONGA_SCRIPT.read_text()
        # Extract the all_modes block defined in the --all handling.
        start = source.index('all_modes = """')
        end = source.index('""".split()', start)
        all_modes_block = source[start:end]
        assert 'match_metaconga_aaclusters' not in all_modes_block
        assert 'match_metaconga_clumps' not in all_modes_block
