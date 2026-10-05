"""
CLI argparse smoke tests for the `--organism` `choices=` extension made to
`scripts/run_conga.py` and `scripts/setup_10x_for_conga.py` by the
tcrdist-db-update feature (design Component 2).

Both scripts build their `argparse.ArgumentParser` and call
`parser.parse_args()` at module level (not guarded by `if __name__ ==
'__main__':`), so the parser cannot be imported and exercised in-process
without triggering the rest of each script's module-level execution.
These tests therefore invoke each script as a subprocess, matching the
convention already established in `tests/test_run_conga_cli.py`.

For a representative sample of Newly_Supported_Organism values (`cat`,
`dog_gd`, `rabbit_ig`, `sheep`), each script is invoked with just enough
other required arguments that argument *parsing* completes -- i.e. the
process must not exit with argparse's choices-rejection message (exit code
2, "invalid choice") for `--organism`. Both scripts fail later for
unrelated reasons (missing GEX/clones data), which is expected and not
what this test checks.

Requirements: 3.1, 3.2
"""

import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_CONGA_SCRIPT = REPO_ROOT / 'scripts' / 'run_conga.py'
SETUP_10X_SCRIPT = REPO_ROOT / 'scripts' / 'setup_10x_for_conga.py'

NEWLY_SUPPORTED_ORGANISM_SAMPLE = ['cat', 'dog_gd', 'rabbit_ig', 'sheep']


def _run_script(script_path, args, cwd=None):
    cmd = [sys.executable, str(script_path)] + args
    return subprocess.run(cmd, capture_output=True, text=True,
                           cwd=cwd or str(REPO_ROOT), timeout=120)


def _argparse_rejected_organism_choice(result):
    """True if argparse itself rejected --organism (its own 'invalid
    choice' error on stderr), as opposed to the script failing later for
    an unrelated reason (missing data files, etc.).
    """
    return 'invalid choice' in result.stderr and '--organism' in result.stderr


class TestRunCongaOrganismChoices:
    """Requirement 3.1: `scripts/run_conga.py --organism` accepts every
    Newly_Supported_Organism value without an argparse 'invalid choice'
    rejection (no SystemExit from argparse's choices validation).
    """

    @pytest.mark.parametrize('organism', NEWLY_SUPPORTED_ORGANISM_SAMPLE)
    def test_organism_accepted_by_argparse(self, tmp_path, organism):
        result = _run_script(
            RUN_CONGA_SCRIPT,
            ['--organism', organism,
             '--outfile_prefix', str(tmp_path / 'tmp_cli_test')],
        )

        assert not _argparse_rejected_organism_choice(result), (
            f"--organism {organism!r} was rejected by argparse choices=:\n"
            f"{result.stderr}"
        )


class TestSetup10xOrganismChoices:
    """Requirement 3.2: `scripts/setup_10x_for_conga.py --organism`
    accepts every Newly_Supported_Organism value without an argparse
    'invalid choice' rejection.
    """

    @pytest.mark.parametrize('organism', NEWLY_SUPPORTED_ORGANISM_SAMPLE)
    def test_organism_accepted_by_argparse(self, tmp_path, organism):
        result = _run_script(
            SETUP_10X_SCRIPT,
            ['--organism', organism,
             '--output_clones_file', str(tmp_path / 'tmp_clones.tsv'),
             '--input_clones_file', str(tmp_path / 'tmp_input_clones.tsv')],
        )

        assert not _argparse_rejected_organism_choice(result), (
            f"--organism {organism!r} was rejected by argparse choices=:\n"
            f"{result.stderr}"
        )
