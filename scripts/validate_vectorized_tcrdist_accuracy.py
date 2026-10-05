#!/usr/bin/env python3
"""
Vectorized TCRdist accuracy validation script.

Validates conga.tcrdist.vectorized's encode_tcrs/accuracy_report accuracy
against exact TCRdist, organism by organism, against the best available
CDR3 data for each organism (real paired data for Tier 1 organisms,
synthetic data modeled on human CDR3 content for Tier 3 organisms).

Usage:
    python scripts/validate_vectorized_tcrdist_accuracy.py --organism human
    python scripts/validate_vectorized_tcrdist_accuracy.py --organism cat --n-synthetic 500
    python scripts/validate_vectorized_tcrdist_accuracy.py --organism all --verbose
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path so `conga` is importable when run directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

import conga
import conga.util as util
import conga.tcrdist.vectorized as vectorized

logger = logging.getLogger(__name__)


# Validation_Tier table (design.md Component 4): organism -> tier label.
TIER_1_ORGANISMS = ('human', 'mouse', 'rhesus')

TIER_3_ORGANISMS = (
    'rhesus_gd', 'rhesus_ig',
    'cat', 'cat_gd', 'cat_ig',
    'dog', 'dog_gd', 'dog_ig',
    'ferret', 'ferret_gd', 'ferret_ig',
    'rabbit', 'rabbit_gd', 'rabbit_ig',
    'sheep',
)

ALL_ORGANISMS = TIER_1_ORGANISMS + TIER_3_ORGANISMS

ORGANISM_TO_TIER = {
    **{org: 'Tier_1' for org in TIER_1_ORGANISMS},
    **{org: 'Tier_3' for org in TIER_3_ORGANISMS},
}

# Accuracy_Gate thresholds (design.md Component 4).
SPEARMAN_THRESHOLD = 0.90
RECALL_AT_10_THRESHOLD = 0.70


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
    )


def _project_matching_db_rows(df, va_col, ja_col, cdr3a_col, vb_col, jb_col, cdr3b_col):
    """Project a matching-db-schema DataFrame to the common tuple shape.

    Common shape: list[tuple[tuple[str, str, str, str], tuple[str, str, str, str]]]
    i.e. ((va, ja, cdr3a, ''), (vb, jb, cdr3b, '')) per row, matching the
    4-element (v_gene, j_gene, cdr3, nucseq) shape _build_synthetic_cdr3_corpus
    already returns directly.
    """
    tcrs = []
    for _, row in df.iterrows():
        atcr = (row[va_col], row[ja_col], row[cdr3a_col], '')
        btcr = (row[vb_col], row[jb_col], row[cdr3b_col], '')
        tcrs.append((atcr, btcr))
    return tcrs


def load_tcrs_for_organism(organism: str, n_synthetic: int, random_seed: int):
    """Dispatch to the correct Validation_Harness data-source function.

    Returns a list[tuple[tuple, tuple]] in the common tuple shape, ready to
    pass straight to conga.tcrdist.vectorized.accuracy_report.
    """
    if organism == 'human':
        df = vectorized._load_human_cdr3_corpus()
        return _project_matching_db_rows(df, 'va', 'ja', 'cdr3a', 'vb', 'jb', 'cdr3b')
    elif organism == 'mouse':
        df = vectorized._load_mouse_cdr3_corpus()
        return _project_matching_db_rows(df, 'va', 'ja', 'cdr3a', 'vb', 'jb', 'cdr3b')
    elif organism == 'rhesus':
        df = vectorized._load_rhesus_cdr3_corpus()
        return _project_matching_db_rows(
            df, 'va_gene', 'ja_gene', 'cdr3a', 'vb_gene', 'jb_gene', 'cdr3b'
        )
    elif organism in TIER_3_ORGANISMS:
        return vectorized._build_synthetic_cdr3_corpus(
            organism, n_synthetic, random_seed
        )
    else:
        raise ValueError(
            f"Unrecognized organism {organism!r} for validation harness. "
            f"Supported organisms: {sorted(ALL_ORGANISMS)}"
        )


def validate_organism(organism: str, n_synthetic: int, random_seed: int) -> bool:
    """Run the Validation_Harness for a single organism and print the result.

    Returns
    -------
    bool
        True if the organism passed the Accuracy_Gate, False otherwise.
    """
    tier = ORGANISM_TO_TIER[organism]
    logger.info(f"Loading validation corpus for organism={organism!r} (tier={tier})")

    tcrs = load_tcrs_for_organism(organism, n_synthetic, random_seed)
    logger.info(f"Loaded {len(tcrs)} paired clonotypes for organism={organism!r}")

    # _skip_validation=True: the Validation_Harness's entire purpose is to measure
    # accuracy for candidate organisms (e.g. Tier 3) that are NOT yet in
    # SUPPORTED_ORGANISMS, so the production organism gate must be bypassed here.
    # This is an intentional, scoped exception -- the underlying encoding math has
    # no organism restriction of its own, it just needs gene ids to resolve in
    # all_genes. This bypass must never be used outside this validation tool.
    report = vectorized.accuracy_report(
        tcrs, organism, random_seed=random_seed, _skip_validation=True
    )

    recall_at_10 = report.mean_recall.get(10, float('nan'))
    passed = (
        report.spearman >= SPEARMAN_THRESHOLD
        and recall_at_10 >= RECALL_AT_10_THRESHOLD
    )

    status = 'PASS' if passed else 'FAIL'
    print(
        f"{organism:12s} {tier:8s} "
        f"spearman={report.spearman:.4f} "
        f"recall@10={recall_at_10:.4f} "
        f"n={report.num_clonotypes:5d} -> {status}"
    )

    return passed


def main():
    parser = argparse.ArgumentParser(
        description="Validate vectorized TCRdist accuracy for CoNGA organisms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --organism human              # Validate human only (real data)
  %(prog)s --organism cat                # Validate cat only (synthetic data)
  %(prog)s --organism all --verbose      # Validate every evaluated organism
        """.strip(),
    )

    parser.add_argument(
        '--organism',
        required=True,
        choices=list(ALL_ORGANISMS) + ['all'],
        help="Organism to validate, or 'all' to validate every evaluated organism",
    )
    parser.add_argument(
        '--n-synthetic', type=int, default=1000,
        help="Number of synthetic clonotypes to generate for Tier 3 organisms (default: 1000)",
    )
    parser.add_argument(
        '--random-seed', type=int, default=util.DEFAULT_RANDOM_SEED,
        help=f"Random seed for synthetic corpus generation and pair sampling (default: {util.DEFAULT_RANDOM_SEED})",
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Verbose logging',
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.organism == 'all':
        organisms = list(ALL_ORGANISMS)
    else:
        organisms = [args.organism]

    print("Vectorized TCRdist Accuracy Validation")
    print("=" * 60)
    print(f"{'organism':12s} {'tier':8s} metrics")
    print("-" * 60)

    results = {}
    for organism in organisms:
        try:
            results[organism] = validate_organism(
                organism, args.n_synthetic, args.random_seed
            )
        except Exception as e:
            print(f"{organism:12s} {'ERROR':8s} {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            results[organism] = False

    print("-" * 60)
    n_passed = sum(1 for passed in results.values() if passed)
    n_total = len(results)
    print(f"Summary: {n_passed}/{n_total} organisms passed the accuracy gate")

    all_passed = all(results.values())
    if all_passed:
        print("Validation PASSED for all requested organisms")
        exit_code = 0
    else:
        failed = [org for org, passed in results.items() if not passed]
        print(f"Validation FAILED for: {failed}")
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
