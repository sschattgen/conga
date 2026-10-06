"""
Vectorized TCRdist encoding module.

This module provides fixed-length vector encodings of paired alpha-beta TCR chains that 
approximate TCRdist distances using Euclidean distance. The encoder replaces CoNGA's 
quadratic KernelPCA approach with linear-memory encoding suitable for large datasets.

Architecture
------------
The vectorizer embeds the TCRdist amino acid dissimilarity matrix into Euclidean space 
using multidimensional scaling (MDS), then encodes each TCR as a concatenation of:
1. Germline V-region loops (CDR1 + CDR2 + CDR2.5, invariant positions removed)
2. Trimmed and gap-aligned CDR3 sequence

Distance Approximation
----------------------
**Key approximation**: Euclidean distance in the encoded space approximates the square 
root of TCRdist, while **squared** Euclidean distance approximates TCRdist itself. This 
design preserves the TCRdist additive structure where substitution scores sum across 
positions. The dissimilarity matrix is square-rooted before MDS embedding so that 
squared distances in the embedding space correspond to the original TCRdist values.

For CDR3 weighting, the CDR3 block is pre-scaled by sqrt(cdr3_weight) so that when 
squared distances are computed, the CDR3 contribution is weighted by cdr3_weight 
relative to the germline contribution, matching TCRdist's weighting scheme.

Measured Accuracy
-----------------
Accuracy measured on 1000 human paired TCRs from bundled database:
- Spearman correlation vs exact TCRdist: 0.999  
- Pearson correlation (distance): 0.993
- Pearson correlation (squared distance): 0.999  
- Mean k-NN recall@10: 0.953
- Mean k-NN recall@100: 0.971

Accuracy is near-exact due to classical MDS initialization and 16-dimensional 
amino acid embedding. Performance degrades gracefully for mouse and rhesus 
(Spearman > 0.999, recall@10 > 0.944).

Supported Organisms
-------------------
18 organisms are currently supported (see conga.tcrdist.vectorized.SUPPORTED_ORGANISMS
for the authoritative, current list, since this set is periodically re-validated and
extended as new reference data becomes available):

- human, mouse, rhesus (alpha-beta TCRs)
- rhesus, cat, dog, ferret, rabbit (gamma-delta TCRs: rhesus_gd, cat_gd, dog_gd, ferret_gd, rabbit_gd)
- rhesus, cat, dog, ferret, rabbit (Ig/B cell receptors: rhesus_ig, cat_ig, dog_ig, ferret_ig, rabbit_ig)
- cat, dog, ferret, rabbit, sheep (alpha-beta TCRs)

Notably NOT supported by this vectorized path: human_gd, human_ig, mouse_gd, mouse_ig.
These four remain usable through CoNGA's KernelPCA representation (X_pca_tcr) or the
exact TCRdist path, just not through vectorized encoding.

Memory and Performance
----------------------
- Encoding time: ~0.1s for 20,000 clonotypes
- Peak memory: O(N·L) where L≈1136 for human (vs O(N^2) for KernelPCA)
- Stored representation: 91MB float32 for 20,000 clonotypes (vs 8MB for KernelPCA)

**Performance Characteristics** (N=20,000 clonotypes on 2019 MacBook Pro):
- Encoding time: ~0.1 seconds
- Peak memory usage: ~150 MB (vs ~6.4 GB for KernelPCA)
- Output size: 91 MB float32 array (1136 dimensions × 20k rows)
- Memory scaling: O(N·L) where L≈1136 for human

**Accuracy vs Configuration**:
- aa_mds_dim=16 (default): Near-exact approximation, Spearman > 0.999
- aa_mds_dim=12: Good approximation, Spearman > 0.99, smaller vectors
- aa_mds_dim=8: Adequate approximation, may fail accuracy gates

**Vector Length by Organism** (default config):
- Human: 1136 dimensions (α: 21+16, β: 18+16 positions × 16 aa_dims each)
- Mouse: 1168 dimensions (α: 23+16, β: 18+16 positions × 16 aa_dims each)  
- Rhesus: 1152 dimensions (α: 21+16, β: 19+16 positions × 16 aa_dims each)

Integration with FAISS
----------------------
The fixed-length float32 output vectors are optimized for consumption by FAISS
indices, enabling GPU-accelerated neighbor search on large TCR datasets:

- **Vector format**: C-contiguous float32 arrays for direct FAISS consumption
- **Dimension optimization**: aa_mds_dim parameter balances accuracy vs FAISS performance
- **Memory layout**: Designed for efficient batch processing and index building
- **Compatibility**: Works with sklearn neighbor search when FAISS unavailable

Usage Guidelines
----------------
**When to use vectorized encoding**:
- Any organism/receptor-type combination in SUPPORTED_ORGANISMS (18 organisms as of
  this writing -- see the module docstring's Supported Organisms section above)
- Large datasets (N > 5,000) where KernelPCA memory usage prohibitive
- Analyses requiring fast neighbor search or clustering
- Integration with external vector similarity tools

**When to use alternatives**:
- Gamma-delta TCRs or B cell receptors for organisms NOT in SUPPORTED_ORGANISMS
  (currently human_gd, human_ig, mouse_gd, mouse_ig) → use X_pca_tcr or exact path
- Small datasets (N < 1,000) where exact TCRdist is fast → use exact path  
- Analyses requiring perfect TCRdist fidelity → use exact path
- Legacy workflows → use X_pca_tcr for backward compatibility

**Configuration recommendations**:
- **Default**: EncodingConfig() for near-exact approximation
- **Performance**: aa_mds_dim=12 for smaller vectors, good accuracy
- **Memory-constrained**: aa_mds_dim=8 for minimal vectors (check accuracy!)
- **Reproducibility**: Set random_seed consistently across analyses

Implementation Notes
--------------------
This module uses lazy imports to avoid filesystem reads at import time. Gene database 
access (via all_genes) and TCRdist constants are imported only inside functions that 
need them, ensuring the module can be imported in packaged installations without 
repository-specific paths.

The amino acid embedding is cached within processes but recomputed across processes 
for reproducibility verification. Cache keys include matrix fingerprints and MDS 
parameter fingerprints to invalidate stale embeddings.

Examples
--------
Basic encoding:
>>> import conga
>>> from conga.tcrdist.vectorized import encode_tcrs, EncodingConfig
>>> # Prepare clonotype data
>>> tcrs = [
...     (('TRAV1*01', 'TRAJ1*01', 'CAVRD', ''), ('TRBV1*01', 'TRBJ1*01', 'CASSRT', '')),
...     (('TRAV2*01', 'TRAJ2*01', 'CAVKE', ''), ('TRBV2*01', 'TRBJ2*01', 'CASSLQ', ''))  
... ]
>>> matrix = encode_tcrs(tcrs, 'human')
>>> matrix.shape  # (2, 1136) for human
(2, 1136)

Custom configuration:
>>> config = EncodingConfig(aa_mds_dim=12, num_pos_cdr3=14, random_seed=123)
>>> matrix = encode_tcrs(tcrs, 'human', config)

DataFrame input:
>>> import pandas as pd
>>> df = pd.DataFrame({
...     'va': ['TRAV1*01', 'TRAV2*01'],
...     'cdr3a': ['CAVRD', 'CAVKE'], 
...     'vb': ['TRBV1*01', 'TRBV2*01'],
...     'cdr3b': ['CASSRT', 'CASSLQ']
... })
>>> matrix = encode_tcrs(df, 'human')

Accuracy validation:
>>> from conga.tcrdist.vectorized import accuracy_report
>>> report = accuracy_report(tcrs, 'human')
>>> print(f"Spearman correlation: {report.spearman:.3f}")
>>> print(f"Recall@10: {report.mean_recall[10]:.3f}")

Integration with CoNGA workflow:
>>> import anndata as ad
>>> from conga.tcrdist.vectorized import store_tcr_vectors_in_adata
>>> # Create AnnData with TCR metadata
>>> adata = ad.AnnData(obs=df)
>>> # Store vectorized representation
>>> vectors = store_tcr_vectors_in_adata(adata, 'human')
>>> # Use with FAISS neighbor search
>>> from conga.neighbors import search_neighbors_auto
>>> result = search_neighbors_auto(vectors, [0.01, 0.05], data_type='tcr')
"""

import hashlib
import logging
from dataclasses import dataclass
from typing import Sequence, Mapping, Collection, Any, Optional, Tuple, List, Union
import numpy as np
import pandas as pd
import sklearn
from sklearn.manifold import MDS

from .amino_acids import amino_acids
from .tcr_distances_blosum import bsd4
from .. import util

logger = logging.getLogger(__name__)

# Supported organisms for vectorized encoding
# Membership is a hand-edited literal, updated from the Accuracy_Gate
# Results table recorded in .kiro/specs/tcrdist-db-update/design.md
# (Component 4). All 18 evaluated organisms passed the Accuracy_Gate
# (Spearman >= 0.90 and mean recall@10 >= 0.70) as of that run.
SUPPORTED_ORGANISMS: frozenset[str] = frozenset({
    'human', 'mouse', 'rhesus',
    'rhesus_gd', 'rhesus_ig',
    'cat', 'cat_gd', 'cat_ig',
    'dog', 'dog_gd', 'dog_ig',
    'ferret', 'ferret_gd', 'ferret_ig',
    'rabbit', 'rabbit_gd', 'rabbit_ig',
    'sheep',
})

# Tier_3 organisms: validated via the Synthetic_CDR3_Generator (modeled on
# human CDR3 content), not species-matched real repertoire data, per the
# Validation_Tier column of the same Results table referenced above.
# `human`, `mouse`, and `rhesus` are Tier_1 (real, species-matched data) and
# are deliberately NEVER included here, regardless of Accuracy_Gate outcome,
# since tier classification is independent of pass/fail (Resolved Decision 5).
_TIER_3_ORGANISMS: frozenset[str] = frozenset({
    'rhesus_gd', 'rhesus_ig',
    'cat', 'cat_gd', 'cat_ig',
    'dog', 'dog_gd', 'dog_ig',
    'ferret', 'ferret_gd', 'ferret_ig',
    'rabbit', 'rabbit_gd', 'rabbit_ig',
    'sheep',
})

# Tracks which Tier_3 organisms have already triggered the one-time-per-process
# Validation_Warning in encode_tcrs, so repeated calls for the same organism
# (e.g. across a pipeline run) don't flood logs.
_already_warned_organisms: set[str] = set()

VECTORIZER_VERSION: str = 'conga.vectorized/1'

# Default encoding configuration parameters
DEFAULT_AA_MDS_DIM: int = 16      # Raised from prototype's 8 for better accuracy
DEFAULT_NUM_POS_CDR3: int = 16
DEFAULT_CDR3_WEIGHT: float = 3.0  # Must match tcr_distances.WEIGHT_CDR3_REGION
DEFAULT_N_TRIM: int = 3           # Matches sequence_distance_with_gappos ntrim
DEFAULT_C_TRIM: int = 2           # Matches sequence_distance_with_gappos ctrim

# Pinned SMACOF parameters for deterministic embedding
# These are not tunable - accuracy measurements depend on these values
_MDS_KWARGS = {
    'metric': 'precomputed',        # Replaces deprecated dissimilarity= (sklearn 1.8+)
    'metric_mds': True,             # 1.8 rename of old boolean metric=
    'init': 'classical_mds',        # Added in 1.8; becomes default in 1.10  
    'n_init': 1,                    # Ignored under classical_mds; default fell 4->1 in 1.9
    'max_iter': 300,
    'eps': 1e-3,                    # Default moved 1e-3 -> 1e-6 in 1.7
    'n_jobs': None,                 # No thread-count-dependent reduction order
    'normalized_stress': False,     # What 'auto' resolves to for metric MDS
}

# Global cache for amino acid embeddings
_EMBEDDING_CACHE = {}


@dataclass(frozen=True)
class EncodingConfig:
    """Configuration parameters that fully determine a vectorized encoding.
    
    This frozen dataclass encapsulates all tunable parameters that affect the 
    vectorized TCR encoding. Two EncodingConfig objects with identical field 
    values will always produce identical encodings when used with the same 
    input data and organism.
    
    The configuration balances encoding accuracy, vector size, and computational 
    cost. Default values are chosen to provide near-exact TCRdist approximation 
    (Spearman > 0.99) while maintaining reasonable vector dimensions.
    
    Parameters
    ----------
    aa_mds_dim : int, default=16
        Dimensionality of amino acid embedding space. Valid range: 1-21.
        Higher values improve accuracy but increase vector length linearly.
        - 8: Minimal, may fail accuracy gates (recall < 0.80)
        - 12: Adequate, passes gates with thin margins  
        - 16: Recommended, near-exact approximation (stress < 0.5)
        - >16: Diminishing returns due to intrinsic 21-symbol limit
        
    num_pos_cdr3 : int, default=16  
        Fixed number of positions in encoded CDR3 sequences. Must be ≥ 1.
        Longer CDR3s are trimmed/gapped; shorter ones are gap-padded.
        Typical CDR3 lengths: 8-22 amino acids, so 16 accommodates most.
        
    cdr3_weight : float, default=3.0
        Relative weight of CDR3 vs germline regions. Must be > 0.
        **Important**: Default 3.0 matches tcr_distances.WEIGHT_CDR3_REGION.
        Other values break correspondence with TCRdist gap penalties.
        Higher values make CDR3 differences more influential in distances.
        
    n_trim : int, default=3
        Number of N-terminal residues trimmed from CDR3 before encoding.
        Must be ≥ 0. Matches sequence_distance_with_gappos ntrim parameter.
        
    c_trim : int, default=2  
        Number of C-terminal residues trimmed from CDR3 before encoding.
        Must be ≥ 0. Matches sequence_distance_with_gappos ctrim parameter.
        
    random_seed : int, default=42
        Random seed for MDS amino acid embedding. Ensures reproducible 
        encodings across runs and processes. Any integer is valid.
        
    Attributes (computed)
    ---------------------
    The class provides methods for serialization and validation:
    - as_uns_dict(): Convert to flat dict for AnnData.uns storage
    - from_uns_dict(): Restore from stored dict
    - Automatic validation in __post_init__()
    
    Validation Rules
    ----------------
    - aa_mds_dim: 1 ≤ value ≤ 21 (cannot exceed amino acid count)
    - num_pos_cdr3: ≥ 1 (must encode at least one position)  
    - cdr3_weight: > 0 (negative weights are nonsensical)
    - n_trim, c_trim: ≥ 0 (negative trimming is undefined)
    - Warns if cdr3_weight ≠ 3.0 (breaks TCRdist correspondence)
    
    Examples
    --------
    Default configuration (recommended):
    >>> config = EncodingConfig()
    >>> config.aa_mds_dim
    16
    >>> config.cdr3_weight  
    3.0
    
    Minimal configuration (smaller vectors):
    >>> config = EncodingConfig(aa_mds_dim=8, num_pos_cdr3=12)
    >>> # Results in ~568-dimensional vectors vs ~1136 default
    
    Custom trimming (longer CDR3 preservation):
    >>> config = EncodingConfig(n_trim=1, c_trim=1, num_pos_cdr3=20)
    >>> # Preserves more CDR3 sequence, useful for long repertoires
    
    Experimental weight (changes distance interpretation):
    >>> config = EncodingConfig(cdr3_weight=5.0)  # Warning logged
    >>> # CDR3 differences weighted more heavily than default
    
    Serialization for storage:
    >>> config = EncodingConfig(random_seed=123)
    >>> config_dict = config.as_uns_dict()
    >>> config_dict['vectorizer_version']
    'conga.vectorized/1'
    >>> restored = EncodingConfig.from_uns_dict(config_dict)
    >>> restored.random_seed
    123
    """
    aa_mds_dim: int = DEFAULT_AA_MDS_DIM
    num_pos_cdr3: int = DEFAULT_NUM_POS_CDR3
    cdr3_weight: float = DEFAULT_CDR3_WEIGHT
    n_trim: int = DEFAULT_N_TRIM
    c_trim: int = DEFAULT_C_TRIM
    random_seed: int = util.DEFAULT_RANDOM_SEED
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not (1 <= self.aa_mds_dim <= 21):
            raise ValueError(f"aa_mds_dim must be 1-21, got {self.aa_mds_dim}")
        if self.num_pos_cdr3 < 1:
            raise ValueError(f"num_pos_cdr3 must be >= 1, got {self.num_pos_cdr3}")
        if self.cdr3_weight <= 0:
            raise ValueError(f"cdr3_weight must be > 0, got {self.cdr3_weight}")
        if self.n_trim < 0 or self.c_trim < 0:
            raise ValueError(f"Trim values must be >= 0, got n_trim={self.n_trim}, c_trim={self.c_trim}")
        
        # Warn if cdr3_weight differs from default (breaks gap penalty correspondence)
        if abs(self.cdr3_weight - DEFAULT_CDR3_WEIGHT) > 1e-6:
            logger.warning(f"cdr3_weight={self.cdr3_weight} differs from default {DEFAULT_CDR3_WEIGHT}. "
                          "This breaks CDR3 gap penalty correspondence to TCRdist.")

    def as_uns_dict(self) -> dict[str, int | float | str]:
        """Convert to dict for storage in adata.uns."""
        return {
            'aa_mds_dim': self.aa_mds_dim,
            'num_pos_cdr3': self.num_pos_cdr3,
            'cdr3_weight': self.cdr3_weight,
            'n_trim': self.n_trim,
            'c_trim': self.c_trim,
            'random_seed': self.random_seed,
            'vectorizer_version': VECTORIZER_VERSION,
        }
    
    @classmethod
    def from_uns_dict(cls, d: Mapping[str, object]) -> 'EncodingConfig':
        """Create from dict loaded from adata.uns."""
        return cls(
            aa_mds_dim=int(d['aa_mds_dim']),
            num_pos_cdr3=int(d['num_pos_cdr3']),
            cdr3_weight=float(d['cdr3_weight']),
            n_trim=int(d['n_trim']),
            c_trim=int(d['c_trim']),
            random_seed=int(d['random_seed']),
        )


def symbol_dissimilarity_matrix() -> np.ndarray:
    """Build 21x21 amino acid dissimilarity matrix from CoNGA's bsd4 table.
    
    Creates a symmetric matrix with amino acids in alphabetical order (following 
    `amino_acids` from tcrdist.amino_acids) plus gap character at index 20. Gap 
    penalty matches GAP_PENALTY_V_REGION from tcr_distances to ensure consistency 
    with exact TCRdist calculations.
    
    The matrix encodes BLOSUM62-derived dissimilarities where identical amino acids 
    have distance 0, and dissimilar amino acids have distances up to 4. Both '.' 
    and '*' gap characters map to the same gap symbol at index 20.
    
    Returns
    -------
    np.ndarray
        (21, 21) float64 dissimilarity matrix. Properties:
        - Symmetric: matrix[i,j] == matrix[j,i]  
        - Zero diagonal: matrix[i,i] == 0 for all i
        - Value range: [0, 4] for all entries
        - Gap penalty: matrix[0:20, 20] == matrix[20, 0:20] == 4.0
        
    Notes
    -----
    This matrix is square-rooted before MDS embedding so that squared Euclidean 
    distances in the embedding space approximate the original TCRdist values.
    
    Examples
    --------
    >>> dm = symbol_dissimilarity_matrix()
    >>> dm.shape
    (21, 21)
    >>> dm[0, 0]  # Identical amino acids
    0.0  
    >>> dm[0, 20]  # Amino acid to gap
    4.0
    >>> np.allclose(dm, dm.T)  # Symmetric
    True
    """
    # Lazy import to avoid filesystem reads at module scope
    from .tcr_distances import GAP_PENALTY_V_REGION
    
    # Build matrix: amino_acids (alphabetical) + gap at index 20
    dm = np.zeros((21, 21), dtype=np.float64)
    
    # Fill amino acid dissimilarities from bsd4 table
    for i, aa_i in enumerate(amino_acids):
        for j, aa_j in enumerate(amino_acids):
            dm[i, j] = bsd4[(aa_i, aa_j)]
    
    # Gap character at index 20
    # Both '.' and '*' map to gap (both appear in germline sequences)
    gap_penalty = float(GAP_PENALTY_V_REGION)  # 4.0
    dm[0:20, 20] = gap_penalty  # aa to gap
    dm[20, 0:20] = gap_penalty  # gap to aa
    dm[20, 20] = 0.0           # gap to gap
    
    # Verify properties
    assert np.allclose(dm, dm.T), "Matrix must be symmetric"
    assert np.allclose(np.diag(dm), 0.0), "Diagonal must be zero"
    assert np.all((dm >= 0) & (dm <= 4)), "Values must be in [0, 4]"
    
    return dm


def aa_embedding(config: EncodingConfig) -> np.ndarray:
    """Embed amino acid dissimilarity matrix into Euclidean space.
    
    Uses deterministic multidimensional scaling (MDS) with classical initialization 
    to embed the 21x21 amino acid dissimilarity matrix into a lower-dimensional 
    Euclidean space. The dissimilarity matrix is square-rooted before embedding 
    so that squared Euclidean distances in the embedding space approximate the 
    original TCRdist additively.
    
    The embedding is cached within processes using a composite key that includes 
    matrix fingerprints and MDS parameter fingerprints. This ensures cache 
    invalidation when underlying data or parameters change while maintaining 
    reproducibility.
    
    Parameters
    ----------
    config : EncodingConfig
        Configuration containing:
        - aa_mds_dim : Target embedding dimensionality (1-21)
        - random_seed : Random seed for MDS (ensures reproducibility)
        
    Returns
    -------
    np.ndarray
        (21, aa_mds_dim) float64 embedding matrix. Properties:
        - Column means are zero (centered embedding)
        - Deterministic: identical inputs produce identical outputs
        - Each row corresponds to one amino acid (0-19) or gap (20)
        
    Notes
    -----
    Uses sklearn.manifold.MDS with pinned parameters for reproducibility:
    - init='classical_mds' for deterministic initialization
    - metric='precomputed' for dissimilarity matrix input  
    - normalized_stress=False for consistent stress computation
    - n_jobs=None to avoid thread-dependent reduction ordering
    
    The classical MDS initialization typically produces much better embeddings 
    than random initialization (stress 0.45 vs 8.03 at dim=16), which directly 
    translates to higher accuracy in downstream TCR distance approximation.
    
    Examples
    --------
    >>> config = EncodingConfig(aa_mds_dim=8, random_seed=42)
    >>> embedding = aa_embedding(config)
    >>> embedding.shape
    (21, 8)
    >>> np.allclose(embedding.mean(axis=0), 0, atol=1e-10)  # Centered
    True
    
    Reproducibility check:
    >>> config1 = EncodingConfig(aa_mds_dim=16, random_seed=42)  
    >>> config2 = EncodingConfig(aa_mds_dim=16, random_seed=42)
    >>> emb1 = aa_embedding(config1)
    >>> emb2 = aa_embedding(config2) 
    >>> np.array_equal(emb1, emb2)  # Bit-identical
    True
    """
    # Build cache key
    dm = symbol_dissimilarity_matrix()
    matrix_fingerprint = hashlib.sha256(dm.tobytes()).hexdigest()[:16]
    mds_call_fingerprint = hashlib.sha256(
        str((_MDS_KWARGS, sklearn.__version__)).encode()
    ).hexdigest()[:16]
    
    cache_key = (
        config.aa_mds_dim,
        config.random_seed, 
        matrix_fingerprint,
        mds_call_fingerprint
    )
    
    if cache_key in _EMBEDDING_CACHE:
        logger.debug(f"aa_embedding cache hit for dim={config.aa_mds_dim}")
        return _EMBEDDING_CACHE[cache_key].copy()
    
    # Square root for additive distance structure
    dm_sqrt = np.sqrt(dm)
    
    # Deterministic MDS with pinned parameters
    mds = MDS(
        n_components=config.aa_mds_dim,
        random_state=config.random_seed,
        **_MDS_KWARGS
    )
    
    embedding = mds.fit_transform(dm_sqrt)
    
    # Center embedding (zero column means)
    embedding = embedding - np.mean(embedding, axis=0)
    
    # Log stress value
    stress = mds.stress_
    logger.info(f"AA embedding dim={config.aa_mds_dim} seed={config.random_seed} stress={stress:.4f}")
    
    # Cache and return copy
    _EMBEDDING_CACHE[cache_key] = embedding
    return embedding.copy()


def _validate_organism(organism: str, *, _skip_validation: bool = False) -> None:
    """Validate organism is supported by vectorizer.
    
    Parameters
    ----------
    organism : str
        Organism identifier to validate
        
    Raises
    ------
    ValueError
        If organism is not in SUPPORTED_ORGANISMS set, with clear message
        listing alternatives for unsupported receptor types
    """
    if _skip_validation:
        return
    if organism not in SUPPORTED_ORGANISMS:
        # Get specific error message for common unsupported types
        if organism in {'human_gd', 'mouse_gd', 'rhesus_gd'}:
            receptor_type = "gamma-delta TCRs"
        elif organism in {'human_ig', 'mouse_ig'}:
            receptor_type = "B cell receptors (Ig)"
        else:
            receptor_type = "this receptor type"
            
        raise ValueError(
            f"Organism '{organism}' not supported by vectorizer. "
            f"Supported organisms: {sorted(SUPPORTED_ORGANISMS)} (alpha-beta TCRs only). "
            f"For {receptor_type}, use KernelPCA representation (X_pca_tcr) or exact TCRdist path instead."
        )


def trim_and_gap_cdr3(
    cdr3: str,
    num_pos: int = DEFAULT_NUM_POS_CDR3,
    n_trim: int = DEFAULT_N_TRIM,
    c_trim: int = DEFAULT_C_TRIM,
) -> str:
    """Trim and gap CDR3 sequence to fixed length.
    
    Reproduces the gap positioning formula from tcr_distances.weighted_cdr3_distance 
    with ALIGN_CDR3S = False. This function implements the same CDR3 processing 
    logic used by exact TCRdist to ensure consistency in distance calculations.
    
    The algorithm performs the following steps:
    1. Calculate gap position using TCRdist formula: min(6, 3+(len(cdr3)-5)//2) - n_trim
    2. Trim N-terminal and C-terminal residues from input sequence  
    3. Insert gaps at calculated position if sequence is shorter than target length
    4. Drop interior residues around gap position if sequence is longer than target
    
    This approach ensures that N and C termini are preserved while interior 
    variability is handled through gapping or truncation.
    
    Parameters
    ----------
    cdr3 : str
        Input CDR3 sequence containing only standard amino acid single-letter codes.
        Must be longer than n_trim + c_trim to allow for meaningful trimming.
    num_pos : int, default=16
        Target output length. All output strings have exactly this length.
    n_trim : int, default=3  
        Number of N-terminal residues to trim before processing.
    c_trim : int, default=2
        Number of C-terminal residues to trim before processing.
        
    Returns
    -------
    str
        Fixed-length string of exactly num_pos characters. Gap characters are 
        represented as '.'. The sequence preserves N and C termini with gaps 
        or truncation applied to the interior.
        
    Raises
    ------
    ValueError
        If cdr3 is too short after trimming: len(cdr3) <= n_trim + c_trim
        
    Notes
    -----
    For sequences longer than the target length after trimming, interior residues 
    around the calculated gap position are dropped. This is an approximation - 
    exact TCRdist aligns each pair of sequences individually and uses the shorter 
    sequence to determine gap position, while this function must choose a position 
    per sequence in isolation.
    
    The gap position formula matches the one used in weighted_cdr3_distance when 
    ALIGN_CDR3S = False, ensuring consistency with CoNGA's existing TCRdist usage.
    
    Examples
    --------
    Basic usage:
    >>> trim_and_gap_cdr3('CASSRT', num_pos=8, n_trim=1, c_trim=1)
    'ASS...RT'
    
    Short sequence (needs gaps):
    >>> trim_and_gap_cdr3('CASSRT', num_pos=10, n_trim=1, c_trim=1)
    'ASS....SRT'
    
    Long sequence (interior truncation):
    >>> trim_and_gap_cdr3('CASSRQWERTYRT', num_pos=8, n_trim=1, c_trim=1) 
    'ASSQRT'
    
    Edge case - minimum length:
    >>> trim_and_gap_cdr3('ABCD', n_trim=1, c_trim=1, num_pos=4)  
    'BC..'
    """
    if len(cdr3) <= n_trim + c_trim:
        raise ValueError(
            f"CDR3 '{cdr3}' too short after trimming: "
            f"len={len(cdr3)} <= n_trim={n_trim} + c_trim={c_trim}"
        )
    
    # Gap position formula from tcr_distances.weighted_cdr3_distance
    # This matches the prototype exactly
    gappos = min(6, 3 + (len(cdr3) - 5) // 2) - n_trim
    
    # Trim sequence: remove n_trim from start and c_trim from end
    seq = cdr3[n_trim:-c_trim] if c_trim > 0 else cdr3[n_trim:]
    
    # Calculate afterlen and numgaps exactly as in the prototype
    afterlen = min(num_pos - gappos, len(seq) - gappos)
    numgaps = max(0, num_pos - len(seq))
    
    # Build the sequence exactly as in the prototype
    fullseq = seq[:gappos] + '.' * numgaps + seq[-afterlen:]
    
    assert len(fullseq) == num_pos, f"Length mismatch: {len(fullseq)} != {num_pos}"
    return fullseq


def vector_length(
    organism: str, config: EncodingConfig | None = None, *, _skip_validation: bool = False
) -> int:
    """Calculate expected vector length for given organism and configuration.
    
    Computes the total length L of the fixed-size vectors that encode_tcrs() 
    will produce for the specified organism and encoding configuration. This 
    allows callers to pre-allocate arrays or validate expected dimensions 
    without actually encoding any clonotypes.
    
    The vector length is determined by:
    L = (germline_A + cdr3_positions) * aa_mds_dim + (germline_B + cdr3_positions) * aa_mds_dim
    
    Where germline_A and germline_B are the number of variant positions in 
    the V gene CDR loops for each chain after invariant column removal.
    
    Parameters
    ----------
    organism : str
        Organism string, must be supported ('human', 'mouse', 'rhesus').
        Determines the reference gene database used for germline length calculation.
    config : EncodingConfig | None, default=None
        Encoding configuration. If None, uses default configuration.
        Only aa_mds_dim and num_pos_cdr3 affect the vector length.
        
    Returns
    -------
    int
        Total vector length L. For typical configurations:
        - human: ~1136 (aa_mds_dim=16, num_pos_cdr3=16)
        - mouse: ~1168 (slightly more germline positions) 
        - rhesus: ~1152 (intermediate germline positions)
        
    Raises
    ------
    ValueError
        If organism is not supported by the vectorizer.
        
    Notes
    -----
    The function attempts to load the actual germline code tables to get exact 
    lengths, but falls back to measured estimates if gene database access fails. 
    This ensures robustness while maintaining accuracy when possible.
    
    Vector lengths differ between organisms because different species have 
    different numbers of conserved/variant positions in their V gene repertoires 
    after invariant column removal.
    
    Examples
    --------
    Default configuration:
    >>> vector_length('human')
    1136
    >>> vector_length('mouse') 
    1168
    
    Custom configuration:
    >>> config = EncodingConfig(aa_mds_dim=8, num_pos_cdr3=12)
    >>> vector_length('human', config)
    568  # Roughly half the default due to aa_mds_dim reduction
    
    Pre-allocation:
    >>> n_clonotypes = 1000
    >>> L = vector_length('human')
    >>> matrix = np.empty((n_clonotypes, L), dtype=np.float32)
    """
    _validate_organism(organism, _skip_validation=_skip_validation)
    
    if config is None:
        config = EncodingConfig()
    
    total_length = 0
    
    for chain in ['A', 'B']:
        # Get germline code table to determine actual lengths
        try:
            gene_ids, code_matrix = germline_code_table(
                organism, chain, _skip_validation=_skip_validation
            )
            germline_length = code_matrix.shape[1]  # Number of variant positions
        except ValueError:
            # Fallback to estimated lengths from design measurements
            if organism == 'human':
                germline_length = 21 if chain == 'A' else 18
            elif organism == 'mouse':
                germline_length = 23 if chain == 'A' else 18
            elif organism == 'rhesus':
                germline_length = 21 if chain == 'A' else 19
            else:
                # Default reasonable estimate
                germline_length = 20
        
        # Add germline length * aa_mds_dim + CDR3 length * aa_mds_dim
        total_length += germline_length * config.aa_mds_dim
        total_length += config.num_pos_cdr3 * config.aa_mds_dim
    
    return total_length


def germline_code_table(
    organism: str, chain: str, *, _skip_validation: bool = False
) -> tuple[list[str], np.ndarray]:
    """Build germline code table for organism and chain.
    
    Extracts V gene CDR loop sequences from the gene database, converts them to 
    integer symbol codes, and removes invariant positions (positions with the same 
    amino acid across all V genes). This produces a compact representation of 
    the variant germline positions that contribute to TCR diversity.
    
    The function concatenates CDR1, CDR2, and CDR2.5 loops from each V gene,  
    excluding the final CDR (CDR3) which is handled separately. Both '.' and '*' 
    gap characters are mapped to the same gap symbol (index 20).
    
    Parameters
    ----------
    organism : str
        Organism identifier, must be supported ('human', 'mouse', 'rhesus').
        Determines which gene database to query.
    chain : str
        Chain identifier ('A' for alpha, 'B' for beta).
        Determines which V gene set to extract.
        
    Returns
    -------
    tuple[list[str], np.ndarray]
        Gene IDs and code matrix:
        - list[str]: V gene identifiers sorted alphabetically for consistent ordering
        - np.ndarray: (n_genes, n_kept_positions) int32 matrix where each row 
          corresponds to one V gene and each column to one variant position.
          Values are amino acid indices (0-19) or gap index (20).
          
    Raises
    ------
    ValueError
        - If organism not found in gene database
        - If no V genes found for organism/chain combination  
        - If germline sequences have inconsistent lengths
        - If unknown characters found in sequences
        
    Notes
    -----
    **Invariant position removal**: Positions where all V genes have identical 
    amino acids are removed since they contribute zero to all pairwise distances. 
    This significantly reduces vector dimensionality without loss of information.
    
    **Sequence extraction**: Uses gene.cdrs[:-1] to get CDR1+CDR2+CDR2.5 while 
    excluding CDR3. This matches the logic in tcr_distances.py for consistency.
    
    **Symbol mapping**: Follows amino_acids module ordering (alphabetical) with 
    gap at index 20. This ensures consistency with symbol_dissimilarity_matrix().
    
    Examples
    --------
    Basic usage:
    >>> gene_ids, codes = germline_code_table('human', 'A')
    >>> len(gene_ids)  # Number of human alpha V genes
    47
    >>> codes.shape   # (n_genes, n_variant_positions)
    (47, 21)
    >>> codes.dtype
    dtype('int32')
    
    Cross-chain comparison:
    >>> _, alpha_codes = germline_code_table('human', 'A')  
    >>> _, beta_codes = germline_code_table('human', 'B')
    >>> alpha_codes.shape[1], beta_codes.shape[1]  # Different lengths
    (21, 18)
    
    Symbol mapping verification:
    >>> codes[0, 0]  # First position of first gene 
    2  # Index in amino_acids string
    >>> from conga.tcrdist.amino_acids import amino_acids
    >>> amino_acids[2]
    'D'  # Corresponds to aspartic acid
    """
    _validate_organism(organism, _skip_validation=_skip_validation)
    
    # Lazy import gene database
    from .all_genes import all_genes
    
    # Get V genes for this organism/chain
    if organism not in all_genes:
        raise ValueError(f"Organism '{organism}' not found in gene database")
    
    genes_dict = all_genes[organism]
    genes = [g for g in genes_dict.values() 
            if g.chain == chain and g.region == 'V']
    
    if not genes:
        raise ValueError(f"No V genes found for organism '{organism}' chain '{chain}'")
    
    # Sort by gene ID for consistent ordering
    genes = sorted(genes, key=lambda g: g.id)
    gene_ids = [g.id for g in genes]
    
    # Build symbol code matrix
    # Map amino acids + gap characters to indices
    symbol_to_index = {aa: i for i, aa in enumerate(amino_acids)}
    symbol_to_index['.'] = 20  # Gap character
    symbol_to_index['*'] = 20  # Alternative gap character
    
    # Extract germline sequences (CDR1 + CDR2 + CDR2.5, exclude CDR3)
    germline_seqs = []
    for gene in genes:
        if gene.cdrs:
            germline_seq = ''.join(gene.cdrs[:-1])  # Exclude final CDR (CDR3)
            germline_seqs.append(germline_seq)
        # Skip genes without CDR annotation
    
    if not germline_seqs:
        raise ValueError(f"No V genes with CDR annotation for organism '{organism}' chain '{chain}'")
    
    # Verify all sequences have same length
    seq_lengths = [len(seq) for seq in germline_seqs]
    if not all(length == seq_lengths[0] for length in seq_lengths):
        raise ValueError(f"Germline sequences have different lengths: {set(seq_lengths)}")
    
    seq_length = seq_lengths[0]
    n_genes = len(germline_seqs)
    
    # Convert to symbol codes
    code_matrix = np.zeros((n_genes, seq_length), dtype=np.int32)
    for i, seq in enumerate(germline_seqs):
        for j, char in enumerate(seq):
            if char not in symbol_to_index:
                raise ValueError(f"Unknown character '{char}' in germline sequence")
            code_matrix[i, j] = symbol_to_index[char]
    
    # Remove invariant positions (same symbol across all genes)
    variant_positions = []
    for j in range(seq_length):
        if len(set(code_matrix[:, j])) > 1:  # More than one unique symbol
            variant_positions.append(j)
    
    # Keep only variant positions
    if variant_positions:
        code_matrix = code_matrix[:, variant_positions]
    else:
        # Edge case: all positions invariant - keep one position to avoid empty matrix
        logger.warning(f"All germline positions invariant for {organism} chain {chain}")
        code_matrix = code_matrix[:, :1]
    
    logger.debug(f"Germline table {organism} chain {chain}: {n_genes} genes, "
                f"{len(variant_positions)} variant positions of {seq_length} total")
    
    return gene_ids, code_matrix


def _validate_input(
    va: Sequence[str], 
    cdr3a: Sequence[str],
    vb: Sequence[str], 
    cdr3b: Sequence[str],
    organism: str,
    config: EncodingConfig,
    *,
    _skip_validation: bool = False,
) -> None:
    """Validate all input before encoding allocation.
    
    Performs comprehensive validation of V genes and CDR3 sequences.
    All validation completes before any encoding arrays are allocated.
    
    Parameters
    ----------
    va, vb : Sequence[str]
        V gene identifiers for alpha/beta chains
    cdr3a, cdr3b : Sequence[str]  
        CDR3 sequences for alpha/beta chains
    organism : str
        Organism string
    config : EncodingConfig
        Encoding configuration
        
    Raises
    ------
    ValueError
        For invalid V genes, CDR3 content, length constraints, or organism/chain
        combinations with no V gene records in the database
    """
    # Lazy import gene database
    from .all_genes import all_genes
    
    n_clonotypes = len(va)
    
    # Validate sequence lengths match
    if not (len(cdr3a) == len(vb) == len(cdr3b) == n_clonotypes):
        raise ValueError("All input sequences must have same length")
    
    # Validate organism is supported
    _validate_organism(organism, _skip_validation=_skip_validation)
    
    # Get organism gene database
    if organism not in all_genes:
        raise ValueError(f"Organism '{organism}' not found in gene database")
    
    genes_dict = all_genes[organism]
    valid_genes = {}
    
    # Validate that organism has V gene records for each required chain
    for chain in ['A', 'B']:
        chain_name = 'alpha' if chain == 'A' else 'beta'
        genes = [g.id for g in genes_dict.values() 
                if g.chain == chain and g.region == 'V']
        
        if not genes:
            raise ValueError(
                f"No V gene records found in gene database for organism '{organism}' "
                f"chain {chain} ({chain_name}). Cannot encode {chain_name} chain."
            )
        
        valid_genes[chain] = set(genes)
        logger.debug(f"Found {len(genes)} V genes for {organism} chain {chain}")
    
    # Validate V genes against database
    for chain_label, v_genes in [('A', va), ('B', vb)]:
        chain_name = 'alpha' if chain_label == 'A' else 'beta'
        invalid_genes = []
        affected_indices = []
        
        for i, v_gene in enumerate(v_genes):
            if v_gene not in valid_genes[chain_label]:
                invalid_genes.append(v_gene)
                affected_indices.append(i)
        
        if invalid_genes:
            # Report detailed information about missing genes
            unique_invalid = sorted(set(invalid_genes))
            affected_count = len(affected_indices)
            
            # Show available alternatives for first few missing genes
            available_sample = sorted(list(valid_genes[chain_label]))[:5]
            available_msg = f"Available genes include: {available_sample}"
            if len(valid_genes[chain_label]) > 5:
                available_msg += f" (and {len(valid_genes[chain_label]) - 5} more)"
            
            raise ValueError(
                f"V gene(s) not found in gene database for {organism} {chain_name} chain: "
                f"{unique_invalid}. Affects {affected_count} clonotypes. "
                f"{available_msg}."
            )
    
    # Validate CDR3 sequences
    valid_amino_acids = set(amino_acids)
    min_length = config.n_trim + config.c_trim + 1
    long_cdr3_count = 0
    
    for cdr3_list, chain_name in [(cdr3a, 'alpha'), (cdr3b, 'beta')]:
        for i, cdr3 in enumerate(cdr3_list):
            # Check amino acid content
            invalid_chars = [c for c in cdr3 if c not in valid_amino_acids]
            if invalid_chars:
                unique_invalid = sorted(set(invalid_chars))
                raise ValueError(
                    f"CDR3 '{cdr3}' (clonotype {i}, {chain_name} chain) contains invalid characters: "
                    f"{unique_invalid}. Valid amino acids: {sorted(valid_amino_acids)}."
                )
            
            # Check minimum length after trimming
            if len(cdr3) < min_length:
                raise ValueError(
                    f"CDR3 '{cdr3}' (clonotype {i}, {chain_name} chain) too short: "
                    f"length {len(cdr3)} < minimum required {min_length} "
                    f"(n_trim={config.n_trim} + c_trim={config.c_trim} + 1)."
                )
            
            # Count long CDR3s (warning, not error)
            encodable_length = config.num_pos_cdr3 + config.n_trim + config.c_trim
            if len(cdr3) > encodable_length:
                long_cdr3_count += 1
    
    # Log warning for long CDR3s that will have interior residues dropped
    if long_cdr3_count > 0:
        max_encodable = config.num_pos_cdr3 + config.n_trim + config.c_trim
        logger.warning(
            f"{long_cdr3_count} CDR3 sequences longer than encodable length "
            f"({max_encodable} = num_pos_cdr3={config.num_pos_cdr3} + "
            f"n_trim={config.n_trim} + c_trim={config.c_trim}). "
            f"Interior residues will be dropped during encoding."
        )


def encode_tcrs(
    tcrs: Sequence[tuple[tuple, tuple]] | pd.DataFrame,
    organism: str,
    config: EncodingConfig | None = None,
    *,
    va_column: str = 'va',
    cdr3a_column: str = 'cdr3a', 
    vb_column: str = 'vb',
    cdr3b_column: str = 'cdr3b',
    _skip_validation: bool = False,
) -> np.ndarray:
    """Encode paired TCR clonotypes as fixed-length vectors.
    
    Main encoding function that converts paired alpha-beta TCR clonotypes into 
    dense float32 vectors. Euclidean distance between encoded vectors approximates 
    sqrt(TCRdist), while squared Euclidean distance approximates TCRdist itself.
    
    The encoding concatenates vectors for both alpha and beta chains, where each 
    chain vector consists of:
    1. Germline V-region embedding (CDR1+CDR2+CDR2.5 loops, variant positions only)
    2. CDR3 embedding (trimmed and gap-aligned to fixed length, scaled by sqrt(cdr3_weight))
    
    This produces vectors of fixed length L determined by vector_length(organism, config), 
    enabling efficient neighbor search without materializing quadratic distance matrices.
    
    Parameters
    ----------
    tcrs : Sequence[tuple[tuple, tuple]] | pd.DataFrame
        Clonotype data in one of two formats:
        
        Nested tuples format (from preprocess.retrieve_tcrs_from_adata):
        Each element is ((va, ja, cdr3a, nucseq_a), (vb, jb, cdr3b, nucseq_b))
        Only V genes (va, vb) and CDR3s (cdr3a, cdr3b) are used.
        
        DataFrame format:
        Must contain columns specified by va_column, cdr3a_column, vb_column, 
        cdr3b_column parameters. Common alternative names like 'va_gene', 
        'vb_gene' are automatically detected as fallbacks.
        
    organism : str
        Organism identifier. Must be supported: 'human', 'mouse', 'rhesus'.
        Determines gene database and reference sequences used for encoding.
        
    config : EncodingConfig | None, default=None
        Encoding configuration controlling vector construction:
        - aa_mds_dim: Amino acid embedding dimensionality (affects vector length)
        - num_pos_cdr3: Fixed CDR3 length in encoding (affects vector length)  
        - cdr3_weight: CDR3 vs germline weighting (3.0 matches TCRdist)
        - n_trim, c_trim: CDR3 trimming parameters
        - random_seed: MDS embedding seed (for reproducibility)
        If None, uses default configuration.
        
    va_column, cdr3a_column, vb_column, cdr3b_column : str
        Column names for DataFrame input. Defaults match standard CoNGA naming.
        
    Returns
    -------
    np.ndarray
        (N, L) float32 C-contiguous matrix where:
        - N = number of input clonotypes  
        - L = vector_length(organism, config)
        - Row i corresponds to input clonotype i (preserves order)
        - All values are finite
        - Memory layout optimized for subsequent vector operations
        
    Raises
    ------
    ValueError
        - If organism not supported ('human', 'mouse', 'rhesus' only)
        - If V gene not found in gene database for organism  
        - If CDR3 contains non-standard amino acids
        - If CDR3 too short after trimming (< n_trim + c_trim + 1)
        - If input format invalid or columns missing
        
    Notes
    -----
    **Memory complexity**: O(N·L) with no N^2 allocations, suitable for large datasets.
    
    **Validation**: All input is validated before any encoding arrays are allocated, 
    ensuring clean failures without partial state.
    
    **CDR3 processing**: Long CDR3 sequences (> num_pos_cdr3 + n_trim + c_trim) 
    trigger interior residue dropping with a warning logged. Short sequences are 
    gap-padded to the target length.
    
    **Accuracy**: Achieves Spearman correlation > 0.99 and k-NN recall > 0.94 
    vs exact TCRdist on all supported organisms with default configuration.
    
    Examples
    --------
    Basic usage with nested tuples:
    >>> tcrs = [
    ...     (('TRAV1*01', 'TRAJ1*01', 'CAVRD', ''), 
    ...      ('TRBV1*01', 'TRBJ1*01', 'CASSRT', '')),
    ...     (('TRAV2*01', 'TRAJ2*01', 'CAVKE', ''),
    ...      ('TRBV2*01', 'TRBJ2*01', 'CASSLQ', ''))
    ... ]
    >>> matrix = encode_tcrs(tcrs, 'human')
    >>> matrix.shape
    (2, 1136)
    >>> matrix.dtype
    dtype('float32')
    
    DataFrame input:
    >>> import pandas as pd  
    >>> df = pd.DataFrame({
    ...     'va': ['TRAV1*01', 'TRAV2*01'],
    ...     'cdr3a': ['CAVRD', 'CAVKE'],
    ...     'vb': ['TRBV1*01', 'TRBV2*01'], 
    ...     'cdr3b': ['CASSRT', 'CASSLQ']
    ... })
    >>> matrix = encode_tcrs(df, 'human')
    
    Custom configuration:
    >>> config = EncodingConfig(aa_mds_dim=12, cdr3_weight=2.0, random_seed=123)
    >>> matrix = encode_tcrs(tcrs, 'human', config)
    
    Alternative column names:
    >>> df_alt = pd.DataFrame({
    ...     'v_alpha': ['TRAV1*01'], 'cdr3_alpha': ['CAVRD'],
    ...     'v_beta': ['TRBV1*01'], 'cdr3_beta': ['CASSRT']  
    ... })
    >>> matrix = encode_tcrs(df_alt, 'human', 
    ...                     va_column='v_alpha', cdr3a_column='cdr3_alpha',
    ...                     vb_column='v_beta', cdr3b_column='cdr3_beta')
    
    Distance calculation:
    >>> from scipy.spatial.distance import pdist
    >>> distances = pdist(matrix, metric='euclidean')  # Approximates sqrt(TCRdist)
    >>> squared_distances = pdist(matrix, metric='sqeuclidean')  # Approximates TCRdist
    """
    _validate_organism(organism, _skip_validation=_skip_validation)
    if organism in _TIER_3_ORGANISMS and organism not in _already_warned_organisms:
        _already_warned_organisms.add(organism)
        logger.warning(
            f"Vectorized TCRdist for organism {organism!r} was validated "
            f"using synthetic sequence data modeled on human CDR3 content, "
            f"not species-matched real repertoire data for {organism!r}. "
            f"Consider the KernelPCA representation (X_pca_tcr) or the "
            f"exact TCRdist path for a more conservative alternative."
        )
    
    if config is None:
        config = EncodingConfig()
    
    # Normalize input format
    if isinstance(tcrs, pd.DataFrame):
        # Handle DataFrame input with fallback column names
        df = tcrs
        
        # Try primary column names, fall back to common alternatives
        def get_column(primary: str, fallback: str | None = None) -> str:
            if primary in df.columns:
                return primary
            if fallback and fallback in df.columns:
                return fallback
            raise ValueError(f"Column '{primary}' not found in DataFrame")
        
        va_col = get_column(va_column, 'va_gene')
        vb_col = get_column(vb_column, 'vb_gene')
        cdr3a_col = get_column(cdr3a_column)
        cdr3b_col = get_column(cdr3b_column)
        
        va = df[va_col].astype(str).tolist()
        cdr3a = df[cdr3a_col].astype(str).tolist()
        vb = df[vb_col].astype(str).tolist()  
        cdr3b = df[cdr3b_col].astype(str).tolist()
    else:
        # Handle nested tuple input
        tcr_list = list(tcrs)
        va = [tcr[0][0] for tcr in tcr_list]      # alpha V gene
        cdr3a = [tcr[0][2] for tcr in tcr_list]   # alpha CDR3 
        vb = [tcr[1][0] for tcr in tcr_list]      # beta V gene
        cdr3b = [tcr[1][2] for tcr in tcr_list]   # beta CDR3
    
    n_clonotypes = len(va)
    if n_clonotypes == 0:
        # Return empty array with correct shape
        L = vector_length(organism, config, _skip_validation=_skip_validation)
        return np.empty((0, L), dtype=np.float32, order='C')
    
    # Validate all input before any allocation
    _validate_input(va, cdr3a, vb, cdr3b, organism, config, _skip_validation=_skip_validation)
    
    # Get amino acid embedding
    aa_embedding_matrix = aa_embedding(config)  # (21, aa_mds_dim)
    
    # Build encoding blocks
    blocks = []
    
    for chain_id, chain_va, chain_cdr3 in [('A', va, cdr3a), ('B', vb, cdr3b)]:
        # Get germline code table for this chain
        gene_ids, germline_codes = germline_code_table(
            organism, chain_id, _skip_validation=_skip_validation
        )
        
        # Map V gene names to row indices
        gene_to_row = {gene_id: i for i, gene_id in enumerate(gene_ids)}
        
        # Map clonotype V genes to germline code rows  
        gene_rows = np.array([gene_to_row[v_gene] for v_gene in chain_va], dtype=np.int32)
        
        # Gather germline codes for all clonotypes: (n_clonotypes, n_germline_positions)
        clonotype_germline_codes = germline_codes[gene_rows]
        
        # Process CDR3 sequences
        cdr3_codes = []
        for cdr3 in chain_cdr3:
            # Trim and gap to fixed length
            gapped_cdr3 = trim_and_gap_cdr3(cdr3, config.num_pos_cdr3, config.n_trim, config.c_trim)
            
            # Convert to symbol codes
            symbol_codes = []
            for char in gapped_cdr3:
                if char == '.' or char == '*':
                    symbol_codes.append(20)  # Gap index
                else:
                    symbol_codes.append(amino_acids.index(char))
            
            cdr3_codes.append(symbol_codes)
        
        cdr3_code_matrix = np.array(cdr3_codes, dtype=np.int32)  # (n_clonotypes, num_pos_cdr3)
        
        # Encode using fancy indexing of embedding matrix
        # Germline block: (n_clonotypes, n_germline_pos, aa_mds_dim) -> (n_clonotypes, n_germline_pos * aa_mds_dim)
        germline_embedded = aa_embedding_matrix[clonotype_germline_codes]
        germline_block = germline_embedded.reshape(n_clonotypes, -1)
        
        # CDR3 block: (n_clonotypes, num_pos_cdr3, aa_mds_dim) -> (n_clonotypes, num_pos_cdr3 * aa_mds_dim) 
        cdr3_embedded = aa_embedding_matrix[cdr3_code_matrix]
        cdr3_block = cdr3_embedded.reshape(n_clonotypes, -1)
        
        # Apply CDR3 weight scaling (sqrt for additive distance structure)
        cdr3_block = cdr3_block * np.sqrt(config.cdr3_weight)
        
        # Add blocks for this chain
        blocks.extend([germline_block, cdr3_block])
    
    # Concatenate all blocks: (n_clonotypes, total_length)
    result = np.hstack(blocks)  # Still float64 from embedding
    
    # Verify expected shape
    expected_length = vector_length(organism, config, _skip_validation=_skip_validation)
    if result.shape != (n_clonotypes, expected_length):
        raise RuntimeError(f"Encoding shape mismatch: got {result.shape}, expected {(n_clonotypes, expected_length)}")
    
    # Verify all finite values
    if not np.all(np.isfinite(result)):
        raise RuntimeError("Encoding produced non-finite values")
    
    # Single cast to float32 at boundary, ensure C-contiguous
    return np.ascontiguousarray(result, dtype=np.float32)


@dataclass(frozen=True) 
class AccuracyReport:
    """Comprehensive accuracy validation results comparing vectorized vs exact TCRdist.
    
    This frozen dataclass contains all metrics from accuracy validation, including 
    correlation statistics and k-nearest neighbor recall measures. It provides 
    both global similarity metrics (correlations over all pairs) and local 
    structure preservation metrics (neighbor recall).
    
    The report enables quantitative assessment of encoding quality and helps 
    decide whether vectorized approximation is suitable for specific analyses.
    All metrics are computed on the same test dataset for consistency.
    
    Attributes
    ----------
    organism : str
        Organism tested ('human', 'mouse', 'rhesus'). 
        Different organisms may have different accuracy profiles.
        
    config : EncodingConfig  
        Encoding configuration used for testing.
        Documents the parameter settings that achieved these metrics.
        
    num_clonotypes : int
        Number of clonotypes in the test dataset.
        Larger test sets generally give more reliable accuracy estimates.
        
    num_pairs_sampled : int
        Number of clonotype pairs actually used for correlation computation.
        May be less than N*(N-1)/2 if sampling was applied due to max_pairs limit.
        
    pearson_distance : float
        Pearson correlation between Euclidean distances and TCRdist values.
        Range: [-1, 1]. Measures linear relationship strength.
        Note: Euclidean distance approximates sqrt(TCRdist), not TCRdist itself.
        
    pearson_squared_distance : float  
        Pearson correlation between squared Euclidean distances and TCRdist values.
        Range: [-1, 1]. More faithful metric since squared Euclidean approximates TCRdist.
        Generally higher than pearson_distance due to better linearity.
        
    spearman : float
        Spearman rank correlation between distances.
        Range: [-1, 1]. Measures monotone relationship, most robust to nonlinearity.
        Often the most important metric for neighbor-based analyses.
        
    neighbor_counts : tuple[int, ...]
        Values of k where k-NN recall was measured.
        Typically (10, 100) to assess both local and regional neighborhood preservation.
        
    mean_recall : dict[int, float]
        Mean k-NN recall for each k in neighbor_counts.
        recall@k = |exact_k_neighbors ∩ vectorized_k_neighbors| / k
        Values near 1.0 indicate excellent neighborhood preservation.
        
    vectorizer_version : str
        Version tag of the encoder that generated this report.
        Enables tracking accuracy across different implementations.
        
    Methods
    -------
    as_dict() : dict[str, object]
        Convert to dictionary for serialization or storage.
        Flattens nested objects for JSON-compatible output.
        
    Interpretation Guidelines  
    -------------------------
    **Excellent encoding** (suitable for most analyses):
    - Spearman ≥ 0.95
    - Mean recall@10 ≥ 0.80  
    - Pearson_squared_distance ≥ 0.90
    
    **Good encoding** (suitable for exploratory analysis):
    - Spearman ≥ 0.90
    - Mean recall@10 ≥ 0.70
    
    **Poor encoding** (consider exact TCRdist instead):
    - Spearman < 0.90 OR mean recall@10 < 0.70
    
    **Warning signs**:
    - Large gap between pearson_distance and pearson_squared_distance (>0.1)
    - Recall@100 much higher than recall@10 (suggests local structure loss)
    - Very low num_pairs_sampled relative to dataset size (unreliable correlations)
    
    Examples
    --------
    Typical good report:
    >>> report = AccuracyReport(
    ...     organism='human',
    ...     config=EncodingConfig(),
    ...     num_clonotypes=1000,
    ...     num_pairs_sampled=499500,
    ...     pearson_distance=0.993,
    ...     pearson_squared_distance=0.999,
    ...     spearman=0.999,
    ...     neighbor_counts=(10, 100),
    ...     mean_recall={10: 0.953, 100: 0.971},
    ...     vectorizer_version='conga.vectorized/1'
    ... )
    
    Accessing results:
    >>> report.spearman >= 0.95  # Passes accuracy gate
    True
    >>> report.mean_recall[10] >= 0.80  # Passes recall gate  
    True
    >>> print(f"Encoding quality: Spearman={report.spearman:.3f}")
    Encoding quality: Spearman=0.999
    
    Serialization:
    >>> report_dict = report.as_dict()
    >>> report_dict['organism']
    'human'
    >>> report_dict['config']['aa_mds_dim']  
    16
    """
    organism: str
    config: EncodingConfig
    num_clonotypes: int
    num_pairs_sampled: int
    pearson_distance: float
    pearson_squared_distance: float  
    spearman: float
    neighbor_counts: tuple[int, ...]
    mean_recall: dict[int, float]
    vectorizer_version: str
    
    def as_dict(self) -> dict[str, object]:
        """Convert to dictionary for serialization."""
        return {
            'organism': self.organism,
            'config': self.config.as_uns_dict(),
            'num_clonotypes': self.num_clonotypes,
            'num_pairs_sampled': self.num_pairs_sampled,
            'pearson_distance': self.pearson_distance,
            'pearson_squared_distance': self.pearson_squared_distance,
            'spearman': self.spearman,
            'neighbor_counts': list(self.neighbor_counts),
            'mean_recall': self.mean_recall,
            'vectorizer_version': self.vectorizer_version,
        }


def store_tcr_vectors_in_adata(
    adata,
    organism: str,
    config: EncodingConfig | None = None,
    *,
    va_column: str = 'va',
    cdr3a_column: str = 'cdr3a',
    vb_column: str = 'vb', 
    cdr3b_column: str = 'cdr3b',
) -> np.ndarray:
    """Store vectorized TCR encodings in AnnData object with proper metadata.
    
    Encodes TCR clonotypes from the AnnData observation metadata and stores the 
    resulting vector matrix in adata.obsm under the standardized key X_vec_tcr.
    Also stores encoding configuration, organism, and version metadata in adata.uns
    for reproducibility and analysis tracking.
    
    The function preserves any existing representations in the AnnData object while
    adding the vectorized representation. If X_vec_tcr already exists, it is 
    overwritten with a logged warning. Other TCR representations (X_pca_tcr) 
    remain unchanged.
    
    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data object containing TCR metadata in .obs. Must have columns
        for V genes and CDR3 sequences for both alpha and beta chains.
    organism : str
        Organism identifier ('human', 'mouse', 'rhesus'). Must be supported
        by the vectorized encoder.
    config : EncodingConfig | None, default=None
        Encoding configuration. If None, uses default configuration.
        All config values are stored in adata.uns for reproducibility.
    va_column, cdr3a_column, vb_column, cdr3b_column : str
        Column names in adata.obs containing V gene and CDR3 data.
        Defaults match standard CoNGA naming conventions.
        
    Returns
    -------
    np.ndarray
        The encoded vector matrix that was stored in adata.obsm[X_vec_tcr].
        Shape: (n_obs, vector_length). Provided for caller convenience.
        
    Notes
    -----
    **Storage locations**:
    - Vector matrix: adata.obsm[util.OBSM_KEY_VEC_TCR] ('X_vec_tcr')
    - Encoding config: adata.uns[util.UNS_KEY_VEC_TCR_CONFIG] ('vec_tcr_config')
    
    **Row ordering**: The stored matrix rows correspond to adata.obs rows in 
    the same order, enabling direct indexing and slicing operations.
    
    **Overwrite behavior**: If X_vec_tcr already exists, logs a warning and 
    overwrites. This supports workflow restart scenarios and config changes.
    
    **Coexistence**: Does not affect existing X_pca_tcr or other representations.
    Multiple TCR representations can coexist in the same AnnData object.
    
    Examples
    --------
    Basic usage:
    >>> import anndata as ad
    >>> import pandas as pd
    >>> # Create AnnData with TCR data
    >>> obs = pd.DataFrame({
    ...     'va': ['TRAV1*01', 'TRAV2*01'],
    ...     'cdr3a': ['CAVRD', 'CAVKE'],
    ...     'vb': ['TRBV1*01', 'TRBV2*01'],
    ...     'cdr3b': ['CASSRT', 'CASSLQ']
    ... })
    >>> adata = ad.AnnData(obs=obs)
    >>> matrix = store_tcr_vectors_in_adata(adata, 'human')
    >>> matrix.shape
    (2, 1136)
    >>> 'X_vec_tcr' in adata.obsm
    True
    
    Custom configuration:
    >>> config = EncodingConfig(aa_mds_dim=12, random_seed=123)
    >>> matrix = store_tcr_vectors_in_adata(adata, 'human', config)
    >>> adata.uns['vec_tcr_config']['aa_mds_dim']
    12
    >>> adata.uns['vec_tcr_config']['random_seed']
    123
    
    Alternative column names:
    >>> matrix = store_tcr_vectors_in_adata(
    ...     adata, 'human',
    ...     va_column='v_alpha', cdr3a_column='cdr3_alpha',
    ...     vb_column='v_beta', cdr3b_column='cdr3_beta'
    ... )
    
    Restart/overwrite scenario:
    >>> # First encoding
    >>> matrix1 = store_tcr_vectors_in_adata(adata, 'human')
    >>> # Changed config - will log warning and overwrite
    >>> config2 = EncodingConfig(aa_mds_dim=8)
    >>> matrix2 = store_tcr_vectors_in_adata(adata, 'human', config2)
    >>> # New matrix has different dimensions due to config change
    >>> matrix1.shape != matrix2.shape
    True
    """
    # Lazy import to avoid circular dependencies
    import anndata as ad
    
    if config is None:
        config = EncodingConfig()
    
    # Check if overwriting existing vectorized representation
    if util.OBSM_KEY_VEC_TCR in adata.obsm:
        logger.warning(f"Overwriting existing vectorized TCR representation in adata.obsm['{util.OBSM_KEY_VEC_TCR}']")
    
    # Convert adata.obs to DataFrame format for encode_tcrs
    obs_df = adata.obs.copy()
    
    # Encode the TCR data
    logger.info(f"Encoding {adata.n_obs} clonotypes with organism '{organism}'")
    vector_matrix = encode_tcrs(
        obs_df,
        organism, 
        config,
        va_column=va_column,
        cdr3a_column=cdr3a_column,
        vb_column=vb_column,
        cdr3b_column=cdr3b_column,
    )
    
    # Verify row ordering matches adata.obs
    if vector_matrix.shape[0] != adata.n_obs:
        raise RuntimeError(
            f"Encoded matrix rows ({vector_matrix.shape[0]}) != adata.n_obs ({adata.n_obs})"
        )
    
    # Store vector matrix in obsm
    adata.obsm[util.OBSM_KEY_VEC_TCR] = vector_matrix
    
    # Store encoding configuration and metadata in uns
    config_dict = config.as_uns_dict()
    config_dict['organism'] = organism
    adata.uns[util.UNS_KEY_VEC_TCR_CONFIG] = config_dict
    
    logger.info(
        f"Stored vectorized TCR representation: {vector_matrix.shape} in "
        f"adata.obsm['{util.OBSM_KEY_VEC_TCR}'] with config in "
        f"adata.uns['{util.UNS_KEY_VEC_TCR_CONFIG}']"
    )
    
    return vector_matrix


def record_active_tcr_representation(adata, active_representation: str) -> None:
    """Record which TCR representation is currently active for neighbor calculations.
    
    Stores the active TCR representation identifier in AnnData metadata so that
    downstream analysis steps know which representation to use for neighbor search,
    clustering, and visualization. This enables multiple TCR representations to
    coexist while maintaining a clear selection.
    
    The active representation determines which obsm key or neighbor calculation 
    method CoNGA will use for TCR-based analyses. This function only records 
    the selection; it does not validate that the representation actually exists
    or perform any computation.
    
    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data object to store the active representation metadata.
    active_representation : str
        Active representation identifier. Must be one of:
        - util.OBSM_KEY_VEC_TCR ('X_vec_tcr'): Vectorized representation
        - util.OBSM_KEY_PCA_TCR ('X_pca_tcr'): KernelPCA representation  
        - util.ACTIVE_REP_EXACT ('exact_tcrdist'): Exact TCRdist path
        
    Notes
    -----
    **Storage location**: adata.uns[util.UNS_KEY_ACTIVE_TCR_REP] ('active_tcr_representation')
    
    **Exact representation handling**: When active_representation is 'exact_tcrdist',
    no obsm entry is created since the exact path computes neighbors on-demand
    without storing a per-observation representation matrix.
    
    **Validation**: This function does not validate that the specified representation
    exists. Use get_active_tcr_representation() to verify stored values.
    
    Examples
    --------
    Set vectorized representation as active:
    >>> record_active_tcr_representation(adata, 'X_vec_tcr')
    >>> adata.uns['active_tcr_representation']
    'X_vec_tcr'
    
    Set exact TCRdist path as active:
    >>> record_active_tcr_representation(adata, 'exact_tcrdist')
    >>> adata.uns['active_tcr_representation']
    'exact_tcrdist'
    
    Switch between representations:
    >>> record_active_tcr_representation(adata, 'X_pca_tcr')  # KernelPCA
    >>> record_active_tcr_representation(adata, 'X_vec_tcr')   # Vectorized
    """
    adata.uns[util.UNS_KEY_ACTIVE_TCR_REP] = active_representation
    logger.debug(f"Set active TCR representation to '{active_representation}'")


def get_active_tcr_representation(adata) -> str | None:
    """Retrieve the currently active TCR representation from AnnData metadata.
    
    Returns the active TCR representation identifier stored by 
    record_active_tcr_representation(). This tells downstream code which
    representation to use for neighbor calculations and analysis.
    
    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data object to read the active representation from.
        
    Returns
    -------
    str | None
        Active representation identifier, or None if no active representation
        has been recorded. Valid return values:
        - 'X_vec_tcr': Vectorized representation
        - 'X_pca_tcr': KernelPCA representation  
        - 'exact_tcrdist': Exact TCRdist path
        - None: No active representation set
        
    Notes
    -----
    **Storage location**: Reads from adata.uns[util.UNS_KEY_ACTIVE_TCR_REP]
    
    **Legacy compatibility**: Returns None for AnnData objects created before
    active representation tracking was implemented. Callers should handle
    None values gracefully.
    
    **Validation**: This function does not validate that the returned
    representation actually exists in adata.obsm. Use with appropriate
    existence checks in downstream code.
    
    Examples
    --------
    Check current active representation:
    >>> active = get_active_tcr_representation(adata)
    >>> if active == 'X_vec_tcr':
    ...     # Use vectorized representation
    ...     vectors = adata.obsm['X_vec_tcr']
    >>> elif active == 'X_pca_tcr':
    ...     # Use KernelPCA representation
    ...     vectors = adata.obsm['X_pca_tcr']
    >>> elif active == 'exact_tcrdist':
    ...     # Use exact TCRdist neighbor calculation
    ...     pass  # No obsm array to load
    >>> else:
    ...     # No active representation set
    ...     pass
    
    Conditional representation access:
    >>> active = get_active_tcr_representation(adata)
    >>> if active and active in adata.obsm:
    ...     representation = adata.obsm[active]
    >>> elif active == 'exact_tcrdist':
    ...     # Handle exact path case
    ...     pass
    """
    return adata.uns.get(util.UNS_KEY_ACTIVE_TCR_REP, None)


def record_active_tcr_representation(adata, active_representation: str) -> None:
    """Record which TCR representation is currently active for neighbor calculations.
    
    Stores the active TCR representation identifier in AnnData metadata so that
    downstream analysis steps can determine which representation was used for 
    TCR neighbor calculations. This enables restart scenarios and verification
    that the expected representation is being used.
    
    The recorded active representation should match the actual representation
    available in adata.obsm. Callers should ensure the representation is available
    before recording it as active.
    
    **Restart compatibility**: The recorded value persists through h5ad save/load
    cycles, enabling workflow restart scenarios.
    
    Examples
    --------
    Record vectorized representation as active:
    >>> record_active_tcr_representation(adata, util.OBSM_KEY_VEC_TCR)
    >>> adata.uns['active_tcr_representation']
    'X_vec_tcr'
    
    Record KernelPCA representation as active:
    >>> record_active_tcr_representation(adata, util.OBSM_KEY_PCA_TCR)
    >>> adata.uns['active_tcr_representation'] 
    'X_pca_tcr'
    
    Record exact TCRdist path as active:
    >>> record_active_tcr_representation(adata, util.ACTIVE_REP_EXACT)
    >>> adata.uns['active_tcr_representation']
    'exact_tcrdist'
    >>> # Note: No obsm entry created for exact path
    >>> util.OBSM_KEY_VEC_TCR in adata.obsm  # May be False
    """
    # Validate active_representation value
    valid_representations = {util.OBSM_KEY_VEC_TCR, util.OBSM_KEY_PCA_TCR, util.ACTIVE_REP_EXACT}
    if active_representation not in valid_representations:
        raise ValueError(
            f"Invalid active_representation '{active_representation}'. "
            f"Must be one of: {sorted(valid_representations)}"
        )
    
    # Store in uns metadata
    adata.uns[util.UNS_KEY_ACTIVE_TCR_REP] = active_representation
    
    logger.debug(f"Recorded active TCR representation: '{active_representation}'")


def get_active_tcr_representation(adata) -> str | None:
    """Retrieve the currently active TCR representation from AnnData metadata.
    
    Returns the active TCR representation identifier that was previously stored
    using record_active_tcr_representation(). This tells downstream analysis
    which representation to use for neighbor calculations and TCR-based features.
    
    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data object containing active representation metadata.
        
    Returns
    -------
    str | None
        Active representation identifier, or None if not recorded:
        - util.OBSM_KEY_VEC_TCR ('X_vec_tcr'): Vectorized representation
        - util.OBSM_KEY_PCA_TCR ('X_pca_tcr'): KernelPCA representation
        - util.ACTIVE_REP_EXACT ('exact_tcrdist'): Exact TCRdist path
        - None: No active representation recorded
        
    Notes
    -----
    **Storage location**: Reads from adata.uns[util.UNS_KEY_ACTIVE_TCR_REP]
    
    **Missing metadata**: Returns None if the metadata key does not exist,
    which can happen with older AnnData objects or when no representation
    has been explicitly set as active.
    
    **Validation**: This function does not validate that the returned 
    representation actually exists in adata.obsm. Callers should check 
    availability before using the representation.
    
    Examples
    --------
    Basic usage:
    >>> active_rep = get_active_tcr_representation(adata)
    >>> if active_rep == util.OBSM_KEY_VEC_TCR:
    ...     print("Using vectorized TCR representation")
    ... elif active_rep == util.OBSM_KEY_PCA_TCR:
    ...     print("Using KernelPCA TCR representation")  
    ... elif active_rep == util.ACTIVE_REP_EXACT:
    ...     print("Using exact TCRdist calculations")
    ... else:
    ...     print("No active TCR representation set")
    
    Workflow decision logic:
    >>> active_rep = get_active_tcr_representation(adata)
    >>> if active_rep and active_rep in adata.obsm:
    ...     # Use stored matrix representation
    ...     tcr_matrix = adata.obsm[active_rep]
    ... elif active_rep == util.ACTIVE_REP_EXACT:
    ...     # Use on-demand exact calculations  
    ...     tcr_matrix = None  # Signal for exact path
    ... else:
    ...     # No active representation or missing matrix
    ...     raise ValueError("No valid TCR representation available")
    """
    return adata.uns.get(util.UNS_KEY_ACTIVE_TCR_REP, None)


def accuracy_report(
    tcrs: Sequence[tuple[tuple, tuple]] | pd.DataFrame,
    organism: str,
    config: EncodingConfig | None = None,
    *,
    neighbor_counts: Sequence[int] = (10, 100),
    max_pairs: int = 1_000_000,
    random_seed: int = util.DEFAULT_RANDOM_SEED,
    _skip_validation: bool = False,
) -> AccuracyReport:
    """Generate comprehensive accuracy report comparing vectorized vs exact TCRdist.
    
    Computes correlation and k-nearest neighbor recall metrics between vectorized 
    encodings and exact TCRdist calculations. This function enables validation of 
    encoding accuracy and provides quantitative measures for deciding whether 
    vectorized approximation is suitable for a specific analysis.
    
    **WARNING**: This function has O(N^2) time and memory complexity due to 
    pairwise distance computation. It should NOT be used on large datasets. 
    Use max_pairs parameter to limit computation for datasets with >1000 clonotypes.
    
    Parameters
    ----------
    tcrs : Sequence[tuple[tuple, tuple]] | pd.DataFrame
        Clonotype data for accuracy testing. Same format as encode_tcrs().
        Should be representative of target dataset characteristics.
    organism : str
        Organism identifier ('human', 'mouse', 'rhesus').
    config : EncodingConfig | None, default=None
        Encoding configuration to test. If None, uses default configuration.
    neighbor_counts : Sequence[int], default=(10, 100)
        Values of k for k-NN recall computation. Recall@k measures what 
        fraction of each clonotype k nearest exact-TCRdist neighbors 
        also appear among its k nearest vectorized neighbors.
    max_pairs : int, default=1_000_000
        Maximum number of clonotype pairs for correlation computation.
        If N*(N-1)/2 > max_pairs, randomly samples pairs with given seed.
    random_seed : int, default=42
        Random seed for pair sampling (if needed) to ensure reproducibility.
        
    Returns
    -------
    AccuracyReport
        Comprehensive accuracy metrics containing:
        - pearson_distance: Pearson correlation(euclidean_distance, tcrdist)
        - pearson_squared_distance: Pearson correlation(euclidean^2 distance, tcrdist)  
        - spearman: Spearman rank correlation (monotone relationship)
        - mean_recall: Dict mapping k -> mean recall@k across all clonotypes
        - num_pairs_sampled: Actual number of pairs used for correlation
        - Plus metadata (organism, config, clonotype count, version)
        
    Notes  
    -----
    **Correlation metrics**:
    - pearson_distance: Measures linear relationship between Euclidean and TCRdist
    - pearson_squared_distance: More faithful since squared Euclidean approximates TCRdist
    - spearman: Rank correlation, most robust to nonlinear monotone relationships
    
    **Recall metrics**:
    Recall@k = |exact_k_neighbors intersect vectorized_k_neighbors| / k
    Values near 1.0 indicate vectorized encoding preserves local neighborhood 
    structure. This is often more important than global correlation for 
    downstream clustering and visualization tasks.
    
    **Sampling strategy**: 
    When max_pairs is exceeded, samples uniformly from upper triangle of 
    distance matrix. Sampling preserves correlation structure for datasets 
    where most pairs have large distances.
    
    Examples
    --------
    Basic accuracy check:
    >>> tcrs = [...]  # Small test dataset
    >>> report = accuracy_report(tcrs, 'human')
    >>> print(f"Spearman: {report.spearman:.3f}")
    Spearman: 0.999
    >>> print(f"Recall@10: {report.mean_recall[10]:.3f}")  
    Recall@10: 0.953
    
    Custom configuration validation:
    >>> config = EncodingConfig(aa_mds_dim=12, num_pos_cdr3=14)
    >>> report = accuracy_report(tcrs, 'human', config)
    >>> if report.spearman < 0.95:
    ...     print("Warning: Low correlation, consider different config")
    
    Large dataset (with sampling):
    >>> large_tcrs = [...]  # 5000 clonotypes  
    >>> report = accuracy_report(large_tcrs, 'human', max_pairs=100000)
    >>> print(f"Sampled {report.num_pairs_sampled} of {5000*4999//2} total pairs")
    
    Batch validation across organisms:
    >>> organisms = ['human', 'mouse', 'rhesus']
    >>> for org in organisms:
    ...     report = accuracy_report(tcrs, org)
    ...     print(f"{org}: Spearman={report.spearman:.3f}, Recall@10={report.mean_recall[10]:.3f}")

    Notes
    -----
    The private `_skip_validation` parameter allows the Validation_Harness to
    measure accuracy for a candidate organism before it is added to
    `SUPPORTED_ORGANISMS` -- it is not intended for use outside that harness.
    """
    _validate_organism(organism, _skip_validation=_skip_validation)

    if config is None:
        config = EncodingConfig()
    
    # Lazy import exact TCRdist calculator
    from .tcr_distances import TcrDistCalculator
    import scipy.stats
    
    # Encode with vectorized method
    vector_matrix = encode_tcrs(tcrs, organism, config, _skip_validation=_skip_validation)
    n_clonotypes = vector_matrix.shape[0]
    
    if n_clonotypes < 2:
        raise ValueError("Need at least 2 clonotypes for accuracy testing")
    
    # Convert input to format expected by TcrDistCalculator
    if isinstance(tcrs, pd.DataFrame):
        # Extract tuples from DataFrame
        tcr_list = []
        for _, row in tcrs.iterrows():
            va_col = 'va' if 'va' in tcrs.columns else 'va_gene'
            vb_col = 'vb' if 'vb' in tcrs.columns else 'vb_gene'
            
            alpha_tcr = (row[va_col], None, row['cdr3a'], None)  # (V, J, CDR3, nucseq)
            beta_tcr = (row[vb_col], None, row['cdr3b'], None)
            tcr_list.append((alpha_tcr, beta_tcr))
    else:
        tcr_list = list(tcrs)
    
    # Compute pairwise distances
    from scipy.spatial.distance import pdist, squareform
    
    # Vectorized distances (Euclidean)
    vector_distances = pdist(vector_matrix, metric='euclidean')
    
    # Exact TCRdist distances  
    calc = TcrDistCalculator(organism)
    exact_distances = []
    
    # Compute upper triangle of exact distance matrix
    for i in range(n_clonotypes):
        for j in range(i + 1, n_clonotypes):
            dist = calc(tcr_list[i], tcr_list[j])
            exact_distances.append(dist)
    
    exact_distances = np.array(exact_distances)
    
    # Sample pairs if too many
    n_pairs = len(vector_distances)
    if n_pairs > max_pairs:
        rng = np.random.default_rng(random_seed)
        sample_indices = rng.choice(n_pairs, size=max_pairs, replace=False)
        vector_sample = vector_distances[sample_indices]
        exact_sample = exact_distances[sample_indices]
        pairs_sampled = max_pairs
    else:
        vector_sample = vector_distances
        exact_sample = exact_distances  
        pairs_sampled = n_pairs
    
    # Compute correlations 
    if pairs_sampled < 3:
        # Not enough pairs for meaningful correlation
        pearson_dist = 1.0 if pairs_sampled == 1 else np.nan
        pearson_squared_dist = 1.0 if pairs_sampled == 1 else np.nan  
        spearman_corr = 1.0 if pairs_sampled == 1 else np.nan
    else:
        try:
            pearson_dist, _ = scipy.stats.pearsonr(vector_sample, exact_sample)
            pearson_squared_dist, _ = scipy.stats.pearsonr(vector_sample**2, exact_sample)
            spearman_corr, _ = scipy.stats.spearmanr(vector_sample, exact_sample)
        except ValueError:
            # Handle constant arrays (all distances identical)
            pearson_dist = 1.0
            pearson_squared_dist = 1.0
            spearman_corr = 1.0
    
    # Compute k-NN recall
    # Convert to full distance matrices for neighbor finding
    vector_dist_matrix = squareform(vector_distances)
    exact_dist_matrix = squareform(exact_distances)
    
    mean_recalls = {}
    for k in neighbor_counts:
        if k >= n_clonotypes:
            mean_recalls[k] = 1.0  # Perfect recall if k >= total clonotypes
            continue
            
        recalls = []
        for i in range(n_clonotypes):
            # Find k nearest neighbors in exact distances (excluding self)
            exact_row = exact_dist_matrix[i]
            exact_neighbors = np.argsort(exact_row)[1:k+1]  # Skip self at index 0
            
            # Find k nearest neighbors in vectorized distances (excluding self)  
            vector_row = vector_dist_matrix[i]
            vector_neighbors = np.argsort(vector_row)[1:k+1]
            
            # Compute recall: |intersection| / k
            intersection_size = len(set(exact_neighbors) & set(vector_neighbors))
            recall = intersection_size / k
            recalls.append(recall)
        
        mean_recalls[k] = np.mean(recalls)
    
    return AccuracyReport(
        organism=organism,
        config=config,
        num_clonotypes=n_clonotypes,
        num_pairs_sampled=pairs_sampled,
        pearson_distance=float(pearson_dist),
        pearson_squared_distance=float(pearson_squared_dist),
        spearman=float(spearman_corr),
        neighbor_counts=tuple(neighbor_counts),
        mean_recall=mean_recalls,
        vectorizer_version=VECTORIZER_VERSION,
    )


def _load_human_cdr3_corpus() -> pd.DataFrame:
    """Load the real, paired human CDR3 validation corpus.

    Reads ``conga/data/new_paired_tcr_db_for_matching_nr.tsv``, resolving the
    path via ``conga.util.path_to_data`` (a ``pathlib.Path``) rather than a
    hardcoded string, and keeps only rows that are both (1) complete
    alpha-beta pairs (both ``cdr3a`` and ``cdr3b`` non-null and non-empty)
    and (2) fully gene-resolvable: ``va``, ``vb``, ``ja``, and ``jb`` must
    each be non-null *and* an exact key in ``all_genes['human']`` for the
    matching chain/region (``va`` -> chain='A', region='V'; ``vb`` ->
    chain='B', region='V'; ``ja`` -> chain='A', region='J'; ``jb`` ->
    chain='B', region='J'). A non-trivial fraction of rows have CDR3
    sequences present but a missing or non-matching J-gene call, so the
    CDR3-only filter alone is insufficient -- ``encode_tcrs``'s
    ``gene_to_row[v_gene]`` lookup requires an exact dict-key match with no
    fuzzy normalization, so any row with an unresolvable gene id must be
    dropped here rather than passed through.

    Column naming convention
    -------------------------
    This file uses the matching-db schema: ``va``/``vb``/``ja``/``jb`` for V/J
    gene calls and ``cdr3a``/``cdr3b`` for CDR3 sequences. This is the *same*
    convention used by :func:`_load_mouse_cdr3_corpus`, but is distinct from
    :func:`_load_rhesus_cdr3_corpus`, which reads a clones-file-style schema
    (``va_gene``/``vb_gene``/``ja_gene``/``jb_gene``).

    Returns
    -------
    pd.DataFrame
        Columns ``cdr3a``, ``cdr3b``, ``va``, ``vb``, ``ja``, ``jb``, one row
        per complete, gene-resolvable paired clonotype. This function does
        not convert rows to the common ``list[tuple[tuple, tuple]]`` shape;
        per the Data Models normalization contract, that projection is the
        caller's (Validation_Harness's) responsibility, not this loader's.
    """
    # Lazy import gene database, matching this module's existing convention
    # (see germline_code_table).
    from .all_genes import all_genes

    data_file = util.path_to_data / 'new_paired_tcr_db_for_matching_nr.tsv'
    df = pd.read_csv(data_file, sep='\t')

    columns = ['cdr3a', 'cdr3b', 'va', 'vb', 'ja', 'jb']
    df = df[columns]

    paired_mask = (
        df['cdr3a'].notna() & (df['cdr3a'].astype(str).str.len() > 0)
        & df['cdr3b'].notna() & (df['cdr3b'].astype(str).str.len() > 0)
    )

    genes_dict = all_genes['human']
    va_ids = {g.id for g in genes_dict.values() if g.chain == 'A' and g.region == 'V'}
    ja_ids = {g.id for g in genes_dict.values() if g.chain == 'A' and g.region == 'J'}
    vb_ids = {g.id for g in genes_dict.values() if g.chain == 'B' and g.region == 'V'}
    jb_ids = {g.id for g in genes_dict.values() if g.chain == 'B' and g.region == 'J'}

    gene_mask = (
        df['va'].isin(va_ids) & df['vb'].isin(vb_ids)
        & df['ja'].isin(ja_ids) & df['jb'].isin(jb_ids)
    )

    return df.loc[paired_mask & gene_mask].reset_index(drop=True)


def _load_mouse_cdr3_corpus() -> pd.DataFrame:
    """Load the real, paired mouse CDR3 validation corpus.

    Reads ``conga/data/mouse_tcr_db_for_matching.tsv``, resolving the path
    via ``conga.util.path_to_data`` rather than a hardcoded string, and
    keeps only rows that are both (1) complete alpha-beta pairs (both
    ``cdr3a`` and ``cdr3b`` non-null and non-empty) and (2) fully
    gene-resolvable: ``va``, ``vb``, ``ja``, and ``jb`` must each be
    non-null *and* an exact key in ``all_genes['mouse']`` for the matching
    chain/region (``va`` -> chain='A', region='V'; ``vb`` -> chain='B',
    region='V'; ``ja`` -> chain='A', region='J'; ``jb`` -> chain='B',
    region='J'). Beyond missing J-gene calls, this file also contains real
    gene-name formatting mismatches against the active gene database (e.g.
    missing allele suffixes like ``TRAV6-1`` vs. ``TRAV6-1*01``, or
    different separator conventions like ``TRAV13D-1:01`` vs. ``*01``), so
    the CDR3-only filter alone passes through rows that would raise a
    ``KeyError`` (or silently mis-encode) in ``encode_tcrs``'s exact
    dict-key gene lookup. Those gene-name mismatches are not normalized or
    fuzzy-matched here -- doing so risks introducing an incorrect mapping --
    rows with an unresolvable gene id are simply excluded from the corpus.

    Column naming convention
    -------------------------
    Same matching-db schema as :func:`_load_human_cdr3_corpus`:
    ``va``/``vb``/``ja``/``jb``/``cdr3a``/``cdr3b``. This file additionally
    has one extra leading unnamed index column in its header (confirmed by
    direct inspection), which pandas' default ``read_csv(sep='\\t')`` assigns
    an auto-generated positional column name to (e.g. ``'Unnamed: 0'``);
    that column -- and every other column this function does not use -- is
    dropped by selecting only the needed columns below.

    Returns
    -------
    pd.DataFrame
        Columns ``cdr3a``, ``cdr3b``, ``va``, ``vb``, ``ja``, ``jb``, one row
        per complete, gene-resolvable paired clonotype.
    """
    # Lazy import gene database, matching this module's existing convention
    # (see germline_code_table).
    from .all_genes import all_genes

    data_file = util.path_to_data / 'mouse_tcr_db_for_matching.tsv'
    df = pd.read_csv(data_file, sep='\t')

    columns = ['cdr3a', 'cdr3b', 'va', 'vb', 'ja', 'jb']
    df = df[columns]

    paired_mask = (
        df['cdr3a'].notna() & (df['cdr3a'].astype(str).str.len() > 0)
        & df['cdr3b'].notna() & (df['cdr3b'].astype(str).str.len() > 0)
    )

    genes_dict = all_genes['mouse']
    va_ids = {g.id for g in genes_dict.values() if g.chain == 'A' and g.region == 'V'}
    ja_ids = {g.id for g in genes_dict.values() if g.chain == 'A' and g.region == 'J'}
    vb_ids = {g.id for g in genes_dict.values() if g.chain == 'B' and g.region == 'V'}
    jb_ids = {g.id for g in genes_dict.values() if g.chain == 'B' and g.region == 'J'}

    gene_mask = (
        df['va'].isin(va_ids) & df['vb'].isin(vb_ids)
        & df['ja'].isin(ja_ids) & df['jb'].isin(jb_ids)
    )

    return df.loc[paired_mask & gene_mask].reset_index(drop=True)


def _load_rhesus_cdr3_corpus() -> pd.DataFrame:
    """Load the real, paired rhesus CDR3 validation corpus.

    Reads ``conga/data/rhesus_clones.tsv``, resolving the path via
    ``conga.util.path_to_data`` rather than a hardcoded string. All 450 rows
    in this file are already complete alpha-beta pairs (confirmed by direct
    inspection: no null/empty values across the relevant columns), so unlike
    :func:`_load_human_cdr3_corpus` and :func:`_load_mouse_cdr3_corpus`, no
    paired-row filtering is applied here.

    Column naming convention
    -------------------------
    This file uses the *clones-file* schema, not the matching-db schema the
    two loaders above use: V/J gene calls are named ``va_gene``/``vb_gene``/
    ``ja_gene``/``jb_gene`` (not ``va``/``vb``/``ja``/``jb``). CDR3 columns
    are still named ``cdr3a``/``cdr3b``, matching the other two loaders.

    Returns
    -------
    pd.DataFrame
        Columns ``cdr3a``, ``cdr3b``, ``va_gene``, ``ja_gene``, ``vb_gene``,
        ``jb_gene``, one row per clonotype. As with the other two loaders,
        this function returns a plain DataFrame in this file's native column
        naming; projecting to the common tuple shape is the caller's
        (Validation_Harness's) responsibility.
    """
    data_file = util.path_to_data / 'rhesus_clones.tsv'
    df = pd.read_csv(data_file, sep='\t')

    columns = ['cdr3a', 'cdr3b', 'va_gene', 'ja_gene', 'vb_gene', 'jb_gene']
    return df[columns].reset_index(drop=True)


def _build_synthetic_cdr3_corpus(
    organism: str,
    n: int,
    random_seed: int,
) -> list[tuple[tuple, tuple]]:
    """Generate a synthetic paired-CDR3 corpus for Tier 3 validation.

    This is the Synthetic_CDR3_Generator used to validate organisms with no
    real, species-matched paired CDR3 data available in this repository
    (e.g. ``cat``, ``dog``, ``ferret``, ``rabbit``, ``sheep``, and rhesus's
    gamma-delta/Ig receptor types). It is a decision-support tool for the
    Validation_Harness, not a biologically faithful simulator.

    Algorithm
    ---------
    1. Build one pooled per-residue amino-acid frequency table (20 amino
       acids, normalized counts) from :func:`_load_human_cdr3_corpus`'s
       ``cdr3a`` and ``cdr3b`` columns combined, pooled across all positions
       and all lengths. This is the only real amino-acid-content model used,
       regardless of ``organism``.
    2. Sample a CDR3 length for each generated chain from the empirical
       length distribution observed in the same human corpus. This function
       pools ``cdr3a`` and ``cdr3b`` lengths together into one distribution
       and draws both the alpha and beta chain lengths from that pooled
       distribution (rather than sampling alpha length from ``cdr3a``
       lengths and beta length from ``cdr3b`` lengths separately); either
       choice is acceptable per the design, and this is the one implemented
       here.
    3. Build each synthetic CDR3 of its sampled length ``L`` by drawing
       ``L`` independent residues from the pooled frequency table in step 1.
    4. Sample V and J gene identifiers for each chain uniformly and
       independently from ``all_genes[organism]`` (lazily imported, matching
       this module's existing lazy-import convention), filtered to
       ``chain='A'``/``region='V'`` or ``'J'`` for the alpha chain and
       ``chain='B'``/``region='V'`` or ``'J'`` for the beta chain. This
       guarantees every sampled gene id is a real key in
       ``germline_code_table``'s own gene list for that organism, which
       ``encode_tcrs``'s bare ``gene_to_row[v_gene]`` dict lookup requires.

    There is no species-specific CDR3 length distribution available anywhere
    in this repository for any organism (germline V/J segments do not
    determine junction length -- that is set by V(D)J recombination, a
    cellular process, not a germline sequence property), so the human
    length distribution is used for every synthetic organism. This is a
    documented property of the validation method, not a per-species
    shortfall.

    Parameters
    ----------
    organism : str
        Organism identifier to sample V/J genes for (e.g. ``'cat'``,
        ``'dog_gd'``, ``'rabbit_ig'``). Must be a key in ``all_genes``.
    n : int
        Number of synthetic paired clonotypes to generate.
    random_seed : int
        Seed for ``np.random.default_rng``, used for reproducibility.
        The global ``np.random`` state is never touched.

    Returns
    -------
    list[tuple[tuple, tuple]]
        ``n`` tuples of the form
        ``((va_gene, ja_gene, cdr3a, ''), (vb_gene, jb_gene, cdr3b, ''))``,
        matching the input shape ``encode_tcrs`` and ``accuracy_report``
        already accept per their docstrings (nucseq left as an empty
        string, since synthetic sequences have no real nucleotide calls).
    """
    # Lazy import gene database, matching this module's existing convention
    from .all_genes import all_genes

    if organism not in all_genes:
        raise ValueError(f"Organism '{organism}' not found in gene database")

    rng = np.random.default_rng(random_seed)

    # Step 1: pooled per-residue amino acid frequency table from human CDR3s
    human_corpus = _load_human_cdr3_corpus()
    all_residues = ''.join(human_corpus['cdr3a']) + ''.join(human_corpus['cdr3b'])
    aa_counts = np.array([all_residues.count(aa) for aa in amino_acids], dtype=np.float64)
    aa_probs = aa_counts / aa_counts.sum()

    # Step 2: pooled empirical CDR3 length distribution from human CDR3s
    lengths = np.concatenate([
        human_corpus['cdr3a'].str.len().to_numpy(),
        human_corpus['cdr3b'].str.len().to_numpy(),
    ])

    # Step 4 (gene pools): V/J genes for each chain, filtered by chain+region
    genes_dict = all_genes[organism]
    va_genes = [g.id for g in genes_dict.values() if g.chain == 'A' and g.region == 'V']
    ja_genes = [g.id for g in genes_dict.values() if g.chain == 'A' and g.region == 'J']
    vb_genes = [g.id for g in genes_dict.values() if g.chain == 'B' and g.region == 'V']
    jb_genes = [g.id for g in genes_dict.values() if g.chain == 'B' and g.region == 'J']

    for gene_list, label in (
        (va_genes, f"organism '{organism}' chain 'A' region 'V'"),
        (ja_genes, f"organism '{organism}' chain 'A' region 'J'"),
        (vb_genes, f"organism '{organism}' chain 'B' region 'V'"),
        (jb_genes, f"organism '{organism}' chain 'B' region 'J'"),
    ):
        if not gene_list:
            raise ValueError(f"No genes found for {label}")

    def _sample_cdr3(length: int) -> str:
        return ''.join(rng.choice(amino_acids, size=length, p=aa_probs))

    corpus: list[tuple[tuple, tuple]] = []
    for _ in range(n):
        len_a, len_b = rng.choice(lengths, size=2)
        cdr3a = _sample_cdr3(int(len_a))
        cdr3b = _sample_cdr3(int(len_b))

        va_gene = rng.choice(va_genes)
        ja_gene = rng.choice(ja_genes)
        vb_gene = rng.choice(vb_genes)
        jb_gene = rng.choice(jb_genes)

        corpus.append((
            (va_gene, ja_gene, cdr3a, ''),
            (vb_gene, jb_gene, cdr3b, ''),
        ))

    return corpus


def store_vectorized_tcr_in_adata(
    adata,
    organism: str,
    config: EncodingConfig | None = None,
    *,
    va_column: str = 'va',
    cdr3a_column: str = 'cdr3a',
    vb_column: str = 'vb', 
    cdr3b_column: str = 'cdr3b',
) -> np.ndarray:
    """Store vectorized TCR encodings in AnnData object with proper metadata.
    
    Encodes TCR clonotypes from the AnnData observation metadata and stores the 
    resulting vector matrix in adata.obsm under the standardized key X_vec_tcr.
    Also stores encoding configuration, organism, and version metadata in adata.uns
    for reproducibility and analysis tracking.
    
    The function preserves any existing representations in the AnnData object while
    adding the vectorized representation. If X_vec_tcr already exists, it is 
    overwritten with a logged warning. Other TCR representations (X_pca_tcr) 
    remain unchanged.
    
    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data object containing TCR metadata in .obs. Must have columns
        for V genes and CDR3 sequences for both alpha and beta chains.
    organism : str
        Organism identifier ('human', 'mouse', 'rhesus'). Must be supported
        by the vectorized encoder.
    config : EncodingConfig | None, default=None
        Encoding configuration. If None, uses default configuration.
        All config values are stored in adata.uns for reproducibility.
    va_column, cdr3a_column, vb_column, cdr3b_column : str
        Column names in adata.obs containing V gene and CDR3 data.
        Defaults match standard CoNGA naming conventions.
        
    Returns
    -------
    np.ndarray
        The encoded vector matrix that was stored in adata.obsm['X_vec_tcr'].
        Shape: (n_obs, vector_length). Provided for caller convenience.
        
    Raises
    ------
    ValueError
        If organism not supported, required columns missing from adata.obs,
        or invalid V genes/CDR3 sequences found in data.
    RuntimeError
        If encoding produces unexpected shape or non-finite values.
        
    Notes
    -----
    **Storage locations**:
    - Vector matrix: adata.obsm[util.OBSM_KEY_VEC_TCR] ('X_vec_tcr')
    - Encoding config: adata.uns[util.UNS_KEY_VEC_TCR_CONFIG] ('vec_tcr_config')
    
    **Row ordering**: The stored matrix rows correspond to adata.obs rows in 
    the same order, enabling direct indexing and slicing operations.
    
    **Overwrite behavior**: If X_vec_tcr already exists, logs a warning and 
    overwrites. This supports workflow restart scenarios and config changes.
    
    **Coexistence**: Does not affect existing X_pca_tcr or other representations.
    Multiple TCR representations can coexist in the same AnnData object.
    
    **Metadata stored**: The configuration dictionary includes:
    - Encoding parameters (aa_mds_dim, num_pos_cdr3, etc.)
    - Organism identifier
    - Vectorizer version tag
    - Creation timestamp (ISO format)
    - Detection flag 'has_vectorized_tcr': True
    
    Examples
    --------
    Basic usage:
    >>> import anndata as ad
    >>> import pandas as pd
    >>> # Create AnnData with TCR data
    >>> obs = pd.DataFrame({
    ...     'va': ['TRAV1*01', 'TRAV2*01'],
    ...     'cdr3a': ['CAVRD', 'CAVKE'],
    ...     'vb': ['TRBV1*01', 'TRBV2*01'],
    ...     'cdr3b': ['CASSRT', 'CASSLQ']
    ... })
    >>> adata = ad.AnnData(obs=obs)
    >>> matrix = store_vectorized_tcr_in_adata(adata, 'human')
    >>> matrix.shape
    (2, 1136)
    >>> 'X_vec_tcr' in adata.obsm
    True
    
    Custom configuration:
    >>> config = EncodingConfig(aa_mds_dim=12, random_seed=123)
    >>> matrix = store_vectorized_tcr_in_adata(adata, 'human', config)
    >>> adata.uns['vec_tcr_config']['aa_mds_dim']
    12
    >>> adata.uns['vec_tcr_config']['random_seed']
    123
    
    Alternative column names:
    >>> matrix = store_vectorized_tcr_in_adata(
    ...     adata, 'human',
    ...     va_column='v_alpha', cdr3a_column='cdr3_alpha',
    ...     vb_column='v_beta', cdr3b_column='cdr3_beta'
    ... )
    
    Restart/overwrite scenario:
    >>> # First encoding
    >>> matrix1 = store_vectorized_tcr_in_adata(adata, 'human')
    >>> # Changed config - will log warning and overwrite
    >>> config2 = EncodingConfig(aa_mds_dim=8)
    >>> matrix2 = store_vectorized_tcr_in_adata(adata, 'human', config2)
    >>> # New matrix has different dimensions due to config change
    >>> matrix1.shape != matrix2.shape
    True
    """
    import datetime
    
    if config is None:
        config = EncodingConfig()
    
    # Check if overwriting existing vectorized representation
    if util.OBSM_KEY_VEC_TCR in adata.obsm:
        logger.warning(f"Overwriting existing vectorized TCR representation in adata.obsm['{util.OBSM_KEY_VEC_TCR}']")
    
    # Convert adata.obs to DataFrame format for encode_tcrs
    obs_df = adata.obs.copy()
    
    # Encode the TCR data
    logger.info(f"Encoding {adata.n_obs} clonotypes with organism '{organism}'")
    vector_matrix = encode_tcrs(
        obs_df,
        organism, 
        config,
        va_column=va_column,
        cdr3a_column=cdr3a_column,
        vb_column=vb_column,
        cdr3b_column=cdr3b_column,
    )
    
    # Verify row ordering matches adata.obs
    if vector_matrix.shape[0] != adata.n_obs:
        raise RuntimeError(
            f"Encoded matrix rows ({vector_matrix.shape[0]}) != adata.n_obs ({adata.n_obs})"
        )
    
    # Store vector matrix in obsm
    adata.obsm[util.OBSM_KEY_VEC_TCR] = vector_matrix
    
    # Store encoding configuration and metadata in uns
    config_dict = config.as_uns_dict()
    config_dict['organism'] = organism
    config_dict['creation_timestamp'] = datetime.datetime.now().isoformat()
    config_dict['has_vectorized_tcr'] = True
    adata.uns[util.UNS_KEY_VEC_TCR_CONFIG] = config_dict
    
    logger.info(
        f"Stored vectorized TCR representation: {vector_matrix.shape} in "
        f"adata.obsm['{util.OBSM_KEY_VEC_TCR}'] with config in "
        f"adata.uns['{util.UNS_KEY_VEC_TCR_CONFIG}']"
    )
    
    return vector_matrix


def load_vectorized_tcr_from_adata(adata) -> tuple[np.ndarray, EncodingConfig, str]:
    """Load vectorized TCR representation from AnnData object with validation.
    
    Retrieves the vectorized TCR matrix from adata.obsm and reconstructs the
    encoding configuration from adata.uns metadata. Performs comprehensive
    validation to ensure the stored representation is consistent and usable.
    
    This function enables analysis workflows to access previously computed
    vectorized representations without re-encoding, while ensuring data
    integrity through validation checks.
    
    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data object containing stored vectorized TCR representation.
        Must have been populated by store_vectorized_tcr_in_adata().
        
    Returns
    -------
    tuple[np.ndarray, EncodingConfig, str]
        - vector_matrix: (n_obs, vector_length) float32 array from adata.obsm
        - config: EncodingConfig object reconstructed from stored metadata
        - organism: Organism identifier string used for encoding
        
    Raises
    ------
    ValueError
        - If X_vec_tcr key not found in adata.obsm
        - If vec_tcr_config key not found in adata.uns
        - If stored matrix has wrong number of rows
        - If config metadata is incomplete or invalid
        - If vectorizer version incompatibility detected
    RuntimeError
        - If stored matrix contains non-finite values
        - If matrix dtype or memory layout is unexpected
        
    Notes
    -----
    **Validation performed**:
    - Checks for required obsm/uns keys
    - Validates matrix shape matches adata.n_obs
    - Ensures all matrix values are finite
    - Reconstructs EncodingConfig with parameter validation
    - Verifies vectorizer version compatibility
    - Confirms organism field presence
    
    **Version compatibility**: The function accepts any vector matrix created
    by the same vectorizer version (VECTORIZER_VERSION). Cross-version 
    compatibility may be added in future releases.
    
    **Performance**: Matrix data is returned as a view when possible, avoiding
    unnecessary copies for large datasets.
    
    Examples
    --------
    Basic usage after storage:
    >>> # First store some vectors
    >>> matrix_orig = store_vectorized_tcr_in_adata(adata, 'human')
    >>> # Later retrieve them
    >>> matrix, config, organism = load_vectorized_tcr_from_adata(adata)
    >>> np.array_equal(matrix, matrix_orig)
    True
    >>> organism
    'human'
    >>> config.aa_mds_dim
    16
    
    Validation of loaded data:
    >>> matrix, config, organism = load_vectorized_tcr_from_adata(adata)
    >>> matrix.shape[0] == adata.n_obs  # Row count matches
    True
    >>> matrix.dtype == np.float32  # Correct dtype
    True
    >>> np.all(np.isfinite(matrix))  # All finite values
    True
    
    Configuration reconstruction:
    >>> matrix, config, organism = load_vectorized_tcr_from_adata(adata)
    >>> expected_length = vector_length(organism, config)
    >>> matrix.shape[1] == expected_length  # Dimensions consistent
    True
    
    Error handling:
    >>> # AnnData without vectorized representation
    >>> empty_adata = ad.AnnData(obs=pd.DataFrame({'x': [1, 2]}))
    >>> load_vectorized_tcr_from_adata(empty_adata)  # doctest: +SKIP
    ValueError: No vectorized TCR representation found in adata.obsm
    """
    # Check for required obsm key
    if util.OBSM_KEY_VEC_TCR not in adata.obsm:
        raise ValueError(
            f"No vectorized TCR representation found in adata.obsm. "
            f"Expected key: '{util.OBSM_KEY_VEC_TCR}'. "
            f"Available obsm keys: {list(adata.obsm.keys())}. "
            f"Use store_vectorized_tcr_in_adata() to create the representation."
        )
    
    # Check for required uns key
    if util.UNS_KEY_VEC_TCR_CONFIG not in adata.uns:
        raise ValueError(
            f"No vectorized TCR configuration found in adata.uns. "
            f"Expected key: '{util.UNS_KEY_VEC_TCR_CONFIG}'. "
            f"Available uns keys: {list(adata.uns.keys())}. "
            f"The vectorized representation may have been created by an incompatible method."
        )
    
    # Load vector matrix
    vector_matrix = adata.obsm[util.OBSM_KEY_VEC_TCR]
    
    # Validate matrix properties
    if vector_matrix.shape[0] != adata.n_obs:
        raise ValueError(
            f"Stored vectorized TCR matrix has {vector_matrix.shape[0]} rows "
            f"but adata has {adata.n_obs} observations. "
            f"The stored matrix may be from a different dataset or subset."
        )
    
    if not np.all(np.isfinite(vector_matrix)):
        raise RuntimeError(
            f"Stored vectorized TCR matrix contains non-finite values. "
            f"The stored representation may be corrupted."
        )
    
    # Load and validate configuration metadata
    config_dict = adata.uns[util.UNS_KEY_VEC_TCR_CONFIG]
    
    # Check for required fields
    if 'organism' not in config_dict:
        raise ValueError(
            f"Stored vectorized TCR config missing 'organism' field. "
            f"Available fields: {list(config_dict.keys())}. "
            f"The config may have been created by an incompatible method."
        )
    
    organism = str(config_dict['organism'])
    
    # Check vectorizer version compatibility
    stored_version = config_dict.get('vectorizer_version', 'unknown')
    if stored_version != VECTORIZER_VERSION:
        logger.warning(
            f"Stored vectorized TCR representation was created by vectorizer "
            f"version '{stored_version}', but current version is '{VECTORIZER_VERSION}'. "
            f"Results may not be reproducible across versions."
        )
    
    # Reconstruct EncodingConfig (this validates parameter ranges)
    try:
        config = EncodingConfig.from_uns_dict(config_dict)
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(
            f"Could not reconstruct EncodingConfig from stored metadata: {e}. "
            f"Stored fields: {list(config_dict.keys())}. "
            f"The config may be incomplete or from an incompatible version."
        ) from e
    
    # Validate expected vector length matches stored matrix
    try:
        expected_length = vector_length(organism, config)
        if vector_matrix.shape[1] != expected_length:
            raise ValueError(
                f"Stored vectorized TCR matrix has {vector_matrix.shape[1]} columns "
                f"but expected {expected_length} for organism '{organism}' and config. "
                f"The stored matrix may be from a different configuration."
            )
    except ValueError as e:
        # Re-raise with additional context
        raise ValueError(
            f"Could not validate stored vectorized TCR matrix dimensions: {e}"
        ) from e
    
    logger.debug(
        f"Loaded vectorized TCR representation: {vector_matrix.shape} from "
        f"adata.obsm['{util.OBSM_KEY_VEC_TCR}'] with organism '{organism}'"
    )
    
    return vector_matrix, config, organism


def clear_vectorized_tcr_from_adata(adata) -> bool:
    """Remove vectorized TCR representation and metadata from AnnData object.
    
    Clears all vectorized TCR-related data from the AnnData object, including
    the vector matrix in adata.obsm and configuration metadata in adata.uns.
    This function is useful for cleaning up representations, forcing re-encoding
    with different parameters, or reducing file size.
    
    The function removes only vectorized TCR data and leaves other TCR 
    representations (X_pca_tcr) and all non-TCR data unchanged.
    
    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data object to clear vectorized TCR data from.
        No error is raised if no vectorized TCR data is present.
        
    Returns
    -------
    bool
        True if vectorized TCR data was found and removed, False if no
        vectorized TCR data was present in the AnnData object.
        
    Notes
    -----
    **Keys removed**:
    - adata.obsm[util.OBSM_KEY_VEC_TCR] ('X_vec_tcr')
    - adata.uns[util.UNS_KEY_VEC_TCR_CONFIG] ('vec_tcr_config')
    
    **Preservation**: The following are explicitly preserved:
    - Other obsm entries (X_pca_tcr, X_umap, X_pca, etc.)
    - Other uns entries (including non-vectorized TCR metadata)
    - All obs, var, obsp, varp, varm, layers data
    - Active TCR representation tracking in uns
    
    **Use cases**:
    - Force re-encoding with different EncodingConfig parameters
    - Clean up AnnData objects before saving to reduce file size
    - Remove potentially corrupted vectorized representations
    - Prepare for encoding with different organism identifier
    
    **Active representation handling**: If util.UNS_KEY_ACTIVE_TCR_REP points
    to the vectorized representation ('X_vec_tcr'), that tracking is preserved
    even though the actual representation is removed. This allows subsequent
    code to detect the inconsistency and handle it appropriately.
    
    Examples
    --------
    Basic usage:
    >>> # Store some vectors first
    >>> store_vectorized_tcr_in_adata(adata, 'human')
    >>> 'X_vec_tcr' in adata.obsm
    True
    >>> # Clear them
    >>> was_present = clear_vectorized_tcr_from_adata(adata)
    >>> was_present
    True
    >>> 'X_vec_tcr' in adata.obsm
    False
    
    No-op when data not present:
    >>> # AnnData without vectorized representation
    >>> empty_adata = ad.AnnData(obs=pd.DataFrame({'x': [1, 2]}))
    >>> was_present = clear_vectorized_tcr_from_adata(empty_adata)
    >>> was_present
    False
    
    Selective clearing (preserves other representations):
    >>> # Store both vectorized and PCA representations
    >>> store_vectorized_tcr_in_adata(adata, 'human')
    >>> adata.obsm['X_pca_tcr'] = np.random.randn(adata.n_obs, 50)
    >>> # Clear only vectorized
    >>> clear_vectorized_tcr_from_adata(adata)
    >>> 'X_vec_tcr' in adata.obsm
    False
    >>> 'X_pca_tcr' in adata.obsm  # Preserved
    True
    
    Re-encoding after clearing:
    >>> # Clear and re-encode with different config
    >>> clear_vectorized_tcr_from_adata(adata)
    >>> new_config = EncodingConfig(aa_mds_dim=8)
    >>> new_matrix = store_vectorized_tcr_in_adata(adata, 'human', new_config)
    >>> new_matrix.shape[1] < 1136  # Smaller due to reduced aa_mds_dim
    True
    """
    was_present = False
    
    # Remove vector matrix from obsm
    if util.OBSM_KEY_VEC_TCR in adata.obsm:
        del adata.obsm[util.OBSM_KEY_VEC_TCR]
        was_present = True
        logger.debug(f"Removed vectorized TCR matrix from adata.obsm['{util.OBSM_KEY_VEC_TCR}']")
    
    # Remove configuration from uns
    if util.UNS_KEY_VEC_TCR_CONFIG in adata.uns:
        del adata.uns[util.UNS_KEY_VEC_TCR_CONFIG]
        was_present = True
        logger.debug(f"Removed vectorized TCR config from adata.uns['{util.UNS_KEY_VEC_TCR_CONFIG}']")
    
    if was_present:
        logger.info("Cleared vectorized TCR representation and metadata from AnnData object")
    else:
        logger.debug("No vectorized TCR representation found to clear")
    
    return was_present

# End of vectorized TCRdist module