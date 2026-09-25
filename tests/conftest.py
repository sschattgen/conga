"""
Test fixtures for CoNGA vectorized TCRdist feature.

This module provides pytest fixtures for testing the vectorized TCRdist implementation,
including:
- Human TCR data from the bundled database
- Synthetic mouse and rhesus TCR data 
- Various clonotype sizes and edge cases
- Deterministic seeded data generation

Requirements validated: 10.1, 10.5
"""

import os
import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass


# Constants for reproducible test data
TEST_RANDOM_SEED = 42
DEFAULT_SAMPLE_SIZE = 300
LARGE_SAMPLE_SIZE = 1000
SMALL_SAMPLE_SIZE = 50

# Standard amino acids for CDR3 generation
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

# CDR3 terminal constraints (conventional C...F pattern)
CDR3_N_TERMINAL = 'C'
CDR3_C_TERMINAL = 'F'


@dataclass
class CloneData:
    """Represents a single test clonotype with paired TCR chains."""
    va: str
    ja: str  
    cdr3a: str
    vb: str
    jb: str
    cdr3b: str
    
    def as_tuple(self) -> Tuple[Tuple, Tuple]:
        """Convert to nested tuple format expected by vectorized encoder."""
        return ((self.va, self.ja, self.cdr3a), (self.vb, self.jb, self.cdr3b))
    
    def as_dict(self) -> Dict[str, str]:
        """Convert to dictionary format."""
        return {
            'va': self.va,
            'ja': self.ja, 
            'cdr3a': self.cdr3a,
            'vb': self.vb,
            'jb': self.jb,
            'cdr3b': self.cdr3b
        }


class CloneFixtureGenerator:
    """Generate test clonotypes with proper seeding for reproducibility."""
    
    def __init__(self, random_seed: int = TEST_RANDOM_SEED):
        """Initialize generator with specified random seed."""
        self.rng = np.random.default_rng(random_seed)
        self._gene_cache = {}
        
    def _get_gene_database_path(self) -> Path:
        """Get path to the gene database."""
        # Path from test file to conga/tcrdist/db/combo_xcr.tsv
        test_dir = Path(__file__).parent
        return test_dir.parent / 'conga' / 'tcrdist' / 'db' / 'combo_xcr.tsv'
    
    def _load_genes_for_organism_chain(self, organism: str, chain: str) -> List[str]:
        """Load V gene IDs for the specified organism and chain."""
        cache_key = f"{organism}_{chain}"
        if cache_key in self._gene_cache:
            return self._gene_cache[cache_key]
            
        db_path = self._get_gene_database_path()
        if not db_path.exists():
            # Fallback: generate fake gene names for testing
            if organism == 'human':
                if chain == 'A':
                    genes = [f'TRAV{i}*01' for i in range(1, 41)]
                else:
                    genes = [f'TRBV{i}*01' for i in range(1, 31)]
            elif organism == 'mouse':
                if chain == 'A':
                    genes = [f'TRAV{i}*01' for i in range(1, 21)]
                else:
                    genes = [f'TRBV{i}*01' for i in range(1, 21)]
            else:  # rhesus
                if chain == 'A':
                    genes = [f'TRAV{i}*01' for i in range(1, 21)]
                else:
                    genes = [f'TRBV{i}*01' for i in range(1, 21)]
        else:
            # Load from actual database
            try:
                df = pd.read_csv(db_path, sep='\t')
                # Filter for organism and chain
                mask = (df['organism'] == organism) & (df['chain'] == chain)
                if chain == 'A':
                    gene_col = 'id'  # V gene column
                else:
                    gene_col = 'id'  # V gene column
                genes = df[mask][gene_col].unique().tolist()
                genes = [g for g in genes if g and not pd.isna(g)]
            except Exception:
                # Fallback to synthetic genes if database read fails
                genes = [f'TR{chain}V{i}*01' for i in range(1, 21)]
        
        self._gene_cache[cache_key] = genes
        return genes
    
    def _generate_cdr3(self, min_length: int = 8, max_length: int = 18) -> str:
        """Generate a realistic CDR3 sequence with proper termini."""
        length = self.rng.integers(min_length, max_length + 1)
        
        # Generate middle sequence (length - 2 for termini)
        middle_length = max(0, length - 2)
        middle = ''.join(self.rng.choice(list(AMINO_ACIDS), size=middle_length))
        
        # Add conventional termini
        return CDR3_N_TERMINAL + middle + CDR3_C_TERMINAL
    
    def generate_clonotypes(self, organism: str, n_clonotypes: int, 
                          cdr3_length_range: Tuple[int, int] = (8, 18)) -> List[CloneData]:
        """Generate synthetic clonotypes for the specified organism."""
        va_genes = self._load_genes_for_organism_chain(organism, 'A')
        vb_genes = self._load_genes_for_organism_chain(organism, 'B') 
        
        # For simplicity, use the same genes for J as V (in real data these are separate)
        ja_genes = va_genes[:10] if len(va_genes) > 10 else va_genes
        jb_genes = vb_genes[:10] if len(vb_genes) > 10 else vb_genes
        
        clonotypes = []
        for _ in range(n_clonotypes):
            clonotype = CloneData(
                va=self.rng.choice(va_genes),
                ja=self.rng.choice(ja_genes),
                cdr3a=self._generate_cdr3(*cdr3_length_range),
                vb=self.rng.choice(vb_genes), 
                jb=self.rng.choice(jb_genes),
                cdr3b=self._generate_cdr3(*cdr3_length_range)
            )
            clonotypes.append(clonotype)
            
        return clonotypes


def load_human_tcr_database() -> pd.DataFrame:
    """Load the bundled human TCR database for testing."""
    # Path from test file to conga/data/new_paired_tcr_db_for_matching_nr.tsv
    test_dir = Path(__file__).parent
    db_path = test_dir.parent / 'conga' / 'data' / 'new_paired_tcr_db_for_matching_nr.tsv'
    
    if not db_path.exists():
        raise FileNotFoundError(f"Bundled TCR database not found at {db_path}")
        
    return pd.read_csv(db_path, sep='\t')


def filter_valid_human_clonotypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the human TCR database to valid clonotypes for testing.
    
    Filters for:
    - Present va, vb, cdr3a, cdr3b columns
    - CDR3s with only standard amino acids  
    - CDR3s of reasonable length (>= 6 characters)
    - Remove duplicates
    """
    # Required columns
    required_cols = ['va', 'vb', 'cdr3a', 'cdr3b']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in database")
    
    # Drop rows with missing values in required columns
    df_clean = df[required_cols].dropna()
    
    # Filter CDR3s - only standard amino acids
    amino_acid_pattern = f'^[{AMINO_ACIDS}]+$'
    cdr3a_valid = df_clean['cdr3a'].str.match(amino_acid_pattern, na=False)
    cdr3b_valid = df_clean['cdr3b'].str.match(amino_acid_pattern, na=False)
    df_clean = df_clean[cdr3a_valid & cdr3b_valid]
    
    # Filter CDR3s by length (>= 6 characters)
    len_filter = (df_clean['cdr3a'].str.len() >= 6) & (df_clean['cdr3b'].str.len() >= 6)
    df_clean = df_clean[len_filter]
    
    # Remove duplicates based on the four key columns
    df_clean = df_clean.drop_duplicates(subset=required_cols)
    
    # Reset index
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean


@pytest.fixture
def human_tcr_data() -> pd.DataFrame:
    """Fixture providing the complete filtered human TCR database."""
    df = load_human_tcr_database()
    return filter_valid_human_clonotypes(df)


@pytest.fixture 
def human_clonotypes_small() -> List[CloneData]:
    """Small sample of human clonotypes for fast tests (50 clonotypes)."""
    df = load_human_tcr_database()
    df_clean = filter_valid_human_clonotypes(df)
    
    # Take seeded sample
    rng = np.random.default_rng(TEST_RANDOM_SEED)
    n_sample = min(SMALL_SAMPLE_SIZE, len(df_clean))
    sample_df = df_clean.sample(n=n_sample, random_state=TEST_RANDOM_SEED)
    
    clonotypes = []
    for _, row in sample_df.iterrows():
        clonotype = CloneData(
            va=row['va'],
            ja=row.get('ja', row['va']),  # Use va as fallback for ja if missing
            cdr3a=row['cdr3a'],
            vb=row['vb'],
            jb=row.get('jb', row['vb']),  # Use vb as fallback for jb if missing  
            cdr3b=row['cdr3b']
        )
        clonotypes.append(clonotype)
        
    return clonotypes


@pytest.fixture
def human_clonotypes_medium() -> List[CloneData]:
    """Medium sample of human clonotypes for regular tests (300 clonotypes)."""
    df = load_human_tcr_database()
    df_clean = filter_valid_human_clonotypes(df)
    
    # Take seeded sample
    rng = np.random.default_rng(TEST_RANDOM_SEED)
    n_sample = min(DEFAULT_SAMPLE_SIZE, len(df_clean))
    sample_df = df_clean.sample(n=n_sample, random_state=TEST_RANDOM_SEED)
    
    clonotypes = []
    for _, row in sample_df.iterrows():
        clonotype = CloneData(
            va=row['va'],
            ja=row.get('ja', row['va']),  
            cdr3a=row['cdr3a'],
            vb=row['vb'],
            jb=row.get('jb', row['vb']),   
            cdr3b=row['cdr3b']
        )
        clonotypes.append(clonotype)
        
    return clonotypes


@pytest.fixture
def human_clonotypes_large() -> List[CloneData]:
    """Large sample of human clonotypes for accuracy tests (1000 clonotypes)."""
    df = load_human_tcr_database()
    df_clean = filter_valid_human_clonotypes(df)
    
    # Take seeded sample
    n_sample = min(LARGE_SAMPLE_SIZE, len(df_clean))
    sample_df = df_clean.sample(n=n_sample, random_state=TEST_RANDOM_SEED)
    
    clonotypes = []
    for _, row in sample_df.iterrows():
        clonotype = CloneData(
            va=row['va'],
            ja=row.get('ja', row['va']),
            cdr3a=row['cdr3a'],
            vb=row['vb'],
            jb=row.get('jb', row['vb']),
            cdr3b=row['cdr3b']
        )
        clonotypes.append(clonotype)
        
    return clonotypes


@pytest.fixture
def mouse_clonotypes_small() -> List[CloneData]:
    """Synthetic mouse clonotypes for fast tests (50 clonotypes)."""
    generator = CloneFixtureGenerator(TEST_RANDOM_SEED)
    return generator.generate_clonotypes('mouse', SMALL_SAMPLE_SIZE)


@pytest.fixture  
def mouse_clonotypes_medium() -> List[CloneData]:
    """Synthetic mouse clonotypes for regular tests (300 clonotypes)."""
    generator = CloneFixtureGenerator(TEST_RANDOM_SEED)
    return generator.generate_clonotypes('mouse', DEFAULT_SAMPLE_SIZE)


@pytest.fixture
def mouse_clonotypes_large() -> List[CloneData]:
    """Synthetic mouse clonotypes for accuracy tests (1000 clonotypes).""" 
    generator = CloneFixtureGenerator(TEST_RANDOM_SEED)
    return generator.generate_clonotypes('mouse', LARGE_SAMPLE_SIZE)


@pytest.fixture
def rhesus_clonotypes_small() -> List[CloneData]:
    """Synthetic rhesus clonotypes for fast tests (50 clonotypes)."""
    generator = CloneFixtureGenerator(TEST_RANDOM_SEED)
    return generator.generate_clonotypes('rhesus', SMALL_SAMPLE_SIZE)


@pytest.fixture
def rhesus_clonotypes_medium() -> List[CloneData]:
    """Synthetic rhesus clonotypes for regular tests (300 clonotypes)."""
    generator = CloneFixtureGenerator(TEST_RANDOM_SEED)  
    return generator.generate_clonotypes('rhesus', DEFAULT_SAMPLE_SIZE)


@pytest.fixture
def rhesus_clonotypes_large() -> List[CloneData]:
    """Synthetic rhesus clonotypes for accuracy tests (1000 clonotypes)."""
    generator = CloneFixtureGenerator(TEST_RANDOM_SEED)
    return generator.generate_clonotypes('rhesus', LARGE_SAMPLE_SIZE)


@pytest.fixture(params=['human', 'mouse', 'rhesus'])
def supported_organism(request) -> str:
    """Parametrized fixture for all supported organisms."""
    return request.param


@pytest.fixture
def edge_case_clonotypes() -> List[CloneData]:
    """
    Edge case clonotypes for testing boundary conditions.
    
    Includes:
    - Minimum length CDR3s 
    - Maximum reasonable length CDR3s
    - Various V gene combinations
    """
    generator = CloneFixtureGenerator(TEST_RANDOM_SEED)
    
    # Generate edge cases manually
    edge_cases = []
    
    # Get some real gene names for human
    try:
        va_genes = generator._load_genes_for_organism_chain('human', 'A')[:5]
        vb_genes = generator._load_genes_for_organism_chain('human', 'B')[:5]
    except Exception:
        # Fallback to synthetic genes
        va_genes = ['TRAV1*01', 'TRAV2*01', 'TRAV3*01']
        vb_genes = ['TRBV1*01', 'TRBV2*01', 'TRBV3*01']
    
    ja_genes = va_genes
    jb_genes = vb_genes
    
    # Edge case 1: Minimum length CDR3s (6 characters)
    edge_cases.append(CloneData(
        va=va_genes[0], ja=ja_genes[0], cdr3a='CAVNVF', 
        vb=vb_genes[0], jb=jb_genes[0], cdr3b='CSARNF'
    ))
    
    # Edge case 2: Longer CDR3s (18 characters) 
    edge_cases.append(CloneData(
        va=va_genes[1], ja=ja_genes[1], cdr3a='CAVNVGGGKLGSYKLF',
        vb=vb_genes[1], jb=jb_genes[1], cdr3b='CSARSETGLGTGELFF'
    ))
    
    # Edge case 3: Very long CDR3s (24 characters)
    edge_cases.append(CloneData(
        va=va_genes[2], ja=ja_genes[2], cdr3a='CAVNVGGGKLGSYKLGSYQLF',
        vb=vb_genes[2], jb=jb_genes[2], cdr3b='CSARSETGLGTGELGVSYQFF'
    ))
    
    # Edge case 4: CDR3s that will need trimming/gapping
    edge_cases.append(CloneData( 
        va=va_genes[0], ja=ja_genes[0], cdr3a='CAVF',  # Very short
        vb=vb_genes[0], jb=jb_genes[0], cdr3b='CSARF'
    ))
    
    return edge_cases


@pytest.fixture
def encoding_configs() -> List[Dict[str, Any]]:
    """Various encoding configurations for testing."""
    return [
        # Default config
        {
            'aa_mds_dim': 16,
            'num_pos_cdr3': 16, 
            'cdr3_weight': 3.0,
            'n_trim': 3,
            'c_trim': 2,
            'random_seed': 42
        },
        # Smaller dimension config
        {
            'aa_mds_dim': 8,
            'num_pos_cdr3': 12,
            'cdr3_weight': 3.0, 
            'n_trim': 2,
            'c_trim': 1,
            'random_seed': 42
        },
        # Different weight config
        {
            'aa_mds_dim': 16,
            'num_pos_cdr3': 16,
            'cdr3_weight': 2.0,
            'n_trim': 3,
            'c_trim': 2, 
            'random_seed': 42
        }
    ]


@pytest.fixture
def mixed_format_data(human_clonotypes_small) -> Dict[str, Any]:
    """Same clonotype data in different input formats for format flexibility testing."""
    clonotypes = human_clonotypes_small[:10]  # Small subset for format testing
    
    # Format 1: Nested tuples (as returned by retrieve_tcrs_from_adata)
    tuples_format = [clone.as_tuple() for clone in clonotypes]
    
    # Format 2: DataFrame with default column names
    df_default = pd.DataFrame([clone.as_dict() for clone in clonotypes])
    
    # Format 3: DataFrame with custom column names
    df_custom = df_default.rename(columns={
        'va': 'va_gene',
        'vb': 'vb_gene', 
        'cdr3a': 'cdr3a_seq',
        'cdr3b': 'cdr3b_seq'
    })
    
    return {
        'tuples': tuples_format,
        'dataframe_default': df_default,
        'dataframe_custom': df_custom,
        'custom_columns': {
            'va_column': 'va_gene',
            'vb_column': 'vb_gene',
            'cdr3a_column': 'cdr3a_seq', 
            'cdr3b_column': 'cdr3b_seq'
        }
    }


# Pytest configuration helpers

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "vectorized: marks tests as part of the vectorized TCRdist feature"
    )
    config.addinivalue_line(
        "markers", "accuracy: marks tests that validate accuracy thresholds"  
    )
    config.addinivalue_line(
        "markers", "property: marks property-based tests"
    )


def pytest_collection_modifyitems(config, items):
    """Add markers to tests based on filename patterns."""
    for item in items:
        # Mark all tests in test_vectorized_*.py files
        if "vectorized" in item.fspath.basename:
            item.add_marker(pytest.mark.vectorized)
            
        # Mark accuracy tests
        if "accuracy" in item.name.lower() or "accuracy" in item.fspath.basename:
            item.add_marker(pytest.mark.accuracy)
            item.add_marker(pytest.mark.slow)
            
        # Mark property tests  
        if item.name.startswith("test_property_") or "property" in item.name.lower():
            item.add_marker(pytest.mark.property)