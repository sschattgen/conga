import numpy as np
import sys
from os import system
import os.path
from pathlib import Path
import os
from scipy.sparse import issparse
from scipy.stats import mannwhitneyu
from collections import Counter, OrderedDict
import subprocess
from . import tags

# try not to have any conga imports here
#

# convenience paths
path_to_conga = Path(__file__).parent

path_to_data = Path.joinpath( path_to_conga, 'data')

# tcrdist_cpp paths - pure Path arithmetic, existence checked at runtime by tcrdist_cpp_available()
path_to_tcrdist_cpp = Path.joinpath( path_to_conga.parents[0] ,'tcrdist_cpp')
path_to_tcrdist_cpp_bin = Path.joinpath( path_to_tcrdist_cpp ,'bin')
path_to_tcrdist_cpp_db = Path.joinpath( path_to_tcrdist_cpp ,'db')


def tcrdist_cpp_available():
    if os.name == 'posix':
        return os.path.exists(Path.joinpath( path_to_tcrdist_cpp_bin ,'find_neighbors'))
    else:
        return os.path.exists(Path.joinpath( path_to_tcrdist_cpp_bin ,'find_neighbors.exe'))

# this is the (OPTIONAL) obs key used to store a subject-specific identifier
# right now this is only used to prevent condensing of clonotypes that span subjects,
# which is pretty unlikely but could happen with certain populations (e.g., MAIT cells)
#
SUBJECT_ID_OBS_KEY = 'subject_id'

# not a big deal, but if we have protein ie antibody data we use these to mask it out
GENE_EXPRESSION_FEATURE_TYPE = 'Gene Expression'
ANTIBODY_CAPTURE_FEATURE_TYPE = 'Antibody Capture'

EXPECTED_FEATURE_TYPES = [GENE_EXPRESSION_FEATURE_TYPE, ANTIBODY_CAPTURE_FEATURE_TYPE]

FUNNY_MOUSE_TRBV_GENE = '5830405F06Rik' # actually seems to be a tcr v gene transcript or correlate with one
FUNNY_HUMAN_IG_GENES = [
    'AC233755.1', 'AC233755.2', # seem to be associated with one or more IGHV genes
    'CH17-224D4.2', # chr14 bac, suspiciously high correlation with tcr features??
    'IGLL5', # correlated with IGLJ1
]
FUNNY_HUMAN_TR_GENES = [
    'TRD-AS1', # overlaps TRA
    'ENSG00000251002', # overlaps TRA (alt name for TRD-AS1)
    'ENSG00000288882', # overlaps TRB
    'ENSG00000289938', # overlaps TRB
]

def run_command( cmd, verbose=False ):

    if verbose:
        print('util.run_command: cmd=', cmd)

    if os.name == 'posix':
        system(cmd)
    else:
        cmd_run = 'cmd /c' + cmd
        subprocess.check_call(list(cmd_run.split(' ')))


# different types of repertoire data we might have
TCR_AB_VDJ_TYPE = 'TCR_AB_VDJ_TYPE'
TCR_GD_VDJ_TYPE = 'TCR_GD_VDJ_TYPE'
IG_VDJ_TYPE = 'IG_VDJ_TYPE'

organism2vdj_type = {
    'human':TCR_AB_VDJ_TYPE,
    'mouse':TCR_AB_VDJ_TYPE,
    'human_gd':TCR_GD_VDJ_TYPE,
    'mouse_gd':TCR_GD_VDJ_TYPE,
    'human_ig':IG_VDJ_TYPE,
    'mouse_ig':IG_VDJ_TYPE,
    'rhesus':TCR_AB_VDJ_TYPE,
    'rhesus_gd':TCR_GD_VDJ_TYPE,
}

# Shared constants for vectorized TCRdist implementation
# These are read by all three CLI scripts for consistent defaults
DEFAULT_RANDOM_SEED: int = 42

KPCA_REDUCTION_LIMIT: int = 20000        # Observation count at or above which KernelPCA is not performed

# AnnData obsm keys for TCR representations  
OBSM_KEY_VEC_TCR: str = 'X_vec_tcr'      # Vectorized TCRdist representation
OBSM_KEY_PCA_TCR: str = 'X_pca_tcr'      # KernelPCA TCRdist representation
ACTIVE_REP_EXACT: str = 'exact_tcrdist'  # Sentinel for exact TCRdist path (no obsm key)

# AnnData uns keys for TCR representation metadata
UNS_KEY_ACTIVE_TCR_REP: str = 'active_tcr_representation'
UNS_KEY_BACKEND_CONFIG: str = 'faiss_backend_config'  # FAISS backend configuration


# Active TCR representation tracking functions
def get_active_tcr_representation(adata) -> str:
    """Get the currently active TCR representation from AnnData.
    
    Returns the key for the TCR representation that should be used for
    analysis. Checks for representations in priority order:
    1. Vectorized TCRdist (X_vec_tcr)
    2. KernelPCA TCRdist (X_pca_tcr) 
    3. Exact TCRdist (sentinel value)
    
    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object to check for TCR representations
        
    Returns
    -------
    str
        Key for active TCR representation:
        - OBSM_KEY_VEC_TCR if vectorized representation available
        - OBSM_KEY_PCA_TCR if PCA representation available
        - ACTIVE_REP_EXACT for exact TCRdist fallback
    """
    # Check stored preference first
    stored_rep = adata.uns.get(UNS_KEY_ACTIVE_TCR_REP)
    if stored_rep and _validate_tcr_representation(adata, stored_rep):
        return stored_rep
    
    # Auto-detect based on available representations (priority order)
    if OBSM_KEY_VEC_TCR in adata.obsm:
        return OBSM_KEY_VEC_TCR
    elif OBSM_KEY_PCA_TCR in adata.obsm:
        return OBSM_KEY_PCA_TCR
    else:
        return ACTIVE_REP_EXACT


def set_active_tcr_representation(adata, representation: str) -> None:
    """Set the active TCR representation in AnnData.
    
    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object to update
    representation : str
        TCR representation key to set as active
        
    Raises
    ------
    ValueError
        If representation is not valid or not available in adata
    """
    if not _validate_tcr_representation(adata, representation):
        available = _list_available_tcr_representations(adata)
        raise ValueError(
            f"TCR representation \"{representation}\" not available. "
            f"Available: {available}"
        )
    
    adata.uns[UNS_KEY_ACTIVE_TCR_REP] = representation


def _validate_tcr_representation(adata, representation: str) -> bool:
    """Validate that a TCR representation is available in adata."""
    if representation == ACTIVE_REP_EXACT:
        return True  # Always available as fallback
    elif representation in [OBSM_KEY_VEC_TCR, OBSM_KEY_PCA_TCR]:
        return representation in adata.obsm
    else:
        return False


def _list_available_tcr_representations(adata) -> list[str]:
    """List all available TCR representations in adata."""
    available = []
    if OBSM_KEY_VEC_TCR in adata.obsm:
        available.append(OBSM_KEY_VEC_TCR)
    if OBSM_KEY_PCA_TCR in adata.obsm:
        available.append(OBSM_KEY_PCA_TCR)
    available.append(ACTIVE_REP_EXACT)  # Always available
    return available


def is_vdj_gene( gene_upper, organism, include_constant_regions=False ):
    # for filtering out TR or IG gene names from GEX prior to processing
    # or for skipping such genes in the graph_vs_features analysis
    vdj_type = organism2vdj_type[organism]

    gene = gene_upper.lower()
    if vdj_type == TCR_AB_VDJ_TYPE:
        return ( gene.startswith('trav') or gene.startswith('trbv') or
                 gene.startswith('traj') or gene.startswith('trbj') or
                 gene.startswith('trbd') or gene_upper == FUNNY_MOUSE_TRBV_GENE or
                 gene_upper in FUNNY_HUMAN_TR_GENES or
                 ( include_constant_regions and (gene.startswith('trac') or gene.startswith('trbc'))))

    elif vdj_type == TCR_GD_VDJ_TYPE:
        return ( gene.startswith('trav') or gene.startswith('trdv') or
                 gene.startswith('traj') or gene.startswith('trdj') or
                 gene.startswith('trgv') or gene.startswith('trgj') or
                 gene.startswith('tcrg-') or gene.startswith('trdd') or
                 ( include_constant_regions and (gene.startswith('trdc') or gene.startswith('trgc'))))
    elif vdj_type == IG_VDJ_TYPE:
        return ( gene.startswith('ighv') or gene.startswith('iglv') or gene.startswith('igkv') or
                 gene.startswith('ighj') or gene.startswith('iglj') or gene.startswith('igkj') or
                 gene.startswith('ighd') or gene_upper in FUNNY_HUMAN_IG_GENES or
                 ( include_constant_regions and
                   ( gene.startswith('ighc') or gene.startswith('iglc') or gene.startswith('igkc'))))
    else:
        print('unrecognized vdj_type:', vdj_type)
        exit()
    return None

def make_clones_file( tcrs, outfilename, subject = 'UNK', epitope = 'UNK_E' ):
    ''' This may not have all the standard fields
    Right now just adding the fields we need in order for make_tcr_logo.py to work...
    '''
    gene_fields = ['{}{}_{}'.format(x,y,z) for x in 'vj' for y in 'ab' for z in ['gene', 'genes']]
    outfields = 'clone_id subject epitope cdr3a cdr3a_nucseq cdr3b cdr3b_nucseq'.split() + gene_fields

    out = open(outfilename, 'w')
    out.write('\t'.join(outfields)+'\n')
    for ii,(atcr, btcr) in enumerate(tcrs):
        outl = { 'clone_id': 'clone_{}'.format(ii+1),
                 'subject': subject,
                 'epitope': epitope,
                 'va_gene': atcr[0],
                 'va_genes': atcr[0],
                 'ja_gene': atcr[1],
                 'ja_genes': atcr[1],
                 'cdr3a': atcr[2],
                 'cdr3a_nucseq': atcr[3],
                 'vb_gene': btcr[0],
                 'vb_genes': btcr[0],
                 'jb_gene': btcr[1],
                 'jb_genes': btcr[1],
                 'cdr3b': btcr[2],
                 'cdr3b_nucseq': btcr[3],
                 }
        out.write('\t'.join( outl[x] for x in outfields)+'\n')
    out.close()


def get_feature_types_varname( adata ):
    ''' Figure out the correct "feature_types" varname, e.g. feature_types-0 or feature_types-0-0
    for sorting out which are gene expression features and which are protein/antibody-capture features
    '''
    for name in adata.var: # the columns of the var dataframe
        if name.startswith('feature_types'):
            ftypes = Counter(adata.var[name])
            print('get_feature_types_varname:', name, 'feature_type_counts:', ftypes.most_common())
            for fname, count in ftypes.items():
                if fname not in EXPECTED_FEATURE_TYPES:
                    print('WARNING WARNING WARNING !!!! unrecognized feature type', fname, 'with', count, 'features')
            return name
    print('unable to find feature_types varname')
    print(adata.var_names)
    return None




def setup_uns_dicts(adata):
    if 'conga_results' not in adata.uns.keys():
        adata.uns['conga_results'] = {}

    if 'conga_stats' not in adata.uns.keys():
        adata.uns['conga_stats'] = {}



def save_table_and_helpfile(
        table_tag,
        adata,
        outfile_prefix
):
    if ('conga_results' not in adata.uns or
        table_tag not in adata.uns['conga_results']):
        print('ERROR missing results for table_tag:', table_tag)
        return

    results = adata.uns['conga_results'][table_tag]
    tsvfile = f'{outfile_prefix}_{table_tag}.tsv'
    results.to_csv(tsvfile, sep='\t', index=False)
    print('saved', table_tag, 'results to tsvfile:', tsvfile)

    help_tag = table_tag + tags.HELP_SUFFIX
    if help_tag not in adata.uns['conga_results']:
        print('WARNING: no help for table', table_tag)
        return

    helpfile = tsvfile+'_README.txt'
    out = open(helpfile, 'w')
    out.write(adata.uns['conga_results'][help_tag])
    out.close()


def make_figure_helpfile(
        figure_tag,
        adata,
):
    pngfile = adata.uns['conga_results'][figure_tag]
    help_message = adata.uns['conga_results'].get(
        figure_tag+tags.HELP_SUFFIX, '')
    if help_message:
        helpfile = pngfile+'_README.txt'
        out = open(helpfile, 'w')
        print('writing help message to file:', helpfile)
        out.write(help_message+'\n')
        out.close()
    else:
        print('WARNING: no help message for figure_tag:', figure_tag)

## this is a silly hack: the new scipy.stats.mannwhitneyu can exceed max recursion
## depth (e.g. on a mac) when the 'exact' method is used (e.g. if one of the two
## arrays is very large)
##
mannwhitneyu_kwargs = {'method':'asymptotic'}
try:
    mannwhitneyu([1,2,3], [4,5,6], **mannwhitneyu_kwargs)
except:
    #print('detected older scipy.stats.mannwhitneyu version')
    mannwhitneyu_kwargs = {}
## END SILLY HACK ################


# CLI flag constants for backend selection
DEFAULT_BACKEND_SELECTION: str = 'auto'  # 'auto', 'faiss_gpu', 'faiss_cpu', 'sklearn'
DEFAULT_FAISS_ADAPTIVE: bool = True      # Enable adaptive FAISS parameter selection

# Backend selection logic function
def select_optimal_backend(n_samples: int, force_backend: str = None) -> str:
    """Select optimal backend based on data size and availability."""
    if force_backend and force_backend != 'auto':
        return force_backend
    
    # Auto-selection logic based on data size and hardware
    if n_samples >= 50000:
        return 'faiss_gpu' if _gpu_available() else 'faiss_cpu'
    elif n_samples >= 10000:
        return 'faiss_cpu' if _cpu_available() else 'sklearn'
    else:
        return 'sklearn'  # Small datasets - sklearn is fine

def _gpu_available() -> bool:
    try:
        import faiss
        return hasattr(faiss, 'StandardGpuResources')
    except ImportError:
        return False

def _cpu_available() -> bool:
    try:
        import faiss
        return True
    except ImportError:
        return False

# TCR Representation Selection System with FAISS Awareness
from enum import Enum
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class TcrRepresentation(Enum):
    """Supported TCR representation types."""
    VECTORIZED = 'vectorized'    # Vectorized TCRdist with FAISS acceleration 
    KERNELPCA = 'kernelpca'     # KernelPCA TCRdist (legacy)
    EXACT = 'exact'             # Exact TCRdist computation

class TcrRepresentationResolver:
    """Resolver for selecting optimal TCR representation based on data and hardware.
    
    This class implements the three-way TCR representation selection logic:
    1. Vectorized TCRdist (preferred for large datasets with FAISS)
    2. KernelPCA TCRdist (legacy representation, good for medium datasets)
    3. Exact TCRdist (fallback for small datasets or when representations unavailable)
    
    The resolver considers:
    - Dataset size (number of clonotypes)
    - Hardware availability (FAISS GPU/CPU)
    - Available representations in AnnData
    - Memory constraints
    - Performance requirements
    """
    
    # Thresholds for representation selection
    VECTORIZED_THRESHOLD: int = 1000      # Prefer vectorized above this size
    KERNELPCA_THRESHOLD: int = 20000      # KernelPCA becomes expensive above this
    EXACT_THRESHOLD: int = 5000           # Exact becomes impractical above this
    
    def __init__(self, 
                 force_representation: Optional[str] = None,
                 enable_faiss_acceleration: bool = True,
                 memory_limit_gb: Optional[float] = None):
        """Initialize the resolver with preferences.
        
        Parameters
        ----------
        force_representation : str, optional
            Force specific representation: 'vectorized', 'kernelpca', 'exact'
        enable_faiss_acceleration : bool, default=True
            Allow FAISS acceleration when available
        memory_limit_gb : float, optional
            Memory limit in GB for representation selection
        """
        self.force_representation = force_representation
        self.enable_faiss_acceleration = enable_faiss_acceleration
        self.memory_limit_gb = memory_limit_gb
        
    def resolve(self, adata, organism: str) -> Tuple[TcrRepresentation, Dict[str, Any]]:
        """Select optimal TCR representation for the given dataset.
        
        Parameters
        ----------
        adata : anndata.AnnData
            AnnData object with TCR data
        organism : str
            Organism for TCR analysis
            
        Returns
        -------
        tuple[TcrRepresentation, dict]
            Selected representation and configuration parameters
        """
        n_clonotypes = self._count_clonotypes(adata)
        available_reps = self._check_available_representations(adata)
        hardware_info = self._assess_hardware()
        
        # Force specific representation if requested
        if self.force_representation:
            forced_rep = TcrRepresentation(self.force_representation)
            if self._validate_representation_available(forced_rep, available_reps, n_clonotypes):
                config = self._get_representation_config(forced_rep, adata, organism, hardware_info)
                logger.info(f"Using forced TCR representation: {forced_rep.value}")
                return forced_rep, config
            else:
                logger.warning(f"Forced representation {self.force_representation} not available, falling back to auto-selection")
        
        # Auto-selection based on data characteristics
        selected_rep = self._auto_select_representation(
            n_clonotypes, available_reps, hardware_info
        )
        
        config = self._get_representation_config(selected_rep, adata, organism, hardware_info)
        
        logger.info(f"Selected TCR representation: {selected_rep.value} "
                   f"for {n_clonotypes} clonotypes")
        
        return selected_rep, config
    
    def _count_clonotypes(self, adata) -> int:
        """Count unique clonotypes in the dataset."""
        if 'clonotype_id' in adata.obs:
            return adata.obs['clonotype_id'].nunique()
        else:
            # Fallback: assume each cell is a clonotype
            return adata.n_obs
    
    def _check_available_representations(self, adata) -> Dict[TcrRepresentation, bool]:
        """Check which representations are already available in AnnData."""
        available = {}
        
        # Check vectorized representation
        available[TcrRepresentation.VECTORIZED] = (
            OBSM_KEY_VEC_TCR in adata.obsm and
            UNS_KEY_VEC_TCR_CONFIG in adata.uns
        )
        
        # Check KernelPCA representation  
        available[TcrRepresentation.KERNELPCA] = OBSM_KEY_PCA_TCR in adata.obsm
        
        # Exact is always "available" (computed on demand)
        available[TcrRepresentation.EXACT] = True
        
        return available
    
    def _assess_hardware(self) -> Dict[str, Any]:
        """Assess available hardware capabilities."""
        hardware_info = {
            'faiss_gpu_available': False,
            'faiss_cpu_available': False,
            'estimated_memory_gb': None
        }
        
        # Check FAISS GPU availability
        if self.enable_faiss_acceleration:
            try:
                import faiss
                if hasattr(faiss, 'StandardGpuResources'):
                    # Try to create GPU resources to verify GPU is actually available
                    try:
                        gpu_res = faiss.StandardGpuResources()
                        hardware_info['faiss_gpu_available'] = True
                        del gpu_res  # Clean up
                    except Exception:
                        pass  # GPU not available
                
                # FAISS CPU is available if we can import it
                hardware_info['faiss_cpu_available'] = True
            except ImportError:
                pass
        
        return hardware_info
    
    def _auto_select_representation(self, 
                                   n_clonotypes: int, 
                                   available_reps: Dict[TcrRepresentation, bool],
                                   hardware_info: Dict[str, Any]) -> TcrRepresentation:
        """Auto-select representation based on data size and hardware."""
        
        # Priority 1: Use existing vectorized representation if available
        if available_reps[TcrRepresentation.VECTORIZED]:
            return TcrRepresentation.VECTORIZED
        
        # Priority 2: For large datasets, prefer vectorized if we can compute it
        if n_clonotypes >= self.VECTORIZED_THRESHOLD:
            if hardware_info['faiss_gpu_available'] or hardware_info['faiss_cpu_available']:
                return TcrRepresentation.VECTORIZED
        
        # Priority 3: Use existing KernelPCA if available and dataset not too large
        if (available_reps[TcrRepresentation.KERNELPCA] and 
            n_clonotypes < self.KERNELPCA_THRESHOLD):
            return TcrRepresentation.KERNELPCA
        
        # Priority 4: For medium datasets, compute vectorized if FAISS available
        if (n_clonotypes >= self.VECTORIZED_THRESHOLD and 
            n_clonotypes < self.EXACT_THRESHOLD and
            (hardware_info['faiss_gpu_available'] or hardware_info['faiss_cpu_available'])):
            return TcrRepresentation.VECTORIZED
        
        # Priority 5: For small-medium datasets, use KernelPCA if memory allows
        if n_clonotypes < self.KERNELPCA_THRESHOLD:
            return TcrRepresentation.KERNELPCA
        
        # Fallback: Exact computation (always works, but slow for large datasets)
        return TcrRepresentation.EXACT
    
    def _validate_representation_available(self, 
                                         representation: TcrRepresentation,
                                         available_reps: Dict[TcrRepresentation, bool],
                                         n_clonotypes: int) -> bool:
        """Validate that a representation can be used."""
        if representation == TcrRepresentation.EXACT:
            return n_clonotypes < self.EXACT_THRESHOLD  # Practical limit
        elif representation == TcrRepresentation.KERNELPCA:
            return n_clonotypes < self.KERNELPCA_THRESHOLD  # Memory limit
        elif representation == TcrRepresentation.VECTORIZED:
            return True  # Can always be computed if dependencies available
        else:
            return False
    
    def _get_representation_config(self, 
                                 representation: TcrRepresentation,
                                 adata,
                                 organism: str,
                                 hardware_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get configuration for the selected representation."""
        config = {
            'representation': representation.value,
            'organism': organism,
            'n_clonotypes': self._count_clonotypes(adata)
        }
        
        if representation == TcrRepresentation.VECTORIZED:
            config.update({
                'enable_faiss': self.enable_faiss_acceleration,
                'faiss_gpu_available': hardware_info['faiss_gpu_available'],
                'faiss_cpu_available': hardware_info['faiss_cpu_available'],
                'backend_preference': self._select_faiss_backend(hardware_info, config['n_clonotypes'])
            })
        elif representation == TcrRepresentation.KERNELPCA:
            config.update({
                'n_components': min(50, config['n_clonotypes'] // 2),  # Standard KernelPCA config
                'kernel': 'precomputed'
            })
        elif representation == TcrRepresentation.EXACT:
            config.update({
                'distance_metric': 'tcrdist',
                'compute_full_matrix': config['n_clonotypes'] < 1000  # Memory consideration
            })
        
        return config
    
    def _select_faiss_backend(self, hardware_info: Dict[str, Any], n_clonotypes: int) -> str:
        """Select optimal FAISS backend based on hardware and data size."""
        if not self.enable_faiss_acceleration:
            return 'sklearn'
        
        # For very large datasets, prefer GPU if available
        if n_clonotypes >= 50000 and hardware_info['faiss_gpu_available']:
            return 'faiss_gpu'
        
        # For medium-large datasets, prefer CPU FAISS
        if n_clonotypes >= 10000 and hardware_info['faiss_cpu_available']:
            return 'faiss_cpu'
        
        # For smaller datasets, sklearn is fine
        return 'sklearn'


# Global resolver instance (can be overridden by applications)
_DEFAULT_RESOLVER = TcrRepresentationResolver()

def resolve_tcr_representation(adata, organism: str, 
                              resolver: Optional[TcrRepresentationResolver] = None) -> Tuple[TcrRepresentation, Dict[str, Any]]:
    """Convenience function to resolve TCR representation using default or custom resolver."""
    if resolver is None:
        resolver = _DEFAULT_RESOLVER
    
    return resolver.resolve(adata, organism)

# Restart Logic for Stored Representations
def detect_and_validate_stored_representations(adata, organism: str) -> Dict[str, Dict[str, Any]]:
    """Detect and validate all stored TCR representations in AnnData.
    
    This function scans the AnnData object for existing TCR representations
    and validates their integrity for reuse in analysis restart scenarios.
    
    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object to scan for representations
    organism : str
        Organism string for validation
        
    Returns
    -------
    dict
        Dictionary mapping representation names to validation results:
        {
            'vectorized': {'valid': bool, 'config': dict, 'issues': list},
            'kernelpca': {'valid': bool, 'config': dict, 'issues': list},
            'exact': {'valid': bool, 'config': dict, 'issues': list}
        }
    """
    results = {}
    
    # Check vectorized representation
    results['vectorized'] = _validate_vectorized_representation(adata, organism)
    
    # Check KernelPCA representation
    results['kernelpca'] = _validate_kernelpca_representation(adata, organism)
    
    # Exact representation is always "valid" (computed on demand)
    results['exact'] = {'valid': True, 'config': {}, 'issues': []}
    
    return results

def _validate_vectorized_representation(adata, organism: str) -> Dict[str, Any]:
    """Validate vectorized TCR representation for restart compatibility."""
    validation_result = {
        'valid': False,
        'config': {},
        'issues': []
    }
    
    # Check if vectorized matrix exists
    if OBSM_KEY_VEC_TCR not in adata.obsm:
        validation_result['issues'].append("Vectorized TCR matrix not found in adata.obsm")
        return validation_result
    
    # Check if configuration exists
    if UNS_KEY_VEC_TCR_CONFIG not in adata.uns:
        validation_result['issues'].append("Vectorized TCR configuration not found in adata.uns")
        return validation_result
    
    # Load and validate configuration
    try:
        from ..tcrdist.vectorized import EncodingConfig
        config_dict = adata.uns[UNS_KEY_VEC_TCR_CONFIG]
        config = EncodingConfig.from_uns_dict(config_dict)
        validation_result['config'] = config_dict
    except Exception as e:
        validation_result['issues'].append(f"Invalid vectorized TCR configuration: {e}")
        return validation_result
    
    # Check matrix dimensions
    vector_matrix = adata.obsm[OBSM_KEY_VEC_TCR]
    expected_rows = adata.n_obs
    
    if vector_matrix.shape[0] != expected_rows:
        validation_result['issues'].append(
            f"Vector matrix row count mismatch: {vector_matrix.shape[0]} != {expected_rows}"
        )
        return validation_result
    
    # Check expected vector length
    try:
        from ..tcrdist.vectorized import vector_length
        expected_length = vector_length(organism, config)
        if vector_matrix.shape[1] != expected_length:
            validation_result['issues'].append(
                f"Vector length mismatch: {vector_matrix.shape[1]} != {expected_length}"
            )
            return validation_result
    except Exception as e:
        validation_result['issues'].append(f"Could not validate vector length: {e}")
        return validation_result
    
    # All checks passed
    validation_result['valid'] = True
    return validation_result

def _validate_kernelpca_representation(adata, organism: str) -> Dict[str, Any]:
    """Validate KernelPCA TCR representation for restart compatibility."""
    validation_result = {
        'valid': False,
        'config': {},
        'issues': []
    }
    
    # Check if KernelPCA matrix exists
    if OBSM_KEY_PCA_TCR not in adata.obsm:
        validation_result['issues'].append("KernelPCA TCR matrix not found in adata.obsm")
        return validation_result
    
    # Basic validation of KernelPCA matrix
    pca_matrix = adata.obsm[OBSM_KEY_PCA_TCR]
    expected_rows = adata.n_obs
    
    if pca_matrix.shape[0] != expected_rows:
        validation_result['issues'].append(
            f"KernelPCA matrix row count mismatch: {pca_matrix.shape[0]} != {expected_rows}"
        )
        return validation_result
    
    # Check for reasonable number of components
    n_components = pca_matrix.shape[1]
    if n_components < 2 or n_components > expected_rows:
        validation_result['issues'].append(
            f"KernelPCA component count seems invalid: {n_components}"
        )
        return validation_result
    
    # All checks passed
    validation_result['valid'] = True
    validation_result['config'] = {'n_components': n_components}
    return validation_result

def create_restart_plan(adata, organism: str, 
                       target_representation: Optional[str] = None) -> Dict[str, Any]:
    """Create a plan for restarting analysis with optimal representation reuse.
    
    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object with potential stored representations
    organism : str
        Organism string
    target_representation : str, optional
        Specific target representation, or None for auto-selection
        
    Returns
    -------
    dict
        Restart plan with keys:
        - 'action': 'reuse', 'compute', or 'hybrid'
        - 'representation': selected representation
        - 'reusable_components': list of components that can be reused
        - 'computation_needed': list of components that need computation
        - 'estimated_time_savings': float (0-1, fraction of time saved)
    """
    stored_reps = detect_and_validate_stored_representations(adata, organism)
    resolver = TcrRepresentationResolver(force_representation=target_representation)
    optimal_rep, config = resolver.resolve(adata, organism)
    
    plan = {
        'action': 'compute',
        'representation': optimal_rep.value,
        'reusable_components': [],
        'computation_needed': ['full_representation'],
        'estimated_time_savings': 0.0
    }
    
    # Check if we can reuse the optimal representation directly
    rep_name = optimal_rep.value
    if rep_name in stored_reps and stored_reps[rep_name]['valid']:
        plan.update({
            'action': 'reuse',
            'reusable_components': [rep_name],
            'computation_needed': [],
            'estimated_time_savings': 1.0
        })
        return plan
    
    # Check for partial reuse opportunities
    if optimal_rep == TcrRepresentation.VECTORIZED:
        # Can we reuse KernelPCA for subset analysis?
        if stored_reps['kernelpca']['valid']:
            plan.update({
                'action': 'hybrid',
                'reusable_components': ['kernelpca_for_subset'],
                'computation_needed': ['vectorized_representation'],
                'estimated_time_savings': 0.2  # Some TCRdist computation can be skipped
            })
    
    elif optimal_rep == TcrRepresentation.KERNELPCA:
        # Can we reuse vectorized representation?
        if stored_reps['vectorized']['valid']:
            plan.update({
                'action': 'reuse',
                'reusable_components': ['vectorized_as_kernelpca'],
                'computation_needed': [],
                'estimated_time_savings': 1.0
            })
    
    return plan

def execute_restart_plan(adata, organism: str, plan: Dict[str, Any]) -> None:
    """Execute the restart plan to prepare representations.
    
    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object to update
    organism : str
        Organism string
    plan : dict
        Restart plan from create_restart_plan()
    """
    logger.info(f"Executing restart plan: {plan['action']} for {plan['representation']}")
    
    if plan['action'] == 'reuse':
        # Set active representation to the reusable one
        if plan['representation'] == 'vectorized':
            set_active_tcr_representation(adata, OBSM_KEY_VEC_TCR)
        elif plan['representation'] == 'kernelpca':
            set_active_tcr_representation(adata, OBSM_KEY_PCA_TCR)
        elif plan['representation'] == 'exact':
            set_active_tcr_representation(adata, ACTIVE_REP_EXACT)
        
        logger.info(f"Reusing stored {plan['representation']} representation")
    
    elif plan['action'] == 'compute':
        # Full computation needed - clear conflicting stored preferences
        if UNS_KEY_ACTIVE_TCR_REP in adata.uns:
            del adata.uns[UNS_KEY_ACTIVE_TCR_REP]  # Will auto-detect
        
        logger.info(f"Full computation needed for {plan['representation']} representation")
    
    elif plan['action'] == 'hybrid':
        # Partial reuse with additional computation
        logger.info(f"Hybrid approach: reusing {plan['reusable_components']} "
                   f"and computing {plan['computation_needed']}")
    
    # Log time savings estimate
    if plan['estimated_time_savings'] > 0:
        savings_pct = plan['estimated_time_savings'] * 100
        logger.info(f"Estimated time savings: {savings_pct:.1f}%")
