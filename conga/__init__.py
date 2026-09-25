from . import preprocess
from . import correlations
from . import plotting
from . import util
from . import tcr_scoring
from . import pmhc_scoring
from . import imhc_scoring
from . import cd8_scoring
from . import tcrdist
from . import tcr_clumping
from . import devel # where development / possibly legacy / unused code goes
from . import tags
from . import compatibility
from . import neighbors  # FAISS-accelerated neighbor search
from . import benchmark  # Performance benchmarking infrastructure

# Expose key compatibility functions at package level
from .compatibility import (
    check_environment_compatibility,
    safe_obs_columns,
    safe_var_columns, 
    safe_uns_keys,
    safe_obsm_keys,
    safe_is_view
)






