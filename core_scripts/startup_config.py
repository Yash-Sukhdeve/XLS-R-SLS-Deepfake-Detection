"""Backward-compatibility shim. Use sls_asvspoof.utils.seed instead."""
import warnings
warnings.warn(
    "Importing from 'core_scripts.startup_config' is deprecated. "
    "Use 'from sls_asvspoof.utils import set_random_seed' instead.",
    DeprecationWarning, stacklevel=2
)
from sls_asvspoof.utils.seed import set_random_seed
