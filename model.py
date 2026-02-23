"""Backward-compatibility shim. Use sls_asvspoof.model instead."""
import warnings
warnings.warn(
    "Importing from 'model' is deprecated. Use 'from sls_asvspoof.model import Model' instead.",
    DeprecationWarning, stacklevel=2
)
from sls_asvspoof.model import SSLModel, getAttenF, Model
