"""Backward-compatibility shim. Use sls_asvspoof.rawboost instead."""
import warnings
warnings.warn(
    "Importing from 'RawBoost' is deprecated. Use 'from sls_asvspoof.rawboost import ...' instead.",
    DeprecationWarning, stacklevel=2
)
from sls_asvspoof.rawboost import (
    randRange, normWav, genNotchCoeffs, filterFIR,
    LnL_convolutive_noise, ISD_additive_noise, SSI_additive_noise
)
