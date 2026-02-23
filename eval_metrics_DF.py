"""Backward-compatibility shim. Use sls_asvspoof.metrics instead."""
import warnings
warnings.warn(
    "Importing from 'eval_metrics_DF' is deprecated. Use 'from sls_asvspoof.metrics import ...' instead.",
    DeprecationWarning, stacklevel=2
)
from sls_asvspoof.metrics.eer import obtain_asv_error_rates, compute_det_curve, compute_eer
from sls_asvspoof.metrics.tdcf import compute_tDCF, compute_tDCF_legacy
