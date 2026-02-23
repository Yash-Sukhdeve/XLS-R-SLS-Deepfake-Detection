"""Evaluation metrics for ASVspoof and In-the-Wild datasets."""
from .eer import compute_det_curve, compute_eer, obtain_asv_error_rates
from .tdcf import compute_tDCF, compute_tDCF_legacy
