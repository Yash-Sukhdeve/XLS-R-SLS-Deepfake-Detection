"""EER and ASV error rate computation.

Functions for computing Detection Error Tradeoff (DET) curves,
Equal Error Rate (EER), and ASV error rates. Used across all
evaluation tracks (DF, LA, In-the-Wild).
"""

import numpy as np


def obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold):
    """Compute false alarm and miss rates for ASV system.

    Args:
        tar_asv: Target speaker ASV scores.
        non_asv: Non-target speaker ASV scores.
        spoof_asv: Spoof ASV scores.
        asv_threshold: ASV decision threshold.

    Returns:
        Tuple of (Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv).
    """
    Pfa_asv = sum(non_asv >= asv_threshold) / non_asv.size
    Pmiss_asv = sum(tar_asv < asv_threshold) / tar_asv.size

    if spoof_asv.size == 0:
        Pmiss_spoof_asv = None
        Pfa_spoof_asv = None
    else:
        Pmiss_spoof_asv = np.sum(spoof_asv < asv_threshold) / spoof_asv.size
        Pfa_spoof_asv = np.sum(spoof_asv >= asv_threshold) / spoof_asv.size

    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv


def compute_det_curve(target_scores, nontarget_scores):
    """Compute Detection Error Tradeoff (DET) curve.

    Args:
        target_scores: Scores for target (bonafide) trials.
        nontarget_scores: Scores for non-target (spoof) trials.

    Returns:
        Tuple of (frr, far, thresholds).
    """
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size))
    )

    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (
        np.arange(1, n_scores + 1) - tar_trial_sums
    )

    frr = np.concatenate(
        (np.atleast_1d(0), tar_trial_sums / target_scores.size)
    )
    far = np.concatenate(
        (np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size)
    )
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices])
    )

    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """Compute Equal Error Rate (EER) and corresponding threshold.

    Args:
        target_scores: Scores for target (bonafide) trials.
        nontarget_scores: Scores for non-target (spoof) trials.

    Returns:
        Tuple of (eer, threshold).
    """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]
