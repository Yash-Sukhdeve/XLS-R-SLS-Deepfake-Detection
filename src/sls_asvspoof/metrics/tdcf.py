"""Tandem Detection Cost Function (t-DCF) computation.

Implements t-DCF for cascaded CM + ASV systems, following:
  [1] Kinnunen et al., "Tandem Assessment of Spoofing Countermeasures
      and Automatic Speaker Verification: Fundamentals," IEEE/ACM TASLP.
  [2] ASVspoof 2019 challenge evaluation plan.
"""

import sys

import numpy as np

from .eer import compute_det_curve


def compute_tDCF(bonafide_score_cm, spoof_score_cm, Pfa_asv, Pmiss_asv,
                 Pfa_spoof_asv, cost_model, print_cost):
    """Compute normalized t-DCF (ASVspoof 2019+ formulation).

    Args:
        bonafide_score_cm: Positive class (bonafide) CM scores.
        spoof_score_cm: Negative class (spoof) CM scores.
        Pfa_asv: ASV false alarm rate (fraction).
        Pmiss_asv: ASV miss rate (fraction).
        Pfa_spoof_asv: ASV false alarm rate for spoof trials (fraction).
        cost_model: Dict with keys Ptar, Pnon, Pspoof, Cmiss, Cfa, Cfa_spoof.
        print_cost: Whether to print cost summary.

    Returns:
        Tuple of (tDCF_norm, CM_thresholds).
    """
    if cost_model['Cfa'] < 0 or cost_model['Cmiss'] < 0 or \
            cost_model['Cfa'] < 0 or cost_model['Cmiss'] < 0:
        print('WARNING: Usually the cost values should be positive!')

    if cost_model['Ptar'] < 0 or cost_model['Pnon'] < 0 or cost_model['Pspoof'] < 0 or \
            np.abs(cost_model['Ptar'] + cost_model['Pnon'] + cost_model['Pspoof'] - 1) > 1e-10:
        sys.exit('ERROR: Your prior probabilities should be positive and sum up to one.')

    if Pfa_spoof_asv is None:
        sys.exit('ERROR: you should provide false alarm rate of spoof tests against your ASV system.')

    combined_scores = np.concatenate((bonafide_score_cm, spoof_score_cm))
    if np.isnan(combined_scores).any() or np.isinf(combined_scores).any():
        sys.exit('ERROR: Your scores contain nan or inf.')

    n_uniq = np.unique(combined_scores).size
    if n_uniq < 3:
        sys.exit('ERROR: You should provide soft CM scores - not binary decisions')

    Pmiss_cm, Pfa_cm, CM_thresholds = compute_det_curve(bonafide_score_cm, spoof_score_cm)

    C0 = cost_model['Ptar'] * cost_model['Cmiss'] * Pmiss_asv + cost_model['Pnon'] * cost_model['Cfa'] * Pfa_asv
    C1 = cost_model['Ptar'] * cost_model['Cmiss'] - (cost_model['Ptar'] * cost_model['Cmiss'] * Pmiss_asv + cost_model['Pnon'] * cost_model['Cfa'] * Pfa_asv)
    C2 = cost_model['Pspoof'] * cost_model['Cfa_spoof'] * Pfa_spoof_asv

    if C0 < 0 or C1 < 0 or C2 < 0:
        sys.exit('You should never see this error but I cannot evalute tDCF with negative weights - please check whether your ASV error rates are correctly computed?')

    tDCF = C0 + C1 * Pmiss_cm + C2 * Pfa_cm
    tDCF_default = C0 + np.minimum(C1, C2)
    tDCF_norm = tDCF / tDCF_default

    if print_cost:
        print('t-DCF evaluation from [Nbona={}, Nspoof={}] trials\n'.format(bonafide_score_cm.size, spoof_score_cm.size))
        print('t-DCF MODEL')
        print('   Ptar         = {:8.5f} (Prior probability of target user)'.format(cost_model['Ptar']))
        print('   Pnon         = {:8.5f} (Prior probability of nontarget user)'.format(cost_model['Pnon']))
        print('   Pspoof       = {:8.5f} (Prior probability of spoofing attack)'.format(cost_model['Pspoof']))
        print('   Cfa          = {:8.5f} (Cost of tandem system falsely accepting a nontarget)'.format(cost_model['Cfa']))
        print('   Cmiss        = {:8.5f} (Cost of tandem system falsely rejecting target speaker)'.format(cost_model['Cmiss']))
        print('   Cfa_spoof    = {:8.5f} (Cost of tandem sysmte falsely accepting spoof)'.format(cost_model['Cfa_spoof']))
        print('\n   Implied normalized t-DCF function (depends on t-DCF parameters and ASV errors), t_CM=CM threshold)')
        print('   tDCF_norm(t_CM) = {:8.5f} + {:8.5f} x Pmiss_cm(t_CM) + {:8.5f} x Pfa_cm(t_CM)\n'.format(C0/tDCF_default, C1/tDCF_default, C2/tDCF_default))
        print('     * The optimum value is given by the first term (0.06273). This is the normalized t-DCF obtained with an error-free CM system.')
        print('     * The minimum normalized cost (minimum over all possible thresholds) is always <= 1.00.')
        print('')

    return tDCF_norm, CM_thresholds


def compute_tDCF_legacy(bonafide_score_cm, spoof_score_cm, Pfa_asv, Pmiss_asv,
                        Pmiss_spoof_asv, cost_model, print_cost):
    """Compute normalized t-DCF (legacy Odyssey 2018 formulation).

    Args:
        bonafide_score_cm: Positive class (bonafide) CM scores.
        spoof_score_cm: Negative class (spoof) CM scores.
        Pfa_asv: ASV false alarm rate (fraction).
        Pmiss_asv: ASV miss rate (fraction).
        Pmiss_spoof_asv: ASV miss rate for spoof trials (fraction).
        cost_model: Dict with keys Ptar, Pnon, Pspoof, Cmiss_asv, Cfa_asv,
                    Cmiss_cm, Cfa_cm.
        print_cost: Whether to print cost summary.

    Returns:
        Tuple of (tDCF_norm, CM_thresholds).
    """
    if cost_model['Cfa_asv'] < 0 or cost_model['Cmiss_asv'] < 0 or \
            cost_model['Cfa_cm'] < 0 or cost_model['Cmiss_cm'] < 0:
        print('WARNING: Usually the cost values should be positive!')

    if cost_model['Ptar'] < 0 or cost_model['Pnon'] < 0 or cost_model['Pspoof'] < 0 or \
            np.abs(cost_model['Ptar'] + cost_model['Pnon'] + cost_model['Pspoof'] - 1) > 1e-10:
        sys.exit('ERROR: Your prior probabilities should be positive and sum up to one.')

    if Pmiss_spoof_asv is None:
        sys.exit('ERROR: you should provide miss rate of spoof tests against your ASV system.')

    combined_scores = np.concatenate((bonafide_score_cm, spoof_score_cm))
    if np.isnan(combined_scores).any() or np.isinf(combined_scores).any():
        sys.exit('ERROR: Your scores contain nan or inf.')

    n_uniq = np.unique(combined_scores).size
    if n_uniq < 3:
        sys.exit('ERROR: You should provide soft CM scores - not binary decisions')

    Pmiss_cm, Pfa_cm, CM_thresholds = compute_det_curve(bonafide_score_cm, spoof_score_cm)

    C1 = cost_model['Ptar'] * (cost_model['Cmiss_cm'] - cost_model['Cmiss_asv'] * Pmiss_asv) - \
         cost_model['Pnon'] * cost_model['Cfa_asv'] * Pfa_asv
    C2 = cost_model['Cfa_cm'] * cost_model['Pspoof'] * (1 - Pmiss_spoof_asv)

    if C1 < 0 or C2 < 0:
        sys.exit('You should never see this error but I cannot evalute tDCF with negative weights - please check whether your ASV error rates are correctly computed?')

    tDCF = C1 * Pmiss_cm + C2 * Pfa_cm
    tDCF_norm = tDCF / np.minimum(C1, C2)

    if print_cost:
        print('t-DCF evaluation from [Nbona={}, Nspoof={}] trials\n'.format(bonafide_score_cm.size, spoof_score_cm.size))
        print('t-DCF MODEL')
        print('   Ptar         = {:8.5f} (Prior probability of target user)'.format(cost_model['Ptar']))
        print('   Pnon         = {:8.5f} (Prior probability of nontarget user)'.format(cost_model['Pnon']))
        print('   Pspoof       = {:8.5f} (Prior probability of spoofing attack)'.format(cost_model['Pspoof']))
        print('   Cfa_asv      = {:8.5f} (Cost of ASV falsely accepting a nontarget)'.format(cost_model['Cfa_asv']))
        print('   Cmiss_asv    = {:8.5f} (Cost of ASV falsely rejecting target speaker)'.format(cost_model['Cmiss_asv']))
        print('   Cfa_cm       = {:8.5f} (Cost of CM falsely passing a spoof to ASV system)'.format(cost_model['Cfa_cm']))
        print('   Cmiss_cm     = {:8.5f} (Cost of CM falsely blocking target utterance which never reaches ASV)'.format(cost_model['Cmiss_cm']))
        print('\n   Implied normalized t-DCF function (depends on t-DCF parameters and ASV errors), s=CM threshold)')

        if C2 == np.minimum(C1, C2):
            print('   tDCF_norm(s) = {:8.5f} x Pmiss_cm(s) + Pfa_cm(s)\n'.format(C1 / C2))
        else:
            print('   tDCF_norm(s) = Pmiss_cm(s) + {:8.5f} x Pfa_cm(s)\n'.format(C2 / C1))

    return tDCF_norm, CM_thresholds
