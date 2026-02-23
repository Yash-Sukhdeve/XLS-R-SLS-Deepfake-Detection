"""Unified evaluation script for DF, LA, and In-the-Wild tracks.

Usage:
    python -m sls_asvspoof.evaluate --track DF scores/scores_DF.txt ./keys eval
    python -m sls_asvspoof.evaluate --track LA scores/scores_LA.txt ./keys eval
    python -m sls_asvspoof.evaluate --track Wild scores/scores_Wild.txt ./keys eval
"""

import sys
import os.path
import argparse
import numpy as np
import pandas
from sls_asvspoof.metrics import compute_eer, compute_tDCF, obtain_asv_error_rates


# --- DF track evaluation ---

def eval_df(score_file, truth_dir, phase):
    cm_key_file = os.path.join(truth_dir, 'CM', 'trial_metadata.txt')
    cm_data = pandas.read_csv(cm_key_file, sep=' ', header=None)
    submission_scores = pandas.read_csv(score_file, sep=' ', header=None, skipinitialspace=True)
    if len(submission_scores) != len(cm_data):
        print('CHECK: submission has %d of %d expected trials.' % (len(submission_scores), len(cm_data)))
        exit(1)
    if len(submission_scores.columns) > 2:
        print('CHECK: submission has more columns (%d) than expected (2).' % len(submission_scores.columns))
        exit(1)

    cm_scores = submission_scores.merge(cm_data[cm_data[7] == phase], left_on=0, right_on=1, how='inner')
    bona_cm = cm_scores[cm_scores[5] == 'bonafide']['1_x'].values
    spoof_cm = cm_scores[cm_scores[5] == 'spoof']['1_x'].values
    eer_cm = compute_eer(bona_cm, spoof_cm)[0]
    print("eer: %.2f" % (100 * eer_cm))
    return eer_cm


# --- LA track evaluation ---

Pspoof_LA = 0.05
cost_model_LA = {
    'Pspoof': Pspoof_LA,
    'Ptar': (1 - Pspoof_LA) * 0.99,
    'Pnon': (1 - Pspoof_LA) * 0.01,
    'Cmiss': 1,
    'Cfa': 10,
    'Cfa_spoof': 10,
}


def load_asv_metrics_la(truth_dir, phase):
    asv_key_file = os.path.join(truth_dir, 'LA', 'ASV', 'trial_metadata.txt')
    asv_scr_file = os.path.join(truth_dir, 'LA', 'ASV', 'ASVTorch_Kaldi', 'score.txt')

    asv_key_data = pandas.read_csv(asv_key_file, sep=' ', header=None)
    asv_scr_data = pandas.read_csv(asv_scr_file, sep=' ', header=None)[asv_key_data[7] == phase]
    idx_tar = asv_key_data[asv_key_data[7] == phase][5] == 'target'
    idx_non = asv_key_data[asv_key_data[7] == phase][5] == 'nontarget'
    idx_spoof = asv_key_data[asv_key_data[7] == phase][5] == 'spoof'

    tar_asv = asv_scr_data[2][idx_tar]
    non_asv = asv_scr_data[2][idx_non]
    spoof_asv = asv_scr_data[2][idx_spoof]
    eer_asv, asv_threshold = compute_eer(tar_asv, non_asv)
    [Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv] = obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold)

    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv


def performance_la(cm_scores, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, invert=False):
    bona_cm = cm_scores[cm_scores[5] == 'bonafide']['1_x'].values
    spoof_cm = cm_scores[cm_scores[5] == 'spoof']['1_x'].values

    if not invert:
        eer_cm = compute_eer(bona_cm, spoof_cm)[0]
    else:
        eer_cm = compute_eer(-bona_cm, -spoof_cm)[0]

    if not invert:
        tDCF_curve, _ = compute_tDCF(bona_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model_LA, False)
    else:
        tDCF_curve, _ = compute_tDCF(-bona_cm, -spoof_cm, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model_LA, False)

    min_tDCF_index = np.argmin(tDCF_curve)
    min_tDCF = tDCF_curve[min_tDCF_index]

    return min_tDCF, eer_cm


def eval_la(score_file, truth_dir, phase):
    cm_key_file = os.path.join(truth_dir, 'LA', 'CM', 'trial_metadata.txt')
    Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv = load_asv_metrics_la(truth_dir, phase)
    cm_data = pandas.read_csv(cm_key_file, sep=' ', header=None)
    submission_scores = pandas.read_csv(score_file, sep=' ', header=None, skipinitialspace=True)

    if len(submission_scores) != len(cm_data):
        print('CHECK: submission has %d of %d expected trials.' % (len(submission_scores), len(cm_data)))
        exit(1)

    cm_scores = submission_scores.merge(cm_data[cm_data[7] == phase], left_on=0, right_on=1, how='inner')
    min_tDCF, eer_cm = performance_la(cm_scores, Pfa_asv, Pmiss_asv, Pfa_spoof_asv)

    print("min_tDCF: %.4f" % min_tDCF)
    print("eer: %.2f" % (100 * eer_cm))

    # Check for inverted scores
    min_tDCF2, eer_cm2 = performance_la(cm_scores, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, invert=True)

    if min_tDCF2 < min_tDCF:
        print(
            'CHECK: we negated your scores and achieved a lower min t-DCF. Before: %.3f - Negated: %.3f - your class labels are swapped during training... this will result in poor challenge ranking' % (
            min_tDCF, min_tDCF2))

    if min_tDCF == min_tDCF2:
        print(
            'WARNING: your classifier might not work correctly, we checked if negating your scores gives different min t-DCF - it does not. Are all values the same?')

    return min_tDCF


# --- In-the-Wild track evaluation ---

def eval_wild(score_file, truth_dir, phase):
    cm_key_file = os.path.join(truth_dir, 'in_the_wild_key.txt')
    cm_data = pandas.read_csv(cm_key_file, sep=' ', header=None)
    submission_scores = pandas.read_csv(score_file, sep=' ', header=None, skipinitialspace=True)
    if len(submission_scores) != len(cm_data):
        print('CHECK: submission has %d of %d expected trials.' % (len(submission_scores), len(cm_data)))
        exit(1)
    if len(submission_scores.columns) > 2:
        print('CHECK: submission has more columns (%d) than expected (2).' % len(submission_scores.columns))
        exit(1)

    cm_scores = submission_scores.merge(cm_data, left_on=0, right_on=1, how='inner')
    bona_cm = cm_scores[cm_scores[5] == 'bona-fide']['1_x'].values
    spoof_cm = cm_scores[cm_scores[5] == 'spoof']['1_x'].values
    eer_cm = compute_eer(bona_cm, spoof_cm)[0]
    print("eer: %.2f" % (100 * eer_cm))
    return eer_cm


# --- Main CLI ---

def main():
    parser = argparse.ArgumentParser(description='Evaluate ASVspoof / In-the-Wild scores')
    parser.add_argument('--track', type=str, required=True,
                        choices=['DF', 'LA', 'Wild'],
                        help='Evaluation track: DF, LA, or Wild')
    parser.add_argument('score_file', type=str,
                        help='Path to score file')
    parser.add_argument('truth_dir', type=str,
                        help='Path to keys/truth directory')
    parser.add_argument('phase', type=str,
                        choices=['progress', 'eval', 'hidden_track'],
                        help='Evaluation phase')
    args = parser.parse_args()

    if not os.path.isfile(args.score_file):
        print("%s doesn't exist" % args.score_file)
        exit(1)

    if not os.path.isdir(args.truth_dir):
        print("%s doesn't exist" % args.truth_dir)
        exit(1)

    if args.track == 'DF':
        eval_df(args.score_file, args.truth_dir, args.phase)
    elif args.track == 'LA':
        eval_la(args.score_file, args.truth_dir, args.phase)
    elif args.track == 'Wild':
        eval_wild(args.score_file, args.truth_dir, args.phase)


if __name__ == '__main__':
    main()
