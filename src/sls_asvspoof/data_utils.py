"""Dataset classes and utilities for ASVspoof and In-the-Wild data loading."""

import os
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import librosa
from torch.utils.data import Dataset
from sls_asvspoof.rawboost import (
    ISD_additive_noise, LnL_convolutive_noise, SSI_additive_noise, normWav
)
from random import randrange
import random


def genSpoof_list2019(dir_meta, is_train=False, is_eval=False):
    d_meta = {}
    file_list = []
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    if (is_eval):
        for line in l_meta:
            key = line.strip()
            file_list.append(key)
        return file_list


def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    d_meta = {}
    file_list = []
    with open(dir_meta, 'r') as f:
         l_meta = f.readlines()

    if (is_train):
        for line in l_meta:
             _, key, _, _, label = line.strip().split()
             file_list.append(key)
             d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list

    elif(is_eval):
        for line in l_meta:
            key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
             _, key, _, _, label = line.strip().split()
             file_list.append(key)
             d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, args, list_IDs, labels, base_dir, algo):
        """Training dataset for ASVspoof2019 LA.

        Args:
            args: Argument namespace (uses audio_max_len, sample_rate, RawBoost params).
            list_IDs: List of utterance IDs.
            labels: Dict mapping utterance ID to label (1=bonafide, 0=spoof).
            base_dir: Base directory containing flac/ subdirectory.
            algo: RawBoost algorithm index (0-8).
        """
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.algo = algo
        self.args = args
        self.cut = getattr(args, 'audio_max_len', 64600)  # ~4 sec at 16kHz
        self.sr = getattr(args, 'sample_rate', 16000)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(
            os.path.join(self.base_dir, 'flac', utt_id + '.flac'), sr=self.sr
        )
        Y = process_Rawboost_feature(X, fs, self.args, self.algo)
        X_pad = pad(Y, self.cut)
        x_inp = Tensor(X_pad)
        target = self.labels[utt_id]
        return x_inp, target


class Dataset_ASVspoof2021_eval(Dataset):
    def __init__(self, list_IDs, base_dir, cut=64600, sr=16000):
        """Evaluation dataset for ASVspoof2021 (DF or LA track).

        Args:
            list_IDs: List of utterance IDs.
            base_dir: Base directory containing flac/ subdirectory.
            cut: Max audio length in samples (default: 64600).
            sr: Sample rate (default: 16000).
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = cut
        self.sr = sr

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        try:
            X, fs = librosa.load(
                os.path.join(self.base_dir, 'flac', utt_id + '.flac'),
                sr=self.sr
            )
        except Exception:
            X = np.zeros(self.cut, dtype=np.float32)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, utt_id


class Dataset_in_the_wild_eval(Dataset):
    def __init__(self, list_IDs, base_dir, cut=64600, sr=16000):
        """Evaluation dataset for In-the-Wild.

        Args:
            list_IDs: List of relative file paths.
            base_dir: Base directory for audio files.
            cut: Max audio length in samples (default: 64600).
            sr: Sample rate (default: 16000).
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = cut
        self.sr = sr

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(
            os.path.join(self.base_dir, utt_id), sr=self.sr
        )
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, utt_id


#--------------RawBoost data augmentation algorithms---------------------------##

def process_Rawboost_feature(feature, sr, args, algo):

    # Data process by Convolutive noise (1st algo)
    if algo == 1:
        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin, args.maxBiasLinNonLin, sr)

    # Data process by Impulsive noise (2nd algo)
    elif algo == 2:
        feature = ISD_additive_noise(feature, args.P, args.g_sd)

    # Data process by coloured additive noise (3rd algo)
    elif algo == 3:
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)

    # Data process by all 3 algo. together in series (1+2+3)
    elif algo == 4:
        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
                 args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin, args.maxBiasLinNonLin, sr)
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands, args.minF,
                args.maxF, args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)

    # Data process by 1st two algo. together in series (1+2)
    elif algo == 5:
        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
                 args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin, args.maxBiasLinNonLin, sr)
        feature = ISD_additive_noise(feature, args.P, args.g_sd)

    # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo == 6:
        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
                 args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin, args.maxBiasLinNonLin, sr)
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)

    # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo == 7:
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)

    # Data process by 1st two algo. together in Parallel (1||2)
    elif algo == 8:
        feature1 = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
                 args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin, args.maxBiasLinNonLin, sr)
        feature2 = ISD_additive_noise(feature, args.P, args.g_sd)
        feature_para = feature1 + feature2
        feature = normWav(feature_para, 0)

    # original data without Rawboost processing
    else:
        feature = feature

    return feature
