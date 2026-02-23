"""Reproducibility utilities.

Originally from core_scripts/startup_config.py by Xin Wang (wangxin@nii.ac.jp).
"""

import os
import random
import numpy as np
import torch


def set_random_seed(random_seed, args=None):
    """Set random seed for numpy, python, and cudnn for reproducibility.

    Args:
        random_seed: Integer random seed.
        args: Argument namespace with cudnn_deterministic_toggle and
              cudnn_benchmark_toggle attributes (optional).
    """
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    if args is None:
        cudnn_deterministic = True
        cudnn_benchmark = False
    else:
        cudnn_deterministic = args.cudnn_deterministic_toggle
        cudnn_benchmark = args.cudnn_benchmark_toggle

        if not cudnn_deterministic:
            print("cudnn_deterministic set to False")
        if cudnn_benchmark:
            print("cudnn_benchmark set to True")

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
    return
