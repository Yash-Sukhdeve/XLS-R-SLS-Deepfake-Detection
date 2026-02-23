"""Backward-compatibility shim. Use sls_asvspoof.data_utils instead."""
import warnings
warnings.warn(
    "Importing from 'data_utils_SSL' is deprecated. Use 'from sls_asvspoof.data_utils import ...' instead.",
    DeprecationWarning, stacklevel=2
)
from sls_asvspoof.data_utils import (
    genSpoof_list, genSpoof_list2019, pad,
    Dataset_ASVspoof2019_train, Dataset_ASVspoof2021_eval,
    Dataset_in_the_wild_eval, process_Rawboost_feature
)
