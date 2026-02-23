"""Backward-compatibility shim. Use 'python -m sls_asvspoof.train' instead."""
import warnings
warnings.warn(
    "Running 'python main.py' is deprecated. Use 'python -m sls_asvspoof.train' instead.",
    DeprecationWarning, stacklevel=2
)
import runpy
runpy.run_module('sls_asvspoof.train', run_name='__main__', alter_sys=True)
