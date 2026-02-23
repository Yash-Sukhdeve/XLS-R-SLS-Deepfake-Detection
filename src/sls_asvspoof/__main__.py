"""Allow running as python -m sls_asvspoof (defaults to training entry point)."""
import runpy
runpy.run_module('sls_asvspoof.train', run_name='__main__', alter_sys=True)
