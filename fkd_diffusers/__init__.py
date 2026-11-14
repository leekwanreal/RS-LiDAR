"""Diffusion pipelines, FK/RS-SMC resampling and reward models used by RS-LiDAR.

The modules in this package use flat imports (e.g. `from fkd_class import FKD`), so the package
directory is added to `sys.path` on import.
"""
import os
import sys

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)
