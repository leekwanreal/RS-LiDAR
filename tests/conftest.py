"""Make repository modules importable in tests, matching how the entrypoints run.

The entrypoints are executed from the repository root and add `fkd_diffusers/` to `sys.path`
(the pipelines use flat imports such as `from fkd_class import FKD`).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (ROOT, os.path.join(ROOT, "fkd_diffusers")):
    if path not in sys.path:
        sys.path.insert(0, path)
os.chdir(ROOT)
