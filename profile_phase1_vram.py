import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from tests.profile_phase1_vram import run_vram_profiler

if __name__ == "__main__":
    print("=" * 65)
    print("RUNNING PHASE 1 VRAM CEILING & MEMORY LEAK PROFILER")
    print("=" * 65)
    run_vram_profiler(is_sdxl=False)
    run_vram_profiler(is_sdxl=True)
