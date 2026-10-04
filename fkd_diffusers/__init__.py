"""
fkd_diffusers package initialization
"""
import os
import sys

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

try:
    from fkd_diffusers.fkd_pipeline_sd import FKDStableDiffusion
    from fkd_diffusers.fkd_pipeline_sdxl import FKDStableDiffusionXL
except ImportError:
    try:
        from fkd_pipeline_sd import FKDStableDiffusion
        from fkd_pipeline_sdxl import FKDStableDiffusionXL
    except ImportError:
        pass

try:
    from fkd_diffusers.image_reward_utils import rm_load
except ImportError:
    try:
        from image_reward_utils import rm_load
    except ImportError:
        pass
