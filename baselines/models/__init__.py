"""
Baseline model wrappers for CosmosXRay360 Phase 1 Evaluation
Organized collection of 6 real implementations with actual module integration.
"""

from .svdrr import SVDRRWrapper
from .xraysyn import XRaySynWrapper
from .mednerf import MedNeRFWrapper
from .pixelnerf import PixelNeRFWrapper
from .naf import NAFWrapper
from .dx2ct import Dx2CTWrapper
from .utils import compute_psnr, compute_ssim, compute_lpips, apply_beer_lambert_correction, normalize_tensor

__all__ = [
    "SVDRRWrapper",
    "XRaySynWrapper",
    "MedNeRFWrapper",
    "PixelNeRFWrapper",
    "NAFWrapper",
    "Dx2CTWrapper",
    "compute_psnr",
    "compute_ssim",
    "compute_lpips",
    "apply_beer_lambert_correction",
    "normalize_tensor",
]
