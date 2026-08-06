"""Dx2CT (ICASSP 2025)

Full 3D Position-aware Slice Diffusion Model.
"""

from __future__ import annotations

import logging
import os
import warnings

try:
    import numpy as np
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch or NumPy not available")

try:
    from diffusers import DDPMScheduler
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    warnings.warn("Diffusers not available")

from models.utils import normalize_tensor, perspective_ray_march

# Configure logger following STYLE.md
logger = logging.getLogger(__name__)

# Import the actual DX2CTModel dynamically to avoid sys.path conflicts with other 'model' packages
try:
    import sys
    import importlib.util
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "cloned", "DX2CT", "model.py")
    
    spec = importlib.util.spec_from_file_location("dx2ct_model_module", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec from {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["dx2ct_model_module"] = module
    spec.loader.exec_module(module)
    DX2CTModel = module.DX2CTModel
    DX2CT_MODEL_AVAILABLE = True
except Exception as e:
    DX2CT_MODEL_AVAILABLE = False
    logger.warning(f"Failed to import DX2CTModel: {e}", exc_info=True)
    warnings.warn("DX2CTModel not found")


import torch.nn.functional as F

class Dx2CTWrapper:
    """Wrapper for full Dx2CT baseline."""
    
    def __init__(self, checkpoint_path: str | None = None) -> None:
        """Loads and initializes the Dx2CT diffusion model.

        Args:
            checkpoint_path: Optional path to pre-trained weights.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.scheduler = None
        
        if not TORCH_AVAILABLE or not DX2CT_MODEL_AVAILABLE or not DIFFUSERS_AVAILABLE:
            logger.warning("[Dx2CT] Required libraries unavailable.")
            return
        
        try:
            self.model = DX2CTModel(embed_dim=128).to(self.device)
            self.scheduler = DDPMScheduler(num_train_timesteps=1000)
            
            checkpoint_loaded = False
            if checkpoint_path and os.path.exists(checkpoint_path):
                self.model.load_state_dict(
                    torch.load(checkpoint_path, map_location=self.device)
                )
                checkpoint_loaded = True

            self.model.eval()
            if checkpoint_loaded:
                logger.info("[Dx2CT] ✓ Loaded pretrained checkpoint successfully")
            else:
                logger.warning(
                    "[Dx2CT] ⚠ No checkpoint loaded (path missing/unset) -- "
                    "running with a randomly-initialized model"
                )
        except Exception as e:
            logger.error(f"[Dx2CT] ✗ Failed: {str(e)[:80]}")
    
    def infer_multi_views(
        self,
        input_xr: torch.Tensor,
        azimuths: tuple[float, float, int] = (0, 360, 93),
    ) -> list[torch.Tensor]:
        """Reconstructs 3D volume slice-by-slice and generates projection list at desired angles.

        Args:
            input_xr: Input 2D projection CXR [1, 1, 256, 256].
            azimuths: Target view boundaries as (start_angle, end_angle, N_views).

        Returns:
            A list of N synthesized 2D novel-view radiography tensors.
        """
        if self.model is None or self.scheduler is None:
            return []
        
        results = []
        # endpoint=True (default) matches datasets/pre_render_diffdrr.py's own
        # torch.linspace(0, 360, N) ground-truth convention -- and thus the same
        # 93-frame data Cosmos-Predict2.5 itself trains/evaluates on -- where frame
        # N-1 lands exactly back at 360=0. endpoint=False (93 intervals instead of
        # 92) drifts up to ~3.9 degrees off that by the last frame.
        azim_range = np.linspace(azimuths[0], azimuths[1], azimuths[2])
        
        try:
            if not isinstance(input_xr, torch.Tensor):
                input_xr = torch.from_numpy(input_xr).float()
            if input_xr.dim() == 2:
                input_xr = input_xr.unsqueeze(0).unsqueeze(0)
            
            input_xr = input_xr.to(self.device)
            B = input_xr.shape[0]
            
            with torch.no_grad():
                # We use input_xr for both PA and LAT as a fallback
                pa_xray = input_xr
                lat_xray = input_xr
                
                # To save memory and time during benchmark, we generate a low-res 32-slice volume
                num_slices = 32
                spatial_size = 64
                
                # We do NOT resize pa_xray to spatial_size. The feature extractor ALWAYS expects 256x256.
                # Only the target slice generation uses spatial_size (64).
                
                # Generate 3D grid coords for the whole slice
                # [spatial_size * spatial_size, 3]
                grid_y, grid_x = torch.meshgrid(
                    torch.linspace(-1, 1, spatial_size, device=self.device),
                    torch.linspace(-1, 1, spatial_size, device=self.device),
                    indexing="ij"
                )
                
                # All 32 slices share the same pa/lat conditioning and only differ in
                # their z-coordinate, so instead of running 32 independent 20-step
                # denoising loops back-to-back, fold the slice index into the batch
                # dimension and run a single 20-step loop over all of them at once --
                # same per-slice math (independent noise, same schedule), ~32x fewer
                # sequential model calls.
                self.scheduler.set_timesteps(20) # 20 DDIM steps for fast inference

                z_vals = -1.0 + 2.0 * torch.arange(num_slices, device=self.device) / max(1, num_slices - 1)
                coords_3d = torch.stack([
                    grid_x.flatten().unsqueeze(0).expand(num_slices, -1),
                    grid_y.flatten().unsqueeze(0).expand(num_slices, -1),
                    z_vals.unsqueeze(1).expand(-1, spatial_size * spatial_size),
                ], dim=-1)  # (num_slices, HW, 3)
                coords_3d = coords_3d.repeat(B, 1, 1)  # (B*num_slices, HW, 3)

                pa_batched = pa_xray.repeat_interleave(num_slices, dim=0)
                lat_batched = lat_xray.repeat_interleave(num_slices, dim=0)

                slice_t = torch.randn(B * num_slices, 1, spatial_size, spatial_size, device=self.device)

                for t in self.scheduler.timesteps:
                    t_batch = torch.full((B * num_slices,), t, device=self.device, dtype=torch.long)
                    noise_pred = self.model(slice_t, pa_batched, lat_batched, coords_3d, timesteps=t_batch)
                    slice_t = self.scheduler.step(noise_pred, t, slice_t).prev_sample

                # (B*num_slices, 1, H, W) -> [B, 1, D, H, W]
                vol_pred = slice_t.view(B, num_slices, spatial_size, spatial_size).unsqueeze(1)

                # vol_pred's own (D, H, W) axes are already world (Z, Y, X): D indexes
                # z_vals (world Z), H/W index grid_y/grid_x (world Y/X) -- see the
                # coords_3d construction above -- matching grid_sample's [..., (x,y,z)]
                # -> (W, H, D) convention directly, no permutation needed.
                def sample_fn(pts: torch.Tensor) -> torch.Tensor:
                    grid_coords = pts.view(1, 1, 1, -1, 3)
                    sampled = F.grid_sample(
                        vol_pred, grid_coords, mode="bilinear",
                        padding_mode="zeros", align_corners=True,
                    )
                    return sampled.view(-1, 1)

                for azimuth in azim_range:
                    # True perspective ray marching through vol_pred, matching the
                    # exact camera geometry datasets/pre_render_diffdrr.py used to
                    # render ground truth -- replaces rotate_volume_3d + axis mean.
                    # reduction="mean" (NOT "attenuation_sum"): the ground-truth
                    # `views/*.png` this projection is compared against is itself a
                    # raw (rescaled) line integral of density -- DiffDRR's Trilinear
                    # renderer computes `sum(density) * step_size`, with no exp()
                    # anywhere in `renderers/diffdrr/renderer.py`'s render() -- so
                    # even though vol_pred is now a genuine linear-attenuation-like
                    # field (post the real-axial-CT-slice training rewrite), applying
                    # Beer-Lambert here would compare an exponentiated projection
                    # against a non-exponentiated target: a value-space mismatch, not
                    # a fix. See perspective_ray_march's own `reduction` docstring
                    # (models/utils.py) and docs/GOTCHAS.md #1, which name Dx2CT as
                    # exactly this case. NAF's use of "attenuation_sum" is safe
                    # because its density field is gradient-fit per-scan directly
                    # against the (same non-exponentiated) target with this transform
                    # inside the loss, so it self-corrects regardless; Dx2CT's
                    # diffusion model is trained independently of this projection
                    # step (no such self-correction), so the reduction must match
                    # the ground truth's actual convention on its own.
                    proj = perspective_ray_march(
                        sample_fn=sample_fn, azimuth=float(azimuth), grid_res=spatial_size,
                        device=self.device, reduction="mean",
                    )

                    # Resize projection to 256x256 as required by unified eval
                    proj_256 = F.interpolate(
                        proj.unsqueeze(0).unsqueeze(0),
                        size=(256, 256),
                        mode="bilinear",
                        align_corners=True
                    ).squeeze(0).squeeze(0)

                    proj_corrected = normalize_tensor(proj_256)
                    results.append(proj_corrected)
        except Exception as e:
            logger.error(f"[Dx2CT] Inference execution failed: {e}")
        
        return results
