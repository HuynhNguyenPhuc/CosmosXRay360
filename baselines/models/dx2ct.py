"""Dx2CT (ICASSP 2025)."""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import warnings

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
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


# --- Logger --- #
logger = logging.getLogger(__name__)


try:
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
        azim_range = np.linspace(azimuths[0], azimuths[1], azimuths[2])
        
        try:
            if not isinstance(input_xr, torch.Tensor):
                input_xr = torch.from_numpy(input_xr).float()
            if input_xr.dim() == 2:
                input_xr = input_xr.unsqueeze(0).unsqueeze(0)
            
            input_xr = input_xr.to(self.device)
            B = input_xr.shape[0]
            
            with torch.no_grad():
                pa_xray = input_xr
                lat_xray = input_xr
                
                num_slices = 32
                spatial_size = 64
                
                # Generate 3D grid coordinates
                grid_y, grid_x = torch.meshgrid(
                    torch.linspace(-1, 1, spatial_size, device=self.device),
                    torch.linspace(-1, 1, spatial_size, device=self.device),
                    indexing="ij"
                )
                
                # Batched DDIM denoising across slices
                self.scheduler.set_timesteps(20)

                z_vals = -1.0 + 2.0 * torch.arange(num_slices, device=self.device) / max(1, num_slices - 1)
                coords_3d = torch.stack([
                    grid_x.flatten().unsqueeze(0).expand(num_slices, -1),
                    grid_y.flatten().unsqueeze(0).expand(num_slices, -1),
                    z_vals.unsqueeze(1).expand(-1, spatial_size * spatial_size),
                ], dim=-1)
                coords_3d = coords_3d.repeat(B, 1, 1)

                pa_batched = pa_xray.repeat_interleave(num_slices, dim=0)
                lat_batched = lat_xray.repeat_interleave(num_slices, dim=0)

                slice_t = torch.randn(B * num_slices, 1, spatial_size, spatial_size, device=self.device)

                for t in self.scheduler.timesteps:
                    t_batch = torch.full((B * num_slices,), t, device=self.device, dtype=torch.long)
                    noise_pred = self.model(slice_t, pa_batched, lat_batched, coords_3d, timesteps=t_batch)
                    slice_t = self.scheduler.step(noise_pred, t, slice_t).prev_sample

                vol_pred = slice_t.view(B, num_slices, spatial_size, spatial_size).unsqueeze(1)

                def sample_fn(pts: torch.Tensor) -> torch.Tensor:
                    grid_coords = pts.view(1, 1, 1, -1, 3)
                    sampled = F.grid_sample(
                        vol_pred, grid_coords, mode="bilinear",
                        padding_mode="zeros", align_corners=True,
                    )
                    return sampled.view(-1, 1)

                for azimuth in azim_range:
                    # Perspective ray marching with mean reduction
                    proj = perspective_ray_march(
                        sample_fn=sample_fn, azimuth=float(azimuth), grid_res=spatial_size,
                        device=self.device, reduction="mean",
                    )

                    # Resize projection to target 256x256
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
