"""XRaySyn (AAAI 2021)

Physics-supervised 3D Backprojection and Voxel Reconstruction.
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
import warnings

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch or NumPy not available")

# Setup early import paths for XraySyn package
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(base_dir, "cloned", "XraySyn"))
sys.path.insert(0, os.path.join(base_dir, "cloned", "XraySyn", "xraysyn", "networks", "drr_projector"))

try:
    from xraysyn.models.ct2xray_real_gan_meta import XraySynModel
    XRAYSYN_MODEL_AVAILABLE = True
except ImportError:
    XRAYSYN_MODEL_AVAILABLE = False

from models.utils import normalize_tensor


# --- Logger --- #
logger = logging.getLogger(__name__)


def _get_T_batched(model: "XraySynModel", inp: list[float], batch_size: int) -> torch.Tensor:
    """Builds a 6-DoF pose transform matching a given batch size."""
    T = model.get_T(inp)
    if batch_size <= T.shape[0]:
        return T[:batch_size]
    reps = (batch_size + T.shape[0] - 1) // T.shape[0]
    return T.repeat(reps, 1, 1)[:batch_size]


def _get_T_multi(model: "XraySynModel", azimuths_deg: list[float], batch_size: int) -> torch.Tensor:
    """Builds one 6-DoF pose transform per azimuth in degrees."""
    T_stack = torch.cat(
        [model.get_T([1, az / 180.0, 0, 0, 0, 0])[:1] for az in azimuths_deg], dim=0
    )  # (N, 4, 4)
    if batch_size <= 1:
        return T_stack
    return T_stack.repeat(batch_size, 1, 1)


class XRaySynWrapper:
    """Wrapper for XRaySyn baseline from cloned/XraySyn."""
    
    def __init__(self, checkpoint_path: str | None = None) -> None:
        """Initializes and loads the XRaySyn multi-view reconstruction GAN.

        Args:
            checkpoint_path: Optional path to pre-trained weights file.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        
        if not TORCH_AVAILABLE or not XRAYSYN_MODEL_AVAILABLE:
            logger.warning("[XRaySyn] Model libraries not fully available.")
            return
        
        try:
            # Change working directory temporarily to let XraySynModel load simplified_bone_absorb_2d.pt, etc.
            old_cwd = os.getcwd()
            xraysyn_dir = os.path.join(base_dir, "cloned", "XraySyn")
            os.chdir(xraysyn_dir)
            try:
                # NOTE: XraySynModel defaults to device="cuda:0" regardless of
                # availability -- must pass self.device explicitly or construction
                # crashes on CPU-only hosts even though self.device already fell back.
                self.model = XraySynModel(device=self.device)
            finally:
                os.chdir(old_cwd)
            
            if checkpoint_path and os.path.exists(checkpoint_path):
                self.model.load(checkpoint_path)
                
            logger.info("[XRaySyn] ✓ Loaded successfully")
        except Exception as e:
            logger.error(f"[XRaySyn] ✗ Failed: {str(e)[:80]}")
    
    def infer_multi_views(
        self,
        input_xr: torch.Tensor,
        azimuths: tuple[float, float, int] = (0, 360, 93),
    ) -> list[torch.Tensor]:
        """Synthesizes novel views via backprojection and 2D refinement.

        Args:
            input_xr: Input 2D projection CXR [1, 1, 256, 256].
            azimuths: Target view boundaries as (start_angle, end_angle, N_views).

        Returns:
            List of N synthesized 2D novel-view radiography tensors.
        """
        if self.model is None:
            return []

        results = []
        # endpoint=True (default) matches datasets/pre_render_diffdrr.py's own
        # torch.linspace(0, 360, N) ground-truth convention -- and thus the same
        # 93-frame data Cosmos-Predict2.5 itself trains/evaluates on -- where frame
        # N-1 lands exactly back at 360=0. endpoint=False (93 intervals instead of
        # 92) drifts up to ~3.9 degrees off that by the last frame.
        azim_range = np.linspace(azimuths[0], azimuths[1], azimuths[2])
        VIEW_BATCH = 8  # 128^3 volumes are repeated per view in the chunk -- keep small.

        try:
            if not isinstance(input_xr, torch.Tensor):
                input_xr = torch.from_numpy(input_xr).float()

            # XRaySyn expects [B, C, H, W]
            if input_xr.dim() == 2:
                input_xr = input_xr.unsqueeze(0).unsqueeze(0)
            elif input_xr.dim() == 3:
                input_xr = input_xr.unsqueeze(0)

            input_xr = input_xr.to(self.device)
            batch_size = input_xr.shape[0]
            H, W = input_xr.shape[-2], input_xr.shape[-1]

            with torch.no_grad():
                xray128 = self.model.avgpool(input_xr)
                T_in = _get_T_batched(self.model, [1, 0, 0, 0, 0, 0], batch_size)
                vol_in = self.model.backproj(xray128, T_in)
                vol_pred_temp = self.model.net3d(vol_in) * 0.5 + 0.5
                bone_mask = vol_pred_temp[:, [0]]
                bone_ct = vol_pred_temp[:, [1]] * bone_mask
                tissue_ct = vol_pred_temp[:, [2]] * (1 - bone_mask)
                vol_pred = bone_ct + tissue_ct  # [B, 1, 128, 128, 128]

                for start in range(0, len(azim_range), VIEW_BATCH):
                    chunk = azim_range[start : start + VIEW_BATCH]
                    n = len(chunk)

                    # Same 6-DoF pose convention as the reference ``self.views``
                    # sweep: theta_x fixed at the frontal pose, theta_y carries the
                    # azimuth (get_T scales its input list by pi, so this is radians).
                    T_other = _get_T_multi(self.model, list(chunk), batch_size)  # (B*n, 4, 4)
                    vol_chunk = vol_pred.repeat_interleave(n, dim=0)
                    bone_chunk = bone_mask.repeat_interleave(n, dim=0)
                    input_chunk = input_xr.repeat_interleave(n, dim=0)

                    _, mat_pred = self.model.ct2xray(vol_chunk, bone_chunk, T_other)
                    mat_refine = self.model.net2d(mat_pred, input_chunk) + self.model.upsample(mat_pred)
                    xray_refine = self.model.mat2xray(mat_refine)  # (B*n, 1, 256, 256)

                    for i in range(xray_refine.shape[0]):
                        proj = xray_refine[i, 0]
                        if proj.shape[0] != H or proj.shape[1] != W:
                            proj = F.interpolate(
                                proj.unsqueeze(0).unsqueeze(0),
                                size=(H, W),
                                mode="bilinear",
                                align_corners=False,
                            ).squeeze(0).squeeze(0)

                        results.append(proj)

                # Normalize across the full sequence to preserve inter-view relative intensity
                if results:
                    results = list(normalize_tensor(torch.stack(results, dim=0)).unbind(0))
        except Exception as e:
            logger.error(f"[XRaySyn] Inference execution failed: {e}\n{traceback.format_exc()}")

        return results
