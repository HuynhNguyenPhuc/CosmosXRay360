"""NAF / SNAF (MICCAI 2022)

Implicit Neural Attenuation Coordinate Net for CBCT Reconstruction.
"""

from __future__ import annotations

import copy
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

# Setup early import paths for NAF package
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
naf_dir = os.path.join(base_dir, "cloned", "naf_cbct")
sys.path.insert(0, naf_dir)

try:
    # HashEncoder (Instant-NGP) multi-resolution grid encoder
    from src.encoder.hashencoder import HashEncoder
    NAF_HASHENCODER_AVAILABLE = True
except Exception:
    HashEncoder = None
    NAF_HASHENCODER_AVAILABLE = False

try:
    from src.encoder.freqencoder import FreqEncoder
    from src.network.network import DensityNetwork
    NAF_NETWORK_AVAILABLE = True
except Exception as e:
    NAF_NETWORK_AVAILABLE = False

from models.utils import build_perspective_ray_points, normalize_tensor, perspective_ray_march


# --- Logger --- #
logger = logging.getLogger(__name__)

FIT_GRID_RES = 64  # Coordinate grid resolution used during per-scan fitting.


def fit_density_field(
    model: "DensityNetwork",
    target_proj: torch.Tensor,
    bound: float,
    device: str,
    iterations: int = 200,
    lr: float = 1e-3,
) -> float:
    """Optimizes coordinate MLP weights to reproduce a single 2D projection.

    Args:
        model: DensityNetwork to fit in place.
        target_proj: Target 2D projection, [1, H, W] or [H, W].
        bound: Coordinate grid half-extent.
        device: Compute device.
        iterations: Number of gradient steps.
        lr: Adam learning rate.

    Returns:
        Final MSE loss value.
    """
    # Margin to keep points strictly within [-bound, bound]
    safe_bound = bound * (1.0 - 1e-6)

    # Normalize to 4D [B, C, H, W] regardless of whether the caller passed
    # [H, W], [1, H, W], or [1, 1, H, W], then squeeze back to [1, H, W].
    target = target_proj.to(device)
    while target.dim() < 4:
        target = target.unsqueeze(0)
    target = F.interpolate(
        target, size=(FIT_GRID_RES, FIT_GRID_RES), mode="bilinear", align_corners=False
    ).view(FIT_GRID_RES, FIT_GRID_RES)

    # Build fixed frontal ray geometry once
    fit_pts = build_perspective_ray_points(
        azimuth=0.0, grid_res=FIT_GRID_RES, device=device, bound=safe_bound,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    loss = torch.tensor(0.0)
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        # Perspective ray marching against input view
        pred_proj = perspective_ray_march(
            sample_fn=model, azimuth=0.0, grid_res=FIT_GRID_RES, device=device,
            bound=safe_bound, pts=fit_pts,
        )
        loss = F.mse_loss(pred_proj, target)
        loss.backward()
        optimizer.step()
    model.eval()
    return loss.item()


class NAFWrapper:
    """Wrapper for NAF baseline."""
    
    def __init__(self, checkpoint_path: str | None = None, iterations: int = 200) -> None:
        """Initializes the NAF coordinate MLP model.

        Args:
            checkpoint_path: Optional path to pre-trained weights file.
            iterations: Number of per-scan fitting steps.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.bound = 0.3
        self.iterations = iterations

        if not TORCH_AVAILABLE or not NAF_NETWORK_AVAILABLE:
            logger.warning("[NAF] Model libraries not fully available.")
            return

        try:
            # Reference config (cloned/naf_cbct/config/*_50.yaml): hashgrid encoder,
            # num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=19.
            if NAF_HASHENCODER_AVAILABLE:
                encoder = HashEncoder(
                    input_dim=3,
                    num_levels=16,
                    level_dim=2,
                    base_resolution=16,
                    log2_hashmap_size=19,
                )
            else:
                logger.warning(
                    "[NAF] HashEncoder unavailable on this host (no working nvcc/CUDA "
                    "toolchain) -- falling back to FreqEncoder, which departs from the "
                    "reference architecture."
                )
                encoder = FreqEncoder(
                    input_dim=3,
                    max_freq_log2=10 - 1,
                    N_freqs=10,
                    log_sampling=True,
                    include_input=True,
                    periodic_fns=(torch.sin, torch.cos),
                )

            # Reference config (cloned/naf_cbct/config/*_50.yaml): num_layers=4,
            # hidden_dim=32, skips=[2].
            model = DensityNetwork(
                encoder=encoder,
                bound=self.bound,
                num_layers=4,
                hidden_dim=32,
                skips=[2],
                out_dim=1,
                last_activation="sigmoid",
            )

            if checkpoint_path and os.path.exists(checkpoint_path):
                model.load_state_dict(
                    torch.load(checkpoint_path, map_location=self.device)
                )

            # Only publish self.model once construction + checkpoint load both
            # succeed, so a shape-mismatched checkpoint (e.g. stale weights from
            # the pre-hashgrid architecture) leaves self.model cleanly None
            # instead of a half-built, wrong-device, untrained network that
            # evaluate.py's `model is None` skip-check wouldn't catch.
            self.model = model.to(self.device)
            self.model.eval()
            logger.info("[NAF] ✓ Loaded successfully")
        except Exception as e:
            logger.error(f"[NAF] ✗ Failed: {str(e)[:80]}")
    
    def infer_multi_views(
        self,
        input_xr: torch.Tensor,
        azimuths: tuple[float, float, int] = (0, 360, 93),
        iterations: int | None = None,
    ) -> list[torch.Tensor]:
        """Fits coordinate field to input X-ray and renders novel view projections.

        Args:
            input_xr: Input 2D projection CXR [1, 1, 256, 256].
            azimuths: Target view boundaries as (start_angle, end_angle, N_views).
            iterations: Optional per-call fitting iteration budget override.

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

        try:
            # Query grid setup
            grid_size = 128
            H, W = input_xr.shape[-2], input_xr.shape[-1]

            model_fit = copy.deepcopy(self.model)
            fit_density_field(
                model_fit, input_xr, self.bound, self.device,
                iterations=iterations if iterations is not None else self.iterations,
            )

            # NAF queries coords in [-bound, bound]. See fit_density_field's
            # safe_bound comment: shrink slightly so linspace's float32 endpoint
            # rounding can't spuriously fail HashEncoder's domain-validity check.
            safe_bound = self.bound * (1.0 - 1e-6)
            grid = torch.linspace(-safe_bound, safe_bound, grid_size, device=self.device)
            coords = torch.stack(torch.meshgrid(grid, grid, grid, indexing="ij"), dim=-1)
            coords_flat = coords.reshape(-1, 3)

            with torch.no_grad():
                # Bake coordinate MLP into dense grid for fast rendering
                chunk = 1024 * 64
                attenuations_flat = torch.zeros(coords_flat.shape[0], 1, device=self.device)
                for i in range(0, coords_flat.shape[0], chunk):
                    attenuations_flat[i : i + chunk] = model_fit(
                        coords_flat[i : i + chunk]
                    )

                attenuations = attenuations_flat.reshape(1, 1, grid_size, grid_size, grid_size)

                def sample_fn(pts: torch.Tensor) -> torch.Tensor:
                    grid_coords = (pts / safe_bound).clamp(-1, 1).view(1, 1, 1, -1, 3)
                    sampled = F.grid_sample(
                        attenuations, grid_coords, mode="bilinear",
                        padding_mode="zeros", align_corners=True,
                    )
                    return sampled.view(-1, 1)

                for azimuth in azim_range:
                    # Perspective ray marching through baked volume
                    proj = perspective_ray_march(
                        sample_fn=sample_fn, azimuth=float(azimuth), grid_res=grid_size,
                        device=self.device,
                    )
                    if proj.shape[0] != H or proj.shape[1] != W:
                        proj = F.interpolate(
                            proj.unsqueeze(0).unsqueeze(0),
                            size=(H, W),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0).squeeze(0)
                    proj_corrected = normalize_tensor(proj)
                    results.append(proj_corrected)
        except Exception as e:
            logger.error(f"[NAF] Inference execution failed: {e}")
        
        return results
