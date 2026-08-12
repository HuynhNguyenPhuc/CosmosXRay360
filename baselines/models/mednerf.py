"""MedNeRF (EMBC 2022)

Implicit Continuous Neural Coordinate Representation with GAN Supervision.
"""

from __future__ import annotations

import copy
import logging
import os
import sys
import traceback
import warnings

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch or NumPy not available")

# Setup sys.path early for top-level package imports
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mednerf_dir = os.path.join(base_dir, "cloned", "mednerf", "graf-main")
sys.path.insert(0, mednerf_dir)
sys.path.insert(0, os.path.join(mednerf_dir, "submodules"))

try:
    from graf.config import build_models, get_render_poses
    from graf.utils import to_theta
    from submodules.GAN_stability.gan_training import lpips
    from submodules.GAN_stability.gan_training.checkpoints import CheckpointIO
    from submodules.GAN_stability.gan_training.config import load_config
    MEDNERF_AVAILABLE = True
except ImportError:
    MEDNERF_AVAILABLE = False

from models.utils import normalize_tensor


# --- Logger --- #
logger = logging.getLogger(__name__)

# Cached process-wide LPIPS loss instance
_PERCEPT_LOSS: "torch.nn.Module | None" = None


def _get_perceptual_loss(device: str) -> "torch.nn.Module":
    global _PERCEPT_LOSS
    if _PERCEPT_LOSS is None:
        _PERCEPT_LOSS = lpips.PerceptualLoss(model="net-lin", net="alex", use_gpu=(device == "cuda"))
    return _PERCEPT_LOSS


def fit_latent_and_weights(
    generator_test: "torch.nn.Module",
    target_xr: "torch.Tensor",
    *,
    z_dim: int,
    img_size: int,
    radius: float,
    theta_mean: float,
    device: str,
    iterations: int = 500,
    use_amp: bool = False,
) -> tuple["torch.Tensor", float]:
    """Jointly optimizes latent code z and generator weights to reconstruct target X-ray.

    Args:
        generator_test: Generator model instance.
        target_xr: Target X-ray tensor, shape [H, W] or [1, H, W].
        z_dim: Latent code dimension.
        img_size: Render resolution.
        radius: Camera distance.
        theta_mean: Mean polar elevation angle in degrees.
        device: Compute device.
        iterations: Number of optimization steps.
        use_amp: If True, uses torch.autocast for mixed precision.

    Returns:
        Tuple of (optimized latent z [1, z_dim], final reconstruction loss).
    """
    z = torch.randn(1, z_dim, device=device, requires_grad=True)

    percept = _get_perceptual_loss(device)

    z_optim = optim.Adam([z], lr=0.0005, betas=(0.0, 0.999))
    g_optim = optim.RMSprop(generator_test.parameters(), lr=0.0005, alpha=0.99, eps=1e-8)

    # Frontal pose for input X-ray
    pose = get_render_poses(radius=radius, angle_range=(0, 0), theta=theta_mean, N=1)
    pose = pose[0].to(device)
    rays = generator_test.val_ray_sampler(
        img_size, img_size, generator_test.focal, pose
    )[0].unsqueeze(0)

    # Reshape rays to generator format
    rays = rays.permute(1, 0, 2, 3).flatten(1, 2)

    # Resize target to img_size and 4D format [B, C, H, W]
    target_xr_4d = target_xr
    while target_xr_4d.dim() < 4:
        target_xr_4d = target_xr_4d.unsqueeze(0)
    target_xr_resized = F.interpolate(
        target_xr_4d,
        size=(img_size, img_size),
        mode="bilinear",
    )
    if target_xr_resized.shape[1] == 1:
        target_xr_resized = target_xr_resized.repeat(1, 3, 1, 1)

    # Normalize target to [-1, 1] range
    target_xr_resized = target_xr_resized * 2.0 - 1.0

    logger.info(f"[MedNeRF] Optimizing latent code for {iterations} iterations...")
    generator_test.train()
    generator_test.chunk = 4096
    rec_loss = torch.tensor(0.0)
    scaler = torch.amp.GradScaler(enabled=(use_amp and device.startswith("cuda")))
    autocast_device = "cuda" if device.startswith("cuda") else "cpu"
    for _ in range(iterations):
        z_optim.zero_grad()
        g_optim.zero_grad()

        with torch.autocast(device_type=autocast_device, enabled=use_amp):
            outputs = generator_test(z, rays=rays)
            rgb = outputs[0] if isinstance(outputs, tuple) else outputs

            # Reshape rgb to image format
            if rgb.shape[0] != img_size * img_size:
                dim_h = int(np.sqrt(rgb.shape[0]))
                rgb = rgb.view(1, dim_h, dim_h, -1).permute(0, 3, 1, 2)
                rgb = F.interpolate(rgb, size=(img_size, img_size), mode="bilinear", align_corners=True)
            else:
                rgb = rgb.view(1, img_size, img_size, -1).permute(0, 3, 1, 2)

            if rgb.shape[1] == 4:
                rgb = rgb[:, :3]
            elif rgb.shape[1] == 1:
                rgb = rgb.repeat(1, 3, 1, 1)

            # Standard-normal prior regularization on z
            nll = (z ** 2 / 2).mean()
            rec_loss = (
                0.3 * percept(rgb, target_xr_resized).sum()
                + 0.1 * F.mse_loss(rgb, target_xr_resized)
                + 0.3 * nll
            )
        scaler.scale(rec_loss).backward()
        scaler.step(z_optim)
        scaler.step(g_optim)
        scaler.update()

    generator_test.eval()
    return z, rec_loss.item()


class MedNeRFWrapper:
    """Wrapper for MedNeRF baseline."""
    
    def __init__(self, checkpoint_path: str | None = None, iterations: int = 50) -> None:
        """Initializes the MedNeRF generator.

        Args:
            checkpoint_path: Optional path to pre-trained weights file.
            iterations: Per-patient latent/weight fitting steps.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.config = None
        self.iterations = iterations

        if not TORCH_AVAILABLE or not MEDNERF_AVAILABLE:
            logger.warning("[MedNeRF] Model libraries not fully available.")
            return
        
        try:
            # Load actual MedNeRF config
            config_file = load_config(
                os.path.join(mednerf_dir, "configs/chest.yaml"),
                os.path.join(mednerf_dir, "configs/default.yaml"),
            )
            self.config = config_file
            
            # Setup standard hwfr parameters to avoid requiring loading datasets
            H = W = imsize = config_file["data"]["imsize"]
            fov = config_file["data"]["fov"]
            focal = W / 2.0 * 1.0 / np.tan((0.5 * fov * np.pi / 180.0))
            radius = config_file["data"]["radius"]
            if isinstance(radius, str):
                radius = tuple(float(r) for r in radius.split(","))
                radius = max(radius)
            config_file["data"]["hwfr"] = [H, W, focal, radius]
            self.radius = radius

            # Mean polar/elevation angle of the trained pose distribution (matches
            # render_xray_G.py's theta_mean). chest.yaml's vmin/vmax sit around 70-85
            # degrees (near-horizontal circular sweep around the chest); theta=0 would
            # be the poles (looking straight along the body axis), a pose the
            # generator was never trained on.
            self.theta_mean = 0.5 * (
                to_theta(config_file["data"]["vmin"]) + to_theta(config_file["data"]["vmax"])
            )

            # Build real MedNeRF generator architecture
            generator, _ = build_models(config_file, disc=False)
            generator = generator.to(self.device)
            generator.chunk = 16384
            
            if checkpoint_path and os.path.exists(checkpoint_path):
                checkpoint_io = CheckpointIO(
                    checkpoint_dir=os.path.dirname(checkpoint_path)
                )
                checkpoint_io.register_modules(**{k + "_test": v for k, v in generator.module_dict.items()})
                checkpoint_io.load(os.path.basename(checkpoint_path))
                
            self.model = generator
            self.model.eval()
            logger.info("[MedNeRF] ✓ Loaded successfully")
        except Exception as e:
            logger.error(f"[MedNeRF] ✗ Failed: {str(e)[:80]}")
            
    def infer_multi_views(
        self,
        input_xr: torch.Tensor,
        azimuths: tuple[float, float, int] = (0, 360, 93),
        iterations: int | None = None,
    ) -> list[torch.Tensor]:
        """Optimizes generator weights for input X-ray and renders novel views.

        Args:
            input_xr: Input 2D projection CXR [1, 1, 256, 256].
            azimuths: Target view boundaries as (start_angle, end_angle, N_views).
            iterations: Optional fitting iteration override.

        Returns:
            List of N synthesized 2D novel-view radiography tensors.
        """
        if self.model is None or self.config is None:
            return []
        
        results = []
        # endpoint=True (default) matches datasets/pre_render_diffdrr.py's own
        # torch.linspace(0, 360, N) ground-truth convention -- and thus the same
        # 93-frame data Cosmos-Predict2.5 itself trains/evaluates on -- where frame
        # N-1 lands exactly back at 360=0. endpoint=False (93 intervals instead of
        # 92) drifts up to ~3.9 degrees off that by the last frame.
        azim_range = np.linspace(azimuths[0], azimuths[1], azimuths[2])
        
        try:
            # 1. Optimize generator weights / latent code for this specific patient
            # Copy generator to avoid modifying the original weights permanently
            generator_test = copy.deepcopy(self.model)
            generator_test.parameters = lambda: generator_test._parameters
            generator_test.named_parameters = (
                lambda: generator_test._named_parameters
            )
            
            if not isinstance(input_xr, torch.Tensor):
                input_xr = torch.from_numpy(input_xr).float()
            
            input_xr = input_xr.to(self.device).squeeze()
            
            # Run optimization loop (authentic to MedNeRF render_xray_G.py)
            z_optimized, _ = fit_latent_and_weights(
                generator_test,
                input_xr,
                z_dim=self.config["z_dist"]["dim"],
                img_size=self.config["data"]["imsize"],
                radius=self.radius,
                theta_mean=self.theta_mean,
                device=self.device,
                iterations=iterations if iterations is not None else self.iterations,
            )
            
            # 2. Render novel views
            img_size = self.config["data"]["imsize"]
            
            with torch.no_grad():
                for azimuth in azim_range:
                    pose = get_render_poses(
                        radius=self.radius, angle_range=(azimuth, azimuth), theta=self.theta_mean, N=1
                    )
                    pose = pose[0].to(self.device)
                    
                    rays = generator_test.val_ray_sampler(
                        img_size, img_size, generator_test.focal, pose
                    )[0]
                    
                    outputs = generator_test(z_optimized, rays=rays)
                    if isinstance(outputs, tuple):
                        rgb = outputs[0]
                    else:
                        rgb = outputs
                        
                    # Convert [H*W, C] to [H, W] single channel
                    if rgb.dim() == 2 and rgb.shape[-1] > 1:
                        proj = rgb.mean(dim=-1).view(img_size, img_size)
                    elif rgb.dim() == 2:
                        proj = rgb.view(img_size, img_size)
                    else:
                        proj = rgb.mean(dim=-1)
                        
                    # Standardize projection size to 256x256
                    proj = F.interpolate(
                        proj.unsqueeze(0).unsqueeze(0),
                        size=(256, 256),
                        mode="bilinear",
                        align_corners=True,
                    ).squeeze(0).squeeze(0)
                        
                    results.append(proj)

                # Normalize across the full sequence to preserve inter-view relative intensity
                if results:
                    results = list(normalize_tensor(torch.stack(results, dim=0)).unbind(0))

            del generator_test
            torch.cuda.empty_cache()
            
        except Exception as e:
            traceback.print_exc()
        
        return results
