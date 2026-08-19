"""Shared utilities for baseline model evaluation."""

from __future__ import annotations

import logging
import math
import os
import random
from typing import Callable
import warnings

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available")


# --- Logger --- #
logger = logging.getLogger(__name__)


def compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Computes Peak Signal-to-Noise Ratio (PSNR) between prediction and ground-truth.

    Args:
        pred: A [B, C, H, W] predicted tensor clamped to [0, 1].
        gt: A [B, C, H, W] ground-truth tensor clamped to [0, 1].

    Returns:
        The computed PSNR value as a float. Returns 100.0 if MSE is 0.
    """
    if not torch.is_tensor(pred) or not torch.is_tensor(gt):
        return 0.0
    
    pred = pred.clamp(0, 1)
    gt = gt.clamp(0, 1)
    mse = torch.mean((pred - gt) ** 2)
    if mse == 0:
        return 100.0
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()


def compute_ssim(pred: torch.Tensor, gt: torch.Tensor, win_size: int = 11) -> float:
    """Computes Structural Similarity Index (SSIM) between prediction and ground-truth.

    Args:
        pred: A [B, C, H, W] predicted tensor clamped to [0, 1].
        gt: A [B, C, H, W] ground-truth tensor clamped to [0, 1].
        win_size: Window size for average pooling (default: 11).

    Returns:
        The computed average SSIM value as a float.
    """
    if not torch.is_tensor(pred) or not torch.is_tensor(gt):
        return 0.0
    
    pred = pred.clamp(0, 1)
    gt = gt.clamp(0, 1)
    
    # Ensure 4D tensors for SSIM computation
    if pred.dim() == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
    if gt.dim() == 2:
        gt = gt.unsqueeze(0).unsqueeze(0)
    
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    
    mu_x = F.avg_pool2d(pred, win_size, 1, win_size // 2)
    mu_y = F.avg_pool2d(gt, win_size, 1, win_size // 2)
    
    sigma_x = F.avg_pool2d(pred ** 2, win_size, 1, win_size // 2) - mu_x ** 2
    sigma_y = F.avg_pool2d(gt ** 2, win_size, 1, win_size // 2) - mu_y ** 2
    sigma_xy = F.avg_pool2d(pred * gt, win_size, 1, win_size // 2) - mu_x * mu_y
    
    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))

    return ssim.mean().item()


_LPIPS_LOSS: "torch.nn.Module | None" = None


def _get_lpips_loss(device: str) -> "torch.nn.Module":
    """Lazily constructs and caches a process-wide LPIPS network.

    Mirrors ``mednerf.py``'s ``_get_perceptual_loss`` singleton pattern: the AlexNet
    backbone is expensive to load and never changes between calls, so it's built once
    per process rather than once per ``compute_lpips`` call. Caveat inherited from that
    same pattern: the cached network stays on whichever device it was first built for,
    so mixing devices across an evaluation run isn't supported.
    """
    global _LPIPS_LOSS
    if _LPIPS_LOSS is None:
        import lpips
        _LPIPS_LOSS = lpips.LPIPS(net="alex", verbose=False).to(device)
        _LPIPS_LOSS.eval()
    return _LPIPS_LOSS


def compute_lpips(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Computes LPIPS (Learned Perceptual Image Patch Similarity) between prediction and ground-truth.

    Unlike PSNR/SSIM's pixelwise comparison, LPIPS scores similarity in a pretrained
    AlexNet's feature space -- lower is more similar (0.0 = identical, no fixed upper
    bound). This penalizes a blurry-but-pixelwise-safe prediction more harshly, and a
    sharp-but-slightly-misaligned one less harshly, than PSNR/SSIM alone -- see
    docs/BENCHMARK.md's note on why this project reports it alongside PSNR/SSIM.

    Args:
        pred: A [B, C, H, W] predicted tensor clamped to [0, 1]. Single-channel (C=1)
            input is repeated to 3 channels internally, since LPIPS' AlexNet backbone
            expects RGB.
        gt: A [B, C, H, W] ground-truth tensor clamped to [0, 1], same shape as `pred`.

    Returns:
        The computed LPIPS distance as a float. Returns 0.0 if inputs aren't tensors.
    """
    if not torch.is_tensor(pred) or not torch.is_tensor(gt):
        return 0.0

    pred = pred.clamp(0, 1)
    gt = gt.clamp(0, 1)

    if pred.dim() == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
    if gt.dim() == 2:
        gt = gt.unsqueeze(0).unsqueeze(0)

    if pred.shape[1] == 1:
        pred = pred.repeat(1, 3, 1, 1)
    if gt.shape[1] == 1:
        gt = gt.repeat(1, 3, 1, 1)

    loss_fn = _get_lpips_loss(str(pred.device))
    with torch.no_grad():
        # normalize=True: inputs are treated as already being in [0, 1] (rescaled to
        # [-1, 1] internally by LPIPS), matching this project's [0, 1] tensor convention.
        dist = loss_fn(pred, gt, normalize=True)
    return dist.mean().item()


def get_train_val_patient_dirs(
    rendered_dir: str,
    val_ratio: float = 0.15,
    seed: int = 42,
    exclude_patients: set[str] | list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Retrieves patient directories from rendered_dir.
    
    If rendered_dir/val is empty or missing, carves out a deterministic
    val_ratio split (default: 15%) from rendered_dir/train using a fixed random seed.

    Args:
        rendered_dir: Path to rendered directory containing train/ or val/ subdirectories.
        val_ratio: Fraction of train data to carve out for val split if val/ missing.
        seed: Random seed for split shuffling.
        exclude_patients: List/set of patient directory basenames to exclude (e.g. corrupted scans).
            Defaults to {"mela_0005"}.
    """
    if exclude_patients is None:
        exclude_patients = {"mela_0005"}
    else:
        exclude_patients = set(exclude_patients)

    train_dir = os.path.join(rendered_dir, "train")
    val_dir = os.path.join(rendered_dir, "val")

    val_patient_dirs = (
        sorted([
            os.path.join(val_dir, d)
            for d in os.listdir(val_dir)
            if os.path.isdir(os.path.join(val_dir, d)) and d not in exclude_patients
        ])
        if os.path.exists(val_dir)
        else []
    )
    train_patient_dirs = (
        sorted([
            os.path.join(train_dir, d)
            for d in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, d)) and d not in exclude_patients
        ])
        if os.path.exists(train_dir)
        else []
    )

    if not val_patient_dirs and train_patient_dirs:
        rng = random.Random(seed)
        shuffled = train_patient_dirs.copy()
        rng.shuffle(shuffled)
        num_val = max(1, int(len(shuffled) * val_ratio))
        val_patient_dirs = sorted(shuffled[:num_val])
        train_patient_dirs = sorted(shuffled[num_val:])

    return train_patient_dirs, val_patient_dirs



def apply_beer_lambert_correction(density: torch.Tensor, I0: float = 1.0) -> torch.Tensor:
    """Applies Beer-Lambert Law correction to map linear attenuation density to transmissivity.

    I = I_0 * exp(-integral(mu * ds))

    Args:
        density: Input density/attenuation tensor.
        I0: Incident source intensity (default: 1.0).

    Returns:
        The computed transmissive intensity tensor.
    """
    density = torch.clamp(density, min=0.0, max=50.0)
    return I0 * torch.exp(-density)


def normalize_tensor(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Min-max normalizes tensor to [0, 1] range matching target image intensity domain.

    Args:
        tensor: Input image or projection tensor.
        eps: Small epsilon for numerical stability.

    Returns:
        Tensor min-max normalized to [0, 1].
    """
    min_v = tensor.min()
    max_v = tensor.max()
    if max_v - min_v < eps:
        return torch.zeros_like(tensor)
    return (tensor - min_v) / (max_v - min_v + eps)


def build_perspective_ray_points(
    azimuth: float,
    grid_res: int,
    device: str,
    dist: float = 8.0,
    elev: float = 0.0,
    fov: float = 12.0,
    min_depth: float = 7.0,
    max_depth: float = 9.0,
    n_samples: int = 192,
    bound: float | None = None,
) -> torch.Tensor:
    """Builds world-space ray sample point grid for camera ray marching."""
    elev_rad = math.radians(elev)
    azim_rad = math.radians(azimuth)
    cam_pos = torch.tensor(
        [
            dist * math.cos(elev_rad) * math.sin(azim_rad),
            dist * math.sin(elev_rad),
            dist * math.cos(elev_rad) * math.cos(azim_rad),
        ],
        device=device, dtype=torch.float32,
    )

    up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32)
    z_axis = F.normalize(-cam_pos, dim=0, eps=1e-5)
    x_axis = F.normalize(torch.cross(up, z_axis, dim=0), dim=0, eps=1e-5)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=0), dim=0, eps=1e-5)
    R = torch.stack([x_axis, y_axis, z_axis], dim=1)

    half_extent = math.tan(math.radians(fov) / 2.0)
    px = torch.linspace(-half_extent, half_extent, grid_res, device=device)
    py = torch.linspace(half_extent, -half_extent, grid_res, device=device)
    grid_y, grid_x = torch.meshgrid(py, px, indexing="ij")
    dirs_cam = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    dirs_world = F.normalize(dirs_cam @ R.T, dim=-1)

    t_vals = torch.linspace(0.0, 1.0, n_samples, device=device)
    depths = min_depth * (1.0 - t_vals) + max_depth * t_vals

    pts = cam_pos.view(1, 1, 3) + dirs_world.unsqueeze(1) * depths.view(1, n_samples, 1)
    if bound is not None:
        pts = pts.clamp(-bound, bound)
    return pts


def perspective_ray_march(
    sample_fn: "Callable[[torch.Tensor], torch.Tensor]",
    azimuth: float,
    grid_res: int,
    device: str,
    dist: float = 8.0,
    elev: float = 0.0,
    fov: float = 12.0,
    min_depth: float = 7.0,
    max_depth: float = 9.0,
    n_samples: int = 192,
    bound: float | None = None,
    reduction: str = "attenuation_sum",
    pts: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Ray-marches through implicit/voxel field to produce 2D projections.

    Args:
        sample_fn: Query function mapping [N, 3] points to [N, 1] density.
        azimuth: Camera azimuth in degrees.
        grid_res: Output projection resolution.
        device: Compute device.
        dist, elev, fov, min_depth, max_depth: Camera geometry parameters.
        n_samples: Number of ray samples.
        bound: Spatial coordinate bounding box constraint.
        reduction: "attenuation_sum" (Beer-Lambert sum) or "mean" (sample average).
        pts: Optional precomputed ray sample points.

    Returns:
        Rendered 2D projection tensor [grid_res, grid_res].
    """
    if reduction not in ("attenuation_sum", "mean"):
        raise ValueError(f"Unknown reduction {reduction!r}, expected 'attenuation_sum' or 'mean'.")
    if pts is None:
        pts = build_perspective_ray_points(
            azimuth, grid_res, device, dist=dist, elev=elev, fov=fov,
            min_depth=min_depth, max_depth=max_depth, n_samples=n_samples, bound=bound,
        )
    step_size = (max_depth - min_depth) / (n_samples - 1)
    n_rays = grid_res * grid_res

    density = sample_fn(pts.reshape(-1, 3)).reshape(n_rays, n_samples)

    if reduction == "attenuation_sum":
        # Distance-weighted Riemann-sum line integral, matching reference NAF's
        # raw2outputs: acc = sum((raw + noise) * dists, dim=-1).
        line_integral = (density * step_size).sum(dim=-1)
        proj = apply_beer_lambert_correction(line_integral)
    else:
        proj = density.mean(dim=-1)

    return proj.reshape(grid_res, grid_res)

