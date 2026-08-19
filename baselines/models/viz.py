"""Shared TensorBoard visualization helper for baseline training scripts."""

from __future__ import annotations

import gc
import logging
import os
import random
from typing import Callable

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.utils
from PIL import Image


# --- Logger Setup --- #
logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================

def epoch_rotating_view_indices(epoch: int, num_views: int, total_views: int = 93) -> list[int]:
    """Compute a list of view indices to render for a given epoch, rotating through total views.

    Args:
        epoch: Current training epoch.
        num_views: Number of views to render for this epoch.
        total_views: Total number of views in the sweep (default 93).

    Returns:
        Sorted list of integer view indices.
    """
    if num_views <= 0 or num_views > total_views:
        raise ValueError(f"num_views must be in [1, {total_views}], got {num_views}")

    rng = random.Random(epoch)
    phase = rng.randrange(total_views)
    step = total_views // num_views

    return sorted((phase + i * step) % total_views for i in range(num_views))


def build_multiview_grid(
    wrapper: object,
    input_xray: torch.Tensor,
    gt_views_dir: str,
    azimuth_indices: list[int],
    total_gt_views: int = 93,
) -> torch.Tensor | None:
    """Build a TensorBoard grid of predicted and ground-truth views.

    Args:
        wrapper: Model wrapper with ``infer_multi_views`` method.
        input_xray: Input X-ray tensor of shape [1, 1, H, W].
        gt_views_dir: Directory containing ground-truth views as PNGs.
        azimuth_indices: List of indices into the ground-truth view sweep to render.
        total_gt_views: Total number of ground-truth views in the sweep (default 93).

    Returns:
        A formatted grid tensor suitable for TensorBoard logging, or None if failed.
    """
    if not azimuth_indices:
        return None

    indices = sorted(azimuth_indices)
    n = len(indices)

    # Convert view indices to azimuth degrees (using endpoint-inclusive N-1 divisor)
    denom = max(1, total_gt_views - 1)
    degrees = [idx / denom * 360.0 for idx in indices]
    span = degrees[-1] - degrees[0] if n > 1 else 0.0

    # 1. Generate predicted views using model wrapper
    # Note: Do not wrap in torch.no_grad() because optimization-based models (MedNeRF, NAF)
    # require gradients during internal per-scan fitting.
    try:
        pred_views = wrapper.infer_multi_views(
            input_xray, azimuths=(degrees[0], degrees[0] + span, n)
        )
    except Exception:
        logger.exception("infer_multi_views failed while building TensorBoard grid")
        pred_views = []

    if not pred_views or len(pred_views) < n:
        logger.warning(
            f"infer_multi_views returned {len(pred_views) if pred_views else 0} views, "
            f"expected {n} -- skipping this epoch's TensorBoard image."
        )
        return None

    # 2. Load matching ground-truth views from disk
    if not os.path.isdir(gt_views_dir):
        logger.warning(f"Ground-truth views dir not found: {gt_views_dir}")
        return None

    gt_files = sorted(f for f in os.listdir(gt_views_dir) if f.endswith(".png"))
    if len(gt_files) < total_gt_views:
        logger.warning(
            f"Expected {total_gt_views} ground-truth views in {gt_views_dir}, found {len(gt_files)}"
        )
        return None

    # 3. Format predicted and ground-truth view tensors
    pred_rows = []
    gt_rows = []

    for i, idx in enumerate(indices):
        # Format predicted tensor
        pred = pred_views[i]
        if not isinstance(pred, torch.Tensor):
            pred = torch.as_tensor(pred)
        pred = pred.detach().float().cpu()
        while pred.dim() > 2:
            pred = pred.squeeze(0)
        pred_rows.append(pred.unsqueeze(0).unsqueeze(0))  # [1, 1, H, W]

        # Format ground-truth tensor
        gt_img = Image.open(os.path.join(gt_views_dir, gt_files[idx])).convert("L")
        gt_t = TF.to_tensor(gt_img)  # [1, H, W] in [0, 1]
        gt_rows.append(gt_t.unsqueeze(0))  # [1, 1, H, W]

    pred_batch = torch.cat(pred_rows, dim=0)  # [n, 1, H, W]
    gt_batch = torch.cat(gt_rows, dim=0)  # [n, 1, H, W]

    # Ensure spatial dimension consistency
    if pred_batch.shape[-2:] != gt_batch.shape[-2:]:
        gt_batch = F.interpolate(
            gt_batch, size=pred_batch.shape[-2:], mode="bilinear", align_corners=False
        )

    # 4. Construct 2-row grid (Row 1: Predictions, Row 2: Ground Truth)
    combined = torch.cat([pred_batch, gt_batch], dim=0)  # [2n, 1, H, W]
    grid = torchvision.utils.make_grid(combined, nrow=n, padding=2, pad_value=1.0)

    # Clean up GPU memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return grid


def visualize_live_checkpoint(
    wrapper_cls: type,
    save_fn: Callable[[str], None],
    tmp_ckpt_path: str,
    input_xray: torch.Tensor,
    gt_views_dir: str,
    azimuth_indices: list[int],
) -> torch.Tensor | None:
    """Bridges in-memory training weights to the standard evaluation inference pipeline.

    Saves current weights to a temporary file, instantiates the model wrapper,
    generates multi-view predictions against ground-truth views, and cleans up scratch files.

    Args:
        wrapper_cls: The baseline wrapper class (e.g. ``SVDRRWrapper``).
        save_fn: Callback ``save_fn(path)`` to persist current weights to disk.
        tmp_ckpt_path: Temporary scratch file path for the checkpoint.
        input_xray: Input radiograph tensor ``[1, 1, 256, 256]`` in ``[0, 1]``.
        gt_views_dir: Path to directory containing ground-truth view images.
        azimuth_indices: List of view indices to sample for visualization.

    Returns:
        A formatted grid tensor ``[1, H, W]`` for TensorBoard, or None if failed.
    """
    save_fn(tmp_ckpt_path)
    wrapper = None

    try:
        wrapper = wrapper_cls(checkpoint_path=tmp_ckpt_path)
        grid = build_multiview_grid(wrapper, input_xray, gt_views_dir, azimuth_indices)
    except Exception:
        logger.exception("Visualize live checkpoint failed")
        grid = None
    finally:
        # Clean up temporary checkpoint and release memory
        if os.path.exists(tmp_ckpt_path):
            os.remove(tmp_ckpt_path)

        del wrapper
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return grid
