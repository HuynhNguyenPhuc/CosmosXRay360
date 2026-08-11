"""XRaySyn (AAAI 2021) Training Script."""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
xraysyn_dir = os.path.join(BASE_DIR, "cloned", "XraySyn")
sys.path.insert(0, xraysyn_dir)
sys.path.insert(0, os.path.join(xraysyn_dir, "xraysyn", "networks", "drr_projector"))

from xraysyn.models.ct2xray_real_gan_meta import XraySynModel  # type: ignore

from models.utils import get_train_val_patient_dirs
from models.viz import epoch_rotating_view_indices, visualize_live_checkpoint
from models.xraysyn import XRaySynWrapper, _get_T_batched


# --- Logger Setup --- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
)
logger = logging.getLogger(__name__)


# --- Configuration --- #
OTHER_POSE_THETA_Y_RANGE = (-1.0, 1.0)


# =============================================================================
# Utility Functions
# =============================================================================

def load_pa_tensor(pat_path: str) -> torch.Tensor | None:
    """Load a patient's frontal (PA) view as a tensor [1, 1, 256, 256] in [0, 1]."""
    pa_file = os.path.join(pat_path, "pa.png")
    if not os.path.exists(pa_file):
        return None

    return TF.to_tensor(Image.open(pa_file).convert("L")).unsqueeze(0)


@torch.no_grad()
def reconstruction_l1(model: XraySynModel, xray: torch.Tensor, return_image: bool = False):
    """Compute self-supervised reconstruction error at input pose."""
    batch_size = xray.shape[0]

    T_in = _get_T_batched(model, [1, 0, 0, 0, 0, 0], batch_size)
    xray128 = model.avgpool(xray)

    vol_in = model.backproj(xray128, T_in)
    vol_pred_temp = model.net3d(vol_in) * 0.5 + 0.5

    bone_mask = vol_pred_temp[:, [0]]
    bone_ct = vol_pred_temp[:, [1]] * bone_mask
    tissue_ct = vol_pred_temp[:, [2]] * (1 - bone_mask)
    vol_pred = bone_ct + tissue_ct

    _, mat_pred = model.ct2xray(vol_pred, bone_mask, T_in)
    mat_refine = model.net2d(mat_pred, xray) + model.upsample(mat_pred)

    xray_refine = model.mat2xray(mat_refine)
    l1 = torch.nn.functional.l1_loss(xray_refine, xray).item()

    if return_image:
        pred_x = xray_refine[0].clamp(0, 1)
        targ_x = xray[0].clamp(0, 1)

        if pred_x.shape[0] == 1:
            pred_x = pred_x.repeat(3, 1, 1)
            targ_x = targ_x.repeat(3, 1, 1)

        comp_img = torch.cat([pred_x, targ_x], dim=-1).cpu()
        return l1, comp_img

    return l1


# =============================================================================
# Main Training Function
# =============================================================================

def train_xraysyn(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # 1. Initialize XRaySyn Model
    old_cwd = os.getcwd()
    os.chdir(xraysyn_dir)

    try:
        model = XraySynModel(lr=args.lr, device=device)
    finally:
        os.chdir(old_cwd)

    # 2. Load & Cache Dataset Tensors
    rendered_dir = os.path.join(BASE_DIR, "..", "datasets", "pre_rendered")
    train_patient_dirs, val_patient_dirs = get_train_val_patient_dirs(rendered_dir)

    logger.info(f"Found {len(train_patient_dirs)} train cases and {len(val_patient_dirs)} val cases for XRaySyn.")

    def cache_tensors(patient_dirs: list[str]) -> list[torch.Tensor]:
        cached = []
        for pat_path in patient_dirs:
            pa_tensor = load_pa_tensor(pat_path)
            if pa_tensor is not None:
                cached.append(pa_tensor)
        return cached

    train_cached_tensors = cache_tensors(
        train_patient_dirs[:args.max_train_samples] if args.max_train_samples else train_patient_dirs
    )
    val_cached_tensors = cache_tensors(
        val_patient_dirs[:args.max_val_samples] if args.max_val_samples else val_patient_dirs
    )

    viz_patient_dir = val_patient_dirs[0] if val_patient_dirs else None

    # 3. Setup Checkpoint Paths & TensorBoard
    ckpt_dir = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    latest_ckpt_path = os.path.join(ckpt_dir, "xraysyn_checkpoint.pt")
    best_ckpt_path = os.path.join(ckpt_dir, "xraysyn_best.pt")

    writer = SummaryWriter(log_dir=os.path.join(ckpt_dir, "tensorboard", "xraysyn"))

    logger.info(f"Starting XRaySyn training for {args.epochs} epochs...")

    best_loss = float("inf")
    rng = random.Random(42)

    # 4. Main Training Loop
    for epoch in range(1, args.epochs + 1):

        # --- Training Phase --- #
        epoch_train_l1 = 0.0
        num_train_steps = 0

        for pa_tensor in train_cached_tensors:
            xray = pa_tensor.to(device)
            batch_size = xray.shape[0]

            T_in = _get_T_batched(model, [1, 0, 0, 0, 0, 0], batch_size)
            theta_y = rng.uniform(*OTHER_POSE_THETA_Y_RANGE)
            T_other = _get_T_batched(model, [1, theta_y, 0, 0, 0, 0], batch_size)

            model.set_input(xray, T_in, T_other)
            model.optimize()

            epoch_train_l1 += model.get_loss().get("L1", 0.0)
            num_train_steps += 1

        avg_train_l1 = epoch_train_l1 / max(1, num_train_steps)

        # --- Validation Phase --- #
        epoch_val_l1 = 0.0
        num_val_steps = 0

        for pa_tensor in val_cached_tensors:
            xray = pa_tensor.to(device)

            l1_val = reconstruction_l1(model, xray)
            epoch_val_l1 += l1_val
            num_val_steps += 1

        avg_val_l1 = epoch_val_l1 / max(1, num_val_steps)

        # --- Logging & TensorBoard --- #
        logger.info(f"Epoch [{epoch}/{args.epochs}] - Train L1 (running): {avg_train_l1:.6f} | Val L1: {avg_val_l1:.6f}")
        writer.add_scalar("L1/train", avg_train_l1, epoch)
        writer.add_scalar("L1/val", avg_val_l1, epoch)

        if args.viz_every > 0 and (epoch % args.viz_every == 0 or epoch == args.epochs) and viz_patient_dir is not None:
            pa_path = os.path.join(viz_patient_dir, "pa.png")
            gt_views_dir = os.path.join(viz_patient_dir, "views")

            if os.path.exists(pa_path):
                indices = epoch_rotating_view_indices(epoch, args.viz_views)
                tmp_ckpt_path = os.path.join(ckpt_dir, f"_viz_tmp_xraysyn_{os.getpid()}.pt")
                input_xray = TF.to_tensor(Image.open(pa_path).convert("L")).unsqueeze(0).to(device)

                multiview_grid = visualize_live_checkpoint(
                    XRaySynWrapper, model.save, tmp_ckpt_path, input_xray, gt_views_dir, indices,
                )

                if multiview_grid is not None:
                    writer.add_image("Images/val_multiview", multiview_grid, epoch)

        # --- Checkpoint Saving --- #
        if avg_val_l1 < best_loss:
            best_loss = avg_val_l1
            model.save(best_ckpt_path)
            logger.info(f"New best model recorded (Val L1: {best_loss:.6f})")

    model.save(latest_ckpt_path)
    writer.close()
    logger.info(f"XRaySyn training complete. Checkpoints saved to {latest_ckpt_path} and {best_ckpt_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XRaySyn Baseline Trainer")

    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Max training samples")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Max validation samples")
    parser.add_argument("--viz_every", type=int, default=10, help="Log multi-view TensorBoard panel every N epochs (0 to disable)")
    parser.add_argument("--viz_views", type=int, default=6, help="Number of azimuths per multi-view panel")

    args = parser.parse_args()
    train_xraysyn(args)
