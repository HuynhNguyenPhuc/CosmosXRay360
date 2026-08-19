"""PixelNeRF (CVPR 2021) Training Script."""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
pixelnerf_dir = os.path.join(BASE_DIR, "cloned", "pixel-nerf")
sys.path.insert(0, os.path.join(pixelnerf_dir, "src"))

from models.pixelnerf import PixelNeRFWrapper, build_model_and_renderer, encode_source_view, render_view
from models.utils import get_train_val_patient_dirs
from models.viz import epoch_rotating_view_indices, visualize_live_checkpoint


# --- Logger Setup --- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
)
logger = logging.getLogger(__name__)


# --- Configuration --- #
LATERAL_AZIMUTH_DEG = 90.0
RENDER_RES = 64


# =============================================================================
# Main Training Function
# =============================================================================

def train_pixelnerf(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # 1. Build PixelNeRF Model & Renderer
    model, render_wrapper = build_model_and_renderer(device, simple_output=False)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # 2. Load Dataset Directories & Pre-cache Tensors
    rendered_dir = os.path.join(BASE_DIR, "..", "datasets", "pre_rendered")
    train_patient_dirs, val_patient_dirs = get_train_val_patient_dirs(rendered_dir)

    logger.info(f"Found {len(train_patient_dirs)} train cases and {len(val_patient_dirs)} val cases for PixelNeRF.")

    def cache_train_tensors(patient_dirs: list[str]) -> list[dict]:
        cached = []

        for pat_path in patient_dirs:
            pa_file = os.path.join(pat_path, "pa.png")
            views_dir = os.path.join(pat_path, "views")

            if not (os.path.exists(pa_file) and os.path.exists(views_dir)):
                continue

            view_files = sorted(f for f in os.listdir(views_dir) if f.endswith(".png"))
            if len(view_files) < 2:
                continue

            pa_tensor = TF.to_tensor(Image.open(pa_file).convert("RGB")).unsqueeze(0)
            view_tensors = torch.stack([
                TF.resize(
                    TF.to_tensor(Image.open(os.path.join(views_dir, vf)).convert("RGB")),
                    [RENDER_RES, RENDER_RES],
                )
                for vf in view_files
            ])

            angles = np.linspace(0.0, 360.0, len(view_files))
            cached.append({"pa": pa_tensor, "views": view_tensors, "angles": angles})

        return cached

    def cache_val_tensors(patient_dirs: list[str]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        cached = []

        for pat_path in patient_dirs:
            pa_file = os.path.join(pat_path, "pa.png")
            lat_file = os.path.join(pat_path, "lat.png")

            if not (os.path.exists(pa_file) and os.path.exists(lat_file)):
                continue

            pa_tensor = TF.to_tensor(Image.open(pa_file).convert("RGB")).unsqueeze(0)
            lat_tensor = TF.to_tensor(Image.open(lat_file).convert("RGB")).unsqueeze(0)

            lat_tensor = TF.resize(lat_tensor, [RENDER_RES, RENDER_RES])
            cached.append((pa_tensor, lat_tensor))

        return cached

    def sample_train_pair(item: dict) -> tuple[torch.Tensor, float, torch.Tensor, float]:
        """Sample a random (source_view, source_azimuth, target_view, target_azimuth) pair."""
        n_views = item["views"].shape[0]
        src_idx, tgt_idx = np.random.choice(n_views, size=2, replace=False)
        src_img = item["views"][src_idx].unsqueeze(0)
        src_az = float(item["angles"][src_idx])
        tgt_img = item["views"][tgt_idx].unsqueeze(0)
        tgt_az = float(item["angles"][tgt_idx])
        return src_img, src_az, tgt_img, tgt_az

    train_cached_tensors = cache_train_tensors(
        train_patient_dirs[:args.max_train_samples] if args.max_train_samples else train_patient_dirs
    )

    val_cached_tensors = cache_val_tensors(
        val_patient_dirs[:args.max_val_samples] if args.max_val_samples else val_patient_dirs
    )

    viz_patient_dir = val_patient_dirs[0] if val_patient_dirs else None

    # 3. Setup Checkpoint Paths & TensorBoard Writer
    ckpt_dir = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    latest_ckpt_path = os.path.join(ckpt_dir, "pixelnerf_checkpoint.pt")
    best_ckpt_path = os.path.join(ckpt_dir, "pixelnerf_best.pt")

    writer = SummaryWriter(log_dir=os.path.join(ckpt_dir, "tensorboard", "pixelnerf"))

    logger.info(f"Starting PixelNeRF training for {args.epochs} epochs...")
    best_loss = float("inf")

    # 4. Main Training Loop
    for epoch in range(1, args.epochs + 1):

        # --- Training Phase --- #
        model.train()
        epoch_train_loss = torch.zeros((), device=device)
        num_train_steps = 0

        for item in train_cached_tensors:
            src_tensor, src_azimuth, target_tensor, target_azimuth = sample_train_pair(item)
            src_tensor = src_tensor.to(device)
            target_tensor = target_tensor.to(device)

            optimizer.zero_grad(set_to_none=True)

            encode_source_view(model, src_tensor, device, source_azimuth=src_azimuth)

            rgb_coarse, rgb_fine = render_view(
                render_wrapper, target_azimuth, RENDER_RES, device, return_coarse=True
            )

            pred_coarse = rgb_coarse.permute(2, 0, 1).unsqueeze(0)
            pred_fine = rgb_fine.permute(2, 0, 1).unsqueeze(0)

            loss = criterion(pred_coarse, target_tensor) + criterion(pred_fine, target_tensor)

            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.detach()
            num_train_steps += 1

        avg_train_loss = (epoch_train_loss / max(1, num_train_steps)).item()

        # --- Validation Phase --- #
        model.eval()
        epoch_val_loss = torch.zeros((), device=device)
        num_val_steps = 0

        with torch.no_grad():
            for pa_tensor, lat_tensor in val_cached_tensors:
                pa_tensor, lat_tensor = pa_tensor.to(device), lat_tensor.to(device)

                encode_source_view(model, pa_tensor, device)

                rgb_fine = render_view(
                    render_wrapper, LATERAL_AZIMUTH_DEG, RENDER_RES, device, return_coarse=False
                )

                pred_img = rgb_fine.permute(2, 0, 1).unsqueeze(0)
                loss = criterion(pred_img, lat_tensor)

                epoch_val_loss += loss.detach()
                num_val_steps += 1

        avg_val_loss = (epoch_val_loss / max(1, num_val_steps)).item()

        # --- Logging & Multi-View Panel --- #
        logger.info(f"Epoch [{epoch}/{args.epochs}] - Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)

        if args.viz_every > 0 and (epoch % args.viz_every == 0 or epoch == args.epochs) and viz_patient_dir is not None:
            pa_path = os.path.join(viz_patient_dir, "pa.png")
            gt_views_dir = os.path.join(viz_patient_dir, "views")

            if os.path.exists(pa_path):
                indices = epoch_rotating_view_indices(epoch, args.viz_views)

                def _save_viz_ckpt(path: str) -> None:
                    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)

                tmp_ckpt_path = os.path.join(ckpt_dir, f"_viz_tmp_pixelnerf_{os.getpid()}.pt")
                input_xray = TF.to_tensor(Image.open(pa_path).convert("L")).unsqueeze(0).to(device)

                multiview_grid = visualize_live_checkpoint(
                    PixelNeRFWrapper, _save_viz_ckpt, tmp_ckpt_path, input_xray, gt_views_dir, indices,
                )

                if multiview_grid is not None:
                    writer.add_image("Images/val_multiview", multiview_grid, epoch)

                model.train()

        # --- Checkpoint Saving --- #
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info(f"New best model recorded (Val Loss: {best_loss:.6f})")

    if 'best_state_dict' in locals():
        torch.save(best_state_dict, best_ckpt_path)
    else:
        torch.save(model.state_dict(), best_ckpt_path)

    torch.save(model.state_dict(), latest_ckpt_path)
    writer.close()
    logger.info(f"PixelNeRF training complete. Checkpoints saved to {latest_ckpt_path} and {best_ckpt_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PixelNeRF Baseline Trainer")

    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Max training samples")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Max validation samples")
    parser.add_argument("--viz_every", type=int, default=10, help="Log multi-view TensorBoard panel every N epochs (0 to disable)")
    parser.add_argument("--viz_views", type=int, default=6, help="Number of azimuths per multi-view panel")

    args = parser.parse_args()
    train_pixelnerf(args)
