"""PixelNeRF (CVPR 2021) Training Script for CosmosXRay360

Trains PixelNeRF's encoder + coarse/fine MLPs end-to-end via a photometric NeRF
reconstruction loss: encode the frontal (PA) view, ray-march + alpha-composite a
render at the lateral (LAT) view's pose (~90 degrees around the same turntable orbit
the wrapper renders on), and minimize the error against the real LAT image. Unlike
MedNeRF/XraySyn, PixelNeRF is a feed-forward, generalizable method (no per-patient
test-time fitting), which is exactly what this project's train/test split is set up
for, so this loss directly trains the model on the task it's evaluated on.

This replaces a prior version that only ran ``model.encoder(...)`` on both PA and LAT
images and minimized MSE between the two raw feature maps -- never calling the NeRF
MLPs, never ray marching, and not training anything resembling novel-view synthesis.

Saves checkpoints to baselines/checkpoints/ (both best and latest).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import torchvision.transforms.functional as TF

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
pixelnerf_dir = os.path.join(BASE_DIR, "cloned", "pixel-nerf")
sys.path.insert(0, os.path.join(pixelnerf_dir, "src"))

from models.pixelnerf import build_model_and_renderer, encode_source_view, render_view
from models.utils import get_train_val_patient_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
)
logger = logging.getLogger(__name__)

# Standard PA/lateral chest X-ray acquisition geometry: the lateral view is
# approximately a 90-degree azimuth rotation from the frontal (PA) view.
LATERAL_AZIMUTH_DEG = 90.0
RENDER_RES = 64


def train_pixelnerf(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    model, render_wrapper = build_model_and_renderer(device, simple_output=False)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    rendered_dir = os.path.join(BASE_DIR, "..", "datasets", "pre_rendered")
    train_patient_dirs, val_patient_dirs = get_train_val_patient_dirs(rendered_dir)

    logger.info(f"Found {len(train_patient_dirs)} train cases and {len(val_patient_dirs)} val cases for PixelNeRF.")

    # Pre-cache image tensors in RAM (PA at full res for encoding, LAT downsampled to
    # the render resolution to match the renderer's output grid).
    def cache_tensors(patient_dirs: list[str]) -> list[tuple[torch.Tensor, torch.Tensor]]:
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

    train_cached_tensors = cache_tensors(
        train_patient_dirs[:args.max_train_samples] if args.max_train_samples else train_patient_dirs
    )
    val_cached_tensors = cache_tensors(
        val_patient_dirs[:args.max_val_samples] if args.max_val_samples else val_patient_dirs
    )

    ckpt_dir = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    latest_ckpt_path = os.path.join(ckpt_dir, "pixelnerf_checkpoint.pt")
    best_ckpt_path = os.path.join(ckpt_dir, "pixelnerf_best.pt")
    writer = SummaryWriter(log_dir=os.path.join(ckpt_dir, "tensorboard", "pixelnerf"))

    logger.info(f"Starting PixelNeRF training for {args.epochs} epochs...")

    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # 1. Training Phase
        model.train()
        # Accumulated as a device tensor (not `.item()`-ed per step) to avoid a
        # CUDA sync every step; `.item()` is called once, after the loop.
        epoch_train_loss = torch.zeros((), device=device)
        num_train_steps = 0

        for pa_tensor, lat_tensor in train_cached_tensors:
            pa_tensor, lat_tensor = pa_tensor.to(device), lat_tensor.to(device)
            optimizer.zero_grad(set_to_none=True)

            encode_source_view(model, pa_tensor, device)
            rgb_coarse, rgb_fine = render_view(
                render_wrapper, LATERAL_AZIMUTH_DEG, RENDER_RES, device, return_coarse=True
            )
            pred_coarse = rgb_coarse.permute(2, 0, 1).unsqueeze(0)  # (1, 3, RENDER_RES, RENDER_RES)
            pred_fine = rgb_fine.permute(2, 0, 1).unsqueeze(0)      # (1, 3, RENDER_RES, RENDER_RES)
            loss = criterion(pred_coarse, lat_tensor) + criterion(pred_fine, lat_tensor)

            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.detach()
            num_train_steps += 1

        avg_train_loss = (epoch_train_loss / max(1, num_train_steps)).item()

        # 2. Validation Phase
        model.eval()
        epoch_val_loss = torch.zeros((), device=device)
        num_val_steps = 0
        val_sample_img = None

        with torch.no_grad():
            for idx, (pa_tensor, lat_tensor) in enumerate(val_cached_tensors):
                pa_tensor, lat_tensor = pa_tensor.to(device), lat_tensor.to(device)
                encode_source_view(model, pa_tensor, device)
                rgb_fine = render_view(
                    render_wrapper, LATERAL_AZIMUTH_DEG, RENDER_RES, device, return_coarse=False
                )
                pred_img = rgb_fine.permute(2, 0, 1).unsqueeze(0)
                loss = criterion(pred_img, lat_tensor)

                if idx == 0:
                    val_sample_img = torch.cat([pred_img[0].clamp(0, 1), lat_tensor[0].clamp(0, 1)], dim=-1).cpu()

                epoch_val_loss += loss.detach()
                num_val_steps += 1

        avg_val_loss = (epoch_val_loss / max(1, num_val_steps)).item()
        logger.info(f"Epoch [{epoch}/{args.epochs}] - Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        if val_sample_img is not None:
            writer.add_image("Images/val_pred_gt", val_sample_img, epoch)

        # Track best model in memory
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info(f"New best model recorded (Val Loss: {best_loss:.6f})")

    # Save best and latest checkpoints at the end of training
    if 'best_state_dict' in locals():
        torch.save(best_state_dict, best_ckpt_path)
    else:
        torch.save(model.state_dict(), best_ckpt_path)
    torch.save(model.state_dict(), latest_ckpt_path)
    writer.close()
    logger.info(f"PixelNeRF training complete. Checkpoints saved to {latest_ckpt_path} and {best_ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PixelNeRF Baseline Trainer")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Max training samples (default: None for full train set)")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Max validation samples (default: None for full val)")
    args = parser.parse_args()
    train_pixelnerf(args)
