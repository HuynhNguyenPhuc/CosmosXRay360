"""MedNeRF (EMBC 2022) Training Script."""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
mednerf_dir = os.path.join(BASE_DIR, "cloned", "mednerf", "graf-main")
sys.path.insert(0, mednerf_dir)
sys.path.insert(0, os.path.join(mednerf_dir, "submodules"))

from graf.config import build_models  # type: ignore
from graf.utils import to_theta  # type: ignore
from submodules.GAN_stability.gan_training.checkpoints import CheckpointIO  # type: ignore
from submodules.GAN_stability.gan_training.config import load_config  # type: ignore

from models.mednerf import fit_latent_and_weights, MedNeRFWrapper
from models.utils import get_train_val_patient_dirs
from models.viz import epoch_rotating_view_indices, visualize_live_checkpoint


# --- Logger Setup --- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Main Training Function
# =============================================================================

def train_mednerf(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    if args.val_iters_per_sample is None:
        args.val_iters_per_sample = max(1, args.iters_per_sample // 4)

    logger.info(
        f"Per-scan fitting iterations: train={args.iters_per_sample}, "
        f"val={args.val_iters_per_sample}"
    )

    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # 1. Load Configurations & Geometry Parameters
    config_file = load_config(
        os.path.join(mednerf_dir, "configs/chest.yaml"),
        os.path.join(mednerf_dir, "configs/default.yaml"),
    )

    H = W = config_file["data"]["imsize"]
    fov = config_file["data"]["fov"]
    focal = W / 2.0 * 1.0 / np.tan((0.5 * fov * np.pi / 180.0))

    radius = config_file["data"]["radius"]
    if isinstance(radius, str):
        radius = tuple(float(r) for r in radius.split(","))
        radius = max(radius)

    config_file["data"]["hwfr"] = [H, W, focal, radius]
    z_dim = config_file["z_dist"]["dim"]

    theta_mean = 0.5 * (to_theta(config_file["data"]["vmin"]) + to_theta(config_file["data"]["vmax"]))

    # 2. Build MedNeRF Generator Model
    generator, _ = build_models(config_file, disc=False)
    generator = generator.to(device)
    generator.chunk = 16384

    # 3. Load & Pre-cache Dataset Images
    rendered_dir = os.path.join(BASE_DIR, "..", "datasets", "pre_rendered")
    train_patient_dirs, val_patient_dirs = get_train_val_patient_dirs(rendered_dir)

    logger.info(f"Found {len(train_patient_dirs)} train cases and {len(val_patient_dirs)} val cases for MedNeRF.")

    def cache_target_tensors(patient_dirs: list[str]) -> list[dict]:
        cached = []

        for pat_path in patient_dirs:
            views_dir = os.path.join(pat_path, "views")
            if not os.path.exists(views_dir):
                pa_file = os.path.join(pat_path, "pa.png")
                if os.path.exists(pa_file):
                    pa_tensor = TF.to_tensor(Image.open(pa_file).convert("L")).unsqueeze(0)
                    pa_tensor = TF.resize(pa_tensor, [H, W])
                    cached.append({"tensors": [pa_tensor], "angles": [0.0]})
                continue

            view_files = sorted(f for f in os.listdir(views_dir) if f.endswith(".png"))
            if not view_files:
                continue

            tensors = [
                TF.resize(TF.to_tensor(Image.open(os.path.join(views_dir, vf)).convert("L")).unsqueeze(0), [H, W])
                for vf in view_files
            ]
            angles = list(np.linspace(0.0, 360.0, len(view_files)))
            cached.append({"tensors": tensors, "angles": angles})

        return cached

    train_cached_tensors = cache_target_tensors(
        train_patient_dirs[:args.max_train_samples] if args.max_train_samples else train_patient_dirs
    )

    val_cached_tensors = cache_target_tensors(val_patient_dirs)
    if args.max_val_samples is not None and len(val_cached_tensors) > args.max_val_samples:
        val_cached_tensors = val_cached_tensors[:args.max_val_samples]
        logger.info(f"Subsampled val set to {len(val_cached_tensors)} cases for fast epoch evaluation.")

    # Fixed validation pairs with reproducible seed
    val_rng = np.random.RandomState(42)
    val_pairs = []
    for item in val_cached_tensors:
        n_views = len(item["tensors"])
        tgt_idx = val_rng.randint(1, n_views) if n_views > 1 else 0
        val_pairs.append((item["tensors"][tgt_idx], float(item["angles"][tgt_idx])))

    viz_patient_dir = val_patient_dirs[0] if val_patient_dirs else None

    # 4. Checkpoint & Logging Setup
    ckpt_dir = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_io = CheckpointIO(ckpt_dir)
    ckpt_io.register_modules(**{k + "_test": v for k, v in generator.module_dict.items()})

    writer = SummaryWriter(log_dir=os.path.join(ckpt_dir, "tensorboard", "mednerf"))

    logger.info(f"Starting MedNeRF training for {args.epochs} epochs...")
    best_loss = float("inf")

    # 5. Main Training Loop
    for epoch in range(1, args.epochs + 1):

        # --- Training Phase --- #
        epoch_train_loss = 0.0
        num_train_steps = 0

        for item in train_cached_tensors:
            n_views = len(item["tensors"])
            tgt_idx = np.random.randint(0, n_views)
            target_xr = item["tensors"][tgt_idx].to(device)
            target_azimuth = float(item["angles"][tgt_idx])

            _, rec_loss = fit_latent_and_weights(
                generator,
                target_xr,
                z_dim=z_dim,
                img_size=H,
                radius=radius,
                theta_mean=theta_mean,
                device=device,
                iterations=args.iters_per_sample,
                use_amp=args.amp,
                azimuth=target_azimuth,
            )

            epoch_train_loss += rec_loss
            num_train_steps += 1

        avg_train_loss = epoch_train_loss / max(1, num_train_steps)

        # --- Validation Phase --- #
        epoch_val_loss = 0.0
        num_val_steps = 0

        for target_xr, target_azimuth in val_pairs:
            target_xr = target_xr.to(device)

            generator_val = copy.deepcopy(generator)
            generator_val.parameters = lambda: generator_val._parameters
            generator_val.named_parameters = lambda: generator_val._named_parameters

            _, rec_loss = fit_latent_and_weights(
                generator_val,
                target_xr,
                z_dim=z_dim,
                img_size=H,
                radius=radius,
                theta_mean=theta_mean,
                device=device,
                iterations=args.val_iters_per_sample,
                azimuth=target_azimuth,
            )

            epoch_val_loss += rec_loss
            num_val_steps += 1

        avg_val_loss = epoch_val_loss / max(1, num_val_steps)

        # --- Logging & Multi-View Visualization --- #
        logger.info(f"Epoch [{epoch}/{args.epochs}] - Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)

        if args.viz_every > 0 and (epoch % args.viz_every == 0 or epoch == args.epochs) and viz_patient_dir is not None:
            pa_path = os.path.join(viz_patient_dir, "pa.png")
            gt_views_dir = os.path.join(viz_patient_dir, "views")

            if os.path.exists(pa_path):
                indices = epoch_rotating_view_indices(epoch, args.viz_views)
                tmp_ckpt_path = os.path.join(ckpt_dir, f"_viz_tmp_mednerf_{os.getpid()}.pt")
                input_xray = TF.to_tensor(Image.open(pa_path).convert("L")).unsqueeze(0).to(device)

                multiview_grid = visualize_live_checkpoint(
                    MedNeRFWrapper, ckpt_io.save, tmp_ckpt_path, input_xray, gt_views_dir, indices,
                )

                if multiview_grid is not None:
                    writer.add_image("Images/val_multiview", multiview_grid, epoch)

        # --- Checkpoint Saving --- #
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            ckpt_io.save("mednerf_best.pt")
            logger.info(f"New best model saved to {os.path.join(ckpt_dir, 'mednerf_best.pt')} (Val Loss: {best_loss:.6f})")

    ckpt_io.save("mednerf_checkpoint.pt")
    writer.close()
    logger.info("MedNeRF training complete.")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedNeRF Baseline Trainer")

    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--iters_per_sample", type=int, default=50, help="Latent+weight fitting iterations per patient")
    parser.add_argument("--val_iters_per_sample", type=int, default=None, help="Validation fitting iterations (default: iters_per_sample // 4)")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Max training samples")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Max validation samples")
    parser.add_argument("--viz_every", type=int, default=10, help="Log multi-view TensorBoard panel every N epochs (0 to disable)")
    parser.add_argument("--viz_views", type=int, default=6, help="Number of azimuths per multi-view panel")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision for per-scan fitting")

    args = parser.parse_args()
    train_mednerf(args)
