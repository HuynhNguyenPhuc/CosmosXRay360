"""NAF (MICCAI 2022) Training Script."""

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
sys.path.insert(0, os.path.join(BASE_DIR, "cloned", "naf_cbct"))

from src.encoder.hashencoder import HashEncoder  # type: ignore
from src.network.network import DensityNetwork  # type: ignore

from models.naf import fit_density_field, NAFWrapper
from models.utils import get_train_val_patient_dirs
from models.viz import epoch_rotating_view_indices, visualize_live_checkpoint


# --- Logger Setup --- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
)
logger = logging.getLogger(__name__)


# --- Configuration --- #
BOUND = 0.3


# =============================================================================
# Main Training Function
# =============================================================================

def train_naf(args: argparse.Namespace) -> None:
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

    # 1. Initialize Network & Hash Encoder
    encoder = HashEncoder(
        input_dim=3,
        num_levels=16,
        level_dim=2,
        base_resolution=16,
        log2_hashmap_size=19,
    )

    model = DensityNetwork(
        encoder=encoder,
        bound=BOUND,
        num_layers=4,
        hidden_dim=32,
        skips=[2],
        out_dim=1,
        last_activation="sigmoid",
    ).to(device)

    # 2. Load Dataset Directories & Pre-cache Projections
    rendered_dir = os.path.join(BASE_DIR, "..", "datasets", "pre_rendered")
    train_patient_dirs, val_patient_dirs = get_train_val_patient_dirs(rendered_dir)

    logger.info(f"Found {len(train_patient_dirs)} train cases and {len(val_patient_dirs)} val cases for NAF.")

    def cache_target_projs(patient_dirs: list[str]) -> list[dict]:
        cached = []

        for pat_path in patient_dirs:
            views_dir = os.path.join(pat_path, "views")
            if not os.path.exists(views_dir):
                pa_file = os.path.join(pat_path, "pa.png")
                if os.path.exists(pa_file):
                    cached.append({"projs": [TF.to_tensor(Image.open(pa_file).convert("L"))], "angles": [0.0]})
                continue

            view_files = sorted(f for f in os.listdir(views_dir) if f.endswith(".png"))
            if not view_files:
                continue

            view_tensors = [TF.to_tensor(Image.open(os.path.join(views_dir, vf)).convert("L")) for vf in view_files]
            angles = list(np.linspace(0.0, 360.0, len(view_files)))
            cached.append({"projs": view_tensors, "angles": angles})

        return cached

    train_cached_projs = cache_target_projs(
        train_patient_dirs[:args.max_train_samples] if args.max_train_samples else train_patient_dirs
    )

    val_cached_projs = cache_target_projs(
        val_patient_dirs[:args.max_val_samples] if args.max_val_samples else val_patient_dirs
    )

    # Fixed validation pairs with reproducible seed
    val_rng = np.random.RandomState(42)
    val_pairs = []
    for item in val_cached_projs:
        n_views = len(item["projs"])
        tgt_idx = val_rng.randint(1, n_views) if n_views > 1 else 0
        val_pairs.append((item["projs"][tgt_idx], float(item["angles"][tgt_idx])))

    viz_patient_dir = val_patient_dirs[0] if val_patient_dirs else None

    # 3. Checkpoint Directory & SummaryWriter Setup
    ckpt_dir = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    latest_ckpt_path = os.path.join(ckpt_dir, "naf_checkpoint.pt")
    best_ckpt_path = os.path.join(ckpt_dir, "naf_best.pt")

    writer = SummaryWriter(log_dir=os.path.join(ckpt_dir, "tensorboard", "naf"))

    logger.info(f"Starting NAF training for {args.epochs} epochs...")
    best_loss = float("inf")

    # 4. Main Training Loop
    for epoch in range(1, args.epochs + 1):

        # --- Training Phase --- #
        epoch_train_loss = 0.0
        num_train_steps = 0

        for item in train_cached_projs:
            n_views = len(item["projs"])
            tgt_idx = np.random.randint(0, n_views)
            target_proj = item["projs"][tgt_idx]
            target_azimuth = float(item["angles"][tgt_idx])

            loss = fit_density_field(
                model, target_proj, BOUND, device, iterations=args.iters_per_sample, lr=args.lr, azimuth=target_azimuth,
            )
            epoch_train_loss += loss
            num_train_steps += 1

        avg_train_loss = epoch_train_loss / max(1, num_train_steps)

        # --- Validation Phase --- #
        epoch_val_loss = 0.0
        num_val_steps = 0

        for target_proj, target_azimuth in val_pairs:
            model_val = copy.deepcopy(model)

            loss = fit_density_field(
                model_val, target_proj, BOUND, device, iterations=args.val_iters_per_sample, lr=args.lr, azimuth=target_azimuth,
            )

            epoch_val_loss += loss
            num_val_steps += 1

        avg_val_loss = epoch_val_loss / max(1, num_val_steps)

        # --- Logging & Visualization --- #
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

                tmp_ckpt_path = os.path.join(ckpt_dir, f"_viz_tmp_naf_{os.getpid()}.pt")
                input_xray = TF.to_tensor(Image.open(pa_path).convert("L")).unsqueeze(0).to(device)

                multiview_grid = visualize_live_checkpoint(
                    NAFWrapper, _save_viz_ckpt, tmp_ckpt_path, input_xray, gt_views_dir, indices,
                )

                if multiview_grid is not None:
                    writer.add_image("Images/val_multiview", multiview_grid, epoch)

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
    logger.info(f"NAF training complete. Checkpoints saved to {latest_ckpt_path} and {best_ckpt_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NAF Baseline Trainer")

    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--iters_per_sample", type=int, default=200, help="Per-scan fitting iterations per patient")
    parser.add_argument("--val_iters_per_sample", type=int, default=None, help="Validation fitting iterations (default: iters_per_sample // 4)")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Max training samples")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Max validation samples")
    parser.add_argument("--viz_every", type=int, default=10, help="Log multi-view TensorBoard panel every N epochs (0 to disable)")
    parser.add_argument("--viz_views", type=int, default=6, help="Number of azimuths per multi-view panel")

    args = parser.parse_args()
    train_naf(args)
