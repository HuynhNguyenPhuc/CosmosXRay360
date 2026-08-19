"""Dx2CT (ICASSP 2025) Training Script."""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF
from diffusers import DDPMScheduler
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
REPO_ROOT = os.path.dirname(BASE_DIR)
sys.path.insert(0, REPO_ROOT)

from models.dx2ct import DX2CTModel, Dx2CTWrapper
from models.utils import get_train_val_patient_dirs
from models.viz import epoch_rotating_view_indices, visualize_live_checkpoint
from renderers.diffdrr.data import load_ct_volume


# --- Logger Setup --- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
)
logger = logging.getLogger(__name__)


# --- Configuration & Paths --- #
RAW_CT_DIRS = [
    os.path.join(REPO_ROOT, "datasets", "TCIA", "images"),
    os.path.join(REPO_ROOT, "datasets", "MELA2022", "raw", "train", "images"),
    os.path.join(REPO_ROOT, "datasets", "NSCLC", "processed", "train", "images"),
]

CT_CACHE_DIR = os.path.join(REPO_ROOT, "datasets", "_ct_volume_cache")
CT_CACHE_VOL_SHAPE = 128


# =============================================================================
# Utility Functions
# =============================================================================

def _build_raw_ct_lookup() -> dict[str, str]:
    """Map patient IDs (basename of NIfTI file) to full path of the raw CT volume on disk."""
    lookup: dict[str, str] = {}

    for d in RAW_CT_DIRS:
        if not os.path.isdir(d):
            continue

        for fname in os.listdir(d):
            if fname.endswith(".nii.gz"):
                lookup[fname[: -len(".nii.gz")]] = os.path.join(d, fname)

    return lookup


def load_ct_volume_cached(ct_path: str, patient_id: str) -> torch.Tensor:
    """Load a CT volume from disk or retrieve from on-disk tensor cache.

    Args:
        ct_path: Path to raw CT volume (NIfTI file).
        patient_id: Patient ID used for caching key.

    Returns:
        Loaded CT volume tensor normalized to [0, 1].
    """
    os.makedirs(CT_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CT_CACHE_DIR, f"{patient_id}.pt")

    if os.path.exists(cache_path):
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    # 1. Load raw volume
    raw_volume = load_ct_volume(ct_path, vol_shape=CT_CACHE_VOL_SHAPE)
    volume = raw_volume.as_tensor() if hasattr(raw_volume, "as_tensor") else torch.as_tensor(raw_volume)

    # 2. Atomic save to cache directory
    tmp_path = f"{cache_path}.tmp{os.getpid()}"
    torch.save(volume, tmp_path)
    os.replace(tmp_path, cache_path)

    return volume


def extract_axial_slices(volume: torch.Tensor, z_vals: torch.Tensor, spatial_size: int) -> torch.Tensor:
    """Extract axial slices from a 3D CT volume at specified z-values using trilinear interpolation.

    Args:
        volume: CT volume tensor of shape (1, D, H, W).
        z_vals: Target z-depths in [-1, 1] of shape (K,).
        spatial_size: Height/width resolution for output slices.

    Returns:
        Interpolated axial slices of shape (K, 1, spatial_size, spatial_size).
    """
    K = z_vals.shape[0]

    # Create 3D sampling grid
    coords_1d = torch.linspace(-1.0, 1.0, spatial_size)
    grid_y, grid_x = torch.meshgrid(coords_1d, coords_1d, indexing="ij")
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(K, -1, -1, -1)

    z_grid = z_vals.view(K, 1, 1, 1).expand(-1, spatial_size, spatial_size, 1)
    grid = torch.cat([grid_xy, z_grid], dim=-1).unsqueeze(0)  # (1, K, spatial_size, spatial_size, 3)

    # Trilinear sampling
    sampled = F.grid_sample(
        volume.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    return sampled[0].permute(1, 0, 2, 3)  # (K, 1, spatial_size, spatial_size)


# =============================================================================
# Dataset Class
# =============================================================================

class CrossDatasetSliceDataset(Dataset):
    """Dataset producing paired X-ray projections and extracted CT axial slices."""

    def __init__(self, split_config_path: str, split: str = "train", slices_per_volume: int = 8) -> None:
        super().__init__()

        self.split = split
        self.slices_per_volume = slices_per_volume
        self.samples = []

        base_dir = os.path.dirname(os.path.abspath(split_config_path))
        rendered_dir = os.path.join(base_dir, "pre_rendered")

        if not os.path.exists(rendered_dir):
            rendered_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "datasets", "pre_rendered"))

        train_patient_dirs, val_patient_dirs = get_train_val_patient_dirs(rendered_dir)
        patient_dirs = val_patient_dirs if split == "val" else train_patient_dirs

        raw_ct_lookup = _build_raw_ct_lookup()
        skipped_no_ct = 0

        for pat_path in patient_dirs:
            pat_id = os.path.basename(os.path.normpath(pat_path))
            ct_path = raw_ct_lookup.get(pat_id)

            if not (ct_path and os.path.exists(os.path.join(pat_path, "pa.png"))
                    and os.path.exists(os.path.join(pat_path, "lat.png"))):
                skipped_no_ct += ct_path is None
                continue

            self.samples.append((pat_path, ct_path, pat_id))

        self.total_cases = len(self.samples)
        logger.info(
            f"Initialized {split.capitalize()} dataset with {self.total_cases} cases "
            f"({skipped_no_ct} skipped: no matching raw CT volume found)."
        )

    def __len__(self) -> int:
        return self.total_cases

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pat_path, ct_path, pat_id = self.samples[idx]

        pa_img = Image.open(os.path.join(pat_path, "pa.png")).convert("L")
        lat_img = Image.open(os.path.join(pat_path, "lat.png")).convert("L")

        # 1. Resize input projections
        pa = TF.resize(TF.to_tensor(pa_img), [256, 256])
        lat = TF.resize(TF.to_tensor(lat_img), [256, 256])
        spatial_size = 128

        # 2. Extract random axial slices from CT volume
        K = self.slices_per_volume
        volume = load_ct_volume_cached(ct_path, pat_id)

        z_vals = torch.empty(K).uniform_(-1.0, 1.0)
        target_slices = extract_axial_slices(volume, z_vals, spatial_size)

        # 3. Create 3D sampling coordinates
        coords_1d = torch.linspace(-1.0, 1.0, spatial_size)
        grid_y, grid_x = torch.meshgrid(coords_1d, coords_1d, indexing="ij")

        coords_3d = torch.cat([
            torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(K, -1, -1, -1),
            z_vals.view(K, 1, 1, 1).expand(-1, spatial_size, spatial_size, 1),
        ], dim=-1).reshape(K, spatial_size * spatial_size, 3)

        pa_rep = pa.unsqueeze(0).expand(K, -1, -1, -1)
        lat_rep = lat.unsqueeze(0).expand(K, -1, -1, -1)

        return pa_rep, lat_rep, target_slices, coords_3d


# =============================================================================
# Training Pipeline
# =============================================================================

def train_dx2ct(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # 1. Initialize Model & Scheduler
    model = DX2CTModel(embed_dim=128, num_blocks=12).to(device)

    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
    )

    # 2. Build DataLoaders
    train_dataset = CrossDatasetSliceDataset(
        split_config_path=args.data_split, split="train", slices_per_volume=args.slices_per_volume,
    )
    val_dataset = CrossDatasetSliceDataset(
        split_config_path=args.data_split, split="val", slices_per_volume=args.slices_per_volume,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=6,
        pin_memory=True if device == "cuda" else False,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=6,
        pin_memory=True if device == "cuda" else False,
        persistent_workers=True,
    )

    # 3. Visualization Patient Setup
    _viz_split_base = os.path.dirname(os.path.abspath(args.data_split))
    _viz_rendered_dir = os.path.join(_viz_split_base, "pre_rendered")

    if not os.path.exists(_viz_rendered_dir):
        _viz_rendered_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "datasets", "pre_rendered"))

    _, viz_val_patient_dirs = get_train_val_patient_dirs(_viz_rendered_dir)
    viz_patient_dir = viz_val_patient_dirs[0] if viz_val_patient_dirs else None

    # 4. Optimization & Tracking Setup
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # LR warmup + cosine decay and gradient clipping: Dx2CT trains from a random
    # init (not a fine-tune of a pretrained checkpoint like SV-DRR), which is
    # exactly the regime both of these matter most for -- warmup protects against
    # early-training instability, clipping guards against occasional large
    # gradients while the model is still finding its footing. No sibling reference
    # implementation exists for Dx2CT (unlike SV-DRR/XRaySyn's Cosmos-NVSyn
    # counterparts) to verify these against, so defaults are chosen from general
    # diffusion-training practice, not a known-working config: warmup_steps=500
    # matches SV-DRR's; gradient_clip_val=1.0 matches XRaySyn's own
    # clip_grad_norm_ convention (baselines/cloned/XraySyn/xraysyn/models/
    # ct2xray_real_gan_meta.py) rather than SV-DRR's DiT-specific 0.5.
    steps_per_epoch = len(train_loader) if args.max_batches is None else min(len(train_loader), args.max_batches)
    max_train_steps = args.epochs * steps_per_epoch
    min_lr_ratio = 0.1

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = min(1.0, (step - args.warmup_steps) / max(1, max_train_steps - args.warmup_steps))
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ckpt_dir = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    latest_ckpt_path = os.path.join(ckpt_dir, "dx2ct_surrogate_checkpoint.pt")
    best_ckpt_path = os.path.join(ckpt_dir, "dx2ct_best.pt")
    writer = SummaryWriter(log_dir=os.path.join(ckpt_dir, "tensorboard", "dx2ct"))

    logger.info(f"Starting Dx2CT training for {args.epochs} epochs...")
    best_loss = float("inf")

    # 5. Main Epoch Loop
    for epoch in range(1, args.epochs + 1):

        # --- Training Phase --- #
        model.train()
        epoch_train_loss = torch.zeros((), device=device)
        num_train_batches = 0

        for batch_idx, (pa, lat, target_slice, coords_3d) in enumerate(train_loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            # Flatten slice batch dimensions
            pa = pa.view(-1, *pa.shape[2:])
            lat = lat.view(-1, *lat.shape[2:])
            target_slice = target_slice.view(-1, *target_slice.shape[2:])
            coords_3d = coords_3d.view(-1, *coords_3d.shape[2:])

            pa, lat = pa.to(device, non_blocking=True), lat.to(device, non_blocking=True)
            target_slice = target_slice.to(device, non_blocking=True)
            coords_3d = coords_3d.to(device, non_blocking=True)

            # Monoplanar dropout: baselines/models/dx2ct.py's infer_multi_views (the
            # only way this baseline is ever actually evaluated, per this project's
            # single-view benchmark protocol) duplicates the PA image into the
            # lateral slot rather than supplying a real second view. Without this,
            # the model never sees that exact input distribution during training --
            # it always gets a genuine, different lat.png here -- so at inference
            # it's fed something out of its training distribution. Replacing lat
            # with a PA duplicate on a fraction of training samples closes that gap
            # while keeping the rest of training on real biplanar pairs.
            if args.lat_dropout_prob > 0:
                drop_mask = torch.rand(pa.shape[0], device=device) < args.lat_dropout_prob
                if drop_mask.any():
                    lat = lat.clone()
                    lat[drop_mask] = pa[drop_mask]

            noise = torch.randn_like(target_slice)
            B = target_slice.shape[0]

            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device).long()
            noisy_slice = scheduler.add_noise(target_slice, noise, timesteps)

            optimizer.zero_grad(set_to_none=True)
            noise_pred = model(noisy_slice, pa, lat, coords_3d, timesteps=timesteps)
            loss = criterion(noise_pred, noise)

            loss.backward()
            if args.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clip_val)
            optimizer.step()
            lr_scheduler.step()

            epoch_train_loss += loss.detach()
            num_train_batches += 1

        avg_train_loss = (epoch_train_loss / max(1, num_train_batches)).item()

        # --- Validation Phase --- #
        model.eval()
        epoch_val_loss = torch.zeros((), device=device)
        num_val_batches = 0

        val_generator = torch.Generator(device=device).manual_seed(42)

        with torch.no_grad():
            for batch_idx, (pa, lat, target_slice, coords_3d) in enumerate(val_loader):
                if args.max_val_batches is not None and batch_idx >= args.max_val_batches:
                    break
                elif args.max_batches is not None and batch_idx >= args.max_batches:
                    break

                pa = pa.view(-1, *pa.shape[2:])
                lat = lat.view(-1, *lat.shape[2:])
                target_slice = target_slice.view(-1, *target_slice.shape[2:])
                coords_3d = coords_3d.view(-1, *coords_3d.shape[2:])

                pa, lat = pa.to(device, non_blocking=True), lat.to(device, non_blocking=True)
                target_slice = target_slice.to(device, non_blocking=True)
                coords_3d = coords_3d.to(device, non_blocking=True)

                # Always monoplanar here, not a dropout probability: this is the one
                # and only condition infer_multi_views ever actually evaluates under,
                # so best-checkpoint selection (avg_val_loss below) should track that
                # real deployment condition rather than an easier real-biplanar signal
                # this baseline never gets to use.
                lat = pa

                noise = torch.randn(target_slice.shape, generator=val_generator, device=device)
                B = target_slice.shape[0]

                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (B,), generator=val_generator, device=device).long()
                noisy_slice = scheduler.add_noise(target_slice, noise, timesteps)

                noise_pred = model(noisy_slice, pa, lat, coords_3d, timesteps=timesteps)
                loss = criterion(noise_pred, noise)

                epoch_val_loss += loss.detach()
                num_val_batches += 1

        avg_val_loss = (epoch_val_loss / max(1, num_val_batches)).item()

        # --- Logging & TensorBoard --- #
        current_lr = lr_scheduler.get_last_lr()[0]
        logger.info(f"Epoch [{epoch}/{args.epochs}] - Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e}")
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("LR/model", current_lr, epoch)

        if args.viz_every > 0 and (epoch % args.viz_every == 0 or epoch == args.epochs) and viz_patient_dir is not None:
            pa_path = os.path.join(viz_patient_dir, "pa.png")
            gt_views_dir = os.path.join(viz_patient_dir, "views")

            if os.path.exists(pa_path):
                indices = epoch_rotating_view_indices(epoch, args.viz_views)

                def _save_viz_ckpt(path: str) -> None:
                    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)

                tmp_ckpt_path = os.path.join(ckpt_dir, f"_viz_tmp_dx2ct_{os.getpid()}.pt")
                input_xray = TF.to_tensor(Image.open(pa_path).convert("L")).unsqueeze(0).to(device)

                multiview_grid = visualize_live_checkpoint(
                    Dx2CTWrapper, _save_viz_ckpt, tmp_ckpt_path, input_xray, gt_views_dir, indices,
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
    logger.info(f"Dx2CT training complete. Checkpoints saved to {latest_ckpt_path} and {best_ckpt_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dx2CT Baseline Trainer")

    parser.add_argument("--data_split", type=str, default="datasets/cross_dataset_split.json", help="Path to JSON split config")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="CT volumes loaded per step")
    parser.add_argument("--slices_per_volume", type=int, default=8, help="Axial slices extracted per CT volume")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_batches", type=int, default=None, help="Max training batches per epoch")
    parser.add_argument("--max_val_batches", type=int, default=None, help="Max validation batches per epoch")
    parser.add_argument("--viz_every", type=int, default=10, help="Log multi-view TensorBoard panel every N epochs (0 to disable)")
    parser.add_argument("--viz_views", type=int, default=6, help="Number of azimuths per multi-view panel")
    parser.add_argument("--lat_dropout_prob", type=float, default=0.5, help="Probability of replacing the real lateral view with a PA duplicate during training, matching infer_multi_views' monoplanar inference input")
    parser.add_argument("--warmup_steps", type=int, default=500, help="LR warmup steps before cosine decay begins")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Gradient clipping max norm (0 to disable)")

    args = parser.parse_args()
    train_dx2ct(args)
