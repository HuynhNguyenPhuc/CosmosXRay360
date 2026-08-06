"""Dx2CT (ICASSP 2025) Training Script for CosmosXRay360

Trains 3D diffusion slice generator on Cross-Dataset split.
Saves checkpoints to baselines/checkpoints/ (both best and latest).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torch.utils.tensorboard import SummaryWriter
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch or NumPy not available")

try:
    from diffusers import DDPMScheduler
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
REPO_ROOT = os.path.dirname(BASE_DIR)
sys.path.insert(0, REPO_ROOT)

# Imported via models.dx2ct's dynamic importlib loader (not a plain `sys.path.insert
# + from model import ...`) because several cloned baseline repos each ship their own
# top-level 'model' module; a bare import here caches whichever one wins in
# sys.modules['model'] and silently breaks any other baseline's `from model import ...`
# for the rest of the process (e.g. PixelNeRF's wrapper).
from models.dx2ct import DX2CTModel

from models.utils import get_train_val_patient_dirs
from renderers.diffdrr.data import load_ct_volume

# Same three raw-CT directories datasets/pre_render_diffdrr.py globs to produce
# pre_rendered/<patient_id>/ -- patient_id is exactly the .nii.gz stem (see its
# get_patient_id), so this is the reverse lookup back to the source volume.
RAW_CT_DIRS = [
    os.path.join(REPO_ROOT, "datasets", "TCIA", "images"),
    os.path.join(REPO_ROOT, "datasets", "MELA2022", "raw", "train", "images"),
    os.path.join(REPO_ROOT, "datasets", "NSCLC", "processed", "train", "images"),
]

# load_ct_volume decodes/resamples a NIfTI volume in ~5s regardless of vol_shape
# (measured directly: 5.27s @ 256, 5.04s @ 128 -- fixed decode overhead, not
# output-resolution-bound), but training runs for many epochs (paper default: 80)
# over the same ~387 patients -- reloading every volume from the raw .nii.gz on
# every single epoch means ~31,000 redundant decodes across a full run. Cache the
# decoded tensor to a fast on-disk format after first use so only epoch 1 pays
# the full cost. vol_shape=128 (not 256): resolution doesn't help the load-time
# bottleneck, but it does shrink the cached file (and therefore every subsequent
# cache-hit disk read) ~8x, and 128 is still well above the 64x64 slice
# resolution actually extracted from it.
CT_CACHE_DIR = os.path.join(REPO_ROOT, "datasets", "_ct_volume_cache")
CT_CACHE_VOL_SHAPE = 128


def _build_raw_ct_lookup() -> dict[str, str]:
    """Maps patient_id -> raw .nii.gz path across all three source datasets."""
    lookup: dict[str, str] = {}
    for d in RAW_CT_DIRS:
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if fname.endswith(".nii.gz"):
                lookup[fname[: -len(".nii.gz")]] = os.path.join(d, fname)
    return lookup


def load_ct_volume_cached(ct_path: str, patient_id: str) -> torch.Tensor:
    """Loads a CT volume via load_ct_volume, transparently caching the decoded
    result to CT_CACHE_DIR so repeat calls (every epoch after the first) read a
    small pre-decoded tensor instead of re-running NIfTI decode/resample.

    Writes via a temp file + atomic os.replace so two DataLoader workers racing
    to populate the same patient's cache entry can't leave a corrupt partial
    file -- worst case both decode redundantly once, which is harmless.
    """
    os.makedirs(CT_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CT_CACHE_DIR, f"{patient_id}.pt")
    if os.path.exists(cache_path):
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    # load_ct_volume returns a MONAI MetaTensor (torch.Tensor subclass carrying
    # extra affine/metadata); cast to a plain tensor before caching so the saved
    # file only contains a bare tensor -- loadable with torch.load's default
    # weights_only=True (MetaTensor's internals otherwise trip its safe-unpickling
    # allowlist) and free of any MONAI-specific state this code never uses anyway.
    # torch.as_tensor() alone does NOT strip the subclass here (it's a no-op on
    # already-tensor-like inputs) -- MetaTensor.as_tensor() is the actual escape
    # hatch MONAI provides for this.
    raw_volume = load_ct_volume(ct_path, vol_shape=CT_CACHE_VOL_SHAPE)
    volume = raw_volume.as_tensor() if hasattr(raw_volume, "as_tensor") else torch.as_tensor(raw_volume)
    tmp_path = f"{cache_path}.tmp{os.getpid()}"
    torch.save(volume, tmp_path)
    os.replace(tmp_path, cache_path)
    return volume


def extract_axial_slices(volume: torch.Tensor, z_vals: torch.Tensor, spatial_size: int) -> torch.Tensor:
    """Extracts K real axial CT slices at world-space depths z_vals from a single
    loaded volume, in one batched grid_sample call.

    load_ct_volume takes ~5s per call regardless of vol_shape (dominated by fixed
    NIfTI decode/resample overhead, not output resolution -- measured directly),
    so extracting many slices per load instead of one is the actual lever for
    training throughput here, not shrinking vol_shape.

    Args:
        volume: Density tensor from load_ct_volume, (1, D, H, W) in [0, 1], world
            coordinates in [-1, 1] (D=Z depth, H=Y, W=X -- see
            renderers/diffdrr/renderer.py's world-frame docstring).
        z_vals: Target axial depths in [-1, 1], shape (K,) -- the paper's own
            coordinate convention (DX2CT, arXiv:2409.08850): "(x, y, z) in
            [-1, 1] for a target 3D CT volume", z being true anatomical depth
            through the body.
        spatial_size: Output slice resolution.

    Returns:
        The interpolated axial slices, (K, 1, spatial_size, spatial_size).
    """
    K = z_vals.shape[0]
    coords_1d = torch.linspace(-1.0, 1.0, spatial_size)
    grid_y, grid_x = torch.meshgrid(coords_1d, coords_1d, indexing="ij")
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(K, -1, -1, -1)
    z_grid = z_vals.view(K, 1, 1, 1).expand(-1, spatial_size, spatial_size, 1)
    grid = torch.cat([grid_xy, z_grid], dim=-1).unsqueeze(0)  # (1, K, spatial_size, spatial_size, 3)
    sampled = F.grid_sample(
        volume.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=True,
    )  # (1, 1, K, spatial_size, spatial_size)
    return sampled[0].permute(1, 0, 2, 3)  # (K, 1, spatial_size, spatial_size)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
)
logger = logging.getLogger(__name__)


class CrossDatasetSliceDataset(Dataset):
    def __init__(self, split_config_path: str, split: str = "train", slices_per_volume: int = 8) -> None:
        """
        Args:
            slices_per_volume: Number of random axial slices to extract from each
                loaded CT volume per __getitem__ call. load_ct_volume's cost is
                dominated by fixed decode/resample overhead (~5s regardless of
                output resolution -- measured directly), so amortizing it across
                many slices per load is what actually matters for throughput.
                Increases each batch's effective size by this factor (see
                train_dx2ct, which flattens the (B, slices_per_volume, ...) dims
                before feeding the model).
        """
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
        from PIL import Image
        import torchvision.transforms.functional as TF

        pat_path, ct_path, pat_id = self.samples[idx]

        pa_img = Image.open(os.path.join(pat_path, "pa.png")).convert("L")
        lat_img = Image.open(os.path.join(pat_path, "lat.png")).convert("L")

        # DX2CTWrapper.infer_multi_views always feeds the feature extractor full
        # 256x256 conditioning images and generates 64x64 slices; keep both resolutions
        # identical here so training isn't done at a different scale than inference.
        pa = TF.resize(TF.to_tensor(pa_img), [256, 256])
        lat = TF.resize(TF.to_tensor(lat_img), [256, 256])
        spatial_size = 64

        # Matches the paper (DX2CT, arXiv:2409.08850) exactly: real axial CT
        # density slices at genuine anatomical depths z, sampled from the same
        # loaded/normalized volume datasets/pre_render_diffdrr.py used to render
        # this patient's ground-truth projections (same [0, 1] density
        # convention, same [-1, 1] world frame -- see load_ct_volume's
        # docstring) -- not DiffDRR-rendered rotation-angle projections standing
        # in for a Z-coordinate they were never at. One (expensive) volume load
        # yields slices_per_volume (cheap) slice extractions -- see
        # extract_axial_slices' docstring for why that ratio matters.
        K = self.slices_per_volume
        volume = load_ct_volume_cached(ct_path, pat_id)
        z_vals = torch.empty(K).uniform_(-1.0, 1.0)
        target_slices = extract_axial_slices(volume, z_vals, spatial_size)  # (K, 1, spatial_size, spatial_size)

        coords_1d = torch.linspace(-1.0, 1.0, spatial_size)
        grid_y, grid_x = torch.meshgrid(coords_1d, coords_1d, indexing="ij")
        coords_3d = torch.cat([
            torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(K, -1, -1, -1),
            z_vals.view(K, 1, 1, 1).expand(-1, spatial_size, spatial_size, 1),
        ], dim=-1).reshape(K, spatial_size * spatial_size, 3)

        pa_rep = pa.unsqueeze(0).expand(K, -1, -1, -1)  # (K, 1, 256, 256)
        lat_rep = lat.unsqueeze(0).expand(K, -1, -1, -1)

        return pa_rep, lat_rep, target_slices, coords_3d


def train_dx2ct(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    model = DX2CTModel(
        embed_dim=128,
        num_blocks=12,
    ).to(device)

    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
    )

    train_dataset = CrossDatasetSliceDataset(
        split_config_path=args.data_split, split="train", slices_per_volume=args.slices_per_volume,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=6,
        pin_memory=True if device == "cuda" else False,
        # Keeps worker processes (and whatever OS page cache they've warmed) alive
        # across epochs instead of respawning them every epoch -- load_ct_volume's
        # ~5s fixed cost makes worker respawn overhead worth avoiding.
        persistent_workers=True,
    )

    val_dataset = CrossDatasetSliceDataset(
        split_config_path=args.data_split, split="val", slices_per_volume=args.slices_per_volume,
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

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    ckpt_dir = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    latest_ckpt_path = os.path.join(ckpt_dir, "dx2ct_surrogate_checkpoint.pt")
    best_ckpt_path = os.path.join(ckpt_dir, "dx2ct_best.pt")
    writer = SummaryWriter(log_dir=os.path.join(ckpt_dir, "tensorboard", "dx2ct"))

    logger.info(f"Starting Dx2CT training for {args.epochs} epochs...")

    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # 1. Training Phase
        model.train()
        # Accumulated as a device tensor (not `.item()`-ed per batch) to avoid a
        # CUDA sync every step; `.item()` is called once, after the loop.
        epoch_train_loss = torch.zeros((), device=device)
        num_train_batches = 0

        for batch_idx, (pa, lat, target_slice, coords_3d) in enumerate(train_loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            # Dataset yields (B, slices_per_volume, ...) -- flatten into one flat
            # batch of B * slices_per_volume examples (K distinct real slices per
            # loaded CT volume, see CrossDatasetSliceDataset's docstring).
            pa = pa.view(-1, *pa.shape[2:])
            lat = lat.view(-1, *lat.shape[2:])
            target_slice = target_slice.view(-1, *target_slice.shape[2:])
            coords_3d = coords_3d.view(-1, *coords_3d.shape[2:])

            pa, lat = pa.to(device, non_blocking=True), lat.to(device, non_blocking=True)
            target_slice = target_slice.to(device, non_blocking=True)
            coords_3d = coords_3d.to(device, non_blocking=True)

            noise = torch.randn_like(target_slice)
            B = target_slice.shape[0]

            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device).long()
            noisy_slice = scheduler.add_noise(target_slice, noise, timesteps)

            optimizer.zero_grad(set_to_none=True)
            noise_pred = model(noisy_slice, pa, lat, coords_3d, timesteps=timesteps)
            loss = criterion(noise_pred, noise)

            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.detach()
            num_train_batches += 1

        avg_train_loss = (epoch_train_loss / max(1, num_train_batches)).item()

        # 2. Validation Phase
        model.eval()
        epoch_val_loss = torch.zeros((), device=device)
        num_val_batches = 0
        val_sample_img = None

        val_generator = torch.Generator(device=device).manual_seed(42)

        with torch.no_grad():
            for batch_idx, (pa, lat, target_slice, coords_3d) in enumerate(val_loader):
                if args.max_val_batches is not None and batch_idx >= args.max_val_batches:
                    break
                elif args.max_batches is not None and batch_idx >= args.max_batches:
                    break

                # See matching flatten in the training loop above.
                pa = pa.view(-1, *pa.shape[2:])
                lat = lat.view(-1, *lat.shape[2:])
                target_slice = target_slice.view(-1, *target_slice.shape[2:])
                coords_3d = coords_3d.view(-1, *coords_3d.shape[2:])

                pa, lat = pa.to(device, non_blocking=True), lat.to(device, non_blocking=True)
                target_slice = target_slice.to(device, non_blocking=True)
                coords_3d = coords_3d.to(device, non_blocking=True)

                noise = torch.randn(target_slice.shape, generator=val_generator, device=device)
                B = target_slice.shape[0]

                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (B,), generator=val_generator, device=device).long()
                noisy_slice = scheduler.add_noise(target_slice, noise, timesteps)

                noise_pred = model(noisy_slice, pa, lat, coords_3d, timesteps=timesteps)
                loss = criterion(noise_pred, noise)

                if batch_idx == 0:
                    alpha_bar = scheduler.alphas_cumprod[timesteps[0]].to(device)
                    sqrt_alpha_bar = torch.sqrt(alpha_bar)
                    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)
                    pred_slice = ((noisy_slice[0] - sqrt_one_minus_alpha_bar * noise_pred[0]) / (sqrt_alpha_bar + 1e-8)).clamp(0, 1)
                    targ_slice = target_slice[0].clamp(0, 1)
                    if pred_slice.shape[0] == 1:
                        pred_slice = pred_slice.repeat(3, 1, 1)
                        targ_slice = targ_slice.repeat(3, 1, 1)
                    val_sample_img = torch.cat([pred_slice, targ_slice], dim=-1).cpu()

                epoch_val_loss += loss.detach()
                num_val_batches += 1

        avg_val_loss = (epoch_val_loss / max(1, num_val_batches)).item()
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
    logger.info(f"Dx2CT training complete. Checkpoints saved to {latest_ckpt_path} and {best_ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dx2CT Baseline Trainer")
    parser.add_argument("--data_split", type=str, default="datasets/cross_dataset_split.json", help="Path to JSON split config.")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs (Dx2CT paper default: 80)")
    parser.add_argument("--batch_size", type=int, default=2, help="CT volumes loaded per step (default: 2 for VRAM safety). Effective training batch size is batch_size * slices_per_volume.")
    parser.add_argument("--slices_per_volume", type=int, default=8, help="Random axial slices extracted per loaded CT volume (default: 8). load_ct_volume costs ~5s regardless of resolution, so this amortizes that fixed cost across more training examples per load -- see CrossDatasetSliceDataset's docstring.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_batches", type=int, default=None, help="Optional max batches per epoch")
    parser.add_argument("--max_val_batches", type=int, default=None, help="Optional max val batches per epoch (default: None for full val)")
    args = parser.parse_args()
    train_dx2ct(args)
