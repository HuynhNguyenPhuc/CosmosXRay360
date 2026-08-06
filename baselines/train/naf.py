"""NAF (MICCAI 2022) Training Script for CosmosXRay360

NAF's ``DensityNetwork`` is a per-scan overfitting coordinate MLP -- it takes only
3D coordinates as input, with no mechanism to condition on *which* patient's anatomy
it should represent. This repeatedly runs the same per-scan fitting procedure the
wrapper uses at inference time (``models.naf.fit_density_field``) across the training
split, directly on the *shared* model, so its weights accumulate a better starting
prior for the fresh per-patient fit ``NAFWrapper.infer_multi_views`` runs at eval time
(same reasoning as ``train/mednerf.py``).

This replaces a prior version that trained one shared, unconditioned coordinate field
directly against a *different* patient's PA image every step with no way to
distinguish patients -- not a meaningful objective for a per-scan method.

Saves checkpoints to baselines/checkpoints/ (both best and latest).
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import torchvision.transforms.functional as TF

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "cloned", "naf_cbct"))

from src.encoder.hashencoder import HashEncoder
from src.network.network import DensityNetwork

from models.naf import fit_density_field, FIT_GRID_RES
from models.utils import get_train_val_patient_dirs, perspective_ray_march

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
)
logger = logging.getLogger(__name__)

BOUND = 0.3  # Matches reference cloned/naf_cbct/config/*_50.yaml


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

    # Reference config (cloned/naf_cbct/config/*_50.yaml): hashgrid encoder,
    # num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=19.
    encoder = HashEncoder(
        input_dim=3,
        num_levels=16,
        level_dim=2,
        base_resolution=16,
        log2_hashmap_size=19,
    )

    # Reference config (cloned/naf_cbct/config/*_50.yaml): num_layers=4,
    # hidden_dim=32, skips=[2].
    model = DensityNetwork(
        encoder=encoder,
        bound=BOUND,
        num_layers=4,
        hidden_dim=32,
        skips=[2],
        out_dim=1,
        last_activation="sigmoid",
    ).to(device)

    rendered_dir = os.path.join(BASE_DIR, "..", "datasets", "pre_rendered")
    train_patient_dirs, val_patient_dirs = get_train_val_patient_dirs(rendered_dir)

    logger.info(f"Found {len(train_patient_dirs)} train cases and {len(val_patient_dirs)} val cases for NAF.")

    # Pre-cache target projection tensors in RAM
    def cache_target_projs(patient_dirs: list[str]) -> list[torch.Tensor]:
        cached = []
        for pat_path in patient_dirs:
            pa_file = os.path.join(pat_path, "pa.png")
            if not os.path.exists(pa_file):
                continue
            pa_tensor = TF.to_tensor(Image.open(pa_file).convert("L"))
            cached.append(pa_tensor)
        return cached

    train_cached_projs = cache_target_projs(
        train_patient_dirs[:args.max_train_samples] if args.max_train_samples else train_patient_dirs
    )
    val_cached_projs = cache_target_projs(
        val_patient_dirs[:args.max_val_samples] if args.max_val_samples else val_patient_dirs
    )

    ckpt_dir = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    latest_ckpt_path = os.path.join(ckpt_dir, "naf_checkpoint.pt")
    best_ckpt_path = os.path.join(ckpt_dir, "naf_best.pt")
    writer = SummaryWriter(log_dir=os.path.join(ckpt_dir, "tensorboard", "naf"))

    logger.info(f"Starting NAF training for {args.epochs} epochs...")

    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # 1. Training Phase -- fit the *shared* model directly (no deepcopy) so its
        # weights accumulate a better prior across patients.
        epoch_train_loss = 0.0
        num_train_steps = 0

        for target_proj in train_cached_projs:
            loss = fit_density_field(
                model, target_proj, BOUND, device, iterations=args.iters_per_sample, lr=args.lr
            )
            epoch_train_loss += loss
            num_train_steps += 1

        avg_train_loss = epoch_train_loss / max(1, num_train_steps)

        # 2. Validation Phase -- fit a throwaway deep copy per patient (mirrors
        # NAFWrapper.infer_multi_views) so validation never mutates the shared model.
        # Uses val_iters_per_sample (default well below iters_per_sample): this measures
        # whether the current shared prior is trending better/worse, which a partial fit
        # already tracks -- it doesn't need to reach the same full convergence training
        # does, and fit_density_field's ray marching is the dominant per-patient cost
        # (measured: ~50ms/iteration), so this is where validation's redundant 1x-full-fit
        # overhead actually goes.
        epoch_val_loss = 0.0
        num_val_steps = 0
        val_sample_img = None

        for idx, target_proj in enumerate(val_cached_projs):
            model_val = copy.deepcopy(model)
            loss = fit_density_field(
                model_val, target_proj, BOUND, device, iterations=args.val_iters_per_sample, lr=args.lr
            )
            if idx == 0:
                with torch.no_grad():
                    # Matches models.naf's perspective_ray_march usage (true ray
                    # marching, not a plain orthographic axis-sum) so this
                    # visualization reflects the same geometry actually used for
                    # fitting/eval. bound is shrunk by the same margin
                    # models.naf.fit_density_field uses -- see its safe_bound
                    # comment: HashEncoder's own domain check compares against
                    # BOUND as a Python float64, but clamping ray samples to
                    # exactly [-BOUND, BOUND] rounds to a *larger* float32 value,
                    # spuriously failing that check right at the boundary.
                    pred_p = perspective_ray_march(
                        sample_fn=model_val, azimuth=0.0, grid_res=FIT_GRID_RES, device=device,
                        bound=BOUND * (1.0 - 1e-6),
                    ).clamp(0, 1)
                    targ_t = target_proj.to(device)
                    while targ_t.dim() < 4:
                        targ_t = targ_t.unsqueeze(0)
                    targ_p = F.interpolate(targ_t, size=(FIT_GRID_RES, FIT_GRID_RES), mode="bilinear", align_corners=False).squeeze()
                    val_sample_img = torch.cat([pred_p, targ_p], dim=-1).unsqueeze(0).repeat(3, 1, 1).cpu()

            epoch_val_loss += loss
            num_val_steps += 1

        avg_val_loss = epoch_val_loss / max(1, num_val_steps)
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
    logger.info(f"NAF training complete. Checkpoints saved to {latest_ckpt_path} and {best_ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NAF Baseline Trainer")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--iters_per_sample", type=int, default=200, help="Per-scan fitting iterations per patient (matches NAFWrapper's inference-time budget)")
    parser.add_argument("--val_iters_per_sample", type=int, default=None, help="Per-scan fitting iterations for validation (default: iters_per_sample // 4 -- validation only needs a trend signal for the current shared prior, not full convergence, and fit_density_field's ray marching dominates per-patient cost)")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Max training samples (default: None for full train set)")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Max validation samples (default: None for full val)")
    args = parser.parse_args()
    train_naf(args)
