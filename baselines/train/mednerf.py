"""MedNeRF (EMBC 2022) Training Script for CosmosXRay360

MedNeRF/GRAF's actual training algorithm is a two-stage, *unconditional* pipeline:
(1) pretrain a population-level generator+discriminator adversarially (patch-based,
scale-annealed ray sampling, R1-regularized hinge loss -- see
``cloned/mednerf/graf-main/train.py`` and ``graf/gan_training.py``) on an unpaired
image collection, then (2) fit a per-patient latent code + generator weights to a
*specific* reference X-ray at test time (``cloned/mednerf/graf-main/render_xray_G.py``).

This project's benchmark is single-frontal-view-conditioned per patient, which has no
natural analogue of stage (1)'s unpaired population dataset (the FastGAN-style
discriminator in ``graf/models/discriminator.py`` expects scale-annealed 32x32
patches with an auxiliary reconstruction-decoder loss and DiffAug-style
augmentation -- reproducing it faithfully is a separate, large undertaking and is not
attempted here). Instead, this script repeatedly exercises stage (2)'s procedure
(``models.mednerf.fit_latent_and_weights``, the same function ``MedNeRFWrapper``
calls at inference) across the training split, letting the shared generator's weights
accumulate a better starting prior across patients before per-patient fitting at eval
time. This replaces a prior version that minimized MSE between a *fresh random z* (not
optimized to match anything) and a specific target image at a fixed pose -- an
objective with no way to learn patient-specific correspondence and no adversarial
signal at all, i.e. not a meaningful training loss under this algorithm.

Saves checkpoints to baselines/checkpoints/ (both best and latest).
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import torchvision.transforms.functional as TF

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
mednerf_dir = os.path.join(BASE_DIR, "cloned", "mednerf", "graf-main")
sys.path.insert(0, mednerf_dir)
sys.path.insert(0, os.path.join(mednerf_dir, "submodules"))

from graf.config import build_models, get_render_poses
from graf.utils import to_theta
from submodules.GAN_stability.gan_training.config import load_config
from submodules.GAN_stability.gan_training.checkpoints import CheckpointIO

from models.mednerf import fit_latent_and_weights
from models.utils import get_train_val_patient_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
)
logger = logging.getLogger(__name__)


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

    # Mean polar/elevation angle of the trained pose distribution (matches
    # render_xray_G.py's theta_mean). chest.yaml's vmin/vmax sit around 70-85 degrees
    # (near-horizontal circular sweep around the chest); theta=0 would be the poles
    # (looking straight along the body's long axis) -- not a pose PA/lateral chest
    # X-rays were ever rendered from.
    theta_mean = 0.5 * (to_theta(config_file["data"]["vmin"]) + to_theta(config_file["data"]["vmax"]))

    generator, _ = build_models(config_file, disc=False)
    generator = generator.to(device)
    # Optimize chunk size for Titan RTX (24GB VRAM) to process 128x128 = 16384 rays in 1 pass
    generator.chunk = 16384

    rendered_dir = os.path.join(BASE_DIR, "..", "datasets", "pre_rendered")
    train_patient_dirs, val_patient_dirs = get_train_val_patient_dirs(rendered_dir)

    logger.info(f"Found {len(train_patient_dirs)} train cases and {len(val_patient_dirs)} val cases for MedNeRF.")

    # Pre-cache input image tensors in RAM
    def cache_pa_tensors(patient_dirs: list[str]) -> list[torch.Tensor]:
        cached = []
        for pat_path in patient_dirs:
            pa_file = os.path.join(pat_path, "pa.png")
            if not os.path.exists(pa_file):
                continue
            pa_tensor = TF.to_tensor(Image.open(pa_file).convert("L")).unsqueeze(0)
            pa_tensor = TF.resize(pa_tensor, [H, W])
            cached.append(pa_tensor)
        return cached

    train_cached_pa_tensors = cache_pa_tensors(
        train_patient_dirs[:args.max_train_samples] if args.max_train_samples else train_patient_dirs
    )
    val_cached_pa_tensors = cache_pa_tensors(val_patient_dirs)
    if args.max_val_samples is not None and len(val_cached_pa_tensors) > args.max_val_samples:
        val_cached_pa_tensors = val_cached_pa_tensors[:args.max_val_samples]
        logger.info(f"Subsampled val set to {len(val_cached_pa_tensors)} cases for fast epoch evaluation.")

    ckpt_dir = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_io = CheckpointIO(ckpt_dir)
    ckpt_io.register_modules(**{k + "_test": v for k, v in generator.module_dict.items()})
    writer = SummaryWriter(log_dir=os.path.join(ckpt_dir, "tensorboard", "mednerf"))

    logger.info(f"Starting MedNeRF training for {args.epochs} epochs...")

    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # 1. Training Phase -- fit the *shared* generator directly (no deepcopy) so
        # its weights accumulate a better prior across patients.
        epoch_train_loss = 0.0
        num_train_steps = 0

        for pa_tensor in train_cached_pa_tensors:
            pa_tensor = pa_tensor.to(device)
            _, rec_loss = fit_latent_and_weights(
                generator,
                pa_tensor,
                z_dim=z_dim,
                img_size=H,
                radius=radius,
                theta_mean=theta_mean,
                device=device,
                iterations=args.iters_per_sample,
            )
            epoch_train_loss += rec_loss
            num_train_steps += 1

        avg_train_loss = epoch_train_loss / max(1, num_train_steps)

        # 2. Validation Phase -- fit a throwaway deep copy per patient (mirrors
        # MedNeRFWrapper.infer_multi_views) so validation never mutates the shared
        # generator being trained. Uses val_iters_per_sample (default well below
        # iters_per_sample): this measures whether the current shared generator is
        # trending toward a better prior, which a partial fit already tracks -- it
        # doesn't need full convergence, and each iteration here is expensive
        # (measured: ~876ms, backprop through the full generator plus a LPIPS/AlexNet
        # forward+backward every step), so validation's redundant 1x-full-fit
        # overhead is where the real cost was going (same reasoning as NAF's
        # train/naf.py, just with a ~17x higher per-iteration cost here).
        epoch_val_loss = 0.0
        num_val_steps = 0
        val_sample_img = None

        for idx, pa_tensor in enumerate(val_cached_pa_tensors):
            pa_tensor = pa_tensor.to(device)
            generator_val = copy.deepcopy(generator)
            generator_val.parameters = lambda: generator_val._parameters
            generator_val.named_parameters = lambda: generator_val._named_parameters

            z_opt, rec_loss = fit_latent_and_weights(
                generator_val,
                pa_tensor,
                z_dim=z_dim,
                img_size=H,
                radius=radius,
                theta_mean=theta_mean,
                device=device,
                iterations=args.val_iters_per_sample,
            )
            if idx == 0:
                with torch.no_grad():
                    pose = get_render_poses(radius=radius, angle_range=(0, 0), theta=theta_mean, N=1)[0].to(device)
                    rays = generator_val.val_ray_sampler(H, H, generator_val.focal, pose)[0]
                    out = generator_val(z_opt, rays=rays)
                    rgb = out[0] if isinstance(out, tuple) else out
                    if rgb.dim() == 2:
                        rgb = rgb.view(1, H, H, -1).permute(0, 3, 1, 2)
                    if rgb.shape[1] == 4:
                        rgb = rgb[:, :3]
                    elif rgb.shape[1] == 1:
                        rgb = rgb.repeat(1, 3, 1, 1)
                    pred_img = ((rgb[0] + 1.0) / 2.0).clamp(0, 1)
                    targ_inp = pa_tensor.unsqueeze(0) if pa_tensor.dim() == 3 else pa_tensor
                    targ_img = F.interpolate(targ_inp, size=(H, H), mode="bilinear")[0]
                    if targ_img.shape[0] == 1:
                        targ_img = targ_img.repeat(3, 1, 1)
                    val_sample_img = torch.cat([pred_img, targ_img.clamp(0, 1)], dim=-1).cpu()

            epoch_val_loss += rec_loss
            num_val_steps += 1

        avg_val_loss = epoch_val_loss / max(1, num_val_steps)
        logger.info(f"Epoch [{epoch}/{args.epochs}] - Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        if val_sample_img is not None:
            writer.add_image("Images/val_pred_gt", val_sample_img, epoch)

        # Save best checkpoint on min val loss
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            ckpt_io.save("mednerf_best.pt")
            logger.info(f"New best model saved to {os.path.join(ckpt_dir, 'mednerf_best.pt')} (Val Loss: {best_loss:.6f})")

    # Always save latest checkpoint at end of training
    ckpt_io.save("mednerf_checkpoint.pt")
    writer.close()
    logger.info("MedNeRF training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedNeRF Baseline Trainer")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--iters_per_sample", type=int, default=50, help="Latent+weight fitting iterations per patient (matches MedNeRFWrapper's inference-time budget)")
    parser.add_argument("--val_iters_per_sample", type=int, default=None, help="Per-scan fitting iterations for validation (default: iters_per_sample // 4 -- validation only needs a trend signal for the current shared generator, not full convergence, and fit_latent_and_weights' generator+LPIPS forward/backward dominates per-patient cost)")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Max training samples (default: None for full train set)")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Max validation samples per epoch (default: None for full val)")
    args = parser.parse_args()
    train_mednerf(args)
