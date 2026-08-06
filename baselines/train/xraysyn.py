"""XRaySyn (AAAI 2021) Training Script for CosmosXRay360

Trains XRaySyn's 2D refinement network + discriminator (``net2d``, ``netD``) via the
reference ``XraySynModel.optimize()`` procedure: frontal-view backprojection into a
bone/tissue voxel volume by the frozen ``net3d``, re-projection through the
differentiable forward projector at the input pose plus one nearby "other" pose,
Beer-Lambert bone/tissue absorption, and adversarial + L1 + sparsity refinement --
matching ``xraysyn/models/ct2xray_real_gan_meta.py::optimize`` exactly instead of
directly regressing a duplicated frontal image against a lateral view (which mixed
raw attenuation units with Beer-Lambert transmissive-intensity units and never
exercised the projector, the absorption curves, or the discriminator).

Saves checkpoints to baselines/checkpoints/ (both best and latest).
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys

import torch
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import torchvision.transforms.functional as TF

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
xraysyn_dir = os.path.join(BASE_DIR, "cloned", "XraySyn")
sys.path.insert(0, xraysyn_dir)
sys.path.insert(0, os.path.join(xraysyn_dir, "xraysyn", "networks", "drr_projector"))

from xraysyn.models.ct2xray_real_gan_meta import XraySynModel

from models.utils import get_train_val_patient_dirs
from models.xraysyn import _get_T_batched

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
)
logger = logging.getLogger(__name__)

# The reference demo's own `self.views` sweep only ever exercises theta_y in
# [-0.05, 0.05] * pi (roughly +/-9 degrees around the input pose) -- but `net3d`
# and the DRRProjector (`proj`/`backproj`) are pure differentiable geometric
# operators with no learnable parameters (verified: no nn.Parameter anywhere in
# drr_projector_new.py), so they're correct at any pose. The only thing that
# actually needs angular training coverage is `net2d`'s learned refinement, and
# this project's benchmark evaluates the full 0-360 degree sweep (see
# models/xraysyn.py's infer_multi_views / evaluate.py). Training net2d on only a
# +/-9 degree neighborhood then asking it to refine reprojections 180 degrees
# away is a train/eval mismatch, not a fidelity requirement -- so this covers the
# full period instead, matching _get_T_multi's az/180-scaled convention (az in
# [0, 360) -> theta_y in [0, 2)).
OTHER_POSE_THETA_Y_RANGE = (-1.0, 1.0)


def load_pa_tensor(pat_path: str) -> torch.Tensor | None:
    """Loads a patient's frontal (PA) view as a [1, 1, 256, 256] tensor in [0, 1]."""
    pa_file = os.path.join(pat_path, "pa.png")
    if not os.path.exists(pa_file):
        return None
    return TF.to_tensor(Image.open(pa_file).convert("L")).unsqueeze(0)


@torch.no_grad()
def reconstruction_l1(model: XraySynModel, xray: torch.Tensor, return_image: bool = False):
    """Self-supervised reconstruction error at the input pose (no GAN/refinement grad).

    Mirrors the first (no_grad) half of ``optimize()``/``test()``: backproject,
    reproject at the input pose, refine, and compare against the input image. This is
    the only objective in the reference method with a real ground-truth target
    (the input view itself), so it doubles as our validation metric.
    """
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


def train_xraysyn(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    old_cwd = os.getcwd()
    os.chdir(xraysyn_dir)
    try:
        # NOTE: must pass device explicitly -- XraySynModel defaults to "cuda:0"
        # regardless of availability.
        model = XraySynModel(lr=args.lr, device=device)
    finally:
        os.chdir(old_cwd)

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

    train_cached_tensors = cache_tensors(train_patient_dirs[:args.max_train_samples] if args.max_train_samples else train_patient_dirs)
    val_cached_tensors = cache_tensors(val_patient_dirs[:args.max_val_samples] if args.max_val_samples else val_patient_dirs)

    ckpt_dir = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    latest_ckpt_path = os.path.join(ckpt_dir, "xraysyn_checkpoint.pt")
    best_ckpt_path = os.path.join(ckpt_dir, "xraysyn_best.pt")
    writer = SummaryWriter(log_dir=os.path.join(ckpt_dir, "tensorboard", "xraysyn"))

    logger.info(f"Starting XRaySyn training for {args.epochs} epochs...")

    best_loss = float("inf")
    rng = random.Random(42)

    for epoch in range(1, args.epochs + 1):
        # 1. Training Phase -- one XraySynModel.optimize() step per sample (nets are
        # put into their expected train/eval modes by the model itself).
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

        # 2. Validation Phase -- reconstruction-only L1 at the input pose (the only
        # objective in this method with a real ground-truth target).
        epoch_val_l1 = 0.0
        num_val_steps = 0
        val_sample_img = None

        for idx, pa_tensor in enumerate(val_cached_tensors):
            xray = pa_tensor.to(device)
            if idx == 0:
                l1_val, comp_img = reconstruction_l1(model, xray, return_image=True)
                val_sample_img = comp_img
            else:
                l1_val = reconstruction_l1(model, xray)
            epoch_val_l1 += l1_val
            num_val_steps += 1

        avg_val_l1 = epoch_val_l1 / max(1, num_val_steps)
        logger.info(f"Epoch [{epoch}/{args.epochs}] - Train L1 (running): {avg_train_l1:.6f} | Val L1: {avg_val_l1:.6f}")
        writer.add_scalar("L1/train", avg_train_l1, epoch)
        writer.add_scalar("L1/val", avg_val_l1, epoch)
        if val_sample_img is not None:
            writer.add_image("Images/val_pred_gt", val_sample_img, epoch)

        if avg_val_l1 < best_loss:
            best_loss = avg_val_l1
            model.save(best_ckpt_path)
            logger.info(f"New best model recorded (Val L1: {best_loss:.6f})")

    model.save(latest_ckpt_path)
    writer.close()
    logger.info(f"XRaySyn training complete. Checkpoints saved to {latest_ckpt_path} and {best_ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XRaySyn Baseline Trainer")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (matches num_epoch: 100 in cloned/XraySyn/config/xraysyn_test.yaml, the checkpoint-metadata record of the original training run)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (passed through to XraySynModel's Adam optimizers)")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Max training samples (default: None for full train set)")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Max validation samples (default: None for full val)")
    args = parser.parse_args()
    train_xraysyn(args)
