"""SV-DRR (MICCAI 2025) Training Script."""

from __future__ import annotations

import argparse
import copy
import logging
import math
import os
import sys

import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
svdrr_dir = os.path.join(BASE_DIR, "cloned", "SV-DRR")
sys.path.insert(0, svdrr_dir)

from pipeline_svdrr_DiT import CCProjection, SvdrrDiTPipeline  # type: ignore
from diffusion.iddpm import IDDPM  # type: ignore

from models.svdrr import SVDRRWrapper
from models.utils import get_train_val_patient_dirs
from models.viz import epoch_rotating_view_indices, visualize_live_checkpoint


# --- Logger Setup --- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================

def sample_timesteps_logit_normal(
    batch_size: int,
    device: torch.device,
    num_timesteps: int = 1000,
    mean: float = 0.0,
    std: float = 1.0,
) -> torch.Tensor:
    """Samples diffusion timesteps from a logit-normal distribution (EDM-style).

    Biases training toward mid-noise timesteps that determine structure.

    Args:
        batch_size: Number of timesteps to sample.
        device: Device to sample on.
        num_timesteps: Total diffusion timesteps in the schedule.
        mean: Logit-normal distribution mean.
        std: Logit-normal distribution standard deviation.

    Returns:
        LongTensor of shape [batch_size], values in [0, num_timesteps - 1].
    """
    u = torch.randn(batch_size, device=device)
    t_normalized = torch.sigmoid(mean + std * u)
    return (t_normalized * num_timesteps).clamp(0, num_timesteps - 1).long()


def train_svdrr(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # 1. Download/Load Base Model
    local_base_path = os.path.join(svdrr_dir, "models", "base_model", "256")
    if not os.path.exists(local_base_path):
        logger.info(f"[SV-DRR] Base model missing at {local_base_path}; downloading from HF Hub...")
        from huggingface_hub import snapshot_download

        SVDRR_BASE_MODEL_REVISION = "151bc201b16fa8d6d01853c2036f7d6b354d50ba"
        snapshot_download(
            repo_id="xiechun-tsukuba/svdrr-dit-fb-256",
            revision=SVDRR_BASE_MODEL_REVISION,
            local_dir=local_base_path,
        )

    model_id = local_base_path

    # 2. Build Pipeline & Optimizers
    cc_projection = CCProjection.from_config(model_id, subfolder="cc_projection")
    pipe = SvdrrDiTPipeline.from_pretrained(
        model_id,
        cc_projection=cc_projection,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
    ).to(device)

    transformer = pipe.transformer
    transformer.train()

    cc_projection = pipe.cc_projection
    cc_projection.train()

    # Reference DiT optimizer config: beta2=0.95 and zero weight decay.
    optimizer = optim.AdamW(
        [
            {"params": transformer.parameters(), "lr": args.lr},
            {"params": cc_projection.parameters(), "lr": 10.0 * args.lr},
        ],
        betas=(0.9, 0.95),
        weight_decay=0.0,
        eps=1e-8,
    )

    # IDDPM training objective with learned variance and hybrid SNR parameterization.
    diffusion = IDDPM(
        timestep_respacing="",
        noise_schedule="linear",
        use_kl=False,
        sigma_small=False,
        predict_xstart=False,
        learn_sigma=True,
        pred_sigma=True,
        rescale_learned_sigmas=False,
        diffusion_steps=pipe.scheduler.config.num_train_timesteps,
        snr=True,
        return_startx=False,
    )

    # Maintain EMA shadow weights of the transformer for inference.
    ema_transformer = copy.deepcopy(transformer)
    for p in ema_transformer.parameters():
        p.requires_grad_(False)
    ema_transformer.eval()

    @torch.no_grad()
    def update_ema(decay: float) -> None:
        for ema_p, p in zip(ema_transformer.parameters(), transformer.parameters()):
            ema_p.mul_(decay).add_(p.detach(), alpha=1 - decay)

    # 3. Load & Filter Dataset Patients
    rendered_dir = os.path.join(BASE_DIR, "..", "datasets", "pre_rendered")
    train_patient_dirs, val_patient_dirs = get_train_val_patient_dirs(rendered_dir)

    if args.max_train_samples:
        train_patient_dirs = train_patient_dirs[:args.max_train_samples]

    logger.info(f"Found {len(train_patient_dirs)} train cases and {len(val_patient_dirs)} val cases for SV-DRR.")

    if getattr(args, "viz_patient", None):
        matching = [d for d in val_patient_dirs + train_patient_dirs if os.path.basename(d) == args.viz_patient or d.endswith(args.viz_patient)]
        viz_patient_dir = matching[0] if matching else (val_patient_dirs[0] if val_patient_dirs else None)
    else:
        viz_patient_dir = val_patient_dirs[0] if val_patient_dirs else None

    if viz_patient_dir:
        logger.info(f"Using '{os.path.basename(viz_patient_dir)}' for live multi-view visualization panels.")

    # 4. Cache VAE Latents & CLIP Embeddings
    CACHE_BATCH = 16

    def cache_dataset(patient_dirs: list[str]) -> list[dict]:
        cached = []
        with torch.no_grad():
            for pat_path in patient_dirs:
                views_dir = os.path.join(pat_path, "views")
                view_files = sorted(f for f in os.listdir(views_dir) if f.endswith(".png")) if os.path.exists(views_dir) else []

                if len(view_files) < 2:
                    continue

                angles = np.linspace(0.0, 360.0, len(view_files))

                latents_list = []
                img_embeds_list = []

                for start in range(0, len(view_files), CACHE_BATCH):
                    chunk_files = view_files[start : start + CACHE_BATCH]
                    imgs_pil = [Image.open(os.path.join(views_dir, vf)).convert("RGB") for vf in chunk_files]

                    img_tensor = torch.stack([TF.to_tensor(im) for im in imgs_pil]).to(device) * 2.0 - 1.0
                    latents = (pipe.vae.encode(img_tensor).latent_dist.sample() * pipe.vae.config.scaling_factor).cpu()
                    img_embeds = pipe._encode_image(
                        image=imgs_pil, device=device, num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                    ).cpu()

                    latents_list.append(latents)
                    img_embeds_list.append(img_embeds)

                cached.append({
                    "latents": torch.cat(latents_list, dim=0),
                    "img_embeds": torch.cat(img_embeds_list, dim=0),
                    "angles": angles,
                })

        return cached

    train_cached = cache_dataset(train_patient_dirs)
    val_cached = cache_dataset(val_patient_dirs[:args.max_val_samples] if args.max_val_samples else val_patient_dirs)

    # 5. Data Sampling Helpers
    def sample_pair(item: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Sample a random (source_latent, target_latent, source_img_embed, relative_angle) tuple."""
        n_views = item["latents"].shape[0]
        src_idx, tgt_idx = np.random.choice(n_views, size=2, replace=False)

        src_latent = item["latents"][src_idx : src_idx + 1]
        tgt_latent = item["latents"][tgt_idx : tgt_idx + 1]
        src_img_embed = item["img_embeds"][src_idx : src_idx + 1]

        rel_angle = float(item["angles"][tgt_idx] - item["angles"][src_idx])
        return src_latent, tgt_latent, src_img_embed, rel_angle

    def sample_batch(batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[float]]:
        """Draw a batch of random pairs from cached patient volumes."""
        items = [train_cached[i] for i in np.random.choice(len(train_cached), size=batch_size, replace=True)]
        pairs = [sample_pair(item) for item in items]

        src_latents = torch.cat([p[0] for p in pairs], dim=0)
        tgt_latents = torch.cat([p[1] for p in pairs], dim=0)
        src_img_embeds = torch.cat([p[2] for p in pairs], dim=0)

        rel_angles = [p[3] for p in pairs]
        return src_latents, tgt_latents, src_img_embeds, rel_angles

    # Fixed validation pairs
    val_rng = np.random.RandomState(42)
    val_pairs = []

    for item in val_cached:
        n_views = item["latents"].shape[0]
        tgt_idx = val_rng.randint(1, n_views)

        val_pairs.append((
            item["latents"][0:1],
            item["latents"][tgt_idx : tgt_idx + 1],
            item["img_embeds"][0:1],
            float(item["angles"][tgt_idx] - item["angles"][0]),
        ))

    # 6. Checkpoint Directory & TensorBoard Setup
    ckpt_dir = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    latest_ckpt_path = os.path.join(ckpt_dir, "svdrr_checkpoint.pt")
    best_ckpt_path = os.path.join(ckpt_dir, "svdrr_best.pt")
    writer = SummaryWriter(log_dir=os.path.join(ckpt_dir, "tensorboard", "svdrr"))

    logger.info(f"Starting SV-DRR fine-tuning for {args.epochs} epochs...")

    best_loss = float("inf")

    # LR schedule with linear warmup and cosine decay to 10% peak LR.
    effective_batch = args.batch_size * args.accum_steps
    steps_per_epoch = max(1, len(train_cached) // effective_batch)
    max_train_steps = args.epochs * steps_per_epoch
    min_lr_ratio = 0.1

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = min(1.0, (step - args.warmup_steps) / max(1, max_train_steps - args.warmup_steps))
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 7. Training Epoch Loop
    for epoch in range(1, args.epochs + 1):

        # --- Training Phase --- #
        transformer.train()
        cc_projection.train()

        epoch_train_loss = torch.zeros((), device=device)
        num_train_steps = 0

        for _ in range(steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)

            for _ in range(args.accum_steps):
                src_latents, tgt_latents, src_img_embeds, rel_angles = sample_batch(args.batch_size)
                src_latents = src_latents.to(device)
                tgt_latents = tgt_latents.to(device)
                B = src_latents.shape[0]

                pose_prompt_embeds = pipe._encode_pose(
                    pose=[[0, -ra, 0] for ra in rel_angles], device=device, num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )
                raw_cond = torch.cat([src_img_embeds.to(device), pose_prompt_embeds], dim=-1)
                cc_emb = cc_projection(raw_cond)
                cond_latents = src_latents

                # CFG dropout: randomly drop conditioning for classifier-free guidance.
                if args.conditioning_dropout_prob > 0:
                    drop_mask = torch.rand(B, device=device) < args.conditioning_dropout_prob
                    if drop_mask.any():
                        cc_emb = cc_emb.clone()
                        cond_latents = cond_latents.clone()
                        cc_emb[drop_mask] = 0
                        cond_latents[drop_mask] = 0

                timesteps = sample_timesteps_logit_normal(
                    B, device, pipe.scheduler.config.num_train_timesteps
                )

                added_cond_kwargs = {
                    "resolution": torch.tensor([[256, 256]], device=device).expand(B, -1),
                    "aspect_ratio": torch.tensor([[1.0]], device=device).expand(B, -1),
                }
                model_kwargs = {
                    "encoder_hidden_states": cc_emb,
                    "latents_concat": cond_latents,
                    "added_cond_kwargs": added_cond_kwargs,
                }

                loss_dict = diffusion.training_losses_diffusers(
                    transformer, x_start=tgt_latents, timestep=timesteps,
                    model_kwargs=model_kwargs,
                )
                loss = loss_dict["loss"].mean() / args.accum_steps
                loss.backward()

                epoch_train_loss += (loss.detach() * args.accum_steps)
                num_train_steps += 1

            if args.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(transformer.parameters()) + list(cc_projection.parameters()),
                    max_norm=args.gradient_clip_val,
                )
            optimizer.step()
            lr_scheduler.step()
            update_ema(args.ema_decay)

        avg_train_loss = (epoch_train_loss / max(1, num_train_steps)).item()

        # --- Validation Phase --- #
        transformer.eval()
        cc_projection.eval()

        epoch_val_loss = torch.zeros((), device=device)
        num_val_steps = 0

        val_generator = torch.Generator(device=device).manual_seed(42)

        with torch.no_grad():
            for src_lat, tgt_lat, src_embed, rel_ang in val_pairs:
                src_latents = src_lat.to(device)
                tgt_latents = tgt_lat.to(device)

                pose_prompt_embeds = pipe._encode_pose(
                    pose=[[0, -rel_ang, 0]], device=device, num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )
                raw_cond = torch.cat([src_embed.to(device), pose_prompt_embeds], dim=-1)
                cc_emb = cc_projection(raw_cond)

                # Uniform timestep sampling for unbiased validation loss evaluation.
                timesteps = torch.randint(
                    0, pipe.scheduler.config.num_train_timesteps, (1,),
                    generator=val_generator, device=device
                ).long()
                noise = torch.randn(tgt_latents.shape, generator=val_generator, device=device)

                added_cond_kwargs = {
                    "resolution": torch.tensor([[256, 256]], device=device),
                    "aspect_ratio": torch.tensor([[1.0]], device=device),
                }
                model_kwargs = {
                    "encoder_hidden_states": cc_emb,
                    "latents_concat": src_latents,
                    "added_cond_kwargs": added_cond_kwargs,
                }

                loss_dict = diffusion.training_losses_diffusers(
                    transformer, x_start=tgt_latents, timestep=timesteps,
                    model_kwargs=model_kwargs, noise=noise,
                )
                val_loss = loss_dict["loss"].mean()

                epoch_val_loss += val_loss.detach()
                num_val_steps += 1

        avg_val_loss = (epoch_val_loss / max(1, num_val_steps)).item()

        # --- Logging & TensorBoard --- #
        current_lr = lr_scheduler.get_last_lr()[0]
        logger.info(f"Epoch [{epoch}/{args.epochs}] - Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e}")
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("LR/transformer", current_lr, epoch)

        if args.viz_every > 0 and (epoch % args.viz_every == 0 or epoch == args.epochs) and viz_patient_dir is not None:
            pa_path = os.path.join(viz_patient_dir, "pa.png")
            gt_views_dir = os.path.join(viz_patient_dir, "views")

            if os.path.exists(pa_path):
                indices = epoch_rotating_view_indices(epoch, args.viz_views)

                def _save_viz_ckpt(path: str) -> None:
                    torch.save({
                        "transformer": {k: v.detach().cpu() for k, v in transformer.state_dict().items()},
                        "cc_projection": {k: v.detach().cpu() for k, v in cc_projection.state_dict().items()},
                        "ema_transformer": {k: v.detach().cpu() for k, v in ema_transformer.state_dict().items()},
                    }, path)

                tmp_ckpt_path = os.path.join(ckpt_dir, f"_viz_tmp_svdrr_{os.getpid()}.pt")
                input_xray = TF.to_tensor(Image.open(pa_path).convert("L")).unsqueeze(0).to(device)

                multiview_grid = visualize_live_checkpoint(
                    SVDRRWrapper, _save_viz_ckpt, tmp_ckpt_path, input_xray, gt_views_dir, indices,
                )

                if multiview_grid is not None:
                    writer.add_image("Images/val_multiview", multiview_grid, epoch)

                transformer.train()
                cc_projection.train()

        # --- Checkpoint Saving --- #
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss

            best_state_dict = {
                "transformer": {k: v.cpu().clone() for k, v in transformer.state_dict().items()},
                "cc_projection": {k: v.cpu().clone() for k, v in cc_projection.state_dict().items()},
                "ema_transformer": {k: v.cpu().clone() for k, v in ema_transformer.state_dict().items()},
            }
            logger.info(f"New best model recorded (Val Loss: {best_loss:.6f})")

    latest_state_dict = {
        "transformer": {k: v.cpu().clone() for k, v in transformer.state_dict().items()},
        "cc_projection": {k: v.cpu().clone() for k, v in cc_projection.state_dict().items()},
        "ema_transformer": {k: v.cpu().clone() for k, v in ema_transformer.state_dict().items()},
    }

    if 'best_state_dict' in locals():
        torch.save(best_state_dict, best_ckpt_path)
    else:
        torch.save(latest_state_dict, best_ckpt_path)

    torch.save(latest_state_dict, latest_ckpt_path)
    writer.close()
    logger.info(f"SV-DRR training complete. Checkpoints saved to {latest_ckpt_path} and {best_ckpt_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SV-DRR Baseline Trainer")

    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Max validation samples")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Max training samples")
    parser.add_argument("--viz_every", type=int, default=10, help="Log multi-view TensorBoard panel every N epochs (0 to disable)")
    parser.add_argument("--viz_views", type=int, default=6, help="Number of azimuths per multi-view panel")
    parser.add_argument("--viz_patient", type=str, default=None, help="Specific patient ID/directory name to use for multi-view visualization panel (e.g. mela_0001)")
    parser.add_argument("--batch_size", type=int, default=16, help="Micro-batch size per optimization step")
    parser.add_argument("--accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--conditioning_dropout_prob", type=float, default=0.05, help="Per-sample probability of dropping conditioning during training, for classifier-free guidance capability")
    parser.add_argument("--ema_decay", type=float, default=0.9995, help="EMA decay rate for the transformer weights")
    parser.add_argument("--warmup_steps", type=int, default=500, help="LR warmup steps before cosine decay begins")
    parser.add_argument("--gradient_clip_val", type=float, default=0.5, help="Gradient clipping max norm (0 to disable)")

    args = parser.parse_args()
    train_svdrr(args)
