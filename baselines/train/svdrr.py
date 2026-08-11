"""SV-DRR (MICCAI 2025) Training Script."""

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
svdrr_dir = os.path.join(BASE_DIR, "cloned", "SV-DRR")
sys.path.insert(0, svdrr_dir)

from pipeline_svdrr_DiT import CCProjection, SvdrrDiTPipeline  # type: ignore

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

    optimizer = optim.AdamW(
        [
            {"params": transformer.parameters(), "lr": args.lr},
            {"params": cc_projection.parameters(), "lr": 10.0 * args.lr},
        ],
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )
    criterion = nn.MSELoss()

    # 3. Load & Filter Dataset Patients
    rendered_dir = os.path.join(BASE_DIR, "..", "datasets", "pre_rendered")
    train_patient_dirs, val_patient_dirs = get_train_val_patient_dirs(rendered_dir)

    if args.max_train_samples:
        train_patient_dirs = train_patient_dirs[:args.max_train_samples]

    logger.info(f"Found {len(train_patient_dirs)} train cases and {len(val_patient_dirs)} val cases for SV-DRR.")

    viz_patient_dir = val_patient_dirs[0] if val_patient_dirs else None

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
    patience_counter = 0

    # 7. Training Epoch Loop
    for epoch in range(1, args.epochs + 1):

        # --- Training Phase --- #
        transformer.train()
        cc_projection.train()

        epoch_train_loss = torch.zeros((), device=device)
        num_train_steps = 0

        effective_batch = args.batch_size * args.accum_steps
        steps_per_epoch = max(1, len(train_cached) // effective_batch)

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

                timesteps = torch.randint(
                    0, pipe.scheduler.config.num_train_timesteps, (B,), device=device
                ).long()
                noise = torch.randn_like(tgt_latents)
                noisy_latents = pipe.scheduler.add_noise(tgt_latents, noise, timesteps)

                added_cond_kwargs = {
                    "resolution": torch.tensor([[256, 256]], device=device).expand(B, -1),
                    "aspect_ratio": torch.tensor([[1.0]], device=device).expand(B, -1),
                }

                if transformer.config.in_channels == 8:
                    hidden_states = torch.cat([noisy_latents, src_latents], dim=1)
                else:
                    hidden_states = noisy_latents

                noise_pred = transformer(
                    hidden_states=hidden_states,
                    encoder_hidden_states=cc_emb,
                    timestep=timesteps,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                if getattr(pipe.scheduler.config, "prediction_type", "epsilon") == "v_prediction":
                    target = pipe.scheduler.get_velocity(tgt_latents, noise, timesteps)
                else:
                    target = noise

                loss = criterion(noise_pred[:, :4], target) / args.accum_steps
                loss.backward()

                epoch_train_loss += (loss.detach() * args.accum_steps)
                num_train_steps += 1

            optimizer.step()

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

                timesteps = torch.randint(
                    0, pipe.scheduler.config.num_train_timesteps, (1,),
                    generator=val_generator, device=device
                ).long()

                noise = torch.randn(tgt_latents.shape, generator=val_generator, device=device)
                noisy_latents = pipe.scheduler.add_noise(tgt_latents, noise, timesteps)

                added_cond_kwargs = {
                    "resolution": torch.tensor([[256, 256]], device=device),
                    "aspect_ratio": torch.tensor([[1.0]], device=device),
                }

                if transformer.config.in_channels == 8:
                    hidden_states = torch.cat([noisy_latents, src_latents], dim=1)
                else:
                    hidden_states = noisy_latents

                noise_pred = transformer(
                    hidden_states=hidden_states,
                    encoder_hidden_states=cc_emb,
                    timestep=timesteps,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                if getattr(pipe.scheduler.config, "prediction_type", "epsilon") == "v_prediction":
                    target = pipe.scheduler.get_velocity(tgt_latents, noise, timesteps)
                else:
                    target = noise

                val_loss = criterion(noise_pred[:, :4], target)

                epoch_val_loss += val_loss.detach()
                num_val_steps += 1

        avg_val_loss = (epoch_val_loss / max(1, num_val_steps)).item()

        # --- Logging & TensorBoard --- #
        logger.info(f"Epoch [{epoch}/{args.epochs}] - Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)

        if args.viz_every > 0 and (epoch % args.viz_every == 0 or epoch == args.epochs) and viz_patient_dir is not None:
            pa_path = os.path.join(viz_patient_dir, "pa.png")
            gt_views_dir = os.path.join(viz_patient_dir, "views")

            if os.path.exists(pa_path):
                indices = epoch_rotating_view_indices(epoch, args.viz_views)

                def _save_viz_ckpt(path: str) -> None:
                    torch.save({
                        "transformer": {k: v.detach().cpu() for k, v in transformer.state_dict().items()},
                        "cc_projection": {k: v.detach().cpu() for k, v in cc_projection.state_dict().items()},
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

        # --- Checkpoint Saving & Early Stopping --- #
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0

            best_state_dict = {
                "transformer": {k: v.cpu().clone() for k, v in transformer.state_dict().items()},
                "cc_projection": {k: v.cpu().clone() for k, v in cc_projection.state_dict().items()},
            }
            logger.info(f"New best model recorded (Val Loss: {best_loss:.6f})")
        else:
            patience_counter += 1
            if args.patience > 0 and patience_counter >= args.patience:
                logger.info(f"Early stopping triggered at epoch {epoch} (patience={args.patience}).")
                break

    latest_state_dict = {
        "transformer": {k: v.cpu().clone() for k, v in transformer.state_dict().items()},
        "cc_projection": {k: v.cpu().clone() for k, v in cc_projection.state_dict().items()},
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
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience (0 to disable)")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Max validation samples")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Max training samples")
    parser.add_argument("--viz_every", type=int, default=10, help="Log multi-view TensorBoard panel every N epochs (0 to disable)")
    parser.add_argument("--viz_views", type=int, default=6, help="Number of azimuths per multi-view panel")
    parser.add_argument("--batch_size", type=int, default=16, help="Micro-batch size per optimization step")
    parser.add_argument("--accum_steps", type=int, default=4, help="Gradient accumulation steps")

    args = parser.parse_args()
    train_svdrr(args)
