"""SV-DRR (MICCAI 2025) Training Script for CosmosXRay360

Fine-tunes the SV-DRR DiT transformer model on Cross-Dataset split.
Saves checkpoints to baselines/checkpoints/ (both best and latest).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import torchvision.transforms.functional as TF

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
svdrr_dir = os.path.join(BASE_DIR, "cloned", "SV-DRR")
sys.path.insert(0, svdrr_dir)

from pipeline_svdrr_DiT import SvdrrDiTPipeline, CCProjection

from models.utils import get_train_val_patient_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
)
logger = logging.getLogger(__name__)


def train_svdrr(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Loading SvdrrDiTPipeline.from_pretrained() directly from the bare Hub ID
    # (rather than a local snapshot) fails on fresh environments: diffusers'
    # custom-component resolution can't locate cc_projection/pipeline_svdrr_DiT.py
    # inside the Hub repo's cc_projection/ subfolder (it only lives at the repo
    # root), raising "does not exist ... and is not a module in
    # 'diffusers/pipelines'". snapshot_download sidesteps this entirely -- it's a
    # plain file copy with no custom-pipeline resolution -- so always materialize
    # a local snapshot first and load from that, matching how this already works
    # when the weights are pre-provisioned locally.
    local_base_path = os.path.join(svdrr_dir, "models", "base_model", "256")
    if not os.path.exists(local_base_path):
        logger.info(f"[SV-DRR] Local base model not found at {local_base_path}; downloading snapshot from HF Hub...")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="xiechun-tsukuba/svdrr-dit-fb-256", local_dir=local_base_path)
    model_id = local_base_path

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

    # Reference train_svdrr_DiT.py trains cc_projection jointly with the transformer,
    # at 10x the transformer's learning rate (it's a lightweight conditioning
    # projection, not the frozen CLIP image encoder) -- see optimizer param groups at
    # train_svdrr_DiT.py:874-882. It must stay trainable and in the compute graph,
    # unlike the frozen VAE/CLIP encoder, so its forward pass cannot be cached
    # up-front the way the (expensive, frozen) VAE/CLIP encodes are below.
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

    rendered_dir = os.path.join(BASE_DIR, "..", "datasets", "pre_rendered")
    train_patient_dirs, val_patient_dirs = get_train_val_patient_dirs(rendered_dir)

    logger.info(f"Found {len(train_patient_dirs)} train cases and {len(val_patient_dirs)} val cases for SV-DRR.")

    # Cache VAE latents (frozen VAE) and the raw pre-cc_projection CLIP image
    # embedding (frozen CLIP image_encoder) for *every* one of a patient's 93
    # views/*.png -- not just pa.png/lat.png -- so any view can serve as either
    # the source or the target during training (see train_svdrr's docstring for
    # why). Pose embedding is cheap pure trig (no network forward pass), so it's
    # computed fresh per sampled pair below rather than cached. cc_projection
    # itself must also run fresh every step so gradients reach it -- only its
    # cheap *inputs* are cached, not its output.
    # Encoding one view at a time (93/patient) measured at ~80ms/view -- dominated
    # by per-call overhead (kernel launch/sync), not actual VAE/CLIP compute, since
    # both natively accept a batch (AutoencoderKL.encode is standard batched;
    # _encode_image explicitly supports a list of PIL images, see
    # pipeline_svdrr_DiT.py). Batching in chunks turns ~36,000 sequential
    # single-image calls (93 x ~387 train patients) into ~2,300 batched ones.
    CACHE_BATCH = 16

    def cache_dataset(patient_dirs: list[str]) -> list[dict]:
        cached = []
        with torch.no_grad():
            for pat_path in patient_dirs:
                views_dir = os.path.join(pat_path, "views")
                view_files = sorted(f for f in os.listdir(views_dir) if f.endswith(".png")) if os.path.exists(views_dir) else []
                if len(view_files) < 2:
                    continue

                # endpoint=True matches pre_render_diffdrr.py's own
                # torch.linspace(0, 360, N) convention the views were rendered with.
                angles = np.linspace(0.0, 360.0, len(view_files))

                latents_list = []
                img_embeds_list = []
                for start in range(0, len(view_files), CACHE_BATCH):
                    chunk_files = view_files[start : start + CACHE_BATCH]
                    imgs_pil = [Image.open(os.path.join(views_dir, vf)).convert("RGB") for vf in chunk_files]

                    # Reference pipeline's own prepare_latents asserts/requires [-1, 1]
                    # input for vae.encode (see pipeline_svdrr_DiT.py: raises if a
                    # passed tensor isn't in [-1, 1], and rescales PIL/np images via
                    # `image / 127.5 - 1.0`) -- TF.to_tensor yields [0, 1], so it must
                    # be rescaled here too.
                    img_tensor = torch.stack([TF.to_tensor(im) for im in imgs_pil]).to(device) * 2.0 - 1.0
                    latents = (pipe.vae.encode(img_tensor).latent_dist.sample() * pipe.vae.config.scaling_factor).cpu()
                    img_embeds = pipe._encode_image(
                        image=imgs_pil, device=device, num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                    ).cpu()
                    latents_list.append(latents)
                    img_embeds_list.append(img_embeds)

                cached.append({
                    "latents": torch.cat(latents_list, dim=0),        # (n_views, 4, H, W)
                    "img_embeds": torch.cat(img_embeds_list, dim=0),  # (n_views, 1, 768)
                    "angles": angles,                                 # (n_views,)
                })
        return cached

    train_cached = cache_dataset(train_patient_dirs)
    val_cached = cache_dataset(val_patient_dirs[:args.max_val_samples] if args.max_val_samples else val_patient_dirs)

    def sample_pair(item: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Samples a random (source_latent, target_latent, source_img_embed, relative_angle)
        tuple with both indices random and distinct, so the model sees the full space
        of (source, target) pairs -- ~n_views x (n_views-1) per patient instead of just
        the one fixed PA->LAT pair, which matters a lot at this project's
        few-hundred-patient scale. Relative angle still spans the same full [0, 360)
        range either way, since angle(target) - angle(source) does not depend on
        which index is picked as source. (Validation uses a separately, seeded
        frontal-source pairing -- see val_pairs below -- so it stays representative
        of the actual deployment task.)
        """
        n_views = item["latents"].shape[0]
        src_idx, tgt_idx = np.random.choice(n_views, size=2, replace=False)
        src_latent = item["latents"][src_idx : src_idx + 1]
        tgt_latent = item["latents"][tgt_idx : tgt_idx + 1]
        src_img_embed = item["img_embeds"][src_idx : src_idx + 1]
        rel_angle = float(item["angles"][tgt_idx] - item["angles"][src_idx])
        return src_latent, tgt_latent, src_img_embed, rel_angle

    # Fix validation (source=frontal, target) pairs once with a seeded RNG --
    # otherwise sample_pair's random target would change every epoch, making
    # avg_val_loss an unstable signal for best-checkpoint selection.
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

    ckpt_dir = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    latest_ckpt_path = os.path.join(ckpt_dir, "svdrr_checkpoint.pt")
    best_ckpt_path = os.path.join(ckpt_dir, "svdrr_best.pt")
    writer = SummaryWriter(log_dir=os.path.join(ckpt_dir, "tensorboard", "svdrr"))

    logger.info(f"Starting SV-DRR fine-tuning for {args.epochs} epochs...")

    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # 1. Training Phase
        transformer.train()
        cc_projection.train()
        # Accumulated as a device tensor (not `.item()`-ed per step) to avoid a
        # CUDA sync every step; `.item()` is called once, after the loop.
        epoch_train_loss = torch.zeros((), device=device)
        num_train_steps = 0

        for item in train_cached:
            optimizer.zero_grad(set_to_none=True)

            # Random (source, target) pair every step -- not just the fixed
            # PA->LAT pair -- so the model sees the full space of relative-pose
            # transformations (see sample_pair's docstring).
            src_latents, tgt_latents, src_img_embed, rel_angle = sample_pair(item)
            src_latents = src_latents.to(device)
            tgt_latents = tgt_latents.to(device)

            pose_prompt_embeds = pipe._encode_pose(
                pose=np.array([0, -rel_angle, 0]), device=device, num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            raw_cond = torch.cat([src_img_embed.to(device), pose_prompt_embeds], dim=-1)
            # Run fresh every step (not cached) so gradients reach cc_projection.
            cc_emb = cc_projection(raw_cond)

            # Must be resampled per step (standard diffusion training): a fixed
            # timestep only ever teaches the model to denoise at that one noise
            # level, while inference (scheduler.set_timesteps(20)) samples across
            # the entire schedule.
            timesteps = torch.randint(
                0, pipe.scheduler.config.num_train_timesteps, (1,), device=device
            ).long()
            noise = torch.randn_like(tgt_latents)
            noisy_latents = pipe.scheduler.add_noise(tgt_latents, noise, timesteps)

            added_cond_kwargs = {
                "resolution": torch.tensor([[256, 256]], device=device),
                "aspect_ratio": torch.tensor([[1.0]], device=device),
            }

            # Matches the reference pipeline's own conditional exactly
            # (pipeline_svdrr_DiT.py:879-880): channel-concat conditioning on the
            # source-view latent only applies to the 8-channel transformer variant;
            # the 4-channel variant (e.g. xiechun-tsukuba/svdrr-dit-fb-256, the
            # checkpoint this repo downloads) conditions purely via cross-attention
            # (cc_emb) and must NOT be concatenated, or the channel count would be
            # wrong for either architecture.
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

            loss = criterion(noise_pred[:, :4], target)

            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.detach()
            num_train_steps += 1

        avg_train_loss = (epoch_train_loss / max(1, num_train_steps)).item()

        # 2. Validation Phase
        transformer.eval()
        cc_projection.eval()
        epoch_val_loss = torch.zeros((), device=device)
        num_val_steps = 0
        val_sample_img = None

        val_generator = torch.Generator(device=device).manual_seed(42)

        with torch.no_grad():
            for idx, (src_latents, tgt_latents, src_img_embed, rel_angle) in enumerate(val_pairs):
                src_latents = src_latents.to(device)
                tgt_latents = tgt_latents.to(device)

                pose_prompt_embeds = pipe._encode_pose(
                    pose=np.array([0, -rel_angle, 0]), device=device, num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )
                raw_cond = torch.cat([src_img_embed.to(device), pose_prompt_embeds], dim=-1)
                cc_emb = cc_projection(raw_cond)

                timesteps = torch.randint(
                    0, pipe.scheduler.config.num_train_timesteps, (1,),
                    generator=val_generator, device=device,
                ).long()
                noise = torch.randn(tgt_latents.shape, generator=val_generator, device=device)
                noisy_latents = pipe.scheduler.add_noise(tgt_latents, noise, timesteps)

                added_cond_kwargs = {
                    "resolution": torch.tensor([[256, 256]], device=device),
                    "aspect_ratio": torch.tensor([[1.0]], device=device),
                }

                # See matching comment in the training loop above.
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

                if idx == 0:
                    alpha_bar = pipe.scheduler.alphas_cumprod[timesteps[0]].to(device)
                    sqrt_alpha_bar = torch.sqrt(alpha_bar)
                    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)
                    if getattr(pipe.scheduler.config, "prediction_type", "epsilon") == "v_prediction":
                        pred_lat = sqrt_alpha_bar * noisy_latents[0] - sqrt_one_minus_alpha_bar * noise_pred[0, :4]
                    else:
                        pred_lat = (noisy_latents[0] - sqrt_one_minus_alpha_bar * noise_pred[0, :4]) / (sqrt_alpha_bar + 1e-8)
                    pred_raw = pipe.vae.decode(pred_lat.unsqueeze(0) / pipe.vae.config.scaling_factor).sample[0]
                    targ_raw = pipe.vae.decode(tgt_latents / pipe.vae.config.scaling_factor).sample[0]
                    pred_img = ((pred_raw + 1.0) / 2.0).clamp(0, 1)
                    targ_img = ((targ_raw + 1.0) / 2.0).clamp(0, 1)
                    val_sample_img = torch.cat([pred_img, targ_img], dim=-1).cpu()

                epoch_val_loss += val_loss.detach()
                num_val_steps += 1

        avg_val_loss = (epoch_val_loss / max(1, num_val_steps)).item()
        logger.info(f"Epoch [{epoch}/{args.epochs}] - Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        if val_sample_img is not None:
            writer.add_image("Images/val_pred_gt", val_sample_img, epoch)

        # Track best model in memory. cc_projection is saved alongside the
        # transformer -- it's jointly trained above, so a transformer-only
        # checkpoint would silently discard its learned weights on reload.
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_state_dict = {
                "transformer": {k: v.cpu().clone() for k, v in transformer.state_dict().items()},
                "cc_projection": {k: v.cpu().clone() for k, v in cc_projection.state_dict().items()},
            }
            logger.info(f"New best model recorded (Val Loss: {best_loss:.6f})")

    # Save best and latest checkpoints at the end of training
    latest_state_dict = {
        "transformer": transformer.state_dict(),
        "cc_projection": cc_projection.state_dict(),
    }
    if 'best_state_dict' in locals():
        torch.save(best_state_dict, best_ckpt_path)
    else:
        torch.save(latest_state_dict, best_ckpt_path)
    torch.save(latest_state_dict, latest_ckpt_path)
    writer.close()
    logger.info(f"SV-DRR training complete. Checkpoints saved to {latest_ckpt_path} and {best_ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SV-DRR Baseline Trainer")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate (Official DiT fine-tuning: 5e-6)")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Max validation samples (default: None for full val)")
    args = parser.parse_args()
    train_svdrr(args)
