"""Inference pipeline for Cosmos Predict 2.5."""

from __future__ import annotations

import contextlib
import os
import json
from pathlib import Path
from re import DEBUG
from typing import Optional, Union

import numpy as np

import torch

from PIL import Image

from cosmos_predict2._src.imaginaire.utils.checkpointer import non_strict_load_model
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.configs.video2world.defaults.conditioner import Video2WorldCondition
from cosmos_predict2._src.predict2.models.fm_solvers_unipc import FlowUniPCMultistepScheduler
from cosmos_predict2._src.predict2.networks.minimal_v1_lvg_dit import MinimalV1LVGDiT
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import SACConfig
from cosmos_predict2._src.predict2.tokenizers.wan2pt1 import Wan2pt1VAEInterface

from predict2_5.constants import (
    COSMOS_2B_PRETRAINED_UUID,
    COSMOS_TOKENIZER_UUID,
    CROSSATTN_PROJ_IN_CHANNELS,
    IMG_HEIGHT,
    IMG_WIDTH,
    NUM_FRAMES,
    PROMPTS,
)
from predict2_5.hf import download_wan_vae_tokenizer, hf_download, resolve_hf_uri
from predict2_5.text_encoder import CR1TextEncoder
from predict2_5.utils import arch_invariant_rand, fix_rope_buffers, move_tokenizer_to_device, get_logger, is_uuid_format, safe_torch_load


# --- Logger --- #
log = get_logger(__name__)

DEFAULT_PROMPT = PROMPTS[0] if PROMPTS else None


class Inferencer:
    """Inference pipeline for post-trained Cosmos Predict 2.5."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        model_size: str = "2B",
        tokenizer_path: Optional[str] = None,
        text_encoder_path: Optional[str] = None,
        device: str = "cuda",
        cpu_offload: bool = False
    ) -> None:
        """
        Initialize inferencer by loading DiT, tokenizer, and text encoder.

        Args:
            checkpoint_path: Path, UUID, or hf:// URI for DiT checkpoint. Defaults to COSMOS_2B_PRETRAINED_UUID if None.
            config_path: Path or URI for DiT config JSON. If None, will attempt to resolve from checkpoint directory.
            model_size: Model size label (e.g., "2B").
            tokenizer_path: Path, UUID, or URI for VAE tokenizer checkpoint. Defaults to COSMOS_TOKENIZER_UUID.
            text_encoder_path: Path, UUID, or URI for text encoder checkpoint. If provided,
                enables on-the-fly text-embedding fallback when precomputed embeddings are missing.
            device: Device for inference (default: "cuda"). Will fall back to "cpu" if CUDA is unavailable.
            cpu_offload: Whether to offload model weights to CPU when not actively denoising (default: False).
        """
        # Determine device with fallback to CPU if CUDA is not available
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        
        # Runtime compute dtype used by autocast during denoising.
        self.runtime_dtype = torch.bfloat16

        # CPU offload flag
        self.cpu_offload = bool(cpu_offload)

        # Model size: 2B, 7B, or 14B
        self.model_size = model_size

        # Resolve all artifact paths (checkpoint, config, tokenizer, text encoder) to local file paths
        target_ckpt = checkpoint_path or COSMOS_2B_PRETRAINED_UUID
        resolved_checkpoint_path = self._resolve_artifact(
            spec=target_ckpt, 
            artifact_name="checkpoint"
        )
        resolved_config_path = self._resolve_config_path(
            config_path=config_path,
            checkpoint_path=resolved_checkpoint_path,
        )
        resolved_tokenizer_path = self._resolve_artifact(
            spec=tokenizer_path or COSMOS_TOKENIZER_UUID,
            artifact_name="tokenizer",
        )
        resolved_text_encoder_path = (
            self._resolve_artifact(
                spec=text_encoder_path, 
                artifact_name="text_encoder"
            )
            if text_encoder_path
            else None
        )

        self.checkpoint_path = resolved_checkpoint_path
        self.config_path = resolved_config_path

        # Initialize UniPC scheduler for sampling
        self.sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=1,
            use_dynamic_shifting=False,
        )

        # Initialize Cosmos Tokenizer
        self.tokenizer = self._build_tokenizer(resolved_tokenizer_path)
        move_tokenizer_to_device(self.tokenizer, self.device)

        # Initialize DiT model
        self.dit = self._build_dit(
            config_path=resolved_config_path,
            checkpoint_path=resolved_checkpoint_path,
        )

        # Initialize Cosmos-Reason 1.0 Text Encoder
        self.text_encoder = CR1TextEncoder(
            text_encoder_ckpt_path=resolved_text_encoder_path,
            device=self.device,
            cpu_offload=self.cpu_offload,
        )

    def _resolve_artifact(self, spec: Optional[str], artifact_name: str) -> Optional[str]:
        """
        Resolve artifact specification to a local file path, supporting multiple formats.

        Args:
            spec: Path-like string, UUID, or hf:// URI.
            artifact_name: Human-readable name of the artifact type (used for logging).

        Returns:
            A local file path string where the artifact can be accessed, or None/empty if spec is None/empty.
        """
        if not spec:
            return spec

        # First check if the spec is a valid local path
        path_candidate = Path(spec)
        if path_candidate.exists():
            resolved = str(path_candidate)
            log.info("Resolved %s from local path: %s", artifact_name, resolved)
            return resolved

        # Next check if it's an hf:// URI
        if spec.startswith("hf://"):
            repo_id, filename, revision = resolve_hf_uri(spec)
            resolved = hf_download(repo_id=repo_id, filename=filename, revision=revision)
            log.info("Resolved %s from HuggingFace: %s", artifact_name, resolved)
            return resolved

        # Finally, check if it's a UUID format for Cosmos OSS checkpoints
        if is_uuid_format(spec):
            try:
                os.environ.setdefault("COSMOS_EXPERIMENTAL_CHECKPOINTS", "1")
                from cosmos_oss.checkpoints_predict2 import register_checkpoints
                from cosmos_predict2._src.imaginaire.utils.checkpoint_db import download_checkpoint

                register_checkpoints()
                resolved = download_checkpoint(spec)
                log.info("Resolved %s from checkpoint UUID: %s", artifact_name, resolved)
                return resolved
            except Exception as e:
                log.warning("Could not download checkpoint UUID %s (%s).", spec, e)
                return spec

        log.info("Using %s spec as-is: %s", artifact_name, spec)
        return spec

    def _resolve_config_path(self, config_path: Optional[str], checkpoint_path: Optional[str]) -> Optional[str]:
        """
        Resolve config path with sensible defaults for local checkpoints.
        
        Args:
            config_path: Optional user-provided config path or URI.
            checkpoint_path: Resolved checkpoint path used to infer default config location if config_path is None.
        
        Returns:
            A local file path string for the config JSON, or None if no config file is found.
        """
        if config_path is not None:
            resolved_config = self._resolve_artifact(config_path, artifact_name="config")
            if resolved_config and Path(resolved_config).exists():
                return resolved_config

        if checkpoint_path is not None:
            local_default = Path(checkpoint_path).parent / "config.json"
            if local_default.exists():
                return str(local_default)

        if Path("config.json").exists():
            return "config.json"

        return None

    def _build_tokenizer(self, tokenizer_path: Optional[str]) -> Wan2pt1VAEInterface:
        """
        Create VAE tokenizer from resolved path with graceful fallback.

        Args:
            tokenizer_path: Local file path or spec for the VAE tokenizer checkpoint.

        Returns:
            An instance of Wan2pt1VAEInterface initialized with the specified checkpoint or unweighted fallback.
        """
        valid_pth = tokenizer_path if (tokenizer_path and Path(tokenizer_path).exists()) else ""
        if not valid_pth:
            valid_pth = download_wan_vae_tokenizer()

        log.info("Loading VAE tokenizer from: %s", valid_pth or "unweighted mock")
        try:
            return Wan2pt1VAEInterface(
                chunk_duration=93,
                load_mean_std=False,
                vae_pth=valid_pth,
                temporal_window=16,
                keep_decoder_cache=False,
                keep_encoder_cache=False,
            )
        except Exception as e:
            log.warning("Failed to instantiate Wan2pt1VAEInterface (%s). Creating mock tokenizer.", e)
            return Wan2pt1VAEInterface(
                chunk_duration=93,
                load_mean_std=False,
                vae_pth="",
                temporal_window=16,
                keep_decoder_cache=False,
                keep_encoder_cache=False,
            )

    @staticmethod
    def _get_default_dit_config() -> dict[str, Any]:
        """Default 2B DiT configuration for Cosmos-Predict2.5."""
        return {
            "max_img_h": 240,
            "max_img_w": 240,
            "max_frames": 128,
            "in_channels": 16,
            "out_channels": 16,
            "patch_spatial": 2,
            "patch_temporal": 1,
            "concat_padding_mask": True,
            "model_channels": 2048,
            "num_blocks": 28,
            "num_heads": 16,
            "atten_backend": "minimal_a2a",
            "pos_emb_cls": "rope3d",
            "pos_emb_learnable": True,
            "pos_emb_interpolation": "crop",
            "use_adaln_lora": True,
            "adaln_lora_dim": 256,
            "rope_h_extrapolation_ratio": 3.0,
            "rope_w_extrapolation_ratio": 3.0,
            "rope_t_extrapolation_ratio": 1.0,
            "crossattn_emb_channels": 1024,
            "use_crossattn_projection": True,
            "crossattn_proj_in_channels": CROSSATTN_PROJ_IN_CHANNELS,
            "sac_config": SACConfig(mode="predict2_2b_720_aggressive"),
        }

    def _build_dit(self, config_path: Optional[str], checkpoint_path: str) -> MinimalV1LVGDiT:
        """
        Create DiT model from config and load checkpoint weights.

        Args:
            config_path: Local file path to the DiT config JSON (or None to use defaults).
            checkpoint_path: Local file path to the DiT checkpoint.

        Returns:
            An instance of MinimalV1LVGDiT initialized with the specified config and checkpoint.
        """
        if config_path is not None and Path(config_path).exists():
            log.info("Loading DiT config from file: %s", config_path)
            with open(config_path, "r", encoding="utf-8") as f:
                dit_config = json.load(f)
            if isinstance(dit_config.get("sac_config"), dict):
                dit_config["sac_config"] = SACConfig(**dit_config["sac_config"])
        else:
            log.info("Using default 2B DiT configuration for Cosmos-Predict2.5")
            dit_config = self._get_default_dit_config()

        log.info("Initializing DiT model structure...")
        with torch.device("meta"):
            dit = MinimalV1LVGDiT(**dit_config)

        # NOTE: After initialization, we have to initialize the weights on the target device to ensure that any buffers (e.g., RoPE) are properly registered and can be fixed if needed.
        dit.to_empty(device=self.device)
        dit.init_weights()

        # Apply RoPE buffer fix to ensure RoPE-related buffers are registered and shape-consistent
        fix_rope_buffers(dit)

        log.info("Loading checkpoint: %s", checkpoint_path)
        ckpt = safe_torch_load(checkpoint_path, map_location="cpu")
        if isinstance(ckpt, dict) and "net" in ckpt:
            state_dict = ckpt["net"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        if isinstance(state_dict, dict):
            state_dict = {
                k.replace("._checkpoint_wrapped_module", "").removeprefix("net."): v
                for k, v in state_dict.items()
                if not any(k.startswith(p) for p in ["prompt_encoder.", "tokenizer."])
            }

        # Non-strict loading to allow missing and unexpected keys.
        load_result = non_strict_load_model(dit, state_dict)
        if isinstance(load_result, tuple) and len(load_result) >= 2:
            missing, unexpected = load_result[0], load_result[1]
            if missing:
                log.warning("Missing keys while loading DiT: %s", missing[:5])
            if unexpected:
                log.warning("Unexpected keys while loading DiT: %s", unexpected[:5])
        elif isinstance(load_result, (list, tuple)):
            log.warning("Checkpoint loaded with info: %s", load_result)

        # Move model to target device and cast directly to runtime dtype (e.g., bfloat16 on CUDA)
        target_dtype = self.runtime_dtype if "cuda" in str(self.device) else torch.float32
        dit = dit.to(device=self.device, dtype=target_dtype)

        # Set to eval mode
        dit.eval()

        return dit

    def save_config(self, filepath: str, num_inference_steps: int = 35, guidance_scale: float = 1.5) -> None:
        """
        Save the inferencer configuration to a JSON file for reproducibility.

        Args:
            filepath: Destination JSON file path.
            num_inference_steps: Number of diffusion sampling steps (default: 35).
            guidance_scale: Classifier-free guidance scale for conditioning strength (default: 1.5).
                Values > 1.0 increase text/video influence; 1.0 = no guidance.
        """
        payload = {
            "model_size": self.model_size,
            "cpu_offload": self.cpu_offload,
            "checkpoint_path": self.checkpoint_path,
            "config_path": self.config_path,
            "device": self.device,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        log.info("Saved inferencer config to %s", filepath)

    def encode_text(self, prompt: str) -> torch.Tensor:
        """
        Encode text prompt for text condition.

        Args:
            prompt: Text prompt.

        Returns:
            A text embedding tensor with shape (1, L, D).
        """
        return self.text_encoder.encode_prompt(prompt)

    @torch.no_grad()
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """
        Encode video tensor to latent space using the VAE tokenizer.

        Args:
            video: Tensor of shape (B, C, T, H, W) in range [-1, 1].

        Returns:
            Latent tensor of shape (B, C_latent, T_latent, H_latent, W_latent).
        """
        latent = self.tokenizer.encode(video.float())
        return latent.float()

    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent tensor to video tensor.

        Args:
            latent: Latent tensor of shape (B, C_latent, T_latent, H_latent, W_latent).

        Returns:
            Video tensor of shape (B, C, T, H, W) in range [-1, 1].
        """
        video = self.tokenizer.decode(latent.float())
        return video.float()

    def _get_condition(
        self,
        text_embeddings: torch.Tensor,
        latent: torch.Tensor,
        num_conditional_frames: int = 1,
        dtype: torch.dtype | None = None,
    ) -> Video2WorldCondition:
        """
        Build Video2World condition object for inference.
        """
        if dtype is None:
            dtype = next(self.dit.parameters()).dtype

        # Get the shape parameters
        bsz, _, _, height, width = latent.shape

        # Get the device to ensure same-device tensors
        device = latent.device

        # Conditioning mask: 0 for non-conditioned frames, 1 for frames that use GT (conditioned frames)
        padding_mask = torch.zeros(
            bsz,
            1,
            height * 8,
            width * 8,
            device=device,
            dtype=dtype,
        )

        # Use a fixed FPS of 24 for all videos
        fps = torch.full((bsz,), 24.0, device=device, dtype=dtype)

        # Create the Video2WorldCondition with the text embedding and padding mask
        base_condition = Video2WorldCondition(
            crossattn_emb=text_embeddings.to(device=device, dtype=dtype),
            fps=fps,
            padding_mask=padding_mask,
            data_type=DataType.VIDEO,
            use_video_condition=True,
        )

        # Set the video condition with the provided latent frames and conditioning semantics
        return base_condition.set_video_condition(
            gt_frames=latent.to(dtype=dtype),
            random_min_num_conditional_frames=num_conditional_frames,
            random_max_num_conditional_frames=num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            conditional_frames_probs=None,
        )

    def denoise(
        self,
        noise: torch.Tensor,
        xt_b_c_t_h_w: torch.Tensor,
        timesteps_b_t: torch.Tensor,
        condition: Video2WorldCondition,
    ) -> torch.Tensor:
        """
        Predict velocity at a single denoising step using DiT.

        This method replaces conditioning frames in both the input and output:
        - Input: Replace noisy frames with ground-truth conditioning frames.
        - Output: Replace predicted velocity of conditioning frames with ground-truth velocity.

        Args:
            noise: Gaussian noise tensor sampled from N(0, I).
            xt_b_c_t_h_w: Current latent sample at denoising step t.
            timesteps_b_t: Timestep for each sample in batch.
            condition: Video2World condition object containing GT frames and masks.

        Returns:
            Predicted velocity tensor (v_t) with same shape as xt_b_c_t_h_w.
        """
        # Get the model dtype to ensure all tensors are in consistent precision
        model_dtype = next(self.dit.parameters()).dtype

        # Get batch size and channel dimensions
        batch_size, channels, _, _, _ = xt_b_c_t_h_w.shape

        condition_video_mask = None

        # Extract GT frames and mask from condition
        gt_frames = condition.gt_frames
        
        # Mask to indicate which frames are conditioned (1 for condition frames, 0 for non-condition frames)
        mask = condition.condition_video_input_mask_B_C_T_H_W

        # FRAME REPLACEMENT INPUT: Replace noisy frames with ground-truth conditioning frames
        if condition.is_video and gt_frames is not None and mask is not None:
            condition_state_in = gt_frames.type_as(xt_b_c_t_h_w)
            condition_video_mask = mask.repeat(1, channels, 1, 1, 1).type_as(xt_b_c_t_h_w)

            # Replace the frame with mask 1 with the GT frame, and keep the noisy frame where mask is 0
            xt_b_c_t_h_w = (
                condition_state_in * condition_video_mask
                + xt_b_c_t_h_w * (1 - condition_video_mask)
            )

        # Run the DiT forward pass to predict the velocity.
        autocast_device = "cuda" if "cuda" in str(self.device) else "cpu"
        with torch.autocast(device_type=autocast_device, dtype=self.runtime_dtype): 
            net_output = self.dit(
                x_B_C_T_H_W=xt_b_c_t_h_w.to(device=self.device, dtype=model_dtype),
                timesteps_B_T=timesteps_b_t.to(device=self.device, dtype=model_dtype),
                **condition.to_dict()
            ).float()

        # FRAME REPLACEMENT OUTPUT: Replace predicted velocity of conditioning frames with GT velocity
        # In rectified flow: v_t = x_T - x_0, so GT velocity = noise - gt_frames
        if condition.is_video and condition_video_mask is not None:
            gt_frames_x0 = condition.gt_frames.type_as(net_output)
            gt_velocity = noise.type_as(net_output) - gt_frames_x0
            
            # Replace predicted velocity of conditioning frames with GT velocity where mask is 1. Otherwise, keep the predicted velocity.
            net_output = (
                gt_velocity * condition_video_mask 
                + net_output * (1 - condition_video_mask)
            )

        return net_output

    def _combine_conditions(
        self, cond1: Video2WorldCondition, cond2: Video2WorldCondition
    ) -> Video2WorldCondition:
        """
        Combine two Video2WorldCondition objects along the batch dimension for CFG batched forward pass.
        """
        return Video2WorldCondition(
            crossattn_emb=torch.cat([cond1.crossattn_emb, cond2.crossattn_emb], dim=0),
            fps=torch.cat([cond1.fps, cond2.fps], dim=0),
            padding_mask=torch.cat([cond1.padding_mask, cond2.padding_mask], dim=0),
            data_type=cond1.data_type,
            use_video_condition=cond1.use_video_condition,
            gt_frames=torch.cat([cond1.gt_frames, cond2.gt_frames], dim=0)
            if cond1.gt_frames is not None and cond2.gt_frames is not None
            else None,
            condition_video_input_mask_B_C_T_H_W=torch.cat(
                [
                    cond1.condition_video_input_mask_B_C_T_H_W,
                    cond2.condition_video_input_mask_B_C_T_H_W,
                ],
                dim=0,
            )
            if cond1.condition_video_input_mask_B_C_T_H_W is not None
            and cond2.condition_video_input_mask_B_C_T_H_W is not None
            else None,
            num_conditional_frames_B=torch.cat(
                [cond1.num_conditional_frames_B, cond2.num_conditional_frames_B], dim=0
            )
            if cond1.num_conditional_frames_B is not None
            and cond2.num_conditional_frames_B is not None
            else None,
        )

    @staticmethod
    def _to_pil_rgb(image: Union[np.ndarray, Image.Image]) -> Image.Image:
        """
        Convert input image to RGB Pillow image format.

        Args:
            image: Input image as a numpy array (H, W), (H, W, C), or a PIL Image.

        Returns:
            A PIL Image in RGB mode.
        """
        if isinstance(image, Image.Image):
            # If it's already a PIL Image, convert to RGB if not already
            return image.convert("RGB")

        # If it's a numpy array, handle different channel configurations and convert to uint8 if necessary
        if image.ndim == 2:
            # If grayscale, replicate channels to make it RGB
            image = np.stack([image] * 3, axis=-1)
        if image.shape[-1] == 4:
            # If RGBA, drop the alpha channel
            image = image[..., :3]
        if image.dtype != np.uint8:
            # If not uint8, assume it's in [0, 255] range and convert to uint8
            image = np.clip(image, 0, 255).astype(np.uint8)

        return Image.fromarray(image, mode="RGB")

    @staticmethod
    def _scale_image_intensity(image: np.ndarray) -> np.ndarray:
        """Scale image intensity to [0, 1] range using fixed /255 scaling.

        Args:
            image: Input image as a uint8 numpy array.

        Returns:
            Scaled image as a numpy array in [0, 1].
        """
        return np.asarray(image, dtype=np.float32).copy() / 255.0

    @torch.inference_mode()
    def predict(
        self,
        image: Union[np.ndarray, Image.Image],
        prompt: Optional[str] = DEFAULT_PROMPT,
        cfg_scale: float = 1.5,
        num_steps: int = 35,
        seed: Optional[int] = None,
        cfg_interval: tuple[float, float] = (0.0, 1.0),
        use_ap_soft_prior: bool = True,
        use_periodic_noise: bool = True,
    ) -> list[np.ndarray]:
        """Generate multi-view video frames from one anchor image.

        Args:
            image: Input anchor image as PIL or numpy array.
            prompt: Optional text prompt. Uses default prompt when None.
            cfg_scale: Classifier-free guidance scale.
            num_steps: Number of diffusion steps.
            seed: Optional RNG seed.
            cfg_interval: Relative step interval [start, end] in [0, 1] where CFG is active.
            use_ap_soft_prior: Blend the horizontally-flipped anchor latent into the 180°
                AP frame (see `METHOD.md` §3.2). Set False to reproduce the pre-prior
                baseline for ablation Variants A/B (`ABLATION_STUDY.md`).
            use_periodic_noise: Use the periodic-orbit noise schedule for closed 360°
                continuity (see `METHOD.md` §3.4). Set False to fall back to i.i.d.
                per-frame Gaussian noise for ablation Variants A/B.

        Returns:
            List of generated RGB frames as uint8 numpy arrays.
        """
        # Get the text prompt
        prompt = prompt if prompt is not None else DEFAULT_PROMPT

        # Get the PIL image
        pil_img = self._to_pil_rgb(image)

        # Resize and convert to tensor
        resized_image = pil_img.resize((IMG_WIDTH, IMG_HEIGHT), resample=Image.BICUBIC)
        
        # Scale image intensity to [0, 1] range
        np_image = self._scale_image_intensity(np.asarray(resized_image, dtype=np.float32))

        # Get the first frame of the input video, and convert to tensor with shape (1, C, H, W)
        anchor_frame = (
            torch.from_numpy(np_image).permute(2, 0, 1).unsqueeze(0)
        ).to(self.device)
        # Then, normalize to [-1, 1]
        anchor_frame = anchor_frame * 2.0 - 1.0
        # Add temporal dimension to make it (1, C, 1, H, W)
        anchor_frame = anchor_frame.unsqueeze(2) 

        batch_size, _, _, _, _ = anchor_frame.shape

        # Get the text embedding
        text_embeddings = self.encode_text(prompt).to(self.device)

        # OPTIMIZATION C3: Encode anchor frame via minimal 5-frame temporal chunk (for 3D VAE video mean/std consistency)
        # Bypasses encoding 93 identical video frames (~18x VAE speedup while maintaining bit-exact frame 0 latent)
        anchor_chunk = anchor_frame.repeat(1, 1, 5, 1, 1)  # (B, C, 5, H, W)
        latent_anchor = self.encode_video(anchor_chunk)     # (B, C_latent, 2, H_latent, W_latent)

        gen_dtype = next(self.dit.parameters()).dtype
        latent_anchor = latent_anchor.to(dtype=gen_dtype)

        b, c, _, h, w = latent_anchor.shape
        t_latent = 1 + (NUM_FRAMES - 1) // 4  # 24 latent frames for 93 video frames

        # Build full 24-frame conditioning latent with GT anchor at t=0 and zeros elsewhere (since mask=0 for t>=1)
        latent_cond = torch.zeros((b, c, t_latent, h, w), device=self.device, dtype=gen_dtype)
        latent_cond[:, :, 0:1] = latent_anchor[:, :, 0:1]

        # Encode horizontally flipped anchor image for 180° AP soft latent feature prior (~27.5 dB PSNR benchmark)
        latent_flip = None
        if use_ap_soft_prior:
            anchor_frame_flip = torch.flip(anchor_frame, dims=[-1])
            flip_chunk = anchor_frame_flip.repeat(1, 1, 5, 1, 1)
            latent_flip = self.encode_video(flip_chunk).to(dtype=gen_dtype)

        ap_idx = t_latent // 2  # Latent frame index 12 (180° AP)
        soft_prior_alpha = 0.3

        state_shape = (c, t_latent, h, w)

        # Random seed for reproducibility. If not provided, generate a random seed using torch's random number generator.
        run_seed = seed if seed is not None else int(torch.randint(0, 2**32 - 1, (1,)).item())
        
        # Get the conditional and unconditional latent states
        condition = self._get_condition(
            text_embeddings,
            latent_cond,
            num_conditional_frames=1,
            dtype=gen_dtype,
        )
        uncondition = self._get_condition(
            torch.zeros_like(text_embeddings),
            latent_cond,
            num_conditional_frames=1,
            dtype=gen_dtype,
        )

        # Pre-combine conditions for CFG batched forward pass
        if cfg_scale != 1.0:
            combined_condition = self._combine_conditions(uncondition, condition)

        if use_periodic_noise:
            # Periodic Orbit Noise Schedule: e(theta_k) = a * cos(theta_k) + b * sin(theta_k)
            # Guarantees bit-exact closed orbit continuity e(0°) = e(360°) = a while e_k ~ N(0, I) marginally
            a_noise = arch_invariant_rand(
                shape=(batch_size, c, 1, h, w),
                dtype=torch.float32,
                device=self.device,
                seed=run_seed,
            ).to(dtype=gen_dtype)
            b_noise = arch_invariant_rand(
                shape=(batch_size, c, 1, h, w),
                dtype=torch.float32,
                device=self.device,
                seed=run_seed + 1,
            ).to(dtype=gen_dtype)

            angles_orbit = torch.linspace(0.0, 2.0 * np.pi, t_latent, device=self.device, dtype=gen_dtype).view(1, 1, t_latent, 1, 1)
            noise = a_noise * torch.cos(angles_orbit) + b_noise * torch.sin(angles_orbit)
        else:
            # Baseline: i.i.d. per-frame Gaussian noise (ablation Variants A/B).
            noise = arch_invariant_rand(
                shape=(batch_size,) + state_shape,
                dtype=torch.float32,
                device=self.device,
                seed=run_seed,
            ).to(dtype=gen_dtype)

        # Seed generator for reproducibility
        seed_generator = torch.Generator(device=self.device)
        seed_generator.manual_seed(run_seed)

        # Set the timesteps for the sampling scheduler.
        self.sample_scheduler.set_timesteps(
            num_inference_steps=num_steps,
            device=self.device,
            shift=5.0,
            use_kerras_sigma=False,
        )

        # Initialize the latent sample with the noise. 
        # This will be iteratively denoised in the loop.
        latents = noise.clone()

        for timestep in self.sample_scheduler.timesteps:
            # Get the current timestep
            timestep_b_t = timestep.view(1, 1).expand(batch_size, 1)

            # Check if CFG applies at current relative step (t_norm in [0, 1])
            t_norm = float((1000.0 - timestep.item()) / 1000.0)
            use_cfg = (cfg_scale != 1.0) and (cfg_interval[0] <= t_norm <= cfg_interval[1])

            if use_cfg:
                # OPTIMIZATION C1: Batched CFG Forward Pass (2B model evaluated once for [uncond, cond] pair)
                noise_batched = torch.cat([noise, noise], dim=0)
                latents_batched = torch.cat([latents, latents], dim=0)
                timestep_batched = torch.cat([timestep_b_t, timestep_b_t], dim=0)

                v_batched = self.denoise(noise_batched, latents_batched, timestep_batched, combined_condition)
                v_uncond, v_cond = v_batched.chunk(2, dim=0)
                velocity_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                velocity_pred = self.denoise(noise, latents, timestep_b_t, condition)

            # Perform the sampling step to get the next latent sample.
            latents = self.sample_scheduler.step(
                model_output=velocity_pred,
                timestep=timestep,
                sample=latents,
                return_dict=False,
                generator=seed_generator,
            )[0]

            # After getting the new latent sample, we need to replace the frames of the latent sample that correspond to the conditioning frames with the ground-truth latent frames from the input video. 
            # This ensures that the model's predictions for those frames are always grounded in the original input, which is crucial for maintaining consistency with the anchor image.
            cond_mask = condition.condition_video_input_mask_B_C_T_H_W
            if cond_mask is not None:
                latents = latents * (1 - cond_mask) + latent_cond * cond_mask

        # Soft feature prior blending at 180° AP (latent frame index 12). Applied once, after
        # the full reverse-diffusion trajectory completes, so alpha is a true blend weight.
        # (Blending inside the sampling loop would compound geometrically over `num_steps`
        # iterations toward a hard replacement, since it is a fixed-point iteration
        # x_{n+1} = alpha*F + (1-alpha)*x_n that converges to F as n grows.)
        if use_ap_soft_prior and latent_flip is not None:
            latents[:, :, ap_idx:ap_idx + 1] = (
                soft_prior_alpha * latent_flip[:, :, 0:1] + (1.0 - soft_prior_alpha) * latents[:, :, ap_idx:ap_idx + 1]
            )

        # Decode the final latent sample to get the generated video frames.
        video = self.decode_latent(latents.float())

        # The output will be in the range [-1, 1], so we scale and clamp it to [0, 1] before converting to uint8.
        video = (video / 2.0 + 0.5).clamp(0, 1)

        # Permute to (T, H, W, C), and convert back to uint8.
        video_np = (
            (video[0].permute(1, 2, 3, 0).detach().cpu().numpy() * 255.0)
            .clip(0, 255)
            .astype(np.uint8)
        )

        # Return a list of frames as numpy arrays. 
        # Each frame is of shape (H, W, C) and dtype uint8.
        return [video_np[i] for i in range(video_np.shape[0])]

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)
