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

from predict2_5.constants import COSMOS_TOKENIZER_UUID, IMG_HEIGHT, IMG_WIDTH, NUM_FRAMES, PROMPTS
from predict2_5.hf import hf_download, resolve_hf_uri
from predict2_5.text_encoder import CR1TextEncoder
from predict2_5.utils import arch_invariant_rand, fix_rope_buffers, move_tokenizer_to_device, get_logger, is_uuid_format, safe_torch_load


# --- Logger --- #
log = get_logger(__name__)

DEFAULT_PROMPT = PROMPTS[0] if PROMPTS else None


class Inferencer:
    """Inference pipeline for post-trained Cosmos Predict 2.5."""

    def __init__(
        self,
        checkpoint_path: str,
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
            checkpoint_path: Path, UUID, or hf:// URI for DiT checkpoint.
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
        resolved_checkpoint_path = self._resolve_artifact(
            spec=checkpoint_path, 
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
            os.environ.setdefault("COSMOS_EXPERIMENTAL_CHECKPOINTS", "1")
            from cosmos_oss.checkpoints_predict2 import register_checkpoints
            from cosmos_predict2._src.imaginaire.utils.checkpoint_db import download_checkpoint

            register_checkpoints()
            resolved = download_checkpoint(spec)
            log.info("Resolved %s from checkpoint UUID: %s", artifact_name, resolved)
            return resolved

        log.info("Using %s spec as-is: %s", artifact_name, spec)
        return spec

    def _resolve_config_path(self, config_path: Optional[str], checkpoint_path: str) -> str:
        """
        Resolve config path with sensible defaults for local checkpoints.
        
        Args:
            config_path: Optional user-provided config path or URI.
            checkpoint_path: Resolved checkpoint path used to infer default config location if config_path is None.
        
        Returns:
            A local file path string for the config JSON.
        """
        if config_path is not None:
            # If user provided a config path, resolve it like any other artifact (supports local path, hf:// URI, or UUID)
            resolved_config = self._resolve_artifact(config_path, artifact_name="config")
        else:
            # If no config path provided, assume it's in the same directory as the checkpoint with name "config.json"
            local_default = Path(checkpoint_path).parent / "config.json"
            resolved_config = str(local_default)

        if not Path(resolved_config).exists():
            raise FileNotFoundError(f"Config file not found at resolved path: {resolved_config}")
        
        return resolved_config

    def _build_tokenizer(self, tokenizer_path: str) -> Wan2pt1VAEInterface:
        """
        Create VAE tokenizer from resolved path.

        Args:
            tokenizer_path: Local file path to the VAE tokenizer checkpoint.

        Returns:
            An instance of Wan2pt1VAEInterface initialized with the specified checkpoint.
        """
        log.info("Loading tokenizer: %s", tokenizer_path)
        return Wan2pt1VAEInterface(
            chunk_duration=93,
            load_mean_std=False,
            vae_pth=tokenizer_path,
            temporal_window=16,
            keep_decoder_cache=False,
            keep_encoder_cache=False,
        )

    def _build_dit(self, config_path: str, checkpoint_path: str) -> MinimalV1LVGDiT:
        """
        Create DiT model from config and load checkpoint weights.

        Args:
            config_path: Local file path to the DiT config JSON.
            checkpoint_path: Local file path to the DiT checkpoint.

        Returns:
            An instance of MinimalV1LVGDiT initialized with the specified config and checkpoint.

        Notes
        -----
        1. Build on the meta device first to avoid immediate parameter allocation. This reduces memory spikes for large models.
        2. Materialize parameters on the target device with `to_empty(...)` and initialize them once so parameter/buffer structures exist before loading.
        3. Apply `fix_rope_buffers(...)` before loading checkpoint weights to ensure RoPE-related buffers are registered and shape-consistent.
        4. Load checkpoint on CPU first (`map_location="cpu"`) to avoid GPU OOM during deserialization and to support flexible key remapping.
        5. Normalize checkpoint layouts (`net`, `state_dict`, or raw dict) and load with non-strict semantics, then report missing/unexpected keys for debug.
        6. Move to runtime device and switch to eval mode for stable inference.

        References
        -----
            Cosmos-Predict 2.5 loading logic
        """
        # Load JSON config file to get the DiT configuration
        with open(config_path, "r", encoding="utf-8") as f:
            dit_config = json.load(f)

        # Deserialize SACConfig if present in the config JSON
        if isinstance(dit_config.get("sac_config"), dict):
            dit_config["sac_config"] = SACConfig(**dit_config["sac_config"])

        log.info("Initializing DiT from config: %s", config_path)
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
            state_dict = {
                k.removeprefix("net."): v
                for k, v in ckpt["state_dict"].items()
                if k.startswith("net.")
            }
        else:
            state_dict = ckpt

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

        # Move model to target device
        dit = dit.to(self.device)

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
    ) -> list[np.ndarray]:
        """Generate multi-view video frames from one anchor image.

        Args:
            image: Input anchor image as PIL or numpy array.
            prompt: Optional text prompt. Uses default prompt when None.
            cfg_scale: Classifier-free guidance scale.
            num_steps: Number of diffusion steps.
            seed: Optional RNG seed.

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

        # Create the input video by repeating the anchor frame across the temporal dimension to match the expected input shape for the model. 
        # This results in a tensor of shape (B, C, T, H, W) where T=NUM_FRAMES.
        input_video = anchor_frame.repeat(1, 1, NUM_FRAMES, 1, 1)  # (B, C, T, H, W) with T=NUM_FRAMES

        # Get the text embedding
        text_embeddings = self.encode_text(prompt).to(self.device)

        # Encode the input video to get the latent tensor
        latent_cond = self.encode_video(input_video)
        gen_dtype = next(self.dit.parameters()).dtype  # Latent dtype should match the DiT model dtype for consistency
        latent_cond = latent_cond.to(dtype=gen_dtype)
        _, channels, timesteps, height, width = latent_cond.shape
        state_shape = (channels, timesteps, height, width)

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

        # Sample Gaussian noise for the initial denoise step.
        # Ensure the noise is generated with the same seed for reproducibility.
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

            # Predict the velocity with classifier-free guidance.
            v_cond = self.denoise(noise, latents, timestep_b_t, condition)
            v_uncond = self.denoise(noise, latents, timestep_b_t, uncondition)
            velocity_pred = v_uncond + cfg_scale * (v_cond - v_uncond)

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
