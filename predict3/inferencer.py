"""Inference pipeline wrapper for Cosmos 3 (Cosmos3-Edge)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image

from diffusers import Cosmos3OmniPipeline

from predict3.constants import COSMOS3_EDGE_REPO, IMG_HEIGHT, IMG_WIDTH, NUM_FRAMES, PROMPTS


# --- Logger --- #
logger = logging.getLogger(__name__)

DEFAULT_PROMPT = PROMPTS[0]
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, distorted anatomy, artifacts, watermark, text overlay, "
    "inconsistent rotation, flickering, discontinuous geometry"
)


class InferencerV3:
    """Inference engine for Cosmos 3 360-degree X-ray rotation video synthesis."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Initializes the Cosmos 3 inference pipeline.

        Args:
            checkpoint_path: Local model directory or Hugging Face repo ID.
            device: Target execution device (CUDA required).
            dtype: Model runtime precision (bfloat16 required).
        """
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Cosmos 3 requires a CUDA-enabled GPU (bfloat16 support required). "
                "No CUDA device found."
            )

        self.device = device
        self.runtime_dtype = dtype

        resolved_path = self._resolve_checkpoint(checkpoint_path or COSMOS3_EDGE_REPO)
        logger.info("Loading Cosmos3OmniPipeline from %s", resolved_path)

        # Load model weights without device_map to ensure full parameter materialization
        self.pipe = Cosmos3OmniPipeline.from_pretrained(
            resolved_path,
            dtype=self.runtime_dtype,
            enable_safety_checker=False,
        )
        self.pipe.to(self.device)

    @staticmethod
    def _resolve_checkpoint(spec: str) -> str:
        """Resolves local directory paths or passes through Hugging Face repo IDs."""
        if os.path.isdir(spec):
            return str(Path(spec).resolve())

        return spec

    def predict(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        num_frames: int = NUM_FRAMES,
        height: int = IMG_HEIGHT,
        width: int = IMG_WIDTH,
        num_inference_steps: int = 35,
        guidance_scale: float = 6.0,
        fps: float = 24.0,
        seed: int = 42,
    ) -> list[np.ndarray]:
        """Generates a 360-degree rotation video sequence from a single 2D input X-ray.

        Args:
            image: Single input X-ray image (PIL Image, NumPy array, or Tensor).
            prompt: Text prompt string.
            negative_prompt: Negative prompt string.
            num_frames: Total number of frames to generate.
            height: Output video frame height.
            width: Output video frame width.
            num_inference_steps: Denoising inference steps.
            guidance_scale: Classifier-free guidance scale.
            fps: Video playback frames per second.
            seed: Random generator seed.

        Returns:
            List of uint8 RGB numpy arrays with shape (H, W, 3).
        """
        pil_image = self._to_pil(image)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            prompt=prompt or DEFAULT_PROMPT,
            negative_prompt=negative_prompt or DEFAULT_NEGATIVE_PROMPT,
            image=pil_image,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="np",
        )

        video = np.asarray(result.video)
        frames_uint8 = np.clip(video * 255.0 + 0.5, 0, 255).astype(np.uint8)

        return [frames_uint8[t] for t in range(frames_uint8.shape[0])]

    @staticmethod
    def _to_pil(image: Union[Image.Image, np.ndarray, torch.Tensor]) -> Image.Image:
        """Converts input image formats (Tensor, NumPy) into a standard RGB PIL Image."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        if not isinstance(image, np.ndarray):
            raise TypeError(f"Unsupported image type: {type(image)}")

        arr = image

        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] != arr.shape[-1]:
            arr = np.moveaxis(arr, 0, -1)  # CHW -> HWC

        if arr.dtype != np.uint8:
            scale = 255.0 if float(arr.max()) <= 1.0 + 1e-3 else 1.0
            arr = np.clip(arr * scale + 0.5, 0, 255).astype(np.uint8)

        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        elif arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)

        return Image.fromarray(arr).convert("RGB")
