"""SV-DRR (MICCAI 2025 / arXiv:2507.05148)

Pose-conditioned 2D Latent Diffusion for novel view synthesis.
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
import warnings

try:
    from PIL import Image
    import numpy as np
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch, PIL or NumPy not available")

try:
    from huggingface_hub import snapshot_download
    HUGGINGFACE_HUB_AVAILABLE = True
except ImportError:
    HUGGINGFACE_HUB_AVAILABLE = False

# Setup early import paths for SV-DRR dependency packages
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(base_dir, "cloned", "SV-DRR"))

try:
    from pipeline_svdrr_DiT import CCProjection, SvdrrDiTPipeline
    SVDRR_PIPELINE_AVAILABLE = True
except ImportError:
    SVDRR_PIPELINE_AVAILABLE = False


# --- Logger --- #
logger = logging.getLogger(__name__)


class SVDRRWrapper:
    """Wrapper for SV-DRR baseline."""
    
    def __init__(self, checkpoint_path: str | None = None) -> None:
        """Initializes the SV-DRR 2D latent diffusion pipeline.

        Args:
            checkpoint_path: Optional path or HuggingFace ID to pre-trained weights.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.pipe = None
        
        if not TORCH_AVAILABLE or not SVDRR_PIPELINE_AVAILABLE:
            logger.warning("[SV-DRR] Model libraries not fully available.")
            return
        
        try:
            local_base_path = os.path.join(base_dir, "cloned", "SV-DRR", "models", "base_model", "256")
            stale_base_path = os.path.join(base_dir, "cloned", "SV-DRR", "models", "base_model", "256.stale-2026-08-08")
            
            if checkpoint_path and os.path.isdir(checkpoint_path):
                model_id = checkpoint_path
            else:
                if not os.path.exists(local_base_path):
                    logger.info(f"[SV-DRR] Local base model not found at {local_base_path}; downloading snapshot from HF Hub...")
                    SVDRR_BASE_MODEL_REVISION = "151bc201b16fa8d6d01853c2036f7d6b354d50ba"
                    snapshot_download(
                        repo_id="xiechun-tsukuba/svdrr-dit-fb-256",
                        revision=SVDRR_BASE_MODEL_REVISION,
                        local_dir=local_base_path,
                    )
                model_id = local_base_path

            # Detect whether checkpoint requires 4-channel or 8-channel transformer
            if checkpoint_path and os.path.isfile(checkpoint_path):
                state = torch.load(checkpoint_path, map_location="cpu")
                st_dict = state.get("transformer", state)
                if "pos_embed.proj.weight" in st_dict:
                    ckpt_in_channels = st_dict["pos_embed.proj.weight"].shape[1]
                    if ckpt_in_channels == 4 and os.path.exists(stale_base_path):
                        model_id = stale_base_path
            else:
                state = None
                
            cc_projection = CCProjection.from_config(model_id, subfolder="cc_projection")
            self.pipe = SvdrrDiTPipeline.from_pretrained(
                model_id, cc_projection=cc_projection, torch_dtype=torch.float32,
                low_cpu_mem_usage=False, ignore_mismatched_sizes=True
            )
            
            if state is not None:
                if "transformer" in state:
                    self.pipe.transformer.load_state_dict(state["transformer"])
                    self.pipe.cc_projection.load_state_dict(state["cc_projection"])
                else:
                    self.pipe.transformer.load_state_dict(state)

            self.pipe = self.pipe.to(self.device)
            self.model = self.pipe
            logger.info("[SV-DRR] ✓ Loaded successfully")
        except Exception as e:
            # Reset pipeline on failure to prevent silent fallback to partially initialized base weights
            logger.error(f"[SV-DRR] ✗ Failed to initialize pipeline: {e}")
            self.pipe = None
            self.model = None
    
    def infer_multi_views(
        self,
        input_xr: torch.Tensor,
        azimuths: tuple[float, float, int] = (0, 360, 93),
    ) -> list[torch.Tensor]:
        """Queries the SV-DRR pipeline to synthesize novel views over specified angles.

        Args:
            input_xr: Input 2D projection CXR [1, 1, 256, 256].
            azimuths: Target view boundaries as (start_angle, end_angle, N_views).

        Returns:
            A list of N synthesized 2D novel-view radiography tensors.
        """
        if self.pipe is None:
            return []

        results = []
        try:
            if not isinstance(input_xr, torch.Tensor):
                input_xr = torch.from_numpy(input_xr).float()
            input_xr = input_xr.clamp(0, 1)
            if input_xr.dim() == 2:
                input_xr = input_xr.unsqueeze(0)

            img_np = input_xr.squeeze().cpu().numpy()
            if img_np.ndim != 2:
                while img_np.ndim > 2:
                    img_np = img_np[0]
            input_img = Image.fromarray(
                (img_np * 255).astype(np.uint8), mode="L"
            )
            # endpoint=True (default) matches datasets/pre_render_diffdrr.py's own
            # torch.linspace(0, 360, N) ground-truth convention -- and thus the same
            # 93-frame data Cosmos-Predict2.5 itself trains/evaluates on -- where
            # frame N-1 lands exactly back at 360=0. endpoint=False (93 intervals
            # instead of 92) drifts up to ~3.9 degrees off that by the last frame.
            azim_range = np.linspace(azimuths[0], azimuths[1], azimuths[2])

            # Batch several target poses per pipeline call instead of one call per
            # azimuth: the reference test_svdrr_DiT.py handles a batch by passing
            # matching-length lists for input_imgs/prompt_imgs/poses (see
            # _encode_image/_encode_pose's list branches), so each chunk only pays the
            # denoising-loop cost once instead of once per view. VIEW_BATCH bounds peak
            # VRAM instead of batching the entire sweep (often 93 views) in one call.
            VIEW_BATCH = 16
            for start in range(0, len(azim_range), VIEW_BATCH):
                chunk = azim_range[start : start + VIEW_BATCH]
                n = len(chunk)
                # Pose relative angle negated to align with training convention
                poses = [[0, -float(azimuth), 0] for azimuth in chunk]
                with torch.no_grad():
                    result = self.pipe(
                        input_imgs=[input_img] * n,
                        prompt_imgs=[input_img] * n,
                        poses=poses,
                        height=256,
                        width=256,
                        guidance_scale=3.0,
                        num_inference_steps=20,
                    )
                    for out_img in result.images:
                        out_tensor = (
                            torch.from_numpy(np.array(out_img.convert("L"))) / 255.0
                        )
                        out_tensor = (out_tensor - out_tensor.min()) / (out_tensor.max() - out_tensor.min() + 1e-8)
                        results.append(out_tensor.float().to(self.device))
        except Exception as e:
            logger.error(f"[SV-DRR] Inference execution failed: {e}\n{traceback.format_exc()}")

        return results
