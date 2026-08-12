"""PixelNeRF (CVPR 2021)

Generalizable Feed-Forward Prior-Guided NeRF.
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
import warnings

try:
    import numpy as np
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch or NumPy not available")

# Setup early import paths for PixelNeRF package
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pixelnerf_dir = os.path.join(base_dir, "cloned", "pixel-nerf")
sys.path.insert(0, os.path.join(pixelnerf_dir, "src"))

try:
    from model import make_model
    from render.nerf import NeRFRenderer
    from util.util import gen_rays, pose_spherical
    from pyhocon import ConfigFactory
    PIXELNERF_MODEL_AVAILABLE = True
except ImportError as e:
    PIXELNERF_MODEL_AVAILABLE = False
    warnings.warn(f"PixelNeRF source package not importable, falling back is unavailable: {e}")

from models.utils import normalize_tensor


# --- Logger --- #
logger = logging.getLogger(__name__)

# Camera parameters for turntable orbit around origin
CAMERA_RADIUS = 4.0
CAMERA_ELEV_DEG = 0.0
CAMERA_FOV_DEG = 40.0
NEAR = CAMERA_RADIUS - 2.0
FAR = CAMERA_RADIUS + 2.0


def load_config():
    """Loads pixel-nerf's default.conf, falling back to a minimal inline config."""
    conf_path = os.path.join(pixelnerf_dir, "conf", "default.conf")
    if os.path.exists(conf_path):
        return ConfigFactory.parse_file(conf_path)
    return ConfigFactory.parse_string(
        """
        model {
            type = pixelnerf
            use_encoder = True
            use_xyz = True
            use_code = True
            code { num_freqs = 6, freq_factor = 1.5, include_input = True }
            use_viewdirs = True
            use_code_viewdirs = False
            encoder { type = resnet34 }
            mlp_coarse { type = mlp, hidden_dim = 64, num_layers = 2 }
            mlp_fine { type = mlp, hidden_dim = 128, num_layers = 2 }
        }
        renderer {
            n_coarse = 64
            n_fine = 32
            white_bkgd = False
        }
        """
    )


def build_model_and_renderer(
    device: str, checkpoint_path: str | None = None, simple_output: bool = True
) -> tuple["torch.nn.Module", "torch.nn.Module"]:
    """Builds the PixelNeRF network and its NeRF ray-marching renderer.

    Args:
        device: Compute device.
        checkpoint_path: Optional path to pretrained model checkpoint.
        simple_output: If True, returns (rgb, depth) directly from fine pass.
            If False, returns full output dict containing both coarse and fine passes.

    Returns:
        A tuple of (model, render_wrapper).
    """
    conf = load_config()

    model = make_model(conf["model"])
    # Freeze the pretrained ImageNet ResNet encoder's gradient (the original class
    # exposes this exact flag for this purpose): the training split here is a few
    # hundred patients, far too small to safely fine-tune a full ResNet34 without
    # destroying its pretrained features.
    model.stop_encoder_grad = True

    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model = model.to(device)

    # Full NeRF ray-marching + alpha-compositing renderer (see
    # cloned/pixel-nerf/src/render/nerf.py), reused as-is instead of a hand-rolled
    # shortcut that never marches rays or alpha-composites.
    renderer = NeRFRenderer.from_conf(
        conf["renderer"], white_bkgd=False, eval_batch_size=100000
    ).to(device)
    render_wrapper = renderer.bind_parallel(model, gpus=None, simple_output=simple_output)
    return model, render_wrapper


def render_view(
    render_wrapper: "torch.nn.Module",
    azimuth: float,
    grid_res: int,
    device: str,
    return_coarse: bool = False,
) -> "torch.Tensor" | tuple["torch.Tensor", "torch.Tensor"]:
    """Ray-marches and alpha-composites a single turntable view.

    Args:
        render_wrapper: Result of ``build_model_and_renderer``'s second return value.
        azimuth: Azimuth angle in degrees on the fixed-elevation orbit.
        grid_res: Output resolution (grid_res x grid_res).
        device: Compute device.
        return_coarse: If True, returns a tuple of (coarse_rgb, fine_rgb).

    Returns:
        RGB tensor of shape [grid_res, grid_res, 3] in [0, 1] or (coarse, fine) tuple.
    """
    focal = 0.5 * grid_res / np.tan(0.5 * np.radians(CAMERA_FOV_DEG))
    focal_t = torch.tensor(focal, device=device, dtype=torch.float32)
    pose = pose_spherical(float(azimuth), CAMERA_ELEV_DEG, CAMERA_RADIUS)
    rays = gen_rays(pose.unsqueeze(0).to(device), grid_res, grid_res, focal_t, NEAR, FAR)
    rays = rays.reshape(1, grid_res * grid_res, 8)
    out = render_wrapper(rays)
    if isinstance(out, tuple):
        rgb, _depth = out
        if return_coarse:
            logger.warning(
                "[PixelNeRF] render_view called with return_coarse=True but renderer was built with "
                "simple_output=True (fine pass only). Returning fine pass for both coarse and fine views. "
                "Rebuild with simple_output=False for genuine dual-pass outputs."
            )
            return rgb.view(grid_res, grid_res, 3), rgb.view(grid_res, grid_res, 3)
        return rgb.view(grid_res, grid_res, 3)
    else:
        # DotMap / Dict output when simple_output=False
        coarse_rgb = out["coarse"]["rgb"].view(grid_res, grid_res, 3) if isinstance(out, dict) else out.coarse.rgb.view(grid_res, grid_res, 3)
        fine = out["fine"] if isinstance(out, dict) else out.fine
        if len(fine) > 0:
            fine_rgb = fine["rgb"].view(grid_res, grid_res, 3) if isinstance(fine, dict) else fine.rgb.view(grid_res, grid_res, 3)
        else:
            fine_rgb = coarse_rgb
        if return_coarse:
            return coarse_rgb, fine_rgb
        return fine_rgb


def encode_source_view(model: "torch.nn.Module", source_img: "torch.Tensor", device: str) -> None:
    """Encodes a single [1, 3, H, W] source image at azimuth=0 on the fixed orbit."""
    encode_source_view_repeated(model, source_img, 1, device)


def encode_source_view_repeated(
    model: "torch.nn.Module", source_img: "torch.Tensor", n: int, device: str
) -> None:
    """Encodes n copies of source image to match target pose batch size."""
    focal = 0.5 * source_img.shape[-1] / np.tan(0.5 * np.radians(CAMERA_FOV_DEG))
    focal_t = torch.tensor(focal, device=device, dtype=torch.float32)
    encode_pose = pose_spherical(0.0, CAMERA_ELEV_DEG, CAMERA_RADIUS).to(device)
    imgs = source_img.repeat(n, 1, 1, 1)
    poses = encode_pose.unsqueeze(0).repeat(n, 1, 1)
    model.encode(imgs.unsqueeze(1), poses=poses.unsqueeze(1), focal=focal_t)


def render_views_batched(
    render_wrapper: "torch.nn.Module",
    azimuths_deg: list[float],
    grid_res: int,
    device: str,
) -> "torch.Tensor":
    """Ray-marches and alpha-composites a batch of turntable views in one call.

    Requires ``encode_source_view_repeated`` to have been called with a matching
    ``n = len(azimuths_deg)`` beforehand (superbatch dim ``SB`` must equal the number
    of encoded "objects").

    Returns:
        RGB tensor of shape [n, grid_res, grid_res, 3] in [0, 1].
    """
    n = len(azimuths_deg)
    focal = 0.5 * grid_res / np.tan(0.5 * np.radians(CAMERA_FOV_DEG))
    focal_t = torch.tensor(focal, device=device, dtype=torch.float32)
    poses = torch.stack(
        [pose_spherical(float(az), CAMERA_ELEV_DEG, CAMERA_RADIUS) for az in azimuths_deg]
    ).to(device)  # (n, 4, 4)
    rays = gen_rays(poses, grid_res, grid_res, focal_t, NEAR, FAR)  # (n, grid_res, grid_res, 8)
    rays = rays.reshape(n, grid_res * grid_res, 8)
    rgb, _depth = render_wrapper(rays)  # (n, grid_res^2, 3)
    return rgb.view(n, grid_res, grid_res, 3)


class PixelNeRFWrapper:
    """Wrapper for PixelNeRF baseline."""

    def __init__(self, checkpoint_path: str | None = None) -> None:
        """Initializes the PixelNeRF network.

        Args:
            checkpoint_path: Optional path to pre-trained weights file.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.render_wrapper = None

        if not TORCH_AVAILABLE or not PIXELNERF_MODEL_AVAILABLE:
            logger.warning("[PixelNeRF] Model libraries not fully available.")
            return

        try:
            self.model, self.render_wrapper = build_model_and_renderer(
                self.device, checkpoint_path=checkpoint_path
            )
            self.model.eval()
            logger.info("[PixelNeRF] ✓ Loaded successfully")
        except Exception as e:
            logger.error(f"[PixelNeRF] ✗ Failed: {str(e)[:80]}")

    def infer_multi_views(
        self,
        input_xr: torch.Tensor,
        azimuths: tuple[float, float, int] = (0, 360, 93),
    ) -> list[torch.Tensor]:
        """Renders novel views by ray-marching PixelNeRF's feature field.

        Args:
            input_xr: Input 2D projection CXR [1, 1, 256, 256].
            azimuths: Target view boundaries as (start_angle, end_angle, N_views).

        Returns:
            List of N synthesized 2D novel-view radiography tensors.
        """
        if self.model is None or self.render_wrapper is None:
            return []

        results = []
        # endpoint=True (default) matches datasets/pre_render_diffdrr.py's own
        # torch.linspace(0, 360, N) ground-truth convention -- and thus the same
        # 93-frame data Cosmos-Predict2.5 itself trains/evaluates on -- where frame
        # N-1 lands exactly back at 360=0. endpoint=False (93 intervals instead of
        # 92) drifts up to ~3.9 degrees off that by the last frame.
        azim_range = np.linspace(azimuths[0], azimuths[1], azimuths[2])
        VIEW_BATCH = 8

        try:
            if not isinstance(input_xr, torch.Tensor):
                input_xr = torch.from_numpy(input_xr).float()

            # Normalize to a single [1, 3, H, W] source image.
            while input_xr.dim() < 4:
                input_xr = input_xr.unsqueeze(0)
            if input_xr.shape[1] == 1:
                input_xr = input_xr.repeat(1, 3, 1, 1)
            input_xr = input_xr.to(self.device)
            H, W = input_xr.shape[-2], input_xr.shape[-1]

            grid_res = 64  # Render at a lower internal resolution, then upsample.

            with torch.no_grad():
                # Source view sits at azimuth=0 on the same orbit every target view
                # is rendered from, so the azimuth=0 output approximately reconstructs
                # the conditioning input (as the other baselines' azimuth=0 frame does).
                for start in range(0, len(azim_range), VIEW_BATCH):
                    chunk = list(azim_range[start : start + VIEW_BATCH])
                    n = len(chunk)

                    encode_source_view_repeated(self.model, input_xr, n, self.device)
                    rgb_batch = render_views_batched(self.render_wrapper, chunk, grid_res, self.device)

                    for i in range(n):
                        proj = rgb_batch[i].mean(dim=-1)
                        proj = torch.nn.functional.interpolate(
                            proj.unsqueeze(0).unsqueeze(0),
                            size=(H, W),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0).squeeze(0)

                        results.append(proj)

                # Normalize across the full sequence to preserve inter-view relative intensity
                if results:
                    results = list(normalize_tensor(torch.stack(results, dim=0)).unbind(0))
        except Exception as e:
            logger.error(f"[PixelNeRF] Inference execution failed: {e}\n{traceback.format_exc()}")

        return results
