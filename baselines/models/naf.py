"""NAF / SNAF (MICCAI 2022)

Implicit Neural Attenuation Coordinate Net for CBCT Reconstruction.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch or NumPy not available")

# Setup early import paths for NAF package
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
naf_dir = os.path.join(base_dir, "cloned", "naf_cbct")
sys.path.insert(0, naf_dir)

try:
    # The multi-resolution hash-grid encoder (Instant-NGP-style) is NAF's actual
    # architecture -- its reference configs (cloned/naf_cbct/config/*_50.yaml) all
    # set encoding: hashgrid, not frequency encoding. This previously failed to
    # compile under PyTorch 2.x (DeprecatedTypeProperties no longer implicitly
    # converts to ScalarType in AT_DISPATCH_FLOATING_TYPES_AND_HALF -- fixed in
    # hashencoder.cu by using .scalar_type() instead of .type()) and was also
    # missing its _backend = get_backend() call entirely (hashgrid.py). Both are
    # fixed now; FreqEncoder is kept only as a last-resort fallback for hosts
    # without a working nvcc/CUDA toolchain.
    from src.encoder.hashencoder import HashEncoder
    NAF_HASHENCODER_AVAILABLE = True
except Exception:
    HashEncoder = None
    NAF_HASHENCODER_AVAILABLE = False

try:
    from src.encoder.freqencoder import FreqEncoder
    from src.network.network import DensityNetwork
    NAF_NETWORK_AVAILABLE = True
except Exception as e:
    NAF_NETWORK_AVAILABLE = False

from models.utils import build_perspective_ray_points, normalize_tensor, perspective_ray_march

# Setup logger following STYLE.md
logger = logging.getLogger(__name__)


import copy

FIT_GRID_RES = 64  # Coordinate grid resolution used during per-scan fitting.


def fit_density_field(
    model: "DensityNetwork",
    target_proj: torch.Tensor,
    bound: float,
    device: str,
    iterations: int = 200,
    lr: float = 1e-3,
) -> float:
    """Optimizes a coordinate MLP's weights to reproduce a single 2D projection.

    NAF is a per-scan overfitting method: the coordinate network ``model(xyz)`` has
    no image-conditioning input at all, so it can only ever represent whichever single
    3D field its weights were fit to. Matches the reference algorithm's actual usage
    (per-CT-scan optimization against the available projections, see
    ``cloned/naf_cbct/train.py``) instead of directly querying an unfit/un-conditioned
    network, or training one shared network directly against many different patients'
    images with no conditioning signal to distinguish them.

    Args:
        model: DensityNetwork to fit in place (mutated).
        target_proj: Target 2D projection, [1, H, W] or [H, W], resized internally to
            ``FIT_GRID_RES``.
        bound: Coordinate grid half-extent (``model.bound``).
        device: Compute device.
        iterations: Number of gradient steps.
        lr: Adam learning rate.

    Returns:
        The final MSE loss value.
    """
    # HashEncoder.forward validates inputs against `bound` as a Python float (float64),
    # but linspace's float32 tensor rounds its endpoint to the nearest float32 value --
    # for bound=0.3 that's *larger* than the float64 0.3 used in the check, so an exact
    # endpoint spuriously fails HashEncoder's own domain-validity check. Shrink by a
    # margin well above the float32 ULP at this magnitude (~3e-8) to stay strictly
    # inside [-bound, bound] regardless of rounding direction. perspective_ray_march
    # additionally clamps ray samples to [-bound, bound] itself (bound= below), but
    # this margin still matters for coordinates fed to HashEncoder near that boundary.
    safe_bound = bound * (1.0 - 1e-6)

    # Normalize to 4D [B, C, H, W] regardless of whether the caller passed
    # [H, W], [1, H, W], or [1, 1, H, W], then squeeze back to [1, H, W].
    target = target_proj.to(device)
    while target.dim() < 4:
        target = target.unsqueeze(0)
    target = F.interpolate(
        target, size=(FIT_GRID_RES, FIT_GRID_RES), mode="bilinear", align_corners=False
    ).view(FIT_GRID_RES, FIT_GRID_RES)

    # Azimuth (and every other camera param) is fixed at 0.0 for the entire
    # fitting loop below, so the ray geometry is identical on every iteration --
    # build it once instead of rebuilding it `iterations` times (see
    # perspective_ray_march's `pts` doc; this alone is only ~2% of per-iteration
    # cost, but it's free).
    fit_pts = build_perspective_ray_points(
        azimuth=0.0, grid_res=FIT_GRID_RES, device=device, bound=safe_bound,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    loss = torch.tensor(0.0)
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        # Fits against the input view's own pose (azimuth=0), true perspective
        # ray-marched to match how datasets/pre_render_diffdrr.py actually
        # rendered it -- see perspective_ray_march's docstring. Differentiable
        # end-to-end through model(pts), unlike the plain orthographic sum this
        # replaces.
        pred_proj = perspective_ray_march(
            sample_fn=model, azimuth=0.0, grid_res=FIT_GRID_RES, device=device,
            bound=safe_bound, pts=fit_pts,
        )
        loss = F.mse_loss(pred_proj, target)
        loss.backward()
        optimizer.step()
    model.eval()
    return loss.item()


class NAFWrapper:
    """Wrapper for NAF baseline from cloned/naf_cbct."""
    
    def __init__(self, checkpoint_path: str | None = None, iterations: int = 200) -> None:
        """Initializes and builds the NAF implicit neural attenuation field MLP.

        Args:
            checkpoint_path: Optional path to pre-trained weights file.
            iterations: Per-scan fitting steps run in ``infer_multi_views``
                (reference ``cloned/naf_cbct/train.py`` uses 3000 per scan;
                this default is a cheaper stand-in -- see
                ``scripts/launch_parallel_vms.sh`` for the cost rationale).
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.bound = 0.3  # Matches reference cloned/naf_cbct/config/*_50.yaml
        self.iterations = iterations

        if not TORCH_AVAILABLE or not NAF_NETWORK_AVAILABLE:
            logger.warning("[NAF] Model libraries not fully available.")
            return

        try:
            # Reference config (cloned/naf_cbct/config/*_50.yaml): hashgrid encoder,
            # num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=19.
            if NAF_HASHENCODER_AVAILABLE:
                encoder = HashEncoder(
                    input_dim=3,
                    num_levels=16,
                    level_dim=2,
                    base_resolution=16,
                    log2_hashmap_size=19,
                )
            else:
                logger.warning(
                    "[NAF] HashEncoder unavailable on this host (no working nvcc/CUDA "
                    "toolchain) -- falling back to FreqEncoder, which departs from the "
                    "reference architecture."
                )
                encoder = FreqEncoder(
                    input_dim=3,
                    max_freq_log2=10 - 1,
                    N_freqs=10,
                    log_sampling=True,
                    include_input=True,
                    periodic_fns=(torch.sin, torch.cos),
                )

            # Reference config (cloned/naf_cbct/config/*_50.yaml): num_layers=4,
            # hidden_dim=32, skips=[2].
            model = DensityNetwork(
                encoder=encoder,
                bound=self.bound,
                num_layers=4,
                hidden_dim=32,
                skips=[2],
                out_dim=1,
                last_activation="sigmoid",
            )

            if checkpoint_path and os.path.exists(checkpoint_path):
                model.load_state_dict(
                    torch.load(checkpoint_path, map_location=self.device)
                )

            # Only publish self.model once construction + checkpoint load both
            # succeed, so a shape-mismatched checkpoint (e.g. stale weights from
            # the pre-hashgrid architecture) leaves self.model cleanly None
            # instead of a half-built, wrong-device, untrained network that
            # evaluate.py's `model is None` skip-check wouldn't catch.
            self.model = model.to(self.device)
            self.model.eval()
            logger.info("[NAF] ✓ Loaded successfully")
        except Exception as e:
            logger.error(f"[NAF] ✗ Failed: {str(e)[:80]}")
    
    def infer_multi_views(
        self,
        input_xr: torch.Tensor,
        azimuths: tuple[float, float, int] = (0, 360, 93),
        iterations: int | None = None,
    ) -> list[torch.Tensor]:
        """Fits a per-scan coordinate field to the input view, then synthesizes novel
        views over specified angles by rotating and re-projecting that fitted field.

        NAF's ``DensityNetwork`` has no image-conditioning input (see
        ``fit_density_field``'s docstring) -- previously this method queried the
        network directly without ever fitting it to ``input_xr`` at all (the argument
        was only used to read its H/W), meaning the output was just whatever a
        randomly-initialized (or previous-call-loaded) network happened to represent,
        completely independent of the patient's actual image. This deep-copies the
        loaded network (used as a fitting prior, not a shared final answer -- same
        reasoning as ``MedNeRFWrapper``) and runs a short per-scan optimization
        against ``input_xr`` before rendering.

        Args:
            input_xr: Input 2D projection CXR [1, 1, 256, 256].
            azimuths: Target view boundaries as (start_angle, end_angle, N_views).
            iterations: Per-call override for the fitting budget (defaults to
                ``self.iterations``); lets a single loaded wrapper be swept
                across budgets without reloading the checkpoint.

        Returns:
            A list of N synthesized 2D novel-view radiography tensors.
        """
        if self.model is None:
            return []

        results = []
        # endpoint=True (default) matches datasets/pre_render_diffdrr.py's own
        # torch.linspace(0, 360, N) ground-truth convention -- and thus the same
        # 93-frame data Cosmos-Predict2.5 itself trains/evaluates on -- where frame
        # N-1 lands exactly back at 360=0. endpoint=False (93 intervals instead of
        # 92) drifts up to ~3.9 degrees off that by the last frame.
        azim_range = np.linspace(azimuths[0], azimuths[1], azimuths[2])

        try:
            # Query grid setup
            grid_size = 128
            H, W = input_xr.shape[-2], input_xr.shape[-1]

            model_fit = copy.deepcopy(self.model)
            fit_density_field(
                model_fit, input_xr, self.bound, self.device,
                iterations=iterations if iterations is not None else self.iterations,
            )

            # NAF queries coords in [-bound, bound]. See fit_density_field's
            # safe_bound comment: shrink slightly so linspace's float32 endpoint
            # rounding can't spuriously fail HashEncoder's domain-validity check.
            safe_bound = self.bound * (1.0 - 1e-6)
            grid = torch.linspace(-safe_bound, safe_bound, grid_size, device=self.device)
            coords = torch.stack(torch.meshgrid(grid, grid, grid, indexing="ij"), dim=-1)
            coords_flat = coords.reshape(-1, 3)

            with torch.no_grad():
                # Bake the fitted coordinate MLP into a dense grid ONCE (chunked to
                # avoid OOM), then ray-march that dense grid per azimuth below via
                # cheap grid_sample interpolation instead of re-querying the network
                # per view -- re-querying model_fit directly for all 93 views would
                # be ~100x more MLP evaluations than baking once and reusing.
                chunk = 1024 * 64
                attenuations_flat = torch.zeros(coords_flat.shape[0], 1, device=self.device)
                for i in range(0, coords_flat.shape[0], chunk):
                    attenuations_flat[i : i + chunk] = model_fit(
                        coords_flat[i : i + chunk]
                    )

                attenuations = attenuations_flat.reshape(1, 1, grid_size, grid_size, grid_size)

                def sample_fn(pts: torch.Tensor) -> torch.Tensor:
                    grid_coords = (pts / safe_bound).clamp(-1, 1).view(1, 1, 1, -1, 3)
                    sampled = F.grid_sample(
                        attenuations, grid_coords, mode="bilinear",
                        padding_mode="zeros", align_corners=True,
                    )
                    return sampled.view(-1, 1)

                for azimuth in azim_range:
                    # True perspective ray marching through the baked dense grid,
                    # matching the exact camera geometry
                    # datasets/pre_render_diffdrr.py used to render ground truth --
                    # replaces the previous rotate_volume_3d + orthographic sum.
                    proj = perspective_ray_march(
                        sample_fn=sample_fn, azimuth=float(azimuth), grid_res=grid_size,
                        device=self.device,
                    )
                    if proj.shape[0] != H or proj.shape[1] != W:
                        proj = F.interpolate(
                            proj.unsqueeze(0).unsqueeze(0),
                            size=(H, W),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0).squeeze(0)
                    # Apply min-max tensor normalization
                    proj_corrected = normalize_tensor(proj)
                    results.append(proj_corrected)
        except Exception as e:
            logger.error(f"[NAF] Inference execution failed: {e}")
        
        return results
