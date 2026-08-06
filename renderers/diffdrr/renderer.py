"""
DiffDRR-backed X-ray volume renderer, geometrically equivalent to
``predict2_5.dvr.renderer.ObjectCentricXRayVolumeRenderer`` (the PyTorch3D
Emission/Absorption DVR used by ``scripts/build_dataset.py``).

Usage mirrors the PyTorch3D pipeline in ``scripts/build_dataset.py``:

    renderer = create_diffdrr_renderer(img_shape=256, device="cuda")
    vol = load_ct_volume("scan.nii.gz", vol_shape=256)
    renderer.set_volume(vol)
    frames = renderer.render(
        azimuth=torch.linspace(0, 360, 93),
        elev=0.0, dist=8.0, fov=12.0,
        min_depth=7.0, max_depth=9.0,
    )  # (93, 1, 256, 256), normalized like the PyTorch3D DVR output

Equivalence strategy
---------------------
* **World frame.** The CT volume is placed in exactly the normalized world
  frame PyTorch3D's ``Volumes`` object uses: an isotropic grid with
  ``voxel_size = 2 * ndc_extent / shape``, centered at the origin. World
  coordinates then line up 1:1 with the PyTorch3D renderer with no unit
  conversion, and ``dist``/``min_depth``/``max_depth`` mean the same thing in
  both renderers.
* **Camera pose.** The source position and orientation are computed with the
  exact spherical formulas PyTorch3D uses internally for
  ``look_at_view_transform(dist, elev, azim)``
  (``camera_position_from_spherical_angles`` + ``look_at_rotation``), so both
  renderers cast rays from the same point in the same direction.
* **Intrinsics.** DiffDRR's detector pixel spacing (``delx``/``dely``) is
  derived from ``fov`` so its per-pixel ray fan matches
  ``FoVPerspectiveCameras(fov=...)`` exactly (see ``_fov_to_pixel_spacing``).
* **Sampling.** DiffDRR's ``Trilinear`` renderer is used with explicit
  ``alphamin``/``alphamax`` bounds derived from ``min_depth``/``max_depth``,
  and ``n_points=n_pts_per_ray``, so both renderers integrate the same
  camera-space depth window with the same number of samples per ray.

Residual, expected differences
-------------------------------
In the default pipeline (``opacity=None``), PyTorch3D's
``AbsorptionEmissionRaymarcher`` is fed *uniform* density (every voxel gets
density = 1, see ``predict2_5/dvr/renderer.py``). This collapses its
nonlinear emission-absorption compositing into (very nearly) a plain sum of
CT density samples along the ray -- i.e. a discretized line integral, missing
only its first sample point (which the raymarcher's reversed-cumprod trick
weights by ~1e-10). DiffDRR's ``Trilinear`` renderer computes the analogous
line integral directly: ``sum(density) * step_size * ray_length``. The two
outputs therefore differ only by:

1. A per-image constant scale factor (``step_size`` / ray-length terms), and
2. A per-pixel foreshortening term (<1% for the FOVs used in this project),

both of which are removed by the shared post-hoc normalization
(``minimized``/``normalized``/``standardized``) applied to both outputs. Run
``renderers/diffdrr/compare_dvr.py`` in an environment with both PyTorch3D
and DiffDRR installed to numerically confirm equivalence (RMSE/PSNR/SSIM)
before relying on this for a specific CT volume/geometry.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torchio as tio

from diffdrr.drr import DRR
from diffdrr.pose import RigidTransform

from predict2_5.utils.transforms import minimized, normalized, standardized


Number = Union[int, float]
AngleLike = Union[Number, Sequence[Number], torch.Tensor]


# ═════════════════════════════════════════════════════════════════════════════
# PyTorch3D-equivalent camera pose math
# ═════════════════════════════════════════════════════════════════════════════

def _look_at_camera_positions(
    dist: torch.Tensor,
    elev: torch.Tensor,
    azim: torch.Tensor,
    degrees: bool = True,
) -> torch.Tensor:
    """
    Replicates ``pytorch3d.renderer.cameras.camera_position_from_spherical_angles``.

    Args:
        dist, elev, azim: Broadcastable tensors of any common shape ``(...,)``.
        degrees: Whether ``elev``/``azim`` are given in degrees (default: True).

    Returns:
        Camera positions in world space, shape ``(..., 3)``.
    """
    if degrees:
        elev = elev * (math.pi / 180.0)
        azim = azim * (math.pi / 180.0)

    x = dist * torch.cos(elev) * torch.sin(azim)
    y = dist * torch.sin(elev)
    z = dist * torch.cos(elev) * torch.cos(azim)
    return torch.stack([x, y, z], dim=-1)


def _look_at_rotation_matrices(
    camera_position: torch.Tensor,
    at: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    up: Tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> torch.Tensor:
    """
    Replicates ``pytorch3d.renderer.cameras.look_at_rotation``.

    Returns a rotation matrix per camera position whose *columns* are the
    camera's local (x, y, z) axes expressed in world coordinates, i.e. the
    matrix ``R`` such that ``p_world = camera_position + R @ p_camera``.

    Args:
        camera_position: World-space camera positions, shape ``(..., 3)``.
        at: World-space look-at target (default: origin).
        up: World-space up vector (default: +Y).

    Returns:
        Rotation matrices, shape ``(..., 3, 3)``.
    """
    device, dtype = camera_position.device, camera_position.dtype
    at_t = torch.as_tensor(at, device=device, dtype=dtype).expand_as(camera_position)
    up_t = torch.as_tensor(up, device=device, dtype=dtype).expand_as(camera_position)

    z_axis = F.normalize(at_t - camera_position, dim=-1, eps=1e-5)
    x_axis = F.normalize(torch.cross(up_t, z_axis, dim=-1), dim=-1, eps=1e-5)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=-1), dim=-1, eps=1e-5)

    # Degenerate case: `up` parallel to the viewing direction.
    is_close = torch.isclose(x_axis, torch.zeros_like(x_axis), atol=5e-3).all(dim=-1, keepdim=True)
    if is_close.any():
        replacement = F.normalize(torch.cross(y_axis, z_axis, dim=-1), dim=-1, eps=1e-5)
        x_axis = torch.where(is_close, replacement, x_axis)

    return torch.stack([x_axis, y_axis, z_axis], dim=-1)


def look_at_view_poses(
    dist: AngleLike,
    elev: AngleLike,
    azim: AngleLike,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> RigidTransform:
    """
    Builds a batched DiffDRR carm-to-world extrinsic equivalent to PyTorch3D's
    ``look_at_view_transform(dist, elev, azim)``.

    Unlike PyTorch3D's ``(R, T)`` pair (a world-to-camera transform), this
    returns the source position directly as the translation component, which
    is exactly what DiffDRR's ``Detector`` expects (see ``Detector.forward``:
    the X-ray source sits at ``extrinsic(0, 0, 0)``).

    Args:
        dist, elev, azim: Broadcastable scalars/sequences/tensors (degrees for elev/azim).
        device: Target device.
        dtype: Target floating-point dtype.

    Returns:
        A ``RigidTransform`` with a batched ``(N, 4, 4)`` matrix, ``N`` being
        the broadcast size of ``dist``/``elev``/``azim``.
    """
    dist_t = torch.as_tensor(dist, device=device, dtype=dtype)
    elev_t = torch.as_tensor(elev, device=device, dtype=dtype)
    azim_t = torch.as_tensor(azim, device=device, dtype=dtype)
    dist_t, elev_t, azim_t = torch.broadcast_tensors(dist_t, elev_t, azim_t)

    C = _look_at_camera_positions(dist_t, elev_t, azim_t)  # (..., 3)
    R = _look_at_rotation_matrices(C)  # (..., 3, 3)

    batch_shape = C.shape[:-1]
    matrix = torch.zeros(*batch_shape, 4, 4, device=C.device, dtype=C.dtype)
    matrix[..., :3, :3] = R
    matrix[..., :3, 3] = C
    matrix[..., 3, 3] = 1.0

    return RigidTransform(matrix.reshape(-1, 4, 4))


def _fov_to_pixel_spacing(
    fov: float,
    sdd: float,
    image_width: int,
    image_height: int,
    ndc_extent: float,
    degrees: bool = True,
) -> Tuple[float, float]:
    """
    Converts a PyTorch3D ``FoVPerspectiveCameras(fov=...)`` vertical field of
    view into the DiffDRR detector pixel spacing (``delx``, ``dely``) that
    reproduces the same per-pixel ray directions at a chosen ``sdd``.

    Derivation: PyTorch3D's NDC focal length is ``f_ndc = cot(fov / 2)``. A
    pixel at NDC coordinate ``x_ndc`` is unprojected to camera-space ray
    direction ``(x_ndc / f_ndc, y_ndc / f_ndc, 1)``. DiffDRR casts a ray from
    the source through detector point ``(s * delx, t * dely, sdd)``, i.e.
    camera-space direction ``(s * delx / sdd, t * dely / sdd, 1)``. Both
    libraries index pixels with symmetric, evenly-spaced pixel-center
    coordinates, so ``x_ndc(i) = (2 * ndc_extent / W) * s(i)``. Equating the
    two ray directions gives ``delx = sdd * 2 * ndc_extent / (W * f_ndc)``.

    Returns:
        ``(delx, dely)`` in the same length units as ``sdd``/``ndc_extent``.
    """
    fov_rad = math.radians(fov) if degrees else fov
    f_ndc = 1.0 / math.tan(fov_rad / 2.0)
    delx = sdd * 2.0 * ndc_extent / (image_width * f_ndc)
    dely = sdd * 2.0 * ndc_extent / (image_height * f_ndc)
    return delx, dely


# ═════════════════════════════════════════════════════════════════════════════
# Renderer
# ═════════════════════════════════════════════════════════════════════════════

class DiffDRRVolumeRenderer(nn.Module):
    """
    DiffDRR-backed X-ray volume renderer, drop-in equivalent (see module
    docstring) of ``predict2_5.dvr.renderer.ObjectCentricXRayVolumeRenderer``.
    """

    def __init__(
        self,
        image_width: int = 256,
        image_height: int = 256,
        n_pts_per_ray: int = 320,
        ndc_extent: float = 1.0,
        renderer: str = "trilinear",
        flip_horizontal: bool = False,
        device: Union[str, torch.device] = "cuda",
    ):
        """
        Args:
            image_width: Output image width, in pixels (default: 256).
            image_height: Output image height, in pixels (default: 256).
            n_pts_per_ray: Number of samples per ray -- matched exactly to
                PyTorch3D's ``NDCMultinomialRaysampler(n_pts_per_ray=...)`` for
                equivalence (default: 320).
            ndc_extent: Extent of the normalized world cube the CT volume is
                placed in, matching ``ObjectCentricXRayVolumeRenderer``'s
                ``ndc_extent`` (default: 1.0).
            renderer: DiffDRR backend, must be ``"trilinear"`` for equivalence
                (Siddon's fixed per-voxel sampling can't be matched to
                PyTorch3D's fixed-``n_pts_per_ray`` scheme).
            flip_horizontal: Whether to mirror the DiffDRR output horizontally
                before returning it. PyTorch3D's NDC space uses a "+X points
                left" convention, opposite to DiffDRR's detector convention --
                the same discrepancy documented and corrected for in
                ``baselines/DRR/renderers/diffdrr.py`` (default: True). Verify
                against ``compare_dvr.py`` for your geometry before disabling.
            device: Target compute device.
        """
        super().__init__()

        if renderer != "trilinear":
            raise ValueError(
                "DiffDRRVolumeRenderer requires renderer='trilinear' to "
                "match PyTorch3D's fixed n_pts_per_ray sampling scheme."
            )

        self.image_width = image_width
        self.image_height = image_height
        self.n_pts_per_ray = n_pts_per_ray
        self.ndc_extent = ndc_extent
        self.renderer_backend = renderer
        self.flip_horizontal = flip_horizontal
        self.device = torch.device(device)

        self._drr: Optional[DRR] = None

    def _build_subject(self, volume: torch.Tensor) -> Tuple[tio.Subject, float]:
        """
        Wraps a preprocessed CT tensor in a ``torchio.Subject`` whose affine
        places it in the exact normalized world frame PyTorch3D's ``Volumes``
        uses (see module docstring, "World frame").

        Args:
            volume: CT density tensor, shape ``(D, H, W)``, ``(1, D, H, W)``,
                or ``(1, 1, D, H, W)`` -- the output of
                ``create_ct_transforms`` / ``load_ct_volume``.

        Returns:
            ``(subject, voxel_size)``.
        """
        vol = volume.detach()
        if vol.ndim == 5:
            if vol.shape[0] != 1:
                raise ValueError("DiffDRRVolumeRenderer.set_volume expects a single (unbatched) CT volume.")
            vol = vol[0]
        if vol.ndim == 4:
            vol = vol[0]
        if vol.ndim != 3:
            raise ValueError(f"Expected a (D, H, W) CT volume, got shape {tuple(volume.shape)}.")

        D, H, W = vol.shape
        # Matches ObjectCentricXRayVolumeRenderer's `shape = max(features.shape[2], features.shape[3])`.
        shape = max(D, H)
        voxel_size = 2.0 * float(self.ndc_extent) / float(shape)

        # PyTorch3D's `Volumes` maps a (D, H, W) tensor to world (Z, Y, X).
        # TorchIO images are stored (C, W, H, D) with world axes (X, Y, Z), so
        # reversing the spatial axes aligns them: orig-W -> world X, orig-H ->
        # world Y, orig-D -> world Z.
        tio_tensor = vol.permute(2, 1, 0).unsqueeze(0).contiguous().to(torch.float32).cpu()  # (1, W, H, D)

        offset_w = -0.5 * (W - 1) * voxel_size
        offset_h = -0.5 * (H - 1) * voxel_size
        offset_d = -0.5 * (D - 1) * voxel_size
        affine = np.array(
            [
                [voxel_size, 0.0, 0.0, offset_w],
                [0.0, voxel_size, 0.0, offset_h],
                [0.0, 0.0, voxel_size, offset_d],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        image = tio.ScalarImage(tensor=tio_tensor, affine=affine)
        subject = tio.Subject(
            volume=image,
            density=image,
            mask=None,
            reorient=torch.eye(4, dtype=torch.float32),
        )
        return subject, voxel_size

    def set_volume(self, volume: torch.Tensor) -> "DiffDRRVolumeRenderer":
        """
        Binds a preprocessed CT volume to this renderer, rebuilding the
        underlying DiffDRR module. Call this once per volume, then call
        ``render(...)`` (optionally many times) with different
        dist/elev/fov/azimuth values.

        Args:
            volume: CT density tensor in ``[0, 1]``, shape ``(D, H, W)``,
                ``(1, D, H, W)``, or ``(1, 1, D, H, W)``.

        Returns:
            ``self``, for chaining.
        """
        subject, _ = self._build_subject(volume)

        # sdd/delx/dely are placeholders here -- `render()` recomputes and
        # overwrites them via `set_intrinsics_` for every call, since they
        # depend on fov/min_depth/max_depth which are render-time arguments.
        self._drr = DRR(
            subject,
            sdd=1.0,
            height=self.image_height,
            width=self.image_width,
            delx=1.0,
            dely=1.0,
            renderer=self.renderer_backend,
        ).to(self.device)
        return self

    @torch.no_grad()
    def render(
        self,
        azimuth: AngleLike,
        volume: Optional[torch.Tensor] = None,
        elev: AngleLike = 0.0,
        dist: AngleLike = 8.0,
        fov: float = 12.0,
        min_depth: float = 7.0,
        max_depth: float = 9.0,
        norm_type: str = "standardized",
        batch_size: int = 16,
    ) -> torch.Tensor:
        """
        Renders one or more X-ray projections of the bound CT volume.

        Args:
            azimuth: Azimuth angle(s) in degrees. Scalar for a single view, or
                a sequence/tensor of ``T`` angles for a multiview batch (e.g.
                ``torch.linspace(0, 360, 93)`` for a 360-degree sweep).
            volume: Optional CT tensor; if given, calls ``set_volume(volume)``
                first. Otherwise a volume must already be bound.
            elev: Elevation angle in degrees, broadcastable against ``azimuth``.
            dist: Camera distance from the origin, in the same normalized
                world units as ``ndc_extent`` (default: 8.0, matching
                ``scripts.build_dataset.create_multiview_cameras``).
            fov: Full vertical field of view in degrees, matching
                ``FoVPerspectiveCameras(fov=...)`` (default: 12.0).
            min_depth, max_depth: Camera-space near/far integration bounds,
                matching ``NDCMultinomialRaysampler(min_depth=..., max_depth=...)``.
            norm_type: One of ``"minimized"``, ``"normalized"``,
                ``"standardized"``, or ``None`` -- applied identically to
                ``BaseXRayVolumeRenderer.forward``'s post-processing.
            batch_size: Number of views rendered per DiffDRR forward pass.

        Returns:
            Rendered images, shape ``(T, 1, image_height, image_width)``.
        """
        if volume is not None:
            self.set_volume(volume)
        if self._drr is None:
            raise RuntimeError("Call set_volume(...) or pass volume=... before rendering.")

        azim_t = torch.as_tensor(azimuth, device=self.device, dtype=torch.float32).reshape(-1)
        n_frames = azim_t.shape[0]
        elev_t = torch.as_tensor(elev, device=self.device, dtype=torch.float32).expand_as(azim_t)
        dist_t = torch.as_tensor(dist, device=self.device, dtype=torch.float32).expand_as(azim_t)

        poses = look_at_view_poses(dist_t, elev_t, azim_t, device=self.device, dtype=torch.float32)

        # Choosing sdd = max_depth places the virtual detector plane exactly
        # at the camera-space far bound, so alphamax below comes out to 1.0.
        sdd = float(max_depth)
        delx, dely = _fov_to_pixel_spacing(
            fov=float(fov),
            sdd=sdd,
            image_width=self.image_width,
            image_height=self.image_height,
            ndc_extent=float(self.ndc_extent),
        )
        # local_z = alpha * sdd for every ray (the detector plane is flat and
        # perpendicular to the viewing axis), so this exactly reproduces
        # PyTorch3D's [min_depth, max_depth] camera-space sampling window.
        alphamin = float(min_depth) / sdd
        alphamax = float(max_depth) / sdd

        self._drr.set_intrinsics_(sdd=sdd, delx=delx, dely=dely, height=self.image_height, width=self.image_width)

        frames = []
        for start in range(0, n_frames, batch_size):
            chunk = RigidTransform(poses.matrix[start:start + batch_size].to(self.device))
            img = self._drr(
                chunk,
                n_points=self.n_pts_per_ray,
                alphamin=alphamin,
                alphamax=alphamax,
            )  # (b, 1, H, W)
            frames.append(img)
        img = torch.cat(frames, dim=0)

        if self.flip_horizontal:
            img = torch.flip(img, dims=[-1])

        if norm_type == "minimized":
            img = minimized(img)
        elif norm_type == "normalized":
            img = normalized(img)
        elif norm_type == "standardized":
            img = normalized(standardized(img))

        return img


# ═════════════════════════════════════════════════════════════════════════════
# scripts/build_dataset.py-compatible convenience API
# ═════════════════════════════════════════════════════════════════════════════

def create_diffdrr_renderer(
    img_shape: int = 256,
    n_pts_per_ray: int = 1000,
    ndc_extent: float = 1.0,
    flip_horizontal: bool = False,
    device: Union[str, torch.device] = "cuda",
) -> DiffDRRVolumeRenderer:
    """DiffDRR analogue of ``scripts.build_dataset.create_renderer``."""
    return DiffDRRVolumeRenderer(
        image_width=img_shape,
        image_height=img_shape,
        n_pts_per_ray=n_pts_per_ray,
        ndc_extent=ndc_extent,
        flip_horizontal=flip_horizontal,
        device=device,
    )


def render_diffdrr_multiview_frames(
    vol: torch.Tensor,
    num_frames: int = 93,
    img_shape: int = 256,
    device: Union[str, torch.device] = "cuda",
    dist: float = 8.0,
    elev: float = 0.0,
    fov: float = 12.0,
    min_depth: float = 7.0,
    max_depth: float = 9.0,
    renderer: Optional[DiffDRRVolumeRenderer] = None,
    batch_size: int = 16,
) -> torch.Tensor:
    """
    DiffDRR analogue of ``scripts.build_dataset.render_multiview_frames``: a
    360-degree azimuth sweep at fixed dist/elev/fov, normalized the same way
    (``norm_type="standardized"``).

    Args:
        vol: Preprocessed CT volume, shape ``(1, D, H, W)`` or ``(1, 1, D, H, W)``.
        num_frames: Number of frames for a full 360-degree rotation (default: 93).
        img_shape: Output image height/width (default: 256).
        device: Target compute device.
        dist, elev, fov, min_depth, max_depth: See ``DiffDRRXRayVolumeRenderer.render``.
        renderer: Optional pre-initialized renderer to reuse across calls.
        batch_size: Number of views rendered per DiffDRR forward pass.

    Returns:
        Rendered video tensor, shape ``(1, 1, num_frames, img_shape, img_shape)``.
    """
    if renderer is None:
        renderer = create_diffdrr_renderer(img_shape=img_shape, device=device)
    renderer.set_volume(vol)

    azimuths = torch.linspace(0, 360, num_frames)
    frames = renderer.render(
        azimuth=azimuths,
        elev=elev,
        dist=dist,
        fov=fov,
        min_depth=min_depth,
        max_depth=max_depth,
        norm_type="standardized", # Enforce beautiful standardized normalization for high-contrast medical visuals
        batch_size=1, # Fixed chunk_size=1 to avoid OOM on 256x256x256 volumes
    )  # (T, 1, H, W)

    return frames.unsqueeze(0).transpose(1, 2)  # (1, 1, T, H, W)


__all__ = [
    "DiffDRRVolumeRenderer",
    "look_at_view_poses",
    "create_diffdrr_renderer",
    "render_diffdrr_multiview_frames",
]
