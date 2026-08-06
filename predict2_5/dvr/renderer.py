"""
X-Ray Volume Renderer Module.

Reference: https://github.com/tmquan/cosmed/blob/main/dvr/renderer.py
"""

import torch
import torch.nn as nn

from pytorch3d.structures import Volumes
from pytorch3d.renderer import (
    VolumeRenderer,
    NDCMultinomialRaysampler,
    EmissionAbsorptionRaymarcher
)

from predict2_5.dvr.raymarcher import AbsorptionEmissionRaymarcher
from predict2_5.utils.transforms import minimized, normalized, standardized


class BaseXRayVolumeRenderer(nn.Module):
    """Base class for X-Ray Volume Rendering using PyTorch3D."""

    def __init__(
        self,
        image_width: int = 256,
        image_height: int = 256,
        n_pts_per_ray: int = 320,
        min_depth: float = 3.0,
        max_depth: float = 9.0,
        ndc_extent: float = 1.0,
    ):
        """
        Initialize the X-Ray Volume Renderer.
        
        Args:
            image_width: Width of the output rendered image (default: 256).
            image_height: Height of the output rendered image (default: 256).
            n_pts_per_ray: Number of sample points per ray (default: 320).
            min_depth: Minimum depth for ray sampling (default: 3.0).
            max_depth: Maximum depth for ray sampling (default: 9.0).
            ndc_extent: Extent of the NDC space for ray sampling (default: 1.0).
        """
        super().__init__()
        
        self.n_pts_per_ray = n_pts_per_ray
        self.image_width = image_width
        self.image_height = image_height
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.ndc_extent = ndc_extent

    def _create_raysampler(self):
        """Create raysampler for volume rendering."""
        pass

    def _create_raymarcher(self):
        """Create raymarcher for volume rendering."""
        pass

    def _setup_renderer(self):
        """Set up the volume renderer."""
        # Create the Ray Sampler
        raysampler = self._create_raysampler()

        # Create the Ray Marcher
        raymarcher = self._create_raymarcher()

        # Create the Volume Renderer
        self.renderer = VolumeRenderer(
            raysampler=raysampler, 
            raymarcher=raymarcher
        )

    def forward(
        self,
        volume,
        cameras,
        opacity=None,
        norm_type="standardized",
        scaling_factor=1.0,
        is_grayscale=True,
        return_bundle=False,
        stratified_sampling=False
    ) -> torch.Tensor:
        """
        Render X-Ray images from the input volume with specified cameras.
        
        Args:
            volume: Input volume tensor of shape (B, C, D, H, W).
            cameras: PyTorch3D cameras object defining the viewpoints.
            opacity: Optional tensor of shape (B, 1, D, H, W) representing opacity values (default: None).
            norm_type: Normalization type for output image ("minimized", "normalized", "standardized", or None) (default: "standardized").
            scaling_factor: Scaling factor for density values (default: 1.0).
            is_grayscale: Whether to convert output to grayscale (default: True).
            return_bundle: Whether to return the rendering bundle with additional info (default: False).
            stratified_sampling: Whether to use stratified sampling along rays (default: False).
        
        Returns:
            Rendered image tensor of shape (B, C, H, W) or a tuple
            (rendered_image, bundle) if return_bundle is True.
        """
        # Use the grayscale value as features
        features = volume.repeat(1, 3, 1, 1, 1) if volume.shape[1] == 1 else volume
        
        # Use provided opacity or create uniform opacity
        densities = (
            opacity * scaling_factor 
            if opacity is not None 
            else torch.ones_like(volume[:, [0]]) * scaling_factor
        )

        # Get the maximum spatial dimension for voxel size calculation
        shape = max(features.shape[2], features.shape[3])

        # Create Pytorch3D Volumes object with features and densities for rendering
        volumes = Volumes(
            features=features,
            densities=densities,
            voxel_size=2.0 * float(self.ndc_extent) / shape,
        )

        # Perform volume rendering to get RGBA output and rendering bundle
        screen_RGBA, bundle = self.renderer(cameras=cameras, volumes=volumes)

        # Permute RGBA output to (B, C, H, W) format for further processing
        screen_RGBA = screen_RGBA.permute(0, 3, 1, 2)
        
        # Extract RGB channels
        rgb_channels = screen_RGBA[:, :3, :, :]

        # Convert to grayscale if requested
        screen_RGB = (
            rgb_channels.mean(dim=1, keepdim=True) 
            if is_grayscale 
            else rgb_channels
        )

        # Apply normalization if specified
        if norm_type == "minimized":
            screen_RGB = minimized(screen_RGB)
        elif norm_type == "normalized":
            screen_RGB = normalized(screen_RGB)
        elif norm_type == "standardized":
            screen_RGB = normalized(standardized(screen_RGB))

        return (screen_RGB, bundle) if return_bundle else screen_RGB


class ScreenCentricXRayVolumeRenderer(BaseXRayVolumeRenderer):
    """Screen-centric X-ray volume renderer."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set up the renderer
        self._setup_renderer()

    def _create_raysampler(self):
        """Create NDCMultinomialRaysampler for screen-centric rendering."""
        return NDCMultinomialRaysampler(
            image_width=self.image_width,
            image_height=self.image_height,
            n_pts_per_ray=self.n_pts_per_ray,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            stratified_sampling=False, 
        )

    def _create_raymarcher(self):
        """Create EmissionAbsorptionRaymarcher for screen-centric rendering."""
        return EmissionAbsorptionRaymarcher()


class ObjectCentricXRayVolumeRenderer(BaseXRayVolumeRenderer):
    """Object-centric X-ray volume renderer."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set up the renderer
        self._setup_renderer()

    def _create_raymarcher(self):
        """Create AbsorptionEmissionRaymarcher for object-centric rendering."""
        return AbsorptionEmissionRaymarcher()

    def _create_raysampler(self):
        """Create NDCMultinomialRaysampler for object-centric rendering."""
        return NDCMultinomialRaysampler(
            image_width=self.image_width,
            image_height=self.image_height,
            n_pts_per_ray=self.n_pts_per_ray,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            stratified_sampling=False, 
        )
