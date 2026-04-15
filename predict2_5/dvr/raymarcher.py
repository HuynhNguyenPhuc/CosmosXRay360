"""
X-Ray Volume Ray Marching Module

Reference: https://github.com/tmquan/cosmed/blob/main/dvr/raymarcher.py
"""

import torch

from pytorch3d.renderer import EmissionAbsorptionRaymarcher
from pytorch3d.renderer.implicit.raymarching import (
    _check_density_bounds,
    _check_raymarcher_inputs,
    _shifted_cumprod,
)


class AbsorptionEmissionRaymarcher(EmissionAbsorptionRaymarcher):
    """Custom Absorption-Emission Ray Marcher for X-ray volume rendering."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        rays_densities: torch.Tensor,
        rays_features: torch.Tensor,
        eps: float = 1e-10,
        **kwargs,
    ) -> torch.Tensor:
        """
        Perform Absorption-Emission Ray Marching.

        Args:
            rays_densities: Tensor of shape (batch, num_rays, num_samples, 1) containing density values.
            rays_features: Tensor of shape (batch, num_rays, num_samples, feature_dim) containing features.
            eps: Small value to prevent numerical issues (default: 1e-10).
        
        Returns:
            Tensor of shape (batch, num_rays, feature_dim + 1) containing the final color (feature_dim channels) and opacity (1 channel).
        """
        # Validate Ray Marcher Inputs
        _check_raymarcher_inputs(
            rays_densities, rays_features, None, 
            z_can_be_none=True, 
            features_can_be_none=False, 
            density_1d=True,
        )

        # Ensure density values are in valid range [0, 1]
        _check_density_bounds(rays_densities)

        # For X-ray rendering, treat densities as absorption coefficients.
        rays_densities = rays_densities[..., 0]
        
        # Reverse the direction of the absorption to match X-ray detector behavior
        absorption = _shifted_cumprod(
            (1.0 + eps) - rays_densities.flip(dims=(-1,)), 
            shift=-self.surface_thickness
        ).flip(dims=(-1,))  
        
        # Compute weighted features based on absorption along ray
        weights = rays_densities * absorption
        features = (weights[..., None] * rays_features).sum(dim=-2)
        
        # Calculate final opacity from density accumulation
        opacities = 1.0 - torch.prod(1.0 - rays_densities, dim=-1, keepdim=True)
    
        return torch.cat((features, opacities), dim=-1)


# Aliases for different raymarching strategies
ScreenCentricRaymarcher = AbsorptionEmissionRaymarcher
ObjectCentricRaymarcher = EmissionAbsorptionRaymarcher
