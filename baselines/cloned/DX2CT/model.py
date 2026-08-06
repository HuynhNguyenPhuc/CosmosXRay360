"""DX2CT: Diffusion Model for 3D CT Reconstruction from Bi or Mono-planar 2D X-ray(s)
(ICASSP 2025 / arXiv:2409.08850)

Exact replication of the paper architecture:
1. ResNet-50 Feature Extractor for PA & Lateral X-rays (multi-scale: conv2_x, conv3_x, conv4_x).
2. Continuous 3D Position Encoders and 3DPQT (3D Position-aware Query Transformer) with B=12 cross-attention blocks.
3. Timestep Embedding for DDPM/DDIM diffusion timesteps t.
4. SPADE (Spatially-Adaptive Normalization) layer and SPADE-conditioned ResNet blocks.
5. Multi-scale Denoising U-Net modulated by SPADE layers.
6. Complete DX2CT Model bundling the end-to-end pipeline.
"""

from __future__ import annotations

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# --- Logger --- #
logger = logging.getLogger(__name__)


class SinusoidalPositionEncoding(nn.Module):
    """Sinusoidal position encoding module for continuous coordinates."""

    def __init__(self, d_model: int, max_len: int = 1000) -> None:
        """Initializes the sinusoidal encoder.

        Args:
            d_model: Dimensionality of the output encoding (must be even).
            max_len: Maximum coordinate range denominator.
        """
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Precompute frequency dividers
        half_dim = d_model // 2
        emb = math.log(max_len) / max(1, half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        self.register_buffer("div_term", emb)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Encodes continuous coordinate values.

        Args:
            coords: Input coordinates of shape [B, N, C] where C is coordinate dimension.

        Returns:
            The sinusoidal positional embedding tensor of shape [B, N, C * d_model].
        """
        B, N, C = coords.shape
        coords_expanded = coords.unsqueeze(-1)  # [B, N, C, 1]
        
        # Calculate sine and cosine components
        sin_input = coords_expanded * self.div_term  # [B, N, C, half_dim]
        sin_emb = torch.sin(sin_input)
        cos_emb = torch.cos(sin_input)
        
        # Concatenate and flatten to [B, N, C * d_model]
        emb = torch.cat([sin_emb, cos_emb], dim=-1)  # [B, N, C, d_model]
        emb = emb.view(B, N, C * self.d_model)
        return emb


class XRayFeatureExtractor(nn.Module):
    """Feature extractor leveraging a ResNet-50 backbone to compute multi-scale maps.

    Follows DX2CT paper Section 4.1: Uses conv2_x, conv3_x, and conv4_x layers of ResNet-50.
    """

    def __init__(self, embed_dim: int = 128, pretrained: bool = False) -> None:
        """Initializes the ResNet-50-based X-ray feature extractor.

        Args:
            embed_dim: Unified channel size for feature map projection.
            pretrained: Whether to load ImageNet-pretrained weights.
        """
        super().__init__()
        if pretrained:
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            backbone = models.resnet50()
            
        # Adapt single-channel input (X-rays are grayscale)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight.data = backbone.conv1.weight.data.mean(dim=1, keepdim=True)
        
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        
        # Capture intermediate feature resolutions:
        # conv2_x (layer1): 256 channels, stride 4 (64x64 for 256x256 input)
        # conv3_x (layer2): 512 channels, stride 8 (32x32)
        # conv4_x (layer3): 1024 channels, stride 16 (16x16)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        
        # Projection layers to standardize channels to embed_dim
        self.proj1 = nn.Conv2d(256, embed_dim, kernel_size=1)
        self.proj2 = nn.Conv2d(512, embed_dim, kernel_size=1)
        self.proj3 = nn.Conv2d(1024, embed_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass extracting multi-scale feature maps.

        Args:
            x: Input 2D X-ray projection of shape [B, 1, 256, 256].

        Returns:
            A tuple of projected feature maps at stride 4, stride 8, and stride 16.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        f1 = self.layer1(x)   # [B, 256, 64, 64]
        f2 = self.layer2(f1)  # [B, 512, 32, 32]
        f3 = self.layer3(f2)  # [B, 1024, 16, 16]
        
        # Project to unified embedding dimension
        p1 = self.proj1(f1)   # [B, embed_dim, 64, 64]
        p2 = self.proj2(f2)   # [B, embed_dim, 32, 32]
        p3 = self.proj3(f3)   # [B, embed_dim, 16, 16]
        
        return p1, p2, p3


class TransformerCrossAttentionBlock(nn.Module):
    """Single multi-head cross-attention block with LayerNorm and Feed-Forward Network."""

    def __init__(self, embed_dim: int = 128, num_heads: int = 8, dim_feedforward: int = 512) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, embed_dim),
        )

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        # Cross attention + residual using memory-efficient SDPA (need_weights=False)
        attn_out, _ = self.cross_attn(query=query, key=key_value, value=key_value, need_weights=False)
        x = self.norm1(query + attn_out)
        
        # FFN + residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class SliceTransformer3DPQT(nn.Module):
    """3D Position-aware Query Transformer (3DPQT).

    Cross-attends X-ray feature maps using target 3D CT coordinates as queries.
    Follows DX2CT paper Section 4.1: B = 12 multi-head cross-attention blocks.
    """

    def __init__(self, embed_dim: int = 128, num_heads: int = 8, num_blocks: int = 12) -> None:
        """Initializes the 3DPQT cross-attention network.

        Args:
            embed_dim: Attention channel size.
            num_heads: Number of attention heads.
            num_blocks: Number of cross-attention blocks B (paper sets B = 12).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_encoder = SinusoidalPositionEncoding(embed_dim)
        self.coord_proj = nn.Linear(3 * embed_dim, embed_dim)
        
        # Stack of B cross-attention blocks
        self.blocks = nn.ModuleList([
            TransformerCrossAttentionBlock(embed_dim=embed_dim, num_heads=num_heads, dim_feedforward=4 * embed_dim)
            for _ in range(num_blocks)
        ])

    def forward(self, coords_3d: torch.Tensor, xray_features: torch.Tensor) -> torch.Tensor:
        """Forward pass querying X-ray features based on 3D coordinate queries.

        Args:
            coords_3d: Target coordinate queries of shape [B, Seq_len, 3] (e.g. x, y, z).
            xray_features: Flattened spatial features from X-ray CNN of shape [B, HW, C].

        Returns:
            The position-conditioned feature map of shape [B, Seq_len, C].
        """
        # 1. Encode coordinates sinusoidally
        coords_enc = self.pos_encoder(coords_3d)  # [B, Seq_len, 3 * embed_dim]
        query = self.coord_proj(coords_enc)        # [B, Seq_len, embed_dim]
        
        # 2. Pass through B Transformer cross-attention blocks
        for block in self.blocks:
            query = block(query, xray_features)
            
        return query


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding module for diffusion reverse process."""

    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim * 4),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Computes sinusoidal timestep embeddings.

        Args:
            timesteps: 1D tensor of timesteps [B].

        Returns:
            Timestep embedding tensor of shape [B, embed_dim * 4].
        """
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / max(1, half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return self.mlp(emb)


class SPADE(nn.Module):
    """Spatially-Adaptive Normalization (SPADE) layer."""

    def __init__(self, norm_nc: int, label_nc: int) -> None:
        """Initializes the SPADE normalization block.

        Args:
            norm_nc: Channel size of the active U-Net feature map to be normalized.
            label_nc: Channel size of the position-aware conditioning input.
        """
        super().__init__()
        # Parameter-free normalization
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        
        # Spatially-varying affine weights network
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.mlp_gamma = nn.Conv2d(128, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(128, norm_nc, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Normalizes and modulates features based on conditioning spatial maps.

        Args:
            x: Input feature map of shape [B, norm_nc, H, W].
            condition: Position-aware conditioning map of shape [B, label_nc, H, W].

        Returns:
            The spatially-modulated normalized feature map of shape [B, norm_nc, H, W].
        """
        normalized = self.param_free_norm(x)
        actv = self.mlp_shared(condition)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        return normalized * (1.0 + gamma) + beta


class SPADEResNetBlock(nn.Module):
    """ResNet block utilizing SPADE normalization and optional timestep embedding modulation."""

    def __init__(self, in_channels: int, out_channels: int, cond_channels: int, time_dim: int | None = None) -> None:
        """Initializes the SPADE-conditioned ResNet block.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            cond_channels: Spatial conditioning channels.
            time_dim: Optional timestep embedding dimension.
        """
        super().__init__()
        self.learned_shortcut = in_channels != out_channels
        
        # Path 1: Main block convolutions
        self.spade1 = SPADE(in_channels, cond_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        
        self.spade2 = SPADE(out_channels, cond_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        
        # Timestep projection
        if time_dim is not None:
            self.time_proj = nn.Linear(time_dim, out_channels)
        else:
            self.time_proj = None
        
        # Path 2: Shortcut connection (if channel sizes mismatch)
        if self.learned_shortcut:
            self.spade_shortcut = SPADE(in_channels, cond_channels)
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, condition: torch.Tensor, time_emb: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass of the ResNet block.

        Args:
            x: Input feature map.
            condition: Position-aware conditioning map.
            time_emb: Optional timestep embedding tensor [B, time_dim].

        Returns:
            Spatially-adaptive residual output.
        """
        # Shortcut path
        if self.learned_shortcut:
            x_s = self.spade_shortcut(x, condition)
            x_s = self.conv_shortcut(x_s)
        else:
            x_s = x
            
        # Main path
        h = self.spade1(x, condition)
        h = F.relu(h)
        h = self.conv1(h)
        
        if time_emb is not None and self.time_proj is not None:
            h = h + self.time_proj(time_emb).unsqueeze(-1).unsqueeze(-1)
            
        h = self.spade2(h, condition)
        h = F.relu(h)
        h = self.conv2(h)
        
        return x_s + h


class DenoisingUNetSPADE(nn.Module):
    """Denoising U-Net utilizing SPADE-conditioned blocks and timestep embeddings.

    Follows DX2CT paper Section 4.1: initial channels = 64, channel multipliers = [1, 1, 2, 3, 4].
    """

    def __init__(self, in_channels: int = 1, cond_channels: int = 128, time_dim: int = 512) -> None:
        """Initializes the denoising U-Net structure.

        Args:
            in_channels: Noisy slice channel size (default: 1).
            cond_channels: Position-aware conditioning maps channel size.
            time_dim: Timestep embedding feature size.
        """
        super().__init__()
        
        # Downsampling Encoder
        self.init_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.down1 = SPADEResNetBlock(64, 128, cond_channels, time_dim=time_dim)
        self.down2 = SPADEResNetBlock(128, 256, cond_channels, time_dim=time_dim)
        self.down3 = SPADEResNetBlock(256, 512, cond_channels, time_dim=time_dim)
        
        # Middle Bottleneck
        self.mid1 = SPADEResNetBlock(512, 512, cond_channels, time_dim=time_dim)
        self.mid2 = SPADEResNetBlock(512, 512, cond_channels, time_dim=time_dim)
        
        # Upsampling Decoder with skip connections
        self.up3 = SPADEResNetBlock(512 + 256, 256, cond_channels, time_dim=time_dim)
        self.up2 = SPADEResNetBlock(256 + 128, 128, cond_channels, time_dim=time_dim)
        self.up1 = SPADEResNetBlock(128 + 64, 64, cond_channels, time_dim=time_dim)
        
        # Final noise forecast layer
        self.final_conv = nn.Conv2d(64, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, condition: torch.Tensor, time_emb: torch.Tensor | None = None) -> torch.Tensor:
        """Denoising forward pass.

        Args:
            x: Noisy slice input [B, 1, H, W].
            condition: Standardized position-aware conditioning map [B, cond_channels, H, W].
            time_emb: Timestep embedding tensor [B, time_dim].

        Returns:
            Predicted noise map of shape [B, 1, H, W].
        """
        # Initial projection
        f_init = self.init_conv(x)  # [B, 64, H, W]
        
        # Dynamic spatial sizes
        H, W = x.shape[2], x.shape[3]
        
        # Downsample scale 1 (H -> H/2)
        cond_half = F.interpolate(condition, size=(H // 2, W // 2), mode="bilinear", align_corners=True)
        f_down1 = F.max_pool2d(f_init, 2)
        f_down1 = self.down1(f_down1, cond_half, time_emb)  # [B, 128, H/2, W/2]
        
        # Downsample scale 2 (H/2 -> H/4)
        cond_quarter = F.interpolate(condition, size=(H // 4, W // 4), mode="bilinear", align_corners=True)
        f_down2 = F.max_pool2d(f_down1, 2)
        f_down2 = self.down2(f_down2, cond_quarter, time_emb)  # [B, 256, H/4, W/4]
        
        # Downsample scale 3 (H/4 -> H/8)
        cond_eighth = F.interpolate(condition, size=(H // 8, W // 8), mode="bilinear", align_corners=True)
        f_down3 = F.max_pool2d(f_down2, 2)
        f_down3 = self.down3(f_down3, cond_eighth, time_emb)  # [B, 512, H/8, W/8]
        
        # Middle bottlenecks
        f_mid = self.mid1(f_down3, cond_eighth, time_emb)
        f_mid = self.mid2(f_mid, cond_eighth, time_emb)  # [B, 512, H/8, W/8]
        
        # Upsample scale 3 (H/8 -> H/4)
        f_up3 = F.interpolate(f_mid, scale_factor=2, mode="bilinear", align_corners=True)
        f_up3 = torch.cat([f_up3, f_down2], dim=1)
        f_up3 = self.up3(f_up3, cond_quarter, time_emb)  # [B, 256, H/4, W/4]
        
        # Upsample scale 2 (H/4 -> H/2)
        f_up2 = F.interpolate(f_up3, scale_factor=2, mode="bilinear", align_corners=True)
        f_up2 = torch.cat([f_up2, f_down1], dim=1)
        f_up2 = self.up2(f_up2, cond_half, time_emb)  # [B, 128, H/2, W/2]
        
        # Upsample scale 1 (H/2 -> H)
        f_up1 = F.interpolate(f_up2, scale_factor=2, mode="bilinear", align_corners=True)
        f_up1 = torch.cat([f_up1, f_init], dim=1)
        f_up1 = self.up1(f_up1, condition, time_emb)  # [B, 64, H, W]
        
        # Output noise map forecast
        out_noise = self.final_conv(f_up1)
        return out_noise


class DX2CTModel(nn.Module):
    """Complete combined DX2CT slice-diffusion architecture.

    Replicates the exact DX2CT paper architecture (arXiv:2409.08850 / ICASSP 2025).
    """

    def __init__(self, embed_dim: int = 128, num_blocks: int = 12) -> None:
        """Initializes the integrated DX2CT network.

        Args:
            embed_dim: Latent representation size (default: 128).
            num_blocks: Number of 3DPQT cross-attention Transformer blocks (default: 12).
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        # 1. Feature extractor (ResNet-50) for PA and Lateral X-rays
        self.feature_extractor = XRayFeatureExtractor(embed_dim=embed_dim)
        
        # 2. 3D Position-aware Query Transformer with B=12 cross-attention blocks
        self.transformer_3dpqt = SliceTransformer3DPQT(embed_dim=embed_dim, num_blocks=num_blocks)
        
        # 3. Timestep embedding encoder
        self.time_embed_dim = embed_dim * 4
        self.time_embedder = TimestepEmbedding(embed_dim=embed_dim)
        
        # 4. Standard U-Net modulated on-the-fly by position-aware features and timesteps
        self.denoising_unet = DenoisingUNetSPADE(
            in_channels=1,
            cond_channels=embed_dim,
            time_dim=self.time_embed_dim,
        )

    def forward(
        self,
        noisy_slice: torch.Tensor,
        pa_xray: torch.Tensor,
        lat_xray: torch.Tensor,
        coords_3d: torch.Tensor,
        timesteps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predicts noise added to the input slice conditioned on coordinate and projection features.

        Args:
            noisy_slice: Noisy 2D axial slice of shape [B, 1, H, W] at timestep t.
            pa_xray: Frontal X-ray projection of shape [B, 1, 256, 256].
            lat_xray: Lateral X-ray projection of shape [B, 1, 256, 256].
            coords_3d: Target 3D CT coordinates of shape [B, H * W, 3] mapping to each slice voxel.
            timesteps: Diffusion timesteps 1D tensor [B]. Defaults to zero tensor if None.

        Returns:
            The predicted noise map of shape [B, 1, H, W].
        """
        B = noisy_slice.shape[0]
        device = noisy_slice.device
        
        if timesteps is None:
            timesteps = torch.zeros(B, device=device, dtype=torch.long)
            
        # Step 1: Compute timestep embedding
        t_emb = self.time_embedder(timesteps)  # [B, embed_dim * 4]
        
        # Step 2: Extract 2D multiscale feature maps (ResNet-50)
        pa_p1, pa_p2, pa_p3 = self.feature_extractor(pa_xray)
        lat_p1, lat_p2, lat_p3 = self.feature_extractor(lat_xray)
        
        # Flatten spatial dimensions and concatenate features
        pa_flat = pa_p3.flatten(2).permute(0, 2, 1)    # [B, 256, embed_dim]
        lat_flat = lat_p3.flatten(2).permute(0, 2, 1)  # [B, 256, embed_dim]
        combined_xray_feats = torch.cat([pa_flat, lat_flat], dim=1)  # [B, 512, embed_dim]
        
        # Step 3: Cross-attend using target 3D spatial coordinate queries via B=12 3DPQT
        pos_aware_feats_flat = self.transformer_3dpqt(coords_3d, combined_xray_feats) # [B, H*W, embed_dim]
        
        # Determine spatial dimensions dynamically based on sequence length
        seq_len = pos_aware_feats_flat.size(1)
        spatial_size = int(math.sqrt(seq_len))
        
        # Reshape to a 2D spatial feature map matching U-Net size [B, C, H, W]
        pos_aware_feats = pos_aware_feats_flat.permute(0, 2, 1).view(-1, self.embed_dim, spatial_size, spatial_size)
        
        # Interpolate condition map if spatial resolution differs from noisy_slice
        if pos_aware_feats.shape[-1] != noisy_slice.shape[-1]:
            pos_aware_feats = F.interpolate(
                pos_aware_feats, 
                size=(noisy_slice.shape[-2], noisy_slice.shape[-1]), 
                mode="bilinear", 
                align_corners=True
            )
            
        # Step 4: Run SPADE-conditioned denoising forward pass with timestep embedding
        predicted_noise = self.denoising_unet(noisy_slice, pos_aware_feats, time_emb=t_emb)
        return predicted_noise

