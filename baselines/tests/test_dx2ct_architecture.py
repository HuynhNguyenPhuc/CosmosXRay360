"""Unit tests verifying the custom modular DX2CT architecture layers and combined pipeline."""

from __future__ import annotations

import pytest
import torch

from cloned.DX2CT.model import (
    SinusoidalPositionEncoding,
    XRayFeatureExtractor,
    SliceTransformer3DPQT,
    SPADE,
    SPADEResNetBlock,
    DenoisingUNetSPADE,
    DX2CTModel,
)


def test_sinusoidal_position_encoding(test_device):
    # Shape: [B, N, C] = [2, 10, 3]
    coords = torch.rand(2, 10, 3, device=test_device)
    encoder = SinusoidalPositionEncoding(d_model=64).to(test_device)
    
    emb = encoder(coords)
    # Output must be [B, N, C * d_model] = [2, 10, 3 * 64] = [2, 10, 192]
    assert emb.shape == (2, 10, 192)
    assert not torch.isnan(emb).any()


def test_xray_feature_extractor(test_device):
    # Grayscale image of size 256x256
    xray = torch.rand(2, 1, 256, 256, device=test_device)
    extractor = XRayFeatureExtractor(embed_dim=128, pretrained=False).to(test_device)
    
    p1, p2, p3 = extractor(xray)
    
    # Scale 1: stride 4 (64x64)
    assert p1.shape == (2, 128, 64, 64)
    # Scale 2: stride 8 (32x32)
    assert p2.shape == (2, 128, 32, 32)
    # Scale 3: stride 16 (16x16)
    assert p3.shape == (2, 128, 16, 16)


def test_slice_transformer_3dpqt(test_device):
    # Target 3D coordinates for a downsampled grid (e.g. 16x16) to keep test lightweight
    coords_3d = torch.rand(2, 256, 3, device=test_device)
    xray_feats = torch.rand(2, 512, 128, device=test_device)
    
    transformer = SliceTransformer3DPQT(embed_dim=128, num_heads=4).to(test_device)
    output = transformer(coords_3d, xray_feats)
    
    # Output should match query shape [B, Seq_len, embed_dim] = [2, 256, 128]
    assert output.shape == (2, 256, 128)


def test_spade_layer(test_device):
    x = torch.rand(2, 64, 32, 32, device=test_device)
    condition = torch.rand(2, 128, 32, 32, device=test_device)
    
    spade = SPADE(norm_nc=64, label_nc=128).to(test_device)
    out = spade(x, condition)
    
    assert out.shape == x.shape


def test_spade_resnet_block(test_device):
    x = torch.rand(2, 64, 32, 32, device=test_device)
    condition = torch.rand(2, 128, 32, 32, device=test_device)
    
    # Matching channels
    block_same = SPADEResNetBlock(in_channels=64, out_channels=64, cond_channels=128).to(test_device)
    out_same = block_same(x, condition)
    assert out_same.shape == (2, 64, 32, 32)
    
    # Mismatching channels (triggering learned shortcut)
    block_diff = SPADEResNetBlock(in_channels=64, out_channels=128, cond_channels=128).to(test_device)
    out_diff = block_diff(x, condition)
    assert out_diff.shape == (2, 128, 32, 32)


def test_denoising_unet_spade(test_device):
    noisy_slice = torch.rand(2, 1, 256, 256, device=test_device)
    condition = torch.rand(2, 128, 256, 256, device=test_device)
    
    unet = DenoisingUNetSPADE(in_channels=1, cond_channels=128).to(test_device)
    noise_pred = unet(noisy_slice, condition)
    
    assert noise_pred.shape == (2, 1, 256, 256)


def test_complete_dx2ct_model_pipeline(test_device):
    # Complete batch pass simulating slice-diffusion query step
    noisy_slice = torch.rand(1, 1, 256, 256, device=test_device)
    pa_xray = torch.rand(1, 1, 256, 256, device=test_device)
    lat_xray = torch.rand(1, 1, 256, 256, device=test_device)
    
    # Grid coordinates mapping to axial pixels (flat sequence size 256*256)
    # To keep memory footprint low in local test suite, we simulate a sequence of 1024 voxel queries
    coords_3d = torch.rand(1, 1024, 3, device=test_device)
    
    model = DX2CTModel(embed_dim=128).to(test_device)
    
    with torch.no_grad():
        # Test full sub-parts with dummy 1024-length coordinates
        pa_p1, pa_p2, pa_p3 = model.feature_extractor(pa_xray)
        lat_p1, lat_p2, lat_p3 = model.feature_extractor(lat_xray)
        pa_flat = pa_p3.flatten(2).permute(0, 2, 1)
        lat_flat = lat_p3.flatten(2).permute(0, 2, 1)
        combined_xray_feats = torch.cat([pa_flat, lat_flat], dim=1)
        
        pos_aware_feats_flat = model.transformer_3dpqt(coords_3d, combined_xray_feats)
        assert pos_aware_feats_flat.shape == (1, 1024, 128)
        
        # Verify full standard forward pass with 256*256 coordinate sequence mapping
        full_coords = torch.rand(1, 256 * 256, 3, device=test_device)
        noise_prediction = model(noisy_slice, pa_xray, lat_xray, full_coords)
        
        assert noise_prediction.shape == (1, 1, 256, 256)
        assert not torch.isnan(noise_prediction).any()
