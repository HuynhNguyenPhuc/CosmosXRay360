import torch
import pytest
from models.utils import compute_psnr, compute_ssim, compute_lpips, apply_beer_lambert_correction

def test_psnr_identical():
    # PSNR of identical tensors should be 100.0 (by definition/implementation)
    t1 = torch.rand(1, 1, 128, 128)
    t2 = t1.clone()
    val = compute_psnr(t1, t2)
    assert val == 100.0

def test_psnr_different():
    t1 = torch.zeros(1, 1, 128, 128)
    t2 = torch.ones(1, 1, 128, 128)
    val = compute_psnr(t1, t2)
    # Mean Squared Error is 1.0, 20 * log10(1 / sqrt(1.0)) = 0.0
    assert abs(val - 0.0) < 1e-4

def test_ssim_range():
    t1 = torch.rand(1, 1, 128, 128)
    t2 = torch.rand(1, 1, 128, 128)
    val = compute_ssim(t1, t2)
    assert -1.0 <= val <= 1.0

def test_ssim_identical():
    t1 = torch.rand(1, 1, 128, 128)
    val = compute_ssim(t1, t1)
    # Identical tensors should yield an SSIM value extremely close to 1.0
    assert abs(val - 1.0) < 1e-2

def test_lpips_identical():
    # LPIPS of identical tensors should be exactly 0.0 (no perceptual distance).
    t1 = torch.rand(1, 1, 64, 64)
    val = compute_lpips(t1, t1.clone())
    assert val == 0.0

def test_lpips_different_is_positive():
    t1 = torch.zeros(1, 1, 64, 64)
    t2 = torch.ones(1, 1, 64, 64)
    val = compute_lpips(t1, t2)
    assert val > 0.0

def test_lpips_grayscale_input_shape():
    # Single-channel [B, 1, H, W] input (this project's convention) must not
    # raise despite LPIPS' AlexNet backbone expecting 3-channel RGB internally.
    t1 = torch.rand(2, 1, 64, 64)
    t2 = torch.rand(2, 1, 64, 64)
    val = compute_lpips(t1, t2)
    assert isinstance(val, float)

def test_beer_lambert_physics():
    # High attenuation should give near 0 transmissivity
    attenuation_high = torch.tensor([10.0])
    trans_high = apply_beer_lambert_correction(attenuation_high)
    assert trans_high.item() < 1e-3
    
    # Zero attenuation should give trans = I0 = 1.0
    attenuation_zero = torch.tensor([0.0])
    trans_zero = apply_beer_lambert_correction(attenuation_zero)
    assert abs(trans_zero.item() - 1.0) < 1e-5
