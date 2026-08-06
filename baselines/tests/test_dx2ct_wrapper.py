import torch
import pytest
from models.dx2ct import Dx2CTWrapper
from cloned.DX2CT.model import DX2CTModel

def test_dx2ct_model_shape(test_device):
    # Initialize model
    model = DX2CTModel(embed_dim=128).to(test_device)
    model.eval()
    
    # Input image batch [B, C, H, W]
    noisy_slice = torch.rand(2, 1, 64, 64, device=test_device)
    pa_xray = torch.rand(2, 1, 256, 256, device=test_device)
    lat_xray = torch.rand(2, 1, 256, 256, device=test_device)
    coords_3d = torch.rand(2, 64 * 64, 3, device=test_device)
    
    with torch.no_grad():
        output = model(noisy_slice, pa_xray, lat_xray, coords_3d)
        
    assert output.shape == (2, 1, 64, 64)

def test_dx2ct_wrapper_inference(mock_frontal_cxr, test_device):
    wrapper = Dx2CTWrapper()
    if wrapper.model is None or wrapper.scheduler is None:
        pytest.skip("Required libraries not available")
    
    # Run mock inference for 2 azimuth views
    views = wrapper.infer_multi_views(mock_frontal_cxr, azimuths=(0, 360, 2))
    
    assert len(views) == 2
    for view in views:
        assert view.dim() == 2
        assert view.shape == (256, 256)
        # Value must be in transmissive min-max bounds [0, 1] due to Beer-lambert outputs
        assert (view >= 0.0).all() and (view <= 1.0).all()
