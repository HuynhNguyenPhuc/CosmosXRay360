import torch
import pytest
from models.mednerf import MedNeRFWrapper

def test_mednerf_wrapper_fallback_gracefully(mock_frontal_cxr):
    wrapper = MedNeRFWrapper()
    
    if wrapper.model is None:
        views = wrapper.infer_multi_views(mock_frontal_cxr)
        assert views == []
    else:
        # Mock latent optimization loop and verify F.interpolate upsamples correctly to 256x256
        views = wrapper.infer_multi_views(mock_frontal_cxr, azimuths=(0, 360, 2))
        assert len(views) == 2
        for view in views:
            assert view.shape == (256, 256)
