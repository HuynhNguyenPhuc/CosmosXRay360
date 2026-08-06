import torch
import pytest
from models.pixelnerf import PixelNeRFWrapper

def test_pixelnerf_wrapper_fallback_gracefully(mock_frontal_cxr):
    wrapper = PixelNeRFWrapper()
    
    if wrapper.model is None:
        views = wrapper.infer_multi_views(mock_frontal_cxr)
        assert views == []
    else:
        views = wrapper.infer_multi_views(mock_frontal_cxr, azimuths=(0, 360, 2))
        assert len(views) == 2
        for view in views:
            assert view.dim() == 2
