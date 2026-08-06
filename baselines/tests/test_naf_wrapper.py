import torch
import pytest
from models.naf import NAFWrapper

def test_naf_wrapper_fallback_gracefully(mock_frontal_cxr):
    wrapper = NAFWrapper()
    
    if wrapper.model is None:
        views = wrapper.infer_multi_views(mock_frontal_cxr)
        assert views == []
    else:
        views = wrapper.infer_multi_views(mock_frontal_cxr, azimuths=(0, 360, 2))
        assert len(views) == 2
        for view in views:
            # Query grid outputs projected via Beer-Lambert correction
            assert view.dim() == 2
            assert (view >= 0.0).all() and (view <= 1.0).all()
