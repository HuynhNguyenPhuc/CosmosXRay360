import torch
import pytest
from models.xraysyn import XRaySynWrapper

def test_xraysyn_wrapper_fallback_gracefully(mock_frontal_cxr, test_device):
    wrapper = XRaySynWrapper()
    mock_frontal_cxr = mock_frontal_cxr.to(test_device)
    
    if wrapper.model is None:
        views = wrapper.infer_multi_views(mock_frontal_cxr)
        assert views == []
    else:
        # Test input rank conversion & max intensity projection outputs
        views = wrapper.infer_multi_views(mock_frontal_cxr, azimuths=(0, 360, 2))
        assert len(views) > 0 # XraySyn returns fixed predefined poses, so we just check it returns something
        for view in views:
            assert view.dim() == 2
