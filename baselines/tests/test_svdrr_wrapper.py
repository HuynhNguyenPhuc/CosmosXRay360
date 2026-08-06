import torch
import pytest
from models.svdrr import SVDRRWrapper

def test_svdrr_wrapper_fallback_gracefully(mock_frontal_cxr):
    # If the system does not have the required pipeline, it should degrade gracefully.
    wrapper = SVDRRWrapper()
    
    # Check if pipeline isn't loaded (which is expected unless installed)
    if wrapper.pipe is None:
        views = wrapper.infer_multi_views(mock_frontal_cxr)
        assert views == []
    else:
        views = wrapper.infer_multi_views(mock_frontal_cxr, azimuths=(0, 360, 2))
        assert len(views) == 2
        for view in views:
            assert view.shape == (256, 256)
            assert (view >= 0.0).all() and (view <= 1.0).all()
