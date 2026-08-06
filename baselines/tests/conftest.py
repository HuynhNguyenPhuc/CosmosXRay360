import os
import sys
import pytest
import torch

# Add baselines directory to search path so 'models' is discoverable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture
def mock_frontal_cxr():
    """Generates a mock frontal CXR chest projection tensor of shape [1, 1, 256, 256]."""
    torch.manual_seed(42)
    return torch.rand(1, 1, 256, 256)

@pytest.fixture
def test_device():
    """Returns 'cuda' if physical GPU is available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"
