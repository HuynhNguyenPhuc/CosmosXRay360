import torch

def searchsorted(a, v, out=None, side='left'):
    """
    Standard drop-in wrapper substituting legacy compiled torchsearchsorted 
    with PyTorch's high-performance native searchsorted.
    """
    return torch.searchsorted(a, v, out=out, side=side)
