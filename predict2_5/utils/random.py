"""Random utilities."""

import numpy as np

import torch


def arch_invariant_rand(
    shape: tuple,
    dtype: torch.dtype,
    device: str | torch.device,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Generate random tensor with architecture-invariant properties.

    Args:
        shape: Desired shape of the output tensor.
        dtype: Desired data type of the output tensor.
        device: Desired device for the output tensor.
        seed: Optional random seed for reproducibility (default: None).

    Returns:
        Random tensor of specified shape, dtype, and device.
    """
    # Mersenne Twister pseudo-random number generator
    
    rng = np.random.RandomState(seed)

    # Draw samplses from a standard normal distribution
    random_array = rng.standard_normal(shape).astype(np.float32)

    return torch.from_numpy(random_array).to(dtype=dtype, device=device)
