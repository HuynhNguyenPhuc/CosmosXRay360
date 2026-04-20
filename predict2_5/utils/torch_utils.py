"""PyTorch utilities."""

import torch
from predict2_5.utils.logging import get_logger


# --- Logger --- #
logger = get_logger(__name__)


def safe_torch_load(path: str, map_location: str = "cpu") -> dict:
    """
    Load checkpoint safely.

    Args:
        path: Path to checkpoint file.
        map_location: Device mapping for torch.load.

    Returns:
        Loaded checkpoint object.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    
    except Exception as exc:
        if "weights_only" in str(exc) or "Unsupported global" in str(exc):
            logger.warning("weights_only=True failed; retry with weights_only=False")
            return torch.load(path, map_location=map_location, weights_only=False)
        
        raise


def fix_rope_buffers(module: torch.nn.Module) -> None:
    """
    Fix RoPE buffers for VideoRopePosition3DEmb layers after meta-device initialization.

    The problem is that when using meta device initialization (e.g., with torch.nn.init or certain distributed training setups), the buffers in the VideoRopePosition3DEmb layer may not be properly initialized or may be on the wrong device. 
    This function iterates through the module's children, identifies any VideoRopePosition3DEmb layers, and reinitializes their RoPE buffers (seq, dim_spatial_range, dim_temporal_range) on the correct device with the correct values.

    Args:
        module: PyTorch module to fix.
    """
    # Recursively fix RoPE buffers in all child modules
    for _, child in module.named_children():
        # Loop through all child modules to find VideoRopePosition3DEmb
        if child.__class__.__name__ == "VideoRopePosition3DEmb":
            # Get the target device for the buffers
            target_device = child._buffers["dim_spatial_range"].device

            # Determine the maximum sequence length needed based on the dimensions of the input
            max_len = max(child.max_h, child.max_w, child.max_t)

            # Initialize the RoPE buffers on the target device with the correct values
            child.seq = torch.arange(max_len, device=target_device, dtype=torch.float32)

            # Calculate the spatial and temporal ranges based on the dimensions of the input
            dim_h = child._dim_h
            dim_t = child._dim_t

            child.dim_spatial_range = (
                torch.arange(0, dim_h, 2, device=target_device, dtype=torch.float32)[
                    : (dim_h // 2)
                ]
                / dim_h
            )
            child.dim_temporal_range = (
                torch.arange(0, dim_t, 2, device=target_device, dtype=torch.float32)[
                    : (dim_t // 2)
                ]
                / dim_t
            )
        else:
            fix_rope_buffers(child)


def move_tokenizer_to_device(tokenizer: object, target_device: str) -> None:
    """
    Move tokenizer buffers to the target device.
    
    This function checks if the tokenizer has a 'model' attribute with a nested 'model' attribute (which is common in some tokenizers that wrap a model). If it does, it moves the model to the target device.
    
    Args:
        tokenizer: Pytorch Tokenizer object that may contain a model with buffers to move.
        target_device: The target device (e.g., 'cuda:0').
    """
    if not hasattr(tokenizer, "model") or not hasattr(tokenizer.model, "model"):
        return

    # Move the underlying model to the target device
    vae_module = tokenizer.model.model
    vae_module.to(target_device)

    # Change the device of the tokenizer's model to ensure consistency
    tokenizer.model.device = target_device

    # Move normalization buffers
    for attr in ["mean", "std", "img_mean", "img_std", "video_mean", "video_std"]:
        if hasattr(tokenizer.model, attr):
            # Get the buffer value
            val = getattr(tokenizer.model, attr)

            # Move the buffer to the target device if it's a tensor
            if isinstance(val, torch.Tensor):
                setattr(tokenizer.model, attr, val.to(target_device))

    # Move scaling buffers (can be list of tensors)
    if hasattr(tokenizer.model, "scale") and isinstance(tokenizer.model.scale, list):
        tokenizer.model.scale = [
            s.to(target_device) if isinstance(s, torch.Tensor) else s
            for s in tokenizer.model.scale
        ]
