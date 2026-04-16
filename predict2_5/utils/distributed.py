"""Distributed training utilities for DDP."""

import os
import logging

import torch
import torch.distributed as dist


def get_local_rank() -> int:
    """
    Get local rank for DDP, corresponding to the GPU index on the current node.

    In single-GPU setup: returns 0.
    In multi-GPU setup: local rank is the GPU index on this machine (0 to num_gpus-1).

    Returns:
        Local rank from LOCAL_RANK environment variable (0 if not set).
    """
    return int(os.environ.get("LOCAL_RANK", 0))


def get_global_rank() -> int:
    """
    Get global rank across all DDP nodes.

    In single-GPU setup: returns 0.
    In multi-GPU setup: unique rank from 0 to world_size-1 (includes all machines).
        
    The system contains many nodes, each with multiple GPUs. 
    Each process is assigned a global rank that is unique across the entire distributed setup. 
    The global rank is typically calculated as:
        global_rank = node_rank * num_gpus_per_node + local_rank
    where node_rank is the index of the current node (0 to num_nodes-1)

    Returns:
        Global rank from RANK environment variable (0 if not set).
    """
    return int(os.environ.get("RANK", 0))


def get_world_size() -> int:
    """
    Get total number of processes across all nodes.

    In single-GPU setup: returns 1.
    In multi-GPU setup: sum of all GPUs across all nodes.
    
    Returns:
        World size from WORLD_SIZE environment variable (1 if not set).
    """
    return int(os.environ.get("WORLD_SIZE", 1))


def is_ddp_mode() -> bool:
    """
    Check if running under DDP.

    Returns:
        True if WORLD_SIZE > 1, indicating DDP mode; False otherwise.
    """
    return get_world_size() > 1


def ddp_barrier() -> None:
    """
    Synchronize all DDP processes at this point.
    
    This will block until all processes reach this barrier, ensuring they are synchronized before proceeding.
    """
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def is_distributed() -> bool:
    """
    Check if distributed training is initialized.

    Returns:
        True if distributed training is available and initialized; False otherwise.
    """
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """
    Get global rank of current process.

    Alias for get_global_rank().

    Returns:
        Global rank if distributed, else 0.
    """
    return dist.get_rank() if is_distributed() else 0


def rank_print(*args, **kwargs) -> None:
    """
    Print only from the master rank (rank 0) to avoid clutter in DDP.

    Args:
        *args: Positional arguments to print.
        **kwargs: Keyword arguments to print.
    """
    if get_local_rank() == 0:
        print(*args, **kwargs)


def broadcast_model_states(
    model: torch.nn.Module, 
    src: int = 0
) -> None:
    """
    Broadcast model parameters and buffers from source rank to all other ranks.

    Ensures all DDP processes have identical model weights after synchronization.

    Args:
        model: Pytorch Module to broadcast (parameters and buffers).
        src: Source rank to broadcast from (default: 0, typically the master rank).
    """
    # Skip if not in distributed mode or only single process
    if not is_distributed() or get_world_size() == 1:
        return

    # Broadcast all parameters and buffers from source rank
    for param in model.parameters():
        dist.broadcast(param.data, src=src)

    for buffer in model.buffers():
        dist.broadcast(buffer, src=src)


def broadcast_model_states_packed(
    model: torch.nn.Module,
    src: int = 0,
    logger: logging.Logger = None,
) -> None:
    """
    Broadcast model parameters and buffers using a packed approach for efficiency.    

    Combine all parameters and buffers into a single flat tensor, broadcast it, then unpack on each rank.
    This can be more efficient than broadcasting each tensor individually, especially for large models.

    Args:
        model: Pytorch Module to broadcast (parameters and buffers).
        src: Source rank to broadcast from (default: 0, typically the master rank).
        logger: Optional logger for error messages.
    """
    # Skip if not in distributed mode or only single process
    if not is_distributed() or get_world_size() == 1:
        return

    try:
        # Collect all information about parameters and buffers to pack
        tensors = []
        shapes = []
        dtypes = []

        for param in model.parameters():
            if param.numel() == 0:
                continue

            tensors.append(param.data)
            shapes.append(param.shape)
            dtypes.append(param.dtype)

        for buffer in model.buffers():
            if buffer.numel() == 0:
                continue

            tensors.append(buffer)
            shapes.append(buffer.shape)
            dtypes.append(buffer.dtype)

        # Skip if no tensors to broadcast
        if not tensors:
            return

        # Verify all tensors on CUDA device for efficient broadcast
        device = tensors[0].device
        if device.type != "cuda":
            if logger:
                logger.warning(f"Tensors not on CUDA device, falling back to naive broadcast")
            
            return

        # Pack all tensors into a single flat tensor for broadcast
        flat_tensors = []

        for t in tensors:
            # Ensure all tensors are on the same device for packing
            if t.device != device:
                t = t.to(device=device)

            flat_tensors.append(t.flatten().to(dtype=torch.float32))

        # Concatenate all into single tensor
        packed = torch.cat(flat_tensors)

        # Broadcast the packed tensor from source rank to all other ranks
        dist.broadcast(packed, src=src, async_op=False)

        # Unpack tensors back to original shapes and dtypes
        offset = 0

        for tensor, shape, dtype in zip(tensors, shapes, dtypes):
            # Get the number of elements in the original tensor to know how much to unpack
            numel = tensor.numel()

            # Extract the corresponding slice from the packed tensor, reshape it, and convert back to original dtype
            unpacked = packed[offset : offset + numel].view(shape).to(dtype=dtype)

            # In-place copy the unpacked data back to the original tensor to update its values
            tensor.data.copy_(unpacked)

            offset += numel

    except Exception as e:
        if logger:
            logger.error(f"Failed to broadcast model states using packed approach: {e}")

        # Fallback to slower element-wise broadcast if packed broadcast fails
        broadcast_model_states(model, src=src)


def sync_ema_ddp(
    net_ema: torch.nn.Module,
    sync_every_n_steps: int = 1,
    current_step: int = 0,
    logger: logging.Logger = None,
) -> None:
    """
    Synchronize EMA model across all DDP ranks.

    Ensure that the EMA model (net_ema) has the same parameters and buffers across all ranks at regular intervals during training. 
    This is important because the EMA model is often used for evaluation and inference, and we want it to be consistent across all processes.

    Args:
        net_ema: EMA model to synchronize.
        sync_every_n_steps: Sync frequency in training steps (default: 1 = every step).
        current_step: Current training step number (default: 0).
        logger: Optional logger for error messages.
    """
    # Skip if not in distributed mode
    if not is_distributed() or get_world_size() == 1:
        return

    # Skip this step if not time to sync
    if current_step % sync_every_n_steps != 0:
        return

    try:
        # Synchronize with barriers before and after broadcast
        dist.barrier()

        # Use the packed broadcast for efficiency, especially if the EMA model is large
        broadcast_model_states_packed(net_ema, src=0, logger=logger)

        # Ensure all ranks have completed the broadcast before proceeding
        dist.barrier()

    except Exception as e:
        if logger:
            logger.error(f"Syncing EMA model across DDP ranks failed: {e}")
