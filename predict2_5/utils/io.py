"""File I/O utilities."""

import os

import imageio

import nibabel as nib

import numpy as np

import torch

try:
    from torchvision.io import write_video
    HAVE_TORCHVISION_IO = True
except ImportError:
    HAVE_TORCHVISION_IO = False


def save_vid_as_mp4(
    vid: torch.Tensor,
    out: str,
    fps: int = 30,
) -> None:
    """
    Save video tensor as MP4 file.

    Args:
        vid: Video tensor of shape (1, 1, T, H, W) with values in [0, 1].
        out: Output path for the MP4 file.
        fps: Frames per second for the output video (default: 30).
    """
    # Remove batch and channel dimensions
    vid = vid.squeeze(0).squeeze(0)  # (T, H, W)

    # Scale to [0, 255] and convert to uint8
    vid = (vid * 255).clamp(0, 255).to(torch.uint8)
    vid_np = vid.cpu().numpy()  # (T, H, W)

    # Save as MP4
    os.makedirs(os.path.dirname(out), exist_ok=True)

    import cv2
    T, H, W = vid_np.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out, fourcc, fps, (W, H), isColor=True)
    try:
        for i in range(T):
            frame = vid_np[i]
            # Convert grayscale (H, W) to BGR (H, W, 3) for VideoWriter
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            writer.write(frame_bgr)
    finally:
        writer.release()


def save_img_as_png(
    vid: torch.Tensor,
    out: str,
) -> None:
    """
    Save preprocessed X-ray image as PNG file.

    Args:
        vid: Video tensor of shape (1, 1, T, H, W) with values in [0, 1].
        out: Output path for the PNG file.
    """
    # Extract first frame (0° azimuth view)
    xr_frame = vid[0, 0, 0, :, :]  # (H, W)

    # Scale to [0, 255] and convert to uint8
    xr_np = xr_frame.cpu().numpy()
    xr_np = (xr_np * 255).astype(np.uint8)

    # Save as PNG
    os.makedirs(os.path.dirname(out), exist_ok=True)

    # Use imageio for PNG saving to ensure compatibility and quality
    imageio.imwrite(out, xr_np)


def save_xr_as_png(
    img: torch.Tensor,
    out: str,
) -> None:
    """
    Save preprocessed X-ray image as PNG file.

    Args:
        img: Image tensor of shape (1, 1, H, W) with values in [0, 1].
        out: Output path for the PNG file.
    """
    # Remove channel dimension and convert to numpy
    xr_np = img.squeeze(0).cpu().numpy()  # (H, W)

    # Scale to [0, 255] and convert to uint8
    xr_np = (xr_np * 255).astype(np.uint8)

    # Save as PNG
    os.makedirs(os.path.dirname(out), exist_ok=True)

    # Use imageio for PNG saving to ensure compatibility and quality
    imageio.imwrite(out, xr_np)


def save_vol_as_nifti(
    vol: torch.Tensor,
    out: str,
) -> None:
    """
    Save preprocessed CT volume as NIfTI file.

    Args:
        vol: Volume tensor of shape (1, 1, D, H, W) with values in [0, 1].
        out: Output path for the NIfTI file.
    """
    # Remove batch dimension if present
    if vol.ndim == 5:
        vol = vol.squeeze(0)  # (C, D, H, W)

    # Convert to numpy and remove channel dimension if single channel
    vol_np = vol.cpu().numpy()
    if vol_np.shape[0] == 1:
        vol_np = vol_np[0]  # (D, H, W)

    # Create NIfTI image
    nifti_img = nib.Nifti1Image(vol_np, affine=np.eye(4))

    # Save as NIfTI file
    os.makedirs(os.path.dirname(out), exist_ok=True)

    # Use nibabel for NIfTI saving to ensure compatibility and quality
    nib.save(nifti_img, out)
