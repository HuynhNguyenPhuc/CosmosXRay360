"""
Predict 2.5 X-Ray demo.

Usage:
    python app.py --checkpoint-path outputs/your.ckpt --share
"""

from __future__ import annotations

# Must be set before ANY cosmos_predict2 import so checkpoint_db registers experimental UUIDs (tokenizer + base model) at module load time.
import os
os.environ.setdefault("COSMOS_EXPERIMENTAL_CHECKPOINTS", "1")

import argparse
import tempfile
from pathlib import Path

import gradio as gr
import imageio.v2 as imageio

import numpy as np

import torch

from predict2_5.constants import NUM_FRAMES
from predict2_5.utils import get_logger


MODEL_CACHE = {
    "model": None, 
    "device": None, 
    "checkpoint_path": None
}

# --- Logger --- #
logger = get_logger(__name__)


def compute_stretch_range(frames: list[np.ndarray]) -> list[float] | None:
    """
    Compute shared [low, high] percentile range for clip-level contrast stretching across all frames.

    Args:
        frames: List of frames as uint8 numpy arrays.

    Returns:
        List of [low, high] values for contrast stretching, or None if computation fails.
    """
    if not frames:
        return None

    # Sample pixel values from all frames to estimate histogram
    samples = []
    for frame in frames:
        # Convert to grayscale and subsample
        if frame.ndim == 2:
            gray = np.asarray(frame, dtype=np.float32)
        elif frame.shape[-1] == 1:
            gray = np.asarray(frame[..., 0], dtype=np.float32)
        else:
            rgb = np.asarray(frame[..., :3], dtype=np.float32)
            gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

        samples.append(gray.reshape(-1)[::16])

    # Compute percentiles
    sampled_values = np.concatenate(samples, axis=0)
    low = float(np.percentile(sampled_values, 1.0))
    high = float(np.percentile(sampled_values, 99.0))

    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return None

    return [low, high]


def apply_display_postprocess(
    frame: np.ndarray,
    display_mode: str = "normal",
    stretch_range: list[float] | None = None,
) -> np.ndarray:
    """
    Unified display postprocessing: convert to grayscale, apply contrast stretch if requested, 
    convert to uint8 RGB for display.

    Args:
        frame: Input frame as a numpy array (H, W) or (H, W, C).
        display_mode: Display preset mode ("normal" or "contrast_stretch").
        stretch_range: Optional [low, high] clip-level stretch parameters (display-only).

    Returns:
        Postprocessed frame as uint8 RGB numpy array (H, W, 3).
    """
    # Convert to grayscale (luminance from RGB or pass through if already grayscale)
    if frame.ndim == 2:
        gray = np.asarray(frame, dtype=np.float32)
    elif frame.shape[-1] == 1:
        gray = np.asarray(frame[..., 0], dtype=np.float32)
    else:
        rgb = np.asarray(frame[..., :3], dtype=np.float32)
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

    # Apply clip-level contrast stretch if requested (display-only, does not modify model output)
    if display_mode == "contrast_stretch" and stretch_range is not None:
        low, high = stretch_range
        if high > low:
            gray = (gray - low) * (255.0 / (high - low))

    # Convert to uint8 RGB
    gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8, copy=False)
    return np.repeat(gray_uint8[..., None], 3, axis=-1)


def get_or_load_model(
    checkpoint_path: str, 
    config_path: str, 
    device: str
):
    """
    Get or load the Inferencer model, with caching to avoid redundant loads.

    Args:
        checkpoint_path: Path to the model checkpoint.
        config_path: Path to the model config file.
        device: Device to load the model on ("cuda" or "cpu").

    Returns:
        Loaded Inferencer model.
    """
    # If the model is already loaded with the same checkpoint and device, return it from cache
    if (
        MODEL_CACHE["model"] is not None
        and MODEL_CACHE["checkpoint_path"] == checkpoint_path
        and MODEL_CACHE["device"] == device
    ):
        return MODEL_CACHE["model"]

    # If not cached, load the model and store it in the cache
    from predict2_5.inferencer import Inferencer

    model = Inferencer(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=device,
    )

    MODEL_CACHE["model"] = model
    MODEL_CACHE["checkpoint_path"] = checkpoint_path
    MODEL_CACHE["device"] = device

    return model


def save_video(
    frames: list[np.ndarray],
    fps: int = 16,
    display_mode: str = "normal",
    stretch_range: list[float] | None = None,
) -> str:
    """
    Save frame list to MP4 video file.
    
    Args:
        frames: List of uint8 numpy arrays (H, W, C).
        fps: Frames per second for output video.
        display_mode: Preview mode, either "normal" or "contrast_stretch".
        stretch_range: Shared [low, high] range for clip-level contrast stretching.
    
    Returns:
        Path to saved MP4 file.
    """
    # Create a temporary directory to store the video file
    tmp_dir = Path(tempfile.mkdtemp(prefix="xray_views_"))
    video_path = tmp_dir / "views.mp4"
    display_frames = [
        apply_display_postprocess(
            frame,
            display_mode=display_mode,
            stretch_range=stretch_range,
        )
        for frame in frames
    ]

    # Use imageio to save the frames as an MP4 video.
    imageio.mimsave(
        str(video_path), 
        display_frames,
        fps=fps, 
        macro_block_size=1
    )

    return str(video_path)


def frames_to_image(
    frames: list[np.ndarray],
    frame_idx: int,
    display_mode: str = "normal",
    stretch_range: list[float] | None = None,
) -> np.ndarray:
    """
    Extract single frame from list, clamping index to valid range.
    
    Args:
        frames: List of frame arrays.
        frame_idx: Desired frame index (will be clamped).
        display_mode: Preview mode, either "normal" or "contrast_stretch".
        stretch_range: Shared [low, high] range for clip-level contrast stretching.
    
    Returns:
        Single frame as uint8 numpy array.
    """
    frame_idx = int(np.clip(frame_idx, 0, len(frames) - 1))

    return apply_display_postprocess(
        frames[frame_idx],
        display_mode=display_mode,
        stretch_range=stretch_range,
    )


def generate_video(
    xray_image: np.ndarray | None,
    cfg_scale: float,
    num_steps: int,
    checkpoint_path: str,
    config_path: str,
) -> tuple[str, list[np.ndarray]]:
    """
    Generate video from an uploaded X-Ray image.
    
    Args:
        xray_image: Uploaded X-Ray image as a numpy array (H, W, C).
        cfg_scale: Classifier-free guidance scale.
        num_steps: Number of diffusion steps.
        checkpoint_path: Path to model checkpoint.
        config_path: Path to model config.
    
    Returns:
        Tuple of (status_message, list_of_frames).
    """
    if xray_image is None:
        return "Please upload an X-Ray image.", []
    
    try:
        # Get the current device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Get or load the model (with caching)
        model = get_or_load_model(checkpoint_path, config_path, device)

        # Run inference to get the video
        frames = model.predict(
            image=xray_image,
            cfg_scale=float(np.clip(cfg_scale, 0.0, 6.0)),
            num_steps=int(np.clip(num_steps, 5, 80)),
        )

        return f"✓ Generated {len(frames)} views.", frames
    except Exception as e:
        logger.error("Inference failed: %s", e)
        return f"❌ Model inference failed: {e}", []


def build_app(checkpoint_path: str, config_path: str) -> gr.Blocks:
    """
    Build the Gradio app interface.

    Args:
        checkpoint_path: Path to model checkpoint.
        config_path: Path to model config.

    Returns:
        Gradio Blocks object representing the app.
    """
    # Get the current device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with gr.Blocks(title="Cosmos Predict 2.5 X-Ray Video Generation") as demo:
        gr.Markdown("## Cosmos Predict 2.5 X-Ray Video Generation")

        # Load model at startup and show status
        try:
            get_or_load_model(checkpoint_path, config_path, device)
            gr.Markdown(f"✓ Model loaded on **{device.upper()}**")
        except Exception as e:
            gr.Markdown(f"⚠ Failed to load model: {e}")
            return demo

        # Input section: image upload + inference controls
        with gr.Row():
            xray_input = gr.Image(
                label="Upload X-Ray Image",
                type="numpy",
                image_mode="RGB",
                sources=["upload", "clipboard"],
            )

            with gr.Column():
                cfg_scale = gr.Slider(
                    label="CFG Scale",
                    minimum=0.0,
                    maximum=6.0,
                    step=0.1,
                    value=1.5,
                    info="Higher = stronger conditioning",
                )
                num_steps = gr.Slider(
                    label="Inference Steps",
                    minimum=5,
                    maximum=80,
                    step=1,
                    value=35,
                    info="More steps = better quality but slower",
                )
                display_mode = gr.Radio(
                    label="Display Mode",
                    choices=[("Standard", "normal"), ("Enhanced Contrast", "contrast_stretch")],
                    value="normal",
                    info="Preview only. Does not change raw model output.",
                )

                run_btn = gr.Button("Generate Video", variant="primary", size="lg")

        status = gr.Textbox(label="Status", interactive=False, lines=1)

        # Output section: video player + frame inspector
        with gr.Row():
            with gr.Column(scale=2):
                # Native HTML5 video player with frame-accurate scrubbing
                video_output = gr.Video(
                    label=f"Generated Video ({NUM_FRAMES} frames @ 16fps)"
                )
            with gr.Column(scale=1):
                # Frame-by-frame viewer for detailed inspection
                frame_idx = gr.Slider(
                    label="Frame Inspector",
                    minimum=0,
                    maximum=NUM_FRAMES - 1,
                    step=1,
                    value=0,
                    interactive=True,
                )
                frame_display = gr.Image(
                    label="Frame Detail", type="numpy", interactive=False
                )

        # State for storing results between interactions
        # State variables for frame data and display settings
        frames_state = gr.State([])
        stretch_range_state = gr.State(None)
        ckpt_state = gr.State(checkpoint_path)
        config_state = gr.State(config_path)

        # Define event handler for video generation
        def on_generate(
            xray_image,
            cfg_scale,
            num_steps,
            display_mode,
            checkpoint_path,
            config_path,
        ):
            """Generate video and prepare outputs."""
            status_msg, frames = generate_video(
                xray_image=xray_image,
                cfg_scale=cfg_scale,
                num_steps=num_steps, 
                checkpoint_path=checkpoint_path, 
                config_path=config_path
            )

            stretch_range = compute_stretch_range(frames)

            first_frame = (
                apply_display_postprocess(
                    frames[0],
                    display_mode=display_mode,
                    stretch_range=stretch_range,
                )
                if frames
                else np.zeros((256, 256, 3), dtype=np.uint8)
            )
            video_path = (
                save_video(
                    frames,
                    display_mode=display_mode,
                    stretch_range=stretch_range,
                )
                if frames
                else None
            )

            return (
                status_msg,
                frames,
                stretch_range,
                gr.update(value=0),
                first_frame,
                video_path,
            )

        def on_frame_change(frames, idx, display_mode, stretch_range):
            """Update frame display when slider changes."""
            if not frames or idx is None:
                return np.zeros((256, 256, 3), dtype=np.uint8)
            
            return frames_to_image(
                frames,
                int(idx),
                display_mode=display_mode,
                stretch_range=stretch_range,
            )

        def on_display_mode_change(frames, idx, display_mode, stretch_range):
            """Re-render preview assets when the display mode changes."""
            if not frames:
                return np.zeros((256, 256, 3), dtype=np.uint8), None

            frame_idx = 0 if idx is None else int(idx)
            frame_image = frames_to_image(
                frames,
                frame_idx,
                display_mode=display_mode,
                stretch_range=stretch_range,
            )
            video_path = save_video(
                frames,
                display_mode=display_mode,
                stretch_range=stretch_range,
            )

            return frame_image, video_path

        run_btn.click(
            fn=on_generate,
            inputs=[
                xray_input,
                cfg_scale,
                num_steps,
                display_mode,
                ckpt_state,
                config_state,
            ],
            outputs=[
                status,
                frames_state,
                stretch_range_state,
                frame_idx,
                frame_display,
                video_output,
            ],
        )

        frame_idx.change(
            fn=on_frame_change,
            inputs=[frames_state, frame_idx, display_mode, stretch_range_state],
            outputs=frame_display,
        )

        display_mode.change(
            fn=on_display_mode_change,
            inputs=[frames_state, frame_idx, display_mode, stretch_range_state],
            outputs=[frame_display, video_output],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cosmos Predict 2.5 X-Ray Demo")
    
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to trained Lightning checkpoint",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to config.json. Defaults to same directory as checkpoint.",
    )
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")

    # Parse command-line arguments
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    cfg_path = args.config_path

    if checkpoint_path.startswith("hf://"):
        # If config path not provided, assume it's in the same repository under "config.json"
        if cfg_path is None:
            if "@" in checkpoint_path:
                uri_part, revision_part = checkpoint_path.rsplit("@", 1)
                suffix = f"@{revision_part}"
            else:
                uri_part, suffix = checkpoint_path, ""
            
            if "/" in uri_part:
                prefix, _ = uri_part.rsplit("/", 1)
                cfg_path = f"{prefix}/config.json{suffix}"
            else:
                raise ValueError(f"Invalid Hugging Face URI: {checkpoint_path}")
        ckpt_str = checkpoint_path
    else:
        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            from predict2_5.utils import is_uuid_format
            if not is_uuid_format(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found locally and is not a valid Hugging Face URI or Cosmos UUID: {checkpoint_path}")
        
        if cfg_path is None:
            if ckpt.exists():
                cfg_path = str(ckpt.parent / "config.json")
            else:
                cfg_path = None
        ckpt_str = str(ckpt) if ckpt.exists() else checkpoint_path

    # Build the Gradio app
    app = build_app(checkpoint_path=ckpt_str, config_path=cfg_path)

    # Launch the Gradio app
    app.launch(
        server_name=args.server_name, 
        server_port=args.server_port, 
        share=args.share
    )
