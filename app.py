"""
 Predict 2.5 X-Ray demo.

Usage:
    python app.py --checkpoint-path outputs/your.ckpt
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


def save_video(frames: list[np.ndarray], fps: int = 16) -> str:
    """
    Save frame list to MP4 video file.
    
    Args:
        frames: List of uint8 numpy arrays (H, W, C).
        fps: Frames per second for output video.
    
    Returns:
        Path to saved MP4 file.
    """
    # Create a temporary directory to store the video file
    tmp_dir = Path(tempfile.mkdtemp(prefix="xray_views_"))
    video_path = tmp_dir / "views.mp4"

    # Use imageio to save the frames as an MP4 video.
    imageio.mimsave(
        str(video_path), 
        frames, 
        fps=fps, 
        macro_block_size=1
    )

    return str(video_path)


def frames_to_image(frames: list[np.ndarray], frame_idx: int) -> np.ndarray:
    """
    Extract single frame from list, clamping index to valid range.
    
    Args:
        frames: List of frame arrays.
        frame_idx: Desired frame index (will be clamped).
    
    Returns:
        Single frame as uint8 numpy array.
    """
    frame_idx = int(np.clip(frame_idx, 0, len(frames) - 1))

    return frames[frame_idx]


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
        frames_state = gr.State([])
        ckpt_state = gr.State(checkpoint_path)
        config_state = gr.State(config_path)

        # Define event handler for video generation
        def on_generate(xray_image, cfg, steps, ckpt_path, cfg_path):
            """Generate video and prepare outputs."""
            status_msg, frames = generate_video(
                xray_image=xray_image,
                cfg=cfg,
                steps=steps, 
                ckpt_path=ckpt_path, 
                cfg_path=cfg_path
            )

            first_frame = (
                frames[0] if frames else np.zeros((256, 256, 3), dtype=np.uint8)
            )
            video_path = save_video(frames) if frames else None

            return status_msg, frames, gr.update(value=0), first_frame, video_path

        def on_frame_change(frames, idx):
            """Update frame display when slider changes."""
            if not frames or idx is None:
                return np.zeros((256, 256, 3), dtype=np.uint8)
            
            return frames_to_image(frames, int(idx))

        run_btn.click(
            fn=on_generate,
            inputs=[xray_input, cfg_scale, num_steps, ckpt_state, config_state],
            outputs=[status, frames_state, frame_idx, frame_display, video_output],
        )

        frame_idx.change(
            fn=on_frame_change,
            inputs=[frames_state, frame_idx],
            outputs=frame_display,
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

    # Validate checkpoint path
    ckpt = Path(args.checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    # If config path not provided, assume it's in the same directory as the checkpoint with name "config.json"
    cfg_path = args.config_path
    if cfg_path is None:
        cfg_path = str(ckpt.parent / "config.json")

    # Build the Gradio app
    app = build_app(checkpoint_path=str(ckpt), config_path=cfg_path)

    # Launch the Gradio app
    app.launch(
        server_name=args.server_name, 
        server_port=args.server_port, 
        share=args.share
    )
