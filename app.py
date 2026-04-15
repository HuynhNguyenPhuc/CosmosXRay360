"""Cosmos Predict 2.5 X-Ray demo with fixed 93-frame video output.

Usage:
    python app.py --checkpoint-path outputs/your.ckpt
"""

from __future__ import annotations

# Must be set before ANY cosmos_predict2 import so checkpoint_db registers
# experimental UUIDs (tokenizer + base model) at module load time.
import os

os.environ.setdefault("COSMOS_EXPERIMENTAL_CHECKPOINTS", "1")

import argparse
import pickle
import tempfile
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from predict2_5.constants import CR1_EMBEDDING_DIM, CR1_MAX_LENGTH, NUM_FRAMES, PROMPTS
from predict2_5.utils import get_logger

_MODEL_CACHE = {"model": None, "device": None, "checkpoint_path": None}
_EMBEDDING_CACHE: dict[str, torch.Tensor] = {}
log = get_logger(__name__)


def _prompt_to_filename(prompt: str) -> str:
    """Convert prompt to embedding filename (matches get_cosmos_reason_embeddings.py)."""
    return (
        (prompt.strip().replace(" ", "_").replace("'", "").replace(",", "")[:100])
        or "null"
    )


def _load_text_embedding(embeddings_dir: str, prompt: str) -> torch.Tensor:
    """Load pre-computed text embedding from disk, fallback to zeros.

    Returns tensor of shape (1, seq_len, CR1_EMBEDDING_DIM).
    """
    cache_key = f"{embeddings_dir}::{prompt}"
    if cache_key in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[cache_key]

    emb_path = Path(embeddings_dir) / f"{_prompt_to_filename(prompt)}.pkl"

    embedding: torch.Tensor | None = None
    if emb_path.exists():
        try:
            with open(emb_path, "rb") as f:
                data = pickle.load(f)  # list[Tensor(seq_len, embed_dim)]
            raw = data[0] if isinstance(data, list) else data
            embedding = torch.as_tensor(raw, dtype=torch.float32).unsqueeze(
                0
            )  # (1, L, D)
        except Exception as e:
            log.warning("[app] could not load embedding %s: %s", emb_path, e)

    if embedding is None:
        log.warning("[app] embedding not found, using zeros: %s", emb_path)
        embedding = torch.zeros(
            (1, CR1_MAX_LENGTH, CR1_EMBEDDING_DIM), dtype=torch.float32
        )

    _EMBEDDING_CACHE[cache_key] = embedding
    return embedding


def _ensure_checkpoint_db_compat() -> None:
    """Install compatibility shim for legacy checkpoint resolver API.

    Some local code imports get_checkpoint_by_uuid from checkpoint_db, while
    newer checkpoint_db exposes get_checkpoint_uri/get_checkpoint_path.
    Also ensures checkpoints are registered (cosmos_predict2.config does this
    normally, but predict2_5.module doesn't import it).
    """
    from cosmos_predict2._src.imaginaire.utils import checkpoint_db

    # Populate the _CHECKPOINTS registry (normally done by cosmos_predict2.config).
    from cosmos_oss.checkpoints_predict2 import register_checkpoints

    register_checkpoints()

    if hasattr(checkpoint_db, "get_checkpoint_by_uuid"):
        return

    def _get_checkpoint_by_uuid(uuid: str) -> SimpleNamespace:
        resolved_uri = checkpoint_db.get_checkpoint_uri(uuid)
        resolved_path = checkpoint_db.get_checkpoint_path(resolved_uri)
        return SimpleNamespace(path=resolved_path)

    checkpoint_db.get_checkpoint_by_uuid = _get_checkpoint_by_uuid


def _to_pil(image_np: np.ndarray) -> Image.Image:
    """Convert numpy image to RGB PIL image."""
    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3, axis=-1)
    if image_np.shape[-1] == 4:
        image_np = image_np[..., :3]
    if image_np.dtype != np.uint8:
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    return Image.fromarray(image_np, mode="RGB")


def _get_or_load_model(checkpoint_path: str, config_path: str, device: str):
    """Return cached model instance for the same checkpoint/device."""
    if (
        _MODEL_CACHE["model"] is not None
        and _MODEL_CACHE["checkpoint_path"] == checkpoint_path
        and _MODEL_CACHE["device"] == device
    ):
        return _MODEL_CACHE["model"]

    _ensure_checkpoint_db_compat()
    from predict2_5.inferencer import Inferencer

    model = Inferencer(
        checkpoint_path=checkpoint_path, config_path=config_path, device=device
    )

    _MODEL_CACHE["model"] = model
    _MODEL_CACHE["checkpoint_path"] = checkpoint_path
    _MODEL_CACHE["device"] = device
    return model


def _save_video(frames: list[np.ndarray], fps: int = 16) -> str:
    """Save frames to MP4 and return path."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="xray_views_"))
    video_path = tmp_dir / "views.mp4"
    imageio.mimsave(str(video_path), frames, fps=fps, macro_block_size=1)
    return str(video_path)


def _frames_to_image(frames: list[np.ndarray], frame_idx: int) -> np.ndarray:
    """Extract a single frame from the list, clamp to bounds."""
    frame_idx = int(np.clip(frame_idx, 0, len(frames) - 1))
    return frames[frame_idx]


def _predict_video_from_model(
    xray_image: np.ndarray,
    checkpoint_path: str,
    config_path: str,
    cfg_scale: float,
    num_steps: int,
) -> list[np.ndarray]:
    """Generate a fixed NUM_FRAMES video with model.generate."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _get_or_load_model(checkpoint_path, config_path, device)

    # Use the new predict interface
    prompt = PROMPTS[0] if PROMPTS else ""
    frames = model(
        image=xray_image,
        prompt=prompt,
        cfg_scale=float(cfg_scale),
        num_steps=int(num_steps),
    )
    return frames


def generate_views_ui(
    xray_image: np.ndarray | None,
    cfg_scale: float,
    num_steps: int,
    checkpoint_path: str,
    config_path: str,
) -> tuple[str, list[np.ndarray]]:
    """Generate one 93-frame video from one X-Ray image, return frames list."""
    if xray_image is None:
        return "Please upload an X-Ray image.", []
    try:
        frames = _predict_video_from_model(
            xray_image=xray_image,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            cfg_scale=float(np.clip(cfg_scale, 0.0, 6.0)),
            num_steps=int(np.clip(num_steps, 5, 80)),
        )
        return f"Generated {len(frames)} views (fixed to {NUM_FRAMES}).", frames
    except Exception as e:
        return f"Model inference failed: {e}", []


def build_app(checkpoint_path: str, config_path: str) -> gr.Blocks:
    """Build minimal UI: image input + CFG/steps controls + video + frame slider."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with gr.Blocks(
        title="Cosmos Predict 2.5 Post-training for Medical Image Dataset"
    ) as demo:
        gr.Markdown("## Cosmos Predict 2.5 Post-training for Medical Image Dataset")

        # Eager load model on startup
        try:
            _get_or_load_model(checkpoint_path, config_path, device)
            gr.Markdown(f"✓ Model loaded on **{device.upper()}**")
        except Exception as e:
            gr.Markdown(f"⚠ Failed to load model: {e}")
            return demo

        with gr.Row():
            xray_input = gr.Image(
                label="Upload X-Ray Image",
                type="numpy",
                image_mode="RGB",
                sources=["upload", "clipboard"],
            )
            with gr.Column():
                cfg_scale = gr.Slider(
                    label="CFG", minimum=0.0, maximum=6.0, step=0.1, value=1.5
                )
                num_steps = gr.Slider(
                    label="Num Steps", minimum=5, maximum=80, step=1, value=35
                )
                run_btn = gr.Button("Generate Video", variant="primary")

        status = gr.Textbox(label="Status", interactive=False)

        with gr.Row():
            with gr.Column():
                # Native browser video player – smooth scrubbing with zero server latency
                video_output = gr.Video(
                    label=f"Output Video ({NUM_FRAMES} views, scrub freely)"
                )
            with gr.Column():
                # Frame slider for precise per-frame inspection
                frame_idx = gr.Slider(
                    label="Frame Index",
                    minimum=0,
                    maximum=NUM_FRAMES - 1,
                    step=1,
                    value=0,
                    interactive=True,
                )
                frame_display = gr.Image(
                    label=f"Frame View (0–{NUM_FRAMES - 1})", type="numpy"
                )

        frames_state = gr.State([])
        ckpt_state = gr.State(checkpoint_path)
        config_state = gr.State(config_path)

        def on_generate(xray_image, cfg, steps, ckpt, cfg_path):
            status_msg, frames = generate_views_ui(
                xray_image, cfg, steps, ckpt, cfg_path
            )
            first_frame = (
                frames[0] if frames else np.zeros((256, 256, 3), dtype=np.uint8)
            )
            video_path = _save_video(frames) if frames else None
            # Return gr.update(value=0) to reset slider and prevent None-payload error
            return status_msg, frames, gr.update(value=0), first_frame, video_path

        def on_frame_change(frames, idx):
            if not frames or idx is None:
                return np.zeros((256, 256, 3), dtype=np.uint8)
            return _frames_to_image(frames, int(idx))

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
    parser = argparse.ArgumentParser(description="Cosmos Predict 2.5 X-Ray demo")
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
    args = parser.parse_args()

    ckpt = Path(args.checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    cfg_path = args.config_path
    if cfg_path is None:
        cfg_path = str(ckpt.parent / "config.json")

    app = build_app(checkpoint_path=str(ckpt), config_path=cfg_path)
    app.launch(
        server_name=args.server_name, server_port=args.server_port, share=args.share
    )
