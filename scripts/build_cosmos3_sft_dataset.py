#!/usr/bin/env python3
"""
Build a Cosmos 3 SFT Dataset from Pre-Rendered DRR Rotations
==============================================================
Converts ``datasets/pre_rendered/train/<patient>/views.pt`` (93x256x256
uint8 DiffDRR rotations, produced by ``datasets/pre_render_diffdrr.py``)
into the video+caption directory layout that cosmos-framework's
``captions_to_sft_jsonl`` converter expects, then invokes that converter
in-process to produce ``train/video_dataset_file.jsonl`` -- exactly the file
``cosmos_framework.scripts.train``'s ``vision_sft_edge`` recipe reads via
``${oc.env:DATASET_PATH}/train/video_dataset_file.jsonl``.

See ``docs/cosmos-predict3/PLAN.md`` P2.

Usage:
    uv run python scripts/build_cosmos3_sft_dataset.py \
        --pre-rendered-dir datasets/pre_rendered \
        --output-dir datasets/cosmos3_sft \
        --fps 24

    # Fast local smoke test on a handful of patients:
    uv run python scripts/build_cosmos3_sft_dataset.py --limit 3

OOD Discipline Note:
    Reads ONLY ``datasets/pre_rendered/train/`` (TCIA + MELA2022). Never
    reads or encodes ``datasets/pre_rendered/test/`` (NSCLC) -- a leakage
    guard asserts none of the emitted patient ids collide with the
    test-split directory names before any video is written
    (.claude/rules/baseline-experiments.md).
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
COSMOS_FRAMEWORK_DIR = ROOT_DIR / "cosmos-framework"
if str(COSMOS_FRAMEWORK_DIR) not in sys.path:
    sys.path.insert(0, str(COSMOS_FRAMEWORK_DIR))

from predict3.constants import NUM_FRAMES, PROMPTS, VOL_SIZE  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("build_cosmos3_sft_dataset")


def load_views(patient_dir: Path) -> np.ndarray:
    """Load a patient's rotation as uint8 ``[NUM_FRAMES, H, W]`` grayscale.

    Prefers the single-file ``views.pt`` binary (matches the loading
    priority in ``predict2_5/datamodule.py``'s ``PreRendered360Dataset``);
    falls back to ``views/*.png``.
    """
    views_pt = patient_dir / "views.pt"
    if views_pt.exists():
        tensor = torch.load(views_pt, map_location="cpu")
        arr = tensor.numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
    else:
        from PIL import Image

        png_paths = sorted((patient_dir / "views").glob("*.png"))
        if not png_paths:
            raise FileNotFoundError(f"No views.pt or views/*.png under {patient_dir}")
        arr = np.stack([np.array(Image.open(p).convert("L")) for p in png_paths], axis=0)

    if arr.dtype != np.uint8:
        scale = 255.0 if float(arr.max()) <= 1.5 else 1.0
        arr = np.clip(arr * scale + 0.5, 0, 255).astype(np.uint8)  # round, not truncate
    if arr.shape[0] != NUM_FRAMES:
        raise ValueError(f"{patient_dir.name}: expected {NUM_FRAMES} frames, got {arr.shape[0]}")
    return arr


def write_lossless_mp4(frames_gray: np.ndarray, out_path: Path, fps: float) -> None:
    """Write a grayscale ``[T, H, W]`` uint8 stack as an MP4, bit-exact lossless.

    ``-qp 0`` (lossless x264) on ``yuv420p`` preserves every luma sample
    exactly, and encoding as ``gray`` directly with constant chroma loses
    nothing for genuinely single-channel content converted to yuv420p 4:2:0
    -- BUT only if the color range is explicit on both ends. ffmpeg's
    default swscale color conversion assumes "limited" (16-235) range
    unless told otherwise, silently remapping full-range (0-255) input
    values on the way in and out and corrupting up to ~30 grayscale levels
    per pixel even at ``-qp 0`` (verified empirically: PSNR ~32dB without
    these flags vs. bit-exact with them). ``-color_range pc`` plus an
    explicit ``scale=in_range=full:out_range=full`` filter on the encode
    side forces a true identity mapping; verified bit-exact (``max abs
    diff == 0``) against both plain ffmpeg PNG extraction and decord (the
    likely SFT-loader video backend) -- see
    ``predict3/tests/test_sft_dataset_build.py``.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames_gray.shape[1], frames_gray.shape[2]
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-color_range",
        "pc",
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-qp",
        "0",
        "-pix_fmt",
        "yuv420p",
        "-color_range",
        "pc",
        "-vf",
        "scale=in_range=full:out_range=full",
        str(out_path),
    ]
    proc = subprocess.run(cmd, input=np.ascontiguousarray(frames_gray).tobytes(), capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {out_path}: {proc.stderr.decode(errors='replace')}")


def build_caption_json(fps: float) -> dict:
    """One frozen structured-caption template, filled with fixed geometry only.

    Never regenerated per-patient or per-run (docs/cosmos-predict3/PLAN.md R5)
    -- every patient gets the identical template, since the synthesized
    content (a full 360-degree DRR orbit) is the same task regardless of
    patient anatomy. Matches the ``StructuredCaption`` schema in
    ``cosmos_framework/inference/structured_caption.py``.
    """
    duration_s = NUM_FRAMES / fps
    return {
        "subjects": [
            {
                "description": "A chest anatomical structure captured as a digitally reconstructed "
                "radiograph (DRR) Beer-Lambert attenuation projection",
                "action": "rotates continuously through a full 360-degree azimuthal orbit",
            }
        ],
        "background_setting": "Uniform black background; single-channel X-ray-style attenuation projection",
        "cinematography": {
            "camera_motion": "orbiting azimuthal rotation around a fixed vertical axis",
            "framing": "centered, full-frame anatomical projection",
            "camera_angle": "isocentric, sweeping from 0 to 360 degrees azimuth",
        },
        "actions": [
            {
                "time": f"0:00-0:{duration_s:04.2f}",
                "description": "the projection rotates smoothly through the full azimuthal range, "
                "revealing anterior, lateral, and posterior views in sequence",
            }
        ],
        "temporal_caption": PROMPTS[0],
        "resolution": {"H": VOL_SIZE, "W": VOL_SIZE},
        "aspect_ratio": "1,1",
        "duration": f"{duration_s:.1f}s",
        "fps": round(fps),
    }


def build_dataset(
    pre_rendered_dir: Path,
    output_dir: Path,
    fps: float = 24.0,
    limit: Optional[int] = None,
) -> Path:
    """Write ``<output_dir>/train/{videos,captions}/`` from ``pre_rendered_dir/train``.

    Returns:
        The ``<output_dir>/train`` directory (ready for
        ``run_captions_to_sft_jsonl``).
    """
    train_dir = pre_rendered_dir / "train"
    test_dir = pre_rendered_dir / "test"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"{train_dir} not found")

    test_patient_ids = {p.name for p in test_dir.iterdir() if p.is_dir()} if test_dir.is_dir() else set()

    patient_dirs = sorted(p for p in train_dir.iterdir() if p.is_dir())
    if limit is not None:
        patient_dirs = patient_dirs[:limit]

    out_train_dir = output_dir / "train"
    videos_dir = out_train_dir / "videos"
    captions_dir = out_train_dir / "captions"
    videos_dir.mkdir(parents=True, exist_ok=True)
    captions_dir.mkdir(parents=True, exist_ok=True)

    caption_json = build_caption_json(fps)
    caption_txt = PROMPTS[0] + "\n"

    written = 0
    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        if patient_id in test_patient_ids:
            raise RuntimeError(
                f"OOD leakage guard: '{patient_id}' appears in both {train_dir} and {test_dir}. "
                "Refusing to build the SFT dataset -- see .claude/rules/baseline-experiments.md."
            )

        frames = load_views(patient_dir)
        write_lossless_mp4(frames, videos_dir / f"{patient_id}.mp4", fps=fps)

        patient_caption_dir = captions_dir / patient_id
        patient_caption_dir.mkdir(parents=True, exist_ok=True)
        (patient_caption_dir / "caption.json").write_text(json.dumps(caption_json, indent=2))
        (patient_caption_dir / "caption.txt").write_text(caption_txt)

        written += 1
        if written % 50 == 0:
            logger.info("Wrote %d/%d patients", written, len(patient_dirs))

    logger.info("Wrote %d video+caption pairs to %s", written, out_train_dir)
    return out_train_dir


def run_captions_to_sft_jsonl(train_dir: Path) -> Path:
    """Invoke cosmos-framework's official ``captions_to_sft_jsonl`` converter in-process.

    Deliberately reuses NVIDIA's converter (rather than hand-authoring the
    JSONL schema) so the loader-matching filters (max duration, min frames)
    and record shape stay correct even if ``sft_dataset.py`` changes
    upstream. Requires ``cosmos-framework/`` on ``sys.path`` (done at module
    import time above) -- this converter's own dependencies (``tyro``,
    ``ffprobe``) are lightweight enough to run from this repo's main venv,
    unlike the full training stack (see docs/cosmos-predict3/PLAN.md P0).
    """
    from cosmos_framework.scripts.captions_to_sft_jsonl import main as captions_to_sft_jsonl_main

    output_jsonl = train_dir / "video_dataset_file.jsonl"
    captions_to_sft_jsonl_main(
        captions_dir=train_dir / "captions",
        videos_dir=train_dir / "videos",
        output=output_jsonl,
        num_video_frames=NUM_FRAMES,
    )
    return output_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pre-rendered-dir", type=Path, default=Path("datasets/pre_rendered"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/cosmos3_sft"))
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N patients (smoke testing).")
    parser.add_argument(
        "--skip-jsonl",
        action="store_true",
        help="Write videos/captions only; skip invoking captions_to_sft_jsonl.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_dir = build_dataset(args.pre_rendered_dir, args.output_dir, fps=args.fps, limit=args.limit)
    if not args.skip_jsonl:
        output_jsonl = run_captions_to_sft_jsonl(train_dir)
        logger.info("DATASET_PATH for launch_sft_vision_edge.sh: %s", args.output_dir.resolve())
        logger.info("Wrote %s", output_jsonl)


if __name__ == "__main__":
    main()
