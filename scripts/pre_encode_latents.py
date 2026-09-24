#!/usr/bin/env python3
"""
Pre-Encode VAE Latents Utility for Cosmos-Predict2.5
==================================================
Runs the Wan2.1 3D VAE encoder once across all training CT rotation projections (TCIA + MELA2022)
and saves pre-encoded latent tensors z_0 [16, 24, 32, 32] to disk.

Usage:
    uv run python scripts/pre_encode_latents.py \
        --dataset_dir datasets/pre_rendered \
        --output_dir datasets/pre_rendered_latents \
        --batch_size 1 \
        --device cuda

OOD Discipline Note:
    Strictly pre-encodes the training split (TCIA + MELA2022) only. Never encode the
    held-out NSCLC test set into training latent cache!
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Safely stub optional missing third-party packages required by cosmos_predict2.easy_io
# to avoid modifying files in the cosmos-predict2.5 submodule
if "multistorageclient" not in sys.modules:
    try:
        __import__("multistorageclient")
    except ImportError:
        from unittest.mock import MagicMock
        sys.modules["multistorageclient"] = MagicMock()

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from predict2_5.constants import COSMOS_TOKENIZER_UUID
from predict2_5.datamodule import PreRendered360Dataset
from predict2_5.hf import resolve_hf_uri, hf_download
from predict2_5.utils import get_logger, is_uuid_format, setup_early_logging


# Setup logging
setup_early_logging()

# --- Logger --- #
logger = get_logger("pre_encode_latents")


def resolve_artifact_path(spec: str, artifact_name: str = "artifact") -> str:
    """Resolve artifact spec (local path, hf:// URI, or Cosmos UUID) to local path."""
    if not spec:
        return spec

    path_candidate = Path(spec)
    if path_candidate.exists():
        logger.info("Resolved %s from local path: %s", artifact_name, path_candidate)
        return str(path_candidate)

    if spec.startswith("hf://"):
        repo_id, filename, revision = resolve_hf_uri(spec)
        resolved = hf_download(repo_id=repo_id, filename=filename, revision=revision)
        logger.info("Resolved %s from HuggingFace: %s", artifact_name, resolved)
        return resolved

    if is_uuid_format(spec):
        os.environ.setdefault("COSMOS_EXPERIMENTAL_CHECKPOINTS", "1")
        try:
            from cosmos_oss.checkpoints_predict2 import register_checkpoints
            from cosmos_predict2._src.imaginaire.utils.checkpoint_db import download_checkpoint

            register_checkpoints()
            resolved = download_checkpoint(spec)
            logger.info("Resolved %s from checkpoint UUID: %s", artifact_name, resolved)
            return resolved
        except Exception as e:
            logger.warning("Failed UUID resolution for %s (%s): %s. Falling back to HuggingFace tokenizer download.", spec, artifact_name, e)
            from predict2_5.hf import download_wan_vae_tokenizer
            resolved = download_wan_vae_tokenizer()
            logger.info("Resolved %s from HuggingFace fallback: %s", artifact_name, resolved)
            return resolved

    return spec


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for VAE latent pre-encoding."""
    parser = argparse.ArgumentParser(
        description="Pre-encode 93-view CT rotation videos into Wan2.1 VAE latents."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="datasets/pre_rendered",
        help="Root directory containing pre-rendered PNG datasets (train/ and optional test/).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/pre_rendered_latents",
        help="Target directory to save pre-encoded latent tensors.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Local path, hf:// URI, or Cosmos UUID for Wan2.1 VAE tokenizer.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size per encoding iteration (default: 1).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader worker count (default: 4).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float32", "float16"],
        help="Storage dtype for saved latent tensors (default: bfloat16).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cached latent tensors if present.",
    )
    return parser.parse_args()


def pre_encode_split(
    split_name: str,
    src_split_dir: Path,
    dst_split_dir: Path,
    tokenizer: Any,
    args: argparse.Namespace,
) -> int:
    """
    Pre-encodes a single dataset split directory.

    Args:
        split_name: Name of split ('train').
        src_split_dir: Path to source PNG split directory.
        dst_split_dir: Path to destination latent split directory.
        tokenizer: Wan2pt1VAEInterface instance.
        args: Parsed CLI arguments.

    Returns:
        Number of patient cases processed.
    """
    if not src_split_dir.exists():
        logger.warning(f"Source split directory {src_split_dir} does not exist. Skipping.")
        return 0

    dataset = PreRendered360Dataset(data_dir=src_split_dir)
    if len(dataset) == 0:
        logger.warning(f"No patient cases found in {src_split_dir}. Skipping.")
        return 0

    logger.info(f"Processing split '{split_name}': {len(dataset)} patient cases in {src_split_dir}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device.startswith("cuda")),
    )

    save_dtype = getattr(torch, args.dtype)
    encoded_count = 0

    for batch in tqdm(dataloader, desc=f"Encoding {split_name} latents"):
        videos = batch["video"]  # [B, 3, 93, 256, 256]
        patient_ids = batch["patient_id"]

        # Check if all in batch are already cached
        all_cached = True
        for pid in patient_ids:
            patient_dst = dst_split_dir / pid
            latent_file = patient_dst / "latent.pt"
            if not latent_file.exists() or args.overwrite:
                all_cached = False
                break

        if all_cached:
            encoded_count += len(patient_ids)
            continue

        videos_device = videos.to(device=args.device, dtype=torch.float32)
        if videos_device.dim() == 5 and videos_device.shape[1] == 1:
            videos_device = videos_device.repeat(1, 3, 1, 1, 1)

        with torch.no_grad():
            latents = tokenizer.encode(videos_device)  # [B, 16, 24, 32, 32]

        for i, pid in enumerate(patient_ids):
            patient_src = src_split_dir / pid
            patient_dst = dst_split_dir / pid
            patient_dst.mkdir(parents=True, exist_ok=True)

            latent_file = patient_dst / "latent.pt"
            if not latent_file.exists() or args.overwrite:
                single_latent = latents[i].cpu().to(dtype=save_dtype)
                torch.save(single_latent, latent_file)

            # Copy frontal PA radiograph and optional prompt for self-contained latent dataset
            pa_src = patient_src / "pa.png"
            pa_dst = patient_dst / "pa.png"
            if pa_src.exists() and (not pa_dst.exists() or args.overwrite):
                shutil.copy2(pa_src, pa_dst)

            prompt_src = patient_src / "prompt.txt"
            prompt_dst = patient_dst / "prompt.txt"
            if prompt_src.exists() and (not prompt_dst.exists() or args.overwrite):
                shutil.copy2(prompt_src, prompt_dst)

            encoded_count += 1

    return encoded_count


def main() -> None:
    args = parse_args()

    # OOD Discipline check
    dataset_path = Path(args.dataset_dir)
    if "test" in dataset_path.name.lower() or "nsclc" in str(dataset_path).lower():
        raise ValueError(
            "OOD Discipline Violation: Never pre-encode the NSCLC test set into training latent cache!"
        )

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Resolve VAE Tokenizer checkpoint
    tokenizer_spec = args.tokenizer_path or COSMOS_TOKENIZER_UUID
    logger.info(f"Resolving VAE Tokenizer checkpoint: {tokenizer_spec}")
    resolved_tokenizer_path = resolve_artifact_path(
        spec=tokenizer_spec,
        artifact_name="tokenizer",
    )

    # Initialize Wan2.1 VAE Tokenizer
    from cosmos_predict2._src.predict2.tokenizers.wan2pt1 import Wan2pt1VAEInterface

    logger.info(f"Initializing Wan2.1 VAE Tokenizer from {resolved_tokenizer_path} on {args.device}...")
    tokenizer = Wan2pt1VAEInterface(
        chunk_duration=93,
        load_mean_std=False,
        vae_pth=resolved_tokenizer_path,
        temporal_window=16,
        keep_decoder_cache=False,
        keep_encoder_cache=False,
    )

    from predict2_5.utils import move_tokenizer_to_device
    move_tokenizer_to_device(tokenizer, args.device)

    # Process training split exclusively
    train_src = dataset_path / "train" if (dataset_path / "train").exists() else dataset_path
    train_dst = output_path / "train" if (dataset_path / "train").exists() else output_path

    total_encoded = pre_encode_split(
        split_name="train",
        src_split_dir=train_src,
        dst_split_dir=train_dst,
        tokenizer=tokenizer,
        args=args,
    )

    logger.info(f"✓ Completed VAE pre-encoding. Total training cases cached: {total_encoded}")


if __name__ == "__main__":
    main()
