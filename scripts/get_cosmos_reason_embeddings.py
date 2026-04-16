"""
Generate Cosmos-Reason1 text embeddings for Cosmos-Predict 2.5.

This script generates 100,352-dimensional text embeddings using the Cosmos-Reason1-7B
text encoder (Qwen-VL-2.5-7B backbone, all 28 layers concatenated).

The embeddings are used to condition the DiT for view-specific camera parameter descriptions.

Usage:
    # Auto-resolve checkpoint (HF for external, S3 for internal)
    python -m scripts.get_cosmos_reason_embeddings --output_dir cosmos_reason_embeddings

    # Specify custom checkpoint path
    python -m scripts.get_cosmos_reason_embeddings --ckpt_path /path/to/checkpoint

    # Use different concatenation strategy
    python -m scripts.get_cosmos_reason_embeddings --concat_strategy mean_pooling

Environment:
    HF_TOKEN: HuggingFace token for model downloads (required for external machines)
    INTERNAL: Set to 1 for S3 access (NVidia internal only)
"""

import os
import sys

# Resolve the repository root and ensure it's in the Python path for imports
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

# Resolve Cosmos-Predict2.5 package paths for imports
_COSMOS_ROOT = os.path.join(_REPO_ROOT, "cosmos-predict2.5")
_COSMOS_OSS_PKG = os.path.join(_COSMOS_ROOT, "packages", "cosmos-oss")
if os.path.isdir(_COSMOS_ROOT):
    sys.path.insert(0, _COSMOS_ROOT)
if os.path.isdir(_COSMOS_OSS_PKG):
    sys.path.insert(0, _COSMOS_OSS_PKG)

import pickle
import argparse
import traceback
from pathlib import Path

import torch

from cosmos_predict2._src.predict2.text_encoders.text_encoder import (
    TextEncoder,
    TextEncoderConfig
)

from predict2_5.constants import (
    PROMPTS,
    CR1_EMBEDDING_DIM,
    CR1_HF_REVISION,
    COSMOS_REASON_UUID,
    DEFAULT_EMBEDDING_DIR
)
from predict2_5.utils import get_logger


# --- Logger --- #
logger = get_logger(__name__)


# --- Checkpoint patching --- #
_ORIGINAL_GET_CHECKPOINT_PATH = None


def setup_checkpoint_cache(ckpt_path):
    """
    Setup tokenizer cache and monkey-patch checkpoint resolution for external machines.

    Args:
        ckpt_path: The local path to the downloaded tokenizer checkpoint snapshot.
    """
    global _ORIGINAL_GET_CHECKPOINT_PATH
    
    import cosmos_predict2._src.imaginaire.utils.checkpoint_db as _ckpt_db
    
    # Save original function only once
    if _ORIGINAL_GET_CHECKPOINT_PATH is None:
        _ORIGINAL_GET_CHECKPOINT_PATH = _ckpt_db.get_checkpoint_path
    
    def _patched_get_checkpoint_path(checkpoint_uri):
        """
        Redirect S3 tokenizer to cached Text Encoder snapshot.

        Args:
            checkpoint_uri: The original checkpoint URI (S3 or HF).

        Returns:
            The local path to the cached tokenizer checkpoint if applicable, otherwise the original checkpoint path.
        """
        if "Qwen_tokenizer" in checkpoint_uri:
            logger.info("Redirecting S3 to cached tokenizer: %s", ckpt_path)

            return ckpt_path
        
        # Fall back to original
        return _ORIGINAL_GET_CHECKPOINT_PATH(checkpoint_uri)
    
    _ckpt_db.get_checkpoint_path = _patched_get_checkpoint_path
    logger.info("Checkpoint cache setup complete.")


def main(args):
    """Generate Cosmos-Reason1 text embeddings from prompts."""

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Output directory: %s", args.output_dir)
    logger.info("Concat strategy: %s", args.concat_strategy)
    logger.info("Expected dimension: %s", CR1_EMBEDDING_DIM)

    # Get the checkpoint path
    ckpt_path = args.ckpt_path

    # Track the execution mode (internal=S3, external=HF)
    INTERNAL = False  

    if not ckpt_path:
        # If no checkpoint path is provided, auto-resolve Cosmos-Reason1 checkpoint
        logger.info("Auto-resolving Cosmos-Reason1 checkpoint...")

        try:
            from cosmos_predict2._src.imaginaire.flags import INTERNAL as _INTERNAL
            from cosmos_predict2._src.imaginaire.utils.checkpoint_db import get_checkpoint_uri

            try:
                # Ensure UUID mappings (including Cosmos-Reason1) are registered.
                from cosmos_oss.checkpoints import register_checkpoints

                register_checkpoints()

            except Exception as register_err:
                logger.warning("Could not register checkpoint mappings explicitly: %s", register_err)

            # Determine execution environment (internal vs external) based on S3 access
            INTERNAL = _INTERNAL

            if INTERNAL:
                # Internal NVIDIA machine: resolve canonical Cosmos-Reason1 UUID to S3 URI.
                ckpt_path = get_checkpoint_uri(COSMOS_REASON_UUID)
                logger.info("Resolved Cosmos-Reason1 UUID to S3 URI: %s", ckpt_path)
            else:
                # External machine: download via HuggingFace using pinned revision.
                logger.info("Downloading Cosmos-Reason1 from HuggingFace (pinned revision=%s) ...", CR1_HF_REVISION)
                
                try:
                    from predict2_5.hf import download_text_encoder_snapshot 

                    # Download the checkpoint snapshot
                    ckpt_path = download_text_encoder_snapshot()

                    logger.info("Text encoder snapshot downloaded to: %s", ckpt_path)

                except ImportError:
                    logger.warning("Try to download via HuggingFace...")
                    
                    from huggingface_hub import snapshot_download

                    # Download the checkpoint snapshot from HuggingFace
                    ckpt_path = snapshot_download(
                        repo_id="nvidia/Cosmos-Reason1-7B",
                        revision=CR1_HF_REVISION,
                    )
                    
                    logger.info("Text encoder snapshot downloaded to: %s", ckpt_path)
        
        except Exception as e:
            logger.error("Auto-resolution failed: %s", e)
            logger.info("Please provide checkpoint path with --ckpt_path")
            logger.info("  python scripts/get_cosmos_reason_embeddings.py --ckpt_path /path/to/checkpoint")
            logger.info("Or use HuggingFace snapshot")
            logger.info("  python scripts/get_cosmos_reason_embeddings.py --ckpt_path nvidia/Cosmos-Reason1-7B")
            return

    # Load the Cosmos-Reason1 text encoder
    logger.info("Loading Cosmos-Reason1 encoder...")
    try:
        # For external machines: workaround to avoid S3 access in checkpoint_db
        if not INTERNAL:
            # Monkey-patch checkpoint resolution to redirect S3 tokenizer access to the downloaded snapshot
            setup_checkpoint_cache(ckpt_path)
            os.environ["COSMOS_EXPERIMENTAL_CHECKPOINTS"] = "0"

        # Create TextEncoder configuration
        encoder_config = TextEncoderConfig(
            ckpt_path=ckpt_path,
            compute_online=True,
            embedding_concat_strategy=args.concat_strategy,
        )

        # Initialize encoder (handle distributed setup for external mode)
        if not INTERNAL:
            # Mask torch.distributed to prevent QwenModel DTensor sharding
            # This avoids hangs on external machines
            import torch.distributed as _dist

            _orig_is_init = _dist.is_initialized
            _dist.is_initialized = lambda: False

            try:
                encoder = TextEncoder(
                    config=encoder_config, 
                    device=args.device
                )
            finally:
                _dist.is_initialized = _orig_is_init
        else:
            encoder = TextEncoder(
                config=encoder_config, 
                device=args.device
            )

        logger.info("TextEncoder loaded successfully.")

    except Exception as e:
        logger.error("Error loading encoder: %s", e)
        logger.info("Verify checkpoint path with --ckpt_path")
        logger.info("For external machines, ensure you have:")
        logger.info("  - HuggingFace hub access")
        logger.info("  - Enough GPU memory (14GB+ for Cosmos-Reason1-7B)")

        # Print full traceback for debugging
        traceback.print_exc()

        return

    # Generate embeddings for each prompt and save as pickle files
    processed_count = 0
    skipped_count = 0

    logger.info("Processing %d prompts...", len(PROMPTS))

    for idx, prompt in enumerate(PROMPTS, 1):
        # Derive output filename from prompt text
        if prompt:
            base_filename = (
                prompt.strip()
                .replace(" ", "_")
                .replace("'", "")
                .replace(",", "")[:100]
            )
        else:
            base_filename = "null"

        output_filename = os.path.join(args.output_dir, f"{base_filename}.pkl")

        # Skip if already processed and not overwriting
        if os.path.exists(output_filename) and not args.overwrite:
            skipped_count += 1
            logger.info("[%d/%d] SKIP: %s", idx, len(PROMPTS), output_filename)
            continue

        logger.info("[%d/%d] Processing: '%s...'", idx, len(PROMPTS), prompt[:60])

        try:
            # Generate embedding using Cosmos-Reason1 encoder
            with torch.no_grad():
                encoded_text = encoder.compute_text_embeddings_online(
                    {"text": [prompt]},
                    input_caption_key="text"
                )

            # Move to CPU and ensure float32 dtype
            encoded_text = encoded_text.to(torch.float32).cpu()

            # Prepare output: keep batch dimension for dataloader consistency
            # Shape: (1, seq_len, embedding_dim)
            encoded_text_output = [encoded_text[0]]

            # Save embedding as pickle file
            with open(output_filename, "wb") as fp:
                pickle.dump(encoded_text_output, fp)

            processed_count += 1
            logger.info(
                "Saved (%d tokens, %d dims): %s",
                encoded_text.shape[1],
                encoded_text.shape[2],
                Path(output_filename).name,
            )

        except Exception as e:
            logger.error("Error processing prompt #%d: %s", idx, e)
            traceback.print_exc()
            continue

    # ============================================================
    # Step 4: Summary
    # ============================================================
    logger.info("%s", "=" * 80)
    logger.info("Done! Processed: %d, Skipped: %d", processed_count, skipped_count)
    logger.info("Output: %s", args.output_dir)
    logger.info("Expected embedding dimension: %d", CR1_EMBEDDING_DIM)
    logger.info("%s", "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Cosmos-Reason1 text embeddings for Cosmos-Predict 2.5.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-resolve and generate embeddings (INTERNAL: S3, EXTERNAL: HuggingFace)
  python -m scripts.get_cosmos_reason_embeddings

  # Specify output directory
  python -m scripts.get_cosmos_reason_embeddings --output_dir my_embeddings

  # Use local checkpoint path
  python -m scripts.get_cosmos_reason_embeddings --ckpt_path /path/to/checkpoint

  # Regenerate existing embeddings
  python -m scripts.get_cosmos_reason_embeddings --overwrite

  # Use different concat strategy
  python -m scripts.get_cosmos_reason_embeddings --concat_strategy mean_pooling
        """,
    )

    # Output directory
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_EMBEDDING_DIR,
        help=f"Output directory for embedding files (default: {DEFAULT_EMBEDDING_DIR})",
    )

    # Checkpoint path
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to Cosmos-Reason1 checkpoint (local or S3). Auto-resolved if not provided.",
    )

    # Embedding strategy
    parser.add_argument(
        "--concat_strategy",
        type=str,
        default="full_concat",
        choices=["mean_pooling", "full_concat", "pool_every_n_layers_and_concat"],
        help="Embedding concatenation strategy (default: full_concat = 100352-dim)",
    )

    # Overwrite flag
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing embedding files",
    )

    # Device selection
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference (default: cuda)",
    )

    # Parse arguments
    args = parser.parse_args()

    main(args)
