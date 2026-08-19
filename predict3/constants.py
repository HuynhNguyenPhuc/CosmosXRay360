"""Constants and default configuration for Cosmos 3 (predict3)."""

from __future__ import annotations

# ==============================================================================
# Dataset Paths
# ==============================================================================

DEFAULT_CT_DIRS = [
    "datasets/NSCLC/processed/train/images/",
    "datasets/MELA2022/raw/train/images/",
    "datasets/MELA2022/raw/val/images/",
    "datasets/TCIA/images/",
]

DEFAULT_XR_DIR = "data/VinDr/v1/processed/"


# ==============================================================================
# Cosmos 3 Model Checkpoints
# ==============================================================================

COSMOS3_EDGE_REPO = "nvidia/Cosmos3-Edge"   # 4B; primary target model
COSMOS3_NANO_REPO = "nvidia/Cosmos3-Nano"   # 16B model
COSMOS3_SUPER_REPO = "nvidia/Cosmos3-Super" # 64B model


# ==============================================================================
# VAE Tokenizer & Patch Geometry
# ==============================================================================

VAE_TEMPORAL_DOWNSAMPLE = 4
VAE_SPATIAL_DOWNSAMPLE = 16
DIT_PATCH_MERGE = 2


# ==============================================================================
# Volume & Resolution Configuration
# ==============================================================================

VOL_SIZE = 256
IMG_HEIGHT = VOL_SIZE
IMG_WIDTH = VOL_SIZE
FRAME_HEIGHT = IMG_HEIGHT
FRAME_WIDTH = IMG_WIDTH


# ==============================================================================
# Frame & Latent Configuration
# ==============================================================================

NUM_FRAMES = 93  # Total 360-degree rotation views (0 to 360 degrees)
NUM_LATENT_FRAMES = 1 + (NUM_FRAMES - 1) // VAE_TEMPORAL_DOWNSAMPLE  # 24 latent frames


# ==============================================================================
# Default Prompts
# ==============================================================================

PROMPTS = [
    (
        "A 360-degree rotational view of a chest CT scan showing anatomical "
        "structures from all angles, rotating from 0 to 360 degrees azimuth."
    )
]

