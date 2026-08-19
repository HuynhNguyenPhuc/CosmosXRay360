"""Unit tests for predict3 package constants and inferencer helper utilities."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from predict3.constants import (
    COSMOS3_EDGE_REPO,
    NUM_FRAMES,
    NUM_LATENT_FRAMES,
    VAE_SPATIAL_DOWNSAMPLE,
    VAE_TEMPORAL_DOWNSAMPLE,
    VOL_SIZE,
)
from predict3.inferencer import InferencerV3


class TestConstants(unittest.TestCase):
    """Tests constants and VAE geometry shapes."""

    def test_shapes(self) -> None:
        self.assertEqual(VOL_SIZE, 256)
        self.assertEqual(NUM_FRAMES, 93)
        self.assertEqual(NUM_LATENT_FRAMES, 1 + (NUM_FRAMES - 1) // VAE_TEMPORAL_DOWNSAMPLE)
        self.assertEqual(NUM_LATENT_FRAMES, 24)

    def test_checkpoint_repo_is_valid_hf_id(self) -> None:
        self.assertRegex(COSMOS3_EDGE_REPO, r"^[\w.-]+/[\w.-]+$")
        self.assertEqual(COSMOS3_EDGE_REPO, "nvidia/Cosmos3-Edge")

    def test_vae_geometry_matches_wan22(self) -> None:
        self.assertEqual(VAE_SPATIAL_DOWNSAMPLE, 16)
        self.assertEqual(VAE_TEMPORAL_DOWNSAMPLE, 4)


class TestInferencerHelpers(unittest.TestCase):
    """Tests CPU-only utility methods of InferencerV3."""

    def test_to_pil_from_hwc_float_grayscale(self) -> None:
        arr = np.random.rand(VOL_SIZE, VOL_SIZE).astype(np.float32)
        img = InferencerV3._to_pil(arr)

        self.assertEqual(img.size, (VOL_SIZE, VOL_SIZE))
        self.assertEqual(img.mode, "RGB")

    def test_to_pil_from_chw_uint8(self) -> None:
        arr = (np.random.rand(3, 64, 64) * 255).astype(np.uint8)
        img = InferencerV3._to_pil(arr)

        self.assertEqual(img.size, (64, 64))
        self.assertEqual(img.mode, "RGB")

    def test_to_pil_rejects_unsupported_type(self) -> None:
        with self.assertRaises(TypeError):
            InferencerV3._to_pil("not an image")

    def test_resolve_checkpoint_passes_through_bare_repo_id(self) -> None:
        self.assertEqual(
            InferencerV3._resolve_checkpoint("nvidia/Cosmos3-Edge"), "nvidia/Cosmos3-Edge"
        )

    def test_resolve_checkpoint_resolves_local_dir(self) -> None:
        resolved = InferencerV3._resolve_checkpoint(".")
        self.assertTrue(resolved.startswith("/"))


if __name__ == "__main__":
    unittest.main()


