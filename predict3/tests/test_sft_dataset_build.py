"""Unit tests for SFT dataset builder script (scripts/build_cosmos3_sft_dataset.py)."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "scripts"))

from build_cosmos3_sft_dataset import build_dataset, run_captions_to_sft_jsonl  # noqa: E402
from predict3.constants import NUM_FRAMES, VOL_SIZE  # noqa: E402

_HAS_DECORD = True
try:
    import decord
except ImportError:
    _HAS_DECORD = False


def _write_synthetic_patient(train_dir: Path, patient_id: str, seed: int) -> None:
    """Helper to generate synthetic patient view tensors for test fixtures."""
    rng = np.random.default_rng(seed)
    frames = rng.random((NUM_FRAMES, VOL_SIZE, VOL_SIZE), dtype=np.float32)

    patient_dir = train_dir / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)

    torch.save(torch.from_numpy(frames), patient_dir / "views.pt")


class TestBuildCosmos3SftDataset(unittest.TestCase):
    """Tests building SFT MP4 dataset and JSONL metadata."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="cosmos3_sft_test_")
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)

        self.pre_rendered_dir = Path(self._tmp) / "pre_rendered"
        self.output_dir = Path(self._tmp) / "cosmos3_sft"

        train_dir = self.pre_rendered_dir / "train"
        test_dir = self.pre_rendered_dir / "test"

        for i, pid in enumerate(["mela_0001", "mela_0002", "mela_0003"]):
            _write_synthetic_patient(train_dir, pid, seed=i)

        _write_synthetic_patient(test_dir, "LUNG1-001", seed=99)

        self.train_dir = train_dir
        self.test_dir = test_dir

    def test_round_trip_93_frames_and_jsonl_schema(self) -> None:
        out_train_dir = build_dataset(self.pre_rendered_dir, self.output_dir, fps=24.0)

        video_paths = sorted((out_train_dir / "videos").glob("*.mp4"))
        self.assertEqual(len(video_paths), 3)

        output_jsonl = run_captions_to_sft_jsonl(out_train_dir)
        self.assertTrue(output_jsonl.exists())

        records = [json.loads(line) for line in output_jsonl.read_text().splitlines()]
        self.assertEqual(len(records), 3)

        for record in records:
            self.assertIn("uuid", record)
            self.assertIn("vision_path", record)

            window = record["t2w_windows"][0]
            self.assertEqual(window["start_frame"], 0)
            self.assertEqual(window["end_frame"], NUM_FRAMES - 1)
            self.assertIn("caption_json", window)
            self.assertIn("caption", window)

    @unittest.skipUnless(_HAS_DECORD, "decord not installed")
    def test_mp4_round_trip_is_lossless_via_decord(self) -> None:
        out_train_dir = build_dataset(self.pre_rendered_dir, self.output_dir, fps=24.0, limit=1)

        video_path = out_train_dir / "videos" / "mela_0001.mp4"
        self.assertTrue(video_path.exists())

        vr = decord.VideoReader(str(video_path))
        self.assertEqual(len(vr), NUM_FRAMES)

        decoded = vr.get_batch(range(len(vr))).asnumpy()[..., 0].astype(np.float64)

        orig_float = torch.load(self.train_dir / "mela_0001" / "views.pt").numpy().astype(np.float64)
        orig_u8 = np.clip(orig_float * 255.0 + 0.5, 0, 255)

        mse = np.mean((decoded - orig_u8) ** 2)
        psnr = float("inf") if mse == 0 else 10.0 * np.log10(255.0**2 / mse)

        self.assertGreaterEqual(psnr, 50.0, f"MP4 round-trip PSNR too low: {psnr:.2f} dB")


if __name__ == "__main__":
    unittest.main()


    def test_nscl_leakage_guard_raises(self) -> None:
        # Plant an id that collides with the held-out NSCLC test split.
        _write_synthetic_patient(self.train_dir, "LUNG1-001", seed=7)
        with self.assertRaisesRegex(RuntimeError, "OOD leakage guard"):
            build_dataset(self.pre_rendered_dir, self.output_dir, fps=24.0)

    def test_wrong_frame_count_raises(self) -> None:
        bad_dir = self.train_dir / "bad_patient"
        bad_dir.mkdir(parents=True, exist_ok=True)
        torch.save(torch.rand(50, VOL_SIZE, VOL_SIZE), bad_dir / "views.pt")
        with self.assertRaises(ValueError):
            build_dataset(self.pre_rendered_dir, self.output_dir, fps=24.0, limit=None)


if __name__ == "__main__":
    unittest.main()
