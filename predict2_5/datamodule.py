"""DataModule for pre-rendered 93-view 360° rotational projections."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Optional, Union

from PIL import Image
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF

from lightning import LightningDataModule, seed_everything

# Ensure repo root is on sys.path for direct execution
_BASE_DIR = Path(__file__).resolve().parents[1]
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from predict2_5.constants import NUM_FRAMES, PROMPTS
from predict2_5.utils import get_logger


# --- Logger --- #
logger = get_logger(__name__)

DEFAULT_PROMPT = PROMPTS[0] if PROMPTS else "A 360-degree rotational view of a chest CT scan."


# ============================================================
# Dataset Class
# ============================================================

class PreRendered360Dataset(Dataset):
    """
    Dataset for loading pre-rendered 93-view 360° rotation projections.

    Expected directory structure:
        data_dir/
        ├── patient_001/
        │   ├── pa.png          # Frontal PA radiograph [256x256]
        │   └── views/          # 93 rotational frames
        │       ├── 000.png
        │       ├── 001.png
        │       └── ... (092.png)
        └── ...
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        patient_dirs: Optional[list[str]] = None,
        num_frames: int = NUM_FRAMES,
        img_size: int = 256,
        prompt: str = DEFAULT_PROMPT,
    ) -> None:
        """
        Initialize PreRendered360Dataset.

        Args:
            data_dir: Root directory containing patient subdirectories.
            patient_dirs: Optional list of specific patient directory names. If None, scans all in data_dir.
            num_frames: Total expected frames per rotation video (default: 93).
            img_size: Target image resolution (H, W). Default: 256.
            prompt: Default text prompt string for conditioning.
        """
        super().__init__()

        self.data_dir = Path(data_dir)
        self.num_frames = num_frames
        self.img_size = (img_size, img_size)
        self.prompt = prompt

        if patient_dirs is not None:
            self.patient_paths = [
                self.data_dir / p for p in patient_dirs if (self.data_dir / p).is_dir()
            ]
        else:
            if self.data_dir.exists():
                self.patient_paths = sorted([
                    p for p in self.data_dir.iterdir()
                    if p.is_dir() and (
                        (p / "views").exists() or (p / "views.pt").exists() or (p / "views.npy").exists()
                    )
                ])
            else:
                self.patient_paths = []

        logger.info(
            "Initialized PreRendered360Dataset with %d patient cases from %s",
            len(self.patient_paths),
            self.data_dir,
        )

    def __len__(self) -> int:
        return len(self.patient_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        patient_path = self.patient_paths[idx]
        patient_id = patient_path.name

        # 1. Load frontal PA X-ray image (pa.png)
        pa_path = patient_path / "pa.png"

        if pa_path.exists():
            pa_img = Image.open(pa_path).convert("L")
        elif (patient_path / "views").exists():
            # Fallback to first view if pa.png is missing
            pa_img = Image.open(patient_path / "views" / "000.png").convert("L")
        else:
            pa_img = None

        if pa_img is not None:
            pa_tensor = TF.to_tensor(pa_img)  # Shape: [1, H, W], range [0.0, 1.0]
            if pa_tensor.shape[-2:] != self.img_size:
                pa_tensor = TF.resize(pa_tensor, list(self.img_size))
            pa_tensor = pa_tensor.repeat(3, 1, 1)  # Expand to 3 channels: [3, H, W]
        else:
            pa_tensor = torch.zeros((3, self.img_size[0], self.img_size[1]), dtype=torch.float32)

        # 2. Fast single-file loading or multi-view PNG fallback
        pt_views_path = patient_path / "views.pt"
        npy_views_path = patient_path / "views.npy"

        if pt_views_path.exists():
            # Fast binary PyTorch tensor read (1 file open vs 93 PNG opens)
            video_tensor = torch.load(pt_views_path, weights_only=True)
            if video_tensor.dtype == torch.uint8:
                video_tensor = video_tensor.float() / 255.0
            if video_tensor.dim() == 3:  # [T, H, W] -> [1, T, H, W]
                video_tensor = video_tensor.unsqueeze(0)
            elif video_tensor.dim() == 4:
                if video_tensor.shape[1] == 1 and video_tensor.shape[0] != 1:  # [T, 1, H, W] -> [1, T, H, W]
                    video_tensor = video_tensor.permute(1, 0, 2, 3)
                elif video_tensor.shape[0] == 3:  # [3, T, H, W] -> [1, T, H, W]
                    video_tensor = video_tensor[:1]  # Keep 1 channel, expand on GPU
        elif npy_views_path.exists():
            # Fast binary NumPy array read
            arr = np.load(npy_views_path)
            video_tensor = torch.from_numpy(arr)
            if video_tensor.dtype == torch.uint8:
                video_tensor = video_tensor.float() / 255.0
            if video_tensor.dim() == 3:  # [T, H, W] -> [1, T, H, W]
                video_tensor = video_tensor.unsqueeze(0)
            elif video_tensor.dim() == 4:
                if video_tensor.shape[1] == 1 and video_tensor.shape[0] != 1:  # [T, 1, H, W] -> [1, T, H, W]
                    video_tensor = video_tensor.permute(1, 0, 2, 3)
                elif video_tensor.shape[0] == 3:
                    video_tensor = video_tensor[:1]
        else:
            # Fallback to multi-view PNG loading (single channel output, expanded on GPU)
            views_dir = patient_path / "views"
            view_files = sorted([f for f in os.listdir(views_dir) if f.endswith(".png")]) if views_dir.exists() else []

            if len(view_files) < self.num_frames:
                logger.warning(
                    "Patient %s has %d views, expected %d.",
                    patient_id,
                    len(view_files),
                    self.num_frames,
                )

            selected_files = view_files[: self.num_frames]
            frame_tensors: list[torch.Tensor] = []

            for vf in selected_files:
                v_path = views_dir / vf
                v_img = Image.open(v_path).convert("L")
                v_tensor = TF.to_tensor(v_img)  # [1, H, W] in [0, 1]

                if v_tensor.shape[-2:] != self.img_size:
                    v_tensor = TF.resize(v_tensor, list(self.img_size))

                frame_tensors.append(v_tensor)

            # Handle case if fewer views are present by padding last frame
            while len(frame_tensors) < self.num_frames:
                frame_tensors.append(
                    frame_tensors[-1].clone()
                    if frame_tensors
                    else torch.zeros((1, self.img_size[0], self.img_size[1]))
                )

            # Stack into 4D video tensor [1, T, H, W] -> [1, 93, 256, 256]
            video_tensor = torch.stack(frame_tensors, dim=1)

        # 3. Load sidecar prompt if available
        prompt_txt_path = patient_path / "prompt.txt"
        prompt_str = self.prompt

        if prompt_txt_path.exists():
            try:
                prompt_str = prompt_txt_path.read_text(encoding="utf-8").strip()
            except Exception as e:
                logger.warning("Failed to read prompt file %s: %s", prompt_txt_path, e)

        return {
            "video": video_tensor,  # FloatTensor [1, 93, 256, 256] in [0, 1] (1-channel, expanded on GPU)
            "image": pa_tensor,     # FloatTensor [3, 256, 256] in [0, 1]
            "prompt": prompt_str,
            "patient_id": patient_id,
        }


class PreRenderedLatentDataset(Dataset):
    """
    Dataset for loading pre-encoded Wan2.1 VAE latent tensors z_0 alongside frontal PA X-rays.

    Expected directory structure:
        latent_dir/ (e.g. datasets/pre_rendered_latents/train/)
        ├── patient_001/
        │   ├── latent.pt       # Pre-encoded VAE latent tensor [16, 24, 32, 32]
        │   ├── pa.png          # Optional frontal PA radiograph [256x256]
        │   └── prompt.txt      # Optional sidecar prompt
        └── ...
    """

    def __init__(
        self,
        latent_dir: Union[str, Path],
        data_dir: Optional[Union[str, Path]] = None,
        patient_dirs: Optional[list[str]] = None,
        img_size: int = 256,
        prompt: str = DEFAULT_PROMPT,
    ) -> None:
        """
        Initialize PreRenderedLatentDataset.

        Args:
            latent_dir: Directory containing pre-encoded latent patient subdirectories.
            data_dir: Optional fallback directory for pa.png if missing in latent_dir.
            patient_dirs: Optional list of specific patient directory names.
            img_size: Target image resolution for pa.png (default: 256).
            prompt: Default text prompt string.
        """
        super().__init__()

        self.latent_dir = Path(latent_dir)
        self.data_dir = Path(data_dir) if data_dir is not None else self.latent_dir
        self.img_size = (img_size, img_size)
        self.prompt = prompt

        if patient_dirs is not None:
            self.patient_paths = [
                self.latent_dir / p for p in patient_dirs
                if (self.latent_dir / p).is_dir() and (self.latent_dir / p / "latent.pt").exists()
            ]
        else:
            if self.latent_dir.exists():
                self.patient_paths = sorted([
                    p for p in self.latent_dir.iterdir()
                    if p.is_dir() and (p / "latent.pt").exists()
                ])
            else:
                self.patient_paths = []

        logger.info(
            "Initialized PreRenderedLatentDataset with %d patient cases from %s",
            len(self.patient_paths),
            self.latent_dir,
        )

    def __len__(self) -> int:
        return len(self.patient_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        patient_path = self.patient_paths[idx]
        patient_id = patient_path.name

        # 1. Load pre-cached latent tensor z_0
        latent_file = patient_path / "latent.pt"
        if not latent_file.exists():
            raise FileNotFoundError(f"Pre-cached latent file not found at {latent_file}")

        latent_tensor = torch.load(latent_file, map_location="cpu")  # Shape [16, 24, 32, 32]
        if latent_tensor.dim() == 5 and latent_tensor.shape[0] == 1:
            latent_tensor = latent_tensor.squeeze(0)

        # 2. Load frontal PA X-ray image (pa.png) from latent_dir or fallback to data_dir
        pa_path = patient_path / "pa.png"
        if not pa_path.exists() and self.data_dir != self.latent_dir:
            pa_path = self.data_dir / patient_id / "pa.png"
        if not pa_path.exists() and (self.data_dir / patient_id / "views" / "000.png").exists():
            pa_path = self.data_dir / patient_id / "views" / "000.png"

        if pa_path.exists():
            pa_img = Image.open(pa_path).convert("L")
            pa_tensor = TF.to_tensor(pa_img)  # Shape [1, H, W]
            if pa_tensor.shape[-2:] != self.img_size:
                pa_tensor = TF.resize(pa_tensor, list(self.img_size))
            pa_tensor = pa_tensor.repeat(3, 1, 1)  # Shape [3, H, W]
        else:
            pa_tensor = torch.zeros((3, self.img_size[0], self.img_size[1]), dtype=torch.float32)

        # 3. Load sidecar prompt if available
        prompt_txt_path = patient_path / "prompt.txt"
        if not prompt_txt_path.exists() and self.data_dir != self.latent_dir:
            prompt_txt_path = self.data_dir / patient_id / "prompt.txt"

        prompt_str = self.prompt
        if prompt_txt_path.exists():
            try:
                prompt_str = prompt_txt_path.read_text(encoding="utf-8").strip()
            except Exception as e:
                logger.warning("Failed to read prompt file %s: %s", prompt_txt_path, e)

        return {
            "pre_cached_latent": latent_tensor,  # FloatTensor [16, 24, 32, 32]
            "latent": latent_tensor,             # Alias for compatibility
            "image": pa_tensor,                  # FloatTensor [3, 256, 256] in [0, 1]
            "prompt": prompt_str,
            "patient_id": patient_id,
        }


# ============================================================
# DataModule Class
# ============================================================

class PreRenderedDataModule(LightningDataModule):
    """Lightning DataModule for loading pre-rendered 93-view 360° rotation dataset or pre-encoded VAE latents."""

    def __init__(
        self,
        dataset_path: str = "datasets/pre_rendered",
        use_latent_cache: bool = False,
        latent_cache_dir: Optional[str] = "datasets/pre_rendered_latents",
        val_split_ratio: float = 0.1,
        batch_size: int = 1,
        num_workers: int = 4,
        prefetch_factor: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        """
        Initialize PreRenderedDataModule.

        Args:
            dataset_path: Path to dataset root (containing `train` and `test` subdirs).
            use_latent_cache: If True, uses pre-encoded VAE latent dataset instead of raw PNG views.
            latent_cache_dir: Directory containing pre-encoded VAE latents.
            val_split_ratio: Fraction of train dataset used for validation.
            batch_size: Batch size for dataloaders.
            num_workers: DataLoader worker count.
            prefetch_factor: DataLoader prefetch factor per worker.
            pin_memory: Pin memory for GPU transfer efficiency.
            persistent_workers: Keep dataloader workers active across epochs.
            seed: RNG seed for reproducible train/val splits.
            **kwargs: Extra arguments for compatibility with TrainingConfig.
        """
        super().__init__()

        self.save_hyperparameters()

        self.dataset_path = Path(dataset_path)
        self.train_dir = (
            self.dataset_path / "train"
            if (self.dataset_path / "train").exists()
            else self.dataset_path
        )
        self.test_dir = self.dataset_path / "test"

        self.latent_cache_dir = Path(latent_cache_dir) if latent_cache_dir else None

        self.train_dataset: Optional[Union[PreRendered360Dataset, PreRenderedLatentDataset]] = None
        self.val_dataset: Optional[Union[PreRendered360Dataset, PreRenderedLatentDataset]] = None
        self.test_dataset: Optional[PreRendered360Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up datasets for train, val, and test splits.

        Args:
            stage: Stage hint ('fit', 'test', or None).
        """
        seed_everything(self.hparams.seed)

        if stage in (None, "fit"):
            if self.train_dir.exists():
                all_patients = sorted([
                    p.name
                    for p in self.train_dir.iterdir()
                    if p.is_dir() and (
                        (p / "views").exists() or (p / "views.pt").exists() or (p / "views.npy").exists() or (p / "latent.pt").exists()
                    )
                ])
            else:
                all_patients = []

            n_val = max(1, int(len(all_patients) * self.hparams.val_split_ratio)) if len(all_patients) > 1 else 0

            # Reproducible shuffle
            g = torch.Generator().manual_seed(self.hparams.seed)
            perm = (
                torch.randperm(len(all_patients), generator=g).tolist()
                if len(all_patients) > 0
                else []
            )
            shuffled_patients = [all_patients[i] for i in perm]

            val_patients = shuffled_patients[:n_val]
            train_patients = shuffled_patients[n_val:]

            # Check if latent cache should and can be used
            use_latent = False
            train_latent_dir = None

            if self.hparams.use_latent_cache and self.latent_cache_dir:
                candidate_dir = (
                    self.latent_cache_dir / "train"
                    if (self.latent_cache_dir / "train").exists()
                    else self.latent_cache_dir
                )
                if candidate_dir.exists() and any(candidate_dir.iterdir()):
                    use_latent = True
                    train_latent_dir = candidate_dir
                else:
                    logger.warning(
                        "use_latent_cache=True requested, but pre-cached latent directory %s is empty or not found. Falling back to raw PNG dataset.",
                        candidate_dir,
                    )

            if use_latent:
                self.train_dataset = PreRenderedLatentDataset(
                    latent_dir=train_latent_dir,
                    data_dir=self.train_dir,
                    patient_dirs=train_patients,
                )
                self.val_dataset = PreRenderedLatentDataset(
                    latent_dir=train_latent_dir,
                    data_dir=self.train_dir,
                    patient_dirs=val_patients,
                )
                logger.info(
                    "DataModule setup (fit with pre-cached latents): %d train cases, %d val cases",
                    len(self.train_dataset),
                    len(self.val_dataset),
                )
            else:
                self.train_dataset = PreRendered360Dataset(
                    data_dir=self.train_dir,
                    patient_dirs=train_patients,
                )
                self.val_dataset = PreRendered360Dataset(
                    data_dir=self.train_dir,
                    patient_dirs=val_patients,
                )
                logger.info(
                    "DataModule setup (fit): %d train cases, %d val cases",
                    len(train_patients),
                    len(val_patients),
                )

        if stage in (None, "test"):
            if self.test_dir.exists():
                self.test_dataset = PreRendered360Dataset(data_dir=self.test_dir)
                logger.info(
                    "DataModule setup (test): %d test cases",
                    len(self.test_dataset),
                )
            else:
                self.test_dataset = None

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        if self.train_dataset is None or len(self.train_dataset) == 0:
            raise RuntimeError("Train dataset is empty or not initialized.")

        kwargs = {}
        if self.hparams.num_workers > 0:
            kwargs["prefetch_factor"] = self.hparams.prefetch_factor

        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=(
                self.hparams.persistent_workers
                if self.hparams.num_workers > 0
                else False
            ),
            **kwargs,
        )

    def val_dataloader(self) -> DataLoader | list:
        """Create validation dataloader."""
        if self.val_dataset is None or len(self.val_dataset) == 0:
            return []

        kwargs = {}
        if self.hparams.num_workers > 0:
            kwargs["prefetch_factor"] = self.hparams.prefetch_factor

        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=(
                self.hparams.persistent_workers
                if self.hparams.num_workers > 0
                else False
            ),
            **kwargs,
        )

    def test_dataloader(self) -> DataLoader | list:
        """Create test dataloader."""
        if self.test_dataset is None or len(self.test_dataset) == 0:
            return []

        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )


# Backward compatibility alias
ChestMedicalDataModule = PreRenderedDataModule


# ============================================================
# Main Test Entrypoint
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test PreRenderedDataModule loading and batch shapes"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="datasets/pre_rendered",
        help="Path to pre-rendered dataset root directory",
    )
    args = parser.parse_args()

    logger.info("Testing PreRenderedDataModule on %s...", args.data_dir)

    # Setup the data module
    dm = PreRenderedDataModule(dataset_path=args.data_dir, batch_size=1)
    dm.setup("fit")

    # Get the data loader
    loader = dm.train_dataloader()

    for batch in loader:
        logger.info("✓ Batch loaded successfully:")
        logger.info("  - Video shape: %s", batch["video"].shape)
        logger.info("  - Image shape: %s", batch["image"].shape)
        logger.info("  - Prompt:      %s", batch["prompt"][0])
        logger.info("  - Patient ID:  %s", batch["patient_id"][0])
        break

