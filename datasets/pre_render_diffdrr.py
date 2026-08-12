"""
Pre-render CT volumes to 2D X-Ray projections (PA, LAT, and 360-degree views)
using the physically-rigorous DiffDRR renderer.

This populates the datasets/pre_rendered/ directory for training and evaluating
the baseline models, with no PyTorch3D dependency.
"""

from __future__ import annotations

import os
import sys
import glob
import logging
import argparse
import traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image

# Ensure workspace root is in path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from renderers.diffdrr.renderer import FIXED_LINE_INTEGRAL_MAX, create_diffdrr_renderer
from renderers.diffdrr.data import load_ct_volume

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_patient_id(file_path: str) -> str:
    """Extract a clean patient/scan identifier from file path."""
    name = Path(file_path).name
    if name.endswith(".nii.gz"):
        return name[:-7]
    elif name.endswith(".nii"):
        return name[:-4]
    return name


def get_dataset_optimal_fov(file_path: str) -> float:
    """
    Return dataset-matched optimal FOV obtained from high-density brute-force optimization
    (161 candidate FOVs, 0.05° resolution, 4 orthogonal angles across 20 CT scans).

    Matches MELA2022 Gold Standard framing without anatomical truncation:
      - MELA2022: 12.0° (Thorax Area = 96.31%, H-Black = 3.59%)
      - NSCLC:    10.2° (Thorax Area = 92.00%, H-Black = 3.23%)
      - TCIA:     11.9° (Thorax Area = 86.08%, H-Black = 7.77%)
    """
    p_str = str(file_path).upper()
    if "NSCLC" in p_str:
        return 10.2
    elif "TCIA" in p_str:
        return 11.9
    elif "MELA" in p_str:
        return 12.0
    return 12.0


def pre_render_ct(
    ct_path: str,
    output_patient_dir: Path,
    img_shape: int = 256,
    vol_shape: int = 256,
    num_frames: int = 93,
    dist: float = 8.0,
    elev: float = 0.0,
    fov: float = 12.0,
    min_depth: float = 7.0,
    max_depth: float = 9.0,
    device: str = "cuda",
) -> bool:
    """Render projections for a single CT scan and save them."""
    try:
        output_patient_dir.mkdir(parents=True, exist_ok=True)
        views_dir = output_patient_dir / "views"
        views_dir.mkdir(parents=True, exist_ok=True)

        pa_path = output_patient_dir / "pa.png"
        lat_path = output_patient_dir / "lat.png"

        # Load CT Volume using the custom loader (which resamples and scales it)
        logger.info(f"Loading CT volume from {ct_path}...")
        vol = load_ct_volume(ct_path, vol_shape=vol_shape)

        # Initialize DiffDRR renderer
        logger.info("Initializing DiffDRR renderer...")
        renderer = create_diffdrr_renderer(img_shape=img_shape, device=device)
        renderer.set_volume(vol)

        # Render 360-degree azimuth sweep as raw un-normalized line integrals (norm_type=None)
        logger.info(f"Rendering 360° sweep ({num_frames} frames raw line integrals)...")
        azimuths = torch.linspace(0.0, 360.0, num_frames, device=device)
        
        raw_frames_list = []
        chunk_size = 1  # Render one by one to keep VRAM usage extremely low (~1GB)
        for i in range(0, num_frames, chunk_size):
            chunk_azimuths = azimuths[i : i + chunk_size]
            chunk_frames = renderer.render(
                azimuth=chunk_azimuths,
                elev=elev,
                dist=dist,
                fov=fov,
                min_depth=min_depth,
                max_depth=max_depth,
                norm_type=None,  # Un-normalized raw line integrals
                batch_size=chunk_size,
            )  # [batch, 1, H, W]
            raw_frames_list.append(chunk_frames.cpu())
            
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        all_raw = torch.cat(raw_frames_list, dim=0)  # Shape: [93, 1, H, W]

        # Dataset-wide physical scaling with the fixed constant defined once in
        # renderers/diffdrr/renderer.py, to preserve relative attenuation ratios.
        norm_frames = torch.clamp(all_raw / FIXED_LINE_INTEGRAL_MAX, 0.0, 1.0)  # Shape: [93, 1, H, W]

        # 1. Save PA view (0.0°) and LAT view (90.0°)
        pa_np = (norm_frames[0, 0].numpy() * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(pa_np, mode="L").save(pa_path)

        lat_idx = int(round(90.0 / (360.0 / (num_frames - 1))))
        lat_np = (norm_frames[lat_idx, 0].numpy() * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(lat_np, mode="L").save(lat_path)

        # 2. Save individual PNG views
        for frame_idx in range(num_frames):
            frame_np = (norm_frames[frame_idx, 0].numpy() * 255.0).clip(0, 255).astype(np.uint8)
            frame_path = views_dir / f"{frame_idx:03d}.png"
            Image.fromarray(frame_np, mode="L").save(frame_path)

        # 3. Save single binary float32 tensor container (views.pt); strip MONAI MetaTensor subclass for safe loading
        pt_tensor = norm_frames.squeeze(1)
        if hasattr(pt_tensor, "as_tensor"):
            pt_tensor = pt_tensor.as_tensor()
        pt_path = output_patient_dir / "views.pt"
        torch.save(pt_tensor, pt_path)

        logger.info(f"Successfully processed and pre-rendered CT volume: {ct_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to process CT volume {ct_path}: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="DiffDRR-based Medical Dataset Pre-rendering Pipeline")
    parser.add_argument(
        "--dest_dir",
        type=str,
        default=str(BASE_DIR / "datasets" / "pre_rendered"),
        help="Destination directory for processed pre-rendered outputs"
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of files to process per split (for dry run)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run renderer on (e.g. cuda, cpu)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing pre-rendered files with new optimal FOVs"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated list of datasets to process (e.g., 'TCIA,NSCLC')"
    )
    args = parser.parse_args()

    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Filter datasets if specified
    allowed_ds = [d.strip().upper() for d in args.datasets.split(",")] if args.datasets else None

    # 1. Collect raw CT files for TRAIN split: MELA2022 and TCIA
    train_dirs = []
    if not allowed_ds or "TCIA" in allowed_ds:
        train_dirs.append(BASE_DIR / "datasets" / "TCIA" / "images")
    if not allowed_ds or "MELA2022" in allowed_ds or "MELA" in allowed_ds:
        train_dirs.append(BASE_DIR / "datasets" / "MELA2022" / "raw" / "train" / "images")

    train_ct_files = []
    for d in train_dirs:
        if d.exists():
            train_ct_files.extend(glob.glob(str(d / "*.nii.gz")))
    train_ct_files = sorted(list(set(train_ct_files)))

    # 2. Collect raw CT files for TEST split: NSCLC
    test_dirs = []
    if not allowed_ds or "NSCLC" in allowed_ds:
        test_dirs.append(BASE_DIR / "datasets" / "NSCLC" / "processed" / "train" / "images")

    test_ct_files = []
    for d in test_dirs:
        if d.exists():
            test_ct_files.extend(glob.glob(str(d / "*.nii.gz")))
    test_ct_files = sorted(list(set(test_ct_files)))

    logger.info(f"Collected {len(train_ct_files)} train CT files and {len(test_ct_files)} test CT files.")

    if args.max_files is not None:
        train_ct_files = train_ct_files[:args.max_files]
        test_ct_files = test_ct_files[:args.max_files]
        logger.info(f"Limited by --max_files to {len(train_ct_files)} train files and {len(test_ct_files)} test files.")

    # 3. Process TRAIN CT files
    logger.info("========================================")
    logger.info("🚀 Processing TRAIN Split")
    logger.info("========================================")
    train_out_dir = dest_dir / "train"
    train_success = 0
    for ct_path in train_ct_files:
        pat_id = get_patient_id(ct_path)
        pat_out_dir = train_out_dir / pat_id
        views_dir = pat_out_dir / "views"
        # Check if pa.png, lat.png exist and views has files
        if not args.overwrite and (pat_out_dir / "pa.png").exists() and (pat_out_dir / "lat.png").exists() and views_dir.exists() and len(list(views_dir.glob("*.png"))) >= 93:
            logger.info(f"Skipping already fully pre-rendered train case: {pat_id}")
            train_success += 1
            continue
        
        fov = get_dataset_optimal_fov(ct_path)
        logger.info(f"Processing train case {pat_id} with dataset optimal FOV={fov}°")
        success = pre_render_ct(
            ct_path=ct_path,
            output_patient_dir=pat_out_dir,
            fov=fov,
            device=args.device,
        )
        if success:
            train_success += 1

    # 4. Process TEST CT files
    logger.info("========================================")
    logger.info("🚀 Processing TEST Split")
    logger.info("========================================")
    test_out_dir = dest_dir / "test"
    test_success = 0
    for ct_path in test_ct_files:
        pat_id = get_patient_id(ct_path)
        pat_out_dir = test_out_dir / pat_id
        views_dir = pat_out_dir / "views"
        if not args.overwrite and (pat_out_dir / "pa.png").exists() and (pat_out_dir / "lat.png").exists() and views_dir.exists() and len(list(views_dir.glob("*.png"))) >= 93:
            logger.info(f"Skipping already fully pre-rendered test case: {pat_id}")
            test_success += 1
            continue
            
        fov = get_dataset_optimal_fov(ct_path)
        logger.info(f"Processing test case {pat_id} with dataset optimal FOV={fov}°")
        success = pre_render_ct(
            ct_path=ct_path,
            output_patient_dir=pat_out_dir,
            fov=fov,
            device=args.device,
        )
        if success:
            test_success += 1

    logger.info("========================================")
    logger.info("🏁 PRE-RENDERING SUMMARY")
    logger.info("========================================")
    logger.info(f"Train Split: {train_success}/{len(train_ct_files)} successfully pre-rendered.")
    logger.info(f"Test Split: {test_success}/{len(test_ct_files)} successfully pre-rendered.")
    logger.info(f"All outputs saved to: {dest_dir}")
    logger.info("========================================")


if __name__ == "__main__":
    main()
