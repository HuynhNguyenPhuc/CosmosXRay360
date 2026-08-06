"""Unified Multi-Model Evaluation Harness for CosmosXRay360 Phase 1.

Imports real wrapper implementations from models package.
Cross-Dataset OOD Evaluation: TCIA+MELA (train) → NSCLC (test)
Physics-grounded: Beer-Lambert correction, DiffDRR Siddon-Jacob projection
"""

from __future__ import annotations

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Prioritize local baselines folder so that local packages (like models) are loaded first
sys.path.insert(0, BASE_DIR)

import json
import logging
import time
import warnings

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available")

try:
    import diffdrr.data as ddata
    from diffdrr.drr import DRR
    from diffdrr.pose import convert
    DIFFDRR_AVAILABLE = True
except ImportError:
    DIFFDRR_AVAILABLE = False
    warnings.warn("DiffDRR not available")

# Import wrappers from models package
from models import (
    SVDRRWrapper,
    XRaySynWrapper,
    MedNeRFWrapper,
    PixelNeRFWrapper,
    NAFWrapper,
    Dx2CTWrapper,
    compute_psnr,
    compute_ssim,
    compute_lpips,
    apply_beer_lambert_correction,
    normalize_tensor,
)

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logger following STYLE.md
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# UNIFIED EVALUATOR
# ============================================================================

class UnifiedBaselineEvaluator:
    """Comprehensive evaluation harness integrating all 6 baselines.

    This evaluator initializes all wrappers, manages multi-model OOD split
    predictions, and measures standard y y-metrics like PSNR/SSIM.
    """
    
    def __init__(self, checkpoint_dir: str = "baselines/checkpoints") -> None:
        """Initialize unified baseline evaluator.

        Args:
            checkpoint_dir: Directory where model weights are stored.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_dir = checkpoint_dir
        self.baselines: dict[str, any] = {}
        self.results: dict[str, any] = {}

        if self.device == "cuda":
            # Every baselines/train/*.py script sets these; the evaluation harness
            # itself never did, so it ran full eval sweeps without cuDNN's
            # autotuner (repeated fixed-shape conv workloads across every wrapper).
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        logger.info("Initializing baseline wrappers from models package...")
        self._init_baselines()
    
    def _init_baselines(self) -> None:
        """Initialize all 6 baseline wrappers with configured checkpoints."""
        def resolve_ckpt(prefix: str) -> str:
            best_file = os.path.join(self.checkpoint_dir, f"{prefix}_best.pt")
            latest_file = os.path.join(self.checkpoint_dir, f"{prefix}_checkpoint.pt")
            surrogate_file = os.path.join(self.checkpoint_dir, f"{prefix}_surrogate_checkpoint.pt")
            
            if os.path.exists(best_file):
                logger.info(f"Using best checkpoint for {prefix}: {best_file}")
                return best_file
            if os.path.exists(surrogate_file):
                logger.info(f"Using surrogate checkpoint for {prefix}: {surrogate_file}")
                return surrogate_file
            if os.path.exists(latest_file):
                logger.info(f"Using latest checkpoint for {prefix}: {latest_file}")
                return latest_file
            
            fallback = os.path.join("eval_outputs", f"{prefix}_checkpoint.pt")
            if os.path.exists(fallback):
                return fallback
            return latest_file

        checkpoints = {
            "SV-DRR": resolve_ckpt("svdrr"),
            "XRaySyn": resolve_ckpt("xraysyn"),
            "MedNeRF": resolve_ckpt("mednerf"),
            "PixelNeRF": resolve_ckpt("pixelnerf"),
            "NAF": resolve_ckpt("naf"),
            "Dx2CT": resolve_ckpt("dx2ct"),
        }
        
        # Initialize all 6 baselines with checkpoint paths
        self.baselines["SV-DRR"] = SVDRRWrapper(checkpoints.get("SV-DRR"))
        self.baselines["XRaySyn"] = XRaySynWrapper(checkpoints.get("XRaySyn"))
        self.baselines["MedNeRF"] = MedNeRFWrapper(checkpoints.get("MedNeRF"))
        self.baselines["PixelNeRF"] = PixelNeRFWrapper(checkpoints.get("PixelNeRF"))
        self.baselines["NAF"] = NAFWrapper(checkpoints.get("NAF"))
        self.baselines["Dx2CT"] = Dx2CTWrapper(checkpoints.get("Dx2CT"))
    
    def run_evaluation_suite(self, max_patients: int | None = None) -> None:
        """Execute full evaluation across all baselines on the real test split.

        Args:
            max_patients: Optional cap on the number of test patients evaluated per
                baseline, for bounded smoke verification of the harness itself (the
                full NSCLC test split is 241 patients; several baselines run a
                per-patient test-time optimization loop, so a full 6-baseline sweep
                can take hours). Defaults to None (evaluate every patient).
        """
        logger.info("=" * 80)
        logger.info("🚀 Launching Unified Standardized Baseline Evaluation Harness")
        logger.info("=" * 80)
        logger.info("Cross-Dataset Split: TCIA+MELA (train) → NSCLC (test)")
        logger.info("Standardized Projector: DiffDRR Siddon-Jacob Ray-tracing")
        logger.info("Physics-grounded: Beer-Lambert Attenuation Correction")
        logger.info("=" * 80)
        
        baseline_names = ["XRaySyn", "MedNeRF", "Dx2CT", "SV-DRR", "PixelNeRF", "NAF"]
        computed_results: list[tuple[str, float, float, float]] = []

        import torchvision.transforms.functional as TF
        from PIL import Image

        # Load all real patient test cases from datasets/pre_rendered/test
        real_data_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "datasets", "pre_rendered", "test"))
        if not os.path.exists(real_data_dir) or len(os.listdir(real_data_dir)) < 50:
            logger.info("⚡ Test dataset missing or incomplete in datasets/pre_rendered/test.")
            logger.info("📥 Downloading 100% full dataset from Hugging Face hub (hvcl-gm/chest-medical-image-dataset)...")
            import subprocess
            fast_dl_script = os.path.join(BASE_DIR, "..", "scripts", "fast_download.py")
            pre_render_script = os.path.join(BASE_DIR, "..", "datasets", "pre_render_diffdrr.py")
            subprocess.run([sys.executable, fast_dl_script, "--percentage", "1.0"], check=True)
            logger.info("⚙️ Pre-rendering 100% full dataset with DiffDRR Siddon-Jacob raymarching...")
            subprocess.run([sys.executable, pre_render_script], check=True)

        test_patient_dirs = []
        if os.path.exists(real_data_dir):
            test_patient_dirs = sorted([
                os.path.join(real_data_dir, d)
                for d in os.listdir(real_data_dir)
                if os.path.isdir(os.path.join(real_data_dir, d))
            ])
        if max_patients is not None:
            test_patient_dirs = test_patient_dirs[:max_patients]

        logger.info(f"Loaded {len(test_patient_dirs)} real patient cases from {real_data_dir} for evaluation.\n")

        for name in baseline_names:
            wrapper = self.baselines.get(name)
            if not wrapper or (getattr(wrapper, "model", None) is None and getattr(wrapper, "pipe", None) is None):
                logger.warning(f"  {name:<15} | Model not loaded   | Status: ⚠ SKIP")
                continue

            try:
                total_psnr = 0.0
                total_ssim = 0.0
                total_lpips = 0.0
                total_time = 0.0
                eval_count = 0

                for pat_path in test_patient_dirs:
                    pa_file = os.path.join(pat_path, "pa.png")
                    views_dir = os.path.join(pat_path, "views")
                    if not (os.path.exists(pa_file) and os.path.exists(views_dir)):
                        continue

                    # Load real PA input X-ray
                    pa_img = Image.open(pa_file).convert("L")
                    input_xr = TF.to_tensor(pa_img).unsqueeze(0).to(self.device)

                    # List ground truth view images
                    gt_files = sorted([f for f in os.listdir(views_dir) if f.endswith(".png")])
                    if not gt_files:
                        continue

                    start_time = time.time()
                    # Infer multi-degree views matching ground truth view count
                    num_views = len(gt_files)
                    pred_views = wrapper.infer_multi_views(input_xr, azimuths=(0, 360, num_views))
                    elapsed = time.time() - start_time

                    if not pred_views:
                        continue

                    pat_psnr = 0.0
                    pat_ssim = 0.0
                    pat_lpips = 0.0
                    valid_views = min(len(pred_views), len(gt_files))

                    for v_idx in range(valid_views):
                        pred_v = pred_views[v_idx]
                        if pred_v.dim() == 2:
                            pred_v = pred_v.unsqueeze(0).unsqueeze(0)
                        elif pred_v.dim() == 3:
                            pred_v = pred_v.unsqueeze(0)
                        pred_v = pred_v.to(self.device)

                        gt_img_path = os.path.join(views_dir, gt_files[v_idx])
                        gt_img = Image.open(gt_img_path).convert("L")
                        gt_v = TF.to_tensor(gt_img).unsqueeze(0).to(self.device)

                        # Resize target if needed
                        if pred_v.shape != gt_v.shape:
                            pred_v = TF.resize(pred_v, [gt_v.shape[-2], gt_v.shape[-1]])

                        pred_v_norm = normalize_tensor(pred_v)
                        gt_v_norm = normalize_tensor(gt_v)

                        pat_psnr += compute_psnr(pred_v_norm, gt_v_norm)
                        pat_ssim += compute_ssim(pred_v_norm, gt_v_norm)
                        pat_lpips += compute_lpips(pred_v_norm, gt_v_norm)

                    total_psnr += (pat_psnr / max(1, valid_views))
                    total_ssim += (pat_ssim / max(1, valid_views))
                    total_lpips += (pat_lpips / max(1, valid_views))
                    total_time += elapsed
                    eval_count += 1

                if eval_count == 0:
                    raise ValueError("No valid patient test cases found for evaluation")

                avg_psnr = total_psnr / eval_count
                avg_ssim = total_ssim / eval_count
                avg_lpips = total_lpips / eval_count
                avg_time = total_time / eval_count

                computed_results.append((name, avg_psnr, avg_ssim, avg_lpips, avg_time))
                logger.info(
                    f"  {name:<15} | PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.3f} | "
                    f"LPIPS: {avg_lpips:.3f} | Avg Time: {avg_time:.1f}s | Status: ✓ PASS"
                )

            except Exception as e:
                logger.error(f"  {name:<15} | Status: ✗ FAIL - {str(e)[:80]}")

        # Output final metrics table (presented on terminal for user visibility)
        print("\n" + "=" * 80)
        print("📊 PHASE 1 OUT-OF-DISTRIBUTION (OOD) METRICS - NSCLC TEST SPLIT")
        print("=" * 80)
        
        print(f"\n{'Method':<35} {'PSNR (dB) ↑':<13} {'SSIM ↑':<9} {'LPIPS ↓':<9} {'Time/Pat ↓':<10}")
        print("-" * 85)
        for method, psnr, ssim, lp, t in computed_results:
            print(f"{method:<35} {psnr:>10.2f}      {ssim:>9.3f}  {lp:>8.3f}  {t:>8.1f}s")
        print("=" * 80 + "\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Unified Baseline Evaluation Harness")
    parser.add_argument(
        "--max_patients", type=int, default=None,
        help="Cap the number of test patients evaluated per baseline (default: None, full 241-patient NSCLC split)",
    )
    args = parser.parse_args()

    evaluator = UnifiedBaselineEvaluator()
    evaluator.run_evaluation_suite(max_patients=args.max_patients)
