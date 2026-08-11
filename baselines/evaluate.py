"""Unified Multi-Model Evaluation Harness for CosmosXRay360."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
import warnings
from PIL import Image

import torch
import torchvision.transforms.functional as TF

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from models import (
    Dx2CTWrapper,
    MedNeRFWrapper,
    NAFWrapper,
    PixelNeRFWrapper,
    SVDRRWrapper,
    XRaySynWrapper,
    compute_lpips,
    compute_psnr,
    compute_ssim,
    normalize_tensor,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class UnifiedBaselineEvaluator:
    """Comprehensive evaluation harness for all 6 baselines."""

    def __init__(self, checkpoint_dir: str = "baselines/checkpoints") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_dir = checkpoint_dir
        self.baselines: dict[str, object] = {}

        if self.device == "cuda":
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        logger.info("Initializing baseline wrappers...")
        self._init_baselines()

    def _resolve_ckpt(self, prefix: str) -> str:
        for name in [f"{prefix}_best.pt", f"{prefix}_surrogate_checkpoint.pt", f"{prefix}_checkpoint.pt"]:
            path = os.path.join(self.checkpoint_dir, name)
            if os.path.exists(path):
                logger.info(f"Using checkpoint for {prefix}: {path}")
                return path
        return os.path.join("eval_outputs", f"{prefix}_checkpoint.pt")

    def _init_baselines(self) -> None:
        wrapper_cls = {
            "SV-DRR": (SVDRRWrapper, "svdrr"),
            "XRaySyn": (XRaySynWrapper, "xraysyn"),
            "MedNeRF": (MedNeRFWrapper, "mednerf"),
            "PixelNeRF": (PixelNeRFWrapper, "pixelnerf"),
            "NAF": (NAFWrapper, "naf"),
            "Dx2CT": (Dx2CTWrapper, "dx2ct"),
        }
        for name, (cls, prefix) in wrapper_cls.items():
            ckpt_path = self._resolve_ckpt(prefix)
            self.baselines[name] = cls(ckpt_path)

    def _ensure_test_data(self, test_dir: str) -> None:
        if not os.path.exists(test_dir) or len(os.listdir(test_dir)) < 50:
            logger.info("Test dataset missing or incomplete. Downloading & pre-rendering...")
            fast_dl = os.path.join(BASE_DIR, "..", "scripts", "fast_download.py")
            pre_render = os.path.join(BASE_DIR, "..", "datasets", "pre_render_diffdrr.py")
            subprocess.run([sys.executable, fast_dl, "--percentage", "1.0"], check=True)
            subprocess.run([sys.executable, pre_render], check=True)

    def run_evaluation_suite(self, max_patients: int | None = None) -> None:
        """Executes full evaluation across all baselines on the test split."""
        logger.info("=" * 80)
        logger.info("🚀 Launching Unified Baseline Evaluation Harness")
        logger.info("=" * 80)

        test_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "datasets", "pre_rendered", "test"))
        self._ensure_test_data(test_dir)

        test_dirs = sorted([
            os.path.join(test_dir, d) for d in os.listdir(test_dir)
            if os.path.isdir(os.path.join(test_dir, d))
        ])
        if max_patients is not None:
            test_dirs = test_dirs[:max_patients]

        logger.info(f"Loaded {len(test_dirs)} test patient cases from {test_dir}\n")

        results: list[tuple[str, float, float, float, float]] = []

        for name in ["XRaySyn", "MedNeRF", "Dx2CT", "SV-DRR", "PixelNeRF", "NAF"]:
            wrapper = self.baselines.get(name)
            if not wrapper or (getattr(wrapper, "model", None) is None and getattr(wrapper, "pipe", None) is None):
                logger.warning(f"  {name:<15} | Model not loaded | Status: ⚠ SKIP")
                continue

            try:
                psnr, ssim, lpips_val, avg_time = self._eval_baseline(wrapper, test_dirs)
                results.append((name, psnr, ssim, lpips_val, avg_time))
                logger.info(
                    f"  {name:<15} | PSNR: {psnr:.2f} dB | SSIM: {ssim:.3f} | "
                    f"LPIPS: {lpips_val:.3f} | Time: {avg_time:.1f}s | Status: ✓ PASS"
                )
            except Exception as e:
                logger.error(f"  {name:<15} | Status: ✗ FAIL - {str(e)[:80]}")

        self._print_results(results)

    def _eval_baseline(
        self, wrapper: object, test_dirs: list[str]
    ) -> tuple[float, float, float, float]:
        total_psnr, total_ssim, total_lpips, total_time = 0.0, 0.0, 0.0, 0.0
        eval_count = 0

        for pat_path in test_dirs:
            pa_file = os.path.join(pat_path, "pa.png")
            views_dir = os.path.join(pat_path, "views")
            if not (os.path.exists(pa_file) and os.path.exists(views_dir)):
                continue

            pa_img = Image.open(pa_file).convert("L")
            input_xr = TF.to_tensor(pa_img).unsqueeze(0).to(self.device)

            gt_files = sorted([f for f in os.listdir(views_dir) if f.endswith(".png")])
            if not gt_files:
                continue

            start_time = time.time()
            pred_views = wrapper.infer_multi_views(input_xr, azimuths=(0, 360, len(gt_files)))
            total_time += time.time() - start_time

            if not pred_views:
                continue

            pat_psnr, pat_ssim, pat_lpips = 0.0, 0.0, 0.0
            valid_views = min(len(pred_views), len(gt_files))

            for v_idx in range(valid_views):
                pred_v = pred_views[v_idx]
                while pred_v.dim() < 4:
                    pred_v = pred_v.unsqueeze(0)
                pred_v = pred_v.to(self.device)

                gt_img = Image.open(os.path.join(views_dir, gt_files[v_idx])).convert("L")
                gt_v = TF.to_tensor(gt_img).unsqueeze(0).to(self.device)

                if pred_v.shape != gt_v.shape:
                    pred_v = TF.resize(pred_v, [gt_v.shape[-2], gt_v.shape[-1]])

                pred_norm, gt_norm = normalize_tensor(pred_v), normalize_tensor(gt_v)
                pat_psnr += compute_psnr(pred_norm, gt_norm)
                pat_ssim += compute_ssim(pred_norm, gt_norm)
                pat_lpips += compute_lpips(pred_norm, gt_norm)

            total_psnr += pat_psnr / valid_views
            total_ssim += pat_ssim / valid_views
            total_lpips += pat_lpips / valid_views
            eval_count += 1

        if eval_count == 0:
            raise ValueError("No valid test cases evaluated")

        return (
            total_psnr / eval_count,
            total_ssim / eval_count,
            total_lpips / eval_count,
            total_time / eval_count,
        )

    def _print_results(self, results: list[tuple[str, float, float, float, float]]) -> None:
        print("\n" + "=" * 80)
        print("📊 PHASE 1 OUT-OF-DISTRIBUTION (OOD) METRICS - NSCLC TEST SPLIT")
        print("=" * 80)
        print(f"\n{'Method':<20} {'PSNR (dB) ↑':<13} {'SSIM ↑':<9} {'LPIPS ↓':<9} {'Time/Pat ↓':<10}")
        print("-" * 65)
        for method, psnr, ssim, lp, t in results:
            print(f"{method:<20} {psnr:>10.2f}      {ssim:>9.3f}  {lp:>8.3f}  {t:>8.1f}s")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Baseline Evaluation Harness")
    parser.add_argument(
        "--max_patients", type=int, default=None,
        help="Cap the number of test patients evaluated per baseline",
    )
    args = parser.parse_args()

    evaluator = UnifiedBaselineEvaluator()
    evaluator.run_evaluation_suite(max_patients=args.max_patients)

