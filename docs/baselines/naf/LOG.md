# NAF Change & Fix Log

## 2026-08-03 — Initial NAF Integration & Beer-Lambert Physics Correction

**Phase:** 1 & 2 (Import & Correctness)  
**Files:** `baselines/models/naf.py`, `baselines/train/naf.py`  
**Change:** Created `NAFWrapper` and `train_naf` training script. Applied `apply_beer_lambert_correction(density)` ($I = I_0 \exp(-A)$) to integrated density rays.  
**Why:** NAF predicts linear attenuation density $\mu$. Comparing linear attenuation directly against transmissive intensity target $I$ violated Beer-Lambert law.  
**Verification:** Verified in `baselines/tests/test_loss_regression.py` (`test_naf_fit_density_field_uses_beer_lambert`).

---

## 2026-08-05 — PyTorch 2.x CUDA Compile Fix, Perspective Ray Marching & Azimuth Alignment

**Phase:** 1, 2 & 3 (Import, Geometry, Guardrails & Acceleration)  
**Files:** `baselines/cloned/naf_cbct/src/encoder/hashencoder/src/hashencoder.cu`, `hashgrid.py`, `baselines/models/naf.py`, `baselines/train/naf.py`  
**Change:** Fixed `hashencoder.cu` by replacing deprecated `.type()` calls with `.scalar_type()` and added `_backend = get_backend()` in `hashgrid.py`. Replaced flat `rotate_volume_3d(...)` orthographic depth collapse with `perspective_ray_march`/`build_perspective_ray_points` (`baselines/models/utils.py`). Added `safe_bound = bound * (1.0 - 1e-6)` margin to prevent float32 rounding errors at `bound=0.3`. Pre-built `fit_pts` outside fitting loop, added `--val_iters_per_sample`, and switched target angle generation from `endpoint=False` to `endpoint=True`.  
**Why:** Enables Instant-NGP HashEncoder CUDA C++ extension compilation on PyTorch 2.x/CUDA 12+, aligns ray-marching with DiffDRR perspective camera geometry, prevents boundary validation crashes, accelerates per-scan fitting, and aligns azimuth angles with ground-truth DiffDRR projections (`views/*.png`).  
**Verification:** `HashEncoder` CUDA extension compiles cleanly, `NAF_HASHENCODER_AVAILABLE` evaluates to `True`, and `perspective_ray_march` executes error-free.
