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

---

## 2026-08-06 — Docker Base Image CUDA/torch Version Mismatch (Same Root Cause as XRaySyn's Stub Fallback)

**Phase:** 1 (Import / Dependency-Reality Correctness)
**Files:** `baselines/Dockerfile`
**Change:** Bumped the base image from `nvidia/cuda:12.8.0-devel-ubuntu22.04` to `nvidia/cuda:13.0.3-devel-ubuntu22.04`.
**Why:** While investigating why `baselines/cloned/XraySyn/xraysyn/networks/drr_projector`'s CUDA extension
failed to build in `baselines/Dockerfile`'s container (see `docs/baselines/xraysyn/LOG.md`'s 2026-08-06
entry), found the actual blocker was a hard `nvcc`/`torch` CUDA version mismatch: `requirements.txt` doesn't
pin `torch`, so `uv pip install -r requirements.txt` resolves whatever the current default PyPI wheel is
(a `cu130` build, matching the local dev machine), while the container's `nvcc` came from the base image's
CUDA 12.8 toolkit -- `torch.utils.cpp_extension`'s `_check_cuda_version` hard-fails any custom extension
build on this mismatch (`RuntimeError: The detected CUDA version (12.8) mismatches the version that was
used to compile PyTorch (13.0)`). This affects NAF too, not just XRaySyn: `naf.py`'s `HashEncoder` import
(`baselines/cloned/naf_cbct/src/encoder/hashencoder/hashgrid.py`'s `get_backend()`) JIT-compiles via the
same `torch.utils.cpp_extension.load()` machinery, at runtime rather than Docker build time, so it would
hit the identical version-mismatch error the first time a training run actually imports `HashEncoder`
inside this container -- silently downgrading to `FreqEncoder` via `naf.py`'s own fallback rather than
crashing, so this failure mode is easy to miss (no visible error, just a degraded-quality encoder) unless
specifically checked for, exactly like XRaySyn's stub fallback was.
**Verification:** Local `docker build` of the bumped Dockerfile (see this same date's entry in
`docs/baselines/xraysyn/LOG.md` for the XRaySyn extension's own build verification) is the verification
for this entry too, since both extensions depend on the same base-image/torch CUDA alignment. Did not
independently re-verify `NAF_HASHENCODER_AVAILABLE` inside the rebuilt container specifically (would
require actually running `baselines/train/naf.py` inside it) -- worth confirming on the next real NAF
training run's log output rather than assuming.
