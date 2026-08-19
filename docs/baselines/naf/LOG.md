# NAF Change & Fix Log

## 2026-08-16 — Seeded Off-Axis Validation View Sampling

**Phase:** 2 (Training & Validation Parity)  
**Files:** `baselines/train/naf.py`  
**Change:** Updated `train_naf.py` validation loop to use `val_rng = np.random.RandomState(42)` for selecting deterministic off-axis validation target views (`val_pairs`) instead of hard-pinning index 0 ($0^\circ$).  
**Why:** Ensures validation loss (`avg_val_loss`) accurately measures off-axis novel-view fitting quality during training, matching SV-DRR's seeded validation protocol.  
**Verification:** All unit and regression tests pass cleanly.

---

## 2026-08-14 — Fixed Baked-Grid Axis-Swap Bug in `sample_fn`; Confirmed Residual Comb Artifact Is a Method Limitation, Not a Bug

**Phase:** 2 (Correctness)
**Files:** `baselines/models/naf.py`
**Change:** `infer_multi_views`'s `sample_fn(pts)` (queries the dense-baked `attenuations` grid via
`F.grid_sample`) now permutes `pts` to `pts[..., [2, 1, 0]]` before normalizing/clamping.
**Why:** `attenuations` is baked via `torch.stack(torch.meshgrid(grid, grid, grid, indexing="ij"), dim=-1)`
stacked in `(x_world, y_world, z_world)` order, giving the 5D tensor axes `(D=x_world, H=y_world,
W=z_world)`. `F.grid_sample` reads `grid[...,0]→W`, `[...,1]→H`, `[...,2]→D` — so an unpermuted
`(x,y,z)` query lands `x_world` on the `D` axis and `z_world` on the `W` axis, i.e. exactly backwards.
Found this independently while investigating why `results/naf.png` (Images/val_multiview panel) looked
like radiating comb-pattern noise instead of chest anatomy — the same class of bug a concurrent Copilot
pass had just (incorrectly, see `docs/baselines/dx2ct/LOG.md`'s same-date entry) tried to fix in
`dx2ct.py`.
**Verification:** Empirical, not just derived: baked a synthetic single-voxel marker tensor at a known
`(i0,j0,k0)`, queried `sample_fn` at the exact world-coordinate the marker was placed at — the unpermuted
query returned `0.0` (missed the marker entirely), the `[2,1,0]`-permuted query returned `~1.0` (correct
hit). `uv run pytest baselines/tests/test_naf_wrapper.py` still passes.
**Residual finding (not a bug, left as-is):** After this fix, `infer_multi_views` on a real NSCLC test
patient (no checkpoint needed — NAF fits per-scan) still shows a severe "comb"/interference artifact at
azimuths near the single fit view (0°/~4°/~190°), while azimuths far from it show smooth-but-wrong
banding. Ruled out three alternative causes by direct experiment before concluding this: (1) bypassing the
baked grid entirely and querying the coordinate MLP directly per ray sample — pattern unchanged; (2)
raising per-scan fit iterations 200→1500 — pattern unchanged (got slightly sharper, not weaker); (3)
matching the render grid resolution to the fit resolution (128→64) — pattern reduced but not eliminated.
A direct self-consistency check (render azimuth=0 immediately after fitting to it) reproduces the fit
target reasonably well (MSE ≈0.0027), so the field *does* fit its one supervised view — it just collapses
to a thin, nearly-2D density slab at that view (the classic shape-radiance/thin-slab degeneracy for a
coordinate field fit from a *single* 2D projection with no cross-view constraint), and any ray that isn't
close to parallel to the fit view grazes that slab at a steep angle, producing dense fringing. This matches
`research-state.yaml`'s own H1 hypothesis (reconstruction-based NVS baselines suffer severe artifacts on
single-view OOD input) and NAF's intended use case (CBCT reconstruction from dozens of angularly-diverse
projections, not one) — not something a further code fix in this wrapper should chase.
**Follow-up same day:** pulled `gs://.../checkpoints/cosmos-worker-1/naf_best.pt` (57MB) from GCS and re-ran
against the same real patient, initializing the per-scan fit from these pretrained weights instead of
`NAFWrapper`'s default random init. Output is visually indistinguishable from the random-init run — expected,
since `infer_multi_views` always re-fits ~200 gradient steps per scan regardless of the starting point, and
both converge to the same single-view thin-slab degeneracy. Confirms the checkpoint choice is irrelevant to
this baseline's quality (unlike the other 5), consistent with NAF's architecture.

---

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
---

## 2026-08-15 — Target Azimuth Sampling Across Full 93-View Cache

**Phase:** 2 (Training Protocol / View Representation)  
**Files:** `baselines/models/naf.py`, `baselines/train/naf.py`  
**Change:** Updated `fit_density_field` to take `azimuth: float = 0.0`. Updated `train_naf` to load pre-rendered views and angles and sample random target view angles `(target_proj, target_azimuth)` during density field fitting.  
**Why:** Prevents per-scan density field fitting from overfitting to 0° frontal projection during training, optimizing NAF rays across the full $360^\circ$ sweep.  
**Verification:** `python run_all_tests_isolate.py` passes all unit tests.
