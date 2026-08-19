# XRaySyn Change & Fix Log

## 2026-08-15 — Compared Against Cosmos-NVSyn's Reference Implementation: No Real Training Gaps Found

**Phase:** 2 (Correctness verification)
**Files:** none changed -- comparison only
**Change:** After finding real training-methodology gaps for SV-DRR against its
`Cosmos-NVSyn/model/nvsyn_svdrr.py` sibling reference, did the same diff for XRaySyn against
`Cosmos-NVSyn/model/nvsyn_xraysyn.py`. Unlike SV-DRR (a from-scratch training loop this project wrote),
XRaySyn's training is almost entirely delegated to `baselines/cloned/XraySyn/xraysyn/models/
ct2xray_real_gan_meta.py`'s own `XraySynModel.optimize()` -- the *original* paper repo's own training
method, not something this project reimplemented -- so the comparison is really "does the vendored
`optimize()` match the reference," not "does this project's code match it."
**Result -- already matching, nothing to port**: `optimize()` already has: the same LSGAN discriminator
loss (`GANLoss(gan_mode='lsgan')` == the reference's MSE-vs-ones/zeros formulation), the same loss
weights (`0.02` GAN, `0.005` sparse, conditionally applied when `loss_sparse.item() > 1`, both matching
the reference's `gan_weight=0.02, sparse_weight=0.005` exactly), the same `torch.nn.utils.
clip_grad_norm_(..., max_norm=1.0)` gradient clipping on both the discriminator and generator (already
added 2026-08-07, see below), the same `not torch.isnan(loss)` guard before each optimizer step, and
the same Adam `beta1=0.5` (paired with the reference's `beta1=0.5, beta2=0.9`). Neither version trains
directly against paired novel-view ground truth in the loss (the reference computes `novel_l1` only for
logging when `image_target` happens to be available; this project's training loop never provides one
either) -- consistent, not a gap.
**One real, but already-deliberate, difference found**: the reference clamps `atten_proj` symmetrically
to `[-10.0, 10.0]` before `exp()`; the vendored code clamps only the upper bound (`max=50.0`), per an
existing in-code comment explaining this project's 360-degree training range produces larger `atten_proj`
magnitudes than the original paper's +/-9-degree range, so a tighter +/-10 bound would clip too
aggressively here. This is a reasoned, already-documented divergence (see the 2026-08-07 entry below),
not an oversight -- flagging it as low-priority (a missing lower clamp risks `exp()` underflowing toward
0 on very negative inputs, not the overflow/NaN risk the missing *upper* bound would have posed) rather
than porting it.
**Verification**: direct read of `ct2xray_real_gan_meta.py`'s `__init__`/`ct2xray`/`optimize()` against
`nvsyn_xraysyn.py`'s `__init__`/`training_step`/`configure_optimizers`, line-by-line on the loss
composition, optimizer config, and clipping/NaN-guard logic.

---

## 2026-08-14 — Reverted the Vertical-Flip "Fix" Below: It Was Actually Backwards

**Phase:** 2 (Correctness verification — correcting a same-day entry)
**Files:** `baselines/models/xraysyn.py`
**Change:** Removed `proj = torch.flip(proj, dims=[-2])` from `infer_multi_views`, reverting to the
unflipped output the entry directly below this one had just replaced.
**Why:** The entry below claims this flip was empirically verified against a real checkpoint and GT.
It wasn't verified correctly. User flagged that a freshly-regenerated `results/xraysyn.png` (with the
flip active) still looked visually wrong, which prompted a re-check using a stronger ground truth
anchor than `views/*.png`: the raw, wrapper-untouched `pa.png` input itself (azimuth ≈0° should
closely match it, and zero wrapper logic sits between disk and that file, so it can't itself be
mis-oriented by anything this wrapper does). Row-wise brightness-profile correlation against `pa.png`:
**without** the flip, `corr(pa, pred) = 0.81` (strong match); **with** the flip (the staged/just-reverted
state), `corr(pa, pred) = 0.02` (no match, effectively scrambled). Direct visual comparison confirms
the same thing — without the flip, the near-0° prediction has clavicles/apex at the top and a bright
diaphragm/abdomen band at the bottom, matching `pa.png`; with the flip, that's inverted. The entry
below's own "without the flip it's upside down" claim does not hold up against this stronger anchor —
most likely that verification pass compared against a mis-oriented reference or mislabeled which grid
row was prediction vs. ground truth; the exact cause of the earlier mistake wasn't identified, only
that the conclusion was wrong.
**Verification:** Quantitative (row-profile correlation against the unambiguous raw input, not just a
visual read of `views/*.png`) plus direct visual comparison. Full 8-view/3-patient panel regenerated via
`scripts/regenerate_tb_images.py --baseline xraysyn --checkpoint baselines/checkpoints/xraysyn_best.pt`:
near-frontal columns (~0-45°, ~315-360°) now closely match ground truth; oblique columns (~90-270°)
remain blurry/hallucinated, the same genuinely-OOD-single-view failure mode already documented for
Dx2CT (`docs/baselines/dx2ct/LOG.md`'s 2026-08-14 entry), not an orientation problem. `results/xraysyn.png`
and `baselines/checkpoints/tensorboard_fixed/xraysyn/` updated with the corrected panel.
**Lesson:** a LOG.md entry claiming something was "empirically verified" is not itself proof — this
project's own read-first discipline applies to its own past entries too, not just to a paper/repo being
read for the first time. Re-derive from the strongest available ground truth anchor when a result looks
suspicious, rather than trusting a prior conclusion because it was labeled "verified."

---

## 2026-08-14 — Empirically Confirmed a Staged Vertical-Flip Fix Against a Real Trained Checkpoint
**⚠️ Superseded by the correction entry directly above — this conclusion was wrong.**

**Phase:** 2 (Correctness verification)
**Files:** `baselines/models/xraysyn.py` (no further change — verifying an already-staged, uncommitted fix)
**Change:** None to this file; confirmed the staged `proj = torch.flip(proj, dims=[-2])` line (added
before `results.append(proj)` in `infer_multi_views`) is correct rather than assuming it from its comment.
**Why verified, not just trusted:** Ran the wrapper against a real trained checkpoint
(`baselines/checkpoints/xraysyn_best.pt`, completed locally today) and a freshly-DiffDRR-rendered real
NSCLC test patient (`LUNG1-001_0000`, via `datasets/pre_render_diffdrr.py`), both with and without the
flip (temporarily monkeypatching `torch.flip` to a no-op for the comparison). Without the flip, the
near-frontal (~4°) predicted view is a chest X-ray rendered upside down (diaphragm curve at the top,
apex/ribcage at the bottom) relative to ground truth. With the flip, the same view is right-side up and
recognizably matches the ground-truth PA orientation. Confirms `DRRProjector`'s internal row convention is
inverted relative to this project's `views/*.png` convention, exactly as the staged comment claimed.
**Verification:** Direct visual before/after comparison against real GT (see session images); `uv run
pytest baselines/tests/test_xraysyn_wrapper.py` passes.

---

## 2026-08-03 — Initial XRaySyn Integration & Dynamic Pose Helpers

**Phase:** 1 (Import & Environment)  
**Files:** `baselines/models/xraysyn.py`, `baselines/train/xraysyn.py`  
**Change:** Created `XRaySynWrapper` and `train_xraysyn` training script wrapping `XraySynModel` from `baselines/cloned/XraySyn/`. Implemented `_get_T_batched` and `_get_T_multi` pose matrix helpers and wrapped model initialization with a temporary `os.chdir(xraysyn_dir)` block.  
**Why:** Fixes `XraySynModel.get_T`'s hardcoded batch size of 4 (`torch.cat([T, T, T, T])`) to support arbitrary batch sizes, and allows loading `simplified_bone_absorb_2d.pt` / `simplified_tissue_absorb_2d.pt` from any working directory.  
**Verification:** Tested `XRaySynWrapper.infer_multi_views` across single and multi-batch inputs without shape or file errors.

---

## 2026-08-05 — Paper Cached & PAPER.md/CODE.md Corrected Against It

**Phase:** 0 (Resource acquisition / correctness of docs, not code)
**Files:** `docs/baselines/xraysyn/{PAPER.md,CODE.md}`, `arxiv_papers/XraySyn_2012.02407.pdf`
**Change:** The 2026-08-05 docs backfill pass initially wrote `PAPER.md` without actually
downloading/reading the paper — it cited the wrong title ("Reconstructing 3D CT Volumes from Single
2D X-ray Images using Physics-Supervised Generative Adversarial Networks", which does not exist) and
described an architecture that doesn't match the paper's real two-stage 3DPN/2DRN + CT2Xray design.
`CODE.md` separately mis-stated the repo's license as "Custom research license (Johns Hopkins
University)". Both were caught during a same-day review, the real paper (arXiv:2012.02407, "XraySyn:
Realistic View Synthesis From a Single Radiograph Through CT Priors") was downloaded into
`arxiv_papers/`, and `PAPER.md`/`CODE.md` were rewritten directly from it and cross-checked against
`baselines/cloned/XraySyn/`'s actual source (`LICENSE` is plain MIT; `DRRProjector`, `UnetGenerator`,
`NLayerDiscriminator` constructor signatures confirmed via `grep`).
**Why:** generic-pattern.md's Phase 0 rule is explicit: "never work from memory of what a method
probably does; open the PDF." This is exactly the failure mode that rule exists to prevent, and it's
worth keeping as a log entry rather than silently rewriting, since it's evidence the read-the-PDF
step needs to be checked, not assumed, when reviewing future backfills (including ones done by other
tools/agents).
**Verification:** New title/authors/venue/arXiv ID/equation numbers cross-checked against the
downloaded PDF page-by-page; license claim verified against the actual `LICENSE` file; all class/
kwarg names in the "Key Modules" mapping re-verified with `grep` against the real source files
(`drr_projector_new.py`, `unet.py`, `common.py`) rather than asserted from memory.

---

## 2026-08-05 — Widened Angular Training Range & Azimuth Alignment

**Phase:** 2 (Training Scope / Fairness & Convention Alignment)  
**Files:** `baselines/train/xraysyn.py`, `baselines/models/xraysyn.py`  
**Change:** Expanded `OTHER_POSE_THETA_Y_RANGE` in `train/xraysyn.py` from `(-0.05, 0.05)` ($\pm 9^\circ$) to `(-1.0, 1.0)` (full $360^\circ$ sweep). Switched target angle generation in `models/xraysyn.py` from `endpoint=False` to `endpoint=True`.  
**Why:** Official demo script trained `net2d` refinement on a $\pm 9^\circ$ angular neighborhood, creating a train/eval mismatch when benchmark evaluation queries $360^\circ$ novel views. Endpoint alignment matches ground-truth DiffDRR projections (`views/*.png`).  
**Verification:** Verified `net2d` trains across full angle range and `infer_multi_views` indices align with ground-truth view files.

---

## 2026-08-06 — Fixed Silent CUDA Extension Build Failure (All-Zero Projector Stub in Production)

**Phase:** 1 & 2 (Import / Dependency-Reality Correctness)
**Files:** `baselines/cloned/XraySyn/xraysyn/networks/drr_projector/setup.py`, `baselines/Dockerfile`
**Change:** `setup.py`'s `extra_link_args` hardcoded a CUDA runtime rpath at
`sys.prefix/.../site-packages/nvidia/cu13/lib` (matching the local dev machine's `torch==2.13.0+cu130`
install exactly) instead of discovering it dynamically. Replaced with a `glob.glob(".../nvidia/cu*/lib/libcudart.so*")`
lookup that resolves to whatever CUDA-flavored torch wheel is actually installed, with a safe fallback
(omit the rpath flag entirely, don't crash) if none is found. Also added `-gencode arch=compute_89,code=sm_89`
alongside the existing `arch=compute_75,code=sm_75`, since the prior build only embedded a Titan-RTX
(sm_75) cubin with no PTX fallback -- it would have failed to *launch* on an L4 GPU (`sm_89`,
`scripts/launch_parallel_vms.sh`'s cloud training target) even after fixing the rpath. Separately,
`baselines/Dockerfile`'s `RUN ... build_ext --inplace || true` now checks the exit code and prints an
unmissable warning banner on failure instead of swallowing it silently.
**Why:** `baselines/Dockerfile` is built `FROM nvidia/cuda:12.8.0-devel-ubuntu22.04` with an unpinned
`uv pip install -r requirements.txt`, which resolves a different (non-cu13) CUDA-flavored torch wheel
than the local dev machine -- so the hardcoded rpath pointed at a directory that doesn't exist in the
container. The resulting `ImportError` was caught by `xraysyn.py`'s own dependency-fallback logic and
silently downgraded to the all-zero-projector stub, with the `|| true` in the Dockerfile hiding the
build failure entirely. Found 2026-08-06 while investigating a live GCP training run: **all 3 worker
VMs** (`cosmos-worker-1/2/3`) had been running `XraySynWrapper` against the fake stub for their entire
session -- worker-2's XRaySyn training (which also independently diverged to NaN loss from epoch 2
onward, a separate issue) was training against a projector that was a no-op regardless. This is exactly
the "dependency reality check" this project's `baseline-experiment` skill's Phase 2 checklist calls
for, except it had only ever been verified on the local machine, never inside the actual Docker image
the cloud training path uses -- the checklist item needs to be re-run per-environment, not just once.
**Verification:** Could not fully verify end-to-end locally: the local machine's active `nvcc` (12.4)
mismatches its own `torch` build (`cu130`), a pre-existing environment issue unrelated to this fix,
which blocks `torch.utils.cpp_extension`'s built-in version check from even attempting a build here.
Verified what's possible in isolation instead: (1) the new glob logic resolves correctly against the
local `nvidia/cu13/lib` layout; (2) a raw `nvcc` compile of `dp_trilinear_cuda.cu` with both
`-gencode` flags (`sm_75` and `sm_89`) succeeds cleanly (only pre-existing `.data<T>()` deprecation
warnings, no errors), confirming the kernel source itself is valid for both target architectures. Full
verification (Docker build -> import -> real kernel launch on an L4) still needs either a Docker build
test or a real worker relaunch -- not yet done as of this entry.

---

## 2026-08-07 — Correction: The CUDA-Extension Fix Above Unmasked a Real exp() Overflow, Now Fixed

**Phase:** 2 (Correctness -- regression found and fixed after relaunching on the real, fixed projector)
**Files:** `baselines/cloned/XraySyn/xraysyn/models/ct2xray_real_gan_meta.py`
**Change:** Added `atten_proj = torch.clamp(atten_proj, max=50.0)` before `torch.exp(atten_proj)` in both
`ct2xray()` and `mat2xray()`.
**Why:** After the 2026-08-06 CUDA extension fix landed and the real (non-stub) `drr_projector_function`
started running on a live cloud worker, XRaySyn's training loss went `nan` starting at epoch 1 (worse
than the earlier stub-era run, which stayed clean through epoch 1 and only diverged at epoch 2 -- see
that entry above). Root cause: `self.bone_absorb`/`self.tissue_absorb` (`ct2xray_real_gan_meta.py:41-44`)
are not learned/adapted networks -- they're frozen, pickled physical calibration functions
(`simplified_bone_absorb_2d.pt`/`simplified_tissue_absorb_2d.pt`, `requires_grad=False`), and their
summed output (`atten_proj`) feeds straight into an uncapped `torch.exp()` with no clamp anywhere in
the original vendored code. With the all-zero stub, `bone_proj`/`tissue_proj` were always exactly zero,
so `atten_proj` was always bounded regardless of camera angle -- masking the missing clamp entirely. The
real projector, combined with this project's own 2026-08-05 fix widening `OTHER_POSE_THETA_Y_RANGE` to
the full 360° sweep (the official paper/demo only ever queried ±9°), now queries the frozen calibration
functions on camera geometries they were never calibrated for, producing `atten_proj` values large
enough to overflow `exp()` to `inf`, which then propagates to `nan` through `self.norm(out_new.max() -
out_new)` (`inf - inf`). This is the exact same physics/value-space bug class this project's own
`apply_beer_lambert_correction` (`baselines/models/utils.py`) already guards against with an identical
`torch.clamp(density, min=0.0, max=50.0)` before its own `exp()` -- the clamp constant (`max=50.0`) is
reused here for consistency, not independently re-derived.
**Verification:** `python3 -m py_compile` on the edited file. Not yet re-verified with a real training
run at time of this entry -- relaunching `cosmos-worker-2` (XRaySyn only, via
`scripts/launch_parallel_vms.sh --override cosmos-worker-2=xraysyn`) to confirm the loss no longer goes
`nan`; update this entry once epoch-level loss values are observed.

---

## 2026-08-11 — Fixed Zero-Range Division NaN in NormLayer and Added Gradient Clipping

**Phase:** 2 (Training Stability & Correctness)  
**Files:** `baselines/cloned/XraySyn/xraysyn/utils/torch.py`, `baselines/cloned/XraySyn/xraysyn/networks/rdn_meta.py`, `baselines/cloned/XraySyn/xraysyn/loaders/ct2xray_real.py`, `baselines/cloned/XraySyn/xraysyn/models/ct2xray_real_gan_meta.py`  
**Change:**  
1. Added `+ 1e-8` epsilon to `inp / (denom + 1e-8)` inside `NormLayer.forward`, `NormToLayer.forward`, and custom `norm()` methods across `torch.py`, `rdn_meta.py`, and `ct2xray_real.py`.  
2. Added `torch.nn.utils.clip_grad_norm_` (max_norm=1.0) and `not torch.isnan(loss)` checks in `XraySynModel.optimize()` before `self.optimD.step()` and `self.optimG.step()`.  
**Why:**  
Inspecting `xraysyn.csv` from the 100-epoch `cosmos-worker-2` run showed `L1` loss decreasing from $0.361$ (epoch 1) to a minimum of $0.041$ (epoch 6), rebounding slightly to $0.068$ (epoch 8), and turning into `NaN` starting at epoch 9 through epoch 100.  
*Analysis & Context:* The `xraysyn.csv` run predated the deployment of the 2026-08-07 `atten_proj` clamp fix (`torch.clamp(atten_proj, max=50.0)`). Thus, the historical `NaN` at epoch 9 could stem from two potential root causes (or their combination):  
  - **Flat feature map division by zero:** `out_new = self.norm(out_new.max() - out_new)` in `ct2xray()` / `mat2xray()` passes $0.0$ when `out_new` becomes spatially uniform, causing `NormLayer.forward` to compute $0.0 / 0.0 = \text{NaN}$.  
  - **Unclamped exponentiation overflow:** `exp(atten_proj)` overflowing to `inf`, leading to $\text{inf} - \text{inf} = \text{NaN}$ in `out_new.max() - out_new`.  
Adding `1e-8` epsilon ensures $0.0 / (0.0 + 1e-8) = 0.0$ instead of `NaN`, while gradient clipping protects against GAN gradient explosion. Combining `atten_proj` clamping, `NormLayer` epsilon, and gradient clipping guards against both potential failure modes simultaneously.  
**Verification:** Verified python compilation with `py_compile`. End-to-end loss stability across 100 epochs pending verification on a fresh worker launch with all fixes deployed.
