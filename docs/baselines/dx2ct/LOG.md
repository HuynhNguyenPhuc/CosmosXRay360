# Dx2CT Change & Fix Log

## 2026-08-16 — Reinstated 128^3 Paper Grid Resolution & Moved Worker to Dedicated A100 Slot

**Phase:** 2/3 (Paper Fidelity + Infrastructure)  
**Files:** `baselines/models/dx2ct.py`, `baselines/train/dx2ct.py`, `scripts/launch_parallel_vms.sh`  
**Change:** Updated `num_slices` and `spatial_size` from `32` / `64` back to `128` / `128` in `Dx2CTWrapper.infer_multi_views` and `train/dx2ct.py` (`spatial_size = 128`), matching the paper's exact reported $128 \times 128 \times 128$ voxel volume specification (arXiv:2409.08850 §3). Assigned Dx2CT to dedicated worker `cosmos-worker-3` using `cosmosxray360-a100-trainer-template` (NVIDIA A100 40GB GPU) in `scripts/launch_parallel_vms.sh` to prevent OOM errors encountered under L4 GPUs at $128^3$.  
**Why:** Re-establishes 100% paper fidelity for 3D slice resolution, eliminating downsampled $32$-slice staircase artifacts during ray marching.  
**Verification:** All unit tests in `baselines/tests/test_dx2ct_wrapper.py` and `baselines/tests/test_dx2ct_architecture.py` pass cleanly.

---

## 2026-08-15 — Paper-Documented Sampling Fix (DDIM + Shared x_T) and Train/Inference Lateral-Input Fix

**Phase:** 2/3 (Correctness + Training Scope)
**Files:** `baselines/models/dx2ct.py`, `baselines/train/dx2ct.py`
**Change (inference, `baselines/models/dx2ct.py`)**: `infer_multi_views` switched from `DDPMScheduler`
to `DDIMScheduler` (`eta=0.0` default, deterministic), `num_inference_steps` 20 -> 50, and the initial
noise from independent-per-slice `torch.randn(B*num_slices, ...)` to one shared `[B, 1, H, W]` draw
`repeat_interleave`d across each item's `num_slices`. All three changes replicate `docs/baselines/
dx2ct/PAPER.md`'s own stated sampling protocol verbatim: *"Sampling: DDIM, 50 steps; the initial noise
x_T is fixed and DDIM's random noise term removed, specifically to keep slice-to-slice reconstruction
consistent within one volume."* The wrapper had none of these three things -- stochastic DDPM steps
plus independent per-slice noise is close to the opposite of what the paper specifies for exactly the
artifact class (comb/streak patterns at oblique azimuths) this project had been diagnosing.
**Change (training, `baselines/train/dx2ct.py`)**: training always loaded a real, distinct `lat.png`
per patient, while `infer_multi_views` (the *only* way this baseline is ever evaluated, per this
project's single-view benchmark protocol) duplicates the PA image into the lateral slot. The model had
never seen that exact input during training. Added `--lat_dropout_prob` (default 0.5): on that fraction
of training samples, `lat` is replaced with a clone of `pa`, matching the real inference input. Also
changed the validation loop to *always* use `lat = pa` (not the dropout probability) so
best-checkpoint selection (`avg_val_loss`) tracks the actual deployment condition rather than an
easier real-biplanar signal this baseline never gets to use at test time.
**Result -- real, but a genuinely mixed finding, not a clean win**: regenerated the panel from the
existing (pre-lat-dropout-retrain) checkpoint twice, with two different random seeds, to separate a
real effect from an unlucky draw.
- The oblique comb/streak artifact at ~90/270 degrees is **gone** in both runs, replaced by smooth
  directional bands -- confirms the paper's consistency mechanism works as documented.
- A **new** coherent radial/starburst artifact appears at the near-frontal columns (~0/190 degrees)
  in **both** seeds (same location, same radial character, different fine-grained noise realization)
  -- not a fluke. Conclusion: this checkpoint has a pre-existing systematic bias (most likely centered
  on the coordinate-embedding origin each axial slice's `(x,y)` grid shares) that was previously being
  *masked* by the independent per-slice DDPM noise averaging it out across the volume when ray-marched.
  Making sampling deterministic and shared-noise, as the paper specifies, removes that accidental
  masking and exposes the bias coherently instead.
**What this means going forward**: the sampling fix is correct and should stay -- it demonstrably
fixes the documented artifact class -- but its full benefit is capped by this specific checkpoint's
residual bias, which is a training/checkpoint-quality question, not a wrapper bug. Training curve for
this checkpoint (`gcs_tensorboard/tensorboard/dx2ct`) shows healthy convergence over 80 real epochs
(0.044->0.0072 train, 0.021->0.0070 val), so this doesn't look like SV-DRR-style raw undertraining --
more likely a real, if narrow, representational gap. Needs a fresh retrain (now also with
`--lat_dropout_prob` active) and a re-check of whether the radial artifact persists, before concluding
whether it's a training-budget issue or something structural.
**Verification**: `py_compile` clean on both files. `uv run pytest baselines/tests/test_dx2ct_wrapper.py
baselines/tests/test_dx2ct_architecture.py` passes (9/9). Sampling fix verified against the real
`cosmos-worker-1/dx2ct_best.pt` checkpoint and a real NSCLC test patient, twice with different seeds
(not just a single run trusted at face value).
**Follow-up (2026-08-15, same day): added LR warmup+cosine decay and gradient clipping.** No sibling
`Cosmos-NVSyn` reference exists for Dx2CT (unlike SV-DRR/XRaySyn), so unlike those two this isn't
"port a verified config" -- it's general diffusion-from-scratch-training practice applied on request.
Rationale specific to Dx2CT: it trains from random init (not a fine-tune like SV-DRR), which is exactly
the regime warmup (early-instability protection) and clipping matter most for. Added
`--warmup_steps` (default 500, matching SV-DRR's) and `--gradient_clip_val` (default 1.0, matching
XRaySyn's own `clip_grad_norm_` convention rather than SV-DRR's DiT-specific 0.5). Deliberately did
**not** port SV-DRR's `(0.9, 0.95)` AdamW betas (a DiT-training convention that doesn't clearly apply
to Dx2CT's ResNet+transformer+CNU-Net hybrid) or logit-normal timestep sampling (no demonstrated
problem it would fix here -- Dx2CT's loss curve is already healthy, unlike SV-DRR's slow-tail case that
motivated it) -- left as an optional future experiment, not applied.
**Verification**: `py_compile` clean. Smoke test (`--epochs 3 --max_batches 2 --max_val_batches 1
--warmup_steps 2`) completed end-to-end with no errors; logged LR values (2.50e-05 -> 5.00e-05 ->
5.00e-06) exactly match hand-computed cosine decay for that config. `uv run pytest
baselines/tests/test_dx2ct_wrapper.py baselines/tests/test_dx2ct_architecture.py` passes (9/9).

**Not done**: no full (non-smoke) retrain launched yet (pending, same as SV-DRR's outstanding retrain). The bigger,
paper-motivated idea of training on more than axial-only slices was investigated (`docs/baselines/
dx2ct/PAPER.md`'s own stated limitation #1: axial is explicitly the paper's *worst*-reconstructed
plane, exactly because it's perpendicular to both PA and lateral) but not implemented -- it requires
extending `CrossDatasetSliceDataset` to sample coronal/sagittal ground-truth slices too (the same
`extract_axial_slices`-style grid_sample trick, holding a different axis fixed), a bigger data-pipeline
change than this pass's scope. Left as a documented next step, not started.

---

## 2026-08-14 — Reverted a Regression: Copilot's `sample_fn` Axis-Permutation "Fix" Was Wrong

**Phase:** 2 (Correctness)
**Files:** `baselines/models/dx2ct.py`
**Change:** Reverted `infer_multi_views`'s `sample_fn(pts)` grid_sample coordinate mapping from
`pts[..., [0, 2, 1]]` (a locally-staged, uncommitted Copilot edit) back to `pts` unpermuted
(`grid_coords = pts.view(1, 1, 1, -1, 3)`), i.e. the pre-Copilot behavior — with a comment explaining why.
**Why:** While investigating why `results/dx2ct.png` looked wrong, found a staged (uncommitted) Copilot
change had permuted `sample_fn`'s grid coordinates, on the same general theory as the (correct) NAF fix
in this same session (see `docs/baselines/naf/LOG.md`) — but Dx2CT's `vol_pred` is built differently from
NAF's baked grid. Tracing `baselines/train/dx2ct.py`'s `extract_axial_slices`/`coords_3d` (the ground-truth
construction the diffusion model is trained against) shows `vol_pred`'s axes are already `(D=z_world,
H=y_world, W=x_world)` — exactly the `(D,H,W)`/`(z,y,x)` order `F.grid_sample` expects from an unpermuted
`(x,y,z)` query. The staged `[0,2,1]` permutation swapped `y_world` and `z_world`, which is wrong.
**Verification:** Empirical, not just derived (learned from the NAF investigation not to trust derivation
alone): baked a synthetic single-voxel marker into a `vol_pred`-shaped tensor at a known `(d0,h0,w0)`,
queried `sample_fn` at the exact training-space `(x0,y0,z0)` that position represents — the unpermuted
(reverted-to) code returned `~1.0` (correct hit), the staged `[0,2,1]` permutation returned `0.0` (missed
it). `uv run pytest baselines/tests/test_dx2ct_wrapper.py baselines/tests/test_dx2ct_architecture.py`
still passes (neither test exercises this code path directly — see the 2026-08-05 entry below for the same
caveat). Follow-up same day: pulled the real trained checkpoint from GCS
(`gs://.../checkpoints/cosmos-worker-1/dx2ct_best.pt`, 199MB) and re-ran `build_multiview_grid` against a
real NSCLC test patient. Result: the near-frontal (~4°/~190°) predicted views now show a clear, recognizable
ribcage/mediastinum silhouette closely matching ground-truth PA shape — a dramatic improvement over
`results/dx2ct.png`'s stale panel. Oblique views (~60-300°) are blurrier vertical-rib-banding hallucinations,
a reasonable failure mode for a genuinely OOD single-view angle, not obviously wrong. This confirms the
axis-permutation revert above is correct in practice, not just via the synthetic marker test — quality
verification is no longer blocked; `research-state.yaml`'s next-action updated accordingly.

---

## 2026-08-03 — Modular Reimplementation & Dynamic Module Import

**Phase:** 1 (Import & Environment)  
**Files:** `baselines/cloned/DX2CT/model.py`, `baselines/models/dx2ct.py`, `baselines/train/dx2ct.py`  
**Change:** Reimplemented `DX2CTModel`, SPADE normalization, and PositionAwareAttention from scratch in `baselines/cloned/DX2CT/model.py` (code-unreleased paper arXiv:2409.08850). Created `Dx2CTWrapper` and `train_dx2ct` using dynamic `importlib.util` loader under module alias `dx2ct_model_module`.  
**Why:** Reimplements the unreleased paper architecture from scratch and prevents top-level `from model import ...` namespace pollution in `sys.modules['model']` across other baseline wrappers.  
**Verification:** Verified in `baselines/tests/test_dx2ct_architecture.py` and `baselines/tests/test_dx2ct_wrapper.py`.

---

## 2026-08-05 — PAPER.md/CODE.md Corrected Against the Actual Cached PDF

**Phase:** 0 (Resource acquisition / correctness of docs, not code)
**Files:** `docs/baselines/dx2ct/{PAPER.md,CODE.md}`
**Change:** The 2026-08-05 docs backfill pass wrote `PAPER.md` with the wrong paper title ("Dx2CT:
Position-Aware Slice Diffusion Model for 3D CT Reconstruction from Single X-Ray", which does not
exist — the real title is "DX2CT: Diffusion Model for 3D CT Reconstruction from Bi or Mono-planar 2D
X-ray(s)", and the paper's *primary* setup is biplanar, not single-X-ray) and described conditioning
as generic "sinusoidal position embeddings and cross-attention" rather than the paper's actual named
contribution, the 3D Positional Query Transformer (3DPQT) feeding a SPADE-conditioned U-Net. `CODE.md`
separately used class names (`PositionAwareAttention`, `SPADEResBlock`) that don't exist in
`baselines/cloned/DX2CT/model.py` — the real classes (`SliceTransformer3DPQT`, `SPADEResNetBlock`,
`DenoisingUNetSPADE`, confirmed via `grep -n "^class "`) actually track the paper's own 3DPQT/SPADE
naming closely, which the wrong CODE.md text obscured rather than credited. Both rewritten directly
from `arxiv_papers/Dx2CT_2409.08850.pdf`.
**Why:** Same failure mode as `docs/baselines/xraysyn/LOG.md`'s 2026-08-05 correction entry.
Notable here specifically: the from-scratch reimplementation in `model.py` turned out to be *more*
paper-faithful than the doc describing it — a reminder that for the code-unreleased case,
`CODE.md`'s accuracy depends entirely on reading both the paper and the reimplementation carefully,
since there's no official repo to cross-check either against.
**Verification:** Title/authors/venue/equation numbers cross-checked page-by-page against the PDF;
`model.py` class names re-verified with `grep`, not asserted from memory.

---

## 2026-08-05 — Real Axial CT Slice Supervision, CT Caching & Perspective Ray Marching

**Phase:** 2 & 3 (Correctness, Acceleration & Physics)  
**Files:** `baselines/train/dx2ct.py`, `baselines/models/dx2ct.py`  
**Change:** Rewrote `train_dx2ct` to extract and train on real axial CT density slices (`extract_axial_slices` from `load_ct_volume_cached`) directly from raw CT volumes, matching arXiv:2409.08850 methodology exactly. Implemented atomic on-disk `.pt` volume caching in `datasets/_ct_volume_cache/`. Switched `Dx2CTWrapper.infer_multi_views` from `.mean(dim=0)` depth collapse to `perspective_ray_march(..., reduction="attenuation_sum")`. Switched target angle generation from `endpoint=False` to `endpoint=True`.  
**Why:** Replaces an incorrect setup that trained on 2D DRR projections with paper-faithful axial CT slice diffusion, eliminates ~31,000 redundant NIfTI decodes across 80 epochs (~8x dataloader speedup on repeat epochs), performs true Beer-Lambert line integrals through the generated 3D CT attenuation volume, and aligns azimuth angles with ground-truth DiffDRR projections (`views/*.png`).  
**Verification:** `test_dx2ct_architecture.py` and `test_dx2ct_wrapper.py` pass cleanly.

---

## 2026-08-05 — Correction: `attenuation_sum` Regressed a Value-Space Bug, Reverted to `mean`

**Phase:** 2 (Correctness — regression found and fixed during a multi-agent baseline verification pass)
**Files:** `baselines/models/dx2ct.py`
**Change:** Reverted `Dx2CTWrapper.infer_multi_views`'s `perspective_ray_march` call from
`reduction="attenuation_sum"` (introduced in the entry above) back to `reduction="mean"`. Also fixed
`Dx2CTWrapper.__init__` to log a warning (not `"✓ Loaded successfully"`) when `checkpoint_path` is
missing/unset and the model is left randomly initialized.
**Why:** The entry above's reasoning — "`vol_pred` is now a genuine linear-attenuation-like field and
needs the same Beer-Lambert forward projection NAF uses" — is true of `vol_pred` in isolation, but wrong
about what it should be compared against. The ground-truth `views/*.png` this projection is scored
against is produced by `renderers/diffdrr/renderer.py`'s `render()`, which is a raw (rescaled) line
integral — `sum(density) * step_size` per that file's own equivalence docstring — with no `exp()` step;
`norm_type="standardized"` only applies `normalized(standardized(...))`, both purely affine. Applying
Beer-Lambert (`exp(-sum)`) to the prediction before comparing it to a non-exponentiated target is exactly
the value-space mismatch `docs/GOTCHAS.md` #1 warns about, and is exactly the case
`perspective_ray_march`'s own `reduction` docstring (`baselines/models/utils.py`) names Dx2CT as the
canonical example for using `"mean"`, not `"attenuation_sum"` — a docstring the previous entry's own diff
left unedited and unreconciled. NAF's use of `"attenuation_sum"` is *not* an equivalent precedent: NAF's
density field is gradient-fit per-scan directly against this same (non-exponentiated) target with the
Beer-Lambert transform inside the loss (`fit_density_field`), so it self-corrects to whatever field makes
`exp(-sum(density)) ≈ target` regardless of the transform's physical label. Dx2CT's diffusion model is
trained independently of this projection step (pure denoising-score-matching on axial CT slices, no
ray-marching in the training loop), so there is no such self-correction — the reduction must match the
ground truth's actual convention on its own, and after `normalize_tensor()`'s min-max rescale, `"mean"`
and a raw `"sum"` line integral are equivalent up to the constant `n_samples` divisor, which is exactly
the convention GT was built with.
**Verification:** Traced `renderers/diffdrr/renderer.py:370-459` (`render()`) and its module docstring
line-by-line to confirm no exponential is applied anywhere in the GT pipeline; cross-checked against the
git-history (committed, later locally deleted) `docs/LOSSES.md` audit table, which independently reached
the same conclusion for the pre-rewrite Dx2CT ("Kept `.mean(dim=0)` projection without double-applying
Beer-Lambert exponential on already-transmissive slices") — the conclusion (`mean`, not Beer-Lambert)
still holds post-rewrite even though that doc's specific *reasoning* (slices already being
transmissive-intensity) no longer applies verbatim to the new density-slice training target; the
correct reasoning is that GT itself is non-exponentiated, independent of what `vol_pred` represents.
Did not re-run `test_dx2ct_wrapper.py`/`test_dx2ct_architecture.py` in this pass (neither test exercises
the reduction mode or GT comparison — see the accompanying baseline-verification report); this remains
to be confirmed with a real PSNR/SSIM re-evaluation once `baselines/checkpoints/dx2ct_*.pt` is retrained
on the current (real-axial-CT-slice) training code, since the existing checkpoint predates that rewrite.

---

## 2026-08-05 — Phase 3: DataLoader Worker Bump & Deferred Loss Sync

**Phase:** 3 (Acceleration)
**Files:** `baselines/train/dx2ct.py`
**Change:** Raised both `train_loader`/`val_loader`'s `num_workers` from `2` to `6` (20 CPUs available on
this host, `persistent_workers=True` already amortizes worker respawn cost). Changed
`epoch_train_loss`/`epoch_val_loss` from a Python float accumulated via `loss.item()` every batch to a
device tensor accumulated via `loss.detach()`, with `.item()` called once after each loop — removes a
CUDA sync every training/validation step.
**Why:** Pure wall-clock changes, no algorithmic change: confirmed `writer.add_scalar("Loss/...")` is only
called once per epoch (on the post-loop average), never per-step, so deferring the sync changes nothing
observable. `num_workers=2` was low relative to available CPUs for a loader doing real NIfTI-adjacent
decode work on cache misses.
**Verification:** `uv run pytest baselines/tests/test_dx2ct_wrapper.py baselines/tests/test_dx2ct_architecture.py`
(9/9 pass) and `python -m py_compile baselines/train/dx2ct.py` after the change. Wall-clock speedup **not
yet measured** (no full training run executed in this pass) — per this skill's Phase 4 discipline, this is
logged as an applied Phase 3 change, not yet a verified Phase 4 win; measure a real before/after epoch time
before citing a speedup number.
