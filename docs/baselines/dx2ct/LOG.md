# Dx2CT Change & Fix Log

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
