# PixelNeRF Change & Fix Log

## 2026-08-03 — Initial PixelNeRF Integration, Coarse MLP Loss Fix & Photometric Training Rewrite

**Phase:** 1 & 2 (Import & Correctness)  
**Files:** `baselines/models/pixelnerf.py`, `baselines/train/pixelnerf.py`  
**Change:** Created `PixelNeRFWrapper` and `train_pixelnerf` training script. Configured `build_model_and_renderer(device, simple_output=False)` and added coarse loss `criterion(pred_coarse, target) + criterion(pred_fine, target)`. Frozen ResNet34 encoder weights (`p.requires_grad = False`). Rewrote `train_pixelnerf` to perform end-to-end photometric NeRF volume rendering loss against lateral (LAT) target images.  
**Why:** Fixes dead `mlp_coarse` sub-network bug (which received zero gradient under `simple_output=True`), prevents ResNet encoder overfitting on small medical dataset, and replaces a dummy feature-map loss with real novel-view ray-marching supervision.  
**Verification:** Verified in `baselines/tests/test_loss_regression.py` (`test_pixelnerf_coarse_mlp_receives_gradient`) and `baselines/tests/test_pixelnerf_wrapper.py`.

---

## 2026-08-05 — Azimuth Endpoint Alignment

**Phase:** 2 (Convention Alignment)  
**Files:** `baselines/models/pixelnerf.py`  
**Change:** Switched target angle generation from `np.linspace(..., endpoint=False)` to default `endpoint=True`.  
**Why:** Aligns synthesized angles with ground-truth DiffDRR projections (`views/*.png`), eliminating a ~3.9° drift at $360^\circ$.  
**Verification:** `infer_multi_views` output indices match ground-truth view files exactly.

---

## 2026-08-05 — Correction: Encoder-Freeze Mechanism Mischaracterized

**Phase:** 2 (Verification / Documentation Correctness)
**Files:** `docs/baselines/pixelnerf/CODE.md` (no code changed)
**Change:** The 2026-08-03 entry above and `CODE.md`'s discrepancy #2 both described the ResNet34 encoder
freeze as `p.requires_grad = False`. Independent verification (multi-agent baseline audit, this date) found
no such loop anywhere in `baselines/models/pixelnerf.py` or `baselines/train/pixelnerf.py`
(`grep -n "requires_grad"` returns nothing). The actual mechanism is `model.stop_encoder_grad = True`
(`baselines/models/pixelnerf.py:102`), which makes the official code's own `forward()`
(`baselines/cloned/pixel-nerf/src/model/models.py:217-218`) `.detach()` the encoder's output latent.
**Why:** Functionally equivalent (encoder never receives gradient either way), but the doc's stated
mechanism was wrong, not just imprecisely worded — worth a correction entry so a future reader grepping
for `requires_grad` doesn't conclude the freeze is missing.
**Verification:** `grep -rn "requires_grad" baselines/models/pixelnerf.py baselines/train/pixelnerf.py`
(no hits) vs. `grep -n "stop_encoder_grad" baselines/models/pixelnerf.py baselines/cloned/pixel-nerf/src/model/models.py`
(both present). `CODE.md` updated to match.

---

## 2026-08-05 — Phase 3: Deferred Loss Sync

**Phase:** 3 (Acceleration)
**Files:** `baselines/train/pixelnerf.py`
**Change:** Changed `epoch_train_loss`/`epoch_val_loss` from a Python float accumulated via `loss.item()`
every step to a device tensor accumulated via `loss.detach()`, with `.item()` called once after each loop.
**Why:** Removes a CUDA sync every training/validation step; confirmed `writer.add_scalar("Loss/...")` is
only called once per epoch, not per-step. Did not apply AMP to the encoder+coarse/fine-MLP forward pass
(flagged in the accompanying survey as the single best AMP candidate of the 6 baselines, since it's a
plain conv/MLP forward with no diffusion scheduler or custom CUDA kernel) — left as a recommendation
pending a real Phase 4 regression + wall-clock measurement, not applied blind.
**Verification:** `uv run pytest baselines/tests/test_pixelnerf_wrapper.py baselines/tests/test_loss_regression.py::test_pixelnerf_coarse_mlp_receives_gradient`
(pass) and `python -m py_compile baselines/train/pixelnerf.py`. Wall-clock speedup **not yet measured**.

---

## 2026-08-07 — Fixed Fixed-Single-View Training Scope (Real Correctness Gap, Not Just Epoch Count)

**Phase:** 2 (Training Scope / Fairness Correctness)
**Files:** `baselines/train/pixelnerf.py`
**Change:** Training target view is now sampled randomly from the full 93-view pre-rendered sweep per
patient (excluding index 0, the frontal/PA pose matching the source image), instead of always
`LATERAL_AZIMUTH_DEG = 90.0`. Added `cache_train_tensors`/`sample_train_target` (loads all
`views/*.png` + their azimuth angles, mirroring `baselines/train/svdrr.py`'s `cache_dataset`/
`sample_pair`) alongside the original `cache_val_tensors` (unchanged, still PA+LAT only) so
validation stays a single fixed, stable, comparable metric across epochs.
**Why:** A real 8-epoch training run's val loss bottomed at epoch 5 (0.019171) then rose for 3
consecutive epochs (0.019251 → 0.019338 → 0.019571) while train loss kept falling — classic
overfitting to one specific task. The root cause: this wrapper trained on the *single* PA→LAT (90°)
pair every step, ever, while `baselines/evaluate.py` scores a full 360° sweep — verified directly
against the original repo's own `train/train.py` (`calc_losses`/`train_step`), which samples random
rays across *all* views and a random source/target split every step
(`pix_inds = torch.randint(0, NV * H * W, ...)`, `view_dest = np.random.randint(...)`). This is the
exact same training-scope/fairness gap class already found and fixed for SV-DRR (fixed 90°-apart pair
→ full 93-view random pairs) and XRaySyn (±9° → full 360° `OTHER_POSE_THETA_Y_RANGE`) — PixelNeRF was
simply never given the same treatment, and a cross-baseline consistency check (per this project's
`baseline-experiment` skill's Phase 2 checklist) would have caught this earlier than a live overfitting
signal did.
**Verification:** Local smoke test (`uv run python baselines/train/pixelnerf.py --epochs 1
--max_train_samples 3 --max_val_samples 2`) runs end-to-end, produces real (non-NaN) train/val loss,
and saves checkpoints. The previous checkpoint (trained under the fixed-90°-only setup, which already
showed an overfitting signal by epoch 8) needs a full retrain under this fix before any new benchmark
number is trusted — not yet done as of this entry.
