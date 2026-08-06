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
