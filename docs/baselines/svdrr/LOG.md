# SV-DRR Change & Fix Log

## 2026-08-03 — Initial SV-DRR Pipeline Integration

**Phase:** 1 (Import & Environment)  
**Files:** `baselines/models/svdrr.py`, `baselines/train/svdrr.py`  
**Change:** Created `SVDRRWrapper` and `train_svdrr` training script integrating `pipeline_svdrr_DiT.py` from `baselines/cloned/SV-DRR/`. Added automatic `snapshot_download(repo_id="xiechun-tsukuba/svdrr-dit-fb-256")` to `baselines/cloned/SV-DRR/models/base_model/256`.  
**Why:** Materializing a local model snapshot resolves Diffusers custom-component path resolution failures when loading `cc_projection/pipeline_svdrr_DiT.py`.  
**Verification:** `SVDRRWrapper` initializes cleanly and passes unit tests in `baselines/tests/test_svdrr_wrapper.py`.

---

## 2026-08-05 — PAPER.md/CODE.md Corrected Against the Actual Cached PDF

**Phase:** 0 (Resource acquisition / correctness of docs, not code)
**Files:** `docs/baselines/svdrr/{PAPER.md,CODE.md}`
**Change:** The 2026-08-05 docs backfill pass wrote `PAPER.md` with the wrong paper title ("SV-DRR:
Single-View Digitally Reconstructed Radiograph Synthesis with Diffusion Transformers", which does
not exist) and an architecture description (a single pose-vector MLP into cross-attention only) that
doesn't match the paper's actual two-stream conditioning (CLIP-embedding+relative-pose into
cross-attention, *plus* a separate channel-concatenation of the source latent with the noised target
latent). `CODE.md`'s license line was also an unresolved guess ("Apache-2.0 / MIT"). Caught during
same-day review, rewritten directly from `arxiv_papers/SV_DRR_2507.05148.pdf`, and the `CCProjection`/
`SvdrrDiTPipeline` class names re-verified with `grep` against `pipeline_svdrr_DiT.py`; license
verified as plain MIT against the actual `LICENSE` file.
**Why:** Same failure mode as `docs/baselines/xraysyn/LOG.md`'s 2026-08-05 correction entry — the
paper must actually be read (generic-pattern.md's Phase 0 rule), not reconstructed from a plausible-
sounding guess at what a paper with this name probably contains.
**Verification:** Title/authors/venue/equation numbers cross-checked page-by-page against the PDF;
class names and license re-verified against the actual repo files with `grep`/`cat`, not asserted
from memory.

---

## 2026-08-05 — Full-Sweep View-Pair Dataset & Joint Pose Optimization

**Phase:** 2 (Fairness / Training Scope & Correctness)  
**Files:** `baselines/train/svdrr.py`, `baselines/models/svdrr.py`  
**Change:** Expanded training dataloader to cache all 93 pre-rendered views per patient, sampling random `(source, target)` view pairs ($\sim 93 \times 92$ combinations) every step while keeping validation fixed to `source = views[0]` (frontal PA). Enabled joint training of `cc_projection` at $10\times$ LR (`lr = 10 * args.lr`) and saved both `transformer` and `cc_projection` state dicts in checkpoints.  
**Why:** Official code trained only on fixed 0°–90° PA-LAT pairs, whereas benchmark evaluation queries the full 360° sweep. Jointly training `cc_projection` ensures pose conditioning is optimized alongside the DiT denoiser.  
**Verification:** Saved checkpoints contain both sub-network weights; `SVDRRWrapper` restores both keys successfully.

---

## 2026-08-05 — Dynamic Velocity Target, Batched Caching & Azimuth Alignment

**Phase:** 2 & 3 (Correctness, Acceleration & Convention Alignment)  
**Files:** `baselines/train/svdrr.py`, `baselines/models/svdrr.py`  
**Change:** Handled `v_prediction` velocity targets dynamically via `pipe.scheduler.get_velocity()` when `prediction_type == "v_prediction"`. Batched VAE latent and CLIP image encoding in chunks of `CACHE_BATCH = 16` during `cache_dataset`. Switched target angle generation from `endpoint=False` to `endpoint=True`.  
**Why:** Supports velocity-prediction DDPM/DDIM schedulers, turns ~36,000 single-image encoding calls into ~2,300 batched calls, and aligns azimuth angles with ground-truth DiffDRR projections (`views/*.png`).  
**Verification:** Verified in `baselines/tests/test_loss_regression.py` and `baselines/tests/test_svdrr_wrapper.py`.

---

## 2026-08-05 — Phase 3: Deferred Loss Sync

**Phase:** 3 (Acceleration)
**Files:** `baselines/train/svdrr.py`
**Change:** Changed `epoch_train_loss`/`epoch_val_loss` from a Python float accumulated via `loss.item()`
every step to a device tensor accumulated via `loss.detach()`, with `.item()` called once after each loop.
**Why:** Removes a CUDA sync every training/validation step; confirmed `writer.add_scalar("Loss/...")` is
only called once per epoch, not per-step, so this changes nothing observable. Did not thread
`cache_dataset`'s sequential `Image.open` calls or pin/`non_blocking` the per-step cached-latent transfers
(both flagged as candidates in the accompanying acceleration survey) — the former adds real complexity
(order-preservation across a `ThreadPoolExecutor`) for I/O that's already dwarfed by the batched VAE/CLIP
encode calls per this file's own comment, and the latter moves a single `[1,4,H,W]` tensor per step (too
small to be worth pinning ~36k cached latents in host RAM for); left as a documented option, not applied.
**Verification:** `uv run pytest baselines/tests/test_svdrr_wrapper.py` (pass) and `python -m py_compile
baselines/train/svdrr.py`. Wall-clock speedup **not yet measured** — see dx2ct's identically-dated entry
for the same Phase 4 caveat.
