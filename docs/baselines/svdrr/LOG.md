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

---

## 2026-08-07 — Real Mini-Batch Training (Was Effectively Batch=1) + Gradient Accumulation to Match the Paper's Batch=64

**Phase:** 2 & 3 (Fairness Correctness & Acceleration)
**Files:** `baselines/train/svdrr.py`
**Change:** The training loop previously ran one `(source, target)` pair per gradient step (`for item in
train_cached: ... optimizer.step()`) -- an effective batch size of 1, which doesn't match the paper's own
reported recipe of batch 64 (`docs/baselines/svdrr/PAPER.md` section 3: "200K steps at 256 res" on batch
64). Replaced with real mini-batch training: `sample_batch(batch_size)` draws `batch_size` pairs from
`batch_size` randomly-chosen (with replacement) cached patients via the existing, unchanged `sample_pair`,
stacks them, and one gradient step processes the whole batch (`_encode_pose` already accepts a
list-of-lists pose batch with no changes needed to the vendored pipeline; `timesteps` now sampled one per
batch item instead of one shared timestep for the whole step). Also added `--accum_steps` (gradient
accumulation): `batch_size=16` micro-batches accumulated over `accum_steps=4` gradient calls before a
single `optimizer.step()` gives an *effective* batch of 64, matching the paper exactly, without ever
holding 64 samples' activations in VRAM at once (`micro_loss / accum_steps` before each `.backward()`,
so the summed gradient equals the true mean gradient over the full effective batch, not just an
extra unweighted average).
**Why:** User-requested ("keep the fairness, try to do like the original paper do") rather than just
inflating the epoch count on the old batch-1 loop, which would only ever match the paper's *step count*,
not its actual per-step training dynamics (batch 64 sees 64x more data per gradient signal than batch 1).
**Verification:** Local testing on a 24GB GPU (Titan RTX) with `--max_train_samples 20` (fast smoke-test
slices, not the full ~1000-patient set): `--batch_size 8` (no accumulation) completed cleanly, real
non-`nan` losses. `--batch_size 64` (one shot, no accumulation) **OOMs**:
`CUDA out of memory... 22.61 GiB memory in use` out of the GPU's 23.46 GiB. `--batch_size 4 --accum_steps
3` (validating the accumulation code path itself) completed cleanly, real non-`nan` losses, checkpoints
saved correctly. Did not separately re-test the actual chosen default (`--batch_size 16 --accum_steps 4`)
in isolation -- reasoned safe from the batch=8-clean / batch=64-OOM data points (16 is well below the OOM
threshold with real margin), but this should be confirmed with an actual run before fully trusting it at
scale, not just extrapolated.
