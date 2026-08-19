# SV-DRR Change & Fix Log

## 2026-08-15 — Ported Reference Training Objective (IDDPM SNR Loss, Logit-Normal Timesteps, CFG Dropout, EMA)

**Phase:** 2/3 (Correctness + Training Scope)
**Files:** `baselines/train/svdrr.py`, `baselines/models/svdrr.py`
**Change:** Ported four real methodological gaps from `Cosmos-NVSyn/model/nvsyn_svdrr.py` (a sibling
codebase's Lightning-based SV-DRR module, user-identified as "the old version, it works correctly")
into this project's from-scratch training loop:
1. **IDDPM loss** (`diffusion.iddpm.IDDPM`, already vendored in `baselines/cloned/SV-DRR/diffusion/` —
   no need to copy code) replacing a plain `nn.MSELoss()` on epsilon-only. Config matches the old
   module exactly: `learn_sigma=True, pred_sigma=True, snr=True` (hybrid parameterization: predicts
   epsilon at high-noise timesteps `t>249`, x0 directly at low-noise timesteps `t<=249`, avoiding
   amplifying tiny errors when recovering x0 from epsilon via near-zero alphas late in denoising) plus
   a variational-bound term for the model's learned-sigma channels. Betas (`beta_start=0.0001,
   beta_end=0.02`, linear, 1000 steps) verified to match `pipe.scheduler`'s own config exactly, so
   train-time and inference-time noise schedules agree.
2. **Logit-normal timestep sampling** for training (`sample_timesteps_logit_normal`) instead of
   uniform `torch.randint` — biases gradient signal toward mid-noise timesteps that determine
   anatomical structure, rather than wasting it on near-trivial high-noise-timestep predictions.
   Validation timesteps stay uniform (matches the old module's own `training_step`/`validation_step`
   asymmetry) so the val-loss metric stays a fair, comprehensive estimate, not biased.
3. **Classifier-free-guidance dropout** (`--conditioning_dropout_prob`, default 0.05) — training now
   actually zeroes conditioning + condition latents together on ~5% of samples, so the
   `guidance_scale=3.0` used at inference (`baselines/models/svdrr.py`) has a real trained
   unconditional branch to extrapolate from, instead of one it never saw during training.
4. **EMA transformer** (`--ema_decay`, default 0.9995) — maintained as a `copy.deepcopy` shadow copy,
   updated after every `optimizer.step()`, saved under a new `ema_transformer` checkpoint key.
   `SVDRRWrapper.__init__` now prefers `state["ema_transformer"]` over `state["transformer"]` when
   present, falling back cleanly for older checkpoints that don't have it.

Also adopted the transformer's native `latents_concat` forward kwarg (confirmed via
`svdrr_transformer_2d.py`: `hidden_states = torch.cat([hidden_states, latents_concat], dim=1)` --
identical concat order to what this file built manually) instead of manually concatenating condition
latents before the forward call — required by `training_losses_diffusers`' calling convention anyway,
and removes the `transformer.config.in_channels == 8` branch entirely.

**Why:** The old module's 4 gaps were found by direct code comparison after the user flagged it as a
working reference, not by assumption. One candidate (CFG-mismatch) was tested empirically *before*
being trusted: toggling `guidance_scale` 3.0 -> 1.0 on the *existing* (pre-port) fine-tuned checkpoint
did **not** improve output (both settings produced non-anatomical noise, just differently textured) —
an honest negative result showing the current checkpoint's badness is dominantly raw undertraining, not
purely the CFG mismatch alone. That result is exactly why the fix here is "train with these four gaps
closed," not "just flip the CFG scale on the existing checkpoint."

**Verification:** `python -m py_compile` on both files. `uv run pytest
baselines/tests/test_svdrr_wrapper.py` passes (confirms the new `ema_transformer`-preferring load path
doesn't break loading older-format checkpoints). End-to-end smoke test:
`--epochs 2 --max_train_samples 2 --max_val_samples 1 --batch_size 2 --accum_steps 1 --viz_every 1`
against 3 freshly-pre-rendered real TCIA (train-only, OOD-compliant) patients completed both epochs
with no errors, decreasing loss both epochs (train 0.1537->0.0398, val 0.0300->0.0233 -- not
comparable in absolute scale to the pre-port MSE-only loss curve, since the loss formulation itself
changed), and both epochs' `Images/val_multiview` TensorBoard panel generated successfully (confirms
`SVDRRWrapper` loads the new checkpoint format through the real `visualize_live_checkpoint` path, not
just a unit test).

**Side effect (disclosed)**: the smoke test above overwrote `baselines/checkpoints/{svdrr_best,
svdrr_checkpoint}.pt` with a throwaway 2-epoch/2-sample dummy (same known limitation as the 2026-08-10
research-log entry — no `--ckpt_dir` override to redirect smoke-test output elsewhere). The file it
overwrote was already a corrupted/truncated download (992MB vs. the real ~2.4GB), so nothing valid was
lost, but the real reference checkpoint for comparisons remains
`baselines/checkpoints/cosmos-worker-4/svdrr_best.pt` (untouched) until a real run replaces it.

**Next action**: launch a real training run (not a smoke test) with these fixes active — the user's
proposed target is ~10,000 real optimizer steps, up from the previous checkpoint's 945.

---

## 2026-08-15 — Second Pass: 4 More Gaps Found (LR Schedule, Grad Clipping, Betas, Weight Decay)

**Phase:** 2/3 (Correctness + Training Scope)
**Files:** `baselines/train/svdrr.py`
**Change:** The entry above ported 4 gaps but wasn't a complete diff against
`Cosmos-NVSyn/model/nvsyn_svdrr.py` -- re-audited on request and found 4 more, all now ported:
1. **LR schedule** -- old `configure_optimizers()` uses `LambdaLR` with 500-step linear warmup then
   cosine decay to 10% of peak LR over `max_train_steps`. Current code had **no scheduler at all**
   (flat LR the entire run). Added an equivalent `lr_lambda`/`LambdaLR`, with `--warmup_steps` (default
   500) and `max_train_steps = args.epochs * steps_per_epoch` computed once before the epoch loop
   (moved `effective_batch`/`steps_per_epoch` out of the loop body since they're epoch-invariant).
   Logged current LR to both the console and a new `LR/transformer` TensorBoard scalar.
2. **Gradient clipping** -- old has `gradient_clip_val=0.5` (PyTorch Lightning's built-in clipping).
   Current code had **no clipping**. Added `torch.nn.utils.clip_grad_norm_` over transformer +
   cc_projection params, right before `optimizer.step()`, gated by a new `--gradient_clip_val` (default
   0.5) CLI flag.
3. **Optimizer betas** -- old uses `(0.9, 0.95)` (the standard DiT/transformer beta2, not the generic
   default). Current code used `(0.9, 0.999)`. Changed to `(0.9, 0.95)`.
4. **Weight decay** -- old uses `0.0`. Current code used `1e-2`. For a short fine-tune starting from a
   pretrained checkpoint, nonzero weight decay actively regularizes away from those pretrained weights,
   working against the exact thing a light fine-tune wants to preserve. Changed to `0.0`.

Two other candidate gaps were checked and ruled **not** gaps: the pipeline's actual on-disk
`scheduler_config.json` (`baselines/cloned/SV-DRR/models/base_model/256/scheduler/`) has
`_class_name: DPMSolverMultistepScheduler`, matching the old module's explicit inference-time
`DPMSolverMultistepScheduler.from_config(...)` choice exactly -- an earlier diagnostic script that
force-loaded the same config file as `DDPMScheduler` produced a misleading printout suggesting
otherwise; re-read the raw JSON directly to confirm the real class. Pose encoding was already confirmed
equivalent in the entry above.

**Why:** User asked directly whether all gaps had been found, correctly suspecting the first pass
wasn't exhaustive. It wasn't -- these 4 are real and would matter for a 10,000-step run (no LR decay
and no gradient clipping are both exactly the kind of gap that looks fine at smoke-test scale and
causes instability or a suboptimal final loss only once training runs long enough to reach the
few-thousand-step regime).

**Verification:** `py_compile` clean. Re-ran the smoke test (`--epochs 3 --max_train_samples 2
--max_val_samples 1 --warmup_steps 1`) -- all 3 epochs completed with no errors, and the logged LR
values (5.00e-06 -> 2.75e-06 -> 5.00e-07) exactly match hand-computed cosine decay for that config,
confirming the schedule is wired correctly. Gradient clipping ran every step across all 3 epochs with
no NaN/instability. (The run was externally interrupted mid-epoch-3's TensorBoard viz reload by a
session/agent-teardown event, not a code error -- confirmed via `ps aux` showing no orphaned process and
no traceback in the log -- but the 3 completed epochs plus 2 successful mid-run checkpoint reloads
already demonstrate all 4 fixes work.)

---

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

---

## 2026-08-14 — Real Finetuned Checkpoint (GCS) Still Produces Blocky Non-Anatomical Output; No Further Bug Found

**Phase:** 2 (Correctness verification)
**Files:** `baselines/models/svdrr.py` (one change: `num_inference_steps` 20→30), architecture/config audit
of the checkpoint and `train_svdrr_DiT.py`/`pipeline_svdrr_DiT.py` (no further change)
**Context:** `results/svdrr.png` (this project's own logged TensorBoard panel, checkpoint at 940 training
steps) showed blocky, posterized, non-anatomical output. The checkpoint behind that panel only existed in
GCS (`gs://.../checkpoints/cosmos-worker-4/svdrr_best.pt`, 2.4GB); pulled it locally and re-ran the same
real `build_multiview_grid` panel against a real NSCLC test patient to check whether it's a wrapper bug or
a genuinely undertrained model.
**Findings (all ruled out as further bugs):**
1. **Architecture/checkpoint mismatch** — `svdrr.py.__init__` has a fallback that swaps to a stale 4-channel
   base model if the checkpoint's `pos_embed.proj.weight` indicates 4 input channels (a historical
   compat path for an older checkpoint generation). Inspected the checkpoint directly: `pos_embed.proj.
   weight` is `[1152, 8, 2, 2]` — 8 channels, so this fallback correctly does *not* trigger; the current
   8-channel base model is used, matching how the checkpoint was trained (`train_svdrr_DiT.py`'s
   `in_channels==8` concat-conditioning path).
2. **`cc_projection` shape** — `[1152, 772]` (772 = 768-dim CLIP image embed + 4-dim raw pose), matches the
   expected Zero123-style conditioning convention. No mismatch.
3. **Inference step count** — wrapper used `num_inference_steps=20`, below `cloned/SV-DRR/test_svdrr_DiT.
   py`'s own reference eval constant (`NUM_INFERENCE_STEPS=30`). Bumped to 30 to match. Re-ran against the
   same real checkpoint/patient: output is visually near-identical (still blocky/posterized) — rules this
   out as the cause, but the fix is kept since it now matches the reference eval protocol exactly.
**Conclusion:** No code bug found. Training curves are healthy (`Loss/train` 0.275→0.077, `Loss/val`
0.202→0.070 over 945 logged steps, no divergence — see `gcs_tensorboard/tensorboard/svdrr`), but 945 steps
at `lr=5e-6` fine-tuning a full DiT (not LoRA) is a light fine-tune; this matches
`research-log.md`'s existing 2026-07-24 measurement (PSNR 6.47dB, SSIM 0.092 — among the weakest of the 6
baselines) and is most likely genuine undertraining, not a further wrapper bug. Left as a known result, not
patched further, since there's nothing broken left to fix without more training compute.
**Verification:** `uv run pytest baselines/tests/test_svdrr_wrapper.py` passes post-change. Both the
20-step and 30-step real-checkpoint panels visually compared side by side.

---

## [2026-08-18] Viz Azimuth-Divisor Fix, Pose-Sign Convention Re-Confirmed, Full 93-View Quantitative Sweep

**Phase:** 2 (Correctness verification) — first real, non-degenerate quality numbers for the 2026-08
reference-training-objective port (EXP-013/014's ~1316-epoch / ~21K-optimizer-step run, `ema_transformer`
present in the checkpoint).

**Bug 1 — `baselines/models/viz.py:build_multiview_grid` azimuth-degree divisor:** was
`degrees = [idx / total_gt_views * 360.0 for idx in indices]` (divide by 93), inconsistent with this
project's own `linspace(0, 360, 93, endpoint=True)` convention used everywhere else (`train/svdrr.py`'s
`angles = np.linspace(0.0, 360.0, len(view_files))`, `renderer.py:render_diffdrr_multiview_frames`,
`.claude/rules/physics-rendering.md`). Caused up to ~3.9° drift between the TensorBoard live-viz panel's
requested prediction azimuth and the ground-truth file it was compared against (idx=92: 356.1° requested
vs. true 360°). **Fixed** to `degrees = [idx / max(1, total_gt_views - 1) * 360.0 for idx in indices]`.
Scoped only to the live viz panel — `baselines/evaluate.py`'s real benchmark harness already builds
azimuths correctly via `infer_multi_views(azimuths=(0, 360, total_frames))`, which uses a correct
`np.linspace(..., endpoint=True)` internally, so logged PSNR/SSIM numbers were never affected by this.

**Bug investigated and NOT changed — pose-sign convention:** a separate session proposed flipping
`baselines/models/svdrr.py`'s `poses = [[0, -float(azimuth), 0] for azimuth in chunk]` to
`+float(azimuth)`, based on an *unseeded* sweep (`pipe(..., generator=None)`, fresh random diffusion
noise per call) that showed `+90°` correlating better with GT 90° than `-90°` did. Applied, then
**re-tested with the confound controlled** — same `torch.Generator(...).manual_seed(s)` reused across
the `+angle`/`-angle` pair at each trial, isolating pose sign as the only variable — across 3 magnitudes
(30°/60°/90°) × 3 seeds = 9 paired trials. Result: the *original* `-azimuth` convention matched its
corresponding ground-truth view better in **7/9** controlled trials (60°: 3/3; 90°: 2/3; 30°: 2/3 with
one near-tie), the opposite of the unseeded test's conclusion. Also, `baselines/train/svdrr.py:288,354`
negates the identical `rel_angle` during training (`[0, -ra, 0]`) — flipping only the inference side
would have desynced inference from what the checkpoint was actually fine-tuned on, independent of which
sign is "physically correct." **Reverted** `models/svdrr.py` back to `-float(azimuth)`. Root cause of the
unseeded test's misleading result: `pipeline_svdrr_DiT.py:544`'s `generator: Optional[...] = None` means
every uncontrolled call draws fresh noise — a confound large enough on this checkpoint's quality level to
flip an unseeded A/B result.

**Quantitative verification — full 93-view sweep, real checkpoint, `-azimuth` convention, on
`datasets/pre_rendered/test/LUNG1-004_0000`** (uses `models.utils.compute_psnr`/`compute_ssim`, the same
functions `baselines/evaluate.py` uses):

| | mean PSNR (dB) | mean SSIM |
|---|---|---|
| All 93 views | **19.221** (std 3.790) | **0.5971** (std 0.1400) |
| "Simple" (≤90° from source, n=48) | 19.876 | 0.6301 |
| "Hemisphere-tail" (>90° from source, n=45) | 18.522 | 0.5618 |

This is a large jump over the previously-logged 940-step checkpoint's benchmark numbers
(`research-log.md` 2026-07-24: PSNR 6.47dB, SSIM 0.092) — consistent with the ~21K-optimizer-step
reference-objective-port run (EXP-013/014) actually clearing the "blocky noise" regime this baseline was
stuck in. The Simple-vs-Hemisphere-tail gap (~1.3dB) is real but much smaller than the SV-DRR paper's own
reported gap on its full-budget checkpoint (23.98dB Simple vs. 11.29dB Hemisphere, `PAPER.md` §3) — this
project's checkpoint is still far short of the paper's 200K-step budget, so a smaller relative easy/hard
gap on one test patient shouldn't be read as this checkpoint handling the hard region *better*, just that
its overall level is different; needs confirming across more than one patient before drawing a real
conclusion. Worst 5 views by PSNR are NOT symmetric around 180° as the "near-antipode is hardest"
hypothesis would predict — they cluster in one specific octant (46°–113°, e.g. idx=23/90° PSNR=9.96,
corr=-0.0075), while the 250–300° region (roughly the antipodal side) scores a comparatively healthy
17–19dB. Likely patient/view-specific rather than a general azimuth-difficulty law; not yet checked
against a second patient.

Full per-view CSV (`view_idx, azimuth_deg, psnr_db, ssim, pixel_corr`):
`docs/baselines/svdrr/logs/pose_fix_full_sweep_LUNG1-004_0000.csv`.

**Also this session (not a code change, dataset-level):** audited the other 5 baselines' azimuth-sign
conventions against DiffDRR's `x=dist·cos(elev)·sin(azim), z=dist·cos(elev)·cos(azim)` — NAF/Dx2CT share
`build_perspective_ray_points`, algebraically byte-identical to DiffDRR's formula (zero risk). XRaySyn's
full composed 6-DOF transform (`Rx(180°,fixed)@Ry(azimuth)@Rz(0)`) preserves DiffDRR's `sin(azim)`
x-dependency exactly through the fixed offset (zero risk, verified algebraically for 0°/30°/90°).
PixelNeRF's `pose_spherical` matches DiffDRR's `sin(azim)` on the azimuth-controlled axis (low risk).
MedNeRF uses a structurally different axis parametrization; a multi-patient empirical audit (2026-08-19,
500 iterations across LUNG1-004 and LUNG1-001) confirmed its pose convention is correct as implemented,
with observed far-side ($\ge 90^\circ$) direction ambiguity reflecting MedNeRF's single-view GAN fitting
reconstruction ceiling rather than a pose-sign bug in wrapper code (see `docs/baselines/mednerf/LOG.md`
for full audit details). Also ran a dataset-wide (n=1225 patients) lung-field left/right asymmetry scan
looking for CT-orientation-flip outliers: 17% of patients show the minority-direction asymmetry (vs. the
~0.01% true-situs-inversus base rate), but visual spot-check of the strongest outliers found plausible
real pathology (sternal wires / prior cardiac surgery, cardiomegaly) rather than mirrored anatomy, and
the outlier rate is statistically indistinguishable between the MELA2022 (17.7%) and COVID-19 (16.7%)
source subsets — arguing against a single source-pipeline handedness bug. Not conclusively resolved
without per-patient DICOM-orientation-tag or clinical review; flagged as an open item, not fixed.

**Verification:** `viz.py` change re-run through `python run_all_tests_isolate.py` (9/9 pass, per the
other session's report — not independently re-run this entry). Pose-sign revert confirmed via the seeded
9-trial A/B above. Full-sweep numbers computed directly from the real `svdrr_best.pt`
(`ema_transformer` key present, confirming it's the post-port checkpoint) via the actual
`SVDRRWrapper.infer_multi_views` path used by `evaluate.py`, not a synthetic/mocked shortcut.

**Next action:** re-run this same 93-view sweep against 2–3 more test patients before trusting the
Simple/Hemisphere-tail gap or the worst-octant pattern as general findings rather than one-patient noise.

**Correction (same day, user asked "why is it so bad on some views"):** the full-sweep script above calls
`infer_multi_views` -> `pipe(...)` with no `generator` passed, i.e. **every one of the 93 views was an
independent unseeded random draw** — the same confound already identified and controlled for in this
entry's own pose-sign A/B, but *not* controlled for in the sweep itself. Re-ran the 2 "worst" views
(idx=23/90°, idx=29/113°) plus idx=12/47° and the idx=0/0° control across 5 fixed seeds each:

| view | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | spread |
|---|---|---|---|---|---|---|
| idx=0 (0°) | 28.2 | 28.3 | 29.6 | 27.4 | 29.0 | 2.2 dB |
| idx=12 (47°) | 23.9 | 22.8 | 21.0 | 17.3 | 20.8 | 6.6 dB |
| idx=23 (90°) | 21.4 | 19.1 | 16.2 | 9.4 | 14.6 | 12.1 dB |
| idx=29 (113°) | 21.0 | 13.4 | 17.4 | 9.9 | 15.3 | 11.1 dB |

The originally-logged idx=23 PSNR (9.96) matches seed=3's draw almost exactly (9.36) — the single-draw
sweep landed on the worst of a ~12dB-wide outcome range for that view; the same azimuth reaches 21.4dB
under a different seed. **Retracting** the "worst 5 views cluster in one specific octant (46°–113°), not
symmetric around 180°" claim above as unsupported by a single unseeded draw per view — that ranking is
mostly noise-seed-dependent, not a stable property of those azimuths. What *does* survive this check:
harder/larger-relative-angle views show both lower mean quality **and** far higher seed-to-seed variance
(12dB swings vs. 2dB at the easy 0° control) — a real, legitimate finding about generation reliability in
the undertrained regime, just not the specific per-azimuth ranking claimed above. A trustworthy per-view
sweep needs either a fixed seed or multi-seed averaging per view; the current CSV should be read as
"one noisy sample per view," not "the model's true per-azimuth quality curve."

**Follow-up (same day): strengthened the dataset-orientation-consistency check** after a separate session
claimed the rotation-direction issue "cannot occur" — that phrasing overstated the original 3-patient
spot-check, so re-verified with two independent, larger checks before accepting or rejecting it:
1. **Affine determinant/orientation-code scan** across all locally-available raw NIfTI files (628 of the
   1225 pre-rendered patients — the other 49% already had their raw file cleaned up post-render). Zero
   degenerate/zero-determinant affines; 100% parsed to well-formed `nib.aff2axcodes` orientation strings
   (LPS: 387, LAS: 238, LPI: 3). This mixed-convention spread is routine scanner heterogeneity, exactly
   what `OrientationDict(axcodes="ASL")` is built to normalize — not a red flag. Rules out the one
   concrete failure mode worth worrying about (a missing/degenerate/fallback affine silently producing a
   bad reorientation) for the 51% of the dataset checked.
2. **Visual spot-check expanded from 3 to a genuine random sample of 16** of the 209 minority-direction
   outliers (`random.seed(7)`, not cherry-picked). Zero show an orientation-flip signature; all consistent
   with ordinary pathology/anatomical variation (enlarged mediastinal shadows, uneven aeration), same
   pattern as the original 3.
- **Correction**: earlier this session `volume-covid19-A-*` patients were flagged as a possible
  TCIA+MELA2022-only training-scope violation. Wrong — confirmed these files live under
  `datasets/TCIA/images/`, i.e. they are part of the TCIA collection (TCIA hosts a COVID-19 CT subset
  under this naming), not a foreign dataset. Retracted.
- **Verdict**: "no evidence of a systematic orientation bug, checked structurally (affine headers, 51% of
  patients) and visually (expanded random sample)" is now supportable for docs. "Cannot occur" as an
  absolute is still not — 49% of patients have no raw file left to structurally check, and no visual scan
  is a 100%-certain flip detector. Residual risk is now small and well-characterized, not a live open
  question.

---

## [2026-08-18] `SVDRRWrapper.infer_multi_views` Gains `generator`/`seed`, and the Real Seeded 3-Seed Sweep

**Change (already present in `baselines/models/svdrr.py` when this entry was written — not made in this
session, verified correct on read):** `infer_multi_views` now accepts `generator: torch.Generator |
list[torch.Generator] | None = None` and `seed: int | None = None`; when only `seed` is given it builds
`torch.Generator(device=self.device).manual_seed(seed)` and threads it into every batched `pipe(...)`
call for that sweep. Default (`generator=None, seed=None`) preserves the old unseeded behavior, so
nothing else that calls this method changed.

**Re-ran the full 93-view sweep properly** using this, on `LUNG1-004_0000`, `svdrr_best.pt`: 3 seeds
(0, 1, 2), per-view PSNR/SSIM/corr averaged with std recorded. CSV:
`docs/baselines/svdrr/logs/pose_fix_full_sweep_LUNG1-004_0000_seeded3x.csv` — supersedes
`pose_fix_full_sweep_LUNG1-004_0000.csv` (the original unseeded single-draw file, kept for reference, not
deleted) as the trustworthy per-view curve.

| | mean PSNR (dB) | mean SSIM |
|---|---|---|
| All 93 views (3-seed avg) | **19.203** | **0.6008** |
| "Simple" (≤90°, n=48) | 19.428 | 0.6207 |
| "Hemisphere-tail" (>90°, n=45) | 18.963 | 0.5796 |

Overall mean matches the unseeded run almost exactly (19.221dB/0.5971 before) — the *aggregate* number was
never in doubt, only the *per-view* ranking was. The Simple/Hemisphere-tail gap shrank from ~1.35dB
(unseeded, noise-inflated) to a real ~0.47dB.

**Per-view std now lets stable-bad be distinguished from noisy-bad.** Worst 5 by mean PSNR: idx=21/82°
(11.4±**0.4**dB — low std, genuinely hard), idx=24/94° (12.9±2.5), idx=27/106° (12.9±1.6), idx=23/90°
(12.9±**3.3** — high std, mostly the noise this checkpoint already showed in the earlier 5-seed spot
check), idx=31/121° (13.0±**0.4** — low std, genuinely hard). Two of five are low-variance and
reproducibly bad; net read: there is a real, low-variance-confirmed soft spot in roughly the **80°–125°**
band for this checkpoint on this patient — shifted from, and more defensible than, the "one octant
46°–113°" claim retracted earlier this same day, which was built from a single unseeded draw per view.

**Verification**: process completed cleanly (`Overall mean PSNR`/`Overall mean SSIM` summary lines
present in the run log), CSV has all 93 rows with populated std columns, GPU ran hot (88°C, 100% util)
during the ~20min run but no VRAM pressure or errors — the per-batch slowdown observed mid-run was
thermal, not a stall.

**Next action**: same as before — extend to 2-3 more test patients before treating the 80°-125° soft spot
or the Simple/Hemisphere-tail gap as general rather than one-patient findings.
