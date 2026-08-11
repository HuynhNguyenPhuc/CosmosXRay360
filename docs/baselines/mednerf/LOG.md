# MedNeRF Change & Fix Log

## 2026-08-03 — Initial MedNeRF Integration & Value-Space Target Scaling Fix

**Phase:** 1 & 2 (Import & Correctness)  
**Files:** `baselines/models/mednerf.py`, `baselines/train/mednerf.py`, `baselines/cloned/mednerf/graf-main/submodules/torchsearchsorted.py`  
**Change:** Created `MedNeRFWrapper` and `train_mednerf` training script. Substituted PyTorch native `torch.searchsorted` for legacy C++ `torchsearchsorted` extension. Rescaled `target_xr` in `fit_latent_and_weights` from `[0, 1]` to `[-1, 1]` (`target_xr * 2.0 - 1.0`).  
**Why:** Fixes C++ extension compile failures on PyTorch 2.x/CUDA 12+ and eliminates value-space mismatch gradients (MedNeRF generator outputs `[-1, 1]` via `tanh`).  
**Verification:** Verified in `baselines/tests/test_loss_regression.py` (`test_mednerf_target_scaling`).

---

## 2026-08-05 — PAPER.md/CODE.md Corrected Against the Actual Cached PDF

**Phase:** 0 (Resource acquisition / correctness of docs, not code)
**Files:** `docs/baselines/mednerf/{PAPER.md,CODE.md}`
**Change:** The 2026-08-05 docs backfill pass used a slightly wrong paper title and, more
substantively, omitted the paper's actual stated contribution over GRAF (§II-C's self-supervised
auto-encoding discriminator + DAG multi-head training) while stating test-time loss coefficients
($\lambda_{\text{MSE}}=1.0$, $\lambda_z=0.01$) that don't match the paper's real reported values
(Eq. 10: $\lambda_1{=}0.3,\lambda_2{=}0.1,\lambda_3{=}0.3$, lr=0.0005, $\beta_1{=}0,\beta_2{=}0.999$).
Notably, **this project's own `fit_latent_and_weights` already uses the real paper's exact
coefficients** (see the 2026-08-05 entry below) — only the documentation had it wrong, not the code.
`CODE.md` was largely accurate but under-described the discriminator (called it a generic "PatchGAN"
when `SimpleDecoder`, confirmed real in `discriminator.py`, is literally the paper's own "SD: Simple
Decoder" self-supervision component from Fig. 2). Rewritten directly from
`arxiv_papers/MedNeRF_2202.01020.pdf`.
**Why:** Same failure mode as the other 2026-08-05 correction entries across `docs/baselines/`.
**Verification:** Title/authors/venue/equation numbers/coefficients cross-checked page-by-page
against the PDF; `Discriminator`/`SimpleDecoder` class names re-verified with `grep`; license
(MIT, copyright Jonathan Frawley — one of the paper's actual authors) verified against the real
`LICENSE` file, and was already correct.

---

## 2026-08-05 — Reference Optimizer Alignment, Gaussian Prior NLL & LPIPS Caching

**Phase:** 2 & 3 (Correctness, Acceleration & Convention Alignment)  
**Files:** `baselines/models/mednerf.py`, `baselines/train/mednerf.py`  
**Change:** Updated `z_optim` in `fit_latent_and_weights` to `Adam(lr=0.0005, betas=(0.0, 0.999))` and added Gaussian NLL prior penalty `0.3 * nll` on $z$, matching reference `render_xray_G_Z.py` exactly. Cached `lpips.PerceptualLoss` process-wide as `_PERCEPT_LOSS` and added `--val_iters_per_sample`. Switched target angle generation from `endpoint=False` to `endpoint=True`.  
**Why:** Aligns test-time latent fitting with `render_xray_G_Z.py`, prevents $z$ from drifting far from the standard-normal prior, eliminates AlexNet reload I/O overhead, and aligns azimuth angles with ground-truth DiffDRR projections (`views/*.png`).  
**Verification:** `MedNeRFWrapper` test-time fitting executes stably and azimuth indices match `views/*.png`.

---

## 2026-08-07 — Phase 3: Opt-In Mixed Precision for Training Only (Not Inference)

**Phase:** 3 (Acceleration)
**Files:** `baselines/models/mednerf.py`, `baselines/train/mednerf.py`
**Change:** Added `use_amp: bool = False` to `fit_latent_and_weights`, wrapping the forward pass
(generator + LPIPS) in `torch.autocast` with a `torch.amp.GradScaler` shared across both
`z_optim`/`g_optim` (`scaler.scale(loss).backward()` once, `scaler.step()` each optimizer,
one `scaler.update()`). Threaded through as `--amp` in `baselines/train/mednerf.py`, wired only to
the *training* fit call -- validation's internal trend-tracking fit and, critically,
`MedNeRFWrapper.infer_multi_views` (which produces this project's actual reported benchmark numbers)
are untouched and stay fp32 by default.
**Why:** Considered whether MedNeRF's per-scan fitting had the same `exp()`-overflow risk that caused
a real NaN bug in XRaySyn (see `docs/baselines/xraysyn/LOG.md`'s 2026-08-07 entry) before enabling
this. Checked the actual alpha-compositing formula this method uses
(`baselines/cloned/mednerf/graf-main/submodules/nerf_pytorch/run_nerf_mod.py:238`,
`1 - exp(-relu(raw)*dists)`): the exponent is always `<= 0` (density is ReLU'd non-negative, `dists`
is a positive ray-step distance), so `exp()` is bounded in `(0, 1]` and structurally cannot overflow
under reduced precision -- a fundamentally different, safer shape than XraySyn's unbounded
`exp(+atten_proj)`. This gave enough confidence to try AMP here, though the LPIPS/generator layers
themselves weren't individually audited for fp16 underflow, hence defaulting to off and requiring an
explicit `--amp` opt-in.
**Verification:** Local smoke test, 2 train / 1 val patient, `--iters_per_sample 20`: both fp32 and
`--amp` runs complete with real (non-`nan`, non-crashing) loss values in the same order of magnitude
(0.358/0.481 fp32 vs 0.435/0.508 with `--amp` -- not a same-seed apples-to-apples comparison, `z` is
randomly initialized each run, so some divergence is expected regardless of precision). Measured
wall-clock: 44s (fp32) -> 26s (`--amp`) for the identical workload, a real ~1.7x speedup. This is a
small-scale smoke test, not a full-dataset regression -- treat any benchmark number produced with
`--amp` as provisional until a real full-scale run confirms output quality isn't measurably affected.
