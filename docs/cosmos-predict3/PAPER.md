# Cosmos 3 Backbone: Formulation & Status

**Reference Manuals:** [`PLAN.md`](PLAN.md) (migration plan, phase status),
[`README.md`](README.md) (quickstart), `docs/cosmos-predict2.5/PAPER.md`
(the shipped method this backbone is being compared against).

> Supersedes an earlier draft of this file that presented a "Mixture-of-
> Transformers World Foundation Model Adaptation" paper outline with
> fabricated benchmark numbers (a "Target > 31.50 dB" PSNR row that was
> never measured) and a training objective (`L_MoT-NVS`, any-to-any
> relative-view learning) that no code in this repo implements. This
> version states only what is actually built and measured, and marks
> everything else as not yet done.

## 1. Task

Same task as `predict2_5`: synthesize a 93-frame, 0-360-degree rotation
video from a single 2D frontal chest X-ray, by post-training a pretrained
video/world model on paired CT-derived DRR rotation videos
(`datasets/pre_rendered/`). This document covers only what differs when the
backbone is Cosmos 3 instead of Cosmos-Predict2.5.

## 2. Architecture (as actually used)

Cosmos 3's `Cosmos3OmniTransformer` (inside `diffusers.Cosmos3OmniPipeline`,
`nvidia/Cosmos3-Edge`) is a Mixture-of-Transformers: a causal "understanding"
stream (text + optional vision-language tokens) runs in parallel with a
bidirectionally-attended "generation" stream (VAE video/audio/action
latents), joined by 3D multimodal RoPE. `predict3/` does not reimplement any
of this -- it is consumed entirely through `diffusers`' pipeline API
(inference) and `cosmos-framework`'s TOML/Hydra SFT stack (training); see
`PLAN.md` Sec. 3 for the exact class/module boundary.

### 2.1 Image-to-video conditioning (verified against the installed pipeline source)

Given a single conditioning image and `num_frames > 1`, `prepare_latents` in
`diffusers/pipelines/cosmos/pipeline_cosmos3_omni.py` builds the initial
latent as:

```
vision_condition_mask[0] = 1.0   # frame 0 anchored, all others 0
latents = mask * encode(image_repeated_as_video) + (1 - mask) * noise
```

and passes `vision_condition_mask` into the transformer at every denoising
step (not just at initialization) -- so frame-0 anchoring is enforced by the
model itself throughout sampling, not by pipeline-level post-hoc splicing.
This replaces `predict2_5.module.CosmosXRay360.denoise()`'s hand-rolled
per-step latent/velocity replacement (`predict2_5/module.py:588-661`) with
no code on our side; `predict3.inferencer.InferencerV3` just calls
`pipe(image=pa_view, num_frames=93, ...)`.

### 2.2 Diffusion objective

Flow matching with velocity prediction, UniPC-family sampler (matches
`predict2_5`'s rectified-flow / UniPC choice at a high level; the exact
sigma-schedule parameterization inside `cosmos-framework`'s training stack
has not yet been diffed against `predict2_5`'s `RectifiedFlow` -- see
Sec. 4 below, this is a prerequisite for porting the physical losses).

## 3. Physical & geometric loss regularizers (ported, not yet wired in)

`predict3/losses.py` carries over `predict2_5`'s two regularizers as pure,
backbone-agnostic functions (unit-tested in `predict3/tests/test_losses.py`):

- **`angular_weighted_velocity_loss`** -- `L_angle_rf`, weighting frame `k`'s
  velocity-matching error by `w(theta_k) = 1 + gamma_side * sin^2(theta_k)`,
  upweighting lateral/oblique views (worst line-of-sight ambiguity).
- **`attenuation_mass_loss`** -- `L_atten`, penalizing deviation of each
  view's VAE-latent "mass" (mean over channel/spatial dims) from the
  ground-truth 0-degree PA anchor's mass, an approximate Beer-Lambert
  total-attenuation-conservation constraint in latent space.

```
total = angle_rf_loss + loss_atten_weight * atten_loss     # loss_atten_weight default 0.02
```

Both assume the flow-matching convention `x_t = sigma*noise + (1-sigma)*data`,
`v = noise - data`, `x0 = x_t - sigma*v` (Cosmos-Predict2.5's
`RectifiedFlow`). **Before wiring these into a Cosmos 3 training loop, this
convention must be verified against `cosmos-framework`'s own flow
parameterization** -- a flipped sigma direction silently inverts
`attenuation_mass_loss` without raising an error (`docs/GOTCHAS.md`).
Not yet done; tracked as `PLAN.md` P5.

## 4. Any-to-any relative-view training

Not implemented. `predict2_5` and the current `predict3` I2V recipe both
always condition on the fixed 0-degree PA frame. A future increment
(`PLAN.md` P5) would roll each training clip's 93-frame orbit by a random
offset so the anchor frame's azimuth `theta_src` and the target-relative
offset `Delta theta = theta_tgt - theta_src` vary, removing the
frontal-view shortcut -- cheap to add since frame index already equals
azimuth by construction, but not built yet.

## 5. Post-training recipe

`predict3/recipes/xray360_edge.toml` is a byte-for-byte diff of
cosmos-framework's stock `vision_sft_edge.toml` (see the recipe's own header
comment for the full rationale): only `[job]` identity changes. The one
behavioral delta this task needs -- always exactly one PA-anchor
conditioning frame, vs. the stock recipe's 70% T2V / 20% I2V / 10% V2V mix
-- is applied as three Hydra CLI overrides, zeroing the T2V/V2V branches
individually (`+....conditioning_config.0=0.0`, `.1=1.0`, `.2=0.0`) rather
than one dict-literal override, since a single `conditioning_config={1:1.0}`
merges into the stock `{0:0.7,1:0.2,2:0.1}` instead of replacing it --
caught and fixed via `--dryrun` against the real schema on a GCP A100
worker (`docs/cosmos-predict3/LOG.md`) -- in `scripts/launch_cosmos3_worker.sh`, not a TOML/experiment-file change,
because `cosmos_framework.scripts.train` has no dynamic experiment-module
discovery (registration is a hardcoded import list in
`cosmos_framework/configs/base/config.py`) and reusing the already-registered
`vision_sft_edge` Hydra SKU avoids a ~300-line near-duplicate.

## 6. Benchmark

**Not yet measured.** Zero-shot Cosmos3-Edge and post-trained numbers on the
402-case NSCLC OOD test set both require a GCP A100 worker this repo's
local dev machine doesn't have (`PLAN.md` Sec. 2, Hard Constraints). No
placeholder or "target" numbers are recorded here on purpose -- see the
note at the top of this file for why the previous draft's fabricated table
was removed. Once P3 (zero-shot) and P4 (post-trained) runs complete, this
section should report the same PSNR/SSIM/LPIPS/Side-Arc-PSNR/Latency/Peak-VRAM
columns as `docs/cosmos-predict2.5/PAPER.md`'s benchmark table, appended as
new rows to `docs/BENCHMARK.md` rather than duplicated here.
