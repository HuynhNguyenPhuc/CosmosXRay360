# Cosmos 3 (`predict3`) for CosmosXRay360

**Status (2026-08-17):** Backbone-swap implementation landed and smoke-tested
on a real GCP A100 worker. P0 (env setup) and P4 (`--dryrun` schema
validation, which also caught and fixed a real Hydra override bug) both
PASS. P3 (zero-shot inference) is currently BLOCKED by an external
package-ecosystem gap, not a bug in this repo: `diffusers==0.39.0` (current
PyPI release) doesn't recognize several config fields on
`nvidia/Cosmos3-Edge`'s checkpoint, silently randomly-initializing ~120
attention/MLP weights instead of loading them; the unreleased `diffusers`
dev build that does understand the checkpoint requires a `huggingface-hub`
major version `transformers` doesn't yet support. See the 2026-08-17 GCP
smoke-test entry in [`LOG.md`](LOG.md) for full details and exact error
traces. See [`PLAN.md`](PLAN.md) for the full migration plan and
phase-by-phase status.

> **This file replaces an earlier draft** that described a fictional
> "AR Reasoner + DM Generator" architecture built on `cosmos_predict2._src.*`
> internals (i.e. Cosmos-Predict2.5's own DiT, re-labeled). That draft was
> never actually wired to NVIDIA's real Cosmos 3 release. Everything below
> describes what the code in `predict3/` actually does today, verified
> against the installed `diffusers==0.39.0` `Cosmos3OmniPipeline` and the
> `cosmos-framework` submodule's real source.

## 1. What Cosmos 3 actually is

Cosmos 3 (NVIDIA, June 2026, [arXiv:2606.02800](https://arxiv.org/abs/2606.02800))
is a Mixture-of-Transformers (MoT) omnimodal world model that subsumes the
old Predict/Transfer/Reason split into one model. There is no
`cosmos-predict3` repo -- the ecosystem lives at
[`NVIDIA/cosmos`](https://github.com/NVIDIA/cosmos) (checkpoints, docs) and
[`NVIDIA/cosmos-framework`](https://github.com/NVIDIA/cosmos-framework)
(training/inference code, vendored here as the `cosmos-framework/` submodule).

- **Towers:** a causal AR Reasoner (language + ViT vision tokens) and a
  bidirectional DM Generator (VAE video/audio/action tokens), sharing one
  self-attention operator; AR tokens are never updated from DM tokens.
- **Checkpoints:** `nvidia/Cosmos3-Edge` (4B), `nvidia/Cosmos3-Nano` (16B),
  `nvidia/Cosmos3-Super` (64B). OpenMDW-1.1.
- **Tokenizer:** Wan2.2-TI2V-5B VAE, 4x temporal / 16x spatial + a 2x2 patch
  merge in the transformer = 32x effective spatial token compression.
- **BF16-only, Ampere+.** This repo's local TITAN RTX (Turing) cannot run it.

## 2. `predict2_5` vs `predict3`

| Component | `predict2_5` (shipped) | `predict3` (this migration) |
| --- | --- | --- |
| Network | `MinimalV1LVGDiT` 2B (own DiT, `cosmos_predict2._src.*`) | `Cosmos3OmniTransformer` inside `diffusers.Cosmos3OmniPipeline` (Cosmos3-Edge 4B) |
| Tokenizer | Wan2.1 VAE, 8x spatial | Wan2.2 VAE, 16x spatial + 2x2 patch merge |
| Text conditioning | `CR1TextEncoder`, 100352-d full-layer-concat cross-attn | Native to the pipeline; structured-JSON captions |
| Frame-0 anchor conditioning | Hand-rolled per-step latent/velocity splice (`denoise()`) | Native I2V: clean latent concatenated ahead of noisy target, re-applied every step by the transformer itself |
| Training | Lightning + `FSDPStrategy`, custom EMA/optimizer | `cosmos_framework.scripts.train`, TOML recipe + Hydra experiment SKU, DCP checkpoints |
| Inference | Hand-rolled UniPC loop (`Inferencer`) | `Cosmos3OmniPipeline.__call__` (`InferencerV3`) |
| Physical losses (`L_angle_rf`, `L_atten`) | `predict2_5/module.py:_compute_physical_losses` | Ported to backbone-agnostic `predict3/losses.py`; not yet wired into a training loop (P5, gated on a measured plain-SFT baseline) |

## 3. `predict3/` package structure

```
predict3/
├── __init__.py           # No import-time mocks needed (diffusers imports cleanly)
├── constants.py          # NUM_FRAMES=93, Wan2.2 VAE geometry, real nvidia/Cosmos3-* HF repo ids
├── losses.py             # angular_weighted_velocity_loss, attenuation_mass_loss (pure functions)
├── inferencer.py         # InferencerV3: thin wrapper around Cosmos3OmniPipeline
├── recipes/
│   └── xray360_edge.toml # cosmos-framework SFT recipe (thin diff of vision_sft_edge.toml)
└── tests/
    ├── test_predict3.py          # constants + InferencerV3 CPU-only helpers
    ├── test_losses.py            # physics-loss math
    └── test_sft_dataset_build.py # dataset builder round-trip + OOD leakage guard
```

Training (`cosmos_framework.scripts.train`) and dataset conversion
(`cosmos_framework.scripts.captions_to_sft_jsonl`) run out of the
`cosmos-framework/` submodule directly -- there is no `predict3/trainer.py`
or `predict3/datamodule.py`; unlike `predict2_5`, this backbone doesn't
need bespoke Lightning training-loop code (see `PLAN.md` P1/P4 for why).

## 4. Quickstart

### 4.1 Zero-shot inference (no post-training)

Requires an Ampere+ CUDA GPU -- this raises immediately on unsupported hardware:

```python
from PIL import Image
from predict3.inferencer import InferencerV3

inferencer = InferencerV3()  # defaults to nvidia/Cosmos3-Edge, zero-shot
image = Image.open("sample_xray.png").convert("RGB")
frames = inferencer.predict(image=image, num_inference_steps=35, guidance_scale=6.0, seed=42)
print(f"Generated {len(frames)} frames of shape {frames[0].shape}")
```

```bash
python app.py --backbone predict3 --share   # zero-shot, or --checkpoint-path <post-trained dir>
```

### 4.2 Build the SFT dataset (local, no GPU needed)

```bash
uv run python scripts/build_cosmos3_sft_dataset.py \
    --pre-rendered-dir datasets/pre_rendered \
    --output-dir datasets/cosmos3_sft
```

Converts `datasets/pre_rendered/train/<patient>/views.pt` into
lossless-for-this-content MP4s + structured captions, then invokes
`cosmos_framework.scripts.captions_to_sft_jsonl` to produce
`datasets/cosmos3_sft/train/video_dataset_file.jsonl`. TCIA/MELA2022 only
-- an OOD leakage guard refuses to run if any patient id collides with the
held-out NSCLC test split.

### 4.3 Post-training (GCP A100 worker only)

```bash
bash scripts/setup_predict3_env.sh          # once, on the worker
bash scripts/launch_cosmos3_worker.sh --max-iter 10000
```

See `PLAN.md` P4 for the checkpoint-conversion prerequisites
(`convert_model_to_dcp`) and the `xray360_edge.toml` recipe's design
rationale (why it's a thin diff of the stock recipe rather than a new
Hydra experiment SKU).

## 5. What's not done yet

- **P3 zero-shot eval is blocked**, not merely "not run yet" -- see the
  Status line above and `LOG.md`'s 2026-08-17 GCP smoke-test entry. It
  needs either a `diffusers` release past `0.39.0` that supports
  `nvidia/Cosmos3-Edge`'s current checkpoint config, or a
  `huggingface-hub` release that satisfies both the `diffusers` dev build
  and `transformers` simultaneously. Retry `InferencerV3()` once one of
  those lands.
- **P4 full training run** (as opposed to `--dryrun` schema validation,
  which passed) has not been executed -- needs a 4x A100-80GB worker and
  real GPU-hours, out of scope for a smoke test.
- **P5 physics losses** are ported (`predict3/losses.py`, unit-tested) but
  not wired into a Cosmos 3 training loop -- doing so first requires
  verifying cosmos-framework's flow-matching sigma convention matches the
  one `attenuation_mass_loss` assumes (see the module docstring and
  `PLAN.md` P5 "convention check").
- **Any-to-any relative-view training** (source view != 0-degree PA) is not
  implemented. `PLAN.md` P5 describes it as a dataset-layer change (roll the
  93-frame orbit by a random offset) once the plain-I2V baseline is measured.
