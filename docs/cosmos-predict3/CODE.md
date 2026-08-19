# Cosmos 3 Codebase & Architecture Mapping (`CODE.md`)

**Main Post-Training & Inference Package:** `predict3/`
**Foundation Submodule:** `cosmos-framework/` (training: `cosmos_framework.scripts.train`) +
installed `diffusers` package (inference: `diffusers.Cosmos3OmniPipeline`)
**Main Engine Class:** `predict3/inferencer.py` (`InferencerV3`)
**Real Checkpoints:** `nvidia/Cosmos3-Edge` (4B, primary target), `nvidia/Cosmos3-Nano` (16B), `nvidia/Cosmos3-Super` (64B)

> Supersedes an earlier draft that described `CosmosXRay360v3` (a Lightning
> `LightningModule`), `PreRenderedDataModuleV3`, and `InferencerV3` methods
> that built a `MinimalV1LVGDiT` on a meta device -- all of that was a
> renamed copy of `predict2_5/`'s classes, never actually touching Cosmos 3.
> Those files (`predict3/module.py`, `predict3/trainer.py`,
> `predict3/datamodule.py`) have been deleted; see `PLAN.md` P1.

## 1. System integration

```
                    scripts/build_cosmos3_sft_dataset.py
                    (local; reads datasets/pre_rendered/train/)
                                    |
                                    v
              datasets/cosmos3_sft/train/{videos/,captions/,video_dataset_file.jsonl}
                                    |
                                    v  (GCP A100 worker; cosmos-framework/'s own venv)
              cosmos_framework.scripts.train --sft-toml=predict3/recipes/xray360_edge.toml
                                    |
                                    v
                    DCP checkpoint -> export_model -> convert_model_to_diffusers
                                    |
                                    v
              predict3/inferencer.py: InferencerV3(checkpoint_path=<exported dir>)
                    -> diffusers.Cosmos3OmniPipeline.from_pretrained(...)
                                    |
                                    v
                    app.py --backbone predict3   /   baselines/evaluate.py
```

## 2. `predict3/` file-by-file

### `predict3/constants.py`

```python
COSMOS3_EDGE_REPO = "nvidia/Cosmos3-Edge"    # real HF repo id (not an invented UUID)
COSMOS3_NANO_REPO = "nvidia/Cosmos3-Nano"
COSMOS3_SUPER_REPO = "nvidia/Cosmos3-Super"
VAE_TEMPORAL_DOWNSAMPLE = 4
VAE_SPATIAL_DOWNSAMPLE = 16          # + a 2x2 patch merge in the transformer -> 32x effective
NUM_FRAMES = 93
NUM_LATENT_FRAMES = 24               # 1 + (93-1)//4
```

### `predict3/losses.py`

Pure, backbone-agnostic functions (no `self`, no framework coupling):

```python
def angular_offset_weights(num_frames, gamma_side=1.0, device=None, dtype=torch.float32) -> Tensor: ...
def angular_weighted_velocity_loss(v_pred, v_target, gamma_side=1.0, time_weights=None) -> tuple[Tensor, Tensor]: ...
def attenuation_mass_loss(v_pred, x_t, sigmas, x1_gt=None, anchor_frame_index=0) -> Tensor: ...
def total_physical_loss(v_pred, v_target, x_t, sigmas, x1_gt=None, gamma_side=1.0,
                         loss_atten_weight=0.02, time_weights=None, anchor_frame_index=0) -> dict[str, Tensor]: ...
```

Ported from `predict2_5/module.py:_compute_physical_losses` (`predict2_5/module.py:926`).
Not yet called by any training loop -- see `PAPER.md` Sec. 3 for the
sigma-convention prerequisite.

### `predict3/inferencer.py`

```python
class InferencerV3:
    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16) -> None: ...
    def predict(self, image, prompt=None, negative_prompt=None, num_frames=NUM_FRAMES,
                height=IMG_HEIGHT, width=IMG_WIDTH, num_inference_steps=35,
                guidance_scale=6.0, fps=24.0, seed=42) -> list[np.ndarray]: ...
```

- `__init__` raises `RuntimeError` immediately if `not torch.cuda.is_available()`
  -- Cosmos 3 is BF16-only/Ampere+ and there is no meaningful CPU fallback.
- `checkpoint_path` is either a local directory (a diffusers-format export,
  e.g. from P4's `convert_model_to_diffusers`) or a bare HF repo id; both are
  handled directly by `Cosmos3OmniPipeline.from_pretrained` -- there is no
  `hf://repo/file` single-file resolution here (unlike `predict2_5.hf`),
  because a diffusers pipeline is a whole-repo snapshot, not one weights file.
- `enable_safety_checker=False` at construction: DRR chest-rotation frames
  reliably trip Cosmos 3's guardrail and it isn't needed for this use case;
  passing `False` skips even downloading `nvidia/Cosmos-Guardrail1`.
- `predict()` calls `self.pipe(image=pil_image, num_frames=..., ...,
  output_type="np")` and converts the returned `[T, H, W, C]` float32 `[0,1]`
  array to a `list[np.ndarray]` of uint8 `(H, W, 3)` frames -- the same
  return contract as `predict2_5.inferencer.Inferencer.predict()`, so
  `app.py` and `baselines/evaluate.py` need no special-casing beyond the
  `--backbone` switch (`app.py`'s `get_or_load_model`/`generate_video`).
- No hand-rolled UniPC loop, no meta-device DiT construction, no
  `fix_rope_buffers`, no `Video2WorldCondition` -- all Predict-2.5-specific
  machinery with no equivalent needed here (frame-0 anchoring is native to
  the pipeline; see `PAPER.md` Sec. 2.1).

### `predict3/tower.py`

Documents, and makes CPU-testable, the AR Reasoner / DM Generator parameter split that
`xray360_edge.toml`'s `[optimizer] keys_to_select` implements at training time (see that
TOML's header comment). Every name is grep-verified against the real
`cosmos-framework` submodule source (`unified_mot.py`, `_convert_model_to_diffusers.py`)
-- not guessed. Also carries `NATIVE_TO_DIFFUSERS_ATTN_REMAP`, the authoritative table
(transcribed from `_convert_model_to_diffusers.py`'s `_ATTN_KEY_REMAP`) reconciling this
repo's native `*_moe_gen` naming with CosmosXRay2XRay's diffusers-side `add_*`/`proj_in`
naming -- both repos freeze the identical parameter set, described at two different
stages of the same checkpoint's life. See `docs/cosmos-predict3/LOG.md`'s 2026-08-19
entry for the full verification writeup.

```python
def is_generator_param(name: str) -> bool: ...                    # AR/DM classifier
def freeze_reasoner_tower(model) -> dict[str, int]: ...            # requires_grad_ toggler; not exercised against the real net locally
def parse_toml_keys_to_select(toml_text: str) -> tuple[str, ...]: ...  # drift-guard helper
KEYS_TO_SELECT: tuple[str, ...]                                    # single source of truth for xray360_edge.toml
NATIVE_TO_DIFFUSERS_ATTN_REMAP: dict[str, str]
```

### `predict3/recipes/xray360_edge.toml`

A thin diff of `cosmos-framework/examples/toml/sft_config/vision_sft_edge.toml`
-- only `[job]` (`project`/`group`/`name`) differs. `[job].experiment` stays
`"vision_sft_edge"` (the registered Hydra ConfigStore SKU name); see the
recipe file's own header comment and `PAPER.md` Sec. 5 for why no custom
Python experiment SKU was written.

## 3. `scripts/build_cosmos3_sft_dataset.py`

```python
def load_views(patient_dir: Path) -> np.ndarray: ...          # views.pt (float32 [0,1]) -> uint8 [T,H,W]
def write_lossless_mp4(frames_gray: np.ndarray, out_path: Path, fps: float) -> None: ...
def build_caption_json(fps: float) -> dict: ...                # frozen StructuredCaption template
def build_dataset(pre_rendered_dir, output_dir, fps=24.0, limit=None) -> Path: ...
def run_captions_to_sft_jsonl(train_dir: Path) -> Path: ...    # calls cosmos_framework's own converter
```

- **`load_views`**: `datasets/pre_rendered/<patient>/views.pt` is
  `float32` in `[0, 1]`, shape `[93, 256, 256]` (verified against the real
  dataset on disk -- not uint8, despite the file's binary appearance).
  Rounds (not truncates) to uint8.
- **`write_lossless_mp4`**: `-qp 0` libx264 on `yuv420p` is genuinely lossless
  for this content (constant chroma), but **only** with `-color_range pc`
  explicit on both encode and decode plus a `scale=in_range=full:out_range=full`
  filter -- without these, ffmpeg's default "limited range" (16-235)
  assumption silently remaps up to ~30 grayscale levels per pixel even at
  `-qp 0` (measured empirically: PSNR ~32dB without the flags vs. bit-exact
  with them). Verified bit-exact via both plain ffmpeg PNG extraction and
  `decord.VideoReader` (the likely SFT-loader video backend) in
  `predict3/tests/test_sft_dataset_build.py`.
- **`build_caption_json`**: one frozen `StructuredCaption`-shaped dict
  (matches `cosmos_framework/inference/structured_caption.py`'s pydantic
  schema), identical for every patient -- the synthesized task (a full
  360-degree DRR orbit) doesn't vary by anatomy, so there's no LLM
  captioning step and nothing to keep reproducible across runs beyond
  versioning this one template.
- **`run_captions_to_sft_jsonl`**: imports and calls
  `cosmos_framework.scripts.captions_to_sft_jsonl.main()` in-process
  (`sys.path`-inserted from this repo's main venv -- that converter's own
  deps, `tyro` + `ffprobe`, are light enough to not need the full
  `cu130-train` install). Produces `train/video_dataset_file.jsonl` in
  exactly the schema `vision_sft_edge`'s `get_sft_dataset(...)` reads via
  `${oc.env:DATASET_PATH}/train/video_dataset_file.jsonl` -- the official
  converter is reused rather than hand-authoring the JSONL schema, so the
  loader's own filters (max 61s duration, min 61 frames) and record shape
  stay correct even if `sft_dataset.py` changes upstream.
- **OOD leakage guard**: `build_dataset` raises `RuntimeError` before
  writing anything if any `datasets/pre_rendered/train/` patient id
  collides with a `datasets/pre_rendered/test/` (NSCLC) directory name.

## 4. `scripts/launch_cosmos3_worker.sh`

Runs on an already-provisioned worker (does **not** provision GCP
infrastructure itself -- unlike `scripts/launch_cosmos25_workers.sh`, which
drives a pinned GCP instance template; creating cloud GPU infra needs a
separately-reviewed template and has real billing impact, so it's out of
scope for this pass). Checks `.venv`, the JSONL, the DCP checkpoint dir, and
the Wan2.2 VAE file exist before launching; always appends three per-key
`conditioning_config.{0,1,2}=...` Hydra overrides (pure I2V, single PA
anchor -- a single dict-literal override merges instead of replacing, see
`PAPER.md` Sec. 5); `--max-iter N` / `--smoke` set `trainer.max_iter`.

## 5. `app.py` (`--backbone {predict2_5,predict3}`)

`get_or_load_model(checkpoint_path, config_path, device, backbone=...)`
branches the constructor call (`predict2_5.inferencer.Inferencer` needs
`config_path`; `predict3.inferencer.InferencerV3` doesn't) and the cache key
now includes `backbone`. `generate_video(...)` branches only the two
`.predict()` keyword names that differ between backbones (`cfg_scale`/
`num_steps` vs. `guidance_scale`/`num_inference_steps`); both return the
same `list[np.ndarray]`. The CLI's `hf://`/Cosmos-UUID checkpoint resolution
block is skipped entirely for `--backbone predict3` (not applicable to a
diffusers pipeline snapshot / bare HF repo id).

## 6. Test suite

```bash
uv run pytest predict3/tests/ -v
```

- `test_predict3.py`: constants sanity (real HF repo id, not an invented
  UUID) + `InferencerV3`'s CPU-only helpers (`_to_pil`, `_resolve_checkpoint`).
- `test_losses.py`: `angular_offset_weights`/`angular_weighted_velocity_loss`/
  `attenuation_mass_loss`/`total_physical_loss` -- zero-loss edge cases,
  gradient flow, shape-mismatch errors.
- `test_sft_dataset_build.py`: synthetic-fixture round-trip (93 frames in/out,
  JSONL schema, decord-decoded PSNR), the NSCLC leakage guard, and the
  wrong-frame-count error path.
- `test_tower.py`: `is_generator_param` classifier against real cosmos-framework names,
  `freeze_reasoner_tower` against a toy `nn.Module`, a drift guard between
  `xray360_edge.toml`'s `keys_to_select` and `tower.KEYS_TO_SELECT`, and a static
  regression guard that image conditioning (native I2V) stays wired in both
  `InferencerV3.predict()` and `scripts/launch_cosmos3_worker.sh`.

All 36 tests are CPU-only and require no Cosmos 3 weights or GPU (run with
`PYTHONPATH=. uv run pytest predict3/tests/ -v` from the repo root -- plain
`uv run pytest predict3/tests/` fails to resolve the `predict3` package without it).
