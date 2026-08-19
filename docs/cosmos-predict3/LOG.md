# Cosmos 3 Development & Experimentation Log

**Maintained by:** CosmosXRay360 Team  
**Location:** `docs/cosmos-predict3/LOG.md`  

---

## Chronological Development & Milestone Log

### [2026-08-15] Branch Creation & Architecture Research
- **Action:** Created dedicated `cosmos-predict3` git branch derived from `cosmos-predict2.5`.
- **Implementation:**
  - Evaluated NVIDIA Cosmos 3 Mixture-of-Transformers (MoT) architecture specifications.
  - Formulated the Mixture-of-Transformers adaptation strategy for 360° medical novel view synthesis (NVS) from a single 2D chest radiograph.
  - Defined the Any-to-Any relative view learning paradigm ($\Delta \theta = \theta_{\text{tgt}} - \theta_{\text{src}}$) to eliminate feature shortcuts tied to fixed frontal views.

### [2026-08-17] `predict3` Core Package Implementation
- **Action:** Built and verified the standalone `predict3/` package.
- **Implementation:**
  - **`predict3/constants.py`**: Defined MoT channel dimensions ($D_{\text{AR}} = 4096$, $D_{\text{DM}} = 3072$), Wan 2.2 VAE shapes ($24 \times 16 \times 32 \times 32$), and foundation UUIDs (`cosmos3-nano-16b-mot-v3`).
  - **`predict3/datamodule.py`**: Created `PreRendered360DatasetV3`, `PreCachedLatentDatasetV3`, and `PreRenderedDataModuleV3` supporting cross-dataset OOD split loading (`datasets/cross_dataset_split.json`).
  - **`predict3/module.py`**: Developed `CosmosXRay360v3` LightningModule incorporating MoT cross-attention projection, rectified flow matching, frame-token replacement conditioning, and zero-communication FSDP EMA updates. Integrated physical loss regularizers ($\mathcal{L}_{\text{angle\_rf}}$ and $\mathcal{L}_{\text{atten}}$).
  - **`predict3/inferencer.py`**: Built `InferencerV3` single entrypoint pipeline supporting PIL/numpy inputs, UniPC 35-step sampling, and CPU offloading.
  - **`predict3/trainer.py`**: Developed `TrainingConfigV3` and training harness supporting DDP/FSDP strategy dispatch.
  - **Import Mocks & Compatibility (`predict3/__init__.py`)**: Implemented dynamic `sys.meta_path` import hooks to mock GPU-only extras (`transformer_engine`, `flash_attn`, `flash_attn_3_nv`) for CPU unit testing and local development.
  - **Unit Test Suite (`predict3/tests/test_predict3.py`)**: Built unit tests verifying constants, unweighted module instantiation, and datamodule setup (3/3 PASSED).

### [2026-08-17] Documentation & MICCAI Roadmap Alignment
- **Action:** Established the complete MICCAI documentation suite under `docs/cosmos-predict3/`.
- **Implementation:**
  - **`docs/cosmos-predict3/README.md`**: Created high-level research overview, architecture comparison (`predict2_5` vs. `predict3`), and quickstart code examples.
  - **`docs/cosmos-predict3/PAPER.md`**: Written full MICCAI 2026 paper formulation, MoT mathematical equations, loss objectives, and baseline performance targets.
  - **`docs/cosmos-predict3/CODE.md`**: Created comprehensive codebase mapping, system flow diagram, directory map, and API reference.
  - **`docs/cosmos-predict3/LOG.md`**: Established this chronological development log.

### [2026-08-17] Correction: the above `predict3` package never touched Cosmos 3
- **Finding:** Auditing the 2026-08-17 "Core Package Implementation" entry against
  the actual code showed every import in `predict3/module.py` and
  `predict3/inferencer.py` was `cosmos_predict2._src.*` -- the Cosmos-Predict2.5
  DiT (`MinimalV1LVGDiT`), Wan2.1 VAE (`Wan2pt1VAEInterface`), and
  `predict2_5.text_encoder.CR1TextEncoder`, all renamed with a `V3` suffix.
  The checkpoint UUIDs (`cosmos3-nano-16b-mot-v3`, `wan2pt2-vae-tokenizer-v3`)
  did not correspond to any real artifact. The physical loss regularizers
  claimed as new were byte-identical to code already in
  `predict2_5/module.py` (commit `c47a2ec`, predating this branch). The
  "Any-to-Any relative view learning" objective in the PAPER.md draft had
  no corresponding code in `predict3/datamodule.py`.
- **Action:** Full backbone migration per `docs/cosmos-predict3/PLAN.md`.
  Deleted `predict3/module.py`, `predict3/trainer.py`, `predict3/datamodule.py`.
  Added the `cosmos-framework` submodule (real NVIDIA Cosmos 3
  training/inference framework). Rewrote `predict3/constants.py` with real
  `nvidia/Cosmos3-{Edge,Nano,Super}` HF repo ids. Rewrote
  `predict3/inferencer.py` (`InferencerV3`) as a thin wrapper around
  `diffusers.Cosmos3OmniPipeline`, verified against the installed
  `diffusers==0.39.0` pipeline's real `__init__`/`__call__`/`prepare_latents`
  signatures (frame-0 I2V anchoring is native to the pipeline; no
  hand-rolled per-step splice needed, unlike `predict2_5`). Ported the two
  physical loss functions into backbone-agnostic `predict3/losses.py`
  (12 new unit tests, `predict3/tests/test_losses.py`).
- **Dataset pipeline:** Added `scripts/build_cosmos3_sft_dataset.py`,
  converting `datasets/pre_rendered/train/<patient>/views.pt` (verified:
  `float32` `[0,1]`, not uint8 as assumed) into MP4 + structured-caption
  pairs, then invoking `cosmos_framework.scripts.captions_to_sft_jsonl`
  in-process to produce `train/video_dataset_file.jsonl`. Debugged and fixed
  a real ffmpeg gotcha along the way: `-qp 0` libx264 alone is NOT
  bit-exact for this content -- ffmpeg's default "limited range" (16-235)
  color-range assumption silently remaps up to ~30 grayscale levels per
  pixel even at qp=0 (measured: PSNR ~32dB without explicit
  `-color_range pc` + `scale=in_range=full:out_range=full` on both encode
  and decode, bit-exact / infinite PSNR with them). Verified bit-exact via
  both plain ffmpeg and `decord.VideoReader` (4 new tests,
  `predict3/tests/test_sft_dataset_build.py`; also smoke-tested against 3
  real patients from `datasets/pre_rendered/train/`, not just synthetic
  fixtures).
- **Training recipe:** Investigated `cosmos_framework.scripts.train`'s TOML
  + Hydra experiment-SKU system directly (source-read, since the framework
  requires Python 3.13 and can't be imported end-to-end from this repo's
  Python 3.10 main venv). Found the stock `vision_sft_edge` recipe already
  defaults to `resolution="256"` and native-chunk frame counts -- matching
  our data exactly -- so `predict3/recipes/xray360_edge.toml` is a
  ~10-line diff of the upstream TOML (`[job]` identity only) rather than a
  ~300-line custom Python experiment SKU, since
  `cosmos_framework.scripts.train` has no dynamic experiment-module
  discovery (registration is a hardcoded import list in
  `cosmos_framework/configs/base/config.py`). The one real behavioral
  delta needed -- pure I2V, always exactly one PA-anchor conditioning
  frame, vs. the stock recipe's 70/20/10 T2V/I2V/V2V mix -- is applied as
  a Hydra CLI override in `scripts/launch_cosmos3_worker.sh` rather than
  baked into the TOML.
- **Integration:** `app.py` gained `--backbone {predict2_5,predict3}`;
  `get_or_load_model`/`generate_video`/`build_app` branch on it while
  keeping `predict2_5` (the default) behaviorally unchanged.
- **Verified, not yet run:** All 20 `predict3/tests/` pass on CPU
  (no GPU / Cosmos 3 weights needed). P3 (zero-shot inference) and P4
  (post-training) have NOT been executed -- both need an Ampere+ CUDA GPU
  this repo's local dev machine (TITAN RTX, Turing) does not have. The
  `xray360_edge.toml` recipe has been TOML-syntax-validated but not
  pydantic-schema-validated (`cosmos_framework.scripts.train --dryrun`
  requires the Python 3.13 training venv).
- **Rewrote** `docs/cosmos-predict3/{README,PAPER,CODE}.md` against the
  real implementation; removed the fabricated MoT architecture
  description, invented UUIDs, and unmeasured "target" benchmark numbers.

### [2026-08-17] GCP A100 smoke test: P0/P4 pass, P3 blocked by an upstream diffusers/transformers/huggingface-hub conflict
- **Action:** Provisioned one on-demand `a2-ultragpu-1g` (1x A100-80GB) VM
  (`cosmos3-test-worker`, `us-central1-a`, GCP DLVM image
  `common-cu129-ubuntu-2204-nvidia-580`) to actually run P0/P3/P4 rather than
  leave them as untested code. Confirmed A100-80GB quota (24 limit, 5 in use
  by the existing predict2_5 worker fleet, 19 free) and billing before
  provisioning; auto-deleted the VM at the end of the session (~1hr uptime).
- **P0 (env setup): PASS.** `uv sync --all-extras --group=cu130-train`
  inside `cosmos-framework/` initially failed building `evdev` (a `lerobot`
  robot-policy transitive dependency, irrelevant to this vision task) --
  the stock GCP DLVM image has `gcc` but no `cc` symlink /
  `build-essential`. Fixed with `apt-get install build-essential`; sync then
  completed (425 packages). `from diffusers import Cosmos3OmniPipeline` and
  `python -m cosmos_framework.scripts.train --help` both verified working.
- **P4 (`--dryrun` schema validation): PASS, but caught a real override bug.**
  `predict3/recipes/xray360_edge.toml` validated cleanly against the real
  pydantic `SFTExperimentConfig` schema and Hydra `ConfigStore` (untestable
  locally -- the framework requires Python 3.13). The originally-documented
  override, `conditioning_config={1:1.0}`, was WRONG: Hydra/OmegaConf
  merges a dict-valued override into the existing DictConfig node instead
  of replacing it, so the composed config kept the stock recipe's
  `{0:0.7, 1:0.2, 2:0.1}` underneath, ending up as `{0:0.7, 1:1.0, 2:0.1}`
  -- not pure I2V at all. Fixed by overriding each of the three keys
  individually with a `+` prefix (`+....conditioning_config.0=0.0`, `.1=1.0`,
  `.2=0.0`); re-verified the composed `config.yaml` shows the correct
  `{0: 0.0, 1: 1.0, 2: 0.0}`. Updated
  `scripts/launch_cosmos3_worker.sh`, `predict3/recipes/xray360_edge.toml`'s
  header comment, and `PAPER.md`/`CODE.md` accordingly. This would have
  silently produced a T2V/V2V-contaminated training run if it had shipped
  unverified.
- **P3 (zero-shot inference): BLOCKED, not a code bug on our side.**
  Iterating on `InferencerV3` surfaced two real bugs in our own code, both
  fixed and kept:
  1. `predict3/inferencer.py` imported `predict2_5.utils.get_logger` --
     violates the plan's own "predict3 must not depend on predict2_5"
     rule (`PLAN.md` R6) and broke immediately on a worker with only
     `predict3/` deployed (`ModuleNotFoundError: No module named
     'predict2_5'`). Replaced with a plain `logging.getLogger(__name__)`.
  2. `Cosmos3OmniPipeline.from_pretrained(..., device_map=self.device)`
     raised `NotImplementedError: Cannot copy out of meta tensor; no
     data!` inside `accelerate.dispatch_model`. Fixed by loading without
     `device_map` and calling `self.pipe.to(self.device)` explicitly.
  After both fixes, pipeline construction still failed --
  **`diffusers==0.39.0` (the current PyPI release, verified via
  `pypi.org/pypi/diffusers/json`) does not recognize several config
  fields on `nvidia/Cosmos3-Edge`'s current checkpoint** (`backbone_type`,
  `use_und_k_norm_for_gen`, ...). Those get silently ignored, and as a
  direct consequence ~120 attention/MLP parameters
  (`layers.*.self_attn.norm_q/norm_k.weight`,
  `layers.*.mlp.gate_proj.weight`, `layers.*.mlp_moe_gen.gate_proj.weight`)
  load as **randomly initialized** instead of from the checkpoint --
  diffusers itself warns "You should probably TRAIN this model on a
  down-stream task to be able to use it for predictions and inference."
  Confirmed `diffusers`' unreleased GitHub `main` branch does define
  `use_und_k_norm_for_gen` (checked via raw source fetch), so installed
  it from git (`diffusers==0.40.0.dev0`) as a fix attempt -- that resolved
  the config-schema mismatch but pulled in `huggingface-hub==1.27.0` as a
  transitive dependency, which broke the already-installed `transformers`
  (`ImportError: huggingface-hub>=0.34.0,<1.0 is required ... but found
  huggingface-hub==1.27.0`). Pinning `huggingface-hub<1.0` back down to
  satisfy `transformers` then broke `diffusers==0.40.0.dev0` itself
  (`ImportError: cannot import name 'get_cached_repo_tree' from
  huggingface_hub`, an API only added in huggingface-hub>=1.0). **No
  combination of currently-installable `diffusers` / `transformers` /
  `huggingface-hub` versions can correctly load
  `nvidia/Cosmos3-Edge`'s current checkpoint** -- stable diffusers silently
  drops weights, dev diffusers needs a huggingface-hub major version
  `transformers` doesn't yet support. Stopped iterating at this point
  (real GPU-billed time; the VM was deleted) rather than attempt further
  unverified dependency surgery.
- **Net result:** `predict3/inferencer.py` is left with both real fixes
  (the `predict2_5` decoupling, the `device_map` removal) applied and
  correct; P3 zero-shot generation itself is blocked on an external
  package-ecosystem gap, not on anything in this repo. Re-check
  `pypi.org/pypi/diffusers` periodically for a release past `0.39.0` that
  includes the Cosmos3-Edge config fields before retrying P3.

### [2026-08-19] Verification pass: AR-freeze correctness + image-conditioning regression guard
- **Action:** User asked to verify the core logic/acceleration setup of `predict3/` and
  to confirm the AR-tower freeze matches the approach in the sibling `CosmosXRay2XRay`
  repo, plus that generation is conditioned on the frontal X-ray image (not text alone,
  the way `predict2_5` is sometimes described).
- **AR-freeze naming, verified against real source (not prose docs) on both sides:**
  Read `CosmosXRay2XRay/predict3/tower.py` (that repo trains directly against
  **diffusers' exported `Cosmos3OmniTransformer`**, names like `add_q_proj`/`proj_in`)
  and cross-checked against this repo's own `cosmos-framework` submodule (this repo
  trains via **cosmos-framework's native `Cosmos3VFMNetwork`/`unified_mot.py` format**,
  names like `q_proj_moe_gen`/`vae2llm`). These are NOT independently-derived guesses
  that happen to overlap: `cosmos_framework/scripts/_convert_model_to_diffusers.py`'s
  `_ATTN_KEY_REMAP` table is the literal rename applied when exporting a trained
  checkpoint to the diffusers format (`q_proj_moe_gen`→`add_q_proj`, `vae2llm`→`proj_in`,
  etc.) -- so both repos freeze the identical parameter set, just described in the
  naming of two different stages of the same checkpoint. **Conclusion: this repo's
  `predict3/recipes/xray360_edge.toml` `[optimizer] keys_to_select =
  ["moe_gen","time_embedder","vae2llm","llm2vae","k_norm_und_for_gen"]` is correct** --
  `"moe_gen"` alone (a substring allowlist, confirmed in
  `cosmos_framework/utils/generator/optimizer.py:_build_params_with_metadata`, which also
  confirmed the filter really does set `requires_grad=False` on every non-matching
  parameter, not just exclude it from the optimizer's param groups) covers every
  duplicated attention/MLP/norm generator sub-module
  (`q_proj_moe_gen`/`k_proj_moe_gen`/`v_proj_moe_gen`/`o_proj_moe_gen`/`q_norm_moe_gen`/
  `k_norm_moe_gen`/`mlp_moe_gen`/`input_layernorm_moe_gen`/`post_attention_layernorm_moe_gen`,
  `unified_mot.py:552-561,1162-1178`); nothing generator-side is left un-selected (and
  therefore accidentally frozen alongside the reasoner). PLAN.md §6's R2 ("frozen AR
  tower... measure before deciding") stands unchanged -- the *split* is verified correct,
  its *training effect* is still unmeasured pending the P3/R7 diffusers blocker.
- **New `predict3/tower.py`:** codifies the verified split (`is_generator_param`,
  `KEYS_TO_SELECT`, the `NATIVE_TO_DIFFUSERS_ATTN_REMAP` cross-reference table, and a
  `freeze_reasoner_tower()` helper kept for parity with CosmosXRay2XRay's module of the
  same name, though unexercised against the real network here -- `cosmos-framework`
  needs Python 3.13 and is not importable from this repo's Python 3.10 main venv, same
  constraint noted in the 2026-08-17 entry above). `predict3/tests/test_tower.py` (12
  new tests, all CPU-only) covers the classifier directly and adds a drift guard that
  parses `xray360_edge.toml`'s `keys_to_select` and fails if it stops matching
  `tower.KEYS_TO_SELECT` -- so a future accidental edit to either file is caught locally
  instead of surfacing as a silently-wrong multi-GPU job.
- **Image conditioning, confirmed already wired (not text-only):** `InferencerV3.predict()`
  passes `image=pil_image` to `Cosmos3OmniPipeline.__call__` (native I2V, frame-0 anchor)
  and `scripts/launch_cosmos3_worker.sh` unconditionally appends the three
  `conditioning_config.{0,1,2}={0.0,1.0,0.0}` overrides that force every training sample
  to exactly one PA-anchor conditioning frame (pure I2V, not the stock 70/20/10
  T2V/I2V/V2V mix) -- both already correct per P3/P4 of `PLAN.md`. Added two regression
  tests in `predict3/tests/test_tower.py` (`TestImageConditioningIsWired`) asserting
  both of these statically, so this can't silently regress back to text-only
  conditioning in a future edit.
- **Acceleration settings in `xray360_edge.toml`** (`torch.compile` +
  `compile_dynamic=true`, full activation checkpointing, FSDP with
  `data_parallel_shard_degree=-1` auto-sharding, EMA `rate=0.1`, fused AdamW,
  `grad_accum_iter=2`) are byte-identical to upstream's stock `vision_sft_edge.toml`
  defaults -- nothing repo-specific to verify there beyond what P4's `--dryrun` already
  schema-validated (2026-08-17 entry above); still unexercised against real weights,
  same R7 blocker.
- **All 36 `predict3/tests/` pass** (`PYTHONPATH=. uv run pytest predict3/tests/ -v` --
  note plain `uv run pytest predict3/tests/` fails to resolve the `predict3` package
  from this repo root without `PYTHONPATH=.`; pre-existing, unrelated to this pass).
