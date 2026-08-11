# Cosmos-Predict2.5 Development & Experimentation Log

**Maintained by:** CosmosXRay360 Team
**Location:** `docs/cosmos-predict2.5/LOG.md`

---

## Chronological Development & Milestone Log

### [2026-07-22] Foundation & Submodule Setup
- **Action:** Initialized repository and integrated NVIDIA `cosmos-predict2.5` foundation model git submodule.
- **Implementation:**
  - Configured `pyproject.toml` with `cu128`/`cu130` CUDA extras via `uv`.
  - Defined system constants in `predict2_5/constants.py` (`NUM_FRAMES=93`, `VOL_SIZE=256`, checkpoint UUIDs).
  - Added volumetric emission/absorption DRR rendering utilities in `predict2_5/dvr/`.

### [2026-07-25] Distributed Utilities, Callbacks & Text Embeddings
- **Action:** Built core PyTorch Lightning module callbacks, GCS cloud I/O, and text embedding pre-extraction.
- **Implementation:**
  - Added training callbacks for TensorBoard visualization (`TensorBoardCallback`), EMA monitoring (`EMAMonitorCallback`), and gradient norm clipping (`GradClipCallback`).
  - Created `scripts/get_cosmos_reason_embeddings.py` to extract and cache Cosmos-Reason1 text embeddings (`CR1TextEncoder`).

### [2026-07-28] Frame-Token Replacement Conditioning Implementation
- **Action:** Developed frame replacement conditioning logic inside `Inferencer.denoise` and `Inferencer.predict`.
- **Rationale:** Standard video diffusion from a single conditioning image drifts significantly across later frames in a $360^\circ$ rotation trajectory.
- **Fix:** At every UniPC step $t$, enforced explicit latent replacement $z_t[0] \leftarrow z_{\text{cond}}[0]$ and velocity replacement $v_t[0] \leftarrow (\text{noise} - z_{\text{cond}}[0])$.
- **Verification:** Observed perfect spatial alignment at frame 0 and continuous 360° parallax rotation across all 93 frames.

### [2026-08-01] RoPE Buffer Fix, Artifact Resolver & Gradio Application
- **Action:** Fixed Rotary Position Embedding (RoPE) buffer shape mismatches and built Gradio Web UI (`app.py`).
- **Implementation:**
  - Added `fix_rope_buffers(dit)` in `predict2_5/utils/torch_utils.py` to re-register RoPE buffers prior to weight loading.
  - Built `predict2_5/hf.py` and `_resolve_artifact` in `Inferencer` supporting `hf://` URIs, local file paths, and Cosmos UUIDs.
  - Implemented Gradio application in `app.py` with display modes and contrast stretching.

### [2026-08-06] DiffDRR Physics Migration & Baseline Benchmark Suite
- **Action:** Migrated ground-truth DRR rendering to DiffDRR Siddon-Jacob raymarching (`renderers/diffdrr/`) and integrated 6 NVS baselines (`baselines/`).
- **Implementation:**
  - Standardized unified wrapper API (`infer_multi_views(input_xray, azimuths)`) across SV-DRR, XRaySyn, MedNeRF, PixelNeRF, NAF, and Dx2CT.
  - Built unified evaluation harness in `baselines/evaluate.py` calculating PSNR, SSIM, LPIPS, and inference latency on the cross-dataset OOD split (TCIA+MELA train $\to$ NSCLC test).

### [2026-08-08] Multi-View Visualization & GCP VM Automation Overhaul
- **Action:** Overhauled TensorBoard multi-view logging across baseline training scripts and streamlined GCP cloud worker VM orchestration.
- **Implementation:**
  - Enforced 1-channel grayscale output (`[1, 1, 256, 256]`) for all baseline predictions and ground truth.
  - Expanded intermediate view logging from 4 cardinal views to 6 oblique views (0°, 58°, 120°, 178°, 240°, 298°).
  - Updated `scripts/launch_parallel_vms.sh` to manage cloud training workers on GCP.

### [2026-08-10] Post-Training Acceleration Pipeline & PyTorch FSDP Strategy
- **Action:** Implemented the complete 5-phase acceleration roadmap for Cosmos-Predict2.5 post-training and PyTorch FSDP multi-GPU sharding.
- **Implementation:**
  - **Selective Activation Checkpointing (SAC):** Integrated `predict2_2b_720_aggressive` SAC mode in `predict2_5/module.py` and `trainer.py`, dropping activation VRAM per sample from ~28 GB to ~6 GB.
  - **Offline VAE Latent Pre-caching (Track B1):** Built `scripts/pre_encode_latents.py` for batch pre-encoding 93-view CT rotation videos into Wan2.1 VAE $z_0 \in \mathbb{R}^{16 \times 24 \times 32 \times 32}$ latent tensors. Implemented `PreRenderedLatentDataset` in `predict2_5/datamodule.py` and added `--use_latent_cache` / `--latent_cache_dir` flags in `PreRenderedDataModule` and `TrainingConfig` (`trainer.py`). `module.py` consumes `pre_cached_latent` in `training_step()` and `validation_step()`, bypassing 3D VAE encoder compute (~18 GB VRAM savings, ~35% speedup per step). Enforces OOD discipline (TCIA+MELA training set only; never encodes NSCLC).
  - **PyTorch FSDP Multi-GPU Strategy:** Configured `FSDPStrategy` with `sharding_strategy="FULL_SHARD"` and `Block` auto-wrap policy. Implemented non-module EMA storage pattern (`_net_ema_module` behind `@property def net_ema`) to prevent parameter key mismatch errors in optimizer state dicts under FSDP sharding.
  - **Device Assignment & Dataloader Fixes:** Bound local rank GPU devices (`cuda:{get_local_rank()}`) and fixed validation dataloader to return `[]` when validation split is empty.
  - **JIT Compilation Evaluation (`torch.compile`):** Benchmarked `torch.compile` on NVIDIA A100-80GB GPU via `scripts/benchmark_acceleration_phases.py`. Found that Transformer Engine C++ kernels (`rmsnorm_fwd`, `DotProductAttention`) create Dynamo graph breaks yielding ~0% speedup. Omitted JIT compilation from the pipeline to maintain code simplicity and instant startup.
- **Verification:** Verified end-to-end multi-GPU FSDP training on 2x A100 GPUs, successfully logging training/validation steps and generating full FSDP checkpoints (`best.ckpt`, `epoch=0000.ckpt`, `last.ckpt`).

### [2026-08-11] DataLoader Throughput Benchmarking & Optimal Defaults (Track B3)
- **Action:** Benchmarked DataLoader throughput configurations on an NVIDIA A100-SXM4-80GB GPU instance (`a2-ultragpu-1g`) on GCP Compute Engine via `scripts/benchmark_dataloader.py`.
- **Benchmarking Results:**
  - **Pre-cached Latent Mode (`PreRenderedLatentDataset`):** Peak throughput reached **~440.94 samples/s** ($2.27\text{ ms/batch}$). Top configuration: `num_workers=4`, `prefetch_factor=2`, `pin_memory=False`, `persistent_workers=True`. Since `.pt` latent tensors are compact ($\sim 786\text{ KB/sample}$), $4$ worker processes saturate disk/RAM bandwidth without IPC process management overhead.
  - **Raw Video Mode (`PreRendered360Dataset`):** Peak throughput reached **~48.68 samples/s** ($20.54\text{ ms/batch}$). Top configuration: `num_workers=12`, `prefetch_factor=2`, `pin_memory=False`, `persistent_workers=True`.
- **Implementation & Default Configuration:**
  - Plumbed `pin_memory`, `persistent_workers`, and `prefetch_factor` through `TrainingConfig`, `PreRenderedDataModule`, and CLI arguments in `predict2_5/trainer.py`.
  - Established optimal hardware defaults in `predict2_5/trainer.py`: `num_workers=4`, `prefetch_factor=2`, `pin_memory=True`, `persistent_workers=True`.
  - Added single-file `.pt`/`.npy` binary container reading and 1-channel IPC transfer optimizations in `predict2_5/datamodule.py` and `predict2_5/module.py`.

