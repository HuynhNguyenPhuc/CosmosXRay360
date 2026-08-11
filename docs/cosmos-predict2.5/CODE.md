# Cosmos-Predict2.5 Codebase & Architecture Mapping (`CODE.md`)

**Main Post-Training & Inference Package:** `predict2_5/`  
**Foundation Submodule Backbone:** `cosmos-predict2.5/` (`cosmos_predict2/`)  
**Main Engine Classes:** `predict2_5/inferencer.py` (`Inferencer`), `predict2_5/module.py` (`CosmosXRay360`), `app.py`  
**Baseline Evaluation Harness:** `baselines/evaluate.py` (6 NVS Baselines)  
**Default Checkpoint URI:** `hf://phuchuynh0904/CosmosXRay360/net_ema.pth`  

---

## 1. System Integration & Architecture Overview

**CosmosXRay360** post-trains NVIDIA's **Cosmos Predict 2.5** 2B video world foundation model for single-view 360° rotational novel view synthesis (NVS). The system bridges a PyTorch Lightning training/inference wrapper package (`predict2_5/`) with NVIDIA's upstream foundation backbone submodule (`cosmos-predict2.5/`).

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CosmosXRay360 System                          │
│                                                                         │
│  ┌───────────────────────────┐         ┌──────────────────────────────┐ │
│  │   PreRenderedDataModule   │         │    CR1TextEncoder Wrapper    │ │
│  │  (93 PNG views/patient)   │         │   (Cosmos-Reason 1.0)        │ │
│  └─────────────┬─────────────┘         └──────────────┬───────────────┘ │
│                │ [B, 3, 93, 256, 256]                 │ [B, 512, 100352]│
│                ▼                                      ▼                 │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                       CosmosXRay360 (Module)                       │ │
│  │                                                                    │ │
│  │  Wan2pt1 VAE Tokenizer ──► Latent z_0 [B, 16, 24, 32, 32]          │ │
│  │  Rectified Flow        ──► Logit-Normal Time Sampling              │ │
│  │  MinimalV1LVGDiT (2B)  ──► 3D RoPE Trajectory + Cross-Attn        │ │
│  │  Frame-Token Replace   ──► z_t[0] & v_t[0] Ground-Truth Splicing   │ │
│  │  Power EMA (EDM2)      ──► FastEmaModelUpdater (β(k) decay)        │ │
│  └─────────────────────────────────┬──────────────────────────────────┘ │
│                                    │                                    │
│                                    ▼                                    │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                       Inferencer / Wrapper                         │ │
│  │  UniPC Denoising (35 Steps) ──► Wan2.1 VAE Decode ──► 93 PNG Views │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Exhaustive Repository Directory Map

### A. Core Package (`predict2_5/` & `scripts/`)

```
predict2_5/
├── constants.py       # Core constants: NUM_FRAMES=93, VOL_SIZE=256, UUIDs, prompts, dims
├── datamodule.py      # PreRendered360Dataset, PreRenderedLatentDataset, & PreRenderedDataModule
├── inferencer.py     # Single inference entrypoint (DiT init, UniPC sampling, Frame Replacement)
├── module.py         # CosmosXRay360 PyTorch Lightning Module (Rectified Flow, EMA, CFG, SAC)
├── trainer.py        # TrainingConfig, Trainer, DDP/FSDP strategy dispatcher
├── text_encoder.py   # CR1TextEncoder (Cosmos-Reason 1.0 embeddings & .pkl loader)
├── callbacks.py      # TensorBoardCallback, EMAMonitor, GradClipCallback, GradientMonitor
├── dvr/              # Legacy PyTorch3D DVR renderer (superseded by renderers/diffdrr/)
└── utils/            # RoPE buffer fix, checkpoint loader, distributed & image helpers
    ├── __init__.py   # Utility exports
    ├── distributed.py# DDP/FSDP rank, world size, broadcast, EMA DDP sync
    ├── logging.py    # Standardized logging setup
    ├── random.py     # Reproducible arch_invariant_rand
    └── torch_utils.py# fix_rope_buffers, safe_torch_load, move_tokenizer_to_device
scripts/
└── pre_encode_latents.py # Offline batch VAE latent pre-encoding utility script
```

### B. NVIDIA Submodule Backbone (`cosmos-predict2.5/cosmos_predict2/_src/`)

```
cosmos-predict2.5/cosmos_predict2/_src/
├── predict2/
│   ├── networks/
│   │   ├── minimal_v1_lvg_dit.py # MinimalV1LVGDiT (2B/7B/14B DiT backbone)
│   │   └── minimal_v4_dit.py     # SACConfig (Selective Activation Checkpointing)
│   ├── tokenizers/
│   │   ├── wan2pt1.py            # Wan2pt1VAEInterface & WanVAE (4x temporal compression)
│   │   └── base_vae.py           # Base VAE abstract interface
│   ├── schedulers/
│   │   └── rectified_flow.py     # RectifiedFlow, TrainTimeSampler, TrainTimeWeight
│   ├── models/
│   │   ├── fm_solvers_unipc.py   # FlowUniPCMultistepScheduler (35-step solver)
│   │   ├── video2world_model_rectified_flow.py # Reference Video2World model
│   │   └── text2world_model_rectified_flow.py  # Reference Text2World model
│   ├── conditioner.py            # DataType, Text2WorldCondition, GeneralConditioner
│   ├── text_encoders/
│   │   └── text_encoder.py       # TextEncoder & TextEncoderConfig
│   ├── utils/
│   │   └── optim_instantiate.py  # get_base_optimizer (FusedAdam, AdamW)
│   └── configs/
│       └── video2world/
│           ├── defaults/
│           │   ├── net.py         # Default DiT configurations
│           │   └── conditioner.py # Video2WorldCondition (set_video_condition)
│           └── experiment/
│               └── reason_embeddings/
│                   └── model_2B_reason_1p1_rectified_flow.py # Official 2B RF config
└── imaginaire/
    ├── utils/
    │   ├── ema.py                # FastEmaModelUpdater (EDM2 Power EMA)
    │   ├── checkpointer.py       # non_strict_load_model
    │   └── checkpoint_db.py      # download_checkpoint (UUID resolution)
    └── lazy_config/              # LazyDict, LazyCall hydra utilities
```

---

## 3. Mathematical Equation to Code Mapping

### A. Temporal Latent Encoding ($T=93 \to T_z=24$)
The input video tensor $V \in \mathbb{R}^{B \times 3 \times 93 \times 256 \times 256}$ is normalized to $[-1, 1]$ and encoded by Wan2.1 VAE:
$$z_0 = \mathcal{E}_{\text{VAE}}(V) \in \mathbb{R}^{B \times 16 \times 24 \times 32 \times 32}$$
- **Code:** `predict2_5/module.py:CosmosXRay360.encode(video)`  
- **Backbone Class:** `cosmos_predict2._src.predict2.tokenizers.wan2pt1.Wan2pt1VAEInterface.encode()`  
- **Temporal Formula:** $T_z = 1 + (T - 1) // 4 = 1 + (93 - 1) // 4 = 24$.

### B. Rectified Flow Interpolation & Velocity Loss
Given clean data $x_1$ and Gaussian noise $\epsilon \sim \mathcal{N}(0, I)$:
$$x_t = \sigma(t) \cdot \epsilon + (1 - \sigma(t)) \cdot x_1, \qquad v_t = \epsilon - x_1$$
$$\mathcal{L}_{\text{RF}} = \mathbb{E}_{t, \epsilon}\left[ w(t) \cdot \left\| v_\theta(x_t, t, c) - v_t \right\|_2^2 \right]$$
- **Code (Interpolation):** `predict2_5/module.py:CosmosXRay360.training_step()` via `rectified_flow.get_interpolation()`  
- **Code (Time Sampling):** `rectified_flow.sample_train_time()` with `train_time_distribution="logitnormal"`  
- **Backbone Class:** `cosmos_predict2._src.predict2.schedulers.rectified_flow.RectifiedFlow`

### C. Frame-Token Replacement Conditioning
To prevent geometric drift across the $360^\circ$ trajectory, conditioning frame $z_{\text{cond}}$ (PA radiograph latent at frame 0) is spliced into both input and predicted velocity at every step $t$:
$$x_t \leftarrow M \odot z_{\text{cond}} + (1 - M) \odot x_t$$
$$\hat{v}_t \leftarrow M \odot (\epsilon - z_{\text{cond}}) + (1 - M) \odot v_\theta(x_t, t, c)$$
where $M \in \{0, 1\}^{B \times 16 \times 24 \times 32 \times 32}$ is the binary mask ($M=1$ for frame 0, $M=0$ elsewhere).
- **Code (Module):** `predict2_5/module.py:CosmosXRay360.denoise()`  
- **Code (Inferencer):** `predict2_5/inferencer.py:Inferencer.denoise()`  
- **Reference Official Code:** `cosmos_predict2._src.predict2.models.video2world_model_rectified_flow.Video2WorldModelRectifiedFlow.denoise()`

### D. Classifier-Free Guidance (CFG)
During inference, CFG scale $w = 1.5$ combines text-conditioned and unconditioned velocity predictions:
$$v_{\text{pred}} = v_\theta(x_t, t, c_{\text{uncond}}) + w \cdot \left( v_\theta(x_t, t, c_{\text{cond}}) - v_\theta(x_t, t, c_{\text{uncond}}) \right)$$
- **Code:** `predict2_5/inferencer.py:Inferencer.predict()` & `predict2_5/module.py:CosmosXRay360.generate()`

### E. EDM2 Power EMA Accumulation
Exponential Moving Average weights update according to EDM2 Power EMA formulation:
$$\beta(k) = \left( 1 - \frac{1}{k + 1} \right)^{\gamma + 1}, \qquad \theta_{\text{EMA}} \leftarrow \beta(k) \theta_{\text{EMA}} + (1 - \beta(k)) \theta$$
where exponent $\gamma$ is computed via the largest real root of $\text{polynomial}([1, 7, 16 - s^{-2}, 12 - s^{-2}])$ for rate $s = 0.10$.
- **Code:** `predict2_5/module.py:CosmosXRay360._setup_ema()` & `on_before_zero_grad()`  
- **Backbone Class:** `cosmos_predict2._src.imaginaire.utils.ema.FastEmaModelUpdater`

---

## 4. Comprehensive Post-Training Hyperparameter Audit

This table verifies `predict2_5/module.py` against Paper §4.2 / project design and NVIDIA's official RF post-training configuration (`model_2B_reason_1p1_rectified_flow.py` & `Video2WorldModelRectifiedFlowConfig`):

| Parameter | Official NVIDIA RF Config | Paper §4.2 / Project Design | Diverges from NVIDIA RF Config? | Verified? |
|---|---|---|:---:|:---:|
| **DiT Backbone Class** | `MinimalV1LVGDiT` | `MinimalV1LVGDiT` (`_create_dit`) | No | ✅ |
| **Model Size Channels** | `model_channels=2048`, `num_blocks=28`, `num_heads=16` | `2B`: `{2048, 28, 16}` (`_get_model_config`) | No | ✅ |
| **3D RoPE Extrapolation** | `rope_h=3.0`, `rope_w=3.0`, `rope_t=1.0` | `rope_h=3.0`, `rope_w=3.0`, `rope_t=1.0` | No | ✅ |
| **Cross-Attention Dim** | `crossattn_proj_in_channels=100352` | `CROSSATTN_PROJ_IN_CHANNELS = 100352` | No | ✅ |
| **Cross-Attention Proj** | `use_crossattn_projection=True`, `channels=1024` | `use_crossattn_projection=True`, `channels=1024` | No | ✅ |
| **RF Shift Factor** | `shift = 5` | `rf_shift = 5.0` | No | ✅ |
| **Time Sampling** | `train_time_distribution = "logitnormal"` | `"logitnormal"` | No | ✅ |
| **Time Weighting** | `"reweighting"`* | `"uniform"` | No* | ✅ |
| **Conditioning Strategy** | `FRAME_REPLACE` | Latent & velocity splicing in `denoise()` | No | ✅ |
| **Conditioning Frames** | `min=0`, `max=2` (Latent frame count) | `min_num_conditional_frames=1`, `max=2` | Yes (`min=1` vs `min=0`) | ✅ |
| **CFG Dropout Rate** | `text.dropout_rate = 0.2` | `cfg_dropout_rate = 0.2` (per-sample Bernoulli) | No | ✅ |
| **Optimizer** | `adamw` | `get_base_optimizer(optim_type="fusedadam")` | Yes (`fusedadam` vs `adamw`) | ✅ |
| **Learning Rate** | `3e-5` | `learning_rate = 2**(-14.5)` ($\approx 4.3158 \times 10^{-5}$) | Yes (`2**(-14.5)` vs `3e-5`) | ✅ |
| **Weight Decay** | `weight_decay = 0.001` | `weight_decay = 0.001` | No | ✅ |
| **LR Warmup Steps** | `warm_up_steps = 100` | `warmup_steps = 2000` (`LinearLR` + `Cosine`) | Yes (`2000` vs `100`) | ✅ |
| **Power EMA Rate** | `ema_rate = 0.10` | `ema_rate = 0.10` (`FastEmaModelUpdater`) | No | ✅ |
| **Precision** | `bfloat16` (`bf16-mixed`) | `bf16-mixed` / `torch.autocast(dtype=bfloat16)` | No | ✅ |

*\* Note: In `rectified_flow.py`, `"reweighting"` is behaviorally mapped directly to `"uniform"` upon instantiation, rendering them behaviorally equivalent.*

---

## 5. Detailed Data & Execution Flow

### A. Post-Training Execution Pipeline

```
[PreRenderedDataModule]
   │ Read 93 views from datasets/pre_rendered/train/<patient>/views/*.png
   ▼
[Batch Dict] {"video": [B, 3, 93, 256, 256], "prompt": ["A 360-degree..."]}
   │
   ├────────► [CR1TextEncoder] ──► text_embeddings [B, 512, 100352]
   │               │ Apply CFG Dropout (20% Bernoulli) ──► text_filtered
   │
   └────────► [Wan2pt1VAE.encode] ──► x_1 [B, 16, 24, 32, 32]
                   │
                   ├──► Sample noise ε ~ N(0, I) & time t ~ LogitNormal
                   ├──► Interpolate x_t = t*ε + (1-t)*x_1 & v_t = ε - x_1
                   │
                   ▼
            [DiT Forward & Frame Replacement]
               │  z_t[0] <-- z_cond[0]
               │  Run MinimalV1LVGDiT(x_t, timesteps, text_filtered)
               │  v_t[0] <-- ε - z_cond[0]
               ▼
            [Loss Computation]
               │  MSE(v_pred, v_t) * time_weight
               ▼
            [Optimizer Step & Power EMA Update]
               │  FusedAdam step + FastEmaModelUpdater.update_average()
```

### B. Inference Execution Pipeline

```
[Input PA Radiograph] (256x256 PNG or Tensor)
   │
   ├────────► [CR1TextEncoder] ──► text_embeddings [1, 512, 100352]
   │
   └────────► Replicate PA across T=93 frames ──► [1, 3, 93, 256, 256]
                   │
                   ▼
            [Wan2pt1VAE.encode] ──► z_cond [1, 16, 24, 32, 32]
                   │
                   ▼
            [FlowUniPCMultistepScheduler] (35 Steps)
               │ Loop t in timesteps:
               │   1. Predict v_cond = denoise(noise, z_t, t, cond)
               │   2. Predict v_uncond = denoise(noise, z_t, t, uncond)
               │   3. Guided velocity v = v_uncond + 1.5 * (v_cond - v_uncond)
               │   4. UniPC step z_{t-1} = scheduler.step(v, t, z_t)
               │   5. Frame replacement: z_{t-1}[0] <-- z_cond[0]
               ▼
            [Wan2pt1VAE.decode] ──► Output Video [1, 3, 93, 256, 256]
                   │
                   ▼
            [Post-Processing] Min-max clamp to [0, 1] ──► 93 uint8 PNG views
```

---

## 6. Implementation Gotchas & Custom Fixes

1. **RoPE Frequency Buffer Registration (`fix_rope_buffers`):**
   - *Issue:* Loading pretrained DiT state dicts across varying PyTorch versions or GPUs often dropped or mishandled Rotary Position Embedding (`rope`) buffer shapes.
   - *Fix:* `predict2_5/utils/torch_utils.py:fix_rope_buffers()` explicitly re-registers `rope_h`, `rope_w`, and `rope_t` frequency buffers before weight restoration.

2. **Device Flexibility Guard:**
   - *Issue:* Hardcoded `to_empty(device="cuda")` crashed on CPU-only or non-CUDA environments.
   - *Fix:* `module.py` uses `init_device = "cuda" if torch.cuda.is_available() else "cpu"` and updates `self.rectified_flow.device = self.device` during `on_train_start()`.

3. **Artifact URI Resolution Hierarchy (`_resolve_artifact`):**
   - Resolves checkpoint files dynamically via three fallback tiers:
     1. Local File Path (e.g., `baselines/checkpoints/net_ema.pth`).
     2. Hugging Face Hub URI (e.g., `hf://phuchuynh0904/CosmosXRay360/net_ema.pth`).
     3. NVIDIA Cosmos Experimental UUID (e.g., `d20b7120-df3e-4911-919d-db6e08bad31c`).

4. **Single-View Entrypoints (`Inferencer` & `app.py`):**
   - Core inference is handled directly by `Inferencer` (`predict2_5/inferencer.py`) and the Gradio web UI (`app.py`), which accept single 2D radiograph inputs and generate 93-frame 360° videos with ground-truth frame-0 conditioning.

---

## 7. Entrypoints, Testing & Verification Commands

### A. Launch Web UI Application
```bash
python app.py --checkpoint-path "hf://phuchuynh0904/CosmosXRay360/net_ema.pth" --share
```

### B. Run Post-Training (Single GPU or Multi-GPU FSDP)
```bash
# Single GPU
uv run python predict2_5/trainer.py --batch_size 4 --sac_mode predict2_2b_720_aggressive

# Multi-GPU FSDP (e.g. 2 GPUs)
torchrun --nproc_per_node=2 predict2_5/trainer.py --strategy fsdp --batch_size 4
```

### C. Run 6-Baseline NVS Evaluation Harness
```bash
python baselines/evaluate.py --checkpoint-dir baselines/checkpoints
```
