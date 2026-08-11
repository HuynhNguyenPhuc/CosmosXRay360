# Repository Structure

This document details the exact repository structure and file-by-file contents of the **CosmosXRay360** codebase, mapping configurations, core predictive engines, baselines, and utility scripts.

---

## 📂 Visual Repository Tree

```markdown
CosmosXRay360/             # Repository Root
├── app.py                # Main Gradio Web Interface (Launch CLI)
├── predict2_5/           # Core Post-Training & Inference Package
│   ├── callbacks.py      # Training callbacks (EMAMonitor, GradClipCallback, etc.)
│   ├── constants.py      # Core constants and configurations (NUM_FRAMES=93, etc.)
│   ├── datamodule.py     # PreRendered360Dataset, PreRenderedLatentDataset, & PreRenderedDataModule
│   ├── hf.py             # Hugging Face checkpoint loading and management
│   ├── inferencer.py     # Core Inferencer class (single-view DiT denoising + VAE)
│   ├── module.py         # CosmosXRay360 PyTorch Lightning Module (Rectified Flow, EMA, CFG, SAC)
│   ├── text_encoder.py   # CR1TextEncoder (Cosmos-Reason 1.0 text conditioning)
│   ├── trainer.py        # TrainingConfig, Trainer, DDP/FSDP strategy dispatcher
│   ├── dvr/              # Differentiable volume rendering (raymarcher) utilities
│   └── utils/            # Data/image transformations, formatting, distributed helpers
├── cosmos-predict2.5/    # Submodule (NVIDIA's World Foundation Model Backbone)
│   ├── tests/            # Test suite for DiT/VAE architectures
│   └── setup.py          # Submodule configuration and dependencies
├── baselines/            # Baseline Evaluation & Training Harness
│   ├── evaluate.py       # Unified multi-model OOD evaluation harness (UnifiedBaselineEvaluator)
│   ├── run_all_training.sh # Unified training automation script
│   ├── models/           # Baseline standard wrapper APIs (models/utils.py holds shared
│   │                      #   math incl. perspective_ray_march, compute_psnr, compute_ssim)
│   ├── train/             # Per-baseline training scripts
│   │   ├── dx2ct.py       # Slice diffusion trained on real axial CT density slices
│   │   ├── svdrr.py       # Pose-conditioned DiT, random view-pair training
│   │   ├── mednerf.py     # Joint z+generator-weight per-scan fitting
│   │   ├── naf.py         # Per-scan coordinate-field fitting
│   │   ├── pixelnerf.py   # Feed-forward NeRF training
│   │   └── xraysyn.py     # Voxel GAN + refinement training
│   ├── tests/            # Baseline test suite (pytest); run_all_tests_isolate.py runs
│   │                      #   each test_*.py in its own subprocess to avoid CUDA/JIT
│   │                      #   state cross-contamination between baselines
│   └── cloned/           # Original Cloned Repositories
│       ├── XraySyn/      # Physics-supervised voxel baseline (AAAI '21)
│       ├── mednerf/      # Implicit Neural Radiance Fields baseline (EMBC '22)
│       ├── DX2CT/        # Feature-modulated 3D CT Diffusion baseline (ICASSP '25)
│       ├── naf_cbct/     # Implicit Continuous Attenuation Fields (MICCAI '22)
│       ├── pixel-nerf/   # Feed-Forward Prior-Guided NeRF (CVPR '21)
│       └── SV-DRR/       # Pose-conditioned 2D DRR Diffusion baseline (MICCAI '25)
├── datasets/             # Local Data & Split Configurations
│   └── cross_dataset_split.json # Train/Test OOD partitions (TCIA+MELA -> NSCLC)
├── docs/                 # Documentation Suite (nanoCosmos Style)
│   ├── INDEX.md          # Visual Navigation Router
│   ├── STRUCTURE.md      # Repository Map (This file)
│   ├── WALKTHROUGH.md    # Chronological Cosmos-Predict2.5 batch execution walkthrough
│   ├── DATASET.md        # Comprehensive EDA, Preprocessing & HU Window Specifications
│   ├── RENDERER.md       # Renderer evolution (PyTorch3D DVR -> DiffDRR) and migration strategy
│   ├── BENCHMARK.md      # Quantitative benchmarking specifications
│   ├── EXECUTION.md      # Master baseline retraining & evaluation protocol
│   ├── LITERATURE.md     # Comprehensive academic review of baseline mechanics
│   ├── GOTCHAS.md        # Physical & Mathematical Silent Failures
│   ├── REVIEWS.md        # Structured MICCAI reviewer concerns & deconstructed requirements
│   ├── PROPOSAL.md       # SOTA baselines & re-submission roadmap
│   ├── baselines/        # Per-baseline PAPER.md, CODE.md, and LOG.md documentation
│   ├── cosmos-predict2.5/ # Cosmos-Predict2.5 main method PAPER.md, CODE.md, and LOG.md
│   └── reviews/          # Reviewer responses & analyses
├── renderers/            # Shared Physical Rendering Engines
│   └── diffdrr/          # DiffDRR Siddon-Jacob ray-tracing standard
├── scripts/              # Command-Line Utilities & GCP Cloud Automation
│   ├── fast_download.py  # Fast concurrent Hugging Face dataset downloader
│   ├── pre_encode_latents.py # Offline batch VAE latent pre-encoding utility script
│   ├── upload_to_hf.py   # Upload checkpoints/models to Hugging Face Hub
│   ├── launch_parallel_vms.sh # Launch GCP L4 VM instances for baseline training
│   └── download_folder_from_gcs.py # Sync checkpoints from GCS buckets
├── AGENTS.md             # Contribution Guidelines
├── CLAUDE.md             # Agent Instruction Map
└── requirements.txt      # Python dependencies list
```

---

## 🛠️ Folder Breakdown and Specifications

### 1. Main Application Wrapper (`app.py`)
Launches the interactive Gradio web application for inference. It loads the pretrained model weights via Hugging Face (`hf://phuchuynh0904/CosmosXRay360/net_ema.pth`), local paths, or Cosmos UUIDs. It binds inputs to the `predict2_5` inference wrapper and renders the synthesized 360-degree novel views directly.

### 2. Core Post-Training & Inference Package (`predict2_5/`)
* **`inferencer.py`**: Declares the main `Inferencer` class, managing the DiT denoising schedule, batched CFG forward passes ($B=2$), minimal 5-frame VAE anchor encoding, and frame-token replacement anchoring.
* **`module.py`**: Defines `CosmosXRay360` PyTorch LightningModule with Rectified Flow loss formulation, Selective Activation Checkpointing (SAC), FSDP EMA shard-wise updates, and classifier-free guidance.
* **`datamodule.py`**: Implements `PreRenderedDataModule` supporting raw 93-view PNGs, single-file `.pt`/`.npy` binary containers, and pre-cached Wan2.1 VAE latents (`PreRenderedLatentDataset`).
* **`trainer.py`**: CLI entrypoint and configuration harness (`TrainingConfig`) for single/multi-GPU DDP and FSDP training runs.
* **`hf.py` & `text_encoder.py`**: Handles Hugging Face hub artifact resolution, Wan2.1 VAE tokenizer downloads, and Cosmos-Reason1 text embeddings.
* **`dvr/` & `utils/`**: Implements legacy differentiable volume rendering strategies and generic distributed/image I/O preprocessing helpers.

### 3. SOTA Baselines & Evaluation (`baselines/`)
Standardized environment for out-of-distribution (OOD) cross-dataset splits (`TCIA` + `MELA` for training $\to$ `NSCLC` for testing):
* **`evaluate.py`**: Centralized evaluation script computing PSNR, SSIM, and Latency identically across all models.
* **`models/`**: Wrapper classes standardizing inputs/outputs to `256x256` tensors scaled between `[0, 1]` via Beer-Lambert transmissive mappings.
* **`cloned/`**: Contains the source code of SOTA comparative models (SV-DRR, MedNeRF, NAF, PixelNeRF, XRaySyn, Dx2CT).
