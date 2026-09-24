# CosmosXRay360 - 360° Novel View Synthesis from Single Chest X-Ray Images via a World Foundation Model

[🤗 HuggingFace Model](https://huggingface.co/phuchuynh0904/CosmosXRay360)

---

CosmosXRay360 is a medical imaging project built around **Cosmos Predict 2.5** for generating multi-view chest X-ray sequences from a single input X-ray image. This branch provides inference code, a Gradio interface, and the full post-training pipeline (dataset rendering + Lightning trainer) used to produce the released checkpoint.


## 🔥 News

- **2026-09-24** — `train` branch: DiffDRR-based dataset rendering, VAE latent pre-caching, and the Lightning post-training pipeline (FSDP EMA, physical loss regularizers, CFG-dropout anchoring)
- **2026-04-21** — Initial release: inference code, Gradio demo, and model checkpoints


## Overview

CosmosXRay360 is an end-to-end pipeline for generating multi-view chest X-Ray sequences from a single input image using a world foundation model.

It provides:

- **Inference** — Standalone Python scripts and Gradio web interface
- **Model Integration** — Seamless use of Cosmos base models and project-specific checkpoints
- **Deployment Ready** — Flexible loading from local paths, Hugging Face Hub, or Cosmos UUIDs


## ⚡ Quick Start

```bash
git clone --recursive https://github.com/phuchuynh0904/CosmosXRay360.git
cd CosmosXRay360

uv venv .venv --python 3.10
source .venv/bin/activate

# Install project dependencies
uv pip install -r requirements.txt

# Install cosmos-predict2.5 with CUDA extra (cu128 for CUDA 12.x, cu130 for CUDA 13.x)
cd cosmos-predict2.5
uv pip install -e ".[cu128]"
cd ..

python app.py --checkpoint-path "hf://phuchuynh0904/CosmosXRay360/net_ema.pth" --share
```


## Features

- 🎬 Generate 93-frame multi-view X-Ray sequences from a single input image
- 🎯 Run inference via Python API or interactive Gradio app
- 🖼️ Two visualization modes: **Standard** (raw) and **Enhanced Contrast**
- 📦 Export checkpoints from Lightning `.ckpt` to standalone `.pth`
- 🔗 Flexible model loading: local paths, Hugging Face URIs, or Cosmos UUIDs
- ⚡ CPU and GPU support with configurable inference parameters
- 🏋️ Post-train on your own CT-derived DRR data, with FSDP EMA, physical loss regularizers, and VAE latent pre-caching


## 🎁 Model Checkpoints

CosmosXRay360 relies on both **Cosmos base models** and a **project-specific checkpoint**:

| Component | Role | Source | Status |
|---|---|---|---|
| Cosmos Predict 2.5 Tokenizer | VAE tokenizer for image encoding | Cosmos base | ✅ Auto-downloaded |
| Cosmos Predict 2.5 Base 2B | Foundation DiT model | Cosmos base | ✅ Auto-downloaded |
| Cosmos-Reason1 Encoder | Text conditioning encoder | Cosmos base | ✅ Auto-downloaded |
| Project Checkpoint | Post-trained model for X-Ray generation | Hugging Face / local | ✅ Required |

**Usage Notes:**
- Only the **project checkpoint (`.pth` + `config.json`)** needs to be provided manually
- All Cosmos base models are automatically downloaded and cached on first run
- Ensure you have a valid checkpoint before running inference


## 🤗 Get Started with CosmosXRay360

CosmosXRay360 has been primarily tested on Linux. Other operating systems may work, but issues related to dependencies, CUDA compatibility, or file paths may occur. Contributions to improve cross-platform support are welcome.

### System Requirements
- **Operating System:** Linux (other OS may work but are not officially supported)
- **Python Version:** 3.10 or higher
- **CUDA Version:** 12.8 or 13.x (versions below 12.8 may be incompatible with the Cosmos-OSS package)

### Install Requirements

**1. Clone the repository:**
```bash
git clone --recursive https://github.com/phuchuynh0904/CosmosXRay360.git
cd CosmosXRay360
````

**2. Install dependencies** using `uv` (recommended) or `pip`:

```bash
# Install uv (fast Python package installer)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment
uv venv .venv --python 3.10

# Activate the virtual environment
source .venv/bin/activate          # Linux/macOS
# OR
.\.venv\Scripts\Activate.ps1       # Windows PowerShell

# Install project dependencies
uv pip install -r requirements.txt
```

**3. Install the Cosmos submodule:**

```bash
# Install cosmos-predict2.5 with CUDA extra (cu128 for CUDA 12.x, cu130 for CUDA 13.x)
cd cosmos-predict2.5
uv pip install -e ".[cu128]"
cd ..
```

**4. (Optional) Install PyTorch3D:**

The default rendering path (`renderers/diffdrr/`, used for dataset pre-rendering and training) depends only on `diffdrr` + `torchio`, both installed via `requirements.txt` in step 2. PyTorch3D is only needed for the legacy DVR renderer (`predict2_5/dvr/`), kept as a reference implementation that DiffDRR is checked against — skip this step unless you need that comparison.

```bash
# Install pytorch3d for the legacy X-Ray volume renderer
# (install with --no-build-isolation to ensure torch is available during build)
uv pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

### Download Checkpoints

Download pre-trained checkpoints from Hugging Face:

```bash
# Using git + git-lfs
git lfs install
git clone https://huggingface.co/phuchuynh0904/CosmosXRay360
cd CosmosXRay360
```

Or download using Python:

```python
from huggingface_hub import hf_hub_download

checkpoint_path = hf_hub_download(
    repo_id="phuchuynh0904/CosmosXRay360",
    filename="net_ema.pth",
)

config_path = hf_hub_download(
    repo_id="phuchuynh0904/CosmosXRay360",
    filename="config.json",
)
```

You may need to authenticate with Hugging Face to access these files. The checkpoint and configuration files will be stored in your local cache directory (e.g., `~/.cache/huggingface/hub/`).


## Code Usage

Use CosmosXRay360 in **Python** by loading checkpoints from one of the following sources:
- **Local paths** — downloaded files
- **Hugging Face Hub (direct)** — `hf://phuchuynh0904/CosmosXRay360/net_ema.pth`
- **Cosmos UUIDs** — base model assets (auto-downloaded)

### Option 1: Load from Hugging Face Hub (no download required)

```python
from PIL import Image
from predict2_5.inferencer import Inferencer

inferencer = Inferencer(
    checkpoint_path="hf://phuchuynh0904/CosmosXRay360/net_ema.pth",
    config_path="hf://phuchuynh0904/CosmosXRay360/config.json",
    device="cuda",  # Use "cpu" if CUDA is unavailable
)

image = Image.open("sample_xray.png").convert("RGB")

frames = inferencer.predict(
    image=image,
    cfg_scale=1.5,
    num_steps=35,
    seed=42,
)

print(f"Generated {len(frames)} frames of shape {frames[0].shape}")

for idx, frame in enumerate(frames):
    Image.fromarray(frame).save(f"output/frame_{idx:03d}.png")
```

### Option 2: Load from Local Files

```python
from PIL import Image
from predict2_5.inferencer import Inferencer

# First download checkpoints using: git clone + git-lfs (see Download Checkpoints section)

inferencer = Inferencer(
    checkpoint_path="path/to/downloaded/net_ema.pth",
    config_path="path/to/downloaded/config.json",
    device="cuda",
)

image = Image.open("sample_xray.png").convert("RGB")

frames = inferencer.predict(
    image=image,
    cfg_scale=1.5,
    num_steps=35,
    seed=42,
)
```

**Expected Output:**

* `frames`: List of 93 RGB frames
* Each frame: `uint8` array with shape `(256, 256, 3)`

### Gradio App

Launch the interactive demo:

```bash
# From local checkpoint
python app.py --checkpoint-path checkpoints/net_ema.pth --share

# From Hugging Face Hub (no download required)
python app.py \
  --checkpoint-path "hf://phuchuynh0904/CosmosXRay360/net_ema.pth" \
  --config-path "hf://phuchuynh0904/CosmosXRay360/config.json" \
  --share
```

If `config.json` is in the same directory as the checkpoint, you can omit `--config-path`:

```bash
python app.py --checkpoint-path checkpoints/net_ema.pth --share
```

**Features:**

* 📤 Upload X-Ray images from file or clipboard
* 🎚️ Adjust CFG scale and inference steps in real time
* 🎬 Browse generated frames with a frame inspector
* 🖼️ Toggle between **Standard** (raw) and **Enhanced Contrast** modes
* 💾 Download generated sequences as a video or individual frames

---

## 🏋️ Training

CosmosXRay360 is post-trained on paired **CT-derived DRR (Digitally Reconstructed Radiograph) rotation videos**: 93 views spanning 0–360° azimuth, rendered from chest CT volumes with the [DiffDRR](https://github.com/eigenvivek/DiffDRR) renderer (`renderers/diffdrr/`). Training uses a cross-dataset **out-of-distribution (OOD) split** — train/val on **TCIA + MELA2022**, held-out test on **NSCLC-Radiomics (LUNG1)** — so never mix NSCLC cases into the training or caching steps below.

### 1. Download the raw CT datasets

Training expects raw CT volumes (`.nii.gz`) laid out as:

```
datasets/
├── TCIA/images/*.nii.gz
├── MELA2022/raw/train/images/*.nii.gz    # + raw/val/images/ (validation)
└── NSCLC/processed/train/images/*.nii.gz # held-out OOD test set — never used for training
```

- **TCIA** and **NSCLC-Radiomics (LUNG1)** are public collections on [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/); **MELA2022** is from the [MELA 2022 Grand Challenge](https://mela22.grand-challenge.org/).
- If your team has an internal mirror on Google Cloud Storage, `scripts/download_folder_from_gcs.py` pulls a GCS folder in parallel:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json

uv run python scripts/download_folder_from_gcs.py \
  --bucket   <your-gcs-bucket> \
  --gcs-folder data/TCIA/ \
  --local-dir  datasets/TCIA \
  --num-workers 8
# repeat per dataset (TCIA / MELA2022 / NSCLC), adjusting --gcs-folder and --local-dir
```

Ask your team lead for the bucket name and folder layout if you don't already have access.

### 2. Pre-render DRR projections

Converts raw CT volumes into 93-view 360° X-ray projections (plus PA/LAT views), populating `datasets/pre_rendered/{train,test}/<patient>/{pa.png,lat.png,views/*.png}`:

```bash
uv run python datasets/pre_render_diffdrr.py \
  --dest_dir datasets/pre_rendered \
  --device cuda
```

Useful flags: `--datasets TCIA,MELA2022` to render a subset, `--max_files N` for a quick dry run, `--overwrite` to re-render existing cases.

### 3. (Optional) Pre-encode VAE latents

Skips repeated VAE encoding during training by caching Wan2.1 VAE latents to disk once — recommended for multi-epoch runs:

```bash
uv run python scripts/pre_encode_latents.py \
  --dataset_dir datasets/pre_rendered \
  --output_dir  datasets/pre_rendered_latents \
  --batch_size 1 \
  --device cuda
```

### 4. Launch training

```bash
uv run python predict2_5/trainer.py \
  --dataset_path datasets/pre_rendered \
  --num_gpus 4 \
  --strategy fsdp \
  --batch_size 1 \
  --precision bf16-mixed \
  --experiment_name my_run \
  --output_dir outputs
```

Add `--use_latent_cache --latent_cache_dir datasets/pre_rendered_latents` if you ran step 3. Other notable flags (see `predict2_5/trainer.py`'s `parse_args()` for the full list):

| Flag | Default | Purpose |
|---|---|---|
| `--model_size` | `2B` | DiT size (`2B`/`7B`/`14B`) |
| `--resume_from_checkpoint` | `None` | Resume from a Lightning `.ckpt` |
| `--enable_ema` / `--ema_rate` | `True` / `0.10` | FSDP-sharded EMA of the weights |
| `--gamma_side` / `--loss_atten_weight` | `1.0` / `0.02` | Physical loss regularizer weights (angular offset, attenuation mass) |
| `--learning_rate` / `--max_iters` | `2^-14.5` / `100000` | Core optimization hyperparameters |
| `--build_cache_only` | off | Prepare the disk cache and exit, without training |

Checkpoints and TensorBoard logs land under `<output_dir>/<experiment_name>/`.

### 5. Export a checkpoint for inference

Converts a Lightning `.ckpt` into the standalone `.pth` + `config.json` pair the `Inferencer` (and `app.py`) expect:

```bash
uv run python -m scripts.export_dit_checkpoint export \
  --ckpt  outputs/my_run/checkpoints/epoch=0042.ckpt \
  --out   exported/dit_ema_epoch42.pth \
  --export-config
```

By default this exports the EMA weights (`--source net_ema`); pass `--source net` for the raw weights. See [Code Usage](#code-usage) above to load the exported checkpoint with `Inferencer`.

---

## Model Description

**Task:** Novel View Synthesis for X-Ray images

**Input:** A single chest X-Ray image

**Output:** A 93-frame multi-view sequence (360° rotation)

**Base Model:** Cosmos Predict 2.5 (DiT). The tokenizer and text encoder are initialized from pretrained models.

**Post-training:** Trained on multi-view chest X-Ray data to improve geometric consistency and anatomical plausibility.


## Limitations

- ⚠️ **Not for clinical diagnosis** — outputs are intended for research purposes only
- ⚠️ **Hallucinations may occur** — the model may generate anatomically implausible structures in unseen views
- ⚠️ **Requires valid checkpoints** — base Cosmos models must be downloaded before first use
- ⚠️ **GPU recommended** — CPU inference is significantly slower


## Acknowledgments

CosmosXRay360 builds on the following foundational works:

- **[NVIDIA Cosmos Predict 2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5)** — Foundation model architecture and training scripts
- **[DiffDRR](https://github.com/eigenvivek/DiffDRR)** — Differentiable, Siddon-Jacob ray-tracing DRR renderer used for dataset generation
- **[MONAI](https://github.com/Project-MONAI/MONAI)** — Medical imaging preprocessing and evaluation tools

We are grateful to the broader open-source community for their valuable tools and contributions.


## 📜 License

CosmosXRay360 is released under the **Apache License 2.0**. See [LICENSE](./LICENSE) for details.
