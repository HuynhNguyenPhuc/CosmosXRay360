# Cosmos-Predict2.5: Comprehensive Benchmarking & Experimental Protocol (`EXPERIMENT.md`)

**Document Version:** 1.0  
**Target Publication:** MICCAI 2026 Resubmission Benchmark Suite  
**Location:** `docs/cosmos-predict2.5/EXPERIMENT.md`  

---

## 1. Executive Summary

This document serves as the master experimental protocol and benchmarking specification for **Cosmos-Predict2.5** and the unified suite of **6 single-view NVS baselines** (SV-DRR, XRaySyn, MedNeRF, PixelNeRF, NAF, Dx2CT).

All benchmarking strictly adheres to the zero-leakage **Out-of-Domain (OOD) Cross-Dataset Protocol** defined in `datasets/cross_dataset_split.json`.

```
========================================================================================
                                CROSS-DATASET OOD PROTOCOL
========================================================================================
TRAIN / VAL SPLIT (1,296 Volumes)                   TEST SPLIT (402 Volumes - STRICT OOD)
├── TCIA Chest CT (771 cases)                       └── NSCLC Radiogenomics (402 cases)
└── MELA2022 Chest CT (525 cases)                   *(Never exposed to training/caching)*
========================================================================================
```

---

## 2. Experimental Setup & Hardware Specifications

### 2.1 Hardware Environment
- **Training Cluster:** 4× NVIDIA A100-SXM4-80GB GPUs / GCP `a2-ultragpu-4g` Compute Engine instance (single NVLink-connected node — `torchrun --nproc_per_node=4` with no `--nnodes`/multi-node config assumes a single node; **not** 4× separate `a2-ultragpu-1g` VMs, which was this document's prior, incorrect machine-type reference).
- **Baseline Worker Nodes:** GCP `g2-standard-8` (1x NVIDIA L4 24GB) and `a2-highgpu-1g` (1x NVIDIA A100 40GB).
- **Environment Stack:** CUDA 13.0 / 12.8, PyTorch 2.x, `uv` package manager, DiffDRR Siddon-Jacob raymarcher.

### 2.1.1 Measured Per-GPU Batch-Size / Step-Time Sweep (2026-08-13)

Before committing to a multi-day 4-GPU run, `scripts/benchmark_step_time.py` was run on a single provisioned `a2-ultragpu-1g` (1× A100-80GB) VM against the real `CosmosXRay360` DiT (random-init weights — architecture/VRAM footprint is checkpoint-independent) with `sac_mode=predict2_2b_720_aggressive`, `precision=bf16-mixed`, synthetic pre-cached latents (bypasses VAE/dataset I/O to isolate DiT forward+backward+optimizer cost):

| Per-GPU Batch Size | Mean Step Time (s) | Samples/s | Peak VRAM (GB / 80GB) |
|:---:|:---:|:---:|:---:|
| 1 | 1.065 | 0.94 | 39.8 |
| 2 | 1.623 | 1.23 | 41.2 |
| 4 | 3.039 | 1.32 | 44.1 |
| 6 | 4.462 | 1.34 | 47.0 |
| 8 | 5.885 | 1.36 | 49.8 |
| 12 | 8.778 | 1.37 | 55.5 |
| 16 | 11.597 | 1.38 | 61.3 |

**Findings:**
- **No OOM up to batch size 16** on a single GPU (61.3/80 GB) — `PLAN.md`'s documented `--batch_size 4` has substantial headroom (44.1/80 GB, ~55%). The fixed ~38GB floor at batch size 1 is FusedAdam's fp32 master weights + optimizer states for the 2.06B-parameter DiT (not activations), consistent with `capturable=True, master_weights=True` in the training log.
- **Throughput plateaus fast**: marginal cost is a consistent ~1.43 GB/sample (SAC keeping activation memory low), but samples/s only rises 0.94→1.38 (+47%) from batch size 1→16 (16× the batch, 16× the VRAM) — the DiT's per-sample compute over 24 latent frames is close to compute-bound already at small batch sizes, so there's little throughput reason to push per-GPU batch size past the documented `4` (which hits the ablation protocol's effective global batch size 16 exactly across 4 GPUs, at ~55% single-GPU VRAM).
- **Not measured here**: real FSDP inter-GPU communication overhead (all-gather/reduce-scatter), which only shows up in an actual multi-GPU run — these single-GPU numbers are a lower bound on per-step wall time. At the measured single-GPU batch-size-4 step time (~3.04s) as a floor, extrapolated (unvalidated) wall-clock for the documented step budgets: 10,000 steps (per ablation variant, `ABLATION_STUDY.md`) ≈ 8.4+ hours; 100,000 steps (`PLAN.md` Phase 3 full run) ≈ 84+ hours. Actual FSDP overhead will push both higher — validate with a short real 4-GPU FSDP run before trusting these for scheduling/cost purposes.
- **Verification**: `research-log.md` 2026-08-13 entry has the full provisioning/run record; raw results in `scripts/benchmark_step_time.py`'s `--output_json` output.

### 2.2 Standardized Evaluation Resolution & Geometry
- **Resolution:** $256 \times 256$ pixels, min-max normalized to $[0, 1]$, bilinear interpolation with `align_corners=True`.
- **Views:** $93$ rotation frames spanning $0^\circ \to 360^\circ$ azimuth ($\text{endpoint}=\text{True}$).
- **Camera Parameters:** Matching DiffDRR geometry (`dist=1.8m`, `fov=35^\circ`, `min_depth=0.1m`, `max_depth=3.0m`).

---

## 3. Quantitative Benchmark Results (NSCLC OOD Test Split - 402 Cases)

The table below presents the master benchmark structure for comparative evaluation across all 6 SOTA baselines, training-free geometric priors, and Cosmos-Predict2.5 on the held-out **NSCLC Radiogenomics** test set:

| Model / Method | Category / Paradigm | PSNR (dB) ↑ | SSIM ↑ | LPIPS (Alex) ↓ | Inference Latency (s/case) ↓ | Peak GPU Memory |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Horizontal Flip ($I_{\text{PA}}$ at $180^\circ$)** | Geometric Mirror-Symmetry Prior | **27.50** *(at $180^\circ$)* | **0.8790** | **0.0527** | **< 0.01s** | **< 0.1 GB** |
| **XRaySyn** (AAAI '21) | Voxel GAN + Refinement | TBD | TBD | TBD | TBD | TBD |
| **Dx2CT** (ICASSP '25) | Slice Diffusion + CT Feature | TBD | TBD | TBD | TBD | TBD |
| **SV-DRR** (MICCAI '25) | Pose-Conditioned 2D DiT | TBD | TBD | TBD | TBD | TBD |
| **MedNeRF** (EMBC '22) | Per-Scan Generative NeRF | TBD | TBD | TBD | TBD | TBD |
| **NAF** (MICCAI '22) | Continuous Coordinate Field | TBD | TBD | TBD | TBD | TBD |
| **PixelNeRF** (CVPR '21) | Feed-Forward Prior NeRF | TBD | TBD | TBD | TBD | TBD |
| **Cosmos-Predict2.5 (Zero-Shot)** | Pretrained WFM (No Post-Train) | TBD | TBD | TBD | TBD | TBD |
| **Cosmos-Predict2.5 (Ours - Full)**| **Physical World Model (DiffDRR)**| **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

*Note: Baseline retraining on current fixed codebase is actively executing across parallel GCP worker nodes (`cosmos-worker-1`, `cosmos-worker-2`, `cosmos-worker-4`, `cosmos-worker-5`). Table cells will be populated directly from `baselines/evaluate.py` log outputs upon run completion.*

---

## 4. Qualitative Evaluation Protocols

### 4.1 Pathological Variation Audit (VinDr-CXR)
To address **Reviewer #2 (R2-1)**, models are evaluated qualitatively on real-world clinical radiographs from VinDr-CXR exhibiting major thoracic pathologies:
1. **Cardiomegaly:** Verifying that cardiac silhouette expansion (CTR > 0.5) rotates smoothly without unnatural volume shrinkage.
2. **Pleural Effusion:** Auditing the fluid meniscus boundary across rotation angles to confirm gravity-consistent attenuation behavior.
3. **Pneumonia / Consolidation:** Confirming focal parenchymal opacities remain localized in 3D space rather than smearing across views.

### 4.2 Musculoskeletal Zero-Shot Probe (Stanford MURA)
To establish transfer boundaries on non-chest anatomies (**Reviewer #2 R2-1**):
- Zero-shot evaluation on Stanford MURA hand/wrist radiographs.
- **Protocol:** Assess macro-geometry and bone alignment vs micro-trabecular bone texture smoothing to define domain transfer limits.

---

## 5. Planned Downstream 3D CT Reconstruction Benchmark Protocol

To address **Reviewer #3 (R3-1) & Reviewer #4 (R4-2)**, synthesized 93-view sequences will be evaluated as input projections for $3\text{D}$ CT volume reconstruction once NVS inference completes:

| Reconstruction Algorithm | Input Source | 3D Volume PSNR (dB) ↑ | 3D Volume SSIM ↑ | Volumetric Reconstruction Time |
|:---|:---|:---:|:---:|:---:|
| **FDK Filtered Backprojection** | 1-View CXR (Direct) | TBD | TBD | TBD |
| **FDK Filtered Backprojection** | SV-DRR 93 Views | TBD | TBD | TBD |
| **FDK Filtered Backprojection** | **Cosmos-Predict2.5 93 Views** | **TBD** | **TBD** | **TBD** |
| **Beer-Lambert NeRF Optimization**| **Cosmos-Predict2.5 93 Views** | **TBD** | **TBD** | TBD |

*Evaluation Target:* Measure whether synthesizing dense $360^\circ$ projections via Cosmos-Predict2.5 converts the ill-posed $1$-view CT reconstruction problem into a well-conditioned tomographic task.

