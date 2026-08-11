# Cosmos-Predict2.5: World Foundation Model Adaptation for Single-View 360° Novel View Synthesis

**Paper:** *Cosmos-Predict2.5: World Foundation Model Adaptation for Single-View 360° Novel View Synthesis*
**Authors:** CosmosXRay360 Team
**Venue/Publication:** Target MICCAI 2026 Submission
**Reference Baseline Manual:** `docs/LITERATURE.md`

---

## 1. Executive Summary & Paradigm

**Cosmos-Predict2.5** adapts NVIDIA's **Cosmos Predict 2.5** 2B video diffusion foundation model to the medical imaging domain, establishing a novel **Spatiotemporal World Model** macro-paradigm for single-view 360° novel view synthesis (NVS).

Given a single 2D frontal PA chest radiograph $I^{\text{PA}}$, prior methods either attempt per-scan 3D neural field fitting (MedNeRF, NAF) incurring extreme test-time optimization latency (minutes to hours), or rely on unconstrained 2D latent diffusion (SV-DRR) that suffers from multi-view geometric hallucination across $360^\circ$ rotation trajectories.

Cosmos-Predict2.5 addresses both limitations by leveraging:
1. **Pretrained Spatiotemporal Physics Priors:** Pretrained on large-scale physical video dynamics, providing intrinsic camera trajectory and volumetric continuity priors.
2. **Frame-Token Replacement Conditioning:** A structurally anchored conditioning mechanism that forces the first frame $f_0$ (PA radiograph) latent and velocity predictions to match ground-truth anchor states at every UniPC denoising step, eliminating output drift across the entire 93-frame $360^\circ$ rotation trajectory.
3. **Ultra-Fast Generation:** Synthesizes a full 93-frame $360^\circ$ rotation video ($256\times256$ resolution) in $\sim 3.8\text{s}$ per patient case.

---

## 2. Mathematical Formulation & Architecture

```
                                    ┌───────────────────────────────┐
                                    │ Frontal PA Radiograph  I_PA  │
                                    └───────────────┬───────────────┘
                                                    │ Repeat T=93
                                                    ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│  Cosmos-Reason 1.0 Encoder   │   │     Wan2.1 VAE Tokenizer      │
│     c_text = CR1(Prompt)      │   │   z_cond = VAE_encode(V_PA)   │
└───────────────┬───────────────┘   └───────────────┬───────────────┘
                │                                   │
                └─────────────────┬─────────────────┘
                                  │
                                  ▼
                ┌──────────────────────────────────┐
                │  FlowUniPC Denoising Loop (T=35) │
                │  ┌─────────────────────────────┐ │
                │  │ MinimalV1LVGDiT (2B Params) │ │
                │  │ 3D RoPE Trajectory Embeds   │ │
                │  └──────────────┬──────────────┘ │
                │                 │                │
                │                 ▼                │
                │    Frame-Token Replacement       │
                │    z_t[0] <-- z_cond[0]          │
                │    v_t[0] <-- noise - z_cond[0]  │
                └─────────────────┬────────────────┘
                                  │
                                  ▼
                ┌──────────────────────────────────┐
                │     Wan2.1 VAE Decoder          │
                │    V_360 = VAE_decode(z_final)   │
                └─────────────────┬────────────────┘
                                  │
                                  ▼
                ┌──────────────────────────────────┐
                │ 93-Frame 360° Rotation Video     │
                └──────────────────────────────────┘
```

### A. Temporal Extension & Latent Encoding

Input 2D frontal radiograph $I^{\text{PA}} \in \mathbb{R}^{1 \times 1 \times H \times W}$ is min-max normalized to $[0, 1]$, mapped to $[-1, 1]$, and replicated across $T=93$ temporal frames to form anchor video tensor $V_{\text{PA}} \in \mathbb{R}^{1 \times 3 \times 93 \times 256 \times 256}$.

The spatio-temporal Wan2.1 VAE tokenizer encodes $V_{\text{PA}}$ into latent space:
$$z_{\text{cond}} = \mathcal{E}_{\text{VAE}}(V_{\text{PA}}) \in \mathbb{R}^{1 \times C_z \times T_z \times H_z \times W_z}$$

### B. MinimalV1LVGDiT Backbone & Flow Matching

Denoising is powered by a 2B-parameter `MinimalV1LVGDiT` backbone operating under rectified flow matching semantics:
$$\mathrm{d}z_t = v_\theta(z_t, t, c) \mathrm{d}t$$
where $c = (c_{\text{text}}, z_{\text{cond}}, \text{mask}_{\text{cond}})$ contains the Cosmos-Reason 1.0 text embedding $c_{\text{text}}$ and spatial conditioning mask $\text{mask}_{\text{cond}}$.

3D Rotary Position Embeddings (3D-RoPE) encode joint spatial and temporal coordinates, preserving continuous geometric continuity along the 360° circular camera orbit trajectory.

### C. Frame-Token Replacement Conditioning

To maintain strict structural alignment with the input radiograph throughout the 35 UniPC sampling steps without drifting or hallucinating non-anatomical artifacts, Cosmos-Predict2.5 enforces **Frame-Token Replacement Conditioning** at every denoising step $t$:

1. **Input State Splicing:**
   $$z_t \leftarrow M \odot z_{\text{cond}} + (1 - M) \odot z_t$$
   where $M \in \{0, 1\}^{1 \times 1 \times T_z \times H_z \times W_z}$ is the binary conditioning mask ($M=1$ for frame 0, $M=0$ for frames $1 \dots 92$).

2. **Velocity Splicing:**
   In rectified flow matching, target velocity is $v_0 = \text{noise} - z_{\text{cond}}$. The predicted velocity $v_\theta(z_t, t, c)$ is modified as:
   $$\hat{v}_t \leftarrow M \odot (\text{noise} - z_{\text{cond}}) + (1 - M) \odot v_\theta(z_t, t, c)$$

3. **Classifier-Free Guidance (CFG):**
   Classifier-Free Guidance scale $w = 1.5$ balances textual/video conditioning against unconditional generation:
   $$v_{\text{guided}} = v(z_t, t, \emptyset) + w \cdot \big(v(z_t, t, c) - v(z_t, t, \emptyset)\big)$$

### D. VAE Decoding & Post-Processing

Final latent sample $z_0$ is decoded by Wan2.1 VAE decoder:
$$V_{360} = \mathcal{D}_{\text{VAE}}(z_0) \in \mathbb{R}^{1 \times 3 \times 93 \times 256 \times 256}$$
Output frames are min-max clamped to $[0, 1]$ and converted to uint8 $(93, 256, 256, 3)$ grayscale/RGB arrays.

---

## 3. Submission vs. Resubmission Protocol Changes

### A. Submitted Manuscript (MICCAI 2026 Submission 2 Ground Truth)
- **Data Split:** $1,208 / 150 / 340$ patient-level random split pooled across all three CT datasets ($1,698$ total cases).
- **Renderer:** PyTorch3D absorption–emission volumetric raymarcher (`predict2_5/dvr/renderer.py`).
- **Evaluation Metrics:** PSNR (dB) and SSIM across $360^\circ$ views.
- **Comparative Baselines (4):** XRaySyn, MedNeRF, Dx2CT, and SV-DRR.
- **Training Setup & Budget:** $10,000$ steps with batch size $1/\text{GPU} \times 4\text{ accum} = 16$ effective global batch size ($\sim 23\text{h}$ on $4 \times \text{A100-80GB}$).
- **Reported Performance:**
  - **Cosmos-Predict2.5 Post-Trained:** $23.26\text{ dB}$ PSNR / $0.777$ SSIM.
  - **Zero-Shot Pretrained Baseline:** $7.49\text{ dB}$ PSNR / $0.208$ SSIM.
  - **3D VAE Reconstruction Upper Bound:** $43.2\text{ dB}$ DRR / $37.7\text{ dB}$ XR.

### B. Resubmission Enhancements (Current Codebase & Benchmark Protocol)
- **Cross-Dataset Out-of-Domain (OOD) Split:** Strict zero-leakage evaluation (`datasets/cross_dataset_split.json`):
  - **Train / Val:** TCIA ($771$ scans) + MELA2022 ($525$ scans) = $1,296$ volume cases.
  - **Test (OOD):** NSCLC Radiogenomics ($402$ scans) strictly held out.
- **Physical Renderer Migration:** DiffDRR Siddon-Jacob raymarching (`renderers/diffdrr/renderer.py`) providing physically exact Beer-Lambert raytracing without C++ extension compilation overhead.
- **Expanded Evaluation Metrics:** PSNR (dB) ↑, SSIM ↑, LPIPS (AlexNet) ↓, and Wall-Clock Latency (s/case) ↓.
- **Unified 6-Baseline Benchmark Suite:** Standardized wrapper API (`infer_multi_views`) across SV-DRR, XRaySyn, MedNeRF, PixelNeRF, NAF, and Dx2CT in `baselines/evaluate.py`.
- **Accelerated Training & Inference Engine:**
  - **SAC Aggressive Mode:** `predict2_2b_720_aggressive` reduces activation VRAM from $\sim 28\text{GB}$ to $\sim 6\text{GB}$.
  - **Offline VAE Latent Pre-Caching:** `PreRenderedLatentDataset` bypasses 3D VAE encoder compute during training ($\sim 18\text{GB}$ VRAM savings, $\sim 35\%$ step speedup).
  - **FSDP Shard-Wise EMA Update:** In-place CPU EMA updates without all-gather communication overhead.
  - **Native Inference Acceleration:** Batched CFG forward pass ($B=2$, halving DiT calls to $35$), native `bfloat16` weight casting, and minimal 5-frame VAE anchor encoding.

---

## 4. Stated Advantages & Limitations

### Advantages
- **No Per-Scan Fitting:** Unlike MedNeRF or NAF, does not require hours of gradient descent per test scan.
- **Geometric Alignment:** Frame replacement conditioning prevents drift and severe hallucination seen in unconstrained 2D diffusion (SV-DRR).
- **Physical Realism:** Leverages pre-trained video world dynamics to synthesize continuous 360° rotational parallax.

### Limitations
- **High VRAM Footprint during DiT Forward:** Requires $\ge 16\text{GB}$ VRAM for 2B DiT model (mitigated by `bfloat16` and optional CPU offload).
- **Fixed Sampling Frame Count:** VAE temporal tokenizer is calibrated for 93-frame duration ($360^\circ / 92 \approx 3.91^\circ$ step size).
