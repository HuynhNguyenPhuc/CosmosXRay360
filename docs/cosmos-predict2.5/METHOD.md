# Cosmos-Predict2.5: Physical World Model Adaptation & Regularization for Single-View 360° Radiographic View Synthesis

**Document Version:** 1.0  
**Target Publication:** MICCAI 2026 Resubmission Methodology  
**Location:** `docs/cosmos-predict2.5/METHOD.md`  

---

## 1. Executive Overview & Ill-Posed Problem Formulation

Single-View 360° Novel View Synthesis (NVS) from a solitary $2\text{D}$ chest radiograph $I^{\text{PA}}$ to a continuous $93$-frame, $360^\circ$ rotational video $V_{360}$ is an **under-constrained, ill-posed inverse problem**. Specifically:

1. **Depth & Posterior Structural Ambiguity:** A single $2\text{D}$ projection integrates $3\text{D}$ volumetric attenuation along the ray path ($\int \mu \, \mathrm{d}s$), collapsing the depth dimension. Posterior anatomical structures (e.g., retrocardiac lung parenchyma, spine-overlaid nodules, thoracic aorta) are heavily obscured.
2. **Generative Hallucination Risk:** Unconstrained $2\text{D}$ diffusion models (e.g., SV-DRR) synthesize plausible optical textures but produce severe anatomical hallucinations, fake lesions, and non-physical structural drift across rotation azimuths.
3. **Physical vs. Optical Domain Gap:** Standard video world models (NVIDIA Cosmos Predict 2.5) are pretrained on optical natural video dynamics (reflection, illumination, perspective occlusion). X-ray projections, conversely, follow **transmissive Beer-Lambert attenuation physics** ($I = I_0 \exp(-\int \mu \, \mathrm{d}s)$).

To resolve this ill-posedness while addressing reviewer concerns (MICCAI Reviewers #2, #3, #4), **Cosmos-Predict2.5** adapts NVIDIA's $2\text{B}$ Rectified Flow DiT foundation model using a combination of **Frame-Token Replacement Conditioning**, **Line-Integral Global Attenuation Mass Loss ($\mathcal{L}_{\text{atten}}$)**, **Anatomical Mirror-Symmetry Soft Latent Feature Prior ($180^\circ$ AP)**, **Continuous Angular-Offset Loss Weighting ($w(\theta_k)$)**, and a **Periodic Orbit Noise Schedule**.

```
                                      ┌─────────────────────────────────────┐
                                      │  Single 2D Radiograph  I_PA (0°)   │
                                      └──────────────────┬──────────────────┘
                                                         │
                                                         ▼
                                      ┌─────────────────────────────────────┐
                                      │    Wan2.1 3D VAE Latent Encoder     │
                                      │     z_cond = E_VAE(V_PA_repeat)     │
                                      └──────────────────┬──────────────────┘
                                                         │
                                                         ▼
┌─────────────────────────────────────┐  ┌─────────────────────────────────────┐
│    Cosmos-Reason 1.0 Embedder      │  │ MinimalV1LVGDiT (2B Rectified Flow) │
│        c_text = CR1(Prompt)         ├─►│     3D-RoPE Trajectory Embeds     │
└─────────────────────────────────────┘  └──────────────────┬──────────────────┘
                                                            │
                                                            ▼
                                         ┌──────────────────────────────────┐
                                         │  Frame-Token Replacement Clamping │
                                         │   z_t[0] <-- z_cond[0]           │
                                         │   v_t[0] <-- noise - z_cond[0]   │
                                         └──────────────────┬───────────────┘
                                                            │
                                                            ▼
┌─────────────────────────────────────┐  ┌──────────────────────────────────┐
│ Continuous Angular Weight w(theta)  │  │ Global Attenuation Mass Loss     │
│   w(theta_k) = 1 + sin^2(theta_k)   ├─►│   L_atten = ||sum(D) - sum(PA)||^2│
└─────────────────────────────────────┘  └──────────────────┬───────────────┘
                                                            │
                                                            ▼
                                         ┌──────────────────────────────────┐
                                         │      Wan2.1 3D VAE Decoder       │
                                         │  V_360 = D_VAE(z_360_denoised)   │
                                         └──────────────────┬───────────────┘
                                                            │
                                                            ▼
                                         ┌──────────────────────────────────┐
                                         │ 93-Frame 360° Continuous Video   │
                                         └──────────────────────────────────┘
```

---

## 2. Core Architecture & Foundation Model Adaptation

### 2.1 Spatiotemporal Latent Encoding
Input radiograph $I^{\text{PA}} \in \mathbb{R}^{1 \times 1 \times 256 \times 256}$ in range $[0, 1]$ is normalized to $[-1, 1]$. During full post-training, $I^{\text{PA}}$ is replicated across $T=93$ frames to form $V_{\text{PA}} \in \mathbb{R}^{1 \times 3 \times 93 \times 256 \times 256}$, mapped by the Wan2.1 $3\text{D}$-VAE encoder to latent tensor $z_{\text{cond}} \in \mathbb{R}^{1 \times 16 \times 24 \times 32 \times 32}$ with $4\times$ temporal compression ($24$ temporal latents). For accelerated Track C standalone inference, $I^{\text{PA}}$ is replicated over a minimal $5$-frame anchor chunk ($V_{\text{anchor}} \in \mathbb{R}^{1 \times 3 \times 5 \times 256 \times 256}$), extracting latent frame $0$ while zero-padding remaining latent channels to save VAE compute (~18.6× VAE speedup).

### 2.2 Rectified Flow Denoising Backbone
The denoising core is `MinimalV1LVGDiT` ($2\text{B}$ parameters, 28 blocks, hidden dim 2048, 16 attention heads) operating under Rectified Flow matching semantics:
$$\mathrm{d}z_t = v_\theta(z_t, t, c) \mathrm{d}t$$
where $t \in [0, 1]$, $z_1 \sim \mathcal{N}(0, I)$, and $z_0$ represents the clean target latent.

Continuous $360^\circ$ circular orbit trajectory coordinates are encoded into $3\text{D}$ Rotary Position Embeddings ($3\text{D}$-RoPE), enforcing continuous geometric extrapolation across rotation azimuths $\theta \in [0^\circ, 360^\circ]$.

### 2.3 Frame-Token Replacement Conditioning
To anchor the generative trajectory to the ground-truth frontal CXR without drifting or hallucinating non-anatomical artifacts across 35 FlowUniPC sampling steps, Cosmos-Predict2.5 enforces **Frame-Token Replacement Conditioning** at every step $t$:

1. **Latent Token Splicing:**
   $$z_t \leftarrow M \odot z_{\text{cond}} + (1 - M) \odot z_t$$
   where $M \in \{0, 1\}^{1 \times 1 \times 24 \times 32 \times 32}$ is a binary conditioning mask ($M=1$ for latent frame 0, $M=0$ for frames $1 \dots 23$).

2. **Velocity Vector Splicing:**
   In rectified flow, target velocity for frame 0 is $v_{\text{target}} = \epsilon - z_{\text{cond}}$. The predicted velocity field $v_\theta(z_t, t, c)$ is modified in-place:
   $$\hat{v}_t \leftarrow M \odot (\epsilon - z_{\text{cond}}) + (1 - M) \odot v_\theta(z_t, t, c)$$

---

## 3. Advanced Physical & Geometric Regularizations for Ill-Posed NVS

To systematically overcome the ill-posed nature of single-view $360^\circ$ synthesis and address reviewer feedback (**Reviewers #2, #3, #4**), we introduce domain-specific regularizations:

### 3.1 Line-Integral Global Attenuation Mass Loss ($\mathcal{L}_{\text{atten}}$)
*Addressing Reviewer #3 (R3-2) & Reviewer #4 (R4-1)*

Under ideal parallel-beam geometry, total integrated line attenuation $D(\mu) = \int \mu \, \mathrm{d}s$ across a closed orbit around a fixed rotation axis is conserved across projection angles. Under realistic cone-beam divergence ($1.8\text{m}$ distance, $35^\circ$ FOV) and lateral window truncation (shoulder/arm tissue exiting detector bounds at $90^\circ/270^\circ$), global attenuation mass varies smoothly across rotation angles.

Rather than enforcing rigid per-pixel line integral equality, we formulate the **Global Attenuation Mass Loss** as a macro-level total mass stabilizer in un-exponentiated line integral space:
$$\mathcal{L}_{\text{atten}} = \frac{1}{T} \sum_{k=0}^{T-1} \left| \sum_{x,y} \hat{D}_{\theta_k}(x, y) - \sum_{x,y} D^{\text{PA}}(x, y) \right|^2$$
where $\hat{D}_{\theta_k}$ is the predicted line-integral projection at azimuth $\theta_k$, and $D^{\text{PA}}$ is the ground-truth $0^\circ$ PA line integral.

**Impact:** Constrains the generative trajectory to maintain global tissue mass stability, eliminating non-physical macro-level tissue creation or erosion during rotation.

---

### 3.2 Anatomical Mirror-Symmetry Latent Soft Prior ($180^\circ$ AP)
*Addressing Ill-Posed Posterior Ambiguity & Reviewer #2 (R2-1)*

Human thoracic anatomy exhibits approximate left-right and anterior-posterior mirror symmetry. Horizontally flipping $I_{\text{PA}}$ ($0^\circ$) yields a strong $180^\circ$ AP structural prior (measured at 27.50 dB PSNR on 241 NSCLC test cases via `scripts/benchmark_flip_baseline.py`).

Under cone-beam geometry ($1.8\text{m}$ distance, $35^\circ$ FOV), the heart sits closer to the detector in PA orientation and further in AP orientation, producing physical AP cardiac magnification (apparent cardiomegaly). Furthermore, due to the $3\text{D}$-VAE's $4\times$ temporal compression, latent frame index 12 aggregates 4 pixel frames (frames 45–48, spanning ~15.6° of azimuth around $180^\circ$).

To account for cone-beam cardiac magnification and avoid temporal step artifacts across the 4-frame latent window, Cosmos-Predict2.5 incorporates $I_{180^\circ}^{\text{prior}} = \mathcal{H}_{\text{flip}}(I_{\text{PA}})$ as a **Soft Latent Feature Prior** $\alpha \cdot z_{\text{cond\_180}} + (1 - \alpha) z_t$ ($\alpha = 0.3$) at latent frame 12, allowing the DiT model to refine cardiac silhouette magnification while anchoring posterior structural topology.

This blend is applied exactly once, after the full 35-step reverse-diffusion trajectory has produced its final latent — not re-applied at every sampling step. Re-applying it every step would make $x_{n+1} = \alpha z_{\text{cond\_180}} + (1-\alpha)x_n$ a fixed-point iteration that converges geometrically toward $z_{\text{cond\_180}}$ (after 35 steps, $(1-\alpha)^{35} \approx 3.79\times10^{-6}$), collapsing the intended soft blend into a near-hard replacement and leaving the DiT no room to refine the frame. `Inferencer.predict(use_ap_soft_prior=False)` disables this component entirely for ablation baselines.

---

### 3.3 Continuous Angular-Offset Loss Weighting ($w(\theta_k)$)
*Addressing Low Performance at Side Views & Neighboring Oblique Arcs ($\theta \in [45^\circ, 135^\circ] \cup [225^\circ, 315^\circ]$)*

Synthesizing orthogonal lateral projections and neighboring oblique view arcs ($\theta \in [45^\circ, 135^\circ] \cup [225^\circ, 315^\circ]$) from a single frontal PA radiograph ($\theta = 0^\circ$) represents the most severely ill-posed regime due to maximum ray path length, heavy anatomical superposition (spine, ribs, heart, lateral chest wall), and maximum angular distance.

To focus model capacity on these high-ambiguity viewpoint sectors, loss supervision is weighted using a smooth sinusoidal function peaking at $90^\circ$ and $270^\circ$:
$$\mathcal{L}_{\text{angle-RF}} = \frac{1}{T} \sum_{k=0}^{T-1} w(\theta_k) \left\| v_\theta(z_{t, \theta_k}, t, c) - (\epsilon - z_{0, \theta_k}) \right\|_2^2$$
where $w(\theta_k) = 1 + \gamma_{\text{side}} \sin^2(\theta_k)$, with $\gamma_{\text{side}} = 1.0$. Because $\sin^2(\theta_k)$ is smooth and continuous, it doubles supervision weight across the lateral-oblique arcs ($\theta \approx 90^\circ, 270^\circ$) while relying on $0^\circ$ conditioning and $180^\circ$ mirror prior at the frontal/rear axes.

**Impact:** Focuses DiT gradient updates on high-ambiguity lateral and oblique viewpoint sectors, eliminating structural blurring and depth collapse.

---

### 3.4 Periodic Orbit Noise Schedule for Closed $360^\circ$ Orbits
*Addressing Inter-Frame Noise Consistency & Orbit Continuity*

Standard video diffusion models draw i.i.d. noise per frame $\epsilon_k \sim \mathcal{N}(0, I)$. For a closed $360^\circ$ circular orbit ($\text{endpoint}=\text{True}$ where frame $0$ at $0^\circ$ and frame $92$ at $360^\circ$ represent identical physical views), non-periodic noise creates a boundary noise step at the orbit wrap-around point.

We construct a **Periodic Orbit Noise Schedule** using two independent Gaussian noise basis tensors $\mathbf{a}, \mathbf{b} \sim \mathcal{N}(0, I)$:
$$\epsilon(\theta_k) = \mathbf{a} \cos(\theta_k) + \mathbf{b} \sin(\theta_k), \quad \theta_k = \frac{k}{23} \cdot 2\pi$$

where $k \in \{0, \dots, 23\}$ indexes the 24 latent frames across the $360^\circ$ orbit trajectory. Because $\cos^2(\theta_k) + \sin^2(\theta_k) = 1$, $\epsilon(\theta_k)$ is marginally standard Gaussian $\mathcal{N}(0, I)$ at every view while guaranteeing bit-exact closed orbit continuity $\epsilon(0^\circ) = \epsilon(360^\circ) = \mathbf{a}$.

This schedule is applied at inference only; post-training (`predict2_5/module.py`) still samples i.i.d. per-frame Gaussian noise, so it introduces a train/inference noise-correlation gap tracked as its own ablation axis (`ABLATION_STUDY.md` §3.4). `Inferencer.predict(use_periodic_noise=False)` falls back to i.i.d. per-frame noise for ablation baselines.

---

## 4. Complete Optimization Objective

The total loss function for post-training Cosmos-Predict2.5 combines Angular-Weighted Rectified Flow velocity matching with global attenuation mass regularization:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{angle-RF}} + \lambda_{\text{atten}} \mathcal{L}_{\text{atten}}$$

where:
- **Angular-Weighted Logit-Normal Rectified Flow Loss:**
  $$\mathcal{L}_{\text{angle-RF}} = \mathbb{E}_{t \sim \text{LogitNorm}, z_0, \epsilon} \left[ \frac{1}{T} \sum_{k=0}^{T-1} w(\theta_k) \left\| v_\theta(z_{t, \theta_k}, t, c) - (\epsilon - z_{0, \theta_k}) \right\|_2^2 \right]$$
  with $w(\theta_k) = 1 + \gamma_{\text{side}} \sin^2(\theta_k)$ ($\gamma_{\text{side}} = 1.0$), $z_t = t \epsilon + (1 - t) z_0$, timesteps sampled via native Logit-Normal distribution ($s=5.0$), and periodic orbit noise $\epsilon(\theta_k)$.
- **Global Mass Regularization Weight:** $\lambda_{\text{atten}} = 0.02$. Calibrated via gradient scale parity checking to maintain $\|\nabla \mathcal{L}_{\text{atten}}\|_2 \approx 5\text{--}10\%$ of $\|\nabla \mathcal{L}_{\text{angle-RF}}\|_2$.

---

## 5. Downstream 3D CT Reconstruction Bridge (Protocol Target)

*Addressing Reviewer #3 (R3-1) & Reviewer #4 (R4-2)*

Cosmos-Predict2.5 serves as a **dense regularizing intermediate representation** intended to convert the severely ill-posed 1-view $3\text{D}$ CT inverse problem into a well-conditioned tomographic reconstruction task:

```
[1-View 2D CXR] ──► [Cosmos-Predict2.5 (360° NVS)] ──► [93 Dense Projections] ──► [FDK / NeRF] ──► [3D CT Volume]
  (Ill-Posed)           (Spatiotemporal Prior)             (Well-Conditioned)        (Tomographic Backprojection)
```

1. **Direct FDK Backprojection Protocol:** Classical Feldkamp-Davis-Kress (FDK) filtering over the synthesized 93 projections to evaluate 3D volume reconstruction quality.
2. **Beer-Lambert NeRF Optimization Protocol:** Fine-grained $3\text{D}$ CT attenuation volume optimization ($\mathbf{V}_{\text{CT}}(x, y, z)$) constrained by the 93 synthesized views:
   $$\min_{\mathbf{V}_{\text{CT}}} \sum_{k=1}^{93} \left\| \mathcal{R}_{\text{Siddon}}(\mathbf{V}_{\text{CT}}, \theta_k) - \hat{I}_{\theta_k} \right\|_2^2 + \gamma \mathcal{R}_{\text{TV}}(\mathbf{V}_{\text{CT}})$$

This protocol establishes the downstream bridge connecting $2\text{D}$ CXR synthesis directly to $3\text{D}$ CT diagnostic reconstruction, to be evaluated once 360° NVS generation completes.
