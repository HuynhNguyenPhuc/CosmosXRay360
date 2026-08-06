# CosmosXRay360 - Comprehensive Literature Reference Manual

This document provides a highly detailed literature review and academic breakdown of the baseline architectures downloaded in `arxiv_papers/`. It maps their core scientific equations, network topologies, coordinate parameterizations, and physical modeling limits to serve as an authoritative reference for our comparative evaluation.

---

```
                                 NVS BASELINE METHODOLOGY TAXONOMY
                                                 │
         ┌───────────────────────────────────────┴───────────────────────────────────────┐
         ▼                                                                               ▼
Reconstruction-based (2D → 3D → 2D)                                             Direct Synthesis-based (2D → 2D)
 * SNAF: Continuous MLP, Hash Grid                                               * SV-DRR: Pose-conditioned DiT
 * MedNeRF: Generative NeRF, Latent Optimization                                 * Cosmos-Predict2.5: World Model
 * PixelNeRF: Pixel-aligned Feed-forward Features
 * Dx2CT: Slice-by-slice 3D Transformer, SPADE
```

---

## 1. SNAF: Neural Attenuation Fields for Sparse-View CBCT Reconstruction
* **Paper ID:** arXiv:2209.14540 (Accepted at MICCAI 2022 Oral)
* **Authors:** Ruyi Zha, Yanhao Zhang, Hongdong Li
* **Method Category:** Reconstruction-based (Implicit Continuous Representation)

### A. Core Innovation & Topology
SNAF maps continuous spatial 3D coordinates $(x, y, z)$ directly to a view-independent linear attenuation coefficient $\mu \in \mathbb{R}^+$. Unlike standard NeRFs, it eliminates view-dependency (color/radiance) since medical X-ray attenuation is isotropic and depends solely on local physical density. To resolve NeRF's spectral bias (difficulty learning high-frequency structures), coordinates are encoded via a multi-resolution spatial Hash Grid prior to MLP query:
$$\mathbf{e}(x, y, z) = \text{HashGridEncoder}(x, y, z)$$
$$\mu(x, y, z) = \text{MLP}(\mathbf{e}(x, y, z))$$

### B. Mathematical Formulation & Ray Physics
The transmissive intensity of an incident X-ray projection ray $r(t) = \mathbf{o} + t\mathbf{d}$ traveling from source position $\mathbf{o}$ along direction $\mathbf{d}$ is parameterized by the forward Beer-Lambert Law line integral:
$$I(r) = I_0 \exp \left( -\int_{t_{n}}^{t_{f}} \mu(r(t)) \, dt \right)$$
In discretized form (Siddon's or trilinear ray-marching sampling), the estimated projection is computed over $M$ samples:
$$\hat{I}(r) = I_0 \exp \left( -\sum_{i=1}^{M} \mu(r(t_i)) \delta_i \right)$$
where $\delta_i = t_{i+1} - t_i$ represents the step distance. The model optimizes a combination of Mean Squared Error (MSE) and Total Variation (TV) regularization to impose spatial smoothness:
$$\mathcal{L} = \sum_{r \in \mathcal{R}} \| I(r) - \hat{I}(r) \|^2 + \lambda_{\text{TV}} \sum_{v \in \mathcal{V}} \| \nabla \mu(v) \|_2$$

### C. Physical and Algorithmic Limits
1. **Depth Ambiguity Collapse (1-View Limit):** Under a single-view ($1$-view) constraint, there are no intersecting line integrals to resolve the depth coordinate. Consequently, the gradient cannot localize attenuation along the ray, causing the predicted density $\mu$ to stretch infinitely along the projection vector (depth-collapse).
2. **Computational Overhead:** Requires dense querying of coordinates at high spatial resolution during training, making direct real-time volume optimization computationally intensive.

---

## 2. SV-DRR: High-Fidelity Novel View X-Ray Synthesis Using Diffusion Model
* **Paper ID:** arXiv:2507.05148 (Accepted at MICCAI 2025)
* **Authors:** Chun Xie, Yuichi Yoshii, Itaru Kitahara
* **Method Category:** Direct Synthesis-based (Generative Diffusion)

### A. Core Innovation & Topology
SV-DRR completely bypasses the 3D geometry building phase. It frames novel view synthesis as a conditional image-to-image translation task. The architecture utilizes a 2D Diffusion Transformer (DiT) conditioned on both a single source 2D projection and a target camera pose vector to synthesize high-resolution novel views.

### B. Mathematical Formulation
Given a source X-ray image $x_{\text{cond}}$ and a target view camera pose $\mathbf{p} = [elevation, azimuth] \in \mathbb{R}^2$ represented on a spherical coordinate grid, a pre-trained feature extractor $\mathcal{E}$ encodes $x_{\text{cond}}$. The DiT denoiser $\epsilon_\theta$ predicts added noise $\epsilon$ at diffusion timestep $t$ in latent space $z_t$:
$$\mathcal{L}_{\text{SV-DRR}} = \mathbb{E}_{z, \epsilon, t, \mathbf{p}} \left[ \| \epsilon - \epsilon_\theta(z_t, t, \mathcal{E}(x_{\text{cond}}), \mathbf{p}) \|^2 \right]$$
The pose vector $\mathbf{p}$ is mapped sinusoidally and injected directly into the cross-attention blocks of the DiT to steer structural rotation:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) V$$
where $Q$ is computed from spatial features, and $K, V$ are projected from the combined condition vector $[\mathcal{E}(x_{\text{cond}}), \mathbf{p}]$.

### C. Physical and Algorithmic Limits
1. **Geometric Inconsistency (No 3D Prior):** Since there is no explicit 3D volumetric bottleneck or ray-marching layer, the model does not enforce physical multi-view geometry. Over a continuous $360^\circ$ rotation, anatomical landmarks (e.g., ribs, clavicles, or mediastinal edges) can "morph" or shift unnaturally.
2. **Deterministic Bias:** The latent diffusion process can introduce stochastic hallucinations, sometimes altering micro-structures or pathologies (e.g., nodule shapes) across different generated angles.

---

## 3. MedNeRF: Medical Neural Radiance Fields for 3D-Aware Reconstructions
* **Paper ID:** arXiv:2202.01020 (Accepted at IEEE EMBC 2022)
* **Authors:** Abril Corona-Figueroa, Jonathan Frawley, Sam Bond-Taylor, Hubert P. H. Shum, Chris G. Willcocks
* **Method Category:** Reconstruction-based (Generative NeRF with GAN Prior)

### A. Core Innovation & Topology
MedNeRF bridges GANs and Neural Radiance Fields. A NeRF-based generator $G$ receives a latent code $z$ and spatial query rays to output densities, while a patch-based discriminator $D$ evaluates generated 2D projections against real datasets. This allows the model to learn 3D medical shapes from unpaired, multi-patient 2D projection datasets.

### B. Mathematical Formulation
The generator $G_{\theta}$ takes a latent code $z \sim p_z$ and a ray bundle $R = \{\mathbf{o} + t\mathbf{d}\}$ representing specific projection geometries to output a 2D projected patch $\hat{P}$:
$$\hat{P} = \text{Render}(G_{\theta}(z, R))$$
The training objective uses an adversarial minimax loss:
$$\min_{\theta} \max_{\phi} \left( \mathbb{E}_{I \sim p_{\text{data}}} [\log D_{\phi}(I)] + \mathbb{E}_{z \sim p_z, R} [\log(1 - D_{\phi}(\text{Render}(G_{\theta}(z, R))))] \right)$$
At inference/reconstruction time (reconstructing a specific patient from a single reference projection $I_{\text{target}}$), the generator weights $\theta$ are frozen. The latent code $z$ and generator layers are optimized iteratively via backpropagation to minimize Mean Squared Error and LPIPS perceptual loss:
$$z^* = \arg\min_{z} \left[ \lambda_1 \mathcal{L}_{\text{LPIPS}}(G(z, R_{\text{frontal}}), I_{\text{target}}) + \lambda_2 \| G(z, R_{\text{frontal}}) - I_{\text{target}} \|^2 \right]$$

### C. Physical and Algorithmic Limits
1. **High Computational Latency:** Optimizing $z$ on-the-fly at inference time is extremely slow (requiring $>22\text{s}$ per volume), making real-time clinical deployment impossible.
2. **Overfitting & Blurring:** Because the generator is forced to fit a single target projection starting from a generic learned prior, it is prone to local minima. This often leads to blurry, low-frequency reconstructions on unseen out-of-distribution (OOD) cases.

---

## 4. PixelNeRF: Neural Radiance Fields from One or Few Images
* **Paper ID:** arXiv:2012.02190 (Accepted at CVPR 2021)
* **Authors:** Alex Yu, Vickie Ye, Matthew Tancik, Angjoo Kanazawa
* **Method Category:** Reconstruction-based (Feed-forward Pixel-aligned NeRF)

### A. Core Innovation & Topology
PixelNeRF enables feed-forward, zero-shot generalization in Neural Radiance Fields. Instead of optimizing a scene-specific model, it extracts spatial CNN feature maps from input images and conditions the NeRF queries directly on pixel-aligned features.

### B. Mathematical Formulation
For any 3D query point $x$ along a projection ray, $x$ is projected onto the input view plane $I$ using the camera projection matrix $\pi$:
$$u = \pi(x)$$
A ResNet-34 backbone extracts a localized feature map $W(I)$, and the pixel-aligned feature vector is fetched via bilinear interpolation:
$$\mathbf{f}(x) = \text{Interpolate}(W(I), u)$$
The radiance and density MLP receives the coordinate, direction $d$, and the local feature $\mathbf{f}(x)$ directly:
$$[\sigma, \mathbf{c}] = \text{MLP}(x, d, \mathbf{f}(x))$$
During training, the network is trained end-to-end across multiple scenes using standard reconstruction losses without requiring test-time optimization.

### C. Physical and Algorithmic Limits
1. **Reflection vs. Transmission Modeling:** PixelNeRF is designed for opaque, reflecting surfaces under perspective projection. It struggles with transmissive rays (X-rays), where density accumulates along the entire ray path rather than terminating at a solid boundary.
2. **Depth Ambiguity:** Bilinearly interpolating features along a projection line means points at different depths project to the same 2D pixel, inheriting identical feature vectors $\mathbf{f}(x)$. Without explicit depth constraints, the network struggles to resolve the depth of internal organs in transmissive projections.

---

## 5. Dx2CT: Diffusion Model for 3D CT Reconstruction from Mono/Bi-planar X-Rays
* **Paper ID:** arXiv:2409.08850 (Accepted at ICASSP 2025)
* **Authors:** (Anonymous pre-print archive)
* **Method Category:** Reconstruction-based (Slice-by-slice 3D Transformer & Diffusion)

### A. Core Innovation & Topology
Dx2CT proposes a slice-by-slice generative diffusion paradigm. Instead of generating a full 3D CT volume at once (which causes VRAM explosion), it utilizes a 3D Position-aware Query Transformer (3DPQT) to look up features from 2D bi-planar X-rays and feeds them via Spatially-Adaptive Normalization (SPADE) to a 2D axial denoising U-Net.

### B. Mathematical Formulation
For a target axial CT slice at height $Z_i$, a continuous coordinate grid $C = [x, y, Z_i]$ is cross-attended against the multi-scale CNN feature maps extracted from the Frontal (PA) and Lateral X-rays:
$$\text{Condition} = \text{Transformer}(\text{PositionEncoding}(C), \text{CNN}(I_{\text{PA}}, I_{\text{LAT}}))$$
The denoising U-Net $\mathcal{D}$ then predicts the noise $\epsilon$ added to the slice $x_t$ at timestep $t$ using the SPADE mapping:
$$\epsilon_\theta = \mathcal{D}(x_t, t, \text{SPADE}(\text{Condition}))$$
The SPADE layer normalizes the U-Net features $h$ using learnable scaling ($\gamma$) and bias ($\beta$) derived from the 3DPQT conditioning map:
$$\text{SPADE}(h, \text{Condition}) = \text{InstanceNorm}(h) \cdot (1 + \gamma(\text{Condition})) + \beta(\text{Condition})$$

### C. Physical and Algorithmic Limits
1. **Inter-Slice Inconsistency (Staircase Artifacts):** Because CT slices are generated independently along the Z-axis, there are no explicit inter-slice physical constraints. This often results in high-frequency "staircase" artifacts along the coronal and sagittal reconstruction views.
2. **Computational Scale:** Querying 3D cross-attention maps for every voxel sequence coordinate is computationally intensive, requiring smart downsampling strategies to fit within standard hardware bounds during training.
