# nanoCosmos-Style Batch Walkthrough: `WALKTHROUGH.md`

This document details exactly what happens when running a single inference or training batch through the adapted **Cosmos-Predict2.5** framework. It traces execution chronologically from input loading to 360-degree novel-view projection.

---

## 🏃 Chronological Execution Flow

```
+---------------------------------------------------------------------------------+
|                                 INPUT LOADING                                   |
|   1. Frontal Chest X-Ray (CXR) loaded and normalized.                           |
|   2. Image repeated to T=93 frames representing the full rotation.              |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                                 VAE ENCODING                                    |
|   3. 93-frame input compressed into spatiotemporal latents via Wan-VAE.        |
|   4. Latents shaped to [B, T_latent, C, H_latent, W_latent].                    |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                           FRAME-TOKEN REPLACEMENT                               |
|   5. Frontal index t_frontal is located.                                        |
|   6. Latents at t_frontal replaced with encoded patient-specific CXR.           |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                            DIT DENOISING MATCHING                               |
|   7. DiT Backbone processes spatiotemporal latents with 3D RoPE.                |
|   8. Guided by rotation text prompts ("360 rotation around chest").             |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                                 VAE DECODING                                    |
|   9. Spatiotemporal latents decoded back to pixel space.                        |
|  10. Decoded sequence yields a globally coherent 93-view projection sequence.   |
+---------------------------------------------------------------------------------+
```

---

## 🔍 Code Walkthrough with Key Files

### Step 1: Input Setup & Image Parsing
* **File**: `predict2_5/utils/` modules.
The input frontal chest X-ray $x^{(0)} \in \mathbb{R}^{H \times W}$ is resized to $256 \times 256$, normalized to $[-1, 1]$, and duplicated $93$ times along the temporal axis to instantiate the sequential rotation canvas:
$$X_{\text{raw}} \in \mathbb{R}^{93 \times 1 \times 256 \times 256}$$

### Step 2: VAE Spatiotemporal Encoding
* **File**: `predict2_5/inferencer.py` (Core Model Pipeline).
The 93-frame image tensor is passed through the frozen Wan 2.1/2.2 VAE encoder, compressing the spatial and temporal dimensions into the latent space:
$$z = \text{Encoder}(X_{\text{raw}}) \in \mathbb{R}^{T_{\text{lat}} \times C \times H_{\text{lat}} \times W_{\text{lat}}}$$

### Step 3: Frame-Token Replacement (Domain-Specific Adaptation)
* **File**: `predict2_5/inferencer.py` (Denoising loop override).
Instead of naive prompts, we enforce strict physical alignment. We compute the VAE-encoded latents of the actual high-fidelity frontal CXR ($z_{\text{frontal}}$). We then substitute the latent tokens at the exact temporal anchor index corresponding to $0^\circ$ (index $t_{\text{frontal}}$):
$$z_{t_{\text{frontal}}} \leftarrow z_{\text{frontal}}$$
During rectified-flow denoising, this replacement acts as a *hard physical constraint* forcing the diffusion transformer (DiT) to align its generated views directly with the patient's actual chest structure.

### Step 4: DiT Denoising & Trajectory Generation
* **File**: `cosmos-predict2.5/` (WFM Backbone execution).
The DiT backbone takes the noisy latent sequence $z_t$, the timestep embedding, the rotation text conditioning prompt (handled via `predict2_5/text_encoder.py`), and the 3D Rotary Positional Embeddings (RoPE). It predicts the velocity field $v_\theta$ to transport noise to data along linear trajectories via rectified flow:
$$v_{\text{pred}} = \text{DiT}(z_t, t, \mathcal{C}_{\text{prompt}}, \text{RoPE})$$

### Step 5: VAE Latent Decoding
* **File**: `predict2_5/inferencer.py` (Finalization).
After completing the denoising steps, the resulting noise-free spatiotemporal latents $\hat{z}$ are passed to the frozen Wan VAE Decoder, reconstructing the continuous, structurally aligned 360-degree rotation projection sequence:
$$\hat{X} = \text{Decoder}(\hat{z}) \in \mathbb{R}^{93 \times 1 \times 256 \times 256}$$
This sequence is ready to be parsed into 3D tomographic reconstruction engines via FDK backprojection (e.g. using `predict2_5/dvr/` raymarcher tools) or NeRF coordinate MLP optimization.
