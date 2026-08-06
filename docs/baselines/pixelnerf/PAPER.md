# PixelNeRF: Neural Radiance Fields from One or Few Images

**Paper:** *pixelNeRF: Neural Radiance Fields from One or Few Images*  
**Venue/Publication:** CVPR 2021 / arXiv:2012.02190  
**arXiv Cache:** `arxiv_papers/PixelNeRF_2012.02190.pdf`  

---

## 1. Executive Summary & Paradigm

PixelNeRF belongs to the **Reconstruction-based Paradigm (Generalizable Feed-Forward NeRF)**. Unlike standard NeRFs (which optimize a coordinate MLP per scene) or generative NeRFs (which sample latent spaces), PixelNeRF conditions a continuous neural radiance field directly on spatial feature maps extracted from one or few input images.

This allows PixelNeRF to perform generalizable, feed-forward novel view synthesis from a single 2D radiograph without requiring per-patient test-time optimization.

---

## 2. Mathematical Formulation & Architecture

### A. Fully-Convolutional Spatial Feature Extraction
- Given a single source image $I \in \mathbb{R}^{H \times W \times C}$, a fully-convolutional image encoder $E$ (ResNet34) extracts a spatial feature volume $W = E(I)$.

### B. Image-Conditioned Radiance Field Query
- For a 3D query point $\mathbf{x} \in \mathbb{R}^3$ along camera ray $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$:
  1. Project $\mathbf{x}$ onto the source image plane using intrinsic/extrinsic projection $\pi(\mathbf{x})$.
  2. Bilinearly sample local image feature $f_x = W(\pi(\mathbf{x}))$.
  3. Feed $\mathbf{x}$, viewing direction $\mathbf{d}$, and local feature $f_x$ into coarse and fine MLPs:
     $$(\sigma, c) = f_{\text{NeRF}}\Big( \gamma(\mathbf{x}), \gamma(\mathbf{d}), f_x \Big)$$

### C. Coarse & Fine Volume Rendering Loss
- Rays $\mathbf{r} \in \mathcal{R}$ are sampled across target views. Both coarse pass $C_{\text{coarse}}$ and fine pass $C_{\text{fine}}$ are rendered and supervised against ground-truth target image $C_{\text{target}}$:
  $$\mathcal{L}_{\text{total}} = \sum_{\mathbf{r} \in \mathcal{R}} \left( \| C_{\text{coarse}}(\mathbf{r}) - C_{\text{target}}(\mathbf{r}) \|_2^2 + \| C_{\text{fine}}(\mathbf{r}) - C_{\text{target}}(\mathbf{r}) \|_2^2 \right)$$

---

## 3. Reported Training & Evaluation Setup

- **Encoder:** ResNet34 pretrained on ImageNet (frozen or fine-tuned).
- **Ray Samples:** $N_{\text{coarse}} = 64$, $N_{\text{fine}} = 32$.
- **Camera Orbit:** Turntable orbit around world origin with $R = 4.0$, $\text{FOV} = 40^\circ$.
- **Optimizer:** Adam ($\text{lr} = 10^{-4}$).

---

## 4. Stated Limitations

1. **Blurry Extrapolations in Unobserved Regions:** Under single-view constraints, back-projected 2D feature rays create depth ambiguity ("smearing" or "extrusion" artifacts along ray direction).
2. **High Memory Overhead:** Rendering fine and coarse ray samples for high-resolution images ($256 \times 256$) consumes substantial VRAM during volume rendering.
