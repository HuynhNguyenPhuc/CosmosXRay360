# NAF: Neural Attenuation Fields for Sparse-View CBCT Reconstruction

**Paper:** *NAF: Neural Attenuation Fields for Sparse-View CBCT Reconstruction*  
**Venue/Publication:** MICCAI 2022 / arXiv:2209.14540  
**arXiv Cache:** `arxiv_papers/SNAF_2209.14540.pdf`  

---

## 1. Executive Summary & Paradigm

NAF (Neural Attenuation Fields) belongs to the **Reconstruction-based Paradigm (Implicit Attenuation Fields)**. It parameterizes a 3D Continuous Attenuation Field as a spatial coordinate network consisting of a multi-resolution Instant-NGP Hash-grid encoder and a lightweight density MLP.

Because NAF contains no image-conditioning inputs, it operates as a per-scan overfitting framework: the coordinate network is fit directly to a specific scan's available 2D projections via gradient descent before rendering novel-view projections.

---

## 2. Mathematical Formulation & Architecture

### A. Hash-Grid Coordinate Attenuation Field
- For any 3D spatial coordinate $\mathbf{x} = (x, y, z) \in [-b, b]^3$ (where bound $b = 0.3$):
  $$\mu(\mathbf{x}) = \text{DensityMLP}\Big( \text{HashEncoder}(\mathbf{x}) \Big)$$
  where $\mu(\mathbf{x}) \ge 0$ represents the 3D local linear attenuation coefficient.

### B. Perspective Ray-Marching & Beer-Lambert Physics
- For a camera ray $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$ cast from camera origin $\mathbf{o}$ through pixel $(u, v)$ at azimuth $\theta$:
  1. Stratified sampling generates $N$ 3D points $\mathbf{x}_i \in [-b, b]^3$ along the ray between $t_{\text{near}}$ and $t_{\text{far}}$.
  2. Numerical quadrature computes total attenuation line-integral:
     $$A(\mathbf{r}) = \sum_{i=1}^N \mu(\mathbf{x}_i) \Delta s_i$$
  3. Transmissive intensity $I(\mathbf{r})$ is computed via Beer-Lambert exponentiation:
     $$I(\mathbf{r}) = I_0 \exp\big(-A(\mathbf{r})\big)$$

### C. Per-Scan Fitting Objective (`fit_density_field`)
- Given a reference conditioning radiograph $I_{\text{target}}$ at camera pose $\theta_0$:
  $$\min_{\theta_{\text{NAF}}} \left\| I_0 \exp\left( -\int_{t_n}^{t_f} \mu_{\theta_{\text{NAF}}}(\mathbf{r}(s)) \, ds \right) - I_{\text{target}} \right\|_2^2$$
  optimized using Adam ($\text{lr} = 10^{-3}$) for $200$ iterations per patient scan.

---

## 3. Reported Training & Evaluation Setup

- **Encoder:** Multi-resolution Hash Grid (`HashEncoder`, Instant-NGP style).
- **Coordinate Bound:** $b = 0.3$ (bounding volume $[-0.3, 0.3]^3$).
- **Fitting Iterations:** 200 steps per scan (shows clean elbow in loss convergence curve).
- **Ray-Marching Geometry:** Matches physical camera setup (source-to-detector distance, FOV, near/far bounds).

---

## 4. Stated Limitations

1. **Ill-Posed Single-View Problem:** Under a single frontal view constraint ($\le 1$ view), the coordinate field suffers from severe depth ambiguity. Without multi-view constraints or 3D priors, the attenuation field collapses along the primary beam axis, rendering cloud-like artifacts at oblique novel views.
2. **No Population Generalization:** NAF cannot infer novel views feed-forward for an unseen patient without executing per-scan optimization steps.
