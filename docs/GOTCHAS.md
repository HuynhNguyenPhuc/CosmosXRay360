# GOTCHAS

This document outlines the **non-obvious, silent physical and mathematical failure modes** where the novel-view synthesis (NVS) or CT reconstruction models might compile and run without throwing errors, but are physically or scientifically incorrect.

---

## 💥 Index of Silent Failures

| Symptom | Where it Happens | Recommended Remediation |
|:---|:---|:---|
| 1. Direct L2 loss between linear attenuation and transmissive intensity. | Downstream NeRF Optimization | Apply the **Beer-Lambert exponential correction** directly inside the loss term. |
| 2. Downstream FDK reconstruction suffers off-plane blurring. | Downstream FDK Tomography | Acknowledge the **Tuy-Smith sufficiency condition violation** for single circular orbits. |
| 3. Explicit point-clouds (Gaussian Splatting) exhibit extreme depth extrusion. | Single-View Splatting Baselines | Restrict explicit 3DGS models to multi-view evaluations; avoid single-view initializations. |
| 4. DRR models experience contrast/noise degradation on real clinical X-rays. | Multi-Center Real CXR Inference | Implement **Unsupervised Domain Adaptation (UDA)** style matching in the latent space. |

---

## 1. Beer-Lambert Dimensionality Mismatch in Downstream Optimization

* **The Symptom**: The downstream NeRF coordinate MLP runs successfully and outputs a 3D chest volume, but the reconstructed density values are physically distorted, and the simulated projections have incorrect contrast.
* **Where it Happens**: Differentiable reconstruction loss formulations.
* **Why it Happens**: Differentiable projectors (like DiffDRR) compute the continuous line-integral of spatial attenuation coefficients $\mu$ through a voxel grid:
  $$D(\mu) = \int \mu(s) ds$$
  which is linear in tissue density. However, real-world radiographic projections (and the views synthesized by our generative Cosmos-Predict2.5) operate in the normalized transmissive intensity space:
  $$I \propto I_0 e^{-D(\mu)}$$
  Taking a direct L2 loss between the linear attenuation $D(\mu)$ and the non-linear transmissive intensity $I$ is a fundamental physics violation.
* **Remediation**: Correct the dimensionality mismatch. You must either:
  1. Map the rendered attenuation to intensity space before computing the loss:
     $$\mathcal{L}_{\text{NeRF}} = \sum_{i=1}^N \left\| I_0 \exp\left(-\mathcal{R}_{\text{diff}} (\hat{V}_{\text{NeRF}})\right) - \hat{P}_{\theta_i} \right\|_2^2$$
  2. Or log-transform the generated transmissive intensity to attenuation space:
     $$\mathcal{L}_{\text{NeRF}} = \sum_{i=1}^N \left\| \mathcal{R}_{\text{diff}} (\hat{V}_{\text{NeRF}}) - \left( -\ln(\hat{P}_{\theta_i}) \right) \right\|_2^2$$

* **Addendum (2026-08-05) — which space is `datasets/pre_rendered/*/views/*.png` actually in?** This
  entry's framing (real X-rays and Cosmos-Predict2.5's generated frames live in transmissive-intensity
  space $I$) is correct for those two cases, but it does **not** describe this project's own baseline
  ground truth. `renderers/diffdrr/renderer.py`'s `render()` applies only `minimized`/`normalized`/
  `standardized` post-processing — all purely affine (min-max / z-score) — to DiffDRR's raw output, which
  its own docstring states is a discretized line integral, `sum(density) * step_size`, with no `exp()`
  step anywhere in the pipeline. So `views/*.png`, the file every one of the 6 baselines is scored
  against in `baselines/evaluate.py`, is (a rescaled) $D(\mu)$, **not** $I$. Applying Beer-Lambert to a
  baseline's prediction before comparing it to `views/*.png` is therefore only correct if the prediction
  is being *gradient-fit directly* against that same target with the transform inside the loss (self-
  correcting regardless of the transform's physical label — this is why NAF's per-scan `fit_density_field`
  is safe using `apply_beer_lambert_correction`); it is a real bug for a prediction produced independently
  of that target (e.g. a generative model's output), where the reduction must match the ground truth's
  actual — non-exponentiated — convention on its own. This was caught as a live regression in Dx2CT's
  `infer_multi_views`; see `docs/baselines/dx2ct/LOG.md`'s 2026-08-05 correction entry for the full trace.

---

## 2. Cone-Beam Blurring Off-Plane in FDK Backprojection

* **The Symptom**: Reconstructing a 3D chest CT volume from the generated 360-degree projections using standard FDK filtered backprojection results in a high-fidelity central plane, but structures near the top and bottom of the lung cavity exhibit severe blurring and geometric "stretching."
* **Where it Happens**: Downstream classical tomographic reconstruction.
* **Why it Happens**: **Tuy-Smith Sufficiency Condition.** In tomographic reconstruction, a 3D Radon transform can only be exactly inverted if every plane intersecting the object also intersects the source trajectory. A single circular orbit of synthesized views (parallel to the transverse mid-plane) violates the Tuy-Smith sufficiency condition. Any planes off the mid-plane do not intersect the source trajectory, leading to systematic, mathematically unresolvable cone-beam artifacts off-axis.
* **Remediation**: Clarify in the manuscript that FDK backprojection serves as a fast baseline, and emphasize that optimization-based coordinate MLPs (like NeRFs or Gaussian Splatting) successfully regularize and suppress these off-axis cone-beam artifacts by incorporating structural continuity priors.

---

## 3. Depth-Plane Extrusion in Explicit Gaussian Splatting

* **The Symptom**: Initializing `X-Gaussian` or `$R^2$-Gaussian` point clouds on a single frontal projection generates plausible frontal views, but rotation shows "floater" points and anatomical structures elongated like long cylinders along the depth axis.
* **Where it Happens**: Explicit 3D Gaussian Splatting baselines under single-view constraints.
* **Why it Happens**: Explicit point-cloud models parameterize density explicitly over discrete locations. When initialized from a single projection, the model has no depth cues, leading to a mathematically flat distribution. The optimization-based rendering resolves depth ambiguities by elongating the Gaussian points along the projection rays ("extrusion").
* **Remediation**: Explicitly document this behavior as a structural limitation of explicit representation networks under single-view constraints, contrasting them with our Adapted World Foundation Model, which utilizes physical video pretraining on camera panning to infer correct depth-plane continuity.

---

## 4. DRR-to-CXR Domain Degradation

* **The Symptom**: The model achieves excellent PSNR/SSIM on synthetic DiffDRR test sets, but when evaluating on real-world clinical chest X-rays (like `VinDr-CXR`), the generated rotations display contrast artifacts, noise blocks, and blurred anatomical structures.
* **Where it Happens**: Model evaluation and deployment.
* **Why it Happens**: Synthetic DRRs represent mathematically ideal transmissive paths without X-ray forward scatter, Poisson/quantum noise, focal spot blurring, or grid attenuation. Clinical chest X-rays contain all of these physical phenomena, combined with proprietary Look-Up Tables (LUTs) applied by acquisition machines (e.g., Agfa, Siemens, GE) to adjust local contrast.
* **Remediation**: Implement **Unsupervised Domain Adaptation (UDA)** using Maximum Mean Discrepancy (MMD) or adversarial latent matching. This aligns the latent feature distributions of real-world clinical projections with synthetic DiffDRR projections, allowing the DiT to generalize to real clinical noise and contrast variations.
