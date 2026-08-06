# MedNeRF: Medical Neural Radiance Fields for Reconstructing 3D-aware CT-Projections from a Single X-ray

**Paper:** *MedNeRF: Medical Neural Radiance Fields for Reconstructing 3D-aware CT-Projections from
a Single X-ray*
**Authors:** Abril Corona-Figueroa, Jonathan Frawley, Sam Bond-Taylor, Sarath Bethapudi, Hubert P. H.
Shum, Chris G. Willcocks (Durham University; County Durham and Darlington NHS Foundation Trust)
**Venue/Publication:** IEEE EMBC 2022 / arXiv:2202.01020
**arXiv Cache:** `arxiv_papers/MedNeRF_2202.01020.pdf`

> **Correction note (2026-08-05):** an earlier version of this file used a slightly wrong paper title
> ("...for Reconstruction of 3D-aware Medical Images" instead of "...for Reconstructing 3D-aware
> CT-Projections from a Single X-ray") and, more substantively, omitted the paper's actual stated
> contribution over GRAF entirely — a self-supervised auto-encoding discriminator and a
> weight-shared multi-head Data-Augmentation-Optimized-for-GAN (DAG) training scheme — while stating
> test-time-fitting loss coefficients ($\lambda_{\text{MSE}}=1.0$, $\lambda_z=0.01$) that don't match
> the paper's actual reported values. This project's own `fit_latent_and_weights` implementation
> (per `docs/LOSSES.md`/`LOG.md`) already matches the *real* paper's coefficients — it was the
> documentation, not the code, that was wrong. Rewritten directly from the downloaded PDF.

---

## 1. Executive Summary & Paradigm

MedNeRF adapts **GRAF** (Generative Radiance Fields — NeRF wrapped in a GAN, generator $G_\theta$
producing image patches judged by a discriminator $D_\phi$) to medical imaging, reconstructing a
patient's 3D-aware CT-projection representation from as little as a single 2D X-ray. Applying GRAF
directly to a small medical dataset fails for two reasons the paper identifies explicitly (§II-B):
the generator gets only an indirect gradient signal through the discriminator (a single convolutional
feedback poorly conveys DRR-specific structure), and adversarial training is brittle/mode-collapse-
prone on small datasets. **MedNeRF's actual contribution is a fix for both**, not a new radiance-field
formulation — the underlying coordinate-MLP NeRF math is inherited from GRAF/NeRF unchanged.

## 2. Mathematical Formulation & Architecture

### A. Radiance Field Representation (inherited from GRAF/NeRF)

A coordinate MLP maps 3D position $\mathbf{x}=(x,y,z)$ and viewing direction $\mathbf{d}=(\theta,\phi)$,
positionally encoded (Eq. 1, Fourier features $\gamma(p) = (\dots,\cos(2^j\pi p),\sin(2^j\pi p),\dots)$),
to attenuation-response density $\sigma$ and pixel value $c$, conditioned on Gaussian-sampled shape/
appearance latents $z_s, z_a$ (Eqs. 2-4). Per-ray, points are alpha-composited (Eq. 5):
$$c_r = \sum_{i=1}^N c_r^i \alpha_r^i \exp\Big(-\sum_{j=1}^{i-1}\sigma_r^j\delta_r^j\Big), \quad \alpha_r^i = 1-\exp(-\sigma_r^i\delta_r^i)$$

### B. Self-Supervised Discriminator (MedNeRF's actual novel contribution, §II-C.1)

To give the generator richer feedback than GRAF's single discriminator provides, MedNeRF's
discriminator $D_\phi$ is augmented with an **auto-encoding self-supervision pretext task**: two
decoders reconstruct the discriminator's own intermediate feature maps at two scales ($32^2$ and
$8^2$), forcing $D_\phi$ to learn features expressive enough to be *decoded back*, not just to
classify real/fake. This adds an LPIPS-style reconstruction loss on VGG16 features (Eq. 6):
$$\mathcal{L}_r = \mathbb{E}_{f\sim D(p),\,p\sim P}\Big[\tfrac{1}{whd}\sum_i \|\phi_i(\mathcal{G}(f)) - \phi_i(\mathcal{T}(p))\|_2\Big]$$

### C. DAG — Weight-Shared Multi-Head Discriminator Training (§II-C.2)

Rather than naively augmenting the training data (which the paper found "works less favorably"),
MedNeRF adopts the DAG (Data Augmentation Optimized for GAN) framework: $n{=}4$ transformations
$\mathcal{T}_k$ (flips + rotations, invertible so the Jensen-Shannon-preserving property of Eq. 7
holds), each judged by its own discriminator head $D_k$ that shares all but its final layers with
the others (memory-efficient). A hinge loss combines the untransformed head with the average of the
transformed heads (Eq. 8-9, $\lambda{=}0.2$).

### D. Single-View Test-Time Reconstruction (§II-C.3, `fit_latent_and_weights`)

After pretraining, a single input X-ray is reconstructed by *slightly fine-tuning* the pretrained
generator's weights jointly with the shape/appearance latents $z_s, z_a$ (a "relaxed reconstruction"
formulation, following [23] in the paper's own references) — this is explicitly **not** latent-only
optimization; the generator weights move too. The objective adds an MSE distortion term to balance
the perception-distortion tradeoff (Eq. 10):
$$\mathcal{L}_{\text{gen}} = \lambda_1\mathcal{L}_r(\text{VGG16}) + \lambda_2\mathcal{L}_{\text{MSE}}(G) + \lambda_3\mathcal{L}_{\text{NLLL}}(z_s, z_a)$$
with the paper's own tuned values: $\text{lr}=0.0005$, $\beta_1=0$, $\beta_2=0.999$, $\lambda_1=0.3$,
$\lambda_2=0.1$, $\lambda_3=0.3$ — **this project's own `fit_latent_and_weights` already matches
these exact coefficients** (see `LOG.md`), it was only this doc's earlier draft that stated the wrong
numbers ($\lambda_{\text{MSE}}=1.0,\lambda_z=0.01$).

## 3. Reported Training & Evaluation Setup

- **Dataset:** DRRs rendered from 20 CT chest scans + 5 CT knee scans (no paired real X-ray/CT data
  needed for pretraining — DRR generation removes patient radiation exposure and gives full control
  over capture range/resolution). 128×128 resolution, 72 DRRs per object at 5° intervals (a fifth of
  a full 360° vertical rotation) used for training; the model renders the rest.
- **Pretraining:** 100,000 iterations, batch size 8. Camera sampled on a sphere with $u,v$ covering a
  70°-85° polar elevation band and $u_{\min}{=}0, u_{\max}{=}1$ for the full 360° vertical sweep.
- **Test-time fitting budget:** the paper itself doesn't fix a single number; this project's own
  convergence experiment (see `docs/TRAINING_PLAN.md`) found the loss/PSNR curve plateaus around
  1500-2000 iterations, with 50 iterations used as a deliberately fast, under-converged default.

### Reported results (Table I)

Single-view-X-ray-conditioned reconstruction, full-rotation projections: **Knee PSNR
30.17 ± 1.93 dB, SSIM 0.670 ± 0.040**; **Chest PSNR 28.54 ± 0.79 dB, SSIM 0.462 ± 0.082**. The paper
also compares 2D-rendering quality against plain pixelNeRF and GRAF baselines (Fig. 4), showing
MedNeRF more accurately estimates volumetric depth than either.

## 4. Stated Limitations

The paper doesn't have a dedicated "Limitations" section; the closest statements are woven into
§II-B/§III:
1. **Small medical datasets make adversarial training brittle** — the whole motivation for §II-C's
   self-supervised discriminator + DAG contributions is that naive GRAF training mode-collapses or
   underperforms without them on datasets this size.
2. **Implicit network's limited capacity** — the paper notes (§III-A) that "despite the implicit
   linear network's limited capacity," the model can still disentangle 3D anatomy identity and
   attenuation response — phrased as a capacity caveat the results nonetheless overcome, not an
   unaddressed weakness.
3. **Perception-distortion tradeoff** — explicitly named in §II-C.3: pure LPIPS/perceptual loss can
   drift from pixel accuracy, which is why the MSE term in Eq. 10 exists at all; this is presented as
   a fundamental tension the added MSE term only balances, not eliminates.
