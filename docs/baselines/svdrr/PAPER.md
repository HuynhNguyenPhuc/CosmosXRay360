# SV-DRR: High-Fidelity Novel View X-Ray Synthesis Using Diffusion Model

**Paper:** *SV-DRR: High-Fidelity Novel View X-Ray Synthesis Using Diffusion Model*
**Authors:** Chun Xie, Yuichi Yoshii, Itaru Kitahara (Center for Computational Sciences, University
of Tsukuba; Tokyo Medical University Ibaraki Medical Center)
**Venue/Publication:** MICCAI 2025 / arXiv:2507.05148
**arXiv Cache:** `arxiv_papers/SV_DRR_2507.05148.pdf`

> **Correction note (2026-08-05):** an earlier version of this file cited the wrong paper title
> ("SV-DRR: Single-View Digitally Reconstructed Radiograph Synthesis with Diffusion Transformers",
> which does not exist) and an architecture description (`CCProjection` mapping a `(φ, θ, r)` pose
> vector via a standalone MLP, cross-attention-only conditioning) that doesn't match the paper's
> actual channel-concatenation-based View-Conditioned DiT — evidence it was written without reading
> the cached PDF. Rewritten directly from the downloaded PDF; see `LOG.md`.

---

## 1. Executive Summary & Paradigm

SV-DRR is a **generative synthesis-based (direct 2D → 2D)** novel view synthesis framework: given a
single source X-ray $I^S$ and its view parameters $v^S$, it synthesizes a target-view image
$\hat{I}^T$ for arbitrary target view parameters $v^T$, without ever building an intermediate 3D
representation. The name is explicitly explained in the paper (Sec. 1) as "Single-View DRR" —
inspired by Digitally Reconstructed Radiography as a concept, not because the method itself renders
DRRs; SV-DRR synthesizes novel views directly from a single 2D projection.

## 2. Mathematical Formulation & Architecture

### A. Problem Formulation

Given source image $I^S$ with view parameters $v^S$, SV-DRR estimates the conditional distribution
of the target image (Eq. 1):
$$\hat{I}^T \sim \mathcal{P}(I \mid I^S, v^S, v^T)$$

### B. View-Conditioned Diffusion Transformer (VCDiT)

Built on the Latent Diffusion Model (LDM) framework — diffusion runs in a pre-trained VAE's latent
space, not pixel space — with a DiT-based denoiser $\epsilon_\theta$ enhanced with cross-attention to
inject view-conditioning. Conditioning has **two complementary streams** (this is the part the
earlier version of this doc got wrong — it isn't a single pose-MLP into cross-attention):

1. The **source image embedding** (via a frozen CLIP image encoder $\mathcal{E}$, per Fig. 1) is
   concatenated with an **encoded relative polar coordinate** between $v^T$ and $v^S$, forming a view
   embedding that carries spatial-transformation information, fed to the DiT's cross-attention.
2. The **source image's own VAE latent** is *channel-concatenated with the noised target latent*
   before denoising — reinforcing structural alignment directly at the latent level, independent of
   the cross-attention path.

A shared AdaLN-Zero layer handles timestep embedding and a shared learnable linear layer maps the
view-conditioning signal into the VCDiT latent space efficiently. Training objective (Eq. 2):
$$\mathcal{L} = \mathbb{E}_{z_0, c, \epsilon, t}\big[\|\epsilon - \epsilon_\theta(z_t, c(I^S, v^S, v^T), t)\|_2^2\big]$$

### C. Weak-to-Strong Training Strategy

Progressive resolution refinement: train at low resolution first, then fine-tune at higher
resolutions. To avoid the positional-embedding inconsistency that normally degrades performance
during an LR→HR transition, the HR model's positional embeddings are *initialized by interpolating*
the LR model's (a known trick from prior DiT literature, cited rather than novel to this paper).

## 3. Reported Training & Evaluation Setup

- **Dataset — LIDC-IDRI-DRR:** built from LIDC-IDRI (1,012 CT scans → 889 volumes after excluding
  >2.5mm slice thickness; 16 held out for eval, rest for training). DiffDRR renders 1,500 X-ray views
  per CT scan (positions sampled on a 1.8m-radius hemisphere via Fibonacci lattice sampling); the
  first/canonical view is always the frontal PA source.
- **Latent encoding:** the VAE used in SDXL. Image conditioning: CLIP image encoder. VCDiT is
  initialized from **PixArt-Σ-256** pretrained weights.
- **Optimizer:** AdamW, lr $5\times10^{-6}$ (256 res) / $3\times10^{-6}$ (512) / $1\times10^{-6}$
  (1024); batch sizes 64 / 32 / 8 respectively.
- **Compute:** single H100 GPU; 200K steps at 256 res, 100K steps at 512 and 1024 res.
- **Sampling:** DPMSolver, 20 inference steps, guidance scale 3.
- **Eval view sets:** "Simple" (azimuth -90° to 90° in 5° steps, 36 views — matches XraySyn's own
  testing convention) and "Hemisphere" (1,499 Fibonacci-sampled views over a full hemisphere).

### Reported results (Table 1, abridged)

Against XraySyn, Zero123, Zero123-XL on LIDC-IDRI-DRR: SV-DRR-512 reaches **SSIM 0.7509 / PSNR
23.98 / LPIPS 0.107** (Simple views) and **SSIM 0.368 / PSNR 11.29 / LPIPS 0.059** (Hemisphere),
outperforming every baseline on every metric in both view sets — with only minor variance across the
256/512/1024 output resolutions, which the paper attributes to the weak-to-strong training strategy.
A 15-medical-expert user study (50 image pairs each) found participants at **48.7% classification
accuracy** distinguishing SV-DRR output from real DiffDRR simulation — statistically indistinguishable
from the 50% chance baseline (one-sample t-test p=0.334; binomial test p=0.413).

## 4. Stated Limitations

The paper's own conclusion states the limitation directly and narrowly: **"Future work will focus on
improving cross-view consistency to enhance anatomical alignment and realism."** No explicit
enumerated limitations section exists (unlike NAF's paper); this single sentence is the paper's own
framing. Implicitly, per its own Sec. 4 discussion: as a 2D-latent-diffusion method with no explicit
3D constraint, quality is not guaranteed to be multi-view-consistent across independently-sampled
target angles — the model synthesizes each view's distribution independently rather than enforcing a
shared underlying 3D representation.
