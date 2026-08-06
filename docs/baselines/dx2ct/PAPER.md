# DX2CT: Diffusion Model for 3D CT Reconstruction from Bi or Mono-planar 2D X-ray(s)

**Paper:** *DX2CT: Diffusion Model for 3D CT Reconstruction from Bi or Mono-planar 2D X-ray(s)*
**Authors:** Yun Su Jeong, Hye Bin Yoo, Il Yong Chun (Sungkyunkwan University, Republic of Korea)
**Venue/Publication:** ICASSP 2025 / arXiv:2409.08850
**arXiv Cache:** `arxiv_papers/Dx2CT_2409.08850.pdf`

> **Correction note (2026-08-05):** an earlier version of this file cited the wrong paper title
> ("Dx2CT: Position-Aware Slice Diffusion Model for 3D CT Reconstruction from Single X-Ray", which
> does not exist) and described the conditioning mechanism as simple "sinusoidal position embeddings
> and cross-attention" — the real paper's central contribution is a specific transformer module (3D
> Positional Query Transformer, 3DPQT) feeding a **SPADE**-conditioned U-Net, not plain
> cross-attention conditioning. The paper's *primary* setup is also **biplanar** (PA + lateral), with
> monoplanar as a secondary comparison case — not "Single X-Ray" as the old title implied. Rewritten
> directly from the downloaded PDF; see `LOG.md`.

---

## 1. Executive Summary & Paradigm

DX2CT belongs to the **reconstruction-based paradigm**, reformulating the memory-heavy problem of
generating a full 3D CT volume from 2D X-ray(s) as a **per-slice 2D reconstruction problem**: instead
of a 3D U-Net over an entire volume (the standard but memory/compute-heavy approach used by prior
X-ray-to-CT work), DX2CT reconstructs one 2D CT slice at a time via a conditional diffusion model,
then stacks slices to form the 3D volume. Two components are the paper's stated contributions:

1. **3D Positional Query Transformer (3DPQT)** — a ViT that produces X-ray feature maps aligned with
   the 3D position of a *specific target slice*.
2. **SPADE-based conditioning** — those 3D-position-aware feature maps condition a denoising 2D U-Net
   via spatially-adaptive normalization (SPADE), which the paper's own ablation shows works better
   than plain channel-concatenation.

## 2. Mathematical Formulation & Architecture

### A. Multi-scale X-ray Feature Extraction

From biplanar 2D X-rays $\mathbf{i}^{\text{PA}}, \mathbf{i}^{\text{Lat}}$, a shared feature extractor
$\mathcal{E}_{\theta_{\mathcal{E}}}$ (ResNet-50, ImageNet-pretrained, using `conv2_x`/`conv3_x`/
`conv4_x` as $L{=}3$ scales) produces multi-scale feature maps (Eq. 1):
$$\mathbf{f}^v = \{\mathbf{f}_1^v, \dots, \mathbf{f}_L^v\} = \mathcal{E}_{\theta_{\mathcal{E}}}(\mathbf{i}^v), \quad v \in \{\text{PA}, \text{Lat}\}$$

### B. 3D Positional Query Transformer (3DPQT)

For a target CT slice on anatomical plane $m \in \{\text{axial, coronal, sagittal}\}$ at index $n$,
two position-encoding networks $\mathcal{P}_{\theta_{\mathcal{P}}}$/$\mathcal{Q}_{\theta_{\mathcal{Q}}}$
generate multi-scale CT positional embeddings (Eq. 2) and X-ray positional embeddings (Eq. 3) from
the slice's 3D coordinates. 3DPQT ($\mathcal{T}_{\theta_{\mathcal{T}}}$, $B{=}12$ multi-head
cross-attention blocks per scale) then uses the CT positional embeddings as **query** and the X-ray
feature+positional embeddings as **key/value** to retrieve 3D-position-aware feature maps (Eq. 4):
$$\mathbf{c}_n^m = \mathcal{T}_{\theta_{\mathcal{T}}}(\mathbf{f}^{\text{PA}}, \mathbf{f}^{\text{Lat}}, \mathbf{p}_n^m, \mathbf{q}^{\text{PA}}, \mathbf{q}^{\text{Lat}})$$
This is the mechanism that lets the model, at generation time, pull out *exactly the X-ray
information relevant to one particular target 3D slice location* — the paper's stated reason
per-slice generation avoids needing a full 3D-attention transformer.

### C. SPADE-Conditioned Denoising DDPM

The 3D-position-aware feature maps condition a 2D denoising U-Net $\mathcal{D}_{\theta_{\mathcal{D}}}$
via SPADE (chosen over channel-concatenation per the paper's own ablation, see §4 below) as a
standard conditional DDPM (Eq. 5-6):
$$\epsilon_\theta = \mathcal{D}_{\theta_{\mathcal{D}}}(\mathbf{x}_t, t, \mathbf{c}_n^m), \qquad \mathcal{L}(\theta) = \mathbb{E}\big[\|\epsilon - \epsilon_\theta(\mathbf{x}_t, t, \mathbf{i}^{\text{All}}, m, n)\|_2^2\big]$$
where $\mathbf{i}^{\text{All}} = \{\mathbf{i}^{\text{PA}}, \mathbf{i}^{\text{Lat}}\}$. Each of the 3
anatomical planes is reconstructed by repeating the process for every slice index $n$ on that plane,
then the per-plane slice stacks together reconstruct the 3D CT volume.

## 3. Reported Training & Evaluation Setup

- **Dataset:** LIDC CT dataset, DRR-synthesized paired 2D X-rays; 1,018 3D-CT/2D-X-ray pairs split
  916 train / 102 test. Real-world experiments use CycleGAN (trained on PadChest) to style-transfer
  synthetic DRR-style X-rays toward real-radiograph appearance, since no paired real-X-ray/real-CT
  data exists.
- **Diffusion:** $T{=}1000$ timesteps, linear noise schedule $10^{-4}$ to $0.02$; denoising U-Net
  initial channel width 64, channel multipliers $[1,1,2,3,4]$.
- **Optimizer:** Adam, lr $5\times10^{-5}$, batch size 16, 80 epochs.
- **Sampling:** DDIM, 50 steps; the initial noise $\mathbf{x}_T$ is **fixed** and DDIM's random noise
  term removed, specifically to keep slice-to-slice reconstruction consistent within one volume.
- **Baselines compared:** PerX2CT (global/local), X2CTGAN, 2DCNN — the paper notes ([8]-[10] in its
  reference list) that some competing methods' code/weights were unavailable, so those specific
  comparisons could not be reproduced and are omitted rather than approximated.

### Reported results (abridged)

- **Biplanar (Table I(a), PSNR/SSIM/LPIPS):** X2CTGAN 26.747/0.647/0.316, PerX2CT$_{\text{global}}$
  27.546/0.730/0.218, PerX2CT$_{\text{local}}$ 27.659/0.739/0.210, **DX2CT (ours) 28.357/0.763/0.225**
  — best PSNR/SSIM, competitive (not best) LPIPS.
- **Monoplanar/PA-only (Table I(b)):** 2DCNN 24.471/0.549/0.427, X2CTGAN 23.042/0.515/0.372,
  **DX2CT (ours) 25.506/0.643/0.277** — best on all three metrics.
- **Ablation (Table II):** both 3DPQT and SPADE conditioning contribute independently; the paper's
  own numbers show 3DPQT-off+concat 26.939/0.721/0.251 → 3DPQT-on+SPADE (full model) 28.357/0.763/
  0.225 — 3DPQT alone (with concat) and SPADE alone (without 3DPQT) each move the needle, but
  combining both gives the best result.

## 4. Stated Limitations

1. **Axial-plane quality is lower than coronal/sagittal**, explicitly attributed in the paper's own
   results discussion (§IV-B) to a geometric reason: the axial plane is *perpendicular* to both
   biplanar X-ray planes (PA and lateral), so it has the least spatial information available from the
   input views of any of the 3 reconstructed planes.
2. **No perceptual loss during training** — the paper notes that without perceptual-loss training,
   DX2CT's LPIPS was only comparable to (not better than) PerX2CT's, unlike its PSNR/SSIM advantage.
3. **Some prior baselines were not reproducible** for direct comparison since their official code/
   trained weights were unavailable (paper's own footnote), so the reported comparison table is
   necessarily incomplete relative to everything in the related-work section.
