# Cosmos-Predict2.5: Ablation Studies & Hyperparameter Calibration (`ABLATION_STUDY.md`)

**Document Version:** 1.0  
**Target Publication:** MICCAI 2026 Resubmission Appendices / Results  
**Location:** `docs/cosmos-predict2.5/ABLATION_STUDY.md`  

---

## 1. Executive Summary

This document presents the systematic **Ablation Study Matrix** and **Hyperparameter Calibration Protocol** for Cosmos-Predict2.5. To address reviewer concerns (**Reviewers #2, #3, #4**) regarding the ill-posed nature of $360^\circ$ single-view novel view synthesis (NVS), each proposed physical, geometric, and sampling component is evaluated incrementally on the held-out **NSCLC Radiogenomics (402 cases)** OOD test split.

```
+---------------------------------------------------------------------------------------------------+
|                                     INCREMENTAL ABLATION TRAJECTORY                               |
+---------------------------------------------------------------------------------------------------+
| (A) Base Cosmos-Predict2.5  ---> (B) + Angular Weighting  ---> (C) + 180° Mirror & Periodic  ---> (D) Full Model |
|     (Rectified Flow Base)            (w(theta_k) Side Arc)     (AP Soft Prior + Orbit Noise)    (+ L_atten Mass)
+---------------------------------------------------------------------------------------------------+
```

Each variant is produced by (1) training with the corresponding `--gamma_side`/`--loss_atten_weight` CLI flags on `predict2_5/trainer.py`, and (2) evaluating with the matching `Inferencer.predict(use_ap_soft_prior=..., use_periodic_noise=...)` flags — both required, since the 180° AP prior and periodic-orbit noise are inference-time-only techniques applied regardless of which checkpoint is loaded.

---

## 2. Full Component Ablation Matrix Protocol

All variants will be trained under identical post-training budgets ($10,000$ steps, effective global batch size $16$ in `bf16-mixed` precision) on the **TCIA + MELA2022** training set ($1,296$ volume cases) and evaluated on the **NSCLC** OOD test set ($402$ cases) across $93$ rotation views ($256\times256$ resolution).

| Model Variant | $\mathcal{L}_{\text{RF}}$ (Base) | $w(\theta_k)$ Side Weight | $180^\circ$ AP Soft Prior | Periodic Orbit Noise | $\mathcal{L}_{\text{atten}}$ (Line-Integral) | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | Side-Arc PSNR ($45^\circ\text{--}135^\circ$) ↑ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **(A) Base Cosmos-Predict2.5** | ✅ | ❌ | ❌ | ❌ | ❌ | TBD | TBD | TBD | TBD |
| **(B) + Angular Side Weighting ($w(\theta_k)$)** | ✅ | ✅ | ❌ | ❌ | ❌ | TBD | TBD | TBD | TBD |
| **(C) + $180^\circ$ AP Mirror-Symmetry Anchor & Periodic Noise** | ✅ | ✅ | ✅ | ✅ | ❌ | TBD | TBD | TBD | TBD |
| **(D) Full Proposed Model (+ $\mathcal{L}_{\text{atten}}$)**| ✅ | ✅ | ✅ | ✅ | ✅ | **TBD** | **TBD** | **TBD** | **TBD** |

*Reproducibility:* Variant training uses `torchrun ... predict2_5/trainer.py --gamma_side <0.0|1.0> --loss_atten_weight <0.0|0.02>`; variant evaluation uses `Inferencer.predict(..., use_ap_soft_prior=<False|True>, use_periodic_noise=<False|True>)`.

---

## 3. Methodological Rationale per Ablation Component

### 3.1 Impact of Angular-Offset Loss Weighting ($w(\theta_k)$)
- **Continuous Side-View Arc Supervision ($\theta \in [45^\circ, 135^\circ] \cup [225^\circ, 315^\circ]$):** Lateral and oblique viewpoints suffer from maximum depth superposition and maximum angular distance from the frontal PA view ($0^\circ$). The smooth sinusoidal loss weighting $w(\theta_k) = 1 + \gamma_{\text{side}} \sin^2(\theta_k)$ ($\gamma_{\text{side}} = 1.0$) elevates supervision across all adjacent lateral and oblique viewpoints.

### 3.2 Impact of Anatomical Mirror-Symmetry Latent Soft Prior ($180^\circ$ AP)
- **Anatomical Symmetry Prior:** Human thoracic anatomy exhibits approximate anterior-posterior mirror symmetry. Horizontally flipping $I_{\text{PA}}$ ($0^\circ$) yields a strong $180^\circ$ AP geometric prior (27.50 dB PSNR measured on 241 NSCLC test cases via `scripts/benchmark_flip_baseline.py`).
- **Soft Latent Feature Prior:** Incorporating $z_{\text{cond\_180}} = \mathcal{E}_{\text{VAE}}(\mathcal{H}_{\text{flip}}(I_{\text{PA}}))$ as a soft conditioning prior at latent frame 12 anchors posterior structural topology while allowing the DiT model to adapt to cone-beam cardiac magnification and smooth temporal transitions. The blend is applied **once**, after the reverse-diffusion trajectory completes (not per denoising step) — repeating the blend at every step would compound geometrically toward a hard replacement rather than the intended $\alpha=0.3$ soft mix.

### 3.3 Impact of Line-Integral Global Attenuation Mass Loss ($\mathcal{L}_{\text{atten}}$)
- **Mass Regularization:** Enforcing $\mathcal{L}_{\text{atten}}$ directly on un-exponentiated line integrals $D(\mu)$ constrains global integrated attenuation mass across rotation views to stabilize total tissue mass.

### 3.4 Impact of Periodic Orbit Noise Schedule
- **Inter-Frame Flicker & Train/Inference Distribution Gap:** The periodic-orbit schedule $\epsilon(\theta_k) = \mathbf{a}\cos(\theta_k) + \mathbf{b}\sin(\theta_k)$ eliminates the $0^\circ\leftrightarrow360^\circ$ boundary noise discontinuity, but it is an **inference-only** substitution — post-training (`module.py`) still samples i.i.d. per-frame Gaussian noise, so the DiT's temporal attention never observes this rank-2-correlated noise structure during training. This ablation axis isolates whether the closed-orbit continuity benefit outweighs that train/inference mismatch; disable via `Inferencer.predict(use_periodic_noise=False)` to fall back to the i.i.d. baseline.

---

## 4. Hyperparameter Calibration & Sensitivity Protocol

### 4.1 Loss Weight Calibration Protocol ($\lambda_{\text{atten}}, \gamma_{\text{side}}$)

To prevent auxiliary regularizers from dominating or destabilizing the primary Rectified Flow velocity matching loss $\mathcal{L}_{\text{angle-RF}}$, loss weights will be calibrated during post-training via gradient scale parity checking:

$$R_i = \frac{\|\nabla_\theta \mathcal{L}_i\|_2}{\|\nabla_\theta \mathcal{L}_{\text{angle-RF}}\|_2} \in [0.05, 0.15]$$

| Parameter | Search Range | Selected Value | Calibration Target |
|---|---|---|---|
| $\gamma_{\text{side}}$ (Side Weight) | $[0.2, 2.0]$ | **`1.0`** | Doubles supervision weight across the lateral-oblique arc $\theta \in [45^\circ, 135^\circ]$ without over-emphasizing lateral noise at the expense of $0^\circ$ PA sharpness. |
| $\lambda_{\text{atten}}$ (Global Mass) | $[0.005, 0.10]$ | **`0.02`** | Maintains $\|\nabla \mathcal{L}_{\text{atten}}\|_2$ at $\approx 5\text{--}10\%$ of $\|\nabla \mathcal{L}_{\text{angle-RF}}\|_2$. |

### 4.2 Side-View Weight Sensitivity Analysis ($\gamma_{\text{side}}$)

| $\gamma_{\text{side}}$ | Overall PSNR (dB) ↑ | Side-Arc PSNR ($45^\circ\text{--}135^\circ$) ↑ | Frontal PSNR ($0^\circ$) ↑ |
|:---:|:---:|:---:|:---:|
| `0.0` (Uniform) | TBD | TBD | TBD |
| `0.5` | TBD | TBD | TBD |
| **`1.0` (Selected)** | **TBD** | **TBD** | **TBD** |
| `2.0` | TBD | TBD | TBD |

*Protocol:* Evaluate $\gamma_{\text{side}}$ sensitivity across the NSCLC test set upon completion of post-training runs.
