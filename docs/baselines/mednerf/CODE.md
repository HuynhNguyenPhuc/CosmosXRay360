# MedNeRF Codebase & Repository Mapping

**Cloned Repository:** `baselines/cloned/mednerf/`  
**Core Framework:** GRAF (`cloned/mednerf/graf-main/`)  
**License:** MIT License  

---

## 1. Repository Directory Map

```
baselines/cloned/mednerf/graf-main/
├── graf/
│   ├── config.py             # Model construction & render pose utilities
│   ├── models/
│   │   ├── generator.py      # GRAF coordinate MLP generator
│   │   └── discriminator.py  # PatchGAN discriminator
│   └── utils.py              # Camera angle conversion (to_theta)
├── submodules/
│   ├── torchsearchsorted.py  # PyTorch-native searchsorted substitute
│   └── GAN_stability/        # LPIPS perceptual loss implementation
├── configs/chest.yaml        # MedNeRF dataset & camera pose parameters
├── render_xray_G_Z.py        # Reference per-patient joint z + generator weight optimization
└── train.py                  # Population-level unconditional GAN pretraining script
```

---

## 2. Key Modules & Paper Equation Mappings

- **`graf.models.generator.Generator`**: Implements continuous coordinate MLP $G_\theta(\gamma(\mathbf{x}), \mathbf{d}, z)$.
- **`graf.models.discriminator.Discriminator`** (confirmed real class, `discriminator.py`): implements MedNeRF's actual paper contribution over plain GRAF — not a generic PatchGAN. `SimpleDecoder` (also a real class in the same file) is the auto-encoding pretext-task decoder from paper §II-C.1/Fig. 2 ("SD: Simple Decoder"); its presence in code is direct evidence this reimplementation follows the paper's self-supervised-discriminator contribution, not just vanilla GRAF.
- **`render_xray_G_Z.py`**: Reference per-patient joint latent $z$ and generator weight $W_G$ optimization loop (paper §II-C.3, Eq. 10).
- **`submodules.GAN_stability.gan_training.lpips.PerceptualLoss`**: Perceptual LPIPS loss module evaluated on AlexNet features (paper's $\mathcal{L}_r$, Eq. 6).

---

## 3. Discrepancies Between Official Code & Paper

1. **Target Image Value Scaling (`[-1, 1]` vs `[0, 1]`):**
   - *Official Code:* The generator outputs synthesized images in range `[-1, 1]` via `tanh`.
   - *Original Wrapper Bug:* Passed `target_xr` in `[0, 1]` directly to LPIPS and MSE loss, creating value-space mismatch gradients.
   - *Fix:* `fit_latent_and_weights` rescales `target_xr` to `[-1, 1]` (`target_xr * 2.0 - 1.0`) prior to loss calculation.
2. **Legacy C++ Extension Mismatch (`torchsearchsorted`):**
   - *Official Code:* Dependent on custom C++ extension `torchsearchsorted` which fails to compile on modern PyTorch 2.x and CUDA 12+ environments.
   - *Fix:* Created drop-in substitution `graf-main/submodules/torchsearchsorted.py` leveraging PyTorch's native `torch.searchsorted`.
3. **LPIPS Reload Overhead:**
   - *Official Code:* Instantiated fresh `lpips.PerceptualLoss` on disk every call, causing severe disk I/O bottlenecks during multi-patient batch loops.
   - *Fix:* Cached process-wide global instance `_PERCEPT_LOSS`.

---

## 4. Wrapper & Integration Entry Points

- **Model Wrapper:** `baselines/models/mednerf.py` (`MedNeRFWrapper`)
- **Training Entry Point:** `baselines/train/mednerf.py`
- **Unit Tests:** `baselines/tests/test_mednerf_wrapper.py`
