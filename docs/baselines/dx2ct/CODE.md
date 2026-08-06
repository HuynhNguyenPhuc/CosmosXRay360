# Dx2CT Codebase & Repository Mapping

**Cloned Repository:** `baselines/cloned/DX2CT/`  
**Official Code Status:** **Code-Unreleased** (Official implementation unreleased by authors).  
**Implementation Status:** From-scratch modular reimplementation in `baselines/cloned/DX2CT/model.py` based on arXiv:2409.08850 specification.  

---

## 1. Repository Directory Map

```
baselines/cloned/DX2CT/
└── model.py    # From-scratch implementation of DX2CTModel + its sub-modules (see below)
```

---

## 2. Key Modules & Paper Equation Mappings

> **Correction note (2026-08-05):** the class names below were previously wrong (`PositionAwareAttention`,
> `SPADEResBlock`) — re-verified with `grep -n "^class "` against the actual `model.py`. The real
> class names below map closely to the paper's own 3DPQT/SPADE naming (see `PAPER.md`), which is a
> good sign this reimplementation followed the paper's architecture deliberately rather than loosely.

- **`model.py:XRayFeatureExtractor`**: multi-scale X-ray feature extraction (paper Eq. 1, $\mathbf{f}^v$).
- **`model.py:SinusoidalPositionEncoding`**: the position-encoding networks $\mathcal{P}_{\theta_{\mathcal{P}}}$/$\mathcal{Q}_{\theta_{\mathcal{Q}}}$ (paper Eq. 2-3).
- **`model.py:TransformerCrossAttentionBlock`** / **`model.py:SliceTransformer3DPQT`**: the 3D Positional Query Transformer itself (paper Eq. 4, $\mathbf{c}_n^m$) — the class name directly matches the paper's own "3DPQT" terminology.
- **`model.py:SPADE`** / **`model.py:SPADEResNetBlock`**: SPADE conditioning layers (paper §III-C).
- **`model.py:DenoisingUNetSPADE`**: the SPADE-conditioned denoising 2D U-Net $\mathcal{D}_{\theta_{\mathcal{D}}}$ (paper Eq. 5).
- **`model.py:DX2CTModel`**: top-level module wiring the above into the full conditional DDPM (paper Eq. 6).

---

## 3. Discrepancies Between Official Code & Paper

1. **Code-Unreleased Status:**
   - *Official Code:* The original authors of arXiv:2409.08850 did not release official source code.
   - *Reimplementation:* `baselines/cloned/DX2CT/model.py` was implemented from scratch following the exact architecture and math described in ICASSP 2025 / arXiv:2409.08850.
2. **Paper-Fidelity Training Supervision Target:**
   - *Original Script Bug:* `train/dx2ct.py` previously trained the diffusion generator against DiffDRR 2D projections rather than real axial CT density slices.
   - *Fix:* Rewrote `train/dx2ct.py` to extract real axial CT density slices (`load_ct_volume_cached`/`extract_axial_slices`) from cached CT volumes (`datasets/_ct_volume_cache/`), matching arXiv:2409.08850 methodology exactly.
3. **Beer-Lambert Perspective Ray Marching for Novel Views:**
   - *Original Wrapper Bug:* Used `.mean(dim=0)` depth collapse across slices.
   - *Fix:* Switched `Dx2CTWrapper.infer_multi_views` to `perspective_ray_march(..., reduction="attenuation_sum")` -- a true Beer-Lambert line integral through the generated 3D CT attenuation volume.
4. **Dynamic Loader to Avoid `sys.modules['model']` Namespace Pollution:**
   - *Fix:* `baselines/models/dx2ct.py` loads `DX2CTModel` via `importlib.util` under module alias `dx2ct_model_module`, preventing namespace collisions with other baseline `model.py` files.

---

## 4. Wrapper & Integration Entry Points

- **Model Wrapper:** `baselines/models/dx2ct.py` (`Dx2CTWrapper`)
- **Training Entry Point:** `baselines/train/dx2ct.py`
- **Unit Tests:** `baselines/tests/test_dx2ct_architecture.py`, `baselines/tests/test_dx2ct_wrapper.py`
