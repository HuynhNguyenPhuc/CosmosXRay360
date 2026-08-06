# NAF Codebase & Repository Mapping

**Cloned Repository:** `baselines/cloned/naf_cbct/`  
**License:** MIT License  

---

## 1. Repository Directory Map

```
baselines/cloned/naf_cbct/
├── src/
│   ├── encoder/
│   │   ├── hashencoder/        # Instant-NGP style HashGrid CUDA C++ extension
│   │   │   ├── hashgrid.py
│   │   │   └── src/hashencoder.cu
│   │   └── freqencoder/        # Fourier Frequency Encoder (fallback)
│   └── network/
│       └── network.py          # DensityNetwork coordinate MLP
├── train.py                    # Reference per-scan CBCT optimization script
└── config/                     # Configuration files (*_50.yaml)
```

---

## 2. Key Modules & Paper Equation Mappings

- **`src.encoder.hashencoder.HashEncoder`**: Implements multi-resolution hash-grid spatial encoding.
- **`src.network.network.DensityNetwork`**: Implements coordinate MLP predicting 3D linear attenuation $\mu(\mathbf{x})$.
- **`perspective_ray_march` (`baselines/models/utils.py`)**: Performs stratified ray sampling, linear attenuation integration, and Beer-Lambert correction $I = I_0 \exp(-A)$.

---

## 3. Discrepancies Between Official Code & Paper

1. **Physics Violation Bug (Missing Beer-Lambert Exponentiation):**
   - *Original Wrapper Bug:* Integrated line-integral attenuation density $\int \mu \, ds$ and compared it directly against transmissive intensity target $I_{\text{target}}$ without Beer-Lambert exponential $I_0 \exp(-A)$.
   - *Fix:* Applied `apply_beer_lambert_correction(line_integral)` inside `fit_density_field` and `perspective_ray_march`.
2. **Orthographic Projection Bug vs Perspective Ray Marching:**
   - *Original Wrapper Bug:* Used `rotate_volume_3d(...)` + axis-sum (orthographic depth collapse) instead of reference perspective ray marching.
   - *Fix:* Replaced with `perspective_ray_march` (`baselines/models/utils.py`) using DiffDRR's camera geometry (`dist`/`fov`/`min_depth`/`max_depth`).
3. **PyTorch 2.x CUDA Compile Fix (`hashencoder.cu`):**
   - *Official Code:* Failed to compile on PyTorch 2.x and CUDA 12+ due to deprecated `DeprecatedTypeProperties` conversion.
   - *Fix:* Updated `hashencoder.cu` to use `.scalar_type()` instead of `.type()` and added `_backend = get_backend()` in `hashgrid.py`. Added fallback to `FreqEncoder` if `HashEncoder` is unavailable.

---

## 4. Wrapper & Integration Entry Points

- **Model Wrapper:** `baselines/models/naf.py` (`NAFWrapper`)
- **Training Entry Point:** `baselines/train/naf.py`
- **Unit Tests:** `baselines/tests/test_naf_wrapper.py`
