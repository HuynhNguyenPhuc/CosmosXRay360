# SV-DRR Codebase & Repository Mapping

**Cloned Repository:** `baselines/cloned/SV-DRR/`
**Base Model Weights:** `xiechun-tsukuba/svdrr-dit-fb-256` (HuggingFace Hub)
**License:** MIT License (`LICENSE`, copyright 2025 Xie Chun — the paper's first author)

> **Correction note (2026-08-05):** the license line previously read "Apache-2.0 / MIT" (an
> unresolved guess); verified directly against `baselines/cloned/SV-DRR/LICENSE`, which is plain MIT.
> `CCProjection`/`SvdrrDiTPipeline` class names below were re-confirmed real via `grep` against
> `pipeline_svdrr_DiT.py`. See `LOG.md`.

---

## 1. Repository Directory Map

```
baselines/cloned/SV-DRR/
├── pipeline_svdrr_DiT.py    # Main Diffusers custom pipeline (SvdrrDiTPipeline & CCProjection)
├── train_svdrr_DiT.py       # Official training script
├── models/                  # DiT Transformer submodules and configs
│   └── base_model/256/      # Materialized local snapshot of DiT & VAE weights
└── environment.yaml         # Original execution dependencies
```

---

## 2. Key Modules & Paper Equation Mappings

- **`pipeline_svdrr_DiT.py:CCProjection`**: Implements $c_{\text{pose}} = \text{MLP}_{\text{pose}}(p)$ mapping target rotation angles into conditioning embeddings.
- **`pipeline_svdrr_DiT.py:SvdrrDiTPipeline`**: Implements the full inference pipeline (VAE encoding, CLIP feature extraction, DiT cross-attention denoising, and VAE decoding).
- **`train_svdrr_DiT.py`**: Reference training loop.

---

## 3. Discrepancies Between Official Code & Paper

1. **Training View Pair Scope:**
   - *Official Code (`train_svdrr_DiT.py`):* Hardcoded to load only a single fixed 90°-apart pair `(pa.png, lat.png)` per patient during training.
   - *Impact & Fix:* Evaluating across a full 0°–360° rotation required extending the training dataset (`baselines/train/svdrr.py`) to sample arbitrary random `(source, target)` view pairs from all 93 pre-rendered angles per patient ($\sim 93 \times 92$ combinations).
2. **`cc_projection` Optimizer Learning Rate:**
   - *Official Code:* `cc_projection` parameters are optimized at $10 \times$ the DiT transformer's learning rate (`lr=5e-5` vs `5e-6`).
   - *Our Implementation:* Preserved this $10 \times$ multiplier in `baselines/train/svdrr.py` during joint fine-tuning.
3. **Diffusers Custom Component Resolution:**
   - Loading `SvdrrDiTPipeline.from_pretrained()` directly from Hub ID failed due to diffusers custom component pathing (`cc_projection/pipeline_svdrr_DiT.py`).
   - *Fix:* `SVDRRWrapper` and `train/svdrr.py` download a local snapshot via `huggingface_hub.snapshot_download` to `baselines/cloned/SV-DRR/models/base_model/256`.

---

## 4. Wrapper & Integration Entry Points

- **Model Wrapper:** `baselines/models/svdrr.py` (`SVDRRWrapper`)
- **Training Entry Point:** `baselines/train/svdrr.py`
- **Unit Tests:** `baselines/tests/test_svdrr_wrapper.py`
