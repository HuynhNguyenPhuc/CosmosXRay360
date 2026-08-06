# PixelNeRF Codebase & Repository Mapping

**Cloned Repository:** `baselines/cloned/pixel-nerf/`  
**Core Framework:** `src/`  
**License:** MIT License  

---

## 1. Repository Directory Map

```
baselines/cloned/pixel-nerf/
├── src/
│   ├── model/
│   │   ├── make_model.py     # Main model construction factory
│   │   ├── pixelnerf.py     # PixelNeRF model class combining ResNet encoder + NeRF MLPs
│   │   └── encoder.py      # ResNet34 feature encoder
│   ├── render/
│   │   └── nerf.py           # NeRFRenderer class for coarse/fine ray marching
│   └── util/
│       └── util.py           # Ray generation (gen_rays) & pose utilities
└── conf/default.conf         # PyHocon configuration file
```

---

## 2. Key Modules & Paper Equation Mappings

- **`src.model.encoder.Resnet34`**: Implements $W = E(I)$ feature extraction.
- **`src.model.pixelnerf.PixelNeRFNet`**: Queries 3D spatial points $\mathbf{x}$, samples $W(\pi(\mathbf{x}))$, and passes features through `mlp_coarse` and `mlp_fine`.
- **`src.render.nerf.NeRFRenderer`**: Performs coarse and fine hierarchical ray-marching and alpha-compositing.

---

## 3. Discrepancies Between Official Code & Paper

1. **Dead Sub-network Bug (`mlp_coarse` Gradient Loss):**
   - *Official Code / Original Wrapper:* Called `build_model_and_renderer(..., simple_output=True)`, returning only `outputs.fine.rgb`.
   - *Impact & Fix:* Computing MSE loss solely on `pred_fine` left `mlp_coarse` completely unconstrained with zero gradients. Updated `build_model_and_renderer` to set `simple_output=False` and added `criterion(pred_coarse, target) + criterion(pred_fine, target)` in `baselines/train/pixelnerf.py`.
2. **ResNet Encoder Gradient Freeze:**
   - *Official Code:* Allowed ResNet34 encoder weights to be fine-tuned.
   - *Fix:* In `baselines/models/pixelnerf.py`, `model.stop_encoder_grad = True` is set (not `requires_grad = False` on the
     encoder's parameters, contrary to an earlier version of this note) — this flag makes the official code's own
     `forward()` (`baselines/cloned/pixel-nerf/src/model/models.py`) call `.detach()` on the encoder's output latent
     before it reaches the rest of the network. Functionally equivalent (the encoder receives no gradient either way,
     so `optimizer.step()` is a no-op on its params since `.grad` stays `None`), but it is a `.detach()` on the
     activation, not a `requires_grad` flag on the parameters — verified 2026-08-05, see
     `docs/baselines/pixelnerf/LOG.md`'s correction entry.
3. **Training Loss Alignment:**
   - *Original Script:* Minimized MSE between raw feature maps without ray-marching.
   - *Fix:* Replaced with end-to-end photometric NeRF volume rendering loss against lateral (LAT) target images.

---

## 4. Wrapper & Integration Entry Points

- **Model Wrapper:** `baselines/models/pixelnerf.py`
- **Training Entry Point:** `baselines/train/pixelnerf.py`
- **Unit Tests:** `baselines/tests/test_pixelnerf_wrapper.py`
