# XRaySyn Codebase & Repository Mapping

**Cloned Repository:** `baselines/cloned/XraySyn/`
**License:** MIT License (`LICENSE`, copyright cpeng93 — Cheng Peng, the paper's first author)
**Official author repo:** confirmed via `README.md`'s install/checkpoint instructions; no separate
commit hash tracked (cloned as a plain copy, no nested `.git`).

> **Correction note (2026-08-05):** the license line above previously read "Custom research license
> (Johns Hopkins University)", which was wrong — the repo ships a plain MIT `LICENSE` file. Verified
> directly against `baselines/cloned/XraySyn/LICENSE`. See `LOG.md`.

The paper (see `PAPER.md`) names its two stages **3D PriorNet (3DPN)** and **2D RefineNet (2DRN)**,
built on a shared **CT2Xray** differentiable projector. The code doesn't use those names directly —
`XraySynModel` (in `ct2xray_real_gan_meta.py`) implements all three as `net3d` (3DPN), `net2d`
(2DRN), and `ct2xray()`/`DRRProjector` (CT2Xray's FP/BP) respectively; this project's own
docs/wrapper code (and `docs/LOSSES.md`) also use the `net3d`/`net2d`/`backproj` names throughout, so
that's the terminology to search for when reading code, not "3DPN"/"2DRN".

---

## 1. Repository Directory Map

```
baselines/cloned/XraySyn/
├── xraysyn/
│   ├── models/
│   │   ├── ct2xray_real_gan_meta.py  # Primary XraySynModel class
│   │   └── base.py                   # Base model wrapper
│   ├── networks/
│   │   ├── unet.py                   # 3D UNet generator (net3d)
│   │   ├── rdn_meta.py               # 2D residual refinement network (net2d)
│   │   ├── common.py                 # NLayerDiscriminator & GANLoss (netD)
│   │   └── drr_projector_new.py      # Differentiable forward/backward CUDA DRR projector
│   └── utils/
│       ├── geometry.get_6dofs_transformation_matrix
│       └── torch.NormLayer
├── simplified_bone_absorb_2d.pt       # Pre-computed bone absorption lookup table
└── simplified_tissue_absorb_2d.pt     # Pre-computed tissue absorption lookup table
```

---

## 2. Key Modules & Paper Equation Mappings

- **`DRRProjector(mode="backward")`** (`drr_projector_new.py`, confirmed real: `mode='forward'`/`'backward'` are actual constructor args): implements the single-image backprojector BP (paper Eq. 7).
- **`DRRProjector(mode="forward")`**: implements the differentiable forward projector FP (paper Eq. 1-2) that CT2Xray is built on.
- **`UnetGenerator(dimension="3d")`** (`unet.py`, confirmed real: `dimension` is an actual kwarg, `{"2d","3d"}`): implements `net3d`/3DPN, predicting `{V_CT, V_mask}` from the backprojected input (paper Eq. 8).
- **`XraySynModel.ct2xray()`**: applies the HU→$\mu$ scaling used by this specific implementation (`vol = 0.0008088*(vol*5000-1000)+1.030` in `ct2xray_real_gan_meta.py` line 91 — a concrete linear approximation the code uses in place of the paper's more general per-energy $\mu(m,E)$ lookup table, Eq. 3), material-thickness forward projection (Eq. 4), attenuation lookup against `simplified_bone_absorb_2d.pt`/`simplified_tissue_absorb_2d.pt`, and the Beer-Lambert exponentiation (Eq. 5-6).
- **`make_model()` (`rdn_meta.py`)**: implements `net2d`/2DRN, the Residual Dense Network + $\mathcal{M}$ shuffle-net refinement (paper Eq. 11).
- **`NLayerDiscriminator`/`GANLoss`** (`common.py`, confirmed real classes): PatchGAN discriminator and LSGAN loss (paper Eq. 13-14).

---

## 3. Discrepancies Between Official Code & Paper

1. **Training View Angular Range (`OTHER_POSE_THETA_Y_RANGE`):**
   - *Official Code:* The demo training script (`self.views`) only trained `net2d` across a narrow range of $\theta_y \in [-0.05, 0.05] \times \pi$ (roughly $\pm 9^\circ$ around the frontal view).
   - *Impact & Fix:* Evaluating across $360^\circ$ novel views resulted in severe degradation at large angles. In `baselines/train/xraysyn.py`, `OTHER_POSE_THETA_Y_RANGE` was widened to `(-1.0, 1.0)` (covering full $360^\circ$ rotation).
2. **Fixed Batch-Size Constraint in Pose Computation (`get_T`):**
   - *Official Code:* `XraySynModel.get_T` hardcoded a batch size of 4 (`torch.cat([T, T, T, T])`).
   - *Fix:* Added `_get_T_batched` and `_get_T_multi` in `baselines/models/xraysyn.py` to dynamically slice or repeat $T$ matrices for arbitrary batch sizes and multi-azimuth sweeps.
3. **Working Directory Dependency for Physical Lookup Tables:**
   - *Official Code:* `XraySynModel.__init__` expects `simplified_bone_absorb_2d.pt` in the current working directory.
   - *Fix:* `XRaySynWrapper.__init__` temporarily switches CWD to `baselines/cloned/XraySyn/` during initialization.

---

## 4. Wrapper & Integration Entry Points

- **Model Wrapper:** `baselines/models/xraysyn.py` (`XRaySynWrapper`)
- **Training Entry Point:** `baselines/train/xraysyn.py`
- **Unit Tests:** `baselines/tests/test_xraysyn_wrapper.py`
