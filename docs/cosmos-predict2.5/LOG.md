# Cosmos-Predict2.5 Development & Experimentation Log

**Maintained by:** CosmosXRay360 Team
**Location:** `docs/cosmos-predict2.5/LOG.md`

---

## Chronological Development & Milestone Log

### [2026-07-22] Initial Foundation & Submodule Integration
- **Action:** Integrated NVIDIA `cosmos-predict2.5` foundation model submodule into `cosmos-predict2.5/`.
- **Implementation:**
  - Configured `pyproject.toml` with `cu128`/`cu130` CUDA extras via `uv`.
  - Built `predict2_5/inferencer.py` to wrap `MinimalV1LVGDiT` and `Wan2pt1VAEInterface`.
  - Defined system constants in `predict2_5/constants.py` (`NUM_FRAMES=93`, `VOL_SIZE=256`).

### [2026-07-28] Frame-Token Replacement Conditioning Implementation
- **Action:** Developed frame replacement conditioning logic inside `Inferencer.denoise` and `Inferencer.predict`.
- **Rationale:** Standard video diffusion from a single conditioning image drifts significantly across later frames in a $360^\circ$ rotation trajectory.
- **Fix:** At every UniPC step $t$, enforced explicit latent replacement $z_t[0] \leftarrow z_{\text{cond}}[0]$ and velocity replacement $v_t[0] \leftarrow (\text{noise} - z_{\text{cond}}[0])$.
- **Verification:** Observed perfect spatial alignment at frame 0 and continuous 360° parallax rotation across all 93 frames.

### [2026-08-01] RoPE Buffer Fix & Artifact Resolver Infrastructure
- **Action:** Fixed Rotary Position Embedding (RoPE) buffer shape mismatches during checkpoint loading.
- **Implementation:**
  - Added `fix_rope_buffers(dit)` in `predict2_5/utils/torch_utils.py` to re-register RoPE buffers prior to weight loading.
  - Built `predict2_5/hf.py` and `_resolve_artifact` in `Inferencer` to support `hf://` URIs, local paths, and Cosmos UUIDs seamlessly.

### [2026-08-09] Main Method Wrapper, Test Suite, and Evaluation Harness
- **Action:** Implemented standardized `CosmosPredict25Wrapper`, main method evaluation harness `predict2_5/evaluate.py`, and integrated into `baselines/evaluate.py`.
- **Implementation:**
  - Created `predict2_5/wrapper.py` exposing `infer_multi_views(input_xray, azimuths=(0, 360, 93))` returning `[93, 1, 256, 256]` min-max normalized $[0, 1]$ tensors.
  - Re-exported wrapper in `baselines/models/cosmos.py` and `baselines/models/__init__.py`.
  - Created standalone `predict2_5/evaluate.py` calculating PSNR, SSIM, LPIPS, and Latency on the NSCLC test set.
  - Added unit test suite `predict2_5/tests/test_cosmos_wrapper.py` (3 unit tests covering initialization, fallback behavior, and tensor/array/PIL input handling).
  - Authored documentation trio (`PAPER.md`, `CODE.md`, `LOG.md`) under `docs/cosmos-predict2.5/`.
- **Verification:** Executed `pytest predict2_5/tests/test_cosmos_wrapper.py` (all 3 tests PASS). Tested `predict2_5/evaluate.py` dry-run.
