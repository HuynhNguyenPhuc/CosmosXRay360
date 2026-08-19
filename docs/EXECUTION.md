# CosmosXRay360 - Baseline Training & Evaluation Execution Protocol (`EXECUTION.md`)

This document provides a comprehensive, actionable guide on how to configure, retrain, and quantitatively evaluate all six direct SOTA baseline models (**Dx2CT**, **SV-DRR**, **MedNeRF**, **NAF**, **PixelNeRF**, and **XRaySyn**) alongside our proposed **Cosmos-Predict2.5** model on the official cross-dataset OOD benchmark split.

---

## 1. Environment & Pre-Flight System Status

- **Environment Initialization:** Python 3.10.12, PyTorch 2.x, CUDA 12.x / 13.x, managed via `uv`.
- **Core Virtual Environment:**
  ```bash
  source /media/vietai/WD5/CosmosXRay360/.venv/bin/activate
  ```
- **Global Dependencies:**
  ```bash
  uv pip install pytest ninja scikit-image pyhocon diffusers transformers monai medpy nibabel
  ```
  *(Note: `ninja` supports rapid JIT compilation of CUDA C++ extensions required by baseline modules.)*
- **Hardware Accelerator:** NVIDIA TITAN RTX (24GB VRAM).
- **Pre-rendered Dataset Status:**
  - `datasets/pre_rendered/train`: 1,223 verified patient cases (TCIA + MELA2022).
  - `datasets/pre_rendered/test`: 402 verified patient cases (NSCLC-Radiomics LUNG1).

---

## 2. Baseline Model Training Matrix

| Baseline Model | Paper Reference | Paradigm | Loss Function | Learning Rate | Per-Scan Fit Iters | Default Epochs | Checkpoint Artifacts | Documentation |
|---|---|---|---|---|---|---|---|---|
| **Dx2CT** | ICASSP 2025 | Slice Diffusion on real axial CT slices | MSE Loss on DDPM noise | `5e-5` | -- | `80` | `dx2ct_best.pt`<br>`dx2ct_surrogate_checkpoint.pt` | [`docs/baselines/dx2ct/`](baselines/dx2ct/) |
| **SV-DRR** | MICCAI 2025 | Pose-conditioned DiT, random view-pair training | MSE Loss on latent noise (`epsilon`/`v_prediction`-aware) | `5e-6` | -- | `10` | `svdrr_best.pt`<br>`svdrr_checkpoint.pt` | [`docs/baselines/svdrr/`](baselines/svdrr/) |
| **MedNeRF** | IEEE EMBC 2022 | Generative Implicit NeRF, joint $z$+weight fitting | Perceptual LPIPS + MSE + NLL prior on $z$ | fixed internally (`5e-4`, not CLI flag) | `50` | `10` | `mednerf_best.pt`<br>`mednerf_checkpoint.pt` | [`docs/baselines/mednerf/`](baselines/mednerf/) |
| **NAF** | MICCAI 2022 | Neural Attenuation Field, perspective ray marching | Beer-Lambert-corrected coordinate fitting L2 Loss | `1e-3` | `200` | `10` | `naf_best.pt`<br>`naf_checkpoint.pt` | [`docs/baselines/naf/`](baselines/naf/) |
| **PixelNeRF** | CVPR 2021 | Feed-forward NeRF | Photometric NeRF rendering MSE (coarse + fine) | `1e-4` | -- | `10` | `pixelnerf_best.pt`<br>`pixelnerf_checkpoint.pt` | [`docs/baselines/pixelnerf/`](baselines/pixelnerf/) |
| **XRaySyn** | AAAI 2021 | Voxel GAN + Refinement, widened target-angle range | Adversarial + L1 + Absorption Sparsity | `1e-4` | -- | `100` | `xraysyn_best.pt`<br>`xraysyn_checkpoint.pt` | [`docs/baselines/xraysyn/`](baselines/xraysyn/) |

*Note: Per-scan fit iters applies only to the two per-patient test-time-optimization baselines (MedNeRF, NAF); the other four are feed-forward/amortized and have no such flag.*

---

## 3. Training Execution Pathways

### Option A: Parallel GCP Multi-Worker Deployment (Optimal Cloud Execution)
Deploy isolated GPU instances on GCP in parallel using `template.json` and `scripts/launch_parallel_vms.sh`. Each VM runs inside a Docker container with strict process/CUDA isolation, pairing a heavy baseline with a light/medium baseline where practical, and automatically shutting down immediately upon completion to minimize compute costs:

- **Worker 1 (`cosmos-worker-1`)**: Dx2CT (Heavy) + NAF (Light)
- **Worker 2 (`cosmos-worker-2`)**: XRaySyn only (SV-DRR moved to Worker 4's dedicated A100 once its own SV-DRR run finished)
- **Worker 3 (`cosmos-worker-3`)**: PixelNeRF (Medium) + MedNeRF (Medium)
- **Worker 4 (`cosmos-worker-4`)**: SV-DRR only, on a dedicated A100 instance template (larger batch size than the L4 default)
- **Worker 5 (`cosmos-worker-5`)**: PixelNeRF (Medium) + MedNeRF (Medium), a second slot

```bash
# Preview launch plan without executing
./scripts/launch_parallel_vms.sh --dry-run --epochs 50

# Launch the default worker set (L4 GPUs, plus worker-4's A100 template)
./scripts/launch_parallel_vms.sh --epochs 50 --zone us-central1-a
```

*Why Option A is optimal:*
1. **CUDA Context & Process Isolation**: Prevents global monkeypatched state, C++ CUDA extension conflicts, or JIT cache collisions between baselines (the same reason `run_all_tests_isolate.py` isolates tests into subprocesses).
2. **Cost Optimization**: Each VM shuts down (`sudo shutdown -h now`) the instant its assigned pair finishes, avoiding paying for idle GPUs while waiting for the slowest model.

---

### Option B: Sequential Master Pipeline (Single Machine / Docker)
Run all 6 baseline training scripts sequentially via the local/container master shell script:
```bash
# Run full training pipeline for 50 epochs per baseline
bash baselines/run_all_training.sh 50
```

---

### Option C: Individual Model Execution
For targeted retraining or hyperparameter tuning, execute models independently. Note that `--data_split` is only accepted by `train/dx2ct.py` (the other five always read `datasets/cross_dataset_split.json` internally), and `--lr` is not a valid flag for `train/mednerf.py` (its optimizer learning rates are fixed inside `fit_latent_and_weights` to match the reference implementation exactly):

```bash
# 1. Dx2CT (Slice Diffusion on real axial CT slices)
uv run python baselines/train/dx2ct.py --data_split datasets/cross_dataset_split.json --epochs 80 --batch_size 2 --slices_per_volume 8 --lr 5e-5

# 2. SV-DRR (Pose-Conditioned DiT, random view-pair training)
uv run python baselines/train/svdrr.py --epochs 10 --lr 5e-6

# 3. MedNeRF (Implicit Generative NeRF, joint z+weight fitting)
uv run python baselines/train/mednerf.py --epochs 10 --iters_per_sample 50

# 4. NAF (Neural Attenuation Field, perspective ray marching)
uv run python baselines/train/naf.py --epochs 10 --lr 1e-3 --iters_per_sample 200

# 5. PixelNeRF (Generalizable Feed-Forward NeRF)
uv run python baselines/train/pixelnerf.py --epochs 10 --lr 1e-4

# 6. XRaySyn (3D Voxel GAN Refinement, widened target-angle range)
uv run python baselines/train/xraysyn.py --epochs 100 --lr 1e-4
```

All six scripts also accept `--max_train_samples`/`--max_val_samples` (dx2ct: `--max_batches`/`--max_val_batches`) to cap the dataset for smoke tests, and MedNeRF/NAF additionally accept `--val_iters_per_sample` (default `iters_per_sample // 4`) to cut validation-time cost without affecting the training fit budget.

---

## 4. Standardizing Evaluation Rules (Zero-Bias Constraints)

To ensure mathematically fair and scientifically rigorous metrics in our central evaluation harness (`baselines/evaluate.py`), every wrapper's `infer_multi_views(input_xray, azimuths=...)` MUST adhere to the following standardization constraints:

1. **Resolution Standardization:**
   All synthesized projections must be upsampled or downsampled to exactly **$256 \times 256$ pixels** using PyTorch's high-quality bilinear interpolation with corner alignment:
   ```python
   F.interpolate(output_tensor, size=(256, 256), mode="bilinear", align_corners=True)
   ```
2. **Min-Max Intensity Normalization:**
   To align disparate output scales (e.g., raw attenuation values vs. log sigmoid activations), all outputs must undergo min-max normalization to map pixel intensities strictly within the interval $[0, 1]$ before SSIM and PSNR calculations:
   $$I_{\text{normalized}} = \frac{I - \min(I)}{\max(I) - \min(I) + \epsilon}$$
3. **VRAM Leak Prevention:**
   All inference steps must run inside the `with torch.no_grad():` context manager to disable backpropagation graph tracking and prevent VRAM allocation crashes.
4. **Azimuth Endpoint Convention:**
   Every wrapper's target-angle sweep uses `np.linspace(azimuths[0], azimuths[1], azimuths[2], endpoint=True)` (the default), so a 93-view request produces `0, 360/92, ..., 360` -- matching `datasets/pre_render_diffdrr.py`'s own `torch.linspace(0, 360, 93)` convention (and therefore the ground-truth `views/*.png` and Cosmos-Predict2.5's own training data) exactly. `endpoint=False` would instead divide the sweep into 93 *intervals*, drifting up to ~3.9° out of alignment with the ground truth by the last view.
5. **Ground-Truth Value-Space Convention (read before touching a wrapper's projection/reduction code):**
   `views/*.png` is a rescaled **raw line integral** of density (`sum(density) * step_size`, per `renderers/diffdrr/renderer.py`'s own docstring) -- there is no `exp()`/Beer-Lambert step anywhere in how it was rendered. A wrapper whose prediction is produced *independently* of that target (e.g. a generative model's forward pass, not a per-scan gradient fit against it) must match this convention directly -- applying `apply_beer_lambert_correction` before comparison is a value-space bug in that case, not a physics improvement. See `docs/GOTCHAS.md` #1's 2026-08-05 addendum; this was caught as a live regression in Dx2CT (`docs/baselines/dx2ct/LOG.md`, 2026-08-05).

---

## 5. Centralized Evaluation Execution & Metrics Logging

Once baseline model checkpoints are generated in `baselines/checkpoints/`, execute the centralized evaluation suite on the unseen `NSCLC` test set.

### A. Checkpoint Placement
Ensure baseline model checkpoints are placed in the following structure:
```
baselines/checkpoints/
├── svdrr_checkpoint.pt
├── xraysyn_checkpoint.pt
├── mednerf_checkpoint.pt
├── pixelnerf_checkpoint.pt
├── naf_checkpoint.pt
└── dx2ct_surrogate_checkpoint.pt
```

### B. Execution Command
Run the centralized evaluation harness using **`uv`**:
```bash
uv run python baselines/evaluate.py
```

This harness:
1. Loads cross-dataset test splits from `datasets/cross_dataset_split.json`.
2. Forwards the conditioning frontal CXR to each model wrapper.
3. Renders synthesized novel-view projections at $256 \times 256$ pixels.
4. Computes identical, unbiased metrics (**PSNR**, **SSIM**, **LPIPS**, and **Inference Latency**) under a single standardized evaluation loop, exporting the final quantitative benchmark table. LPIPS (`compute_lpips` in `baselines/models/utils.py`, `lpips` package, AlexNet backbone) was added 2026-08-05 alongside PSNR/SSIM specifically because pixelwise metrics alone reward blurry-safe output over sharp-but-imperfect output -- see `docs/BENCHMARK.md`'s note on this.

> **⚠️ Checkpoint staleness (as of 2026-08-05):** the checkpoints currently on disk in `baselines/checkpoints/`
> predate a batch of Phase 2/3 correctness and training-scope fixes (see `docs/BENCHMARK.md` section 2.D
> and each `docs/baselines/<name>/LOG.md`). Retrain all 6 (`baselines/train/<name>.py`) on the current code
> before treating any `baselines/evaluate.py` output as representative.

### C. Monitoring Training Progress via TensorBoard
Every `baselines/train/<name>.py` writes live scalars (`Loss/train`, `Loss/val`, per epoch) to its
own subdirectory under `baselines/checkpoints/tensorboard/`, plus two image tags:
- `Images/val_multiview` -- a real pred-vs-ground-truth multi-view panel, built by calling the
  baseline's own `infer_multi_views` (the same paper-faithful, multi-step inference path
  `baselines/evaluate.py` uses for the benchmark table) against a periodically-snapshotted
  in-memory checkpoint. Logged every `--viz_every` epochs (default 10, `--viz_views` azimuths per
  panel, default 4) -- throttled because each firing rebuilds the full inference pipeline, real GPU
  cost. See `baselines/models/viz.py` and PLAN.md for the full design (why this replaced a former
  single-step, single-view approximation that rendered as blurry/false-color noise regardless of
  actual model quality).
- `Images/val_pred_gt` -- **legacy tag, present only in runs from before this fix.** Current training
  scripts no longer write it.

For a run whose checkpoint already exists but whose event file only has the legacy single-step
panel (or none at all), regenerate a correct one without retraining:
```bash
uv run python scripts/regenerate_tb_images.py \
    --baseline svdrr --checkpoint baselines/checkpoints/svdrr_best.pt \
    --out baselines/checkpoints/tensorboard_fixed/svdrr --views 8 --patients 3
```
If a legacy event file's *only* problem is the false-color rendering (content is otherwise fine, so a
real inference re-run isn't needed), `scripts/fix_tb_event_images.py` rewrites the RGB image
summaries to grayscale in place (as a new event file) without touching scalars.

```
baselines/checkpoints/tensorboard/
├── dx2ct/
├── svdrr/
├── mednerf/
├── naf/
├── pixelnerf/
└── xraysyn/
```
To watch a single baseline while it trains:
```bash
uv run tensorboard --logdir baselines/checkpoints/tensorboard/<name> --port 6006
```
Or point at the parent directory to compare all 6 runs in one dashboard (TensorBoard groups by
subdirectory name automatically):
```bash
uv run tensorboard --logdir baselines/checkpoints/tensorboard --port 6006
```
Then open `http://localhost:6006` (add `--bind_all` and use the machine's IP instead of `localhost` if
running on a headless/remote box you're not physically at).

### D. Monitoring TensorBoard on a GCP Worker VM (`scripts/launch_parallel_vms.sh`)
Each worker VM ([[gcp-compute-engine]]) trains its 2 assigned baselines sequentially inside Docker
containers, with `baselines/checkpoints/` bind-mounted from the container to
`/workspace/CosmosXRay360/baselines/checkpoints/` on the **VM host** (`template.json`'s
`docker run -v $(pwd)/baselines/checkpoints:/workspace/baselines/checkpoints ...`) -- so
`tensorboard/<name>/` fills in live on the host filesystem exactly like a local run, even though
training happens inside a container. The template's `accessConfigs` gives each VM an external IP
(`ONE_TO_ONE_NAT`), but port 6006 isn't opened in any firewall rule, so the clean path in is an SSH
tunnel rather than opening a port or browsing the external IP directly:

**The DLVM host has neither `tensorboard` nor `pip3` on PATH** (verified 2026-08-08) -- despite the
image being marketed as a bundled ML environment, both are missing on the bare host; TensorBoard only
exists **inside** the `cosmos_baselines` Docker image. The training container also can't be reached
via `localhost:6006` on the host, because `launch_parallel_vms.sh`'s `docker run` never publishes
port 6006. The working path is: start TensorBoard *inside* the running training container, then
tunnel to that **container's own bridge IP**, not `localhost`:

1. **Find which zone the worker actually landed in** -- the launch script tries multiple candidate
   zones on stockout, so it isn't necessarily the `--zone` default:
   ```bash
   gcloud compute instances list --filter='name~cosmos-worker'
   ```
2. **Find the running training container's name** (one per worker, started with `--rm` so it only
   exists while training is in progress):
   ```bash
   gcloud compute ssh cosmos-worker-1 --zone=<zone> --project=modular-ethos-468709-u4 \
     --command="sudo docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'"
   ```
3. **Launch TensorBoard detached inside that container**, then get its bridge IP (this, not
   `localhost`, is what the tunnel must target):
   ```bash
   gcloud compute ssh cosmos-worker-1 --zone=<zone> --project=modular-ethos-468709-u4 \
     --command="sudo docker exec -d <container_name> tensorboard --logdir /workspace/baselines/checkpoints/tensorboard --port 6006 --host 0.0.0.0"

   gcloud compute ssh cosmos-worker-1 --zone=<zone> --project=modular-ethos-468709-u4 \
     --command="sudo docker inspect <container_name> --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}'"
   # -> e.g. 172.17.0.2
   ```
4. **Tunnel a local port to `<container_ip>:6006`** (not `-L <port>:localhost:6006` -- that hits the
   VM host's own unlistened port 6006 and just gets connection-reset). `-N -f` backgrounds the tunnel
   with no remote shell:
   ```bash
   gcloud compute ssh cosmos-worker-1 --zone=<zone> --project=modular-ethos-468709-u4 \
     -- -L 6006:172.17.0.2:6006 -N -f
   ```
5. Browse `http://localhost:6006` locally, same as the single-machine case in section C.

**This only works while the VM is up.** Each worker's startup script ends with `sudo shutdown -h now`
once its 2 baselines finish, specifically to avoid idle billing -- so TensorBoard access disappears
with the VM. Pull `baselines/checkpoints/` (including `tensorboard/`) via
`scripts/download_folder_from_gcs.py` from `gs://graphicsminer-data-science-bucket/checkpoints/<worker>/`
after the fact if you need the curves post-hoc rather than live.
