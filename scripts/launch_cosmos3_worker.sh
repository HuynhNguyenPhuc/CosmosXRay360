#!/usr/bin/env bash
# =====================================================================
# CosmosXRay360: Cosmos3-Edge SFT launch (single worker, in-place).
#
# Runs cosmos_framework.scripts.train against
# predict3/recipes/xray360_edge.toml on a worker that ALREADY has:
#   - cosmos-framework/'s own venv set up (scripts/setup_predict3_env.sh)
#   - the Cosmos3-Edge base checkpoint converted to DCP (Step 2 of
#     cosmos-framework/docs/training.md)
#   - the Wan2.2 VAE downloaded
#   - datasets/cosmos3_sft/train/video_dataset_file.jsonl built
#     (scripts/build_cosmos3_sft_dataset.py)
#
# This script does NOT provision GCP infrastructure -- unlike
# scripts/launch_cosmos25_workers.sh, which spins up 5 GCP A100 VMs from a
# pinned instance template, there is no predict3-specific GCP template yet
# and creating cloud GPU infrastructure autonomously is out of scope here
# (real billing impact; needs an explicit, separately-reviewed template).
# Provision an a2-ultragpu-4g (4x A100-80GB) worker yourself, then run this
# script on it.
#
# Usage (from the CosmosXRay360 repo root, on the worker):
#   bash scripts/launch_cosmos3_worker.sh --smoke              # ~500-iter default recipe run
#   bash scripts/launch_cosmos3_worker.sh --max-iter 10000     # full budget, matched to
#                                                                # docs/cosmos-predict2.5/WORKER.md's step count
#
# Env vars (all optional; defaults assume the cosmos-framework docs/training.md
# Step 1/2 layout under cosmos-framework/examples/):
#   DATASET_PATH          default: datasets/cosmos3_sft (must contain train/video_dataset_file.jsonl)
#   BASE_CHECKPOINT_PATH  default: cosmos-framework/examples/checkpoints/Cosmos3-Edge
#   WAN_VAE_PATH          default: cosmos-framework/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
#   IMAGINAIRE_OUTPUT_ROOT default: outputs/train (relative to cosmos-framework/)
#   NPROC_PER_NODE        default: 4 (matches a2-ultragpu-4g; NVIDIA's stock recipe assumes 8)
# =====================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COSMOS_FRAMEWORK_DIR="$ROOT_DIR/cosmos-framework"
TOML_FILE="$ROOT_DIR/predict3/recipes/xray360_edge.toml"

MAX_ITER=""
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --max-iter) MAX_ITER="$2"; shift ;;
        --smoke) MAX_ITER="500" ;;  # matches the recipe's own default; explicit for clarity
        --help)
            echo "Usage: $0 [--max-iter N | --smoke]"
            exit 0 ;;
        *) echo "Unknown parameter: $1" >&2; exit 1 ;;
    esac
    shift
done

: "${DATASET_PATH:=$ROOT_DIR/datasets/cosmos3_sft}"
: "${BASE_CHECKPOINT_PATH:=$COSMOS_FRAMEWORK_DIR/examples/checkpoints/Cosmos3-Edge}"
: "${WAN_VAE_PATH:=$COSMOS_FRAMEWORK_DIR/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth}"
: "${IMAGINAIRE_OUTPUT_ROOT:=$COSMOS_FRAMEWORK_DIR/outputs/train}"
: "${NPROC_PER_NODE:=4}"
: "${MASTER_PORT:=50012}"

[[ -d "$COSMOS_FRAMEWORK_DIR/.venv" ]] || {
    echo "ERROR: $COSMOS_FRAMEWORK_DIR/.venv not found. Run scripts/setup_predict3_env.sh first." >&2
    exit 1
}
[[ -f "$TOML_FILE" ]] || { echo "ERROR: recipe not found: $TOML_FILE" >&2; exit 1; }
[[ -f "$DATASET_PATH/train/video_dataset_file.jsonl" ]] || {
    echo "ERROR: missing $DATASET_PATH/train/video_dataset_file.jsonl -- run scripts/build_cosmos3_sft_dataset.py first." >&2
    exit 1
}
[[ -d "$BASE_CHECKPOINT_PATH" ]] || {
    echo "ERROR: BASE_CHECKPOINT_PATH not found: $BASE_CHECKPOINT_PATH" >&2
    echo "       Convert the base checkpoint to DCP first (cosmos-framework/docs/training.md Step 2):" >&2
    echo "       python -m cosmos_framework.scripts.convert_model_to_dcp -o $BASE_CHECKPOINT_PATH --checkpoint-path Cosmos3-Edge" >&2
    exit 1
}
[[ -f "$WAN_VAE_PATH" ]] || { echo "ERROR: WAN_VAE_PATH not found: $WAN_VAE_PATH" >&2; exit 1; }

# Always-on override: exactly one PA-anchor conditioning frame (pure I2V) --
# the stock vision_sft_edge recipe mixes 70% T2V / 20% I2V / 10% V2V, which
# is wrong for this task (single-view-anchored 360-degree rotation, never
# unconditional text-to-video). See predict3/recipes/xray360_edge.toml's
# header comment and docs/cosmos-predict3/PLAN.md P4.
#
# A single dict-literal override (conditioning_config={1:1.0}) was tried
# first and is WRONG -- Hydra/OmegaConf MERGES a dict-valued override into
# the existing DictConfig node instead of replacing it, so the stock
# recipe's {0: 0.7, 1: 0.2, 2: 0.1} survives underneath and the composed
# config ends up {0: 0.7, 1: 1.0, 2: 0.1} (probabilities not even summing to
# 1). Verified via `--dryrun` against the real pydantic/Hydra schema on a
# GCP A100 worker (docs/cosmos-predict3/LOG.md, 2026-08-17 entry). The fix
# is to override each of the three existing keys individually, zeroing the
# T2V/V2V branches (a "+" prefix is required even though the keys already
# exist -- Hydra's override resolver doesn't see numeric dict keys as
# pre-existing without it, and plain "~key" deletion is rejected by this
# framework's stricter extra_overrides validation, which requires "=" in
# every entry):
TAIL_OVERRIDES=(
    '+dataloader_train.dataloader.datasets.video.dataset.conditioning_config={0:0.0,1:1.0,2:0.0}'
)
[[ -n "$MAX_ITER" ]] && TAIL_OVERRIDES+=("trainer.max_iter=$MAX_ITER")

echo ">>> $(date '+%H:%M:%S') WORKDIR:    $COSMOS_FRAMEWORK_DIR"
echo ">>> $(date '+%H:%M:%S') TOML:       $TOML_FILE"
echo ">>> $(date '+%H:%M:%S') dataset:    $DATASET_PATH"
echo ">>> $(date '+%H:%M:%S') checkpoint: $BASE_CHECKPOINT_PATH"
echo ">>> $(date '+%H:%M:%S') overrides:  ${TAIL_OVERRIDES[*]}"

cd "$COSMOS_FRAMEWORK_DIR"
source .venv/bin/activate
export LD_LIBRARY_PATH=
export DATASET_PATH BASE_CHECKPOINT_PATH WAN_VAE_PATH IMAGINAIRE_OUTPUT_ROOT
export PYTHONPATH=.

torchrun --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" \
    -m cosmos_framework.scripts.train \
    --sft-toml="$TOML_FILE" \
    -- "${TAIL_OVERRIDES[@]}"
