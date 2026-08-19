#!/usr/bin/env bash
# =====================================================================
# Cosmos 3 (predict3) training environment setup.
#
# Installs the cosmos-framework submodule's OWN venv, entirely separate
# from this repo's main .venv used by predict2_5/ and baselines/
# (docs/cosmos-predict3/PLAN.md P0, R6 -- the two backbones must not share
# a dependency set: cosmos-framework pins CUDA 12.8/13.0 + its own
# transformers/torch versions that conflict with predict2_5's).
#
# Run this on a GCP A100 worker (or any Ampere+ CUDA 12.8/13.0 machine) --
# Cosmos 3 is BF16-only and will not run on this repo's local dev GPU.
#
# Usage:
#   bash scripts/setup_predict3_env.sh [cu130|cu128]
#
# After setup, activate with:
#   source cosmos-framework/.venv/bin/activate && export LD_LIBRARY_PATH=
# =====================================================================
set -euo pipefail

CUDA_GROUP="${1:-cu130}"
if [[ "$CUDA_GROUP" != "cu130" && "$CUDA_GROUP" != "cu128" ]]; then
    echo "Usage: $0 [cu130|cu128]" >&2
    exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COSMOS_FRAMEWORK_DIR="$ROOT_DIR/cosmos-framework"

if [[ ! -d "$COSMOS_FRAMEWORK_DIR" ]]; then
    echo "ERROR: $COSMOS_FRAMEWORK_DIR not found. Run 'git submodule update --init cosmos-framework' first." >&2
    exit 1
fi

if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if command -v apt-get &>/dev/null; then
    echo "Installing system dependencies (build-essential, gcc, g++, curl, ffmpeg, git-lfs, libx11-dev, tree, wget)..."
    sudo apt-get update && sudo apt-get install -y --no-install-recommends \
        build-essential gcc g++ curl ffmpeg git git-lfs libx11-dev tree wget ca-certificates
fi

echo "Syncing cosmos-framework training extras (--group=${CUDA_GROUP}-train)..."
cd "$COSMOS_FRAMEWORK_DIR"
uv sync --all-extras --group="${CUDA_GROUP}-train"

echo ""
echo "Done. Activate with:"
echo "  source $COSMOS_FRAMEWORK_DIR/.venv/bin/activate && export LD_LIBRARY_PATH="
echo ""
echo "Then verify with:"
echo "  python -c 'from diffusers import Cosmos3OmniPipeline'"
echo "  python -m cosmos_framework.scripts.train --help"
echo ""
echo "Set these before downloading checkpoints or launching training:"
echo "  export HF_TOKEN=<your token>          # nvidia/Cosmos3-Edge is ungated, but set it for other downloads"
echo "  export HF_HOME=<>=1TB disk path>"
echo "  export IMAGINAIRE_OUTPUT_ROOT=<>=1TB disk path>   # else training falls back to /tmp"
