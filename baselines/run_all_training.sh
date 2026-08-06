#!/bin/bash
# CosmosXRay360 - Cross-Dataset OOD Baseline Training Automation
# Environment: uv + CUDA 13.0

set -e

EPOCHS="${1:-100}"

echo "====================================================================="
echo "🚀 COSMOS-XRAY360: UNIFIED BASELINE TRAINING PIPELINE"
echo "====================================================================="
echo "Cross-Dataset OOD Split: TCIA+MELA (Train) -> NSCLC (Test)"
echo "Target Checkpoint Directory: baselines/checkpoints/"
echo "Epochs per model: $EPOCHS"
echo "Environment Tooling: uv package manager"
echo ""

BASE_DIR="$(pwd)"
DATA_SPLIT_FILE="$BASE_DIR/datasets/cross_dataset_split.json"

if command -v uv >/dev/null 2>&1; then
    PYTHON_CMD="uv run python"
else
    PYTHON_CMD="python3"
fi

# ---------------------------------------------------------
# 0. Automated Dataset Download & Pre-rendering from HuggingFace
# ---------------------------------------------------------
echo "▶ [0/6] Checking Dataset Availability..."
PRE_RENDERED_TRAIN="$BASE_DIR/datasets/pre_rendered/train"
PRE_RENDERED_TEST="$BASE_DIR/datasets/pre_rendered/test"

if [ ! -d "$PRE_RENDERED_TRAIN" ] || [ $(ls -1 "$PRE_RENDERED_TRAIN" 2>/dev/null | wc -l) -lt 300 ] || [ ! -d "$PRE_RENDERED_TEST" ] || [ $(ls -1 "$PRE_RENDERED_TEST" 2>/dev/null | wc -l) -lt 200 ]; then
    echo "⚡ Pre-rendered dataset missing or incomplete."
    echo "📥 Downloading 100% full dataset from Hugging Face hub (hvcl-gm/chest-medical-image-dataset)..."
    $PYTHON_CMD "$BASE_DIR/scripts/fast_download.py" --percentage 1.0

    echo "⚙️ Pre-rendering 100% full dataset with DiffDRR Siddon-Jacob raymarching & standardized normalization..."
    $PYTHON_CMD "$BASE_DIR/datasets/pre_render_diffdrr.py"
else
    echo "✓ Pre-rendered dataset verified ($(ls -1 "$PRE_RENDERED_TRAIN" | wc -l) train cases, $(ls -1 "$PRE_RENDERED_TEST" | wc -l) test cases)."
fi
echo ""

# ---------------------------------------------------------
# 1. DX2CT Diffusion Training
# ---------------------------------------------------------
echo "▶ [1/6] Training DX2CT Diffusion Architecture..."
cd "$BASE_DIR"
$PYTHON_CMD baselines/train/dx2ct.py --data_split "$DATA_SPLIT_FILE" --epochs "$EPOCHS" --batch_size 2 --lr 5e-5
echo "✓ DX2CT Checkpoint Saved."
echo ""

# ---------------------------------------------------------
# 2. SV-DRR Training (Pose-Conditioned DiT)
# ---------------------------------------------------------
echo "▶ [2/6] Training SV-DRR (Pose-Conditioned DiT)..."
cd "$BASE_DIR"
$PYTHON_CMD baselines/train/svdrr.py --epochs "$EPOCHS" --lr 5e-6
echo "✓ SV-DRR Checkpoint Saved."
echo ""

# ---------------------------------------------------------
# 3. MedNeRF Training (Implicit Generative NeRF)
# ---------------------------------------------------------
echo "▶ [3/6] Training MedNeRF..."
cd "$BASE_DIR"
$PYTHON_CMD baselines/train/mednerf.py --epochs "$EPOCHS" --lr 5e-4
echo "✓ MedNeRF Checkpoint Saved."
echo ""

# ---------------------------------------------------------
# 4. NAF CBCT Training (Neural Attenuation Field)
# ---------------------------------------------------------
echo "▶ [4/6] Training NAF..."
cd "$BASE_DIR"
$PYTHON_CMD baselines/train/naf.py --epochs "$EPOCHS" --lr 1e-3
echo "✓ NAF Checkpoint Saved."
echo ""

# ---------------------------------------------------------
# 5. PixelNeRF Training (Feed-forward NeRF)
# ---------------------------------------------------------
echo "▶ [5/6] Training PixelNeRF..."
cd "$BASE_DIR"
$PYTHON_CMD baselines/train/pixelnerf.py --epochs "$EPOCHS" --lr 1e-4
echo "✓ PixelNeRF Checkpoint Saved."
echo ""

# ---------------------------------------------------------
# 6. XRaySyn Training (Voxel GAN Refinement)
# ---------------------------------------------------------
echo "▶ [6/6] Training XRaySyn..."
cd "$BASE_DIR"
$PYTHON_CMD baselines/train/xraysyn.py --epochs "$EPOCHS" --lr 1e-4
echo "✓ XRaySyn Checkpoint Saved."
echo ""

# ---------------------------------------------------------
# FINALIZATION
# ---------------------------------------------------------
echo "====================================================================="
echo "✅ All 6 baselines trained and checkpoints generated in baselines/checkpoints/:"
ls -lh "$BASE_DIR/baselines/checkpoints/"
echo "====================================================================="
