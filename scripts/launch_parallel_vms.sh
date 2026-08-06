#!/usr/bin/env bash
# =====================================================================
# CosmosXRay360: GCP Parallel Worker Deployment & Baseline Training
# =====================================================================
# Spawns 3 isolated L4 GPU VMs on GCP, each running a paired baseline workload
# inside a Docker container. Automatically shuts down each VM upon completion
# to eliminate idle compute costs.
#
# Pairing Strategy (Heavy + Light / Medium + Medium):
#   - Worker 1 (cosmos-worker-1): Dx2CT (Heavy)    + NAF (Light)
#   - Worker 2 (cosmos-worker-2): SV-DRR (Heavy)   + XRaySyn (Light)
#   - Worker 3 (cosmos-worker-3): PixelNeRF (Med)  + MedNeRF (Med)
# =====================================================================

set -e

PROJECT_ID="modular-ethos-468709-u4"
ZONE="us-central1-a"
TEMPLATE_NAME="cosmosxray360-baseline-trainer-template"
TEMPLATE_FILE="template.json"
GCS_BUCKET="graphicsminer-data-science-bucket"
DRY_RUN=false

# Per-baseline epoch counts -- deliberately NOT uniform. Each value below is
# grounded in the ORIGINAL reference implementation under baselines/cloned/
# (paper-cited defaults, documented README training recipes, or the exact
# config/literal a real invocation uses -- not just an unused argparse
# fallback default; see docs/baselines/ history for the full per-baseline
# audit trail) and cross-checked against our own measured wall-clock cost:
#   - dx2ct: cloned/DX2CT/train.py cites "Paper default: 80" epochs directly.
#     Measured ~1 min/epoch here, so 80 epochs (~80 min) is cheap and faithful.
#   - xraysyn: no train.py ships in cloned/XraySyn at all (inference-only
#     release) -- num_epoch: 100 comes from train_opts.yaml, a metadata
#     snapshot bundled with the released checkpoint. Medium confidence (it's
#     a record, not something we can re-run), but the best evidence available.
#   - svdrr: cloned/SV-DRR/README.md's real recipe is 200,000 steps at batch
#     size 64 (num_train_epochs=1 is a dead default overridden by
#     --max_train_steps in every real invocation) -- utterly infeasible to
#     replicate at our ~1000-patient, batch-size-1 scale, so this value is a
#     pragmatic compromise, not a literal match. Our LR (5e-6) already
#     matches the README recipe exactly.
#   - pixelnerf: cloned/pixel-nerf/src/util/args.py's default_num_epochs is a
#     10,000,000 sentinel, and conf/exp/dtu.conf's num_epoch_repeats=32 is a
#     per-epoch data-repeat multiplier, not a total budget -- there is
#     genuinely no fixed target to match. Measured ~14 min/epoch here.
#   - naf: README's exact invocation (`python train.py --config
#     chest_50.yaml`) uses epoch: 3000, but that's PER-SCAN FITTING
#     ITERATIONS (maps to our --iters_per_sample=200, not this --epochs), and
#     the original has no notion of our "sweep all patients" epoch at all.
#     Measured ~5h44m per our outer epoch already; even 1 validates the path.
#   - mednerf: same story -- render_xray_G.py hardcodes 5000 iterations/patient
#     (maps to --iters_per_sample=50 here, not this --epochs). Measured ~11h
#     per our outer epoch pre-LPIPS-fix.
# NOTE: for naf/mednerf, --iters_per_sample (200/50) is itself far below the
# reference (3000/5000) -- deliberately NOT bumped to match, since doing so
# would multiply their already-multi-hour epoch cost by 15x/100x respectively
# (~86h and ~1100h per epoch). That's a wall-clock/cost decision, not a config
# correctness fix, so it's left as an explicit flagged tradeoff rather than
# applied silently.
declare -A BASELINE_EPOCHS=(
    [dx2ct]=80
    [pixelnerf]=8
    [svdrr]=6
    [xraysyn]=100
    [naf]=1
    [mednerf]=1
)

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --epochs BASELINE=NUM   Override one baseline's epoch count (repeatable),"
    echo "                          e.g. --epochs naf=2 --epochs dx2ct=20"
    echo "  --zone ZONE             GCP zone to deploy workers (default: us-central1-a)"
    echo "  --bucket BUCKET         GCS bucket to store output checkpoints (default: $GCS_BUCKET)"
    echo "  --dry-run               Print commands without executing GCP instance creation"
    echo "  --help                  Display this help message"
    echo ""
    echo "Default epochs per baseline (see script comments for rationale):"
    for k in "${!BASELINE_EPOCHS[@]}"; do echo "  $k: ${BASELINE_EPOCHS[$k]}"; done
    exit 0
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --epochs)
            IFS='=' read -r EP_BASELINE EP_VALUE <<< "$2"
            if [ -z "${BASELINE_EPOCHS[$EP_BASELINE]:-}" ]; then
                echo "❌ Error: unknown baseline '$EP_BASELINE' in --epochs $2"; exit 1
            fi
            BASELINE_EPOCHS[$EP_BASELINE]="$EP_VALUE"
            shift ;;
        --zone) ZONE="$2"; shift ;;
        --bucket) GCS_BUCKET="$2"; shift ;;
        --dry-run) DRY_RUN=true ;;
        --help) usage ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
    shift
done

# Resolve an HF token to forward into each worker's containers. Anonymous requests
# from 3 VMs hitting the same dataset repo in parallel get 429 rate-limited by HF,
# which silently starves the training dataset (and SV-DRR's HF fallback checkpoint
# download) without ever raising a hard error - so this is required, not optional.
if [ -z "${HF_TOKEN:-}" ]; then
    HF_TOKEN="$(python3 -c 'from huggingface_hub.utils import get_token; import sys; t=get_token(); sys.exit(1) if not t else print(t)' 2>/dev/null || true)"
fi
if [ -z "${HF_TOKEN:-}" ] && [ "$DRY_RUN" = false ]; then
    echo "❌ Error: No HF_TOKEN available. Set the HF_TOKEN env var or run 'huggingface-cli login' locally first."
    exit 1
fi

echo "====================================================================="
echo "🚀 COSMOSXRAY360: PARALLEL GCP WORKER DEPLOYMENT"
echo "====================================================================="
echo "GCP Project:  $PROJECT_ID"
echo "Target Zone:  $ZONE"
echo "Epochs:       $(for k in "${!BASELINE_EPOCHS[@]}"; do printf '%s=%s ' "$k" "${BASELINE_EPOCHS[$k]}"; done)"
echo "GCS Bucket:   $GCS_BUCKET"
echo "HF Token:     ${HF_TOKEN:+<set>}"
echo "====================================================================="

# 1. Ensure Instance Template exists on GCP
if [ "$DRY_RUN" = false ]; then
    echo "▶ [1/3] Registering GCP Instance Template '$TEMPLATE_NAME'..."
    if gcloud compute instance-templates describe "$TEMPLATE_NAME" --project="$PROJECT_ID" >/dev/null 2>&1; then
        echo "✓ Template '$TEMPLATE_NAME' already exists on GCP."
    else
        echo "Creating template '$TEMPLATE_NAME'..."
        gcloud compute instance-templates create "$TEMPLATE_NAME" \
            --machine-type=g2-standard-8 \
            --accelerator=type=nvidia-l4,count=1 \
            --boot-disk-size=350GB \
            --boot-disk-type=pd-ssd \
            --image-family=common-cu129-ubuntu-2204-nvidia-580 \
            --image-project=deeplearning-platform-release \
            --maintenance-policy=TERMINATE \
            --scopes=cloud-platform \
            --project="$PROJECT_ID"
        echo "✓ Registered '$TEMPLATE_NAME'."
    fi
fi

# 2. Helper function to launch a worker VM with custom startup script
launch_worker() {
    local WORKER_NAME="$1"
    local BASELINE_1="$2"
    local BASELINE_2="$3"
    local EPOCHS_1="${BASELINE_EPOCHS[$BASELINE_1]}"
    local EPOCHS_2="${BASELINE_EPOCHS[$BASELINE_2]}"

    echo ""
    echo "---------------------------------------------------------------------"
    echo "📦 Deploying $WORKER_NAME -> Workload: [$BASELINE_1 (epochs=$EPOCHS_1) + $BASELINE_2 (epochs=$EPOCHS_2)]"
    echo "---------------------------------------------------------------------"

    # Startup script executed by VM upon boot
    read -r -d '' STARTUP_SCRIPT << EOF || true
#!/usr/bin/env bash
set -e
exec > >(tee -a /var/log/cosmos_training.log) 2>&1

echo "====================================================================="
echo "🚀 STARTING WORKER $WORKER_NAME ($BASELINE_1 + $BASELINE_2)"
echo "====================================================================="

# Install docker.io and configure nvidia runtime if docker is missing
if ! command -v docker >/dev/null 2>&1; then
    echo "▶ Installing Docker and configuring NVIDIA Container Runtime..."
    apt-get update && apt-get install -y docker.io git curl
    nvidia-ctk runtime configure --runtime=docker || true
    systemctl enable --now docker
    systemctl restart docker
fi

# Clone repository
mkdir -p /workspace
cd /workspace
if [ ! -d "CosmosXRay360" ]; then
    echo "▶ Cloning CosmosXRay360 repository (baselines branch)..."
    git clone -b baselines https://github.com/HuynhNguyenPhuc/CosmosXRay360.git
fi
cd CosmosXRay360
git checkout baselines || true
git pull origin baselines || true

# Build Docker container
echo "▶ Building Docker container..."
docker build -t cosmos_baselines -f baselines/Dockerfile .

# Prepare directories
mkdir -p datasets baselines/checkpoints

echo "▶ Downloading raw dataset volumes..."
docker run --gpus all --rm \
    -e HF_TOKEN="$HF_TOKEN" \
    -v \$(pwd)/datasets:/workspace/datasets \
    cosmos_baselines \
    python3 scripts/fast_download.py --percentage 1.0

# Pre-render downloaded CT volumes into the DiffDRR projections that
# baselines/train/*.py actually reads (fast_download.py only fetches raw .nii.gz
# volumes - skipping this step leaves datasets/pre_rendered/ empty and every
# training script's DataLoader gets 0 samples).
echo "▶ Pre-rendering DiffDRR projections..."
docker run --gpus all --rm \
    -v \$(pwd)/datasets:/workspace/datasets \
    cosmos_baselines \
    python3 datasets/pre_render_diffdrr.py

# Execute assigned Baseline 1
echo "▶ Training Baseline 1: $BASELINE_1..."
docker run --gpus all --rm \
    -e HF_TOKEN="$HF_TOKEN" \
    -v \$(pwd)/datasets:/workspace/datasets \
    -v \$(pwd)/baselines/checkpoints:/workspace/baselines/checkpoints \
    cosmos_baselines \
    python3 baselines/train/${BASELINE_1}.py --epochs $EPOCHS_1

# Execute assigned Baseline 2
echo "▶ Training Baseline 2: $BASELINE_2..."
docker run --gpus all --rm \
    -e HF_TOKEN="$HF_TOKEN" \
    -v \$(pwd)/datasets:/workspace/datasets \
    -v \$(pwd)/baselines/checkpoints:/workspace/baselines/checkpoints \
    cosmos_baselines \
    python3 baselines/train/${BASELINE_2}.py --epochs $EPOCHS_2

# Sync checkpoints to GCS if gsutil/gcloud is configured
if command -v gsutil >/dev/null 2>&1; then
    echo "▶ Syncing checkpoints to gs://$GCS_BUCKET/checkpoints/$WORKER_NAME/..."
    gsutil -m cp -r baselines/checkpoints/* "gs://$GCS_BUCKET/checkpoints/$WORKER_NAME/" || true
fi

echo "✓ Workload completed ($BASELINE_1 + $BASELINE_2). Shutting down instance to save cost..."
sudo shutdown -h now
EOF

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would create VM: $WORKER_NAME in candidate zones"
    else
        # Candidate zones with L4 GPU availability to try in case of stockout
        CANDIDATE_ZONES=("us-east1-c" "us-west1-a" "us-east1-d" "us-east1-b" "us-west1-b" "us-west1-c" "us-central1-a" "us-central1-b" "us-central1-c" "europe-west1-b")
        LAUNCH_SUCCESS=false

        for CAND_ZONE in "${CANDIDATE_ZONES[@]}"; do
            echo "Attempting launch for $WORKER_NAME in zone '$CAND_ZONE'..."
            
            # Delete existing VM in this zone if present
            if gcloud compute instances describe "$WORKER_NAME" --zone="$CAND_ZONE" --project="$PROJECT_ID" >/dev/null 2>&1; then
                echo "Deleting existing instance $WORKER_NAME in zone $CAND_ZONE..."
                gcloud compute instances delete "$WORKER_NAME" --zone="$CAND_ZONE" --project="$PROJECT_ID" --quiet
            fi

            if gcloud compute instances create "$WORKER_NAME" \
                --source-instance-template="$TEMPLATE_NAME" \
                --zone="$CAND_ZONE" \
                --project="$PROJECT_ID" \
                --metadata=startup-script="$STARTUP_SCRIPT"; then
                echo "✓ VM $WORKER_NAME launched successfully in zone '$CAND_ZONE'."
                LAUNCH_SUCCESS=true
                break
            else
                echo "⚠️ Stockout or failure in zone '$CAND_ZONE'. Trying next zone..."
            fi
        done

        if [ "$LAUNCH_SUCCESS" = false ]; then
            echo "❌ Error: Could not launch $WORKER_NAME in any candidate zone."
            exit 1
        fi
    fi
}

# 3. Launch 3 parallel workers
launch_worker "cosmos-worker-1" "dx2ct" "naf"
launch_worker "cosmos-worker-2" "svdrr" "xraysyn"
launch_worker "cosmos-worker-3" "pixelnerf" "mednerf"

echo ""
echo "====================================================================="
echo "🎉 All 3 workers launched! Monitor status with:"
echo "   gcloud compute instances list --filter='name~cosmos-worker'"
echo "   gcloud compute instances get-serial-port-output cosmos-worker-1 --zone=$ZONE"
echo "====================================================================="
