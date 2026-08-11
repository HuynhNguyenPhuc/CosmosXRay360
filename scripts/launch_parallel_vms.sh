#!/usr/bin/env bash
# =====================================================================
# CosmosXRay360: GCP Parallel Worker Deployment & Baseline Training
# =====================================================================
# Spawns GPU VMs on GCP running baseline workloads inside Docker containers.
# Automatically shuts down each VM upon completion.
#
# Default Worker Pairing:
#   - Worker 1 (cosmos-worker-1): Dx2CT + NAF
#   - Worker 2 (cosmos-worker-2): XRaySyn only (SV-DRR now lives on worker-4's
#     dedicated A100 below -- worker-2 no longer pairs the two)
#   - Worker 3 (cosmos-worker-3): PixelNeRF + MedNeRF
#   - Worker 4 (cosmos-worker-4): SV-DRR only (dedicated A100 slot -- needs a
#     larger --batch_size/--accum_steps via --extra-args than the L4 default)
#   - Worker 5 (cosmos-worker-5): PixelNeRF + MedNeRF (second slot)
# =====================================================================

set -e

PROJECT_ID="modular-ethos-468709-u4"
ZONE="us-central1-a"
TEMPLATE_NAME="cosmosxray360-baseline-trainer-template"
TEMPLATE_FILE="template.json"
GCS_BUCKET="graphicsminer-data-science-bucket"
DRY_RUN=false

# Concurrency & stagger settings to prevent HuggingFace rate-limiting
DOWNLOAD_MAX_WORKERS=6
DOWNLOAD_STAGGER_SECONDS=180   # 3 min between worker starts

# Default epoch count per baseline
declare -A BASELINE_EPOCHS=(
    [dx2ct]=80
    [pixelnerf]=8
    [svdrr]=1316
    [xraysyn]=100
    [naf]=1
    [mednerf]=1
)

declare -A WORKER_OVERRIDE          # worker_name -> "baseline1[,baseline2]"
declare -A ONLY_WORKERS             # worker_name -> 1 (filter launched workers)
# Extra CLI args per baseline (appended after --epochs)
declare -A BASELINE_EXTRA_ARGS=(
    [svdrr]="--patience 50 --batch_size 4 --accum_steps 16"
)
# Instance template per worker (default: L4 template; worker-4 defaults to dedicated A100 template)
declare -A WORKER_TEMPLATE_OVERRIDE=(
    [cosmos-worker-4]="cosmosxray360-a100-trainer-template"
)

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --epochs BASELINE=NUM   Override one baseline's epoch count (repeatable),"
    echo "                          e.g. --epochs naf=2 --epochs dx2ct=20"
    echo "  --zone ZONE             GCP zone to deploy workers (default: us-central1-a)"
    echo "  --bucket BUCKET         GCS bucket to store output checkpoints (default: $GCS_BUCKET)"
    echo "  --only WORKER[,WORKER]  Only (re)launch these worker names, skip the rest entirely"
    echo "                          (e.g. --only cosmos-worker-1,cosmos-worker-2)"
    echo "  --override WORKER=BASELINE[,BASELINE2]"
    echo "                          Replace a worker's default baseline pairing -- give one"
    echo "                          baseline to run it alone (skips the second entirely), e.g."
    echo "                          --override cosmos-worker-1=dx2ct to relaunch just Dx2CT"
    echo "                          on worker-1 without redoing its already-finished NAF"
    echo "  --extra-args BASELINE=\"ARGS\""
    echo "                          Extra CLI args appended after --epochs for one baseline"
    echo "                          (repeatable), e.g. --extra-args mednerf=\"--amp\""
    echo "  --worker-template WORKER=TEMPLATE_NAME"
    echo "                          Use a non-default instance template for one worker (repeatable)"
    echo "                          -- the template must already exist (this script only"
    echo "                          auto-registers the default L4 template), e.g."
    echo "                          --worker-template cosmos-worker-4=cosmosxray360-a100-trainer-template"
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
        --only)
            IFS=',' read -ra ONLY_LIST <<< "$2"
            for w in "${ONLY_LIST[@]}"; do ONLY_WORKERS[$w]=1; done
            shift ;;
        --override)
            IFS='=' read -r OV_WORKER OV_BASELINES <<< "$2"
            WORKER_OVERRIDE[$OV_WORKER]="$OV_BASELINES"
            shift ;;
        --extra-args)
            IFS='=' read -r EA_BASELINE EA_ARGS <<< "$2"
            if [ -z "${BASELINE_EPOCHS[$EA_BASELINE]:-}" ]; then
                echo "❌ Error: unknown baseline '$EA_BASELINE' in --extra-args $2"; exit 1
            fi
            BASELINE_EXTRA_ARGS[$EA_BASELINE]="$EA_ARGS"
            shift ;;
        --worker-template)
            IFS='=' read -r WT_WORKER WT_TEMPLATE <<< "$2"
            WORKER_TEMPLATE_OVERRIDE[$WT_WORKER]="$WT_TEMPLATE"
            shift ;;
        --dry-run) DRY_RUN=true ;;
        --help) usage ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
    shift
done

# Resolve HF_TOKEN to forward into worker containers
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
    local BASELINE_2="$3"          # Optional second baseline
    local STAGGER_SECONDS="${4:-0}" # Stagger delay before download
    local EPOCHS_1="${BASELINE_EPOCHS[$BASELINE_1]}"
    local EXTRA_ARGS_1="${BASELINE_EXTRA_ARGS[$BASELINE_1]:-}"
    local WORKER_TEMPLATE="${WORKER_TEMPLATE_OVERRIDE[$WORKER_NAME]:-$TEMPLATE_NAME}"

    local WORKLOAD_DESC="$BASELINE_1 (epochs=$EPOCHS_1${EXTRA_ARGS_1:+ $EXTRA_ARGS_1})"
    local BASELINE_2_SECTION=""
    if [ -n "$BASELINE_2" ]; then
        local EPOCHS_2="${BASELINE_EPOCHS[$BASELINE_2]}"
        local EXTRA_ARGS_2="${BASELINE_EXTRA_ARGS[$BASELINE_2]:-}"
        WORKLOAD_DESC="$WORKLOAD_DESC + $BASELINE_2 (epochs=$EPOCHS_2${EXTRA_ARGS_2:+ $EXTRA_ARGS_2})"
        BASELINE_2_SECTION="
# Execute assigned Baseline 2
echo \"▶ Training Baseline 2: $BASELINE_2...\"
docker run --gpus all --rm --ipc=host \\
    -e HF_TOKEN=\"\$HF_TOKEN\" \\
    -v \$(pwd)/datasets:/workspace/datasets \\
    -v \$(pwd)/baselines/checkpoints:/workspace/baselines/checkpoints \\
    cosmos_baselines \\
    python3 baselines/train/${BASELINE_2}.py --epochs $EPOCHS_2 $EXTRA_ARGS_2
"
    fi

    echo ""
    echo "---------------------------------------------------------------------"
    echo "📦 Deploying $WORKER_NAME (template: $WORKER_TEMPLATE) -> Workload: [$WORKLOAD_DESC]"
    echo "---------------------------------------------------------------------"

    # Startup script executed by VM upon boot
    read -r -d '' STARTUP_SCRIPT << EOF || true
#!/usr/bin/env bash
set -e
exec > >(tee -a /var/log/cosmos_training.log) 2>&1

echo "====================================================================="
echo "🚀 STARTING WORKER $WORKER_NAME ($WORKLOAD_DESC)"
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

if [ "$STAGGER_SECONDS" -gt 0 ]; then
    echo "▶ Staggering download start by ${STAGGER_SECONDS}s..."
    sleep "$STAGGER_SECONDS"
fi

# Check for pre-rendered dataset cache in GCS before raw download + render
if gsutil -q ls "gs://$GCS_BUCKET/pre_rendered_cache/pre_rendered/train/" >/dev/null 2>&1; then
    echo "▶ Found pre-rendered dataset cache in GCS -- downloading directly..."
    gsutil -m cp -r "gs://$GCS_BUCKET/pre_rendered_cache/pre_rendered" datasets/
else
    echo "▶ No pre-rendered cache in GCS -- downloading raw dataset volumes..."
    docker run --gpus all --rm --ipc=host \
        -e HF_TOKEN="$HF_TOKEN" \
        -v \$(pwd)/datasets:/workspace/datasets \
        cosmos_baselines \
        python3 scripts/fast_download.py --percentage 1.0 --max_workers $DOWNLOAD_MAX_WORKERS

    echo "▶ Pre-rendering DiffDRR projections..."
    docker run --gpus all --rm --ipc=host \
        -v \$(pwd)/datasets:/workspace/datasets \
        cosmos_baselines \
        python3 datasets/pre_render_diffdrr.py

    echo "▶ Seeding pre-rendered dataset cache in GCS for future workers..."
    gsutil -m cp -r datasets/pre_rendered "gs://$GCS_BUCKET/pre_rendered_cache/" || true
fi

# Ensure raw 3D CT dataset volumes (TCIA & MELA2022) are present (needed by 3D volume tasks like Dx2CT)
if [ ! -d "datasets/TCIA" ] || [ ! -d "datasets/MELA2022" ]; then
    echo "▶ Downloading raw 3D CT dataset volumes (TCIA & MELA2022)..."
    docker run --gpus all --rm --ipc=host \
        -e HF_TOKEN="$HF_TOKEN" \
        -v \$(pwd)/datasets:/workspace/datasets \
        cosmos_baselines \
        python3 scripts/fast_download.py --percentage 1.0 --max_workers $DOWNLOAD_MAX_WORKERS
fi

# Execute assigned Baseline 1
echo "▶ Training Baseline 1: $BASELINE_1..."
# --ipc=host shares host IPC namespace to prevent PyTorch DataLoader /dev/shm OOM
docker run --gpus all --rm --ipc=host \
    -e HF_TOKEN="$HF_TOKEN" \
    -v \$(pwd)/datasets:/workspace/datasets \
    -v \$(pwd)/baselines/checkpoints:/workspace/baselines/checkpoints \
    cosmos_baselines \
    python3 baselines/train/${BASELINE_1}.py --epochs $EPOCHS_1 $EXTRA_ARGS_1

$BASELINE_2_SECTION
# Sync checkpoints to GCS if gsutil is available
if command -v gsutil >/dev/null 2>&1; then
    echo "▶ Syncing checkpoints to gs://$GCS_BUCKET/checkpoints/$WORKER_NAME/..."
    gsutil -m cp -r baselines/checkpoints/* "gs://$GCS_BUCKET/checkpoints/$WORKER_NAME/" || true
fi

echo "✓ Workload completed ($WORKLOAD_DESC). Shutting down instance to save cost..."
sudo shutdown -h now
EOF

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would create VM: $WORKER_NAME in candidate zones"
    else
        # Pass startup script via temp file to avoid gcloud comma-parsing issues
        STARTUP_SCRIPT_FILE="$(mktemp)"
        printf '%s' "$STARTUP_SCRIPT" > "$STARTUP_SCRIPT_FILE"

        # Candidate zones with L4 GPU availability to try in case of stockout
        CANDIDATE_ZONES=("us-east1-c" "us-west1-a" "us-east1-d" "us-east1-b" "us-west1-b" "us-west1-c" "us-central1-a" "us-central1-b" "us-central1-c" "europe-west1-b")
        LAUNCH_SUCCESS=false

        # First check if the VM is already running in any candidate zone
        for CAND_ZONE in "${CANDIDATE_ZONES[@]}"; do
            VM_STATUS="$(gcloud compute instances describe "$WORKER_NAME" --zone="$CAND_ZONE" --project="$PROJECT_ID" --format="value(status)" 2>/dev/null || true)"
            if [ "$VM_STATUS" = "RUNNING" ]; then
                echo "✓ Instance $WORKER_NAME is already RUNNING in zone '$CAND_ZONE'. Skipping launch."
                LAUNCH_SUCCESS=true
                break
            fi
        done

        if [ "$LAUNCH_SUCCESS" = true ]; then
            rm -f "$STARTUP_SCRIPT_FILE"
            return 0
        fi

        for CAND_ZONE in "${CANDIDATE_ZONES[@]}"; do
            echo "Attempting launch for $WORKER_NAME in zone '$CAND_ZONE'..."

            # Delete existing VM in this zone if present
            if gcloud compute instances describe "$WORKER_NAME" --zone="$CAND_ZONE" --project="$PROJECT_ID" >/dev/null 2>&1; then
                echo "Deleting existing instance $WORKER_NAME in zone $CAND_ZONE..."
                gcloud compute instances delete "$WORKER_NAME" --zone="$CAND_ZONE" --project="$PROJECT_ID" --quiet
            fi

            if gcloud compute instances create "$WORKER_NAME" \
                --source-instance-template="$WORKER_TEMPLATE" \
                --zone="$CAND_ZONE" \
                --project="$PROJECT_ID" \
                --metadata-from-file=startup-script="$STARTUP_SCRIPT_FILE"; then
                echo "✓ VM $WORKER_NAME launched successfully in zone '$CAND_ZONE'."
                LAUNCH_SUCCESS=true
                break
            else
                echo "⚠️ Stockout or failure in zone '$CAND_ZONE'. Trying next zone..."
            fi
        done

        rm -f "$STARTUP_SCRIPT_FILE"

        if [ "$LAUNCH_SUCCESS" = false ]; then
            echo "❌ Error: Could not launch $WORKER_NAME in any candidate zone."
            exit 1
        fi
    fi
}

# 3. Launch workers with optional overrides (--only, --override)
maybe_launch_worker() {
    local WORKER_NAME="$1"
    local DEFAULT_BASELINE_1="$2"
    local DEFAULT_BASELINE_2="$3"
    local STAGGER_SECONDS="$4"

    if [ "${#ONLY_WORKERS[@]}" -gt 0 ] && [ -z "${ONLY_WORKERS[$WORKER_NAME]:-}" ]; then
        echo "⏭  Skipping $WORKER_NAME (not in --only list)."
        return
    fi

    local BASELINE_1="$DEFAULT_BASELINE_1"
    local BASELINE_2="$DEFAULT_BASELINE_2"
    if [ -n "${WORKER_OVERRIDE[$WORKER_NAME]:-}" ]; then
        IFS=',' read -r BASELINE_1 BASELINE_2 <<< "${WORKER_OVERRIDE[$WORKER_NAME]}"
    fi

    launch_worker "$WORKER_NAME" "$BASELINE_1" "$BASELINE_2" "$STAGGER_SECONDS"
}

maybe_launch_worker "cosmos-worker-1" "dx2ct" "naf" 0
maybe_launch_worker "cosmos-worker-2" "xraysyn" "" "$DOWNLOAD_STAGGER_SECONDS"
maybe_launch_worker "cosmos-worker-4" "svdrr" "" "$((DOWNLOAD_STAGGER_SECONDS * 2))"
maybe_launch_worker "cosmos-worker-5" "pixelnerf" "mednerf" "$((DOWNLOAD_STAGGER_SECONDS * 3))"

echo ""
echo "====================================================================="
echo "🎉 Launch requests sent! Monitor status with:"
echo "   gcloud compute instances list --filter='name~cosmos-worker'"
echo "   gcloud compute instances get-serial-port-output cosmos-worker-1 --zone=$ZONE"
echo "====================================================================="
