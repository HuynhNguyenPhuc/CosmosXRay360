#!/usr/bin/env bash
# =====================================================================
# CosmosXRay360: GCP Parallel Dataset Download + Pre-Render Deployment
# =====================================================================

set -e

PROJECT_ID="modular-ethos-468709-u4"
TEMPLATE_NAME="cosmosxray360-baseline-trainer-template"
GCS_BUCKET="graphicsminer-data-science-bucket"
DRY_RUN=false
DOWNLOAD_MAX_WORKERS=6
STAGGER_SECONDS=180

declare -A ONLY_WORKERS

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --bucket BUCKET         GCS bucket to seed pre_rendered_cache into (default: $GCS_BUCKET)"
    echo "  --only WORKER[,WORKER]  Only (re)launch these worker names, skip the rest"
    echo "                          (e.g. --only cosmos-render-nsclc)"
    echo "  --dry-run               Print what would be launched without creating instances"
    echo "  --help                  Display this help message"
    exit 0
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --bucket) GCS_BUCKET="$2"; shift ;;
        --only)
            IFS=',' read -ra ONLY_LIST <<< "$2"
            for w in "${ONLY_LIST[@]}"; do ONLY_WORKERS[$w]=1; done
            shift ;;
        --dry-run) DRY_RUN=true ;;
        --help) usage ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
    shift
done

if [ -z "${HF_TOKEN:-}" ]; then
    HF_TOKEN="$(python3 -c 'from huggingface_hub.utils import get_token; import sys; t=get_token(); sys.exit(1) if not t else print(t)' 2>/dev/null || true)"
fi
if [ -z "${HF_TOKEN:-}" ] && [ "$DRY_RUN" = false ]; then
    echo "❌ Error: No HF_TOKEN available. Set the HF_TOKEN env var or run 'huggingface-cli login' locally first."
    exit 1
fi

echo "====================================================================="
echo "🚀 COSMOSXRAY360: PARALLEL GCP DOWNLOAD + PRE-RENDER DEPLOYMENT"
echo "====================================================================="
echo "GCP Project:  $PROJECT_ID"
echo "GCS Bucket:   $GCS_BUCKET"
echo "HF Token:     ${HF_TOKEN:+<set>}"
echo "====================================================================="

if [ "$DRY_RUN" = false ]; then
    echo "▶ [1/2] Registering GCP Instance Template '$TEMPLATE_NAME'..."
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

launch_render_worker() {
    local WORKER_NAME="$1"
    local DATASET="$2"             # TCIA | MELA2022 | NSCLC
    local STAGGER="${3:-0}"

    echo ""
    echo "---------------------------------------------------------------------"
    echo "📦 Deploying $WORKER_NAME -> download + render partition: $DATASET"
    echo "---------------------------------------------------------------------"

    read -r -d '' STARTUP_SCRIPT << EOF || true
#!/usr/bin/env bash
set -e
exec > >(tee -a /var/log/cosmos_render.log) 2>&1

echo "====================================================================="
echo "🚀 STARTING RENDER WORKER $WORKER_NAME (dataset=$DATASET)"
echo "====================================================================="

if ! command -v docker >/dev/null 2>&1; then
    echo "▶ Installing Docker and configuring NVIDIA Container Runtime..."
    apt-get update && apt-get install -y docker.io git curl
    nvidia-ctk runtime configure --runtime=docker || true
    systemctl enable --now docker
    systemctl restart docker
fi

mkdir -p /workspace
cd /workspace
if [ ! -d "CosmosXRay360" ]; then
    echo "▶ Cloning CosmosXRay360 repository (baselines branch)..."
    git clone -b baselines https://github.com/HuynhNguyenPhuc/CosmosXRay360.git
fi
cd CosmosXRay360
git checkout baselines || true
git pull origin baselines || true

echo "▶ Building Docker container..."
docker build -t cosmos_baselines -f baselines/Dockerfile .

mkdir -p datasets

if [ "$STAGGER" -gt 0 ]; then
    echo "▶ Staggering download start by ${STAGGER}s to avoid HF rate-limit collision..."
    sleep "$STAGGER"
fi

echo "▶ Downloading $DATASET raw CT volumes from Hugging Face..."
docker run --gpus all --rm --ipc=host \
    -e HF_TOKEN="\$HF_TOKEN" \
    -v \$(pwd)/datasets:/workspace/datasets \
    cosmos_baselines \
    python3 scripts/fast_download.py --percentage 1.0 --datasets $DATASET --max_workers $DOWNLOAD_MAX_WORKERS

echo "▶ Pre-rendering $DATASET DiffDRR projections (fixed dataset-wide scaling)..."
docker run --gpus all --rm --ipc=host \
    -v \$(pwd)/datasets:/workspace/datasets \
    cosmos_baselines \
    python3 datasets/pre_render_diffdrr.py --datasets $DATASET

echo "▶ Uploading rendered $DATASET partition to gs://$GCS_BUCKET/pre_rendered_cache/..."
gsutil -m cp -r datasets/pre_rendered "gs://$GCS_BUCKET/pre_rendered_cache/" || true

echo "✓ Render worker $WORKER_NAME completed ($DATASET). Shutting down to save cost..."
sudo shutdown -h now
EOF

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would create VM: $WORKER_NAME (dataset=$DATASET, stagger=${STAGGER}s) in candidate zones"
        return 0
    fi

    STARTUP_SCRIPT_FILE="$(mktemp)"
    printf '%s' "$STARTUP_SCRIPT" > "$STARTUP_SCRIPT_FILE"

    CANDIDATE_ZONES=("us-east1-c" "us-west1-a" "us-east1-d" "us-east1-b" "us-west1-b" "us-west1-c" "us-central1-a" "us-central1-b" "us-central1-c" "europe-west1-b")
    LAUNCH_SUCCESS=false

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

        if gcloud compute instances describe "$WORKER_NAME" --zone="$CAND_ZONE" --project="$PROJECT_ID" >/dev/null 2>&1; then
            echo "Deleting existing instance $WORKER_NAME in zone $CAND_ZONE..."
            gcloud compute instances delete "$WORKER_NAME" --zone="$CAND_ZONE" --project="$PROJECT_ID" --quiet
        fi

        if gcloud compute instances create "$WORKER_NAME" \
            --source-instance-template="$TEMPLATE_NAME" \
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
}

maybe_launch_render_worker() {
    local WORKER_NAME="$1"
    local DATASET="$2"
    local STAGGER="$3"

    if [ "${#ONLY_WORKERS[@]}" -gt 0 ] && [ -z "${ONLY_WORKERS[$WORKER_NAME]:-}" ]; then
        echo "⏭  Skipping $WORKER_NAME (not in --only list)."
        return
    fi

    launch_render_worker "$WORKER_NAME" "$DATASET" "$STAGGER"
}

echo "▶ [2/2] Launching render workers..."
maybe_launch_render_worker "cosmos-render-tcia"  "TCIA"     0
maybe_launch_render_worker "cosmos-render-mela"  "MELA2022" "$STAGGER_SECONDS"
maybe_launch_render_worker "cosmos-render-nsclc" "NSCLC"    "$((STAGGER_SECONDS * 2))"

echo ""
echo "====================================================================="
echo "🎉 Launch requests sent! Monitor status with:"
echo "   gcloud compute instances list --filter='name~cosmos-render'"
echo "   gcloud compute instances get-serial-port-output cosmos-render-tcia --zone=<zone>"
echo ""
echo "Once all 3 finish (self-shutdown) and gs://$GCS_BUCKET/pre_rendered_cache/pre_rendered/"
echo "is fully populated, run scripts/launch_parallel_vms.sh to start baseline training --"
echo "each training worker will find the cache and skip its own download+render step."
echo "====================================================================="
