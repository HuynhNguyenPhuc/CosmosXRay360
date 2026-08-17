#!/usr/bin/env bash
# =====================================================================
# CosmosXRay360: GCP Cosmos-Predict2.5 Parallel Worker Deployment
# =====================================================================
# Spawns 5 A100-80GB GPU VMs on GCP running Cosmos-Predict2.5 workloads
# (Main Model + Ablation Variants A/B/D + Grid Search Runs) inside Docker containers.
# Automatically shuts down each VM upon completion.
# =====================================================================

set -e

PROJECT_ID="modular-ethos-468709-u4"
ZONE="us-central1-a"
TEMPLATE_NAME="cosmosxray360-a100-trainer-template"
GCS_BUCKET="graphicsminer-data-science-bucket"
DRY_RUN=false

# Concurrency & stagger settings to prevent GCS/HuggingFace rate-limiting
DOWNLOAD_STAGGER_SECONDS=120

declare -A ONLY_WORKERS

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --zone ZONE             GCP zone to deploy workers (default: us-central1-a)"
    echo "  --bucket BUCKET         GCS bucket to store output checkpoints (default: $GCS_BUCKET)"
    echo "  --only WORKER[,WORKER]  Only (re)launch these worker names, skip the rest entirely"
    echo "                          (e.g. --only cosmos-worker-a0,cosmos-worker-a1)"
    echo "  --dry-run               Print commands without executing GCP instance creation"
    echo "  --help                  Display this help message"
    exit 0
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --zone) ZONE="$2"; shift ;;
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

# Resolve HF_TOKEN to forward into worker containers
if [ -z "${HF_TOKEN:-}" ]; then
    HF_TOKEN="$(python3 -c 'from huggingface_hub.utils import get_token; import sys; t=get_token(); sys.exit(1) if not t else print(t)' 2>/dev/null || true)"
fi
if [ -z "${HF_TOKEN:-}" ] && [ "$DRY_RUN" = false ]; then
    echo "❌ Error: No HF_TOKEN available. Set the HF_TOKEN env var or run 'huggingface-cli login' locally first."
    exit 1
fi

echo "====================================================================="
echo "🚀 COSMOSXRAY360: COSMOS-PREDICT2.5 PARALLEL WORKER DEPLOYMENT"
echo "====================================================================="
echo "GCP Project:  $PROJECT_ID"
echo "Target Zone:  $ZONE"
echo "GCS Bucket:   $GCS_BUCKET"
echo "HF Token:     ${HF_TOKEN:+<set>}"
echo "====================================================================="

# Helper function to launch a worker VM
launch_worker() {
    local WORKER_NAME="$1"
    local EXPERIMENT_NAME="$2"
    local EXTRA_TRAINER_FLAGS="$3"
    local STAGGER_SECONDS="${4:-0}"

    local WORKLOAD_DESC="$EXPERIMENT_NAME ($EXTRA_TRAINER_FLAGS)"

    echo ""
    echo "---------------------------------------------------------------------"
    echo "📦 Deploying $WORKER_NAME -> Workload: [$WORKLOAD_DESC]"
    echo "---------------------------------------------------------------------"

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
    echo "▶ Cloning CosmosXRay360 repository (cosmos-predict2.5 branch)..."
    git clone --recursive -b cosmos-predict2.5 https://github.com/HuynhNguyenPhuc/CosmosXRay360.git
fi
cd CosmosXRay360
git checkout cosmos-predict2.5 || true
git pull origin cosmos-predict2.5 || true
git submodule update --init --recursive || true

# Build Docker container
echo "▶ Building Docker container..."
docker build -t cosmos_predict2_5 -f Dockerfile .

# Prepare directories
mkdir -p datasets outputs

if [ "$STAGGER_SECONDS" -gt 0 ]; then
    echo "▶ Staggering start by ${STAGGER_SECONDS}s..."
    sleep "$STAGGER_SECONDS"
fi

# Download pre-encoded VAE latent cache from GCS
echo "▶ Downloading pre-encoded VAE latent cache from GCS..."
gcloud storage cp -r "gs://$GCS_BUCKET/pre_rendered_cache/pre_rendered_latents" datasets/ || true

# Execute assigned Cosmos-Predict2.5 Training Run
echo "▶ Executing Cosmos-Predict2.5 Training ($EXPERIMENT_NAME)..."
docker run --gpus all --rm --name cosmos_training --ipc=host \
    -e HF_TOKEN="$HF_TOKEN" \
    -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    -v \$(pwd)/predict2_5:/workspace/CosmosXRay360/predict2_5 \
    -v \$(pwd)/datasets:/workspace/CosmosXRay360/datasets \
    -v \$(pwd)/outputs:/workspace/CosmosXRay360/outputs \
    cosmos_predict2_5 \
    python3 predict2_5/trainer.py \
        --num_gpus 1 \
        --use_latent_cache \
        --latent_cache_dir datasets/pre_rendered_latents/train \
        --batch_size 16 \
        --accumulate_grad_batches 1 \
        --precision bf16-mixed \
        --max_iters 10000 \
        --learning_rate 4.3158e-5 \
        --log_every_n_steps 10 \
        --output_dir outputs/ \
        --experiment_name $EXPERIMENT_NAME \
        $EXTRA_TRAINER_FLAGS

# Sync outputs and checkpoints to GCS
if command -v gcloud >/dev/null 2>&1; then
    echo "▶ Syncing outputs to gs://$GCS_BUCKET/outputs/$WORKER_NAME/..."
    gcloud storage cp -r outputs/* "gs://$GCS_BUCKET/outputs/$WORKER_NAME/" || true
fi

echo "✓ Workload completed ($WORKLOAD_DESC). Shutting down instance to save cost..."
sudo shutdown -h now
EOF

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would create VM: $WORKER_NAME"
    else
        STARTUP_SCRIPT_FILE="$(mktemp)"
        printf '%s' "$STARTUP_SCRIPT" > "$STARTUP_SCRIPT_FILE"

        CANDIDATE_ZONES=("us-central1-a" "us-central1-b" "us-central1-c" "us-central1-f" "us-east1-b" "us-west1-b" "europe-west4-a" "europe-west4-b" "asia-southeast1-a" "asia-northeast1-a")
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
                --machine-type="a2-ultragpu-1g" \
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

maybe_launch_worker() {
    local WORKER_NAME="$1"
    local EXPERIMENT_NAME="$2"
    local EXTRA_TRAINER_FLAGS="$3"
    local STAGGER_SECONDS="$4"

    if [ "${#ONLY_WORKERS[@]}" -gt 0 ] && [ -z "${ONLY_WORKERS[$WORKER_NAME]:-}" ]; then
        echo "⏭  Skipping $WORKER_NAME (not in --only list)."
        return
    fi

    launch_worker "$WORKER_NAME" "$EXPERIMENT_NAME" "$EXTRA_TRAINER_FLAGS" "$STAGGER_SECONDS"
}

# 5 Cloud Workers for Full Model + Ablations + Grid Searches
maybe_launch_worker "cosmos-worker-a0" "nvsyn_cosmos25_full_physical" "--gamma_side 1.0 --loss_atten_weight 0.02" 0
maybe_launch_worker "cosmos-worker-a1" "ablation_var_A" "--gamma_side 0.0 --loss_atten_weight 0.0" "$((DOWNLOAD_STAGGER_SECONDS * 1))"
maybe_launch_worker "cosmos-worker-a2" "ablation_var_B" "--gamma_side 1.0 --loss_atten_weight 0.0" "$((DOWNLOAD_STAGGER_SECONDS * 2))"
maybe_launch_worker "cosmos-worker-a3" "grid_search_1" "--gamma_side 2.0 --loss_atten_weight 0.01" "$((DOWNLOAD_STAGGER_SECONDS * 3))"
maybe_launch_worker "cosmos-worker-a4" "grid_search_2" "--gamma_side 1.0 --loss_atten_weight 0.05" "$((DOWNLOAD_STAGGER_SECONDS * 4))"

echo ""
echo "====================================================================="
echo "🎉 Launch requests sent! Monitor status with:"
echo "   gcloud compute instances list --filter='name~cosmos-worker-a'"
echo "====================================================================="
