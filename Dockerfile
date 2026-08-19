# ==============================================================================
# CosmosXRay360 Multi-Stage Dockerfile
# ==============================================================================
# This Dockerfile provides isolated environments for two independent pipelines:
#
#   1. predict3   : Cosmos 3 (cosmos-framework) training & inference.
#   2. predict2_5 : Cosmos Predict 2.5 training & inference (DEFAULT target).
#
# Usage Examples:
#   # Build default target (predict2_5):
#   docker build -t cosmos_predict2_5 .
#
#   # Build Cosmos 3 target explicitly:
#   docker build -t cosmos_predict3 --target predict3 .
# ==============================================================================


# ==============================================================================
# STAGE 1: predict3 (Cosmos 3 / cosmos-framework)
# ==============================================================================
FROM nvidia/cuda:13.0.3-devel-ubuntu22.04 AS predict3

# Environment Configuration
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:${PATH}" \
    PYTHONPATH="/workspace/CosmosXRay360:/workspace/CosmosXRay360/cosmos-framework" \
    LD_LIBRARY_PATH=""

# 1. Install System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    git-lfs \
    libx11-dev \
    tree \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Astral uv Package Manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /workspace/CosmosXRay360

# 3. Copy & Sync cosmos-framework Submodule Dependencies (Cached Layer)
COPY cosmos-framework/ cosmos-framework/
RUN cd cosmos-framework && uv sync --all-extras --group=cu130-train

# 4. Copy Application Code & Scripts
COPY predict3/ predict3/
COPY scripts/ scripts/

CMD ["bash"]


# ==============================================================================
# STAGE 2: predict2_5 (Cosmos Predict 2.5 - Default Target)
# ==============================================================================
# Kept as the last stage so `docker build .` without --target defaults here.
FROM nvidia/cuda:13.0.3-devel-ubuntu22.04 AS predict2_5

# Environment Configuration
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:${PATH}" \
    PYTHONPATH="/workspace/CosmosXRay360"

# 1. Install System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    git-lfs \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ninja-build \
    python3-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Astral uv Package Manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /workspace/CosmosXRay360

# 3. Install Python Base Requirements
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# 4. Install Additional Model & Baseline Dependencies
RUN uv pip install --system \
    accelerate \
    diffdrr \
    dotmap \
    editables \
    hatchling \
    jaxtyping \
    pyhocon \
    scikit-image \
    torchio

# 5. Copy Full Repository Code
COPY . .

# 6. Install cosmos-predict2.5 Submodule (Editable Mode)
RUN cd cosmos-predict2.5 && uv pip install --system -e ".[cu128]" && cd ..

CMD ["bash"]
