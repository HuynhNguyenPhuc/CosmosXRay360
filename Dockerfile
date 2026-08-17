# =====================================================================
# CosmosXRay360 Cosmos-Predict2.5 Training & Evaluation Container
# =====================================================================
FROM nvidia/cuda:13.0.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-venv \
    git \
    git-lfs \
    build-essential \
    cmake \
    ninja-build \
    curl \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python packaging
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /workspace/CosmosXRay360

# Copy requirements and install
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# Install extra dependencies required by predict2_5 & baselines
RUN uv pip install --system \
    pyhocon \
    diffdrr \
    torchio \
    accelerate \
    scikit-image \
    jaxtyping \
    dotmap \
    hatchling \
    editables

# Copy full repository
COPY . .

# Install cosmos-predict2.5 submodule in editable mode
RUN cd cosmos-predict2.5 && uv pip install --system -e ".[cu128]" && cd ..

ENV PYTHONPATH=/workspace/CosmosXRay360

CMD ["bash"]
