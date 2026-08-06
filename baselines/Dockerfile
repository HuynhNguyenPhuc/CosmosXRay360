# =====================================================================
# CosmosXRay360 Unified Baselines Container
# Supports CUDA 12.8 / 13.0 + uv + PyTorch 2.x/3.x
# =====================================================================

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Avoid interactive prompts during apt install
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

# Install uv for extremely fast and robust Python packaging
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Set workspace directory
WORKDIR /workspace

# Copy requirements and project specs
COPY requirements.txt .

# Install dependencies via uv
RUN uv pip install --system -r requirements.txt

# Install specific extra requirements for baselines (e.g. pyhocon, diffdrr, torchio)
# dotmap is required by pixel-nerf's src/render/nerf.py (NeRFRenderer's DotMap output).
RUN uv pip install --system \
    pyhocon \
    diffdrr \
    torchio \
    accelerate \
    scikit-image \
    jaxtyping \
    dotmap

# Copy the entire workspace into the container
COPY . .

# Compile custom CUDA extensions (e.g. XRaySyn's drr_projector_function)
RUN cd baselines/cloned/XraySyn/xraysyn/networks/drr_projector && \
    (python3 setup.py build_ext --inplace || true)

# Expose a default command
CMD ["bash"]
