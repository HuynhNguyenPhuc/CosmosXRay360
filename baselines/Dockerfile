# =====================================================================
# CosmosXRay360 Unified Baselines Container
# uv + PyTorch 2.x/3.x
# =====================================================================
#
# Base image CUDA version must match whatever `torch` build `uv pip install`
# actually resolves below, not just be "close enough" -- torch.utils.cpp_
# extension.CUDAExtension hard-fails custom CUDA extension builds (XRaySyn's
# drr_projector_function, and NAF's hashencoder at its own runtime JIT-compile
# import) on any nvcc/torch CUDA version mismatch, even a minor-version one.
# This was 12.8.0 until 2026-08-06: `requirements.txt` doesn't pin torch, and
# unpinned `uv pip install` now resolves a cu130-tagged torch wheel (matching
# the local dev machine too), which raised `RuntimeError: The detected CUDA
# version (12.8) mismatches the version that was used to compile PyTorch
# (13.0)` and silently failed BOTH extensions on all 3 GCP workers -- see
# docs/baselines/xraysyn/LOG.md and docs/baselines/naf/LOG.md's 2026-08-06
# entries. If `requirements.txt` ever pins torch to a specific CUDA build
# again, re-check this line matches it exactly, not just approximately.
FROM nvidia/cuda:13.0.3-devel-ubuntu22.04

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

# Compile custom CUDA extensions (e.g. XRaySyn's drr_projector_function). This
# is allowed to fail without failing the whole image build -- XRaySynWrapper
# has a real fallback path (an all-zero-projector stub) for exactly this case,
# matching this project's other optional-CUDA-extension baselines (e.g. NAF's
# hashencoder) -- but a *silent* failure here is a real correctness hazard: on
# 2026-08-06, this line was a bare `(... || true)` with no output check at all,
# so a broken build (a hardcoded rpath from setup.py that didn't exist in this
# image) silently produced the all-zero stub on all 3 GCP training workers with
# no visible signal anywhere in the build log. Never swallow this without at
# least a loud, unmissable banner -- see docs/baselines/xraysyn/LOG.md's
# 2026-08-06 entry for the full incident.
RUN cd baselines/cloned/XraySyn/xraysyn/networks/drr_projector && \
    if python3 setup.py build_ext --inplace; then \
        echo "✓ drr_projector_function CUDA extension built successfully."; \
    else \
        echo "############################################################"; \
        echo "# WARNING: drr_projector_function CUDA extension build FAILED."; \
        echo "# XRaySyn will silently fall back to an all-zero-projector stub"; \
        echo "# at runtime -- its training/inference will be a no-op unless"; \
        echo "# this is fixed. See docs/baselines/xraysyn/LOG.md."; \
        echo "############################################################"; \
    fi

# Expose a default command
CMD ["bash"]
