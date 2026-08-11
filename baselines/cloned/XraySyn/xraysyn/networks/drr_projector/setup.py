"""Build script for XRaySyn's CUDA DRR projector C++/CUDA extension."""

import glob
import os
import sys

from setuptools import setup
import torch.utils.cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Bypass CUDA version mismatch check between system nvcc and PyTorch wheel
torch.utils.cpp_extension._check_cuda_version = lambda *args, **kwargs: None


def _get_cuda_lib_dir() -> str | None:
    """Finds libcudart.so in active venv site-packages to set rpath."""
    site_packages = os.path.join(
        sys.prefix,
        "lib",
        f"python{sys.version_info.major}.{sys.version_info.minor}",
        "site-packages",
    )
    matches = sorted(glob.glob(os.path.join(site_packages, "nvidia", "cu*", "lib", "libcudart.so*")))
    return os.path.dirname(matches[-1]) if matches else None


cuda_lib_dir = _get_cuda_lib_dir()
extra_link_args = [f"-Wl,-rpath,{cuda_lib_dir}"] if cuda_lib_dir else []

setup(
    name="drr_projector_function",
    ext_modules=[
        CUDAExtension(
            name="drr_projector_function",
            sources=[
                "drr_projector.cpp",
                "dp_nearest_cuda.cu",
                "dp_trilinear_cuda.cu",
            ],
            extra_compile_args={
                "cxx": ["-g"],
                "nvcc": [
                    "-gencode", "arch=compute_75,code=sm_75",  # Turing (Titan RTX)
                    "-gencode", "arch=compute_89,code=sm_89",  # Ada Lovelace (GCP L4)
                ],
            },
            extra_link_args=extra_link_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
