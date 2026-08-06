import os
import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# The pip-installed `cuda-toolkit[nvcc]` runtime (nvidia-cuda-runtime) only ships
# libcudart.so.<major>, not a dev-time `-lcudart`-resolvable symlink or a system
# ldconfig entry, so without an explicit rpath the built extension can only be
# imported when LD_LIBRARY_PATH happens to be set. Baking the rpath in here makes
# it load correctly under normal invocations (e.g. `baselines/evaluate.py`).
# Uses sys.prefix (not os.__file__, whose stdlib may be shared with the system
# Python rather than living under the venv) so this resolves correctly even
# though the extension is built via `python setup.py build_ext`, not `uv pip install`.
_cuda_lib_dir = os.path.join(
    sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}",
    "site-packages", "nvidia", "cu13", "lib",
)

setup(
    name='drr_projector_function',
    ext_modules=[
        CUDAExtension('drr_projector_function', [
            'drr_projector.cpp',
            'dp_nearest_cuda.cu',
            'dp_trilinear_cuda.cu',
        ], extra_compile_args={
            'cxx': ['-g'],
            'nvcc': [
                # CUDA 13.0's nvcc dropped support for compute_61 (Pascal) and
                # compute_70 (Volta) -- `nvcc --list-gpu-arch` on this toolkit
                # starts at compute_75 (Turing), which also matches the Titan
                # RTX this project actually runs on.
                "-gencode", "arch=compute_75,code=sm_75"]
        }, extra_link_args=[f"-Wl,-rpath,{_cuda_lib_dir}"])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
