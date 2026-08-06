import os

_src_path = os.path.dirname(os.path.abspath(__file__))

_backend = None

def get_backend():
    global _backend
    if _backend is None:
        from torch.utils.cpp_extension import load
        _backend = load(name='_hash_encoder',
                        extra_cflags=['-O3'],
                        extra_cuda_cflags=[
                            '-O3',
                            '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__',
                        ],
                        sources=[os.path.join(_src_path, 'src', f) for f in [
                            'hashencoder.cu',
                            'bindings.cpp',
                        ]],
                        )
    return _backend