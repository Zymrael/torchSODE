from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pytorchSODE',
    ext_modules=[
        CUDAExtension('pytorchSODE', [
            'solver_interface.cpp',
            'solver_cuda.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

