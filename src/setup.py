from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='torchSODE',
    ext_modules=[
        CUDAExtension('torchSODE', [
            'solver_interface.cpp',
            'solver.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

