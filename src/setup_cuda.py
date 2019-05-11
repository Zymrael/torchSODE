from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cudasolver',
    ext_modules=[
        CUDAExtension('pytorchodecuda', [
            'cudaSolverInterface.cpp',
            'cudaSolver.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

