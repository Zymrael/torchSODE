from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cudasolv',
    ext_modules=[
        CUDAExtension('solver', [
            'cudaSolverInterface.cpp',
            'cudaSolver.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

