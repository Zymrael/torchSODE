from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pySODE',
    ext_modules=[
        CUDAExtension('pySODE', [
            'solver_interface.cpp',
            'solver.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

