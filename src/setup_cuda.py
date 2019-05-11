from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pySODE',
    ext_modules=[
        CUDAExtension('pySODE', [
            'euler_solver_interface.cpp',
            'euler_solver_cuda.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

