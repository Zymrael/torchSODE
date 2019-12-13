from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='torchSODE',
    version='0.0.1',
    ext_modules=[
        CUDAExtension(name='torchSODE', 
            sources=[
                'solver_interface.cpp',
                'solver.cu',
            ]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

"""

            extra_link_args={
                'nvcc': [
                    '-rdc=true',
                    '-lcudart',
                    '-lcudadevrt'
                ]
            },
            extra_compile_args={
                'cxx': ['-lcudart -lcudadevrt -rdc=true'],
                'nvcc': [
                    '-arch=sm_50', '-gencode=arch=compute_50,code=sm_50',
                    '-gencode=arch=compute_52,code=sm_52',
                    '-gencode=arch=compute_60,code=sm_60',
                    '-gencode=arch=compute_61,code=sm_61',
                    '-gencode=arch=compute_70,code=sm_70',
                    '-gencode=arch=compute_70,code=compute_70',
                    '-rdc=true',
                    '-lcudart',
                    '-lcudadevrt'
                ]
            }
"""
