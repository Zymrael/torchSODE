from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(name='pytorchodecpp',
      ext_modules=[CppExtension('pytorchode', ['solver_cpp.cpp'])],
      cmdclass={'build_ext': BuildExtension})
