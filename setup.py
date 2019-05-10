from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(name='pytorch_ode.cpp',
      ext_modules=[CppExtension('pytorch_ode', ['pytorch_ode.cpp'])],
      cmdclass={'build_ext': BuildExtension})
