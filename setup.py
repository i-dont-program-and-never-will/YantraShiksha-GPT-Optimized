from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import sys


compile_args = ['-O3', '-std=c++17']

# Windows MSVC support
if sys.platform == "win32":
    compile_args = ['/O2', '/std:c++17']


ext_modules = [
    Extension(
        name='Math',
        sources=[
            'Math/bindings.cpp',
            'Math/storage.cpp',
            'Math/Tensor.cpp',
            'Math/Autograd.cpp'
        ],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True),
        ],
        language='c++',
        extra_compile_args=compile_args,
    ),
]


setup(
    name='Math',
    version='0.1.0',
    description='Fast tensor + autograd engine',
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires='>=3.8',
)
