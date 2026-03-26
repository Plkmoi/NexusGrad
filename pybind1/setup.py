#!/usr/bin/env python3
"""
setup.py for CGADImpl Python bindings

Usage:
    python setup.py build_ext --inplace
    python setup.py install
"""

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import os
from pathlib import Path

# Get the pybind11 include path
import pybind11

class get_pybind_include:
    def __str__(self):
        return pybind11.get_include()

# Determine if building with CUDA
USE_CUDA = os.getenv('WITH_CUDA', 'OFF').upper() == 'ON'
CUDA_PATH = os.getenv('CUDA_PATH', '/usr/local/cuda')

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent

# Main module extension
cgad_sources = [
    'src/cgad_pybind.cpp',
    'bindings/tensor_bindings.cpp',
    'bindings/autodiff_bindings.cpp',
    'bindings/layer_bindings.cpp',
    'bindings/optim_bindings.cpp',
]

include_dirs = [
    str(get_pybind_include()),
    str(PROJECT_ROOT / 'cgadimpl' / 'include'),
    str(PROJECT_ROOT / 'tensor' / 'include'),
]

library_dirs = [
    str(PROJECT_ROOT / 'cgadimpl' / 'build' / 'Release'),
    str(PROJECT_ROOT / 'cgadimpl' / 'build'),
]

libraries = ['cgadimpl']

if USE_CUDA:
    include_dirs.append(str(Path(CUDA_PATH) / 'include'))
    library_dirs.append(str(Path(CUDA_PATH) / 'lib64'))
    libraries.extend(['cudart', 'cublas'])

extra_compile_args = [
    '-std=c++17',
]

extra_link_args = []

if sys.platform == 'win32':
    extra_compile_args.append('/std:c++17')
elif sys.platform == 'darwin':
    extra_compile_args.append('-stdlib=libc++')

ext_modules = [
    Extension(
        'cgad',
        cgad_sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c++',
    ),
]

class BuildExt(build_ext):
    c_compiler_so = None
    
    def build_extensions(self):
        # Suppress pybind11 warnings if using MSVC
        ct = self.compiler.compiler_type
        if ct == 'msvc':
            for ext in self.extensions:
                ext.extra_compile_args.append('/W0')  # Suppress all warnings
        
        # Try to find cgadimpl library
        lib_found = False
        for lib_dir in self.compiler.library_dirs + library_dirs:
            if sys.platform == 'win32':
                lib_path = Path(lib_dir) / 'cgadimpl.lib'
            else:
                lib_path = Path(lib_dir) / 'libcgadimpl.a'
                if not lib_path.exists():
                    lib_path = Path(lib_dir) / 'libcgadimpl.so'
            
            if lib_path.exists():
                lib_found = True
                print(f"Found cgadimpl library at: {lib_path}")
                break
        
        if not lib_found:
            print("\n" + "="*60)
            print("WARNING: cgadimpl library not found!")
            print("Please build the main cgadimpl library first:")
            print("  cd ../cgadimpl")
            print("  cmake -B build")
            print("  cmake --build build --config Release")
            print("="*60 + "\n")
        
        super().build_extensions()

setup(
    name='cgad',
    version='0.1.0',
    author='CGADImpl Contributors',
    description='Python bindings for CGADImpl C++ autodiff framework',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/your-repo/cgadimpl',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.6.0', 'numpy>=1.19.0'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: C++',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
