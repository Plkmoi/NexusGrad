#!/usr/bin/env python3
"""
setup.py for CGADImpl Python bindings

Usage:
    python setup.py build_ext --inplace
    python setup.py install
"""

from setuptools import setup, Extension, find_packages
import sys
import setuptools
import os
from pathlib import Path

# Get the pybind11 include path
try:
    import pybind11
except ImportError:
    print("ERROR: pybind11 not found. Install it with: pip install pybind11")
    sys.exit(1)

def get_pybind_include():
    return pybind11.get_include()

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
    get_pybind_include(),
    str(PROJECT_ROOT / 'cgadimpl' / 'include'),
    str(PROJECT_ROOT / 'tensor' / 'include'),
]

library_dirs = [
    str(PROJECT_ROOT / 'cgadimpl' / 'build' / 'Release'),
    str(PROJECT_ROOT / 'cgadimpl' / 'build'),
]

libraries = ['cgadimpl']

extra_compile_args = ['-std=c++17']
extra_link_args = []

if sys.platform == 'win32':
    extra_compile_args = ['/std:c++17']
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

setup(
    name='cgad',
    version='0.1.0',
    author='CGADImpl Contributors',
    description='Python bindings for CGADImpl C++ autodiff framework',
    long_description=open('README.md').read() if Path('README.md').exists() else '',
    long_description_content_type='text/markdown',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.6.0', 'numpy>=1.19.0'],
    zip_safe=False,
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
