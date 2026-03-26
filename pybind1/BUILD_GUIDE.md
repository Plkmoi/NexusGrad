# Build Guide for CGADImpl Python Bindings

This guide explains how to build and install the Python bindings for CGADImpl.

## Prerequisites

### Required
- **Python**: 3.8 or later
- **C++ Compiler**: 
  - MSVC 2019+ (Windows)
  - GCC 9+ (Linux)
  - Clang 10+ (macOS)
- **CMake**: 3.12 or later
- **pybind11**: 2.6.0 or later
- **Main cgadimpl library**: Must be built first

### Optional
- **CUDA Toolkit**: 11.0+ for GPU support
- **cuDNN**: For optimized GPU operations

## Step 1: Install Python and CMake

### Windows
```powershell
# Using Windows Package Manager
winget install Python.Python.3.11
winget install Kitware.CMake

# Or download from:
# https://www.python.org/downloads/
# https://cmake.org/download/
```

### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install python3 python3-dev python3-pip cmake
```

### macOS
```bash
brew install python cmake
```

## Step 2: Install pybind11

```bash
# Via pip (recommended)
pip install pybind11

# Or via package manager
# Ubuntu: sudo apt-get install pybind11-dev
# macOS: brew install pybind11
```

## Step 3: Build Main cgadimpl Library

The Python bindings require the main C++ library to be built first:

```bash
# Navigate to cgadimpl directory
cd ../cgadimpl

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake ..

# Build (Release for better performance)
cmake --build . --config Release

# Install (optional, but recommended)
cmake --install .

cd ../..
```

## Step 4: Build Python Bindings

### Using CMake (Recommended)

```bash
# From the pybind11 directory
cd pybind11

# Create build directory
mkdir -p build
cd build

# Configure
cmake ..

# Build
cmake --build . --config Release

# Run tests (optional)
cd ..
python tests/test_bindings.py
```

### Using setup.py (Alternative)

```bash
cd pybind11

# Build in-place
python setup.py build_ext --inplace

# Or install to site-packages
python setup.py install
```

## Step 5: Verify Installation

```bash
# Test import
python -c "import cgad; print(cgad.__version__)"

# Run test suite
python tests/test_bindings.py
```

## Building with CUDA Support

To enable GPU acceleration:

### Windows
```powershell
# Set CUDA_PATH (update version as needed)
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"

# Build with CUDA
cmake -DWITH_CUDA=ON -B build
cmake --build build --config Release
```

### Linux/macOS
```bash
# Set CUDA_PATH
export CUDA_PATH=/usr/local/cuda  # or appropriate path
export WITH_CUDA=ON

# Build
cmake -DWITH_CUDA=ON -B build
cmake --build build --config Release
```

## Troubleshooting

### Issue: CMake can't find pybind11

**Solution**: Ensure pybind11 is installed and visible to CMake:

```bash
pip install pybind11
cmake -DCMAKE_PREFIX_PATH=$(python -m pybind11 --cmake-prefix-path) -B build
```

### Issue: Cannot find cgadimpl library

**Solution**: Build cgadimpl first and ensure it's in the expected location:

```bash
# Check that library exists
# Windows: cgadimpl\build\Release\cgadimpl.lib
# Linux: cgadimpl/build/libcgadimpl.a or .so

# Update CMakeLists.txt if using different build location
find_library(CGADIMPL_LIB cgadimpl PATHS /path/to/library)
```

### Issue: Compiler errors with C++17

**Solution**: Ensure C++17 is enabled:

```bash
# CMake
cmake -DCMAKE_CXX_STANDARD=17 -B build

# Or manually in CMakeLists.txt
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to use" FORCE)
```

### Issue: Numpy version conflict

**Solution**: Update numpy:

```bash
pip install --upgrade numpy
```

### Issue: "Module has wrong architecture" (macOS)

**Solution**: Ensure consistency between Python and compiler architectures:

```bash
# Check Python architecture
python -c "import platform; print(platform.machine())"

# Use matching compiler
arch -x86_64 python -m pip install pybind11  # For Intel Macs
# arch -arm64e python -m pip install pybind11  # For Silicon Macs
```

## Platform-Specific Notes

### Windows
- Use Visual Studio 2019 or later for best compatibility
- May need to configure vcvars before building:
  ```powershell
  "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
  ```
- Output file: `cgad.pyd`

### Linux
- Install development headers: `sudo apt-get install python3-dev`
- May need to set `LD_LIBRARY_PATH`:
  ```bash
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/../cgadimpl/build
  python tests/test_bindings.py
  ```
- Output file: `cgad.so`

### macOS
- May need to allow execution of non-signed binaries:
  ```bash
  sudo xattr -d com.apple.quarantine cgad.so
  ```
- Use `otool` to debug linking issues:
  ```bash
  otool -L cgad.so
  ```
- Output file: `cgad.so`

## Installation Options

### Option 1: In-Place (for development)

```bash
cd pybind11/build
cmake --install . --config Release --prefix ../dist

# Add to Python path
export PYTHONPATH=$PYTHONPATH:../dist
python -c "import cgad"
```

### Option 2: System-Wide

```bash
cd pybind11/build
cmake --install . --config Release
# Installs to /usr/local or Python's site-packages
```

### Option 3: Virtual Environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\activate   # Windows PowerShell

# Build and install
pip install -e ../pybind11
```

## Next Steps

After successful installation:

1. Try the quick start example in [README.md](README.md)
2. Run example scripts from `examples/`
3. Read the API documentation
4. Build your own neural networks!

## Getting Help

- Check test files in `tests/` for usage examples
- Review docstrings: `python -c "import cgad; help(cgad.autodiff.matmul)"`
- Enable verbose build output: `cmake --build build --verbose`
- For compilation issues, check CMakeLists.txt configuration

## Additional Resources

- [pybind11 Documentation](https://pybind11.readthedocs.io/)
- [CMake Documentation](https://cmake.org/documentation/)
- [NumPy C API Guide](https://numpy.org/doc/stable/reference/c-api/index.html)
