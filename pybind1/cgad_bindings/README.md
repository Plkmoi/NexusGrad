# CGADImpl Python Bindings

Python bindings for the CGADImpl C++ autodifferentiation framework using pybind11.

This uses the existing comprehensive pybind11 bindings from `cgadimpl/src/tools/binder.cpp`.

## Features

The bindings expose:
- **Tensor operations** - zeros, ones, randn with CPU/CUDA device support
- **Autodiff operations** - 30+ operations (add, matmul, relu, gelu, attention, etc.)
- **Value & computation graphs** - automatic differentiation support
- **Kernels** - CPU and CUDA kernel bindings
- **Debug utilities** - tracing and visualization
- **Optimizers** - SGD and gradient-based optimization

## Quick Start

### Build

```bash
cd cgad_bindings
cmake -B build
cmake --build build --config Release
```

Or using setuptools:

```bash
python setup.py build_ext --inplace
```

### Install

```bash
# Using CMake
cmake --install build

# Or using setuptools
python setup.py install
```

### Import

```python
import cgadimpl as cg

# Create tensors
x = cg.Tensor.randn([2, 3])
w = cg.Tensor.randn([3, 4])

# Automatic differentiation
v_x = cg.make_tensor(x)
y = cg.ops.matmul(v_x, w)
loss = cg.ops.sum(y)

# Backpropagation
cg.backward(loss)

# Optimization
cg.optim.SGD(loss, learning_rate=0.1)
```

## Modules

- `cgadimpl.Tensor` - Tensor class with factory methods
- `cgadimpl.Value` - Computation graph nodes
- `cgadimpl.ops` - 30+ operations (arithmetic, activations, losses, attention)
- `cgadimpl.kernels` - CPU and CUDA kernels
- `cgadimpl.optim` - Optimizers
- `cgadimpl.debug` - Debug utilities
- `cgadimpl.forward/backward` - Autodiff functions

## Testing

```bash
python tests/test_bindings.py
```

## Source

The bindings are implemented in `cgadimpl/src/tools/binder.cpp` which is compiled into a Python extension module.
