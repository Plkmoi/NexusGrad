# CGADImpl Python Bindings

High-performance Python bindings for the CGADImpl C++ autodifferentiation framework using pybind11.

## Overview

This module exposes the entire CGADImpl C++ framework to Python, providing:

- **Efficient Tensors**: Fast CPU and GPU tensor operations
- **Automatic Differentiation**: Compute gradients automatically
- **Neural Network Layers**: Pre-built components for deep learning
- **Optimizers**: SGD, Adam, RMSprop, and more
- **Research-Friendly API**: PyTorch-like interface

## Building

### Prerequisites

- C++17 compiler (MSVC, GCC, or Clang)
- Python 3.8+
- CMake 3.12+
- pybind11
- The main cgadimpl library must be built first

### Build Steps

```bash
# From the pybind11 directory
cmake -B build
cmake --build build --config Release
```

The compiled module will be in `build/Release/` or `dist/` depending on configuration.

### Installation

```bash
# Install to Python site-packages
cmake --install build

# Or copy the .pyd/.so file manually to your Python path
cp cgad.pyd /path/to/site-packages/
```

## Quick Start

```python
import cgad as cg
import numpy as np

# Create tensors
x = cg.tensor.randn([2, 3])
w = cg.tensor.randn([3, 4], requires_grad=True)

# Wrap in computation graph
x_val = cg.autodiff.make_tensor(x)
w_val = cg.autodiff.param(w)

# Forward pass
y = cg.autodiff.matmul(x_val, w_val)
loss = cg.autodiff.sum(y)

# Backward pass
cg.autodiff.backward(loss)

# Access gradients
print(w.grad())  # Gradient w.r.t. w

# Optimization step
cg.optim.sgd(w_val, learning_rate=0.01)
```

## API Structure

### `cg.tensor` - Tensor Operations

- **Creation**: `zeros()`, `ones()`, `rand()`, `randn()`, `full()`
- **Manipulation**: `reshape()`, `transpose()`, `flatten()`, `view()`
- **Device**: `to_cpu()`, `to_cuda()`, `to(device)`
- **Conversion**: `from_numpy()`, `to_numpy()`

**Example:**
```python
x = cg.tensor.randn([4, 3], requires_grad=True)
x = x.reshape([12, 1])
x = x.to(cg.tensor.Device.CPU)
```

### `cg.autodiff` - Automatic Differentiation

- **Graph**: `make_tensor()`, `param()`, `constant()`
- **Operations**: `matmul()`, `add()`, `mul()`, `div()`, `sum()`
- **Activations**: `relu()`, `gelu()`, `tanh()`, `sigmoid()`, `silu()`, `mish()`
- **Attention**: `attention()`, `alibiatt()`
- **Loss**: `cross_entropy_with_logits()`, `mse_loss()`, `mae_loss()`, `kldivergence()`
- **Gradient**: `backward()`, `zero_grad()`

**Example:**
```python
x = cg.tensor.randn([10, 20])
x_val = cg.autodiff.make_tensor(x)

y = cg.autodiff.relu(x_val)
z = cg.autodiff.sum(y)

cg.autodiff.backward(z)
```

### `cg.layer` - Neural Network Layers

- **Affine**: `Linear()`, `Sequential()`, `ResidualBlock()`
- **Activations**: `ReLU()`, `GELU()`, `Tanh()`, `Sigmoid()`, etc.
- **Normalization**: `LayerNorm()`, `RMSNorm()`, `DynTanh()`
- **Attention**: `Attention()`, `AlibiAttention()`
- **Loss**: `CrossEntropyWithLogits()`, `MSELoss()`, `MAELoss()`, `KLDivergence()`
- **Embedding**: `make_embedding_table()`, `embed_tokens()`

**Example:**
```python
model = cg.layer.Sequential([
    cg.layer.Linear(784, 128),
    cg.layer.ReLU(),
    cg.layer.Linear(128, 10),
])

output = model(input_tensor)
```

### `cg.optim` - Optimizers

- **SGD**: `sgd()` with optional momentum
- **Adam**: `adam()` and `adamw()` (with weight decay)
- **RMSprop**: `rmsprop()`
- **Scheduling**: `step_decay()`, `exponential_decay()`, `cosine_annealing()`
- **Gradient Clipping**: `clip_grad_norm()`, `clip_grad_value()`

**Example:**
```python
loss = cg.autodiff.cross_entropy_with_logits(logits, targets)
cg.autodiff.backward(loss)

# SGD step
cg.optim.sgd(output_value, learning_rate=0.01)

# Or with scheduling
lr = cg.optim.step_decay(0.1, epoch, decay_epoch=10, decay_rate=0.1)
cg.optim.sgd(output_value, learning_rate=lr)
```

## Training Loop Example

```python
import cgad as cg

# Setup
model = cg.layer.Sequential([...])
optimizer = 'sgd'
num_epochs = 10
learning_rate = 0.01

# Training
for epoch in range(num_epochs):
    for batch_data, batch_labels in dataloader:
        # Convert to tensors
        x = cg.tensor.from_numpy(batch_data)
        y = cg.tensor.from_numpy(batch_labels)
        
        # Forward
        logits = model(cg.autodiff.make_tensor(x))
        loss = cg.autodiff.cross_entropy_with_logits(
            logits, cg.autodiff.make_tensor(y)
        )
        
        # Backward
        cg.autodiff.backward(loss)
        
        # Optimization
        cg.optim.sgd(logits, learning_rate=learning_rate)
        cg.autodiff.zero_grad(logits)
```

## Advanced Features

### Attention Mechanisms

```python
# Multi-head attention
attn = cg.layer.Attention(
    batch_size=32, 
    in_features=768, 
    num_heads=12, 
    out_features=768
)

# ALiBi-scaled attention
alibi_attn = cg.layer.AlibiAttention(
    batch_size=32,
    in_features=768,
    out_features=768,
    num_heads=12
)
```

### Normalization

```python
# Layer normalization
norm = cg.layer.LayerNorm([768])

# RMS normalization  
rms_norm = cg.layer.RMSNorm(768)

# Dynamic tanh
dyn_tanh = cg.layer.DynTanh(alpha=0.5, beta=1.0, gamma=2.0)
```

### State Space Models

```python
# Mamba block
x = cg.autodiff.make_tensor(input_tensor)
y = cg.autodiff.mambassm(x)

# Mixture of Experts
y = cg.autodiff.moewe(x, num_experts=8, expert_capacity=64)
```

## Numpy Interoperability

```python
import numpy as np

# Numpy to tensor
np_array = np.random.randn(3, 4)
tensor = cg.tensor.from_numpy(np_array)

# Tensor to numpy
result = cg.tensor.to_numpy(tensor)
print(result.shape)
```

## Device Management

```python
# CPU operations
x = cg.tensor.randn([10, 10], device=cg.tensor.Device.CPU)

# GPU operations (if compiled with CUDA)
x_gpu = x.to_cuda()
x_gpu = x.to(cg.tensor.Device.CUDA)

# Back to CPU
x_cpu = x_gpu.to_cpu()
```

## Testing

Run the test suite:

```bash
python tests/test_bindings.py
```

## File Structure

```
pybind11/
├── CMakeLists.txt              # Build configuration
├── src/
│   └── cgad_pybind.cpp         # Main module definition
├── bindings/
│   ├── tensor_bindings.cpp     # Tensor API bindings
│   ├── autodiff_bindings.cpp   # Autodiff API bindings
│   ├── layer_bindings.cpp      # Layer API bindings
│   └── optim_bindings.cpp      # Optimizer bindings
└── tests/
    └── test_bindings.py        # Python test suite
```

## Troubleshooting

### "Module not found" Error

Ensure the compiled `.pyd` (Windows) or `.so` (Linux) file is in Python's path:

```python
import sys
sys.path.insert(0, '/path/to/pybind11/dist')
import cgad
```

### Gradient is None

Ensure `requires_grad=True` when creating tensors:

```python
x = cg.tensor.randn([10, 10], requires_grad=True)  # ✓
y = cg.tensor.randn([10, 10])  # ✗ No gradient
```

### CUDA Not Available

The module was compiled without CUDA support. Rebuild with:

```bash
cmake -DWITH_CUDA=ON -B build
cmake --build build --config Release
```

## Performance Tips

1. Use `requires_grad=False` for data that doesn't need gradients
2. Call `zero_grad()` after each training step
3. Use batch operations (matmul, attention) instead of element-wise ops
4. Move to GPU for large workloads if available
5. Use `contiguous()` before heavy operations if needed

## Contributing

Report issues and contribute improvements at the main repository.

## License

Same as the main CGADImpl project.
