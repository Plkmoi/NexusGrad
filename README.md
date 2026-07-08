# NexusGrad

A from-scratch C++/CUDA deep learning framework implementing:

- custom tensor runtime
- reverse-mode autodiff
- CUDA kernels
- Flash Attention
- explicit computational graphs
- PyTorch correctness verification

---

## Core Capabilities

- ⚡ **Flash Attention** — custom CUDA implementation with memory-efficient attention kernels
- ⚡ **Automatic Differentiation** — reverse-mode autodiff with custom Vector-Jacobian Products verified against PyTorch
- ⚡ **Multi-Device Runtime** — CPU (AVX2/OpenMP) and CUDA execution with explicit device-aware tensors
- ⚡ **Transformer Components** — attention, ALiBi, RMSNorm, SWIGLU, embeddings, and KV-cache inference
- ⚡ **Kernel Optimization** — fused kernels, shared-memory optimizations, warp-level primitives, and reduction kernels
- ⚡ **Python Integration** — pybind11 bindings exposing the runtime from Python
- ⚡ **Verification Pipeline** — numerical gradient checking and PyTorch parity tests across forward and backward execution

### Scope

This repository is intended for engineers working on:
- GPU programming
- Automatic differentiation
- Deep learning runtimes
- CUDA kernel development
- Transformer implementation

---

## Quick Start: 60-Second Example

```python
import cgadimpl as cg

# Create tensors on CUDA
x = cg.Tensor.randn([2, 4], device='cuda')
w = cg.Tensor.randn([4, 3], requires_grad=True, device='cuda')

# Build computational graph
y = cg.matmul(x, w)
loss = cg.sum(y)

# Backward pass (VJP verified against PyTorch)
cg.backward(loss)

# Access gradients
print(w.grad())
```

---

## Architecture Overview

```
[Python Frontend] ──► [pybind11 Bindings]
        │
        ▼
[Computational Graph] ◄─ [C++ Runtime Engine]
   (Dynamic DAG)            │
        │                  ├─► [Custom Autograd VJP]
        │                  │    (Reverse-Mode AD)
        │                  │
        ▼                  ▼
[Layer Abstractions]    [Kernel Dispatch]
  (Linear, Attention,      │
   Embeddings, Norm,       ├─► [CPU Kernels: AVX2+OpenMP]
   Activations)            │
                           └─► [GPU Kernels: CUDA]
                                (Flash Attention, GEMM, Activations)
```

---

## Design Highlights

### 1. Flash Attention (Custom CUDA)
Memory-efficient attention that fuses softmax + GEMM in one kernel. Instead of materializing intermediate tensors, this computes attention in a single pass with shared memory optimization.

**Why?** Standard attention creates $O(n^2)$ intermediate tensors. Flash Attention reduces memory bandwidth, enabling longer sequences on limited VRAM.

### 2. Verified Autodiff
Every gradient operator (VJP) is tested via **numerical differentiation** against PyTorch. Tolerance: $atol = 10^{-5}$.

```cpp
// Example: Multiply VJP is verified
forward:  y = a * b
backward: da = grad_y * b,  db = grad_y * a

// Then: check against finite differences
// Ensures correctness before shipping to production
```

### 3. Multi-Device with Automatic Transfer
Tensors carry device metadata (CPU/CUDA). Operations automatically enforce device consistency:
- `.to(device='cuda')` — host → device
- `.to_cpu()` — device → host
- Gradients computed on the same device as values

### 4. Custom Tensor Library
Not NumPy, not CuPy—a custom tensor abstraction with:
- **AVX2 vectorization** for CPU kernels (8 floats at a time)
- **CUDA shared memory** for GPU cache optimization  
- **Broadcasting rules** matching NumPy/PyTorch conventions
- **Type casting** (float32 ↔ float64)

---

## Design Decisions

### 1. **Explicit Computational Graph**
Instead of eager execution (PyTorch default), operations construct an explicit DAG.

The explicit graph makes it possible to inspect intermediate values, profile memory use, and apply compiler-style optimizations.

### 2. **Device-Aware Design**
All tensors track their device (CPU/CUDA) and enforce correctness:

```cpp
Tensor x_gpu = x.to(device='cuda');
Tensor w_cpu = w.to_cpu();
// Gradients computed on same device as values
```

Device metadata ensures CPU/GPU consistency during graph execution.

### 3. **Verified VJP Operators**  
Each gradient operator tested against numerical differentiation:

```
For every operation: f(x)
Check: ∂f/∂x ≈ (f(x+ε) - f(x-ε)) / 2ε
```

Numerical verification helps catch gradient mismatches and unstable updates early.

### 4. **Kernel Fusion**
GPU kernels combine operations to reduce memory bandwidth:
- Flash Attention: softmax + GEMM → 1 kernel
- Activation fusion: add + ReLU → 1 kernel

Kernel fusion reduces memory traffic and can improve throughput for attention and activation-heavy workloads.

### 5. **Performance Optimization Layers**
- **CPU**: AVX2 vector instructions + OpenMP parallelization
- **GPU**: Shared memory for cache-locality, coalesced memory access, warp synchronization
- **Both**: Loop tiling and workload balancing

The implementation targets both vectorized CPU execution and tuned CUDA kernels.

---

## Implementation Summary

| Category | Technology | Evidence in Repo |
|----------|-----------|------------------|
| **GPU Computing** | CUDA 13.0 | Flash Attention, GEMM, reduction kernels |
| **CPU Optimization** | AVX2 + OpenMP | Vectorized ReLU, parallel reductions |
| **Autodiff** | Reverse-mode AD | Custom VJP for 30+ operations, numerical verification |
| **C++ Systems** | C++20, pybind11 | Full runtime, Python bindings |
| **Performance** | Kernel fusion, memory optimization | Shared memory usage, coalesced access patterns |
| **Testing** | Numerical correctness | Gradient checking vs. PyTorch, device consistency |
| **ML Theory** | Attention, activations, layer norms | Standard + ALiBi attention, Swish, Mish, etc. |

---

## Project Status

This repository represents an actively developed deep learning framework. Several components are implemented and verified against PyTorch, while others remain experimental or under active development. The project is intended to demonstrate systems design, CUDA kernel implementation, reverse-mode autodiff, and runtime architecture rather than serve as a production-ready ML framework.

## Core Components

<details>
<summary><b>📋 Full Technical Breakdown (expand to read)</b></summary>

### 1. **Tensor Library** (`tensor/`)
Self-contained tensor abstraction with multi-device support:
- **Device Management**: CPU & CUDA allocators with configurable memory strategies
- **Operations**: 
  - Arithmetic (add, sub, mul, div, pow)
  - Reductions (sum, mean, max, min across axes)
  - Unary ops (exp, log, tanh, sqrt, sin, cos, sinh, cosh)
  - Matrix multiplication with tiling strategies
- **Type System**: Support for float32, float64 with dtype casting
- **CUDA Integration**: Device-aware execution with stream support

### 2. **Autodiff Engine** (`cgadimpl/src/autodiff/`)
Custom reverse-mode automatic differentiation:
- **Graph Construction**: Dynamic computational graph built during forward pass
- **VJP Operators**: Vector-Jacobian products for all supported operations
- **Broadcasting Support**: Automatic gradient reduction for broadcast operations
- **Checkpoint System**: Memory-efficient gradient computation with checkpointing
- **In-place Operations**: Version tracking and alias detection for memory optimization
- **JVP Support**: Forward-mode differentiation for advanced use cases

**Key VJP implementations:**
- Elementwise: Add, Sub, Mul, Div with proper gradient accumulation
- Matrix Operations: Matmul (FMA), reduce operations
- Activations: ReLU, Tanh, Sigmoid, Exp, Log
- Attention: Multi-head attention with gradient flow

### 3. **Runtime & Graph Execution** (`cgadimpl/src/core/`)
- **Dynamic Graph**: Nodes, edges, and topological sort for backward traversal
- **Stream Management**: CUDA stream abstraction for async kernel execution
- **Device Transfer**: Efficient host↔device tensor movement
- **Compilation**: Trace→compile→replay for repeated execution patterns

### 4. **Layer API**
The legacy layer headers under `cgadimpl/include/layer/` are retained for compatibility, but the current main implementation for transformer-style layers lives in `arch/include/layer/` and `arch/src/layer/`.

**Legacy / deprecated layer API:**
- `cgadimpl/include/layer/` contains the older module-style abstractions
- This is no longer the primary development path

**Current main layer stack (`arch/`):**
- `Layer` base abstraction with parameter management and device movement
- `Linear`: dense projection layer with bias
- `Traverse`: sequential layer composition
- `ResidualBlock`: residual skip-connection support
- Activation layers: ReLU, LeakyReLU, Swish, Softplus, Mish, GCU, Gaussian, ParCon, Softmax
- Advanced layers: `Attention`, `AlibiAttention`, and `SWIGLU`
- Additional transformer-oriented utilities for tokenization, SafeTensors loading, and corpus processing

### 5. **Optimizer** (`cgadimpl/include/layer/optim.hpp`)
- SGD with momentum support
- Gradient-based parameter updates
- Learning rate scheduling
- Zero gradient utilities

### 6. **Kernel Backend**

#### CPU Kernels (`kernels/cpu/src/agkernels_cpu.cpp`)
AVX2 + OpenMP optimizations:
- **ReLU**: 8-float vectorized (AVX2) with OpenMP parallelization
- **Arithmetic**: Vectorized add, sub, mul, div with broadcasting
- **Reductions**: Parallel sum/mean/max across axes
- **Activations**: Vectorized implementations of standard activation functions

#### GPU Kernels (`kernels/gpu/*.cu`)
CUDA implementations for production workloads:
- **Flash Attention**: Memory-efficient attention kernel (fused forward pass)
- **GEMM**: Tiled matrix multiplication with coalesced memory access
- **Elementwise**: Fused activation operations (add+relu, mul+tanh)
- **Reductions**: Warp-level and block-level reductions
- **VJP Kernels**: Backward pass implementations for GPU operations

## High-Level Architecture Components (`arch/`)

This is the current main implementation path for transformer-oriented layers and utilities:
- **Attention Mechanisms**: Standard and ALiBi variants
- **Tokenization**: Byte-level and text file loading utilities
- **SafeTensors Support**: Model weight serialization/deserialization
- **Text Processing**: Corpus loading with configurable encoding
- **Layer Stack**: The active module interface and layer implementations used by the current architecture build

## Building

### Prerequisites
- CUDA 13.0 (set via `CUDACXX` environment variable)
- C++20 compiler (g++)
- OpenMP
- Python 3.8+
- pybind11
- CMake 3.20+

### Build Steps
```bash
# 1. Build tensor library (dependency for everything else)
cd tensor && make -j$(nproc)

# 2. Configure and build cgadimpl (core framework)
cd ../cgadimpl
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)

# 3. Build kernel plugins
cd ../kernels
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug \
    -DCGADIMPL_INCLUDE_DIR=../cgadimpl/include
cmake --build build -j$(nproc)

# 4. Build arch (high-level layers)
cd ../arch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)
```

**Or use the integrated build script:**
```bash
bash ztools/run.sh
```

## Testing

Unit tests verify correctness against PyTorch:

- **`test_layer.cpp`**: End-to-end training loop (forward, backward, SGD step)
- **`test_aliba.cpp`**: Alibi attention mechanism
- **`test_aligen.cpp`**: Standard attention layers
- **`test_aliop.cpp`**: Attention operations
- **`test_eh.cpp`**: Embedding and hash operations
- **`test_keretu.cpp`**: Kernel execution unit tests
- **`test_layalib.cpp`**: Layer API tests
- **`test_neh.cpp`**: NLP-specific operations

Run tests:
```bash
cd build && ctest --output-on-failure
```

</details>

---

## Mathematical Correctness & Verification

> **Framework reliability is paramount.** All custom Vector-Jacobian Products (VJPs) and custom CUDA kernels are rigorously verified against PyTorch's eager-mode reference implementations. The testing suite asserts that both forward-pass activations and backward-pass gradient tensors match within a strict numerical tolerance ($atol = 10^{-5}$).
>
> This multi-layered verification approach includes:
> - **Numerical gradient checking**: Custom VJP implementations validated via finite differences
> - **Tensor shape verification**: Broadcast operations tested to ensure proper gradient dimensionality
> - **Device consistency**: CPU and CUDA implementations produce identical numerical results
> - **End-to-end training**: Full training loops with loss decrease validation

## Code Structure

```
cgadimpl/
├── include/
│   ├── tensor.hpp             # Device-aware tensor wrapper
│   ├── ad/
│   │   ├── graph.hpp          # Computational graph nodes/values
│   │   ├── autodiff.hpp       # Public autodiff API
│   │   ├── ops.hpp            # Operation declarations
│   │   ├── detail/
│   │   │   └── autodiff_ops.hpp   # VJP implementations
│   │   ├── runtime.hpp        # CUDA stream management
│   │   ├── checkpoint.hpp     # Memory-efficient gradient computation
│   │   └── inplace.hpp        # In-place operation tracking
│   ├── layer/
│   │   ├── affine.hpp         # Linear, ResidualBlock, Traverse
│   │   ├── activation.hpp     # Activation layers
│   │   ├── attention.hpp      # Attention mechanisms
│   │   ├── norm.hpp           # Normalization layers
│   │   └── optim.hpp          # Optimizers
│   └── nn/nn.hpp              # Module base class
├── src/
│   ├── autodiff/
│   │   ├── autodiff_vjp_ops.cpp    # VJP operator implementations
│   │   ├── autodiff_jvp_ops.cpp    # JVP implementations
│   │   └── autodiff_nodeops.cpp    # Node-level operations
│   ├── core/
│   │   ├── autodiff.cpp       # Backward pass, zero_grad
│   │   ├── graph.cpp          # Graph construction and traversal
│   │   ├── checkpoint.cpp     # Checkpointing logic
│   │   └── ops.cpp            # Operation implementations
│   ├── layer/
│   │   ├── affine.cpp         # Layer implementations
│   │   └── activation.cpp     # Activation implementations
│   ├── kernel_stuff/
│   │   ├── kernels_loader.cpp # Dynamic kernel plugin loading
│   │   └── runtime.cpp        # Runtime initialization
│   └── tools/
│       └── binder.cpp         # pybind11 Python module definition

arch/
├── include/layer/
│   ├── activation.hpp         # Activation variants
│   ├── attention.hpp          # Attention modules
│   ├── tokeni.hpp             # Tokenization utilities
│   └── archlist.hpp           # Layer compositions
└── src/layer/
    ├── attention.cpp          # ALiBi attention implementation
    └── corpus.txt             # Example corpus for tokenization

kernels/
├── cpu/
│   ├── src/agkernels_cpu.cpp  # AVX2 + OpenMP implementations
│   └── benchmark/             # Kernel microbenchmarks
└── gpu/
    ├── entry.cu               # Kernel declarations and exports
    ├── eltwise.cu             # Elementwise operations (kernels)
    ├── mm_cublas.cu           # cuBLAS integration
    ├── vjp.cu                 # Gradient computation kernels
    └── zero.cu                # Initialization kernels

tensor/
├── include/
│   ├── TensorLib.h            # Public API
│   ├── core/Tensor.h          # Core tensor class
│   ├── device/Device.h        # Device abstraction
│   ├── dtype/Dtype.h          # Type system
│   └── ops/TensorOps.h        # Operation signatures
└── src/
    ├── core/Tensor.cpp        # Tensor implementation
    ├── device/                # CPU/CUDA allocators
    ├── Kernels/               # Kernel implementations
    └── TensorOps/             # Operation implementations
```

## Integration with Python

The framework exposes C++ via pybind11:
```python
import cgadimpl as cg

# Create tensors
x = cg.Tensor.randn([2, 4], device='cuda')
w = cg.Tensor.randn([4, 3], requires_grad=True, device='cuda')

# Build graph
y = cg.matmul(x, w)
loss = cg.sum(y)

# Backward pass
cg.backward(loss)

# Access gradients
dw = w.grad()
```

---

## Known Limitations & Future Work

1. **Distributed Training**: No multi-GPU/multi-node support yet
2. **Advanced Schedulers**: Only basic SGD with momentum
3. **Quantization**: No mixed precision or INT8 support
4. **Sparse Tensors**: Dense tensors only
5. **Dynamic Shapes**: Graph compiled at first pass, then replayed with same shape

## Debugging & Profiling

Debug utilities in `include/ad/debug.hpp`:
- `print_tensor(label, tensor)`: Pretty-print tensor contents
- `print_all_values(root)`: Dump entire graph structure
- `check_tensors_nangrad(node)`: Validate gradients are finite

## References & Credits

- VJP formulations inspired by PyTorch/JAX autodiff systems
- CUDA optimization patterns from CuDNN, TensorRT documentation
- CPU vectorization following modern SIMD best practices
