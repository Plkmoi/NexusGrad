# Computational Graph & Autodifferentiation Framework (CGAD)

A production-grade C++/CUDA deep learning framework from scratch, featuring a custom autodiff engine, computational graph compilation, and optimized kernel implementations for efficient neural network training and inference.

---

## 🎯 Overview

CGAD is a systems engineering project implementing the full stack of a neural network runtime: from Python-facing APIs (via pybind11) down through a C++ execution engine, custom autograd (VJP), and CUDA kernel execution. The framework demonstrates deep hardware awareness and memory optimization techniques essential to modern ML systems.

**Why build from scratch?** To understand—not abstract away—the bottlenecks that matter in production ML systems:
- How computational graphs lower to GPU memory hierarchies
- Where kernel fusion saves memory bandwidth
- How checkpointing trades recomputation for memory efficiency
- Why autograd correctness verification matters

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│               Python Frontend (pybind11)                         │
│         Model definition, training loops, validation             │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│          C++ Runtime Engine (cgadimpl_core)                      │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │   Computational Graph Representation                     │   │
│  │   - Node (compute vertex), Value (tensor wrapper)       │   │
│  │   - Op enum (250+ operations)                           │   │
│  │   - Topological sort for execution order                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           │                                      │
│  ┌────────────────────────▼─────────────────────────────────┐   │
│  │   Autograd Engine (Forward & Backward)                  │   │
│  │   - Vector-Jacobian Product (VJP) for backprop          │   │
│  │   - Jacobian-Vector Product (JVP) for forward mode      │   │
│  │   - Op-specific gradient implementations (autodiff_ops) │   │
│  └────────────────────────▼─────────────────────────────────┘   │
│                           │                                      │
│  ┌────────────────────────▼─────────────────────────────────┐   │
│  │   Memory & Execution Optimization                       │   │
│  │   - Activation Checkpointing (trade compute for memory) │   │
│  │   - In-place Operations (versioning + alias tracking)   │   │
│  │   - CUDA Graph Capture (reduce kernel launch overhead)  │   │
│  │   - Careful Deletion (safe memory reclamation)          │   │
│  └────────────────────────▼─────────────────────────────────┘   │
│                           │                                      │
│  ┌────────────────────────▼─────────────────────────────────┐   │
│  │   Neural Network Layers & Operators                     │   │
│  │   - Linear, Affine, RMSNorm, LayerNorm                  │   │
│  │   - Multi-Head Attention, ALiBi Attention               │   │
│  │   - Activation Functions (ReLU, GELU, SwiGLU, etc.)    │   │
│  │   - Fusion ops (fused attention, fused linear)          │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│          Tensor Library + Device Abstraction                     │
│  - Device-aware memory allocation (CPU/CUDA)                    │
│  - Dtype support (Float32, Float64, Int32, etc.)               │
│  - Strided tensor views (cache-friendly indexing)              │
│  - Allocator registry (extensible to other accelerators)       │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│          Kernel Implementation Layer                             │
│                                                                   │
│  ┌────────────────────────┐  ┌─────────────────────────────┐    │
│  │    CPU Kernels         │  │   CUDA/GPU Kernels          │    │
│  │  - Basic arithmetic    │  │  - Element-wise ops         │    │
│  │  - Matrix ops          │  │  - Matrix multiplication    │    │
│  │  - Reductions          │  │  - Flash Attention forward  │    │
│  │  - Activation funcs    │  │  - Fused linear ops        │    │
│  │  (Threaded via OpenMP) │  │  (Optimized for A100/H100) │    │
│  └────────────────────────┘  └─────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Core Components

### 1. **Computational Graph (`ad/graph.hpp`)**
- **Node**: Compute vertex holding value, gradient, inputs, and operation metadata
- **Value**: Wrapper around `Node` for ergonomic API and shape tracking
- **Topological Sort**: `topo_from()` orders graph for efficient execution
- **Op Enum**: 250+ registered operations with VJP implementations

**Key Design**: Lazy evaluation—the graph is built dynamically during forward pass, enabling dynamic control flow and easy debugging.

---

### 2. **Autograd Engine (`ad/autodiff.hpp`, `autodiff_*.cpp`)**

#### Forward Pass (`forward()`)
- Executes nodes in topological order
- Each node computes `output = op(inputs)`
- Stores activations needed for backward pass

#### Backward Pass (`backward()`)
- Vector-Jacobian Product (VJP) computes gradients via chain rule
- Gradient flows from root back through DAG
- Each op has a registered VJP implementation (e.g., `*_vjp_impl()`)

**File Structure**:
- `ad/autodiff.hpp` — Public API declarations
- `core/autodiff.cpp` — Forward/backward orchestration
- `autodiff/autodiff_vjp_ops.cpp` — VJP implementations for 100+ ops
- `autodiff/autodiff_jvp_ops.cpp` — JVP for forward-mode AD
- `autodiff/autodiff_nodeops.cpp` — Node-level reduction logic

**Example: Matrix Multiply VJP**
```cpp
// Forward: C = A @ B
// Backward: dA = dC @ B^T, dB = A^T @ dC
```

---

### 3. **Memory Optimization Techniques**

#### **Activation Checkpointing** (`ad/checkpoint.hpp`)
Trade memory for computation: recompute intermediate activations on backward instead of storing them all.

```cpp
void auto_checkpoint_by_depth(const Value& root, int depth_threshold);
```

**Why it matters**: Training a 7B-parameter model with sequence length 2048 can exhaust GPU memory without checkpointing. This technique halves memory usage at <10% compute overhead.

#### **In-Place Operations** (`ad/inplace.hpp`)
Modify tensors directly without allocation overhead.

```cpp
// Instead of: C = A + B (allocates new tensor)
// Do: A += B (reuses A's storage)
```

**Challenge**: In-place ops break saved activations → solution: version tracking + alias detection
- **Versioning**: Each tensor gets a version number; bumped on modification
- **Alias Tracking**: Detects when two nodes share memory

#### **CUDA Graphs** (`ad/cuda_graphs.hpp`, `runtime/cuda_graphs.cpp`)
Capture kernel launch sequences once, replay without CPU overhead.

```cpp
// Benefit: Eliminates per-kernel host-side launch latency (microseconds add up at 1000+ kernels/pass)
```

#### **Careful Deletion** (`ad/careful_deletion.hpp`)
Safe memory reclamation during backward pass without premature deallocation.

---

### 4. **Neural Network Layers**

**High-Level API** (`layer/*.hpp`):
```cpp
class Linear : public Layer { /* y = x @ W + b */ };
class Attention : public Layer { /* Multi-head self-attention */ };
class AlibiAttention : public Layer { /* ALiBi positional bias */ };
class RMSNorm : public Layer { /* Layer norm without centering */ };
class SWIGLU : public Layer { /* Fused gated linear unit */ };
// ... 20+ more
```

**Hardware-Aware Implementations**:
- Linear layers use `gemm_cuda()` — cuBLAS-style matrix multiply
- Attention uses `run_flash_forward()` — Flash Attention algorithm (Dao et al.)
- Fusion kernels combine multiple ops to reduce memory bandwidth pressure

---

### 5. **Operator Set** (`ad/ops.hpp`)

**Arithmetic**: `add`, `sub`, `mul`, `div`, `pow`, `clip`

**Element-wise Functions**:
- **Activations**: `relu`, `gelu`, `silu`, `swish`, `tanh`, `sigmoid`, `softplus`
- **Modern variants**: `mish`, `lisht`, `gcu`, `parcon`, `gaus`, `dyntanh`

**Matrix Operations**: `matmul`, `transpose`, `fmab` (fused multiply-add)

**Reductions**: `sum`, `rowsum`, `mean_all`, `rowmax`

**Complex Patterns**:
- `attention()` — Multi-head attention core
- `alibiatt()` — Attention with ALiBi positional embeddings
- `swiglu()` — Gated MLP with SwiGLU activation
- `cross_entropy_with_logits()` — Combined softmax + CE loss
- `kldivergence()` — KL divergence for distillation

**Saturation/Sparsity**: `softmax_row`, `logsumexp_row`, `sign`, `moewe` (Mixture of Experts)

---

### 6. **Tensor Library** (`tensor/`)

**Device Abstraction**:
```cpp
// Single API, works on CPU or GPU
Tensor t({32, 128, 512}, TensorOptions()
    .with_device(DeviceIndex(Device::CUDA))
    .with_dtype(Dtype::Float32)
    .with_req_grad(true));
```

**Key Features**:
- **Allocator Registry**: Pluggable allocators for different backends
- **Stride Support**: Non-contiguous views without copying
- **Type Safety**: Dtype checked at runtime
- **Device Transfer**: `t.to(Device::CUDA)` seamlessly moves data

**Supported Operations** (`tensor/ops/`):
- Unary: exp, log, sqrt, sin, cos, tanh, sigmoid, gelu
- Arithmetic: add, sub, mul, div, pow
- Reductions: sum, mean, max, min
- Linear algebra: matmul, transpose
- Utilities: clone, cast, reshape, transpose

---

## 🔬 Verification Against PyTorch

The framework includes extensive testing to ensure correctness:

```cpp
// Forward pass verification
Tensor expected = pytorch_reference(input);
Tensor computed = framework_forward(input);
assert(allclose(expected, computed, atol=1e-5));

// Gradient verification (numerical + VJP)
Tensor grad_numeric = numerical_gradient(output, wrt_input);
Tensor grad_vjp = vjp_gradient(output, wrt_input);
assert(allclose(grad_numeric, grad_vjp, atol=1e-4));
```

**Test Suite** (`cgadimpl/tests/`, `arch/tests/`):
- `test_aliop.cpp` — Alibi attention correctness
- `test_aliba.cpp` — Basic attention
- `test_layer.cpp` — Layer composability
- `test_newlay.cpp` — New layer implementations
- `test_alimass.cpp` — Attention scaling properties
- 12+ additional test modules

---

## 📊 Hardware-Aware Design Decisions

### Why Tiled GEMM Kernels?
**Without tiling**: Threads fetch data from global VRAM → memory bandwidth bottleneck (100 GB/s)
**With tiling**: Load blocks into fast shared memory (10 TB/s) → 100x speedup possible

### Why Flash Attention?
**Standard attention**: O(N²) memory requirement for attention matrix
**Flash Attention**: Streaming algorithm, O(N) memory, no precision loss

### Why Checkpointing?
**Without**: Store 1000 layer activations → 100s of GB memory
**With**: Store ~10 checkpoints → 1 GB memory, recompute as needed

### Why Kernel Fusion?
**Unfused**: Linear → write output → RMSNorm → read input → memory bandwidth wasted
**Fused**: Linear + RMSNorm → output directly to next op → 30-50% speedup

---

## 🚀 Getting Started

### Prerequisites
- **Linux** (CUDA toolkit 13.0+, or CPU-only)
- **CMake** 3.20+
- **CUDA Compute Capability** 8.6+ (A100, RTX 40-series, etc.)
- **Python** 3.8+ with pybind11

### Build & Test

```bash
# Clone and enter workspace
cd cgadimpl

# Run the build script (handles tensor library, core, kernels, tests)
bash ztools/run.sh
```

**What happens**:
1. Builds `tensor/` library (custom tensor ops on CPU/GPU)
2. Builds `cgadimpl_core` (autodiff engine + layers)
3. Builds CUDA kernels (optimized implementations)
4. Runs test suite to verify correctness
5. Creates `cgadimpl.so` Python module

### Python API Example

```python
import cgadimpl as ag

# Create model
model = ag.Sequential([
    ag.Linear(784, 512),
    ag.ReLU(),
    ag.Linear(512, 128),
    ag.ReLU(),
    ag.Linear(128, 10),
])

# Forward pass (dynamic graph)
logits = model(x)  # x shape: (batch, 784)

# Loss & backward
loss = ag.cross_entropy_with_logits(logits, targets)
loss.backward()  # Compute gradients

# Optimizer step (custom)
optimizer = ag.SGD(model.parameters(), lr=0.01)
optimizer.step()
optimizer.zero_grad()
```

---

## 📁 Project Structure

```
cgadimpl/
├── include/
│   ├── ad/
│   │   ├── autodiff.hpp       # Forward/backward API
│   │   ├── graph.hpp          # Node, Value, topological sort
│   │   ├── ops.hpp            # 250+ operation definitions
│   │   ├── checkpoint.hpp     # Activation checkpointing
│   │   ├── inplace.hpp        # In-place op safety
│   │   ├── cuda_graphs.hpp    # CUDA graph capture
│   │   └── detail/
│   │       ├── autodiff_ops.hpp   # VJP implementations
│   │       └── ops.def            # Op registry
│   ├── layer/
│   │   ├── attention.hpp      # Attention layers
│   │   ├── affine.hpp         # Linear, SWIGLU
│   │   ├── norm.hpp           # RMSNorm, LayerNorm
│   │   ├── activation.hpp     # Activation functions
│   │   └── ...                # 20+ layer types
│   ├── nn/
│   │   └── nn.hpp             # Module, Sequential, Linear
│   └── tensor.hpp             # Tensor wrapper
├── src/
│   ├── core/
│   │   ├── autodiff.cpp       # Core AD logic
│   │   ├── graph.cpp          # Graph construction
│   │   ├── ops.cpp            # Op implementations
│   │   └── nodeops.cpp        # Node reductions
│   ├── autodiff/
│   │   ├── autodiff_vjp_ops.cpp    # Gradient computations
│   │   ├── autodiff_jvp_ops.cpp    # Forward-mode AD
│   │   └── ...                     # Op-specific code
│   ├── layer/
│   │   ├── attention.cpp
│   │   ├── affine.cpp
│   │   ├── norm.cpp
│   │   └── ...                # Layer implementations
│   └── tools/
│       ├── binder.cpp         # pybind11 bindings
│       └── debug.cpp          # Debugging utilities
└── tests/
    ├── test_layer.cpp         # Layer composability
    ├── test_aliop.cpp         # Attention ops
    └── ...                    # 12+ test suites

tensor/
├── include/
│   ├── TensorLib.h            # Public API
│   ├── core/
│   │   ├── Tensor.h           # Core tensor class
│   │   ├── TensorDispatch.h   # Backend dispatch
│   │   └── ...
│   ├── device/
│   │   ├── Device.h           # Device abstraction
│   │   ├── Allocator.h        # Memory allocation
│   │   ├── CPUAllocator.cpp
│   │   └── CUDAAllocator.cpp
│   ├── dtype/
│   │   └── Dtype.h            # Type system
│   └── ops/
│       ├── Kernels.h          # Kernel interface
│       ├── TensorOps.h        # Tensor operations
│       └── UnaryOps/          # Element-wise ops
└── src/
    ├── device/                # Allocator implementations
    ├── core/                  # Tensor core ops
    └── TensorOps/             # Kernel implementations

kernels/
├── cpu/
│   ├── src/agkernels_cpu.cpp  # OpenMP-threaded kernels
│   └── benchmark/             # Perf testing suite
└── gpu/
    ├── entry.cu               # Kernel dispatch
    ├── eltwise.cu             # Element-wise ops
    ├── mm_cublas.cu           # Matrix multiply (cuBLAS)
    ├── vjp.cu                 # Gradient kernels
    └── zero.cu                # Memory operations

arch/
├── include/cgad.hpp           # High-level API
└── layer/                     # Architecture-specific layers

tests/
├── test_aliba.cpp             # Attention correctness
├── test_layer.cpp             # Composability
└── ...                        # 12 test modules
```

---

## 🧠 Key Technical Insights

### 1. Dynamic vs. Static Graphs
**CGAD Design**: Dynamic graphs (like PyTorch)
- Graphs built at runtime
- Supports dynamic control flow
- Easier debugging (can inspect nodes)
- Trade-off: Slightly more overhead per node

### 2. VJP Implementation Strategy
Each op stores minimal "saved tensors" needed for gradient:
```cpp
// Forward: y = op(x)
// Backward (VJP): dx = vjp_op(dy, saved_tensors)
```

Example (ReLU):
```cpp
// Forward: y = max(0, x)
// Backward: dx = dy * (x > 0)  [only need saved input mask]
```

### 3. Memory Hierarchy Exploitation
- **Shared Memory** (96 KB): Block-local fast storage
- **L1/L2 Cache**: Automatic (8 MB L2 on A100)
- **Global Memory**: 40-80 GB/s bandwidth (vs 10+ TB/s shared)

Strategy: Tile matrix multiplies so products stay in shared memory

### 4. Gradient Correctness
Verified at three levels:
1. **Numerical Gradient**: `(f(x+ε) - f(x-ε)) / 2ε`
2. **VJP Computation**: Backprop via VJP rules
3. **Tolerance Checking**: `allclose(numerical, vjp, atol=1e-4, rtol=1e-3)`

---

## 🔗 Integration Points

**For Production Deployment**:
1. **Python API** (pybind11) — Already built; expose to serving stack
2. **ONNX Export** (future) — Graph serialization
3. **Kernel Pluggability** — Add custom ops via `kernels/cpu` or `kernels/gpu`
4. **Device Abstraction** — Extend `Allocator` for TPU/custom accelerators

---

## 📚 Further Reading

**On the Algorithms**:
- **Autograd**: Griewank & Walther (2008), "Evaluating Derivatives"
- **Checkpointing**: Griewank et al. (1992), "Algorithm 799"
- **Flash Attention**: Dao et al. (2022), "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
- **ALiBi**: Press et al. (2022), "Train Short, Test Long"

**On Systems Design**:
- **CUDA Best Practices**: NVIDIA docs (occupancy, memory coalescing, warp efficiency)
- **Graph Compilation**: TVM, TensorRT papers

---

## 🎓 Learning Path

To understand this codebase systematically:

1. **Start**: Read `include/ad/graph.hpp` — understand Node/Value
2. **Forward**: Read `core/autodiff.cpp:forward()` — execution order
3. **Backward**: Read `autodiff/autodiff_vjp_ops.cpp` — chain rule
4. **Kernels**: Study `kernels/gpu/eltwise.cu` — simple element-wise ops
5. **Fusions**: Study `kernels/gpu/mm_cublas.cu` — tiled GEMM
6. **Layers**: Read `layer/attention.cpp` — op composition
7. **Optimize**: Read `ad/checkpoint.hpp` — memory techniques

---

## ✅ Testing & Verification

Run full test suite:
```bash
cd cgadimpl/build
ctest --output-on-failure
```

**Key Test Suites**:
- Autodiff correctness (numerical vs. VJP)
- Layer composition (forward → backward → parameter updates)
- Attention implementations (vs. reference)
- Memory safety (no leaks, safe deletion)
- Device transfers (CPU ↔ GPU)

---

## 📝 Build Configuration

**Key CMake Flags**:
- `CMAKE_BUILD_TYPE=Debug|Release` — Optimization level
- `CMAKE_CUDA_ARCHITECTURES=86` — SM version (A100: 80, H100: 90)
- `CMAKE_CXX_STANDARD=20` — C++ standard

**For Development**:
```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake --build . -- -j$(nproc)
```

**For Production**:
```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -- -j$(nproc)
```

---

## 🤝 Contributing

Areas for extension:
- [ ] **Quantization kernels** (Int8, Float8 E4M3)
- [ ] **Sparsity support** (pruned attention, sparse GEMM)
- [ ] **Distributed training** (gradient accumulation, all-reduce fusion)
- [ ] **Custom backends** (TPU, Cerebras, custom ASICs)
- [ ] **Graph export** (ONNX, TensorRT)

---

## 📄 License

MIT License — See LICENSE file

---

**Built to deeply understand modern ML systems, not to replace them.**

For integration into production ML pipelines, consider this a reference implementation and educational resource. The architecture demonstrates the design decisions that power frameworks like PyTorch, JAX, and TensorFlow under the hood.

---

## 📬 Questions?

**Key Files to Explore**:
- "How do gradients work?" → `include/ad/ops.hpp`, `autodiff/autodiff_vjp_ops.cpp`
- "Why is checkpointing useful?" → `include/ad/checkpoint.hpp` (detailed comments)
- "What kernels are optimized?" → `kernels/gpu/entry.cu` (dispatcher)
- "How do layers compose?" → `include/layer/attention.hpp`, `src/layer/attention.cpp`
- "What tensor ops exist?" → `tensor/include/ops/TensorOps.h`
