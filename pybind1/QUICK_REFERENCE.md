# CGADImpl Python Quick Reference

## Installation

```bash
cd pybind11
cmake -B build
cmake --build build --config Release
python tests/test_bindings.py
```

## Import

```python
import cgad as cg
import numpy as np
```

## Tensor Creation

```python
x = cg.tensor.zeros([2, 3])              # All zeros
x = cg.tensor.ones([2, 3])               # All ones
x = cg.tensor.randn([2, 3])              # Normal distribution
x = cg.tensor.rand([2, 3])               # Uniform [0, 1)
x = cg.tensor.full([2, 3], value=5.0)    # Filled with value

# With options
opts = cg.tensor.TensorOptions()
opts.dtype(cg.tensor.Dtype.Float32)
opts.device(cg.tensor.Device.CPU)
opts.requires_grad(True)
x = cg.tensor.randn([2, 3], options=opts)
```

## Tensor Properties

```python
shape = x.shape()              # Get shape
dtype = x.dtype()              # Get dtype
device = x.device()            # Get device
numel = x.numel()              # Total elements
grad = x.grad()                # Get gradient
```

## Tensor Operations

```python
y = x.reshape([6, 1])          # Reshape
y = x.transpose()              # Transpose
y = x.flatten()                # Flatten to 1D
y = x.t()                      # 2D transpose
y = x.to_cpu()                 # Move to CPU
y = x.to_cuda()                # Move to GPU
```

## Numpy Conversion

```python
x = cg.tensor.from_numpy(np_array)      # Numpy to tensor
np_array = cg.tensor.to_numpy(tensor)   # Tensor to numpy
```

## Computation Graph

```python
# Wrap tensor in computation graph
x = cg.tensor.randn([10, 10], requires_grad=True)
x_val = cg.autodiff.make_tensor(x)

# Create learnable parameter
w = cg.tensor.randn([10, 5], requires_grad=True)
w_val = cg.autodiff.param(w)

# Create constant (no gradient)
const = cg.tensor.randn([5, 1])
const_val = cg.autodiff.constant(const)
```

## Autodiff Operations

### Math Operations

```python
y = cg.autodiff.add(x, y)       # Element-wise add
y = cg.autodiff.sub(x, y)       # Element-wise subtract
y = cg.autodiff.mul(x, y)       # Element-wise multiply
y = cg.autodiff.div(x, y)       # Element-wise divide
z = cg.autodiff.matmul(x, y)    # Matrix multiply
s = cg.autodiff.sum(x)          # Sum to scalar
```

### Operators (also supported)

```python
z = x + y
z = x - y
z = x * y
z = x / y
z = cg.autodiff.flomul(x, 2.0)  # x * scalar with grad
z = cg.autodiff.floadd(x, 1.0)  # x + scalar with grad
```

### Activation Functions

```python
y = cg.autodiff.relu(x)         # ReLU
y = cg.autodiff.gelu(x)         # GELU
y = cg.autodiff.tanh(x)         # Tanh
y = cg.autodiff.sigmoid(x)      # Sigmoid
y = cg.autodiff.silu(x)         # SiLU
y = cg.autodiff.swish(x)        # Swish
y = cg.autodiff.mish(x)         # Mish
y = cg.autodiff.leaky_relu(x)   # Leaky ReLU
y = cg.autodiff.softplus(x)     # Softplus
```

### Loss Functions

```python
loss = cg.autodiff.cross_entropy_with_logits(logits, targets)
loss = cg.autodiff.mse_loss(pred, target)
loss = cg.autodiff.mae_loss(pred, target)
loss = cg.autodiff.kldivergence(p, q)
```

### Advanced Operations

```python
y = cg.autodiff.transpose_op(x)         # Transpose
y = cg.autodiff.softmax_row(x)          # Row-wise softmax
y = cg.autodiff.attention(q, k, v)      # Multi-head attention
y = cg.autodiff.alibiatt(q, k, v)       # ALiBi attention
```

## Backward Pass

```python
# Compute loss
loss = cg.autodiff.cross_entropy_with_logits(y_pred, y_true)

# Backpropagate
cg.autodiff.backward(loss)

# Access gradients
grad_w = w.grad()  # Gradient w.r.t. w

# Clear gradients
cg.autodiff.zero_grad(x_val)
```

## Layers

### Creating Layers

```python
linear = cg.layer.Linear(10, 5)                        # FC layer
relu = cg.layer.ReLU()                                 # Activation
model = cg.layer.Sequential([linear, relu])           # Composition
residual = cg.layer.ResidualBlock([linear, relu])     # Skip connection
```

### Activation Layers

```python
cg.layer.ReLU()
cg.layer.GELU()
cg.layer.Tanh()
cg.layer.Sigmoid()
cg.layer.Swish()
cg.layer.Mish()
cg.layer.LeakyReLU(alpha=0.01)
```

### Normalization

```python
norm = cg.layer.LayerNorm([768])          # Layer norm
rms = cg.layer.RMSNorm(768)               # RMS norm
dyn = cg.layer.DynTanh(0.5, 1.0, 2.0)    # Dynamic tanh
```

### Attention

```python
attn = cg.layer.Attention(
    batch_size=32,
    in_features=768,
    num_heads=12,
    out_features=768
)

alibi = cg.layer.AlibiAttention(
    batch_size=32,
    in_features=768,
    out_features=768,
    num_heads=12
)
```

### Loss Layers

```python
ce = cg.layer.CrossEntropyWithLogits()
mse = cg.layer.MSELoss()
mae = cg.layer.MAELoss()
kldiv = cg.layer.KLDivergence()
```

### Layer Operations

```python
params = layer.parameters()        # Get parameters
layer.to(cg.tensor.Device.CPU)     # Move to device
layer.zero_grad()                  # Clear gradients
```

## Optimizers

### SGD

```python
cg.optim.sgd(output_value, learning_rate=0.01)                    # Basic SGD
cg.optim.sgd_momentum(output_value, velocity, lr=0.01, momentum=0.9)  # With momentum
```

### Adam

```python
cg.optim.adam(output_value, m_t, v_t, 
              learning_rate=0.001, beta1=0.9, beta2=0.999,
              epsilon=1e-8, t=1)

cg.optim.adamw(output_value, m_t, v_t, 
               learning_rate=0.001, weight_decay=0.01, t=1)
```

### Other Optimizers

```python
cg.optim.rmsprop(output_value, v_t, learning_rate=0.01, alpha=0.99)
```

## Learning Rate Scheduling

```python
lr = cg.optim.step_decay(0.1, epoch, decay_epoch=10, decay_rate=0.1)
lr = cg.optim.exponential_decay(0.1, epoch, decay_rate=0.01)
lr = cg.optim.cosine_annealing(0.1, epoch, max_epochs=100)
```

## Gradient Clipping

```python
norm = cg.optim.clip_grad_norm(parameters, max_norm=1.0)
cg.optim.clip_grad_value(parameters, clip_value=1.0)
```

## Training Loop Example

```python
import cgad as cg

# Setup
model = cg.layer.Sequential([
    cg.layer.Linear(784, 128),
    cg.layer.ReLU(),
    cg.layer.Linear(128, 10),
])

# Training
for epoch in range(epochs):
    for x_batch, y_batch in dataloader:
        # Forward
        x = cg.tensor.from_numpy(x_batch)
        y = cg.tensor.from_numpy(y_batch)
        
        logits = model(cg.autodiff.make_tensor(x))
        loss = cg.autodiff.cross_entropy_with_logits(
            logits,
            cg.autodiff.make_tensor(y)
        )
        
        # Backward
        cg.autodiff.backward(loss)
        
        # Update
        cg.optim.sgd(logits, learning_rate=0.01)
        cg.autodiff.zero_grad(logits)
        
        print(f"Loss: {loss.val().mean():.4f}")
```

## Common Patterns

### Setting up training variables

```python
# Parameter with gradient tracking
w = cg.tensor.randn([10, 5], requires_grad=True)
w_param = cg.autodiff.param(w)

# Input data
x = cg.tensor.from_numpy(x_batch)
x_tensor = cg.autodiff.make_tensor(x)
```

### Computing and optimizing loss

```python
# Forward
output = cg.autodiff.matmul(x, w_param)
loss = cg.autodiff.sum(output)

# Backward
cg.autodiff.backward(loss)

# Step
cg.optim.sgd(loss, learning_rate=0.01)

# Reset for next iteration
cg.autodiff.zero_grad(loss)
```

### Multi-layer network

```python
layer1 = cg.layer.Linear(784, 128)
relu = cg.layer.ReLU()
layer2 = cg.layer.Linear(128, 10)

model = cg.layer.Sequential([layer1, relu, layer2])
output = model(x_tensor)  # Automatically calls layers in sequence
```

## Enums

```python
# Device
cg.tensor.Device.CPU
cg.tensor.Device.CUDA

# Dtype
cg.tensor.Dtype.Float32
cg.tensor.Dtype.Float64
cg.tensor.Dtype.Int32
cg.tensor.Dtype.Int64
```

## Tips

1. **Always use requires_grad=True** for parameters and inputs that need gradients
2. **Call zero_grad()** between training steps to avoid accumulating gradients
3. **Use from_numpy()** and **to_numpy()** for numpy integration
4. **Device management**: Use .to_cuda() for GPU or .to_cpu() for CPU
5. **Batch operations**: matmul, attention, softmax are optimized
6. **Check gradient shapes**: grad() returns same shape as tensor

## Debugging

```python
# Check shape
print(x.shape())

# Check dtype
print(x.dtype())

# Check if requires grad
print(x.requires_grad())

# Check gradient
if x.grad() is not None:
    print("Has gradient")
else:
    print("No gradient")

# Get string representation
print(x)  # Prints <Tensor shape=...>
```

## Documentation

```python
# Get docstring
import cgad as cg
help(cg.autodiff.matmul)

# Get attributes
print(dir(cg.tensor))
print(dir(cg.autodiff))
print(dir(cg.layer))
print(dir(cg.optim))
```
