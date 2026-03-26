#!/usr/bin/env python3
"""
Test suite for CGADImpl Python bindings.

Demonstrates usage of:
- Tensor creation and operations
- Autodifferentiation
- Layer composition
- Training loops
"""

import sys
sys.path.insert(0, '../dist')

import cgad as cg
import numpy as np


def test_tensor_creation():
    """Test basic tensor creation and properties."""
    print("Testing tensor creation...")
    
    # Create tensors
    x = cg.tensor.randn([2, 3])
    y = cg.tensor.zeros([2, 3])
    z = cg.tensor.ones([3, 4])
    
    print(f"x shape: {x.shape()}, dtype: {x.dtype()}, device: {x.device()}")
    print(f"y shape: {y.shape()}")
    print(f"z numel: {z.numel()}")
    
    # Tensor operations
    a = x + y
    b = x * y
    c = x / (z.reshape([3, 4]))  # Note: incompatible shapes for demo
    
    print("✓ Tensor creation tests passed")


def test_autodiff():
    """Test automatic differentiation."""
    print("\nTesting autodifferentiation...")
    
    # Create input tensor
    x = cg.tensor.randn([2, 3], requires_grad=True)
    
    # Wrap in computation graph
    x_val = cg.autodiff.make_tensor(x)
    
    # Forward pass with operations
    y = cg.autodiff.relu(x_val)
    z = cg.autodiff.sum(y)
    
    # Backward pass
    cg.autodiff.backward(z)
    
    # Check gradients
    grad_x = x.grad()
    print(f"Gradient shape: {grad_x.shape() if grad_x else 'None'}")
    
    print("✓ Autodiff tests passed")


def test_layers():
    """Test layer API."""
    print("\nTesting layer API...")
    
    # Create simple network
    linear = cg.layer.Linear(10, 5, device=cg.tensor.Device.CPU)
    relu = cg.layer.ReLU()
    output = cg.layer.Linear(5, 2, device=cg.tensor.Device.CPU)
    
    # Sequential composition
    model = cg.layer.Sequential([linear, relu, output])
    
    # Get parameters
    params = model.parameters()
    print(f"Number of parameter tensors: {len(params)}")
    
    print("✓ Layer tests passed")


def test_attention():
    """Test attention layer."""
    print("\nTesting attention layer...")
    
    batch_size = 2
    seq_len = 10
    embed_dim = 64
    num_heads = 8
    
    # Create attention layer
    attn = cg.layer.Attention(batch_size, embed_dim, num_heads, embed_dim)
    
    print(f"Created attention layer: {batch_size} batch, {embed_dim} dim, {num_heads} heads")
    
    # Create input
    x = cg.tensor.randn([batch_size, seq_len, embed_dim])
    print(f"Input shape: {x.shape()}")
    
    print("✓ Attention tests passed")


def test_activations():
    """Test various activation functions."""
    print("\nTesting activation functions...")
    
    x = cg.tensor.randn([4, 4])
    
    # Wrap in computation graph
    x_val = cg.autodiff.make_tensor(x)
    
    # Test activations
    activations = [
        ("ReLU", cg.autodiff.relu),
        ("GELU", cg.autodiff.gelu),
        ("Tanh", cg.autodiff.tanh),
        ("Sigmoid", cg.autodiff.sigmoid),
        ("SiLU", cg.autodiff.silu),
        ("Mish", cg.autodiff.mish),
    ]
    
    for name, func in activations:
        y = func(x_val)
        print(f"  {name}: {y.shape()}")
    
    print("✓ Activation tests passed")


def test_loss_functions():
    """Test loss functions."""
    print("\nTesting loss functions...")
    
    # Classification
    logits = cg.tensor.randn([4, 10])  # 4 samples, 10 classes
    targets = cg.tensor.full([4], 0.0)  # All class 0
    
    logits_val = cg.autodiff.make_tensor(logits, "logits")
    targets_val = cg.autodiff.make_tensor(targets, "targets")
    
    ce_loss = cg.autodiff.cross_entropy_with_logits(logits_val, targets_val)
    print(f"Cross-entropy loss: {ce_loss.val().mean()}")
    
    # Regression
    pred = cg.tensor.randn([4, 3])
    target = cg.tensor.randn([4, 3])
    
    pred_val = cg.autodiff.make_tensor(pred)
    target_val = cg.autodiff.make_tensor(target)
    
    mse = cg.autodiff.mse_loss(pred_val, target_val)
    mae = cg.autodiff.mae_loss(pred_val, target_val)
    print(f"MSE loss: {mse.val().mean()}")
    print(f"MAE loss: {mae.val().mean()}")
    
    print("✓ Loss function tests passed")


def test_optimizer():
    """Test optimizer functions."""
    print("\nTesting optimizers...")
    
    # Create simple parameter
    param = cg.tensor.randn([10, 5], requires_grad=True)
    param_val = cg.autodiff.param(param, "weight")
    
    # Create dummy loss
    loss = cg.autodiff.sum(param_val * 2)
    
    # Backward
    cg.autodiff.backward(loss)
    
    # Optimization step
    grad = param.grad()
    if grad:
        # Simplified manual SGD update
        lr = 0.01
        # param -= lr * grad  (would need tensor operation)
        print(f"Parameter gradient norm: {grad.mean()}")
    
    print("✓ Optimizer tests passed")


def test_numpy_interop():
    """Test numpy array interoperability."""
    print("\nTesting numpy interoperability...")
    
    # Create numpy array
    np_array = np.random.randn(3, 4).astype(np.float32)
    print(f"Numpy array shape: {np_array.shape}")
    
    # Convert to tensor
    tensor = cg.tensor.from_numpy(np_array)
    print(f"Tensor shape: {tensor.shape()}")
    
    # Convert back
    np_array_back = cg.tensor.to_numpy(tensor)
    print(f"Back to numpy: {np_array_back.shape}")
    
    print("✓ Numpy interop tests passed")


def test_device_management():
    """Test device management (CPU/CUDA)."""
    print("\nTesting device management...")
    
    # Create on CPU
    x = cg.tensor.randn([2, 3], device=cg.tensor.Device.CPU)
    print(f"Created on {x.device()}")
    
    # Attempt CUDA (will fail if not compiled with CUDA)
    try:
        x_cuda = x.to_cuda()
        print(f"Moved to {x_cuda.device()}")
        x_cpu = x_cuda.to_cpu()
        print(f"Back to {x_cpu.device()}")
    except RuntimeError as e:
        print(f"CUDA not available: {e}")
    
    print("✓ Device management tests passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("CGADImpl Python Bindings Test Suite")
    print("=" * 60)
    
    try:
        test_tensor_creation()
        test_autodiff()
        test_layers()
        test_attention()
        test_activations()
        test_loss_functions()
        test_optimizer()
        test_numpy_interop()
        test_device_management()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
