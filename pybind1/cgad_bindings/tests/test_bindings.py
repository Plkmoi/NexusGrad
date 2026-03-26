#!/usr/bin/env python3
"""
Test suite for CGADImpl Python bindings.
"""

import sys
import traceback

def test_module_import():
    """Test that module imports successfully."""
    try:
        import cgad
        print("✓ Module import successful")
        print(f"  Version: {cgad.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Module import failed: {e}")
        return False

def test_tensor_module():
    """Test tensor module availability."""
    try:
        import cgad
        # Test that submodules exist
        assert hasattr(cgad, 'tensor'), "tensor module not found"
        assert hasattr(cgad.tensor, 'zeros'), "zeros function not found"
        assert hasattr(cgad.tensor, 'ones'), "ones function not found"
        assert hasattr(cgad.tensor, 'randn'), "randn function not found"
        print("✓ Tensor module structure verified")
        return True
    except AssertionError as e:
        print(f"✗ Tensor module test failed: {e}")
        return False

def test_autodiff_module():
    """Test autodiff module availability."""
    try:
        import cgad
        assert hasattr(cgad, 'autodiff'), "autodiff module not found"
        assert hasattr(cgad.autodiff, 'backward'), "backward function not found"
        assert hasattr(cgad.autodiff, 'sum'), "sum function not found"
        assert hasattr(cgad.autodiff, 'matmul'), "matmul function not found"
        assert hasattr(cgad.autodiff, 'relu'), "relu function not found"
        print("✓ Autodiff module structure verified")
        return True
    except AssertionError as e:
        print(f"✗ Autodiff module test failed: {e}")
        return False

def test_layer_module():
    """Test layer module availability."""
    try:
        import cgad
        assert hasattr(cgad, 'layer'), "layer module not found"
        assert hasattr(cgad.layer, 'Linear'), "Linear layer not found"
        assert hasattr(cgad.layer, 'ReLU'), "ReLU layer not found"
        assert hasattr(cgad.layer, 'Sequential'), "Sequential module not found"
        print("✓ Layer module structure verified")
        return True
    except AssertionError as e:
        print(f"✗ Layer module test failed: {e}")
        return False

def test_optim_module():
    """Test optimizer module availability."""
    try:
        import cgad
        assert hasattr(cgad, 'optim'), "optim module not found"
        assert hasattr(cgad.optim, 'sgd'), "sgd optimizer not found"
        assert hasattr(cgad.optim, 'adam'), "adam optimizer not found"
        print("✓ Optimizer module structure verified")
        return True
    except AssertionError as e:
        print(f"✗ Optimizer module test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("CGADImpl Python Bindings Test Suite")
    print("=" * 60)
    
    tests = [
        test_module_import,
        test_tensor_module,
        test_autodiff_module,
        test_layer_module,
        test_optim_module,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"✗ {test_func.__name__} failed with error:")
            print(f"  {e}")
            traceback.print_exc()
            results.append(False)
    
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    
    if all(results):
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    exit(main())
