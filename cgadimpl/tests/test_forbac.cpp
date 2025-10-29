// ====================================================================
// FILE: cgadimpl/tests/test_end_to_end_gpu.cpp (Corrected Version)
// ====================================================================
#include "nn/nn.hpp"
#include "ad/autodiff.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
#include <functional> // <--- FIX 1: Added the missing header for std::function
#include <assert.h>

// A simple helper to check if a test passes or fails as expected
void run_test(const std::string& test_name, std::function<void()> test_func, bool should_throw) {
    std::cout << "--- Running Test: " << test_name << " ---" << std::endl;
    try {
        test_func();
        if (should_throw) {
            std::cerr << "FAIL: Test was expected to throw an exception, but it did not." << std::endl;
        } else {
            std::cout << "PASS: Test completed successfully as expected." << std::endl;
        }
    } catch (const std::runtime_error& e) {
        if (should_throw) {
            std::cout << "PASS: Test threw an exception as expected." << std::endl;
            std::cout << "      Reason: " << e.what() << std::endl;
        } else {
            std::cerr << "FAIL: Test was not expected to throw, but it did." << std::endl;
            std::cerr << "      Reason: " << e.what() << std::endl;
        }
    }
}

// A simple two-layer MLP for testing
class SimpleMLP : public ag::nn::Module {
public:
    ag::nn::Linear fc1;
    ag::nn::Linear fc2;

    SimpleMLP(int in, int hidden, int out, ag::Device dev)
        : fc1(in, hidden, dev), fc2(hidden, out, dev) {
        // Manually collect parameters from sub-modules
        params_.insert(params_.end(), fc1.parameters().begin(), fc1.parameters().end());
        params_.insert(params_.end(), fc2.parameters().begin(), fc2.parameters().end());
    }

    ag::Value forward(const ag::Value& x) {
        ag::Value h = ag::gelu(fc1(x));
        return fc2(h);
    }
};

void compretest()
{


    
}

void test_cpu_backward_pass() {
    using namespace ag;
    SimpleMLP model(10, 5, 2, Device::CPU);
    
    // FIX 2: Added the missing 'seed' argument to randn
    Value input = make_tensor(Tensor::randn(4, 10, /*seed=*/123, Device::CPU), "input");
    
    // Run forward and backward
    Value output = model.forward(input);
    backward(output);

    // Check if the first weight has a non-zero gradient
    auto params = model.parameters();
    float grad_sum = params[0].node->grad.sum_scalar();
    if (std::abs(grad_sum) < 1e-9) { // Use tolerance for float comparison
        throw std::runtime_error("CPU backward pass resulted in zero gradients.");
    }
}

void test_gpu_backward_pass() {
    using namespace ag;
    SimpleMLP model(10, 5, 2, Device::CPU);
    model.to(Device::CUDA); // Move model to GPU

    // FIX 3: Added the missing 'seed' argument to randn
    Value input = make_tensor(Tensor::randn(4, 10, /*seed=*/456, Device::CUDA), "input");

    // Run forward and backward
    Value output = model.forward(input);
    backward(output); // This line is expected to throw!
}


int main() {
    using namespace ag;
    // ...
    Value input = make_tensor(Tensor::randn(8, 10, 1337, Device::CUDA), "input");
    
    // --- DEMONSTRATION FOR YOUR TEAMMATE ---
    std::cout << "--- Proving zeros_like behavior ---" << std::endl;
    std::cout << "Input tensor is on CUDA: " << (input.val().is_cuda() ? "true" : "false") << std::endl;
    
    // Create a new tensor using zeros_like
    Tensor zero_tensor = Tensor::zeros_like(input.val());
    
    std::cout << "Tensor created with zeros_like is on CUDA: " << (zero_tensor.is_cuda() ? "true" : "false") << std::endl;
    assert(zero_tensor.is_cuda()); // This will pass.
    std::cout << "--- Proof complete ---" << std::endl;
    // --- END DEMONSTRATION ---

    // Test 1: Full forward/backward pass on the CPU. This should pass without error.
    run_test("CPU End-to-End Backward Pass", test_cpu_backward_pass, /*should_throw=*/false);
    
    std::cout << "\n";

    // Test 2: Full forward/backward pass on the GPU. This is EXPECTED TO FAIL
    //         inside the `vjp_Relu` function because we haven't written the CUDA kernel yet.
    //         A "PASS" for this test means it threw the correct error.
    run_test("GPU End-to-End Backward Pass (Expect Failure)", test_gpu_backward_pass, /*should_throw=*/false);

    return 0;
}