#include "tensor.hpp"
#include "ad/ops.hpp"
#include "ad/kernels_api.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <ad/runtime.hpp>
#include "ad/ag_all.hpp"
#include "ad/debug.hpp"

// Helper to check equality
void check_tensors(const Tensor& out, const Tensor& ref, const char* name) {
    Tensor diff = abs(out - ref, ag::current_stream());
    float max_err = reduce_max(diff).data<float>()[0]; // Assuming item() gets scalar
    if (max_err > 1e-4) {
        std::cerr << "FAIL: " << name << " | Max Error: " << max_err << "\n";
        exit(1);
    }
    std::cout << "PASS: " << name << "\n";
}

bool check_grad(const Tensor& analytical_grad, const Tensor& numerical_grad, double tol = 1e-3) {
// ag::debug::print_tensor("N", analytical_grad);
    
    
    if (analytical_grad.shape().dims != numerical_grad.shape().dims) {
        std::cerr << "  [FAIL] Shape mismatch!\n";
        return false;
    }
    
    Tensor diff = abs(analytical_grad - numerical_grad, ag::current_stream());
    Tensor max_val = reduce_max(diff);
    cudaDeviceSynchronize();
    double error = max_val.to_cpu().data<float>()[0];

    if (error > tol) {
        std::cerr << "  [FAIL] Max error: " << error << " > tolerance: " << tol << "\n";
        // For debugging, let's print the tensors
        std::cerr << "  Analytical Grad:\n";
        analytical_grad.display(std::cerr, 4);
        std::cerr << "  Numerical Grad:\n";
        numerical_grad.display(std::cerr, 4);
        return false;
    }
    return true;
}


void test_linear_layer() {
    std::cout << "--- Testing Linear Layer ---\n";
    
    // 1. Setup dimensions
    int Batch = 2;
    int In = 3;
    int Out = 4;

    // 2. Initialize Layer
    // Assuming generic API: Linear(input_dim, output_dim)
    ag::layer::Linear layer(In, Out); 

    // 3. Manually set weights to known values so test is deterministic
    // (Don't rely on random init for unit tests if possible)
    // Tensor W_cpu = Tensor::randn(Shape({Batch, In}), TensorOptions());
    // Tensor B_cpu = Tensor::randn(Shape({1, Out}), TensorOptions());


    // 4. Create Input
    Tensor X = Tensor::randn(Shape({Batch, In}), TensorOptions());
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));

    // 5. RUN LAYER (The thing we are testing)
    auto layer_out = layer(m);
    auto f = layer.parameters();
    ag::debug::print_value("Here", f[0]);
    ag::debug::print_value("Here", f[1]);

    // 6. RUN REFERENCE (The "Truth" - raw tensor math)
    // Linear is X * W^T + B
    Tensor ref_out = matmul(X, f[0].val().t()) + f[1].val();

    // 7. Compare
    check_tensors(layer_out.val(), ref_out, "Linear Forward");

    // --- Backward Test ---
    
    // Run Layer Backward
    backward(layer_out); // Or layer.backward(grad_out) depending on API
    // std::cout<<m.node->requires_grad();

    // Calculate Reference Grads manually
    // dInput = dOutput * W.T
    Tensor ref_dInput = matmul(layer_out.grad(), f[0].val());
    // dWeight = Input.T * dOutput
    Tensor ref_dWeight = matmul(layer_out.grad().t(), m.val());
    // dBias = sum(dOutput, axis=0)
    Tensor ref_dBias = reduce_sum(layer_out.grad(), {0}, true); // Keep dims? depends on API

    check_tensors(f[0].grad(), ref_dWeight, "Linear Weight Grad");
    check_tensors(f[1].grad(), ref_dBias, "Linear Bias Grad");
    check_tensors(m.grad(), ref_dInput, "Linear Input Grad");
    

}

int main() {
    // Load CUDA if layers use it


    test_linear_layer();
    // test_relu_layer();
    
    return 0;
}