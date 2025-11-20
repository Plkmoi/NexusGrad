#include "tensor.hpp"
#include "layer/affine.hpp" // Assuming this is where Linear/Layer classes are
#include "ad/ops.hpp"
#include "ad/kernels_api.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <ad/runtime.hpp>
#include "ad/ag_all.hpp"
#include "ad/debug.hpp"
using namespace ag;
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


void test_linear() {
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
    // ag::debug::print_value("Here", f[0]);
    // ag::debug::print_value("Here", f[1]);

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



void test_rmsnorm() {
    std::cout << "--- Testing RMSNorm Layer ---\n";
    
    // 1. Setup dimensions
    int Batch = 2;
    int In = 3;
    int Out = 4;

    // 2. Initialize Layer
    // Assuming generic API: Linear(input_dim, output_dim)
    ag::layer::RMSNorm newlay; 

    // 3. Manually set weights to known values so test is deterministic
    // (Don't rely on random init for unit tests if possible)
    // Tensor W_cpu = Tensor::randn(Shape({Batch, In}), TensorOptions());
    // Tensor B_cpu = Tensor::randn(Shape({1, Out}), TensorOptions());


    // 4. Create Input
    Tensor X = Tensor::randn(Shape({Batch, In}), TensorOptions());
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));

    // 5. RUN LAYER (The thing we are testing)
    auto layer_out = newlay(m);
    auto f = newlay.parameters();
    auto gam = newlay.gam();
    // ag::debug::print_value("Here", f[0]);
    // ag::debug::print_value("Here", f[1]);

    // 6. RUN REFERENCE (The "Truth" - raw tensor math)
    // Linear is X * W^T + B
         Tensor& Xa = m.node->value;
     Tensor G = Tensor::full(Shape{{1, 1}}, TensorOptions().with_req_grad(true).with_device(X.device()), gam); // Assume gain is trainable

     float inv_cols = 1.0f / static_cast<float>(X.shape().dims.back());
    Tensor var   = OwnTensor::reduce_sum(Xa * Xa, {-1}, true) * inv_cols;
    Tensor rsqrt = 1.0f / OwnTensor::sqrt(var + 1e-5f, ag::current_stream());
    Tensor ref_out     = (Xa * rsqrt) * G;

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




// // Helper to compute numerical gradient for a specific input tensor
// Tensor compute_numerical_grad(ag::layer::Layer& layer, Value input, Value& param, float eps = 1e-3f) {
//     // We want dLoss/dParam. Since we don't have a real loss, let's assume Loss = Sum(Output).
//     // Therefore dLoss/dOutput = 1.
    
//     Tensor grad_approx = Tensor::zeros(param.val().shape());
//     float* p_data = param.val().to(Device::CPU).data<float>(); // Access CPU data for editing
//     float* g_data = grad_approx.to(Device::CPU).data<float>();

//     int size = param.val().numel();
    
//     for (int i = 0; i < size; ++i) {
//         float original_val = p_data[i];

//         // 1. Perturb +eps
//         p_data[i] = original_val + eps;
//         // You might need to push back to device if layer keeps internal copies
//         param.val().copy_from_host(p_data); 
        
//         Value out_plus = layer(input);
//         float loss_plus = reduce_sum(out_plus.val()).data<float>(); // Sum of output as dummy loss

//         // 2. Perturb -eps
//         p_data[i] = original_val - eps;
//         param.val().copy_from_host(p_data);

//         Value out_minus = layer(input);
//         float loss_minus = reduce_sum(out_minus.val()).data<float>();

//         // 3. Compute Derivative
//         g_data[i] = (loss_plus - loss_minus) / (2 * eps);

//         // 4. Restore
//         p_data[i] = original_val;
//     }
//     param.val().copy_from_host(p_data); // Restore original weights

//     return grad_approx; // Move back to GPU if needed for comparison
// }

// void test_layer_automatically(ag::layer::Layer& layer, Tensor input_tensor, const char* name) {
//     std::cout << "\n🤖 Auto-Testing Layer: " << name << " 🤖\n";
    
//     Value x = ag::make_tensor(input_tensor, "Input");
    
//     // 1. Run Real Backward
//     Value out = layer(x);
//     Value dummy_loss = reduce_sum(out);
//     backward(dummy_loss); // Computes analytical gradients

//     // 2. Check Parameters
//     for (size_t i = 0; i < layer.parameters().size(); ++i) {
//         Value param = layer.parameters()[i];
//         if (!param.node->requires_grad()) continue;

//         std::cout << "  Checking Param " << i << "... ";
        
//         // Compute Numerical Gradient
//         Tensor num_grad = compute_numerical_grad(layer, x, param);
        
//         // Compare
//         // Note: Tolerance needs to be loose (e.g., 1e-2 or 1e-3) for finite differences
//         if (check_grad(param.grad(), num_grad, 1e-2)) { 
//             std::cout << "✅ PASS\n";
//         } else {
//             std::cout << "❌ FAIL\n";
//             exit(1);
//         }
//     }
// }



void test_attention() {
    std::cout << "--- Testing Linear Layer ---\n";
    
    // 1. Setup dimensions
    int Batch = 4;
    int In = 4;
    int Out = 4;

    // 2. Initialize Layer
    // Assuming generic API: Linear(input_dim, output_dim)
    ag::layer::Attention layer(In, Out); 

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



     Tensor& A = X;
     Tensor& B = f[0].node->value;
     Tensor& C = f[1].node->value;
     Tensor& D = f[2].node->value;

    Tensor q = matmul(A, B);
    Tensor k = matmul(A, C);
    Tensor v = matmul(A, D);

    float scale = 1.f / sqrtf(static_cast<float>(k.shape().dims.back()));
    Tensor g = matmul(q, k.t()) * scale;

    Tensor ma   = OwnTensor::reduce_max(g, {-1}, true);
    Tensor e   = OwnTensor::exp(g - ma);
    Tensor p   = OwnTensor::reduce_sum(e, {-1}, true);
    Tensor s   = e / p;

    Tensor ref_out = matmul(s, v);

    // 7. Compare
    check_tensors(layer_out.val(), ref_out, "Linear Forward");

    // --- Backward Test ---
    
    // Run Layer Backward
    backward(layer_out); // Or layer.backward(grad_out) depending on API
    // std::cout<<m.node->requires_grad();

    // Calculate Reference Grads manually
    // dInput = dOutput * W.T
    auto gy = layer_out.grad();



    // All ops below will now use the stream-aware OwnTensor API
    Tensor dL_ds = OwnTensor::matmul(gy, v.t());
    Tensor dL_dv = OwnTensor::matmul(s.t(), gy);
    
    // VJP of softmax: s * (dL_ds - row_sum(s * dL_ds))
    Tensor dot = OwnTensor::reduce_sum(s * dL_ds, {-1}, true);
    Tensor dL_dg = s * (dL_ds - dot);
    std::cout<<"        fvrwheyjmku  ";
    
    // Propagate gradients back through the Q, K projections
    Tensor dL_dq = OwnTensor::matmul(dL_dg, k);
    Tensor dL_dk = OwnTensor::matmul(dL_dg.t(), q);

    // Propagate gradients to the weight matrices and the input A

        auto bgrad = OwnTensor::matmul(A.t(), dL_dq) * scale;
    
        auto cgrad = OwnTensor::matmul(A.t(), dL_dk) * scale;
        auto dgrad = OwnTensor::matmul(A.t(), dL_dv);
    Tensor dL_dA_q = OwnTensor::matmul(dL_dq, B.t());
    Tensor dL_dA_k = OwnTensor::matmul(dL_dk, C.t());
    Tensor dL_dA_v = OwnTensor::matmul(dL_dv, D.t());
    auto agrad = (dL_dA_q * scale) + (dL_dA_k * scale) + dL_dA_v;
    
    check_tensors(f[0].grad(), bgrad, "Linear Q Grad");
    check_tensors(f[1].grad(), cgrad, "Linear K Grad");
    check_tensors(f[2].grad(), dgrad, "Linear V Grad");
    check_tensors(m.grad(), agrad, "Linear Input Grad");

}

int main() {
    // Load CUDA if layers use it


    test_linear();
    // test_relu_layer();
    test_attention();
    
    return 0;
}