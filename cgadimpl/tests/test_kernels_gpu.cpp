
// -----------------------------
// New tests: gauss, reci, parcon, lisht, gcu
// -----------------------------

// -------------------------
// Custom elementwise tests
// -------------------------

// =========================================================
// FILE: cgadimpl/tests/test_kernels_gpu.cpp
// =========================================================
#include "ad/kernels_api.hpp"
#include "tensor.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>


#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Re-using the same helper from the CPU test
void check_tensors_close(const ag::Tensor& a, const ag::Tensor& b, const std::string& label, float epsilon = 0.65) {
    assert(a.shape() == b.shape());
    for (int r = 0; r < a.rows(); ++r) {
        for (int c = 0; c < a.cols(); ++c) {
            if (std::abs(a(r, c) - b(r, c)) > epsilon) {
                std::cerr << "FAIL: " << label << " mismatch at (" << r << "," << c << ")\n";
                std::cerr << "Tensor A (ref):\n" << a << "\n";
                std::cerr << "Tensor B (out):\n" << b << "\n";
                throw std::runtime_error("Tensor check failed for " + label);
            }
        }
    }
    std::cout << "PASS: " << label << "\n";
}

// --- GPU Memory Helpers ---
float* to_gpu(const ag::Tensor& t) {
    float* d_ptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, t.numel() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_ptr, t.data(), t.numel() * sizeof(float), cudaMemcpyHostToDevice));
    return d_ptr;
}

ag::Tensor from_gpu(const float* d_ptr, int rows, int cols) {
    ag::Tensor t(rows, cols);
    CUDA_CHECK(cudaMemcpy(t.data(), d_ptr, t.numel() * sizeof(float), cudaMemcpyDeviceToHost));
    return t;
}

// --- Test Functions ---

void test_gpu_add() {
    auto& K = ag::kernels::cuda();
    ag::Tensor a_cpu = ag::Tensor::randn(11, 11, 1);
    ag::Tensor b_cpu = ag::Tensor::randn(11, 11, 2);
    std::cout << "\nForward Pass Values:" << std::endl;
    std::cout << "Input A (CPU):\n" << a_cpu << std::endl;
    std::cout << "Input B (CPU):\n" << b_cpu << std::endl;
    ag::Tensor ref = a_cpu + b_cpu;
    std::cout << "Output (CPU):\n" << ref << std::endl;

    float *a_gpu = to_gpu(a_cpu), *b_gpu = to_gpu(b_cpu), *c_gpu;
    CUDA_CHECK(cudaMalloc(&c_gpu, ref.numel() * sizeof(float)));

    K.add(a_gpu, b_gpu, c_gpu, ref.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor out = from_gpu(c_gpu, 11, 11);
    check_tensors_close(ref, out, "test_gpu_add");
    std::cout << "Output (GPU):\n" << out << std::endl;

    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(b_gpu));
    CUDA_CHECK(cudaFree(c_gpu));
}


void test_gpu_unified_tanh() {
    auto& K = ag::kernels::cuda();
    ag::Tensor a_cpu = ag::Tensor::randn(11, 11, 1);
    std::cout << "\nForward Pass Values:" << std::endl;
    std::cout << "Input A (CPU):\n" << a_cpu << std::endl;
    ag::Tensor ref = ag::Tensor::tanh(a_cpu);
    std::cout << "Output (CPU):\n" << ref << std::endl;

    float *a_gpu = to_gpu(a_cpu), *c_gpu;
    CUDA_CHECK(cudaMalloc(&c_gpu, ref.numel() * sizeof(float)));

    K.tanh(a_gpu, c_gpu, ref.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor gy_cpu = ag::Tensor::randn(11, 11, 5);

    ag::Tensor one = ag::Tensor::ones_like(a_cpu);
    ag::Tensor ga_ref = gy_cpu*(one - (ref*ref)); // vjp_add just passes gradient through

    ag::Tensor ga_cpu_init = ag::Tensor::zeros(11, 11);

    float *gy_gpu = to_gpu(gy_cpu);
    float *ga_gpu = to_gpu(ga_cpu_init);

    K.vjp_tanh(ga_gpu, gy_gpu, c_gpu, gy_cpu.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    

    ag::Tensor out = from_gpu(c_gpu, 11, 11);
    check_tensors_close(ref, out, "test_gpu_tanh");
    std::cout << "Output (GPU):\n" << out << std::endl;

    ag::Tensor ga_out = from_gpu(ga_gpu, 11, 11);

    std::cout << "\nBackward Pass Values:" << std::endl;
    std::cout << "Gradient Input dY:\n" << gy_cpu << std::endl;
    std::cout << "Expected dA (CPU):\n" << ga_ref << std::endl;
    std::cout << "Computed dA (GPU):\n" << ga_out << std::endl;

    check_tensors_close(ga_ref, ga_out, "test_gpu_vjp_tanh");





    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(c_gpu));
    CUDA_CHECK(cudaFree(gy_gpu));
    CUDA_CHECK(cudaFree(ga_gpu));
}



void test_gpu_unified_fmab() {
    auto& K = ag::kernels::cuda();
    ag::Tensor a_cpu = ag::Tensor::randn(11, 17, 3);
    ag::Tensor b_cpu = ag::Tensor::randn(17, 9, 4);
    ag::Tensor c_cpu = ag::Tensor::randn(11, 9, 5);
    ag::Tensor e_cpu = ag::Tensor::zeros(11, 9);
    std::cout << "\nForward Pass Values (GEMM):" << std::endl;
    std::cout << "Input A (CPU):\n" << a_cpu << std::endl;
    std::cout << "Input B (CPU):\n" << b_cpu << std::endl;
    std::cout << "Input C (CPU):\n" << c_cpu << std::endl;
    ag::Tensor ref = ag::Tensor::matmul(a_cpu, b_cpu) + c_cpu;
    std::cout << "Output (CPU):\n" << ref << std::endl;

    float *a_gpu = to_gpu(a_cpu), *b_gpu = to_gpu(b_cpu), *c_gpu = to_gpu(c_cpu), *e_gpu=to_gpu(e_cpu);

    K.gemm(a_gpu, b_gpu, c_gpu, e_gpu, 11, 17, 9, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor out = from_gpu(e_gpu, 11, 9);
    check_tensors_close(ref, out, "test_gpu_gemm");
    std::cout << "Output (GPU):\n" << out << std::endl;
    

    ag::Tensor gy_cpu = ag::Tensor::randn(11, 9, 11);

        // Reference calculation on CPU
    ag::Tensor ga_ref = ag::Tensor::matmul(gy_cpu, ag::Tensor::transpose(b_cpu))+ag::Tensor::ones_like(a_cpu);
    ag::Tensor gb_ref = ag::Tensor::matmul(ag::Tensor::transpose(a_cpu), gy_cpu)+ag::Tensor::ones_like(b_cpu);
    ag::Tensor gc_ref = gy_cpu+ag::Tensor::ones_like(c_cpu);

    float *gy_gpu = to_gpu(gy_cpu);
    float *ga_gpu = to_gpu(ag::Tensor::ones_like(a_cpu)), *gb_gpu = to_gpu(ag::Tensor::ones_like(b_cpu)),  *gc_gpu = to_gpu(ag::Tensor::ones_like(c_cpu));



    K.vjp_gemm(ga_gpu, gb_gpu, gc_gpu, gy_gpu, a_gpu, b_gpu, c_gpu, 11, 9, 17, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor ga_out = from_gpu(ga_gpu, 11, 17);
    ag::Tensor gb_out = from_gpu(gb_gpu, 17, 9);
    ag::Tensor gc_out = from_gpu(gc_gpu, 11, 9);

    std::cout << "\nBackward Pass Values (GEMM):" << std::endl;
    std::cout << "Gradient Input dY:\n" << gy_cpu << std::endl;
    std::cout << "Expected dA (CPU):\n" << ga_ref << std::endl;
    std::cout << "Computed dA (GPU):\n" << ga_out << std::endl;
    std::cout << "Expected dB (CPU):\n" << gb_ref << std::endl;
    std::cout << "Computed dB (GPU):\n" << gb_out << std::endl;
    std::cout << "Expected dC (CPU):\n" << gc_ref << std::endl;
    std::cout << "Computed dC (GPU):\n" << gc_out << std::endl;

    check_tensors_close(ga_ref, ga_out, "test_gpu_vjp_gemm (gA)");
    check_tensors_close(gb_ref, gb_out, "test_gpu_vjp_gemm (gB)");
    check_tensors_close(gc_ref, gc_out, "test_gpu_vjp_gemm (gC)");

    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(b_gpu));
    CUDA_CHECK(cudaFree(c_gpu));
    CUDA_CHECK(cudaFree(gy_gpu));
    CUDA_CHECK(cudaFree(ga_gpu));
    CUDA_CHECK(cudaFree(gb_gpu));
    CUDA_CHECK(cudaFree(gc_gpu));
}

void test_gpu_unified_linear() {
    auto& K = ag::kernels::cuda();
    ag::Tensor a_cpu = ag::Tensor::randn(11, 17, 3);
    ag::Tensor b_cpu = ag::Tensor::randn(9, 17, 4);
    ag::Tensor c_cpu = ag::Tensor::randn(11, 9, 5);
    ag::Tensor e_cpu = ag::Tensor::zeros(11, 9);
    std::cout << "\nForward Pass Values (Linear):" << std::endl;
    std::cout << "Input A (CPU):\n" << a_cpu << std::endl;
    std::cout << "Input B (CPU):\n" << b_cpu << std::endl;
    std::cout << "Input C (CPU):\n" << c_cpu << std::endl;
    ag::Tensor ref = ag::Tensor::matmul(a_cpu, ag::Tensor::transpose( b_cpu)) + c_cpu;
    std::cout << "Output (CPU):\n" << ref << std::endl;

    float *a_gpu = to_gpu(a_cpu), *b_gpu = to_gpu(b_cpu), *c_gpu = to_gpu(c_cpu), *e_gpu=to_gpu(e_cpu);

    K.linear(a_gpu, b_gpu, c_gpu, e_gpu, 11, 17, 9, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor out = from_gpu(e_gpu, 11, 9);
    check_tensors_close(ref, out, "test_gpu_linear");
    std::cout << "Output (GPU):\n" << out << std::endl;
    

    ag::Tensor gy_cpu = ag::Tensor::randn(11, 9, 11);

        // Reference calculation on CPU
    ag::Tensor ga_ref = ag::Tensor::matmul(gy_cpu, (b_cpu))+ag::Tensor::ones_like(a_cpu);
    ag::Tensor gb_ref = ag::Tensor::transpose(ag::Tensor::matmul(ag::Tensor::transpose(a_cpu), gy_cpu))+ag::Tensor::ones_like(b_cpu);
    ag::Tensor gc_ref = gy_cpu+ag::Tensor::ones_like(c_cpu);

    float *gy_gpu = to_gpu(gy_cpu);
    float *ga_gpu = to_gpu(ag::Tensor::ones_like(a_cpu)), *gb_gpu = to_gpu(ag::Tensor::ones_like(b_cpu)),  *gc_gpu = to_gpu(ag::Tensor::ones_like(c_cpu));



    K.vjp_linear(ga_gpu, gb_gpu, gc_gpu, gy_gpu, a_gpu, b_gpu, c_gpu, 11, 17, 9, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor ga_out = from_gpu(ga_gpu, 11, 17);
    ag::Tensor gb_out = from_gpu(gb_gpu, 9, 17);
    ag::Tensor gc_out = from_gpu(gc_gpu, 11, 9);

    std::cout << "\nBackward Pass Values (Linear):" << std::endl;
    std::cout << "Gradient Input dY:\n" << gy_cpu << std::endl;
    std::cout << "Expected dA (CPU):\n" << ga_ref << std::endl;
    std::cout << "Computed dA (GPU):\n" << ga_out << std::endl;
    std::cout << "Expected dB (CPU):\n" << gb_ref << std::endl;
    std::cout << "Computed dB (GPU):\n" << gb_out << std::endl;
    std::cout << "Expected dC (CPU):\n" << gc_ref << std::endl;
    std::cout << "Computed dC (GPU):\n" << gc_out << std::endl;

    check_tensors_close(ga_ref, ga_out, "test_gpu_vjp_linear (gA)");
    check_tensors_close(gb_ref, gb_out, "test_gpu_vjp_linear (gB)");
    check_tensors_close(gc_ref, gc_out, "test_gpu_vjp_linear (gC)");

    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(b_gpu));
    CUDA_CHECK(cudaFree(c_gpu));
    CUDA_CHECK(cudaFree(gy_gpu));
    CUDA_CHECK(cudaFree(ga_gpu));
    CUDA_CHECK(cudaFree(gb_gpu));
    CUDA_CHECK(cudaFree(gc_gpu));
}



void test_gpu_unified_attention() {
    auto& K = ag::kernels::cuda();
    ag::Tensor a_cpu = ag::Tensor::randn(11, 9, 3);
    ag::Tensor b_cpu = ag::Tensor::randn(9, 14, 4);
    ag::Tensor c_cpu = ag::Tensor::randn(9, 14, 5);
    ag::Tensor d_cpu = ag::Tensor::randn(9, 14, 6);
    ag::Tensor e_cpu = ag::Tensor::zeros(11, 14);
    std::cout << "\nForward Pass Values (Linear):" << std::endl;
    // std::cout << "Input A (CPU):\n" << a_cpu << std::endl;
    // std::cout << "Input B (CPU):\n" << b_cpu << std::endl;
    // std::cout << "Input C (CPU):\n" << c_cpu << std::endl;
    // std::cout << "Input D (CPU):\n" << d_cpu << std::endl;
                    ag::Tensor q = ag::Tensor::matmul(a_cpu, b_cpu); 
    auto k = ag::Tensor::matmul(a_cpu, c_cpu); 
    auto v = ag::Tensor::matmul(a_cpu, d_cpu);
    auto g = ag::Tensor::matmul(q, ag::Tensor::transpose(k)*(1.f/sqrt(float(k.cols())))) ;
    auto s = ag::Tensor::softmax_row(g);
    auto ref = ag::Tensor::matmul(s, v);
    std::cout << "Output (CPU):\n" << ref << std::endl;

    float *q_gpu = to_gpu(q), *k_gpu = to_gpu(k), *v_gpu = to_gpu(v), *e_gpu=to_gpu(e_cpu);

    K.flash(q_gpu, k_gpu, v_gpu, e_gpu, 1, 1, 11, 14, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor out = from_gpu(e_gpu, 11, 14);
    check_tensors_close(ref, out, "test_gpu_attention");
    std::cout << "Output (GPU):\n" << out << std::endl;
    

    // ag::Tensor gy_cpu = ag::Tensor::randn(11, 9, 11);

    //     // Reference calculation on CPU
    // ag::Tensor ga_ref = ag::Tensor::matmul(gy_cpu, (b_cpu))+ag::Tensor::ones_like(a_cpu);
    // ag::Tensor gb_ref = ag::Tensor::transpose(ag::Tensor::matmul(ag::Tensor::transpose(a_cpu), gy_cpu))+ag::Tensor::ones_like(b_cpu);
    // ag::Tensor gc_ref = gy_cpu+ag::Tensor::ones_like(c_cpu);

    // float *gy_gpu = to_gpu(gy_cpu);
    // float *ga_gpu = to_gpu(ag::Tensor::ones_like(a_cpu)), *gb_gpu = to_gpu(ag::Tensor::ones_like(b_cpu)),  *gc_gpu = to_gpu(ag::Tensor::ones_like(c_cpu));



    // K.vjp_linear(ga_gpu, gb_gpu, gc_gpu, gy_gpu, a_gpu, b_gpu, c_gpu, 11, 17, 9, nullptr);
    // CUDA_CHECK(cudaDeviceSynchronize());

    // ag::Tensor ga_out = from_gpu(ga_gpu, 11, 17);
    // ag::Tensor gb_out = from_gpu(gb_gpu, 9, 17);
    // ag::Tensor gc_out = from_gpu(gc_gpu, 11, 9);

    // std::cout << "\nBackward Pass Values (Linear):" << std::endl;
    // std::cout << "Gradient Input dY:\n" << gy_cpu << std::endl;
    // std::cout << "Expected dA (CPU):\n" << ga_ref << std::endl;
    // std::cout << "Computed dA (GPU):\n" << ga_out << std::endl;
    // std::cout << "Expected dB (CPU):\n" << gb_ref << std::endl;
    // std::cout << "Computed dB (GPU):\n" << gb_out << std::endl;
    // std::cout << "Expected dC (CPU):\n" << gc_ref << std::endl;
    // std::cout << "Computed dC (GPU):\n" << gc_out << std::endl;

    // check_tensors_close(ga_ref, ga_out, "test_gpu_vjp_linear (gA)");
    // check_tensors_close(gb_ref, gb_out, "test_gpu_vjp_linear (gB)");
    // check_tensors_close(gc_ref, gc_out, "test_gpu_vjp_linear (gC)");

    CUDA_CHECK(cudaFree(q_gpu));
    CUDA_CHECK(cudaFree(k_gpu));
    CUDA_CHECK(cudaFree(v_gpu));
    CUDA_CHECK(cudaFree(e_gpu));
    // CUDA_CHECK(cudaFree(ga_gpu));
    // CUDA_CHECK(cudaFree(gb_gpu));
    // CUDA_CHECK(cudaFree(gc_gpu));
}

void test_gpu_matmul() {
    auto& K = ag::kernels::cuda();
    ag::Tensor a_cpu = ag::Tensor::randn(11, 17, 3);
    ag::Tensor b_cpu = ag::Tensor::randn(17, 11, 4);
    std::cout << "\nForward Pass Values (MatMul):" << std::endl;
    std::cout << "Input A (CPU):\n" << a_cpu << std::endl;
    std::cout << "Input B (CPU):\n" << b_cpu << std::endl;
    ag::Tensor ref = ag::Tensor::matmul(a_cpu, b_cpu);
    std::cout << "Output (CPU):\n" << ref << std::endl;

    float *a_gpu = to_gpu(a_cpu), *b_gpu = to_gpu(b_cpu), *c_gpu;
    CUDA_CHECK(cudaMalloc(&c_gpu, ref.numel() * sizeof(float)));

    K.matmul(a_gpu, b_gpu, c_gpu, 11, 17, 11, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor out = from_gpu(c_gpu, 11, 11);
    check_tensors_close(ref, out, "test_gpu_matmul");
    std::cout << "Output (GPU):\n" << out << std::endl;

    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(b_gpu));
    CUDA_CHECK(cudaFree(c_gpu));
}

void test_gpu_vjp_add() {
    auto& K = ag::kernels::cuda();
    ag::Tensor gy_cpu = ag::Tensor::randn(11, 11, 5);
    ag::Tensor ga_ref = gy_cpu; // vjp_add just passes gradient through
    ag::Tensor gb_ref = gy_cpu;

    ag::Tensor ga_cpu_init = ag::Tensor::zeros(11, 11);
    ag::Tensor gb_cpu_init = ag::Tensor::zeros(11, 11);

    float *gy_gpu = to_gpu(gy_cpu);
    float *ga_gpu = to_gpu(ga_cpu_init);
    float *gb_gpu = to_gpu(gb_cpu_init);

    K.vjp_add(ga_gpu, gb_gpu, gy_gpu, gy_cpu.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor ga_out = from_gpu(ga_gpu, 11, 11);
    ag::Tensor gb_out = from_gpu(gb_gpu, 11, 11);

    std::cout << "\nBackward Pass Values (vjp_add):" << std::endl;
    std::cout << "Gradient Input dY:\n" << gy_cpu << std::endl;
    std::cout << "Expected dA (CPU):\n" << ga_ref << std::endl;
    std::cout << "Computed dA (GPU):\n" << ga_out << std::endl;
    std::cout << "Expected dB (CPU):\n" << gb_ref << std::endl;
    std::cout << "Computed dB (GPU):\n" << gb_out << std::endl;

    check_tensors_close(ga_ref, ga_out, "test_gpu_vjp_add (gA)");
    check_tensors_close(gb_ref, gb_out, "test_gpu_vjp_add (gB)");

    CUDA_CHECK(cudaFree(gy_gpu));
    CUDA_CHECK(cudaFree(ga_gpu));
    CUDA_CHECK(cudaFree(gb_gpu));
}


void test_gpu_unified_sub() {
        auto& K = ag::kernels::cuda();
    ag::Tensor a_cpu = ag::Tensor::randn(11, 11, 1);
    ag::Tensor b_cpu = ag::Tensor::randn(11, 11, 2);
    std::cout << "\nForward Pass Values:" << std::endl;
    std::cout << "Input A (CPU):\n" << a_cpu << std::endl;
    std::cout << "Input B (CPU):\n" << b_cpu << std::endl;
    ag::Tensor ref = a_cpu - b_cpu;
    std::cout << "Output (CPU):\n" << ref << std::endl;

    float *a_gpu = to_gpu(a_cpu), *b_gpu = to_gpu(b_cpu), *c_gpu;
    CUDA_CHECK(cudaMalloc(&c_gpu, ref.numel() * sizeof(float)));

    K.sub(a_gpu, b_gpu, c_gpu, ref.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor gy_cpu = ag::Tensor::randn(11, 11, 5);
    ag::Tensor ga_ref = gy_cpu; // vjp_add just passes gradient through
    ag::Tensor gb_ref = -gy_cpu;

    ag::Tensor ga_cpu_init = ag::Tensor::zeros(11, 11);
    ag::Tensor gb_cpu_init = ag::Tensor::zeros(11, 11);

    float *gy_gpu = to_gpu(gy_cpu);
    float *ga_gpu = to_gpu(ga_cpu_init);
    float *gb_gpu = to_gpu(gb_cpu_init);

    K.vjp_sub(ga_gpu, gb_gpu, gy_gpu, gy_cpu.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor ga_out = from_gpu(ga_gpu, 11, 11);
    ag::Tensor gb_out = from_gpu(gb_gpu, 11, 11);

        ag::Tensor out = from_gpu(c_gpu, 11, 11);
    check_tensors_close(ref, out, "test_gpu_sub");

    std::cout << "Output (GPU):\n" << out << std::endl;

    std::cout << "\nBackward Pass Values:" << std::endl;
    std::cout << "Gradient Input dY:\n" << gy_cpu << std::endl;
    std::cout << "Expected dA (CPU):\n" << ga_ref << std::endl;
    std::cout << "Computed dA (GPU):\n" << ga_out << std::endl;
    std::cout << "Expected dB (CPU):\n" << gb_ref << std::endl;
    std::cout << "Computed dB (GPU):\n" << gb_out << std::endl;

    check_tensors_close(ga_ref, ga_out, "test_gpu_vjp_sub (gA)");
    check_tensors_close(gb_ref, gb_out, "test_gpu_vjp_sub (gB)");

    CUDA_CHECK(cudaFree(gy_gpu));
    CUDA_CHECK(cudaFree(ga_gpu));
    CUDA_CHECK(cudaFree(gb_gpu));
}

void test_gpu_unified_hadmul() {
        auto& K = ag::kernels::cuda();
    ag::Tensor a_cpu = ag::Tensor::randn(11, 11, 1);
    ag::Tensor b_cpu = ag::Tensor::randn(11, 11, 2);

    ag::Tensor ref = a_cpu * b_cpu;

    float *a_gpu = to_gpu(a_cpu), *b_gpu = to_gpu(b_cpu), *c_gpu;
    CUDA_CHECK(cudaMalloc(&c_gpu, ref.numel() * sizeof(float)));

    K.hadmul(a_gpu, b_gpu, c_gpu, ref.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor gy_cpu = ag::Tensor::ones(11, 11);
    
    ag::Tensor ga_ref = b_cpu; // vjp_add just passes gradient through
    ag::Tensor gb_ref = a_cpu;


    ag::Tensor ga_cpu_init = ag::Tensor::zeros(11, 11);
    ag::Tensor gb_cpu_init = ag::Tensor::zeros(11, 11);

    float *gy_gpu = to_gpu(gy_cpu);
    float *ga_gpu = to_gpu(ga_cpu_init);
    float *gb_gpu = to_gpu(gb_cpu_init);

    K.vjp_hadmul(ga_gpu, gb_gpu, gy_gpu, a_gpu, b_gpu, gy_cpu.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor ga_out = from_gpu(ga_gpu, 11, 11);
    ag::Tensor gb_out = from_gpu(gb_gpu, 11, 11);

        ag::Tensor out = from_gpu(c_gpu, 11, 11);

            check_tensors_close(ref, out, "test_gpu_hadmul");

                std::cout << "\nForward Pass Values:" << std::endl;
    std::cout << "Input A (CPU):\n" << a_cpu << std::endl;
    std::cout << "Input B (CPU):\n" << b_cpu << std::endl;
                std::cout << "Output (CPU):\n" << ref << std::endl;

        std::cout << "Output (GPU):\n" << ref << std::endl;

        std::cout << "\n\n\nBackward Pass Values:" << std::endl;
    std::cout << "Gradient Input dY:\n" << gy_cpu << std::endl;




        std::cout << "Expected Gradients (CPU):" << std::endl;
    std::cout << "dA:\n" << ga_ref << std::endl;
    std::cout << "dB:\n" << gb_ref << std::endl;

    std::cout << "Expected Gradients (GPU):" << std::endl;
    std::cout << "dA:\n" << ga_out << std::endl;
    std::cout << "dB:\n" << gb_out << std::endl;

    check_tensors_close(ga_ref, ga_out, "test_gpu_vjp_hadmul (gA)");
    check_tensors_close(gb_ref, gb_out, "test_gpu_vjp_hadmul (gB)");
    CUDA_CHECK(cudaFree(gy_gpu));
    CUDA_CHECK(cudaFree(ga_gpu));
    CUDA_CHECK(cudaFree(gb_gpu));
}


void test_gpu_unified_div() {
        auto& K = ag::kernels::cuda();
    ag::Tensor a_cpu = ag::Tensor::randn(11, 11, 1);
    ag::Tensor b_cpu = ag::Tensor::randn(11, 11, 2);
    std::cout << "\nForward Pass Values:" << std::endl;
    std::cout << "Input A (CPU):\n" << a_cpu << std::endl;
    std::cout << "Input B (CPU):\n" << b_cpu << std::endl;
    ag::Tensor ref = a_cpu / b_cpu;
    std::cout << "Output (CPU):\n" << ref << std::endl;

    float *a_gpu = to_gpu(a_cpu), *b_gpu = to_gpu(b_cpu), *c_gpu;
    CUDA_CHECK(cudaMalloc(&c_gpu, ref.numel() * sizeof(float)));

    K.div(a_gpu, b_gpu, c_gpu, ref.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor gy_cpu = ag::Tensor::ones(11, 11);
    ag::Tensor ga_ref = 1.f /b_cpu; // vjp_add just passes gradient through
    ag::Tensor gb_ref = a_cpu/(-b_cpu * b_cpu);

    ag::Tensor ga_cpu_init = ag::Tensor::zeros(11, 11);
    ag::Tensor gb_cpu_init = ag::Tensor::zeros(11, 11);

    float *gy_gpu = to_gpu(gy_cpu);
    float *ga_gpu = to_gpu(ga_cpu_init);
    float *gb_gpu = to_gpu(gb_cpu_init);

    K.vjp_div(ga_gpu, gb_gpu, gy_gpu, a_gpu, b_gpu, gy_cpu.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor ga_out = from_gpu(ga_gpu, 11, 11);
    ag::Tensor gb_out = from_gpu(gb_gpu, 11, 11);

        ag::Tensor out = from_gpu(c_gpu, 11, 11);
    check_tensors_close(ref, out, "test_gpu_div");

    std::cout << "Output (GPU):\n" << out << std::endl;

    std::cout << "\nBackward Pass Values:" << std::endl;
    std::cout << "Gradient Input dY:\n" << gy_cpu << std::endl;
    std::cout << "Expected dA (CPU):\n" << ga_ref << std::endl;
    std::cout << "Computed dA (GPU):\n" << ga_out << std::endl;
    std::cout << "Expected dB (CPU):\n" << gb_ref << std::endl;
    std::cout << "Computed dB (GPU):\n" << gb_out << std::endl;

    check_tensors_close(ga_ref, ga_out, "test_gpu_vjp_div (gA)");
    check_tensors_close(gb_ref, gb_out, "test_gpu_vjp_div (gB)");

    CUDA_CHECK(cudaFree(gy_gpu));
    CUDA_CHECK(cudaFree(ga_gpu));
    CUDA_CHECK(cudaFree(gb_gpu));
}

void test_gpu_vjp_matmul() {
    auto& K = ag::kernels::cuda();
    ag::Tensor a_cpu = ag::Tensor::randn(11, 17, 6);
    ag::Tensor b_cpu = ag::Tensor::randn(17, 11, 7);
    ag::Tensor gy_cpu = ag::Tensor::randn(11, 11, 11);

    // Reference calculation on CPU
    ag::Tensor ga_ref = ag::Tensor::matmul(gy_cpu, ag::Tensor::transpose(b_cpu));
    ag::Tensor gb_ref = ag::Tensor::matmul(ag::Tensor::transpose(a_cpu), gy_cpu);

    float *a_gpu = to_gpu(a_cpu), *b_gpu = to_gpu(b_cpu), *gy_gpu = to_gpu(gy_cpu);
    float *ga_gpu, *gb_gpu;
    CUDA_CHECK(cudaMalloc(&ga_gpu, ga_ref.numel() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gb_gpu, gb_ref.numel() * sizeof(float)));

    K.vjp_matmul(ga_gpu, gb_gpu, gy_gpu, a_gpu, b_gpu, 11, 17, 11, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor ga_out = from_gpu(ga_gpu, 11, 17);
    ag::Tensor gb_out = from_gpu(gb_gpu, 17, 11);

    std::cout << "\nBackward Pass Values (vjp_matmul):" << std::endl;
    std::cout << "Gradient Input dY:\n" << gy_cpu << std::endl;
    std::cout << "Expected dA (CPU):\n" << ga_ref << std::endl;
    std::cout << "Computed dA (GPU):\n" << ga_out << std::endl;
    std::cout << "Expected dB (CPU):\n" << gb_ref << std::endl;
    std::cout << "Computed dB (GPU):\n" << gb_out << std::endl;

    check_tensors_close(ga_ref, ga_out, "test_gpu_vjp_matmul (gA)");
    check_tensors_close(gb_ref, gb_out, "test_gpu_vjp_matmul (gB)");

    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(b_gpu));
    CUDA_CHECK(cudaFree(gy_gpu));
    CUDA_CHECK(cudaFree(ga_gpu));
    CUDA_CHECK(cudaFree(gb_gpu));
}







// ---------------------------------------------------------
// POW
// // ---------------------------------------------------------
// void test_gpu_unified_pow() {
//     auto& K = ag::kernels::cuda();
//     ag::Tensor a_cpu = ag::Tensor::randn(11, 11, 1).abs() + 0.1f;
//     ag::Tensor b_cpu = ag::Tensor::randn(11, 11, 2);
//     ag::Tensor ref = ag::Tensor::pow(a_cpu, b_cpu);

//     float *a_gpu = to_gpu(a_cpu), *b_gpu = to_gpu(b_cpu), *c_gpu;
//     CUDA_CHECK(cudaMalloc(&c_gpu, ref.numel() * sizeof(float)));

//     K.pow(a_gpu, b_gpu, c_gpu, ref.numel(), nullptr);
//     CUDA_CHECK(cudaDeviceSynchronize());

//     ag::Tensor out = from_gpu(c_gpu, 11, 11);
//     check_tensors_close(ref, out, "test_gpu_pow");

//     ag::Tensor gy_cpu = ag::Tensor::randn(11, 11, 3);
//     ag::Tensor ga_ref = gy_cpu * b_cpu * ag::Tensor::pow(a_cpu, b_cpu - 1.0f);
//     ag::Tensor gb_ref = gy_cpu * ref * ag::Tensor::log(a_cpu);

//     float *gy_gpu = to_gpu(gy_cpu);
//     float *ga_gpu = to_gpu(ag::Tensor::zeros_like(a_cpu));
//     float *gb_gpu = to_gpu(ag::Tensor::zeros_like(b_cpu));

//     K.vjp_pow(ga_gpu, gb_gpu, a_gpu, b_gpu, gy_gpu, ref.numel(), nullptr);
//     CUDA_CHECK(cudaDeviceSynchronize());

//     ag::Tensor ga_out = from_gpu(ga_gpu, 11, 11);
//     ag::Tensor gb_out = from_gpu(gb_gpu, 11, 11);
//     check_tensors_close(ga_ref, ga_out, "test_gpu_vjp_pow (gA)");
//     check_tensors_close(gb_ref, gb_out, "test_gpu_vjp_pow (gB)");

//     CUDA_CHECK(cudaFree(a_gpu));
//     CUDA_CHECK(cudaFree(b_gpu));
//     CUDA_CHECK(cudaFree(c_gpu));
//     CUDA_CHECK(cudaFree(gy_gpu));
//     CUDA_CHECK(cudaFree(ga_gpu));
//     CUDA_CHECK(cudaFree(gb_gpu));
// }

// ---------------------------------------------------------
// SQUARE
// ---------------------------------------------------------
void test_gpu_unified_square() {
    auto& K = ag::kernels::cuda();
    ag::Tensor x_cpu = ag::Tensor::randn(11, 11, 1);
    std::cout << "\nForward Pass Values (Square):" << std::endl;
    std::cout << "Input X (CPU):\n" << x_cpu << std::endl;
    ag::Tensor ref = x_cpu * x_cpu;
    std::cout << "Output (CPU):\n" << ref << std::endl;

    float *x_gpu = to_gpu(x_cpu), *y_gpu;
    CUDA_CHECK(cudaMalloc(&y_gpu, ref.numel() * sizeof(float)));

    K.square(x_gpu, y_gpu, ref.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor out = from_gpu(y_gpu, 11, 11);
    check_tensors_close(ref, out, "test_gpu_square");
    std::cout << "Output (GPU):\n" << out << std::endl;

    ag::Tensor gy_cpu = ag::Tensor::randn(11, 11, 2);
    ag::Tensor ga_ref = gy_cpu * (2.0f * x_cpu);

    float *gy_gpu = to_gpu(gy_cpu);
    float *ga_gpu = to_gpu(ag::Tensor::zeros_like(x_cpu));

    K.vjp_square(ga_gpu, x_gpu, gy_gpu, ref.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor ga_out = from_gpu(ga_gpu, 11, 11);
    std::cout << "\nBackward Pass Values (Square):" << std::endl;
    std::cout << "Gradient Input dY:\n" << gy_cpu << std::endl;
    std::cout << "Expected dX (CPU):\n" << ga_ref << std::endl;
    std::cout << "Computed dX (GPU):\n" << ga_out << std::endl;
    check_tensors_close(ga_ref, ga_out, "test_gpu_vjp_square");

    CUDA_CHECK(cudaFree(x_gpu));
    CUDA_CHECK(cudaFree(y_gpu));
    CUDA_CHECK(cudaFree(gy_gpu));
    CUDA_CHECK(cudaFree(ga_gpu));
}

// ---------------------------------------------------------
// NEG
// ---------------------------------------------------------
void test_gpu_unified_neg() {
    auto& K = ag::kernels::cuda();
    ag::Tensor x_cpu = ag::Tensor::randn(11, 11, 1);
    std::cout << "\nForward Pass Values (Neg):" << std::endl;
    std::cout << "Input X (CPU):\n" << x_cpu << std::endl;
    ag::Tensor ref = -x_cpu;
    std::cout << "Output (CPU):\n" << ref << std::endl;

    float *x_gpu = to_gpu(x_cpu), *y_gpu;
    CUDA_CHECK(cudaMalloc(&y_gpu, ref.numel() * sizeof(float)));

    K.neg(x_gpu, y_gpu, ref.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor out = from_gpu(y_gpu, 11, 11);
    check_tensors_close(ref, out, "test_gpu_neg");
    std::cout << "Output (GPU):\n" << out << std::endl;

    ag::Tensor gy_cpu = ag::Tensor::randn(11, 11, 2);
    ag::Tensor ga_ref = -gy_cpu;

    float *gy_gpu = to_gpu(gy_cpu);
    float *ga_gpu = to_gpu(ag::Tensor::zeros_like(x_cpu));

    K.vjp_neg(ga_gpu, gy_gpu, ref.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor ga_out = from_gpu(ga_gpu, 11, 11);
    std::cout << "\nBackward Pass Values (Neg):" << std::endl;
    std::cout << "Gradient Input dY:\n" << gy_cpu << std::endl;
    std::cout << "Expected dX (CPU):\n" << ga_ref << std::endl;
    std::cout << "Computed dX (GPU):\n" << ga_out << std::endl;
    check_tensors_close(ga_ref, ga_out, "test_gpu_vjp_neg");

    CUDA_CHECK(cudaFree(x_gpu));
    CUDA_CHECK(cudaFree(y_gpu));
    CUDA_CHECK(cudaFree(gy_gpu));
    CUDA_CHECK(cudaFree(ga_gpu));
}

// ---------------------------------------------------------
// CLIP
// ---------------------------------------------------------
// void test_gpu_unified_clip() {
//     auto& K = ag::kernels::cuda();
//     ag::Tensor x_cpu = ag::Tensor::randn(11, 11, 1);
//     float minv = -0.5f, maxv = 0.5f;
//     ag::Tensor ref = ag::Tensor::clip(x_cpu, minv, maxv);

//     float *x_gpu = to_gpu(x_cpu), *y_gpu;
//     CUDA_CHECK(cudaMalloc(&y_gpu, ref.numel() * sizeof(float)));

//     K.clip(x_gpu, y_gpu, minv, maxv, ref.numel(), nullptr);
//     CUDA_CHECK(cudaDeviceSynchronize());

//     ag::Tensor out = from_gpu(y_gpu, 11, 11);
//     check_tensors_close(ref, out, "test_gpu_clip");

//     ag::Tensor gy_cpu = ag::Tensor::randn(11, 11, 2);
//     ag::Tensor mask = (x_cpu > minv) * (x_cpu < maxv);
//     ag::Tensor ga_ref = gy_cpu * mask;

//     float *gy_gpu = to_gpu(gy_cpu);
//     float *ga_gpu = to_gpu(ag::Tensor::zeros_like(x_cpu));

//     K.vjp_clip(ga_gpu, x_gpu, minv, maxv, gy_gpu, ref.numel(), nullptr);
//     CUDA_CHECK(cudaDeviceSynchronize());

//     ag::Tensor ga_out = from_gpu(ga_gpu, 11, 11);
//     check_tensors_close(ga_ref, ga_out, "test_gpu_vjp_clip");

//     CUDA_CHECK(cudaFree(x_gpu));
//     CUDA_CHECK(cudaFree(y_gpu));
//     CUDA_CHECK(cudaFree(gy_gpu));
//     CUDA_CHECK(cudaFree(ga_gpu));
// }






// ---------------------------------------------------------
// SIGMOID
// ---------------------------------------------------------
void test_gpu_unified_sigmoid() {
    auto& K = ag::kernels::cuda();
    ag::Tensor x_cpu = ag::Tensor::randn(11, 11, 1);
    std::cout << "\nForward Pass Values (Sigmoid):" << std::endl;
    std::cout << "Input X (CPU):\n" << x_cpu << std::endl;
    ag::Tensor ref = ag::Tensor::sigmoid(x_cpu);
    std::cout << "Output (CPU):\n" << ref << std::endl;

    float *x_gpu = to_gpu(x_cpu), *y_gpu;
    CUDA_CHECK(cudaMalloc(&y_gpu, ref.numel() * sizeof(float)));

    K.sigmoid(x_gpu, y_gpu, ref.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor out = from_gpu(y_gpu, 11, 11);
    check_tensors_close(ref, out, "test_gpu_sigmoid");
    std::cout << "Output (GPU):\n" << out << std::endl;

    ag::Tensor gy_cpu = ag::Tensor::ones(11, 11);
    ag::Tensor ga_ref = gy_cpu * ref * (ag::Tensor::ones_like(ref) - ref);

    float *gy_gpu = to_gpu(gy_cpu);
    float *ga_gpu = to_gpu(ag::Tensor::zeros_like(x_cpu));

    K.vjp_sigmoid(ga_gpu, x_gpu, gy_gpu, ref.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor ga_out = from_gpu(ga_gpu, 11, 11);
    std::cout << "\nBackward Pass Values (Sigmoid):" << std::endl;
    std::cout << "Gradient Input dY:\n" << gy_cpu << std::endl;
    std::cout << "Expected dX (CPU):\n" << ga_ref << std::endl;
    std::cout << "Computed dX (GPU):\n" << ga_out << std::endl;
    check_tensors_close(ga_ref, ga_out, "test_gpu_vjp_sigmoid");

    CUDA_CHECK(cudaFree(x_gpu));
    CUDA_CHECK(cudaFree(y_gpu));
    CUDA_CHECK(cudaFree(gy_gpu));
    CUDA_CHECK(cudaFree(ga_gpu));
}

// ---------------------------------------------------------
// SILU (x * sigmoid(x))
// ---------------------------------------------------------
void test_gpu_unified_silu() {
    auto& K = ag::kernels::cuda();
    ag::Tensor x_cpu = ag::Tensor::randn(11, 11, 1);
    std::cout << "\nForward Pass Values (SiLU):" << std::endl;
    std::cout << "Input X (CPU):\n" << x_cpu << std::endl;
    ag::Tensor s = ag::Tensor::sigmoid(x_cpu);
    ag::Tensor ref = x_cpu * s;
    std::cout << "Output (CPU):\n" << ref << std::endl;

    float *x_gpu = to_gpu(x_cpu), *y_gpu;
    CUDA_CHECK(cudaMalloc(&y_gpu, ref.numel() * sizeof(float)));

    K.silu(x_gpu, y_gpu, ref.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor out = from_gpu(y_gpu, 11, 11);
    check_tensors_close(ref, out, "test_gpu_silu");
    std::cout << "Output (GPU):\n" << out << std::endl;

    ag::Tensor gy_cpu = ag::Tensor::randn(11, 11, 2);
    ag::Tensor ga_ref = gy_cpu * (s + x_cpu * s * (ag::Tensor::ones_like(s)-s));

    float *gy_gpu = to_gpu(gy_cpu);
    float *ga_gpu = to_gpu(ag::Tensor::zeros_like(x_cpu));

    K.vjp_silu(ga_gpu, gy_gpu, x_gpu, ref.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor ga_out = from_gpu(ga_gpu, 11, 11);
    std::cout << "\nBackward Pass Values (SiLU):" << std::endl;
    std::cout << "Gradient Input dY:\n" << gy_cpu << std::endl;
    std::cout << "Expected dX (CPU):\n" << ga_ref << std::endl;
    std::cout << "Computed dX (GPU):\n" << ga_out << std::endl;
    check_tensors_close(ga_ref, ga_out, "test_gpu_vjp_silu");

    CUDA_CHECK(cudaFree(x_gpu));
    CUDA_CHECK(cudaFree(y_gpu));
    CUDA_CHECK(cudaFree(gy_gpu));
    CUDA_CHECK(cudaFree(ga_gpu));
}

// ---------------------------------------------------------
// GELU (approximation)
// ---------------------------------------------------------
// void test_gpu_unified_gelu() {
//     auto& K = ag::kernels::cuda();
//     ag::Tensor x_cpu = ag::Tensor::randn(11, 11, 1);
//     // Approximate GELU: 0.5 * x * (1 + tanh(√(2/π)*(x + 0.044715x³)))
//     ag::Tensor ref = ag::Tensor::gelu(x_cpu);

//     float *x_gpu = to_gpu(x_cpu), *y_gpu;
//     CUDA_CHECK(cudaMalloc(&y_gpu, ref.numel() * sizeof(float)));

//     K.gelu(x_gpu, y_gpu, ref.numel(), nullptr);
//     CUDA_CHECK(cudaDeviceSynchronize());

//     ag::Tensor out = from_gpu(y_gpu, 11, 11);
//     check_tensors_close(ref, out, "test_gpu_gelu");

//     ag::Tensor gy_cpu = ag::Tensor::randn(11, 11, 2);
//     // For testing we can use ag::Tensor::gelu_backward(x, gy)
//     ag::Tensor ga_ref = ag::Tensor::gelu_backward(x_cpu, gy_cpu);

//     float *gy_gpu = to_gpu(gy_cpu);
//     float *ga_gpu = to_gpu(ag::Tensor::zeros_like(x_cpu));

//     K.vjp_gelu(ga_gpu, gy_gpu, x_gpu, ref.numel(), nullptr);
//     CUDA_CHECK(cudaDeviceSynchronize());

//     ag::Tensor ga_out = from_gpu(ga_gpu, 11, 11);
//     check_tensors_close(ga_ref, ga_out, "test_gpu_vjp_gelu");

//     CUDA_CHECK(cudaFree(x_gpu));
//     CUDA_CHECK(cudaFree(y_gpu));
//     CUDA_CHECK(cudaFree(gy_gpu));
//     CUDA_CHECK(cudaFree(ga_gpu));
// }

// ---------------------------------------------------------
// MISH (x * tanh(softplus(x)))
// ---------------------------------------------------------
void test_gpu_unified_mish() {
    auto& K = ag::kernels::cuda();
    ag::Tensor x_cpu = ag::Tensor::randn(11, 11, 1);
    std::cout << "\nForward Pass Values (Mish):" << std::endl;
    std::cout << "Input X (CPU):\n" << x_cpu << std::endl;
    ag::Tensor sp = ag::Tensor::softplus(x_cpu);
    ag::Tensor ref = x_cpu * ag::Tensor::tanh(sp);
    std::cout << "Output (CPU):\n" << ref << std::endl;

    float *x_gpu = to_gpu(x_cpu), *y_gpu;
    CUDA_CHECK(cudaMalloc(&y_gpu, ref.numel() * sizeof(float)));

    K.mish(x_gpu, y_gpu, ref.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor out = from_gpu(y_gpu, 11, 11);
    check_tensors_close(ref, out, "test_gpu_mish");
    std::cout << "Output (GPU):\n" << out << std::endl;

    ag::Tensor gy_cpu = ag::Tensor::randn(11, 11, 2);
    ag::Tensor s = ag::Tensor::sigmoid(x_cpu);
    ag::Tensor t = ag::Tensor::tanh(sp);
    ag::Tensor ga_ref = gy_cpu * (t + x_cpu * s * (ag::Tensor::ones_like(t) - t * t));

    float *gy_gpu = to_gpu(gy_cpu);
    float *ga_gpu = to_gpu(ag::Tensor::zeros_like(x_cpu));

    K.vjp_mish(ga_gpu, x_gpu, gy_gpu, ref.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor ga_out = from_gpu(ga_gpu, 11, 11);
    std::cout << "\nBackward Pass Values (Mish):" << std::endl;
    std::cout << "Gradient Input dY:\n" << gy_cpu << std::endl;
    std::cout << "Expected dX (CPU):\n" << ga_ref << std::endl;
    std::cout << "Computed dX (GPU):\n" << ga_out << std::endl;
    check_tensors_close(ga_ref, ga_out, "test_gpu_vjp_mish");

    CUDA_CHECK(cudaFree(x_gpu));
    CUDA_CHECK(cudaFree(y_gpu));
    CUDA_CHECK(cudaFree(gy_gpu));
    CUDA_CHECK(cudaFree(ga_gpu));
}

// ---------------------------------------------------------
// EXP
// ---------------------------------------------------------
void test_gpu_unified_exp() {
    auto& K = ag::kernels::cuda();
    ag::Tensor x_cpu = ag::Tensor::randn(11, 11, 1);
    std::cout << "\nForward Pass Values (Exp):" << std::endl;
    std::cout << "Input X (CPU):\n" << x_cpu << std::endl;
    ag::Tensor ref = ag::Tensor::exp(x_cpu);
    std::cout << "Output (CPU):\n" << ref << std::endl;

    float *x_gpu = to_gpu(x_cpu), *y_gpu;
    CUDA_CHECK(cudaMalloc(&y_gpu, ref.numel() * sizeof(float)));

    K.exp(x_gpu, y_gpu, ref.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor out = from_gpu(y_gpu, 11, 11);
    check_tensors_close(ref, out, "test_gpu_exp");
    std::cout << "Output (GPU):\n" << out << std::endl;

    ag::Tensor gy_cpu = ag::Tensor::ones(11, 11);
    ag::Tensor ga_ref = gy_cpu * ref;

    float *gy_gpu = to_gpu(gy_cpu);
    float *ga_gpu = to_gpu(ag::Tensor::zeros_like(x_cpu));

    K.vjp_exp(ga_gpu, x_gpu, gy_gpu, ref.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor ga_out = from_gpu(ga_gpu, 11, 11);
    std::cout << "\nBackward Pass Values (Exp):" << std::endl;
    std::cout << "Gradient Input dY:\n" << gy_cpu << std::endl;
    std::cout << "Expected dX (CPU):\n" << ga_ref << std::endl;
    std::cout << "Computed dX (GPU):\n" << ga_out << std::endl;
    check_tensors_close(ga_ref, ga_out, "test_gpu_vjp_exp");

    CUDA_CHECK(cudaFree(x_gpu));
    CUDA_CHECK(cudaFree(y_gpu));
    CUDA_CHECK(cudaFree(gy_gpu));
    CUDA_CHECK(cudaFree(ga_gpu));
}

// ---------------------------------------------------------
// HARD SWISH (x * hard_sigmoid(x))
// // ---------------------------------------------------------
// void test_gpu_unified_hard_swish() {
//     auto& K = ag::kernels::cuda();
//     ag::Tensor x_cpu = ag::Tensor::randn(11, 11, 1);
//     ag::Tensor hs = ag::Tensor::clip(x_cpu * 0.2f + 0.5f, 0.0f, 1.0f);
//     ag::Tensor ref = x_cpu * hs;

//     float *x_gpu = to_gpu(x_cpu), *y_gpu;
//     CUDA_CHECK(cudaMalloc(&y_gpu, ref.numel() * sizeof(float)));

//     K.hard_swish(x_gpu, y_gpu, ref.numel(), nullptr);
//     CUDA_CHECK(cudaDeviceSynchronize());

//     ag::Tensor out = from_gpu(y_gpu, 11, 11);
//     check_tensors_close(ref, out, "test_gpu_hard_swish");

//     ag::Tensor gy_cpu = ag::Tensor::randn(11, 11, 2);
//     ag::Tensor mask = (x_cpu > -2.5f) * (x_cpu < 2.5f);
//     ag::Tensor dh = 0.2f * mask;
//     ag::Tensor ga_ref = gy_cpu * (hs + x_cpu * dh);

//     float *gy_gpu = to_gpu(gy_cpu);
//     float *ga_gpu = to_gpu(ag::Tensor::zeros_like(x_cpu));

//     K.vjp_hard_swish(ga_gpu, gy_gpu, x_gpu, ref.numel(), nullptr);
//     CUDA_CHECK(cudaDeviceSynchronize());

//     ag::Tensor ga_out = from_gpu(ga_gpu, 11, 11);
//     check_tensors_close(ga_ref, ga_out, "test_gpu_vjp_hard_swish");

//     CUDA_CHECK(cudaFree(x_gpu));
//     CUDA_CHECK(cudaFree(y_gpu));
//     CUDA_CHECK(cudaFree(gy_gpu));
//     CUDA_CHECK(cudaFree(ga_gpu));
// }

void test_gpu_unified_reluattention() {
    auto& K = ag::kernels::cuda();
    ag::Tensor a_cpu = ag::Tensor::randn(11, 9, 3);
    ag::Tensor b_cpu = ag::Tensor::randn(9, 14, 4);
    ag::Tensor c_cpu = ag::Tensor::randn(9, 14, 5);
    ag::Tensor d_cpu = ag::Tensor::randn(9, 14, 6);
    ag::Tensor e_cpu = ag::Tensor::zeros(11, 14);
    std::cout << "\nForward Pass Values (Linear):" << std::endl;
    // std::cout << "Input A (CPU):\n" << a_cpu << std::endl;
    // std::cout << "Input B (CPU):\n" << b_cpu << std::endl;
    // std::cout << "Input C (CPU):\n" << c_cpu << std::endl;
    // std::cout << "Input D (CPU):\n" << d_cpu << std::endl;
                    ag::Tensor q = ag::Tensor::matmul(a_cpu, b_cpu); 
    auto k = ag::Tensor::matmul(a_cpu, c_cpu); 
    auto v = ag::Tensor::matmul(a_cpu, d_cpu);
    auto g = ag::Tensor::matmul(q, ag::Tensor::transpose(k)*(1.f/sqrt(float(k.cols())))) ;
    auto s = ag::Tensor::relu(g);
    auto ref = ag::Tensor::matmul(s, v);
    std::cout << "Output (CPU):\n" << ref << std::endl;

    float *q_gpu = to_gpu(q), *k_gpu = to_gpu(k), *v_gpu = to_gpu(v), *e_gpu=to_gpu(e_cpu);

    K.reluflash(q_gpu, k_gpu, v_gpu, e_gpu, 1, 1, 11, 14, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor out = from_gpu(e_gpu, 11, 14);
    check_tensors_close(ref, out, "test_gpu_reluattention");
    std::cout << "Output (GPU):\n" << out << std::endl;
    

    // ag::Tensor gy_cpu = ag::Tensor::randn(11, 9, 11);

    //     // Reference calculation on CPU
    // ag::Tensor ga_ref = ag::Tensor::matmul(gy_cpu, (b_cpu))+ag::Tensor::ones_like(a_cpu);
    // ag::Tensor gb_ref = ag::Tensor::transpose(ag::Tensor::matmul(ag::Tensor::transpose(a_cpu), gy_cpu))+ag::Tensor::ones_like(b_cpu);
    // ag::Tensor gc_ref = gy_cpu+ag::Tensor::ones_like(c_cpu);

    // float *gy_gpu = to_gpu(gy_cpu);
    // float *ga_gpu = to_gpu(ag::Tensor::ones_like(a_cpu)), *gb_gpu = to_gpu(ag::Tensor::ones_like(b_cpu)),  *gc_gpu = to_gpu(ag::Tensor::ones_like(c_cpu));



    // K.vjp_linear(ga_gpu, gb_gpu, gc_gpu, gy_gpu, a_gpu, b_gpu, c_gpu, 11, 17, 9, nullptr);
    // CUDA_CHECK(cudaDeviceSynchronize());

    // ag::Tensor ga_out = from_gpu(ga_gpu, 11, 17);
    // ag::Tensor gb_out = from_gpu(gb_gpu, 9, 17);
    // ag::Tensor gc_out = from_gpu(gc_gpu, 11, 9);

    // std::cout << "\nBackward Pass Values (Linear):" << std::endl;
    // std::cout << "Gradient Input dY:\n" << gy_cpu << std::endl;
    // std::cout << "Expected dA (CPU):\n" << ga_ref << std::endl;
    // std::cout << "Computed dA (GPU):\n" << ga_out << std::endl;
    // std::cout << "Expected dB (CPU):\n" << gb_ref << std::endl;
    // std::cout << "Computed dB (GPU):\n" << gb_out << std::endl;
    // std::cout << "Expected dC (CPU):\n" << gc_ref << std::endl;
    // std::cout << "Computed dC (GPU):\n" << gc_out << std::endl;

    // check_tensors_close(ga_ref, ga_out, "test_gpu_vjp_linear (gA)");
    // check_tensors_close(gb_ref, gb_out, "test_gpu_vjp_linear (gB)");
    // check_tensors_close(gc_ref, gc_out, "test_gpu_vjp_linear (gC)");

    CUDA_CHECK(cudaFree(q_gpu));
    CUDA_CHECK(cudaFree(k_gpu));
    CUDA_CHECK(cudaFree(v_gpu));
    CUDA_CHECK(cudaFree(e_gpu));
    // CUDA_CHECK(cudaFree(ga_gpu));
    // CUDA_CHECK(cudaFree(gb_gpu));
    // CUDA_CHECK(cudaFree(gc_gpu));
}

// const float * charstri(std::string w) {
//     float a[0];
//     if(w=="relu")
//     a[0]= 1.f;
//     else if(w=="sigmoid")
//     a[0]=  2.f;
//     else if(w=="tanh")
//     a[0]=  3.f;
//    // else if(w=="softplus")
//    else
//     a[0]=  4.f;
//    // const float *ptr = a;
//     return a;
    
// }


void test_gpu_unified_flexattention() {
    auto& K = ag::kernels::cuda();
    ag::Tensor a_cpu = ag::Tensor::randn(11, 9, 3);
    ag::Tensor b_cpu = ag::Tensor::randn(9, 14, 4);
    ag::Tensor c_cpu = ag::Tensor::randn(9, 14, 5);
    ag::Tensor d_cpu = ag::Tensor::randn(9, 14, 6);
    ag::Tensor e_cpu = ag::Tensor::zeros(11, 14);
    std::cout << "\nForward Pass Values (Linear):" << std::endl;
    // std::cout << "Input A (CPU):\n" << a_cpu << std::endl;
    // std::cout << "Input B (CPU):\n" << b_cpu << std::endl;
    // std::cout << "Input C (CPU):\n" << c_cpu << std::endl;
    // std::cout << "Input D (CPU):\n" << d_cpu << std::endl;
                    ag::Tensor q = ag::Tensor::matmul(a_cpu, b_cpu); 
    auto k = ag::Tensor::matmul(a_cpu, c_cpu); 
    auto v = ag::Tensor::matmul(a_cpu, d_cpu);
    auto g = ag::Tensor::matmul(q, ag::Tensor::transpose(k)*(1.f/sqrt(float(k.cols())))) ;
    auto s = ag::Tensor::softplus(g);
    auto ref = ag::Tensor::matmul(s, v);
    std::cout << "Output (CPU):\n" << ref << std::endl;

    float *q_gpu = to_gpu(q), *k_gpu = to_gpu(k), *v_gpu = to_gpu(v), *e_gpu=to_gpu(e_cpu);
    std::string activation="softplus";
    float mmi[1];
    if(activation=="relu")
    mmi[0]= 1.f;
    else if(activation=="sigmoid")
    mmi[0]=  2.f;
    else if(activation=="softplus")
    mmi[0]=  3.f;
    else
    mmi[0]=  4.f;

    K.flexflash(q_gpu, k_gpu, v_gpu, e_gpu, 1, 1, 11, 14, mmi, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor out = from_gpu(e_gpu, 11, 14);
    check_tensors_close(ref, out, "test_gpu_flexattention", 3.0);
    std::cout << "Output (GPU):\n" << out << std::endl;
    

    // ag::Tensor gy_cpu = ag::Tensor::randn(11, 9, 11);

    //     // Reference calculation on CPU
    // ag::Tensor ga_ref = ag::Tensor::matmul(gy_cpu, (b_cpu))+ag::Tensor::ones_like(a_cpu);
    // ag::Tensor gb_ref = ag::Tensor::transpose(ag::Tensor::matmul(ag::Tensor::transpose(a_cpu), gy_cpu))+ag::Tensor::ones_like(b_cpu);
    // ag::Tensor gc_ref = gy_cpu+ag::Tensor::ones_like(c_cpu);

    // float *gy_gpu = to_gpu(gy_cpu);
    // float *ga_gpu = to_gpu(ag::Tensor::ones_like(a_cpu)), *gb_gpu = to_gpu(ag::Tensor::ones_like(b_cpu)),  *gc_gpu = to_gpu(ag::Tensor::ones_like(c_cpu));



    // K.vjp_linear(ga_gpu, gb_gpu, gc_gpu, gy_gpu, a_gpu, b_gpu, c_gpu, 11, 17, 9, nullptr);
    // CUDA_CHECK(cudaDeviceSynchronize());

    // ag::Tensor ga_out = from_gpu(ga_gpu, 11, 17);
    // ag::Tensor gb_out = from_gpu(gb_gpu, 9, 17);
    // ag::Tensor gc_out = from_gpu(gc_gpu, 11, 9);

    // std::cout << "\nBackward Pass Values (Linear):" << std::endl;
    // std::cout << "Gradient Input dY:\n" << gy_cpu << std::endl;
    // std::cout << "Expected dA (CPU):\n" << ga_ref << std::endl;
    // std::cout << "Computed dA (GPU):\n" << ga_out << std::endl;
    // std::cout << "Expected dB (CPU):\n" << gb_ref << std::endl;
    // std::cout << "Computed dB (GPU):\n" << gb_out << std::endl;
    // std::cout << "Expected dC (CPU):\n" << gc_ref << std::endl;
    // std::cout << "Computed dC (GPU):\n" << gc_out << std::endl;

    // check_tensors_close(ga_ref, ga_out, "test_gpu_vjp_linear (gA)");
    // check_tensors_close(gb_ref, gb_out, "test_gpu_vjp_linear (gB)");
    // check_tensors_close(gc_ref, gc_out, "test_gpu_vjp_linear (gC)");

    CUDA_CHECK(cudaFree(q_gpu));
    CUDA_CHECK(cudaFree(k_gpu));
    CUDA_CHECK(cudaFree(v_gpu));
    CUDA_CHECK(cudaFree(e_gpu));
    // CUDA_CHECK(cudaFree(ga_gpu));
    // CUDA_CHECK(cudaFree(gb_gpu));
    // CUDA_CHECK(cudaFree(gc_gpu));
}

void test_gpu_unified_sigattention() {
    auto& K = ag::kernels::cuda();
    ag::Tensor a_cpu = ag::Tensor::randn(11, 9, 3);
    ag::Tensor b_cpu = ag::Tensor::randn(9, 14, 4);
    ag::Tensor c_cpu = ag::Tensor::randn(9, 14, 5);
    ag::Tensor d_cpu = ag::Tensor::randn(9, 14, 6);
    ag::Tensor e_cpu = ag::Tensor::zeros(11, 14);
    std::cout << "\nForward Pass Values (Linear):" << std::endl;
    // std::cout << "Input A (CPU):\n" << a_cpu << std::endl;
    // std::cout << "Input B (CPU):\n" << b_cpu << std::endl;
    // std::cout << "Input C (CPU):\n" << c_cpu << std::endl;
    // std::cout << "Input D (CPU):\n" << d_cpu << std::endl;
                    ag::Tensor q = ag::Tensor::matmul(a_cpu, b_cpu); 
    auto k = ag::Tensor::matmul(a_cpu, c_cpu); 
    auto v = ag::Tensor::matmul(a_cpu, d_cpu);
    auto g = ag::Tensor::matmul(q, ag::Tensor::transpose(k)*(1.f/sqrt(float(k.cols())))) ;
    auto s = ag::Tensor::sigmoid(g);
    auto ref = ag::Tensor::matmul(s, v);
    std::cout << "Output (CPU):\n" << ref << std::endl;

    float *q_gpu = to_gpu(q), *k_gpu = to_gpu(k), *v_gpu = to_gpu(v), *e_gpu=to_gpu(e_cpu);

    K.sigflash(q_gpu, k_gpu, v_gpu, e_gpu, 1, 1, 11, 14, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor out = from_gpu(e_gpu, 11, 14);
    check_tensors_close(ref, out, "test_gpu_sigattention", 4.0);
    std::cout << "Output (GPU):\n" << out << std::endl;
    

    // ag::Tensor gy_cpu = ag::Tensor::randn(11, 9, 11);

    //     // Reference calculation on CPU
    // ag::Tensor ga_ref = ag::Tensor::matmul(gy_cpu, (b_cpu))+ag::Tensor::ones_like(a_cpu);
    // ag::Tensor gb_ref = ag::Tensor::transpose(ag::Tensor::matmul(ag::Tensor::transpose(a_cpu), gy_cpu))+ag::Tensor::ones_like(b_cpu);
    // ag::Tensor gc_ref = gy_cpu+ag::Tensor::ones_like(c_cpu);

    // float *gy_gpu = to_gpu(gy_cpu);
    // float *ga_gpu = to_gpu(ag::Tensor::ones_like(a_cpu)), *gb_gpu = to_gpu(ag::Tensor::ones_like(b_cpu)),  *gc_gpu = to_gpu(ag::Tensor::ones_like(c_cpu));



    // K.vjp_linear(ga_gpu, gb_gpu, gc_gpu, gy_gpu, a_gpu, b_gpu, c_gpu, 11, 17, 9, nullptr);
    // CUDA_CHECK(cudaDeviceSynchronize());

    // ag::Tensor ga_out = from_gpu(ga_gpu, 11, 17);
    // ag::Tensor gb_out = from_gpu(gb_gpu, 9, 17);
    // ag::Tensor gc_out = from_gpu(gc_gpu, 11, 9);

    // std::cout << "\nBackward Pass Values (Linear):" << std::endl;
    // std::cout << "Gradient Input dY:\n" << gy_cpu << std::endl;
    // std::cout << "Expected dA (CPU):\n" << ga_ref << std::endl;
    // std::cout << "Computed dA (GPU):\n" << ga_out << std::endl;
    // std::cout << "Expected dB (CPU):\n" << gb_ref << std::endl;
    // std::cout << "Computed dB (GPU):\n" << gb_out << std::endl;
    // std::cout << "Expected dC (CPU):\n" << gc_ref << std::endl;
    // std::cout << "Computed dC (GPU):\n" << gc_out << std::endl;

    // check_tensors_close(ga_ref, ga_out, "test_gpu_vjp_linear (gA)");
    // check_tensors_close(gb_ref, gb_out, "test_gpu_vjp_linear (gB)");
    // check_tensors_close(gc_ref, gc_out, "test_gpu_vjp_linear (gC)");

    CUDA_CHECK(cudaFree(q_gpu));
    CUDA_CHECK(cudaFree(k_gpu));
    CUDA_CHECK(cudaFree(v_gpu));
    CUDA_CHECK(cudaFree(e_gpu));
    // CUDA_CHECK(cudaFree(ga_gpu));
    // CUDA_CHECK(cudaFree(gb_gpu));
    // CUDA_CHECK(cudaFree(gc_gpu));
}


void test_gpu_unified_sum() {
    auto& K = ag::kernels::cuda();
    ag::Tensor a_cpu = ag::Tensor::randn(51, 23, 111);
    std::cout << "\nForward Pass Values:" << std::endl;
    std::cout << "Input A (CPU):\n" << a_cpu << std::endl;
    ag::Tensor ref = ag::Tensor::sum_all(a_cpu);
    std::cout << "Output (CPU):\n" << ref << std::endl;

    float *a_gpu = to_gpu(a_cpu), *c_gpu;
    CUDA_CHECK(cudaMalloc(&c_gpu, ref.numel() * sizeof(float)));

    K.sum(a_gpu, c_gpu, a_cpu.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor gy_cpu = ag::Tensor::randn(1, 1, 5);

    ag::Tensor one = ag::Tensor::ones_like(a_cpu);
    ag::Tensor ga_ref = gy_cpu; // vjp_add just passes gradient through

    ag::Tensor ga_cpu_init = ag::Tensor::zeros(11, 11);

    float *gy_gpu = to_gpu(gy_cpu);
    float *ga_gpu = to_gpu(ga_cpu_init);

    K.vjp_sum(ga_gpu, a_gpu, gy_gpu, gy_cpu.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    

    ag::Tensor out = from_gpu(c_gpu, 1, 1);
    check_tensors_close(ref, out, "test_gpu_sumall", 0.9);
    std::cout << "Output (GPU):\n" << out << std::endl;

    ag::Tensor ga_out = from_gpu(ga_gpu, 11, 11);

    std::cout << "\nBackward Pass Values:" << std::endl;
    std::cout << "Gradient Input dY:\n" << gy_cpu << std::endl;
    std::cout << "Expected dA (CPU):\n" << ga_ref << std::endl;
    std::cout << "Computed dA (GPU):\n" << ga_out << std::endl;

    check_tensors_close(ga_ref, ga_out, "test_gpu_vjp_sum");





    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(c_gpu));
    CUDA_CHECK(cudaFree(gy_gpu));
    CUDA_CHECK(cudaFree(ga_gpu));
}


void test_gpu_unified_mseloss() {
        auto& K = ag::kernels::cuda();
    ag::Tensor a_cpu = ag::Tensor::randn(11, 11, 11);
    ag::Tensor b_cpu = ag::Tensor::randn(11, 11, 21);

    ag::Tensor ref = ag::Tensor::mse_loss(a_cpu, b_cpu);

    float *a_gpu = to_gpu(a_cpu), *b_gpu = to_gpu(b_cpu);
    float *c_gpu;
CUDA_CHECK(cudaMalloc(&c_gpu, sizeof(float)));
CUDA_CHECK(cudaMemset(c_gpu, 0, sizeof(float)));  // ← important!

    K.mseloss(a_gpu, b_gpu, c_gpu, a_cpu.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor gy_cpu = ag::Tensor::ones(1, 1);
    

    int N = a_cpu.numel();
        ag::Tensor diff = a_cpu - b_cpu;
        ag::Tensor ga_ref = diff * (2.0f / float(N))*gy_cpu;
        ag::Tensor gb_ref = -diff * (2.0f / float(N))*gy_cpu;


    ag::Tensor ga_cpu_init = ag::Tensor::zeros(11, 11);
    ag::Tensor gb_cpu_init = ag::Tensor::zeros(11, 11);

    float *gy_gpu = to_gpu(gy_cpu);
    float *ga_gpu = to_gpu(ga_cpu_init);
    float *gb_gpu = to_gpu(gb_cpu_init);

    K.vjp_mseloss(ga_gpu, gb_gpu, gy_gpu, a_gpu, b_gpu, a_cpu.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor ga_out = from_gpu(ga_gpu, 11, 11);
    ag::Tensor gb_out = from_gpu(gb_gpu, 11, 11);

                std::cout << "\nForward Pass Values:" << std::endl;
    std::cout << "Input A (CPU):\n" << a_cpu << std::endl;
    std::cout << "Input B (CPU):\n" << b_cpu << std::endl;

        ag::Tensor out = from_gpu(c_gpu, 1, 1);
       // out = out;

            check_tensors_close(ref, out, "test_gpu_mseloss");


                std::cout << "Output (CPU):\n" << ref << std::endl;

        std::cout << "Output (GPU):\n" << out << std::endl;

        std::cout << "\n\n\nBackward Pass Values:" << std::endl;
    std::cout << "Gradient Input dY:\n" << gy_cpu << std::endl;




        std::cout << "Expected Gradients (CPU):" << std::endl;
    std::cout << "dA:\n" << ga_ref << std::endl;
    std::cout << "dB:\n" << gb_ref << std::endl;

    std::cout << "Expected Gradients (GPU):" << std::endl;
    std::cout << "dA:\n" << ga_out << std::endl;
    std::cout << "dB:\n" << gb_out << std::endl;

    check_tensors_close(ga_ref, ga_out, "test_gpu_vjp_mseloss (pred dA)");
    check_tensors_close(gb_ref, gb_out, "test_gpu_vjp_mseloss (target gB)");
    CUDA_CHECK(cudaFree(gy_gpu));
    CUDA_CHECK(cudaFree(ga_gpu));
    CUDA_CHECK(cudaFree(gb_gpu));
}

void test_gpu_unified_maeloss() {
        auto& K = ag::kernels::cuda();
    ag::Tensor a_cpu = ag::Tensor::randn(11, 11, 11);
    ag::Tensor b_cpu = ag::Tensor::randn(11, 11, 21);

    ag::Tensor ref = ag::Tensor::mae_loss(a_cpu, b_cpu);

    float *a_gpu = to_gpu(a_cpu), *b_gpu = to_gpu(b_cpu);
    float *c_gpu;
CUDA_CHECK(cudaMalloc(&c_gpu, sizeof(float)));
CUDA_CHECK(cudaMemset(c_gpu, 0, sizeof(float)));  // ← important!

    K.maeloss(a_gpu, b_gpu, c_gpu, a_cpu.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor gy_cpu = ag::Tensor::ones(1, 1);
    

    int N = a_cpu.numel();
        ag::Tensor diff = ag::Tensor::sign(a_cpu - b_cpu);
        ag::Tensor ga_ref = diff * (1.0f / float(N))*gy_cpu;
        ag::Tensor gb_ref = -diff * (1.0f / float(N))*gy_cpu;


    ag::Tensor ga_cpu_init = ag::Tensor::zeros(11, 11);
    ag::Tensor gb_cpu_init = ag::Tensor::zeros(11, 11);

    float *gy_gpu = to_gpu(gy_cpu);
    float *ga_gpu = to_gpu(ga_cpu_init);
    float *gb_gpu = to_gpu(gb_cpu_init);

    K.vjp_maeloss(ga_gpu, gb_gpu, gy_gpu, a_gpu, b_gpu, a_cpu.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor ga_out = from_gpu(ga_gpu, 11, 11);
    ag::Tensor gb_out = from_gpu(gb_gpu, 11, 11);

                std::cout << "\nForward Pass Values:" << std::endl;
    std::cout << "Input A (CPU):\n" << a_cpu << std::endl;
    std::cout << "Input B (CPU):\n" << b_cpu << std::endl;

        ag::Tensor out = from_gpu(c_gpu, 1, 1);
       // out = out;

            check_tensors_close(ref, out, "test_gpu_maeloss");


                std::cout << "Output (CPU):\n" << ref << std::endl;

        std::cout << "Output (GPU):\n" << out << std::endl;

        std::cout << "\n\n\nBackward Pass Values:" << std::endl;
    std::cout << "Gradient Input dY:\n" << gy_cpu << std::endl;




        std::cout << "Expected Gradients (CPU):" << std::endl;
    std::cout << "dA:\n" << ga_ref << std::endl;
    std::cout << "dB:\n" << gb_ref << std::endl;

    std::cout << "Expected Gradients (GPU):" << std::endl;
    std::cout << "dA:\n" << ga_out << std::endl;
    std::cout << "dB:\n" << gb_out << std::endl;

    check_tensors_close(ga_ref, ga_out, "test_gpu_vjp_maeloss (pred dA)");
    check_tensors_close(gb_ref, gb_out, "test_gpu_vjp_maeloss (target gB)");
    CUDA_CHECK(cudaFree(gy_gpu));
    CUDA_CHECK(cudaFree(ga_gpu));
    CUDA_CHECK(cudaFree(gb_gpu));
}



void test_gpu_unified_rowsum() {
    auto& K = ag::kernels::cuda();
    ag::Tensor a_cpu = ag::Tensor::randn(11, 11, 111);
    std::cout << "\nForward Pass Values:" << std::endl;
    std::cout << "Input A (CPU):\n" << a_cpu << std::endl;
    ag::Tensor ref = ag::Tensor::row_sum(a_cpu);
    std::cout << "Output (CPU):\n" << ref << std::endl;

    float *a_gpu = to_gpu(a_cpu), *c_gpu;
    CUDA_CHECK(cudaMalloc(&c_gpu, ref.numel() * sizeof(float)));

    K.rowsum(a_gpu, c_gpu, a_cpu.rows(), a_cpu.cols(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor gy_cpu = ag::Tensor::randn(11, 1, 5);

    ag::Tensor one = ag::Tensor::ones_like(a_cpu);
    ag::Tensor ga_ref = gy_cpu*one; // vjp_add just passes gradient through

    ag::Tensor ga_cpu_init = ag::Tensor::zeros(11, 11);

    float *gy_gpu = to_gpu(gy_cpu);
    float *ga_gpu = to_gpu(ga_cpu_init);

    K.vjp_rowsum(ga_gpu, a_gpu, c_gpu, gy_gpu, a_cpu.rows(), a_cpu.cols(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    

    ag::Tensor out = from_gpu(c_gpu, ref.rows(), ref.cols());
    check_tensors_close(ref, out, "test_gpu_rowsum", 0.9);
    std::cout << "Output (GPU):\n" << out << std::endl;

    ag::Tensor ga_out = from_gpu(ga_gpu, 11, 11);

    std::cout << "\nBackward Pass Values:" << std::endl;
    std::cout << "Gradient Input dY:\n" << gy_cpu << std::endl;
    std::cout << "Expected dA (CPU):\n" << ga_ref << std::endl;
    std::cout << "Computed dA (GPU):\n" << ga_out << std::endl;

    check_tensors_close(ga_ref, ga_out, "test_gpu_vjp_rowsum");





    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(c_gpu));
    CUDA_CHECK(cudaFree(gy_gpu));
    CUDA_CHECK(cudaFree(ga_gpu));
}




void test_gpu_unified_rowmax() {
    auto& K = ag::kernels::cuda();
    ag::Tensor a_cpu = ag::Tensor::randn(11, 17, 12);
    std::cout << "\nForward Pass Values:" << std::endl;
    std::cout << "Input A (CPU):\n" << a_cpu << std::endl;
    ag::Tensor ref = ag::Tensor::row_max(a_cpu);
    std::cout << "Output (CPU):\n" << ref << std::endl;

    float *a_gpu = to_gpu(a_cpu), *c_gpu;
    CUDA_CHECK(cudaMalloc(&c_gpu, ref.numel() * sizeof(float)));

    K.rowmax(a_gpu, c_gpu, a_cpu.rows(), a_cpu.cols(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor gy_cpu = ag::Tensor::randn(11, 1, 5);

 //   ag::Tensor one = ag::Tensor::ones_like(a_cpu);
    ag::Tensor g = ag::Tensor::zeros_like(a_cpu);
        for(int i=0; i<a_cpu.rows(); ++i) for(int j=0; j<a_cpu.cols(); ++j)
            g(i,j) = (a_cpu(i,j)==ref(i,0)) ? gy_cpu(i,0) : 0.f;

    ag::Tensor ga_ref = g; // vjp_add just passes gradient through

    ag::Tensor ga_cpu_init = ag::Tensor::zeros(11, 17);

    float *gy_gpu = to_gpu(gy_cpu);
    float *ga_gpu = to_gpu(ga_cpu_init);

    K.vjp_rowmax(ga_gpu, a_gpu, c_gpu, gy_gpu, a_cpu.rows(), a_cpu.cols(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    

    ag::Tensor out = from_gpu(c_gpu, ref.rows(), ref.cols());
    check_tensors_close(ref, out, "test_gpu_rowmax", 0.9);
    std::cout << "Output (GPU):\n" << out << std::endl;

    ag::Tensor ga_out = from_gpu(ga_gpu, 11, 17);

    std::cout << "\nBackward Pass Values:" << std::endl;
    std::cout << "Gradient Input dY:\n" << gy_cpu << std::endl;
    std::cout << "Expected dA (CPU):\n" << ga_ref << std::endl;
    std::cout << "Computed dA (GPU):\n" << ga_out << std::endl;

    check_tensors_close(ga_ref, ga_out, "test_gpu_vjp_rowmax");





    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(c_gpu));
    CUDA_CHECK(cudaFree(gy_gpu));
    CUDA_CHECK(cudaFree(ga_gpu));
}



void test_gpu_unified_softmax() {
    auto& K = ag::kernels::cuda();
    ag::Tensor a_cpu = ag::Tensor::randn(11, 11, 111);
    std::cout << "\nForward Pass Values:" << std::endl;
    std::cout << "Input A (CPU):\n" << a_cpu << std::endl;
    ag::Tensor ref = ag::Tensor::softmax_row(a_cpu);
    std::cout << "Output (CPU):\n" << ref << std::endl;

    float *a_gpu = to_gpu(a_cpu), *c_gpu;
    CUDA_CHECK(cudaMalloc(&c_gpu, ref.numel() * sizeof(float)));

    K.softmax(a_gpu, c_gpu, a_cpu.rows(), a_cpu.cols(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor gy_cpu = ag::Tensor::randn(11, 1, 5);

    ag::Tensor ne = ag::Tensor::row_sum(ref*gy_cpu);
    ag::Tensor ga_ref = ref *( gy_cpu - ne); // vjp_add just passes gradient through

    ag::Tensor ga_cpu_init = ag::Tensor::zeros(11, 11);

    float *gy_gpu = to_gpu(gy_cpu);
    float *ga_gpu = to_gpu(ga_cpu_init);

    K.vjp_softmax(ga_gpu, c_gpu, gy_gpu, a_cpu.rows(), a_cpu.cols(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    

    ag::Tensor out = from_gpu(c_gpu, ref.rows(), ref.cols());
    check_tensors_close(ref, out, "test_gpu_softmax", 0.9);
    std::cout << "Output (GPU):\n" << out << std::endl;

    ag::Tensor ga_out = from_gpu(ga_gpu, 11, 11);

    std::cout << "\nBackward Pass Values:" << std::endl;
    std::cout << "Gradient Input dY:\n" << gy_cpu << std::endl;
    std::cout << "Expected dA (CPU):\n" << ga_ref << std::endl;
    std::cout << "Computed dA (GPU):\n" << ga_out << std::endl;

    check_tensors_close(ga_ref, ga_out, "test_gpu_vjp_softmax");





    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(c_gpu));
    CUDA_CHECK(cudaFree(gy_gpu));
    CUDA_CHECK(cudaFree(ga_gpu));
}

void test_gpu_unified_gcu() {
    auto& K = ag::kernels::cuda();
    ag::Tensor a_cpu = ag::Tensor::randn(11, 11, 1);
    ag::Tensor ref = a_cpu * ag::Tensor::cos(a_cpu);

    float *a_gpu = to_gpu(a_cpu), *c_gpu;
    CUDA_CHECK(cudaMalloc(&c_gpu, ref.numel() * sizeof(float)));
    K.gcu(a_gpu, c_gpu, ref.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor out = from_gpu(c_gpu, 11, 11);
    check_tensors_close(ref, out, "test_gpu_gcu");

    // Backward
    ag::Tensor gy_cpu = ag::Tensor::randn(11, 11, 2);
    ag::Tensor ga_ref = gy_cpu * (ag::Tensor::cos(a_cpu) - (a_cpu * ag::Tensor::sin(a_cpu)));
    float *gy_gpu = to_gpu(gy_cpu);
    float *ga_gpu = to_gpu(ag::Tensor::zeros_like(a_cpu));
    // vjp signature: (gX, X, gy, n, s)
    K.vjp_gcu(ga_gpu, a_gpu, gy_gpu, gy_cpu.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    ag::Tensor ga_out = from_gpu(ga_gpu, 11, 11);
    check_tensors_close(ga_ref, ga_out, "test_gpu_vjp_gcu", 5.1);

    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(c_gpu));
    CUDA_CHECK(cudaFree(gy_gpu));
    CUDA_CHECK(cudaFree(ga_gpu));
}

void test_gpu_unified_gauss() {
    auto& K = ag::kernels::cuda();
    ag::Tensor a_cpu = ag::Tensor::randn(11, 11, 1);
    ag::Tensor ref = ag::Tensor::exp(-(a_cpu * a_cpu));

    float *a_gpu = to_gpu(a_cpu), *c_gpu;
    CUDA_CHECK(cudaMalloc(&c_gpu, ref.numel() * sizeof(float)));
    K.gauss(a_gpu, c_gpu, ref.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor out = from_gpu(c_gpu, 11, 11);
    check_tensors_close(ref, out, "test_gpu_gauss");

    // Backward
    ag::Tensor gy_cpu = ag::Tensor::randn(11, 11, 2);
    ag::Tensor ga_ref = gy_cpu * (-2.0f * a_cpu * ag::Tensor::exp(a_cpu * a_cpu));
    float *gy_gpu = to_gpu(gy_cpu);
    float *ga_gpu = to_gpu(ag::Tensor::zeros_like(a_cpu));
    // vjp signature: (gX, X, gy, n, s)
    K.vjp_gauss(ga_gpu, a_gpu, gy_gpu, gy_cpu.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    ag::Tensor ga_out = from_gpu(ga_gpu, 11, 11);
    check_tensors_close(ga_ref, ga_out, "test_gpu_vjp_gauss");

    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(c_gpu));
    CUDA_CHECK(cudaFree(gy_gpu));
    CUDA_CHECK(cudaFree(ga_gpu));
}

void test_gpu_unified_parcon() {
    auto& K = ag::kernels::cuda();
    ag::Tensor a_cpu = ag::Tensor::randn(11, 11, 1);
    ag::Tensor ref = a_cpu * (2.f*ag::Tensor::ones_like(a_cpu) - a_cpu);

    float *a_gpu = to_gpu(a_cpu), *c_gpu;
    CUDA_CHECK(cudaMalloc(&c_gpu, ref.numel() * sizeof(float)));
    K.parcon(a_gpu, c_gpu, ref.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor out = from_gpu(c_gpu, 11, 11);
    check_tensors_close(ref, out, "test_gpu_parcon");

    // Backward: derivative = 2*(1 - x)
    ag::Tensor gy_cpu = ag::Tensor::randn(11, 11, 2);
    ag::Tensor ga_ref = gy_cpu * (2.0f * (ag::Tensor::ones_like(a_cpu) - a_cpu));
    float *gy_gpu = to_gpu(gy_cpu);
    float *ga_gpu = to_gpu(ag::Tensor::zeros_like(a_cpu));
    K.vjp_parcon(ga_gpu, a_gpu, gy_gpu, gy_cpu.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    ag::Tensor ga_out = from_gpu(ga_gpu, 11, 11);
    check_tensors_close(ga_ref, ga_out, "test_gpu_vjp_parcon");

    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(c_gpu));
    CUDA_CHECK(cudaFree(gy_gpu));
    CUDA_CHECK(cudaFree(ga_gpu));
}

void test_gpu_unified_lisht() {
    auto& K = ag::kernels::cuda();
    ag::Tensor a_cpu = ag::Tensor::randn(11, 11, 1);
    ag::Tensor ref = a_cpu * ag::Tensor::tanh(a_cpu);

    float *a_gpu = to_gpu(a_cpu), *c_gpu;
    CUDA_CHECK(cudaMalloc(&c_gpu, ref.numel() * sizeof(float)));
    K.lisht(a_gpu, c_gpu, ref.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor out = from_gpu(c_gpu, 11, 11);
    check_tensors_close(ref, out, "test_gpu_lisht");

    // Backward: derivative = - (x*(1 - tanh^2(x)) + tanh(x))
    ag::Tensor gy_cpu = ag::Tensor::randn(11, 11, 2);
    ag::Tensor ga_ref = gy_cpu * ( - (a_cpu * (ag::Tensor::ones_like(a_cpu) - ag::Tensor::tanh(a_cpu)*ag::Tensor::tanh(a_cpu)) + ag::Tensor::tanh(a_cpu)) );
    float *gy_gpu = to_gpu(gy_cpu);
    float *ga_gpu = to_gpu(ag::Tensor::zeros_like(a_cpu));
    K.vjp_lisht(ga_gpu, a_gpu, gy_gpu, gy_cpu.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    ag::Tensor ga_out = from_gpu(ga_gpu, 11, 11);
    check_tensors_close(ga_ref, ga_out, "test_gpu_vjp_lisht");

    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(c_gpu));
    CUDA_CHECK(cudaFree(gy_gpu));
    CUDA_CHECK(cudaFree(ga_gpu));
}

void test_gpu_unified_reci() {
    auto& K = ag::kernels::cuda();
    ag::Tensor a_cpu = ag::Tensor::randn(11, 11, 1) ; // avoid zeros
    ag::Tensor ref = ag::Tensor::ones_like(a_cpu) / a_cpu;

    float *a_gpu = to_gpu(a_cpu), *c_gpu;
    CUDA_CHECK(cudaMalloc(&c_gpu, ref.numel() * sizeof(float)));
    K.reci(a_gpu, c_gpu, ref.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor out = from_gpu(c_gpu, 11, 11);
    check_tensors_close(ref, out, "test_gpu_reci");

    // Backward: derivative = -1 / x^2
    ag::Tensor gy_cpu = ag::Tensor::randn(11, 11, 3);
    ag::Tensor ga_ref = gy_cpu * ( - (ag::Tensor::ones_like(a_cpu) / (a_cpu * a_cpu)) );
    float *gy_gpu = to_gpu(gy_cpu);
    float *ga_gpu = to_gpu(ag::Tensor::zeros_like(a_cpu));
    K.vjp_reci(ga_gpu, a_gpu, gy_gpu, gy_cpu.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    ag::Tensor ga_out = from_gpu(ga_gpu, 11, 11);
    check_tensors_close(ga_ref, ga_out, "test_gpu_vjp_reci", 1.f);

    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(c_gpu));
    CUDA_CHECK(cudaFree(gy_gpu));
    CUDA_CHECK(cudaFree(ga_gpu));
}



void test_gpu_unified_swiglu() {
    auto& K = ag::kernels::cuda();
    ag::Tensor a_cpu = ag::Tensor::randn(11, 17, 3);
    ag::Tensor b_cpu = ag::Tensor::randn(9, 17, 4);
    ag::Tensor d_cpu = ag::Tensor::randn(11, 9, 5);
    ag::Tensor f_cpu = ag::Tensor::randn(9, 17, 2);
    ag::Tensor g_cpu = ag::Tensor::randn(11, 9, 46);
    ag::Tensor e_cpu = ag::Tensor::zeros(11, 9);
    ag::Tensor c_cpu = ag::Tensor::zeros(11, 9);
    std::cout << "\nForward Pass Values (SWIGLU):" << std::endl;
    // std::cout << "Input A (CPU):\n" << a_cpu << std::endl;
    // std::cout << "Input B (CPU):\n" << b_cpu << std::endl;
    // std::cout << "Input D (CPU):\n" << d_cpu << std::endl;
    ag::Tensor ref = ag::Tensor::matmul(a_cpu, ag::Tensor::transpose( b_cpu)) + d_cpu;
    std::cout << "Output (CPU):\n" << ref << std::endl;

    float *a_gpu = to_gpu(a_cpu), *b_gpu = to_gpu(b_cpu), *d_gpu = to_gpu(d_cpu), *e_gpu=to_gpu(e_cpu), *f_gpu=to_gpu(f_cpu), *g_gpu=to_gpu(g_cpu), *c_gpu=to_gpu(c_cpu);

    K.swiglu(a_gpu, b_gpu, d_gpu, f_gpu, g_gpu, c_gpu, e_gpu, 11, 17, 9, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor out = from_gpu(e_gpu, 11, 9);
    check_tensors_close(ref, out, "test_gpu_SWIGLU");
    std::cout << "Output (GPU):\n" << out << std::endl;
    

    // ag::Tensor gy_cpu = ag::Tensor::randn(11, 9, 11);

    //     // Reference calculation on CPU
    // ag::Tensor ga_ref = ag::Tensor::matmul(gy_cpu, (b_cpu))+ag::Tensor::ones_like(a_cpu);
    // ag::Tensor gb_ref = ag::Tensor::transpose(ag::Tensor::matmul(ag::Tensor::transpose(a_cpu), gy_cpu))+ag::Tensor::ones_like(b_cpu);
    // ag::Tensor gc_ref = gy_cpu+ag::Tensor::ones_like(d_cpu);

    // float *gy_gpu = to_gpu(gy_cpu);
    // float *ga_gpu = to_gpu(ag::Tensor::ones_like(a_cpu)), *gb_gpu = to_gpu(ag::Tensor::ones_like(b_cpu)),  *gc_gpu = to_gpu(ag::Tensor::ones_like(c_cpu));



    // K.vjp_linear(ga_gpu, gb_gpu, gc_gpu, gy_gpu, a_gpu, b_gpu, d_gpu, 11, 17, 9, nullptr);
    // CUDA_CHECK(cudaDeviceSynchronize());

    // ag::Tensor ga_out = from_gpu(ga_gpu, 11, 17);
    // ag::Tensor gb_out = from_gpu(gb_gpu, 9, 17);
    // ag::Tensor gc_out = from_gpu(gc_gpu, 11, 9);

    // std::cout << "\nBackward Pass Values (Linear):" << std::endl;
    // std::cout << "Gradient Input dY:\n" << gy_cpu << std::endl;
    // std::cout << "Expected dA (CPU):\n" << ga_ref << std::endl;
    // std::cout << "Computed dA (GPU):\n" << ga_out << std::endl;
    // std::cout << "Expected dB (CPU):\n" << gb_ref << std::endl;
    // std::cout << "Computed dB (GPU):\n" << gb_out << std::endl;
    // std::cout << "Expected dC (CPU):\n" << gc_ref << std::endl;
    // std::cout << "Computed dC (GPU):\n" << gc_out << std::endl;

    // check_tensors_close(ga_ref, ga_out, "test_gpu_vjp_linear (gA)");
    // check_tensors_close(gb_ref, gb_out, "test_gpu_vjp_linear (gB)");
    // check_tensors_close(gc_ref, gc_out, "test_gpu_vjp_linear (gC)");

    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(b_gpu));
    CUDA_CHECK(cudaFree(c_gpu));
    CUDA_CHECK(cudaFree(d_gpu));
    CUDA_CHECK(cudaFree(e_gpu));
    CUDA_CHECK(cudaFree(f_gpu));
    CUDA_CHECK(cudaFree(g_gpu));
    // CUDA_CHECK(cudaFree(gy_gpu));
    // CUDA_CHECK(cudaFree(ga_gpu));
    // CUDA_CHECK(cudaFree(gb_gpu));
    // CUDA_CHECK(cudaFree(gc_gpu));
}


int main() {
    std::cout << "=== Running GPU Kernel Tests ===\n";
    try {
        #if defined(_WIN32)
            const char* plugin_path = "./agkernels_cuda.dll";
        #elif defined(__APPLE__)
            const char* plugin_path = "./libagkernels_cuda.dylib";
        #else
            const char* plugin_path = "./cgadimpl/build/libagkernels_cuda.so";
        #endif

        std::cout << "Loading GPU plugin from: " << plugin_path << "\n";
        ag::kernels::load_cuda_plugin(plugin_path);

        test_gpu_add();
        test_gpu_matmul();
        test_gpu_vjp_add();
        test_gpu_vjp_matmul();
        test_gpu_unified_tanh();
        test_gpu_unified_fmab();
        test_gpu_unified_linear();
        test_gpu_unified_sub();
        test_gpu_unified_exp();
        test_gpu_unified_sigmoid();
    // custom elementwise tests
    // test_gpu_unified_gcu();
    // test_gpu_unified_gauss();
    // test_gpu_unified_parcon();
    // test_gpu_unified_lisht();
    // test_gpu_unified_reci();
        // test_gpu_unified_gelu();
        test_gpu_unified_mish();
        // test_gpu_unified_hard_swish();
        test_gpu_unified_square();
        test_gpu_unified_neg();
        // test_gpu_unified_clip();
        // test_gpu_unified_pow();
        test_gpu_unified_hadmul();
        test_gpu_unified_div();
        test_gpu_unified_attention();
        test_gpu_unified_reluattention();
        test_gpu_unified_sigattention();
        test_gpu_unified_flexattention();
        test_gpu_unified_sum();

    // New elementwise op tests
    test_gpu_unified_gauss();
    test_gpu_unified_reci();
    test_gpu_unified_parcon();
    test_gpu_unified_lisht();
    test_gpu_unified_gcu();

        test_gpu_unified_mseloss();
        test_gpu_unified_maeloss();
        test_gpu_unified_rowsum();
        test_gpu_unified_rowmax();
        test_gpu_unified_softmax();

        test_gpu_unified_swiglu();

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\nAll GPU kernel tests passed successfully!\n";
    return 0;
}