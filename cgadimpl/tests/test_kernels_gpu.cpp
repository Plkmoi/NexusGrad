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
void check_tensors_close(const ag::Tensor& a, const ag::Tensor& b, const std::string& label, float epsilon = 1e-4f) {
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
    ag::Tensor ref = a_cpu + b_cpu;

    float *a_gpu = to_gpu(a_cpu), *b_gpu = to_gpu(b_cpu), *c_gpu;
    CUDA_CHECK(cudaMalloc(&c_gpu, ref.numel() * sizeof(float)));

    K.add(a_gpu, b_gpu, c_gpu, ref.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor out = from_gpu(c_gpu, 11, 11);
    check_tensors_close(ref, out, "test_gpu_add");

    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(b_gpu));
    CUDA_CHECK(cudaFree(c_gpu));
}


void test_gpu_unified_tanh() {
    auto& K = ag::kernels::cuda();
    ag::Tensor a_cpu = ag::Tensor::randn(11, 11, 1);
    ag::Tensor ref = ag::Tensor::tanh(a_cpu);

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

    ag::Tensor ga_out = from_gpu(ga_gpu, 11, 11);

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
    ag::Tensor ref = ag::Tensor::matmul(a_cpu, b_cpu) + c_cpu;

    float *a_gpu = to_gpu(a_cpu), *b_gpu = to_gpu(b_cpu), *c_gpu = to_gpu(c_cpu), *e_gpu=to_gpu(e_cpu);

    K.gemm(a_gpu, b_gpu, c_gpu, e_gpu, 11, 17, 9, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor out = from_gpu(e_gpu, 11, 9);
    check_tensors_close(ref, out, "test_gpu_gemm");
    

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
    ag::Tensor ref = ag::Tensor::matmul(a_cpu, ag::Tensor::transpose( b_cpu)) + c_cpu;

    float *a_gpu = to_gpu(a_cpu), *b_gpu = to_gpu(b_cpu), *c_gpu = to_gpu(c_cpu), *e_gpu=to_gpu(e_cpu);

    K.linear(a_gpu, b_gpu, c_gpu, e_gpu, 11, 17, 9, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor out = from_gpu(e_gpu, 11, 9);
    check_tensors_close(ref, out, "test_gpu_linear");
    

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

void test_gpu_matmul() {
    auto& K = ag::kernels::cuda();
    ag::Tensor a_cpu = ag::Tensor::randn(11, 17, 3);
    ag::Tensor b_cpu = ag::Tensor::randn(17, 11, 4);
    ag::Tensor ref = ag::Tensor::matmul(a_cpu, b_cpu);

    float *a_gpu = to_gpu(a_cpu), *b_gpu = to_gpu(b_cpu), *c_gpu;
    CUDA_CHECK(cudaMalloc(&c_gpu, ref.numel() * sizeof(float)));

    K.matmul(a_gpu, b_gpu, c_gpu, 11, 17, 11, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor out = from_gpu(c_gpu, 11, 11);
    check_tensors_close(ref, out, "test_gpu_matmul");

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
    ag::Tensor ref = a_cpu - b_cpu;

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

    K.mul(a_gpu, b_gpu, c_gpu, ref.numel(), nullptr);
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
    ag::Tensor ref = a_cpu / b_cpu;

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
    ag::Tensor ref = x_cpu * x_cpu;

    float *x_gpu = to_gpu(x_cpu), *y_gpu;
    CUDA_CHECK(cudaMalloc(&y_gpu, ref.numel() * sizeof(float)));

    K.square(x_gpu, y_gpu, ref.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor out = from_gpu(y_gpu, 11, 11);
    check_tensors_close(ref, out, "test_gpu_square");

    ag::Tensor gy_cpu = ag::Tensor::randn(11, 11, 2);
    ag::Tensor ga_ref = gy_cpu * (2.0f * x_cpu);

    float *gy_gpu = to_gpu(gy_cpu);
    float *ga_gpu = to_gpu(ag::Tensor::zeros_like(x_cpu));

    K.vjp_square(ga_gpu, x_gpu, gy_gpu, ref.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor ga_out = from_gpu(ga_gpu, 11, 11);
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
    ag::Tensor ref = -x_cpu;

    float *x_gpu = to_gpu(x_cpu), *y_gpu;
    CUDA_CHECK(cudaMalloc(&y_gpu, ref.numel() * sizeof(float)));

    K.neg(x_gpu, y_gpu, ref.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor out = from_gpu(y_gpu, 11, 11);
    check_tensors_close(ref, out, "test_gpu_neg");

    ag::Tensor gy_cpu = ag::Tensor::randn(11, 11, 2);
    ag::Tensor ga_ref = -gy_cpu;

    float *gy_gpu = to_gpu(gy_cpu);
    float *ga_gpu = to_gpu(ag::Tensor::zeros_like(x_cpu));

    K.vjp_neg(ga_gpu, gy_gpu, ref.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor ga_out = from_gpu(ga_gpu, 11, 11);
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
    ag::Tensor ref = ag::Tensor::sigmoid(x_cpu);

    float *x_gpu = to_gpu(x_cpu), *y_gpu;
    CUDA_CHECK(cudaMalloc(&y_gpu, ref.numel() * sizeof(float)));

    K.sigmoid(x_gpu, y_gpu, ref.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor out = from_gpu(y_gpu, 11, 11);
    check_tensors_close(ref, out, "test_gpu_sigmoid");

    ag::Tensor gy_cpu = ag::Tensor::ones(11, 11);
    ag::Tensor ga_ref = gy_cpu * ref * (ag::Tensor::ones_like(ref) - ref);

    float *gy_gpu = to_gpu(gy_cpu);
    float *ga_gpu = to_gpu(ag::Tensor::zeros_like(x_cpu));

    K.vjp_sigmoid(ga_gpu, x_gpu, gy_gpu, ref.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor ga_out = from_gpu(ga_gpu, 11, 11);
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
    ag::Tensor s = ag::Tensor::sigmoid(x_cpu);
    ag::Tensor ref = x_cpu * s;

    float *x_gpu = to_gpu(x_cpu), *y_gpu;
    CUDA_CHECK(cudaMalloc(&y_gpu, ref.numel() * sizeof(float)));

    K.silu(x_gpu, y_gpu, ref.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor out = from_gpu(y_gpu, 11, 11);
    check_tensors_close(ref, out, "test_gpu_silu");

    ag::Tensor gy_cpu = ag::Tensor::randn(11, 11, 2);
    ag::Tensor ga_ref = gy_cpu * (s + x_cpu * s * (ag::Tensor::ones_like(s)-s));

    float *gy_gpu = to_gpu(gy_cpu);
    float *ga_gpu = to_gpu(ag::Tensor::zeros_like(x_cpu));

    K.vjp_silu(ga_gpu, gy_gpu, x_gpu, ref.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor ga_out = from_gpu(ga_gpu, 11, 11);
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
    ag::Tensor sp = ag::Tensor::softplus(x_cpu);
    ag::Tensor ref = x_cpu * ag::Tensor::tanh(sp);

    float *x_gpu = to_gpu(x_cpu), *y_gpu;
    CUDA_CHECK(cudaMalloc(&y_gpu, ref.numel() * sizeof(float)));

    K.mish(x_gpu, y_gpu, ref.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor out = from_gpu(y_gpu, 11, 11);
    check_tensors_close(ref, out, "test_gpu_mish");

    ag::Tensor gy_cpu = ag::Tensor::randn(11, 11, 2);
    ag::Tensor s = ag::Tensor::sigmoid(x_cpu);
    ag::Tensor t = ag::Tensor::tanh(sp);
    ag::Tensor ga_ref = gy_cpu * (t + x_cpu * s * (ag::Tensor::ones_like(t) - t * t));

    float *gy_gpu = to_gpu(gy_cpu);
    float *ga_gpu = to_gpu(ag::Tensor::zeros_like(x_cpu));

    K.vjp_mish(ga_gpu, gy_gpu, x_gpu, ref.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor ga_out = from_gpu(ga_gpu, 11, 11);
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
    ag::Tensor ref = ag::Tensor::exp(x_cpu);

    float *x_gpu = to_gpu(x_cpu), *y_gpu;
    CUDA_CHECK(cudaMalloc(&y_gpu, ref.numel() * sizeof(float)));

    K.exp(x_gpu, y_gpu, ref.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor out = from_gpu(y_gpu, 11, 11);
    check_tensors_close(ref, out, "test_gpu_exp");

    ag::Tensor gy_cpu = ag::Tensor::ones(11, 11);
    ag::Tensor ga_ref = gy_cpu * ref;

    float *gy_gpu = to_gpu(gy_cpu);
    float *ga_gpu = to_gpu(ag::Tensor::zeros_like(x_cpu));

    K.vjp_exp(ga_gpu, x_gpu, gy_gpu, ref.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor ga_out = from_gpu(ga_gpu, 11, 11);
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
        // test_gpu_unified_gelu();
        test_gpu_unified_mish();
        // test_gpu_unified_hard_swish();
        test_gpu_unified_square();
        test_gpu_unified_neg();
        // test_gpu_unified_clip();
        // test_gpu_unified_pow();
        test_gpu_unified_hadmul();
        test_gpu_unified_div();

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\nAll GPU kernel tests passed successfully!\n";
    return 0;
}