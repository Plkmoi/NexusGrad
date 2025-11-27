#include "ad/ag_all.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

// Use the correct namespaces as defined in your project
using namespace OwnTensor;
using namespace ag;

// Helper updated to use types from the correct namespaces
void check_tensors_close(const Tensor& a, const Tensor& b, const std::string& label, float epsilon = 1e-4f) {
    if (a.shape().dims != b.shape().dims) {
        throw std::runtime_error(label + ": Shape mismatch.");
    }
    Tensor a_cpu = a.to_cpu();
    Tensor b_cpu = b.to_cpu();
    const float* a_data = a_cpu.data<float>();
    const float* b_data = b_cpu.data<float>();

    for (size_t i = 0; i < a.numel(); ++i) {
        if (std::abs(a_data[i] - b_data[i]) > epsilon) {
            std::cerr << "FAIL: " << label << " mismatch at index " << i << "\n";
            debug::print_tensor("Tensor A (ref)", a);
            debug::print_tensor("Tensor B (out)", b);
            throw std::runtime_error("Tensor check failed for " + label);
        }
    }
    std::cout << "PASS: " << label << "\n";
}

// --- Test Functions ---
void test_gpu_unified_add() {
    auto& K = kernels::cuda();
    auto cpu_opts = TensorOptions().with_device(Device::CPU);
    auto gpu_opts = TensorOptions().with_device(DeviceIndex(Device::CUDA));

    const int M = 8, N = 8;

    // ----- CPU inputs -----
    Tensor a_cpu = Tensor::randn(Shape{{M, N}}, cpu_opts);
    Tensor b_cpu = Tensor::randn(Shape{{M, N}}, cpu_opts);

    // Forward reference
    Tensor ref = a_cpu + b_cpu;

    // ----- GPU tensors -----
    Tensor a_gpu = a_cpu.to(gpu_opts.device);
    Tensor b_gpu = b_cpu.to(gpu_opts.device);
    Tensor c_gpu(ref.shape(), options(ref).with_device(gpu_opts.device));

    // ----- GPU forward -----
    K.add(a_gpu.data<float>(), b_gpu.data<float>(), c_gpu.data<float>(),
          ref.numel(), nullptr);

    cudaDeviceSynchronize();

    Tensor out_cpu = c_gpu.to_cpu();
    check_tensors_close(ref, out_cpu, "test_gpu_unified_add (forward)");

    // ===== Backward =====
    Tensor gy_cpu = Tensor::randn(Shape{{M, N}}, cpu_opts);

    // CPU reference grads (identity)
    Tensor ga_ref = gy_cpu;
    Tensor gb_ref = gy_cpu;

    // GPU gradient buffers
    Tensor ga_gpu(ga_ref.shape(), options(ga_ref).with_device(gpu_opts.device));
    Tensor gb_gpu(gb_ref.shape(), options(gb_ref).with_device(gpu_opts.device));
    Tensor gy_gpu = gy_cpu.to(gpu_opts.device);

    // ----- GPU VJP -----
    K.vjp_add(ga_gpu.data<float>(), gb_gpu.data<float>(),
              gy_gpu.data<float>(), ref.numel(), nullptr);

    cudaDeviceSynchronize();

    // Move results back to CPU
    Tensor ga_out = ga_gpu.to_cpu();
    Tensor gb_out = gb_gpu.to_cpu();

    check_tensors_close(ga_ref, ga_out, "test_gpu_unified_add (gA)");
    check_tensors_close(gb_ref, gb_out, "test_gpu_unified_add (gB)");
}




void test_gpu_unified_linear() {
    auto& K = kernels::cuda();
    auto cpu_opts = TensorOptions().with_device(Device::CPU);
    auto gpu_opts = TensorOptions().with_device(Device::CUDA);

    // Input shapes:
    // A: (11, 17)
    // B: (9, 17)
    // C: (11, 9)
    Tensor a_cpu = Tensor::randn(Shape{{11, 17}}, cpu_opts);
    Tensor b_cpu = Tensor::randn(Shape{{9, 17}}, cpu_opts);
    Tensor c_cpu = Tensor::randn(Shape{{11, 9}}, cpu_opts);

    // Reference CPU forward: A * B^T + C
    Tensor ref = OwnTensor::matmul(a_cpu, b_cpu.t()) + c_cpu;

    // Allocate GPU tensors
    Tensor a_gpu = a_cpu.to(gpu_opts.device);
    Tensor b_gpu = b_cpu.to(gpu_opts.device);
    Tensor c_gpu = c_cpu.to(gpu_opts.device);
    Tensor e_gpu(ref.shape(), options(ref).with_device(gpu_opts.device));

    // ---- Forward pass ----
    K.linear(a_gpu.data<float>(),
             b_gpu.data<float>(),
             c_gpu.data<float>(),
             e_gpu.data<float>(),
             /*M=*/11, /*K=*/17, /*N=*/9,
             nullptr);

    cudaDeviceSynchronize();

    Tensor out_cpu = e_gpu.to_cpu();
    check_tensors_close(ref, out_cpu, "test_gpu_linear");

    // ---- Backward pass ----
    // gy: (11, 9)
    Tensor gy_cpu = Tensor::randn(Shape{{11, 9}}, cpu_opts);

    // CPU reference gradients
    Tensor ga_ref = OwnTensor::matmul(gy_cpu, b_cpu) + Tensor::ones(a_cpu.shape(), cpu_opts);
    Tensor gb_ref = OwnTensor::matmul(a_cpu.t(), gy_cpu).t() + Tensor::ones(b_cpu.shape(), cpu_opts);
    Tensor gc_ref = gy_cpu + Tensor::ones(c_cpu.shape(), cpu_opts);

    Tensor gy_gpu = gy_cpu.to(gpu_opts.device);

    auto ga_gpu = Tensor::ones(ga_ref.shape(), gpu_opts);
    auto gb_gpu = Tensor::ones(gb_ref.shape(), gpu_opts);
    auto gc_gpu = Tensor::ones(gc_ref.shape(), gpu_opts);


    // ---- GPU backward ----
    K.vjp_linear(
        ga_gpu.data<float>(),
        gb_gpu.data<float>(),
        gc_gpu.data<float>(),
        gy_gpu.data<float>(),
        a_gpu.data<float>(),
        b_gpu.data<float>(),
        c_gpu.data<float>(),
        /*M=*/11, /*K=*/17, /*N=*/9,
        nullptr);

    cudaDeviceSynchronize();

    // Move outputs back to CPU
    Tensor ga_out = ga_gpu.to_cpu();
    Tensor gb_out = gb_gpu.to_cpu();
    Tensor gc_out = gc_gpu.to_cpu();

    check_tensors_close(ga_ref, ga_out, "test_gpu_vjp_linear (gA)");
    check_tensors_close(gb_ref, gb_out, "test_gpu_vjp_linear (gB)");
    check_tensors_close(gc_ref, gc_out, "test_gpu_vjp_linear (gC)");
}

void test_gpu_unified_matmul() {
    auto& K = kernels::cuda();
    auto cpu_opts = TensorOptions().with_device(Device::CPU);
    auto gpu_opts = TensorOptions().with_device(DeviceIndex(Device::CUDA));

    // A: (M × K),  B: (K × N)
    const int M = 8, Kdim = 16, N = 8;

    Tensor a_cpu = Tensor::randn(Shape{{M, Kdim}}, cpu_opts);
    Tensor b_cpu = Tensor::randn(Shape{{Kdim, N}}, cpu_opts);

    // ----- CPU reference forward -----
    Tensor ref = OwnTensor::matmul(a_cpu, b_cpu);

    // GPU input tensors
    Tensor a_gpu = a_cpu.to(gpu_opts.device);
    Tensor b_gpu = b_cpu.to(gpu_opts.device);
    Tensor out_gpu(ref.shape(), options(ref).with_device(gpu_opts.device));

    // ----- GPU forward -----
    K.matmul(a_gpu.data<float>(), b_gpu.data<float>(), out_gpu.data<float>(),
             M, Kdim, N, nullptr);

    cudaDeviceSynchronize();

    Tensor out_cpu = out_gpu.to_cpu();
    check_tensors_close(ref, out_cpu, "test_gpu_unified_matmul (forward)");

    // ----- Backward pass -----
    Tensor gy_cpu = Tensor::randn(Shape{{M, N}}, cpu_opts);

    // Reference CPU grads
    Tensor ga_ref = OwnTensor::matmul(gy_cpu, b_cpu.t());
    Tensor gb_ref = OwnTensor::matmul(a_cpu.t(), gy_cpu);

    // GPU gradient buffers
    Tensor gy_gpu = gy_cpu.to(gpu_opts.device);
    Tensor ga_gpu(ga_ref.shape(), options(ga_ref).with_device(gpu_opts.device));
    Tensor gb_gpu(gb_ref.shape(), options(gb_ref).with_device(gpu_opts.device));

    // ----- GPU VJP -----
    K.vjp_matmul(ga_gpu.data<float>(), gb_gpu.data<float>(), gy_gpu.data<float>(),
                 a_gpu.data<float>(), b_gpu.data<float>(),
                 M, Kdim, N, nullptr);

    cudaDeviceSynchronize();

    // Move results back to CPU
    Tensor ga_out = ga_gpu.to_cpu();
    Tensor gb_out = gb_gpu.to_cpu();

    check_tensors_close(ga_ref, ga_out, "test_gpu_unified_matmul (gA)");
    check_tensors_close(gb_ref, gb_out, "test_gpu_unified_matmul (gB)");
}





int main() {
    std::cout << "=== Running GPU Kernel Tests ===\n";
    try {
        #if defined(_WIN32)
            const char* plugin_path = "./agkernels_cuda.dll";
        #elif defined(__APPLE__)
            const char* plugin_path = "./libagkernels_cuda.dylib";
        #else
            const char* plugin_path = "/home/blubridge-034/Downloads/Newf/cgadimpl/cgadimpl/build/libagkernels_cuda.so";
        #endif

        std::cout << "Loading GPU plugin from: " << plugin_path << "\n";
        kernels::load_cuda_plugin(plugin_path);

        test_gpu_unified_add();
        test_gpu_unified_matmul();
        test_gpu_unified_linear();

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\nAll GPU kernel tests passed successfully!\n";
    return 0;
}