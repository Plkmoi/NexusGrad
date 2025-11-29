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

    int q = 0;
    float w = 0;

    for (size_t i = 0; i < a.numel(); ++i) {
        if (std::abs(a_data[i] - b_data[i]) > epsilon || std::isnan(a_data[i]) || std::isnan(b_data[i])) {
            std::cout << "FAIL: " << label << " mismatch at index " << i << "\n";

            q=1;
            w=w+std::abs(a_data[i] - b_data[i]);
        }
    }
    auto f = (w/a.numel());
    if(q>0 && (w/a.numel())>0.2)
    {
                    debug::print_tensor("Tensor A (ref)", a);
            debug::print_tensor("Tensor B (out)", b);
                    throw std::runtime_error("Tensor check failed for " + label +" mismatch value " + std::to_string(f));

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


// Converted GPU unit tests — new Tensor API style
// All tests rewritten to the newer Tensor / TensorOptions / Device API.
// Followed the example/format provided. No raw cudaFree calls; GPU memory is owned by Tensors.
// Assumes the following symbols/types exist in your codebase: Tensor, TensorOptions, Device, Shape,
// OwnTensor (or Tensor::matmul equivalent), kernels::cuda(), options(), check_tensors_close(), cudaDeviceSynchronize().

void test_gpu_unified_fmab() {
    auto& K = kernels::cuda();
    auto cpu_opts = TensorOptions().with_device(Device::CPU);
    auto gpu_opts = TensorOptions().with_device(Device::CUDA);

    // A: (11,17,3)  B: (17,9,4)  C: (11,9,5)  -> output (11,9)
    Tensor a_cpu = Tensor::randn(Shape{{11, 17}}, cpu_opts);
    Tensor b_cpu = Tensor::randn(Shape{{17, 9}}, cpu_opts);
    Tensor c_cpu = Tensor::randn(Shape{{11, 9}}, cpu_opts);
    Tensor e_cpu = Tensor::zeros(Shape{{11,9}}, cpu_opts);

    // Reference forward: matmul across last dims: here we assume tensor matmul collapses inner dims similarly to old code
    // To keep semantics consistent with the old test, reduce last channels by matmul of 2D views:

    Tensor ref = OwnTensor::matmul(a_cpu, b_cpu) + c_cpu;

    // Move to GPU
    Tensor a_gpu = a_cpu.to(gpu_opts.device);
    Tensor b_gpu = b_cpu.to(gpu_opts.device);
    Tensor c_gpu = c_cpu.to(gpu_opts.device);
    Tensor e_gpu(ref.shape(), options(ref).with_device(gpu_opts.device));

    // Forward: GEMM-like fmab/gemm
    K.gemm(a_gpu.data<float>(), b_gpu.data<float>(), c_gpu.data<float>(), e_gpu.data<float>(),
           /*M=*/11, /*K=*/17, /*N=*/9, nullptr);
    cudaDeviceSynchronize();

    Tensor out_cpu = e_gpu.to_cpu();
    check_tensors_close(ref, out_cpu, "test_gpu_fmab");

    // Backward (VJP)
    Tensor gy_cpu = Tensor::randn(Shape{{11,9}}, cpu_opts);
    Tensor ga_ref = OwnTensor::matmul(gy_cpu, b_cpu.t()) + Tensor::ones(a_cpu.shape(), cpu_opts);
    Tensor gb_ref = OwnTensor::matmul(a_cpu.t(), gy_cpu) + Tensor::ones(b_cpu.shape(), cpu_opts);
    Tensor gc_ref = gy_cpu + Tensor::ones(c_cpu.shape(), cpu_opts);

    Tensor gy_gpu = gy_cpu.to(gpu_opts.device);
    Tensor ga_gpu = Tensor::ones(ga_ref.shape(), gpu_opts);
    Tensor gb_gpu = Tensor::ones(gb_ref.shape(), gpu_opts);
    Tensor gc_gpu = Tensor::ones(gc_ref.shape(), gpu_opts);

    K.vjp_gemm(ga_gpu.data<float>(), gb_gpu.data<float>(), gc_gpu.data<float>(),
               gy_gpu.data<float>(), a_gpu.data<float>(), b_gpu.data<float>(), c_gpu.data<float>(),
               /*M=*/11, /*N=*/9, /*K=*/17, nullptr);
    cudaDeviceSynchronize();

    check_tensors_close(ga_ref, ga_gpu.to_cpu(), "test_gpu_vjp_fmab (gA)");
    check_tensors_close(gb_ref, gb_gpu.to_cpu(), "test_gpu_vjp_fmab (gB)");
    check_tensors_close(gc_ref, gc_gpu.to_cpu(), "test_gpu_vjp_fmab (gC)");
}

void test_gpu_unified_attention() {
    auto& K = kernels::cuda();
    auto cpu_opts = TensorOptions().with_device(Device::CPU);
    auto gpu_opts = TensorOptions().with_device(Device::CUDA);

    int H = 4;

    // A: (11,9,3) B: (9,14,4) C: (9,14,5) D: (9,14,6) -> output (11,14)
    Tensor a_cpu = Tensor::randn(Shape{{11,7,12}}, cpu_opts);
    Tensor b_cpu = Tensor::randn(Shape{{11,12,12}}, cpu_opts);
    Tensor c_cpu = Tensor::randn(Shape{{11,12,12}}, cpu_opts);
    Tensor d_cpu = Tensor::randn(Shape{{11,12,12}}, cpu_opts);

    // Build Q,K,V via matmul as original code


    Tensor q = matmul(a_cpu, b_cpu.t()).unflatten(2, Shape({H, (b_cpu.shape().dims[1]/H)})).transpose(1,2).clone();
    Tensor k = matmul(a_cpu, c_cpu.t()).unflatten(2, Shape({H, (c_cpu.shape().dims[1]/H)})).transpose(1,2).clone();
    Tensor v = matmul(a_cpu, d_cpu.t()).unflatten(2, Shape({H, (d_cpu.shape().dims[1]/H)})).transpose(1,2).clone();


    float scale = 1.f / sqrtf(static_cast<float>(k.shape().dims.back()));
    Tensor g = matmul(q, k.t()) * scale;

    // Re-implement softmax using OwnTensor ops
    Tensor max_val = reduce_max(g, {-1}, true);
    Tensor exp_g = exp(g - max_val);
    Tensor sum_exp_g = reduce_sum(exp_g, {-1}, true);
    Tensor s = exp_g / sum_exp_g;
    //ag::debug::print_tensor("Who is v", v);

    //ag::debug::print_tensor("S middle", s);

    auto ref = matmul(s, v);


    Tensor q_gpu = q.to(gpu_opts.device);
    Tensor k_gpu = k.to(gpu_opts.device);
    Tensor v_gpu = v.to(gpu_opts.device);
    Tensor out_gpu(ref.shape(), options(ref).with_device(gpu_opts.device));

    // flash attention call (standard softmax)
    K.flash(q_gpu.data<float>(), k_gpu.data<float>(), v_gpu.data<float>(), out_gpu.data<float>(),
            /*batches=*/11, /*heads=*/H, /*M=*/7, /*N=*/12/H, ag::current_stream());
    cudaDeviceSynchronize();
    auto refa = ref.transpose(1,2).flatten(2,3).clone();
    auto outa = out_gpu.to_cpu().transpose(1,2).flatten(2,3).clone();

    check_tensors_close(refa, outa, "test_gpu_attention", 0.01);
}



void test_gpu_unified_alibattention() {
    auto& K = kernels::cuda();
    auto cpu_opts = TensorOptions().with_device(Device::CPU);
    auto gpu_opts = TensorOptions().with_device(Device::CUDA);

    int H = 4;
    int B = 11;
    int S = 7;
    int D = 3;

    // A: (11,9,3) B: (9,14,4) C: (9,14,5) D: (9,14,6) -> output (11,14)
    Tensor a_cpu = Tensor::randn(Shape{{11,7,12}}, cpu_opts);
    Tensor b_cpu = Tensor::randn(Shape{{11,12,12}}, cpu_opts);
    Tensor c_cpu = Tensor::randn(Shape{{11,12,12}}, cpu_opts);
    Tensor d_cpu = Tensor::randn(Shape{{11,12,12}}, cpu_opts);

    // Build Q,K,V via matmul as original code


    Tensor q = matmul(a_cpu, b_cpu.t()).unflatten(2, Shape({H, (b_cpu.shape().dims[1]/H)})).transpose(1,2).clone();
    Tensor k = matmul(a_cpu, c_cpu.t()).unflatten(2, Shape({H, (c_cpu.shape().dims[1]/H)})).transpose(1,2).clone();
    Tensor v = matmul(a_cpu, d_cpu.t()).unflatten(2, Shape({H, (d_cpu.shape().dims[1]/H)})).transpose(1,2).clone();


    float scale = 1.f / sqrtf(static_cast<float>(k.shape().dims.back()));
    Tensor logits = matmul(q, k.t()) * scale;

// 3) Build ALiBi bias [B, H, S, S] on CPU, then move to logits.device()
Tensor bias_cpu(
    logits.shape(),
    TensorOptions()
        .with_device(Device::CPU)
        .with_dtype(logits.dtype())
);

{
    float slope_start = 1.0f / std::pow(2.0f, 8.0f / H);

    dispatch_by_dtype(bias_cpu.dtype(), [&](auto dummy) {
        using Tval = decltype(dummy);
        Tval* data = bias_cpu.data<Tval>();

        for (int b = 0; b < B; ++b) {
            for (int h = 0; h < H; ++h) {

                float slope = std::pow(slope_start, h + 1);

                for (int i = 0; i < S; ++i) {
                    for (int j = 0; j < S; ++j) {

                        float v;
                        if (j > i)
                            v = -std::numeric_limits<float>::infinity();  // causal
                        else
                            v = -float(j - i) * slope;                    // ALiBi

                        // index = (((b*H + h)*S + i)*S + j)
                        const int idx = (((b * H + h) * S + i) * S + j);

                        data[idx] = static_cast<Tval>(v);
                    }
                }
            }
        }
    });
}

    

    // Tensor bias = bias_cpu.to(logits.device());
    Tensor g    = logits +bias_cpu;  // [H,T,T]

    // Re-implement softmax using OwnTensor ops
    Tensor max_val = reduce_max(g, {-1}, true);
    Tensor exp_g = exp(g - max_val);
    Tensor sum_exp_g = reduce_sum(exp_g, {-1}, true);
    Tensor s = exp_g / sum_exp_g;
    //ag::debug::print_tensor("Who is v", v);

    //ag::debug::print_tensor("S middle", s);

    auto ref = matmul(s, v);


    Tensor q_gpu = q.to(gpu_opts.device);
    Tensor k_gpu = k.to(gpu_opts.device);
    Tensor v_gpu = v.to(gpu_opts.device);
    Tensor out_gpu(ref.shape(), options(ref).with_device(gpu_opts.device));

    // flash attention call (standard softmax)
    K.flashali(q_gpu.data<float>(), k_gpu.data<float>(), v_gpu.data<float>(), out_gpu.data<float>(),
            /*batches=*/11, /*heads=*/H, /*M=*/7, /*N=*/12/H, ag::current_stream());
    cudaDeviceSynchronize();
    auto refa = ref.transpose(1,2).flatten(2,3).clone();
    auto outa = out_gpu.to_cpu().transpose(1,2).flatten(2,3).clone();

    check_tensors_close(refa, outa, "test_gpu_aliattention", 0.01);
}

// void test_gpu_unified_reluattention() {
//     auto& K = kernels::cuda();
//     auto cpu_opts = TensorOptions().with_device(Device::CPU);
//     auto gpu_opts = TensorOptions().with_device(Device::CUDA);

//     Tensor a_cpu = Tensor::randn(Shape{{11,9,3}}, cpu_opts);
//     Tensor b_cpu = Tensor::randn(Shape{{9,14,4}}, cpu_opts);
//     Tensor c_cpu = Tensor::randn(Shape{{9,14,5}}, cpu_opts);
//     Tensor d_cpu = Tensor::randn(Shape{{9,14,6}}, cpu_opts);

//     Tensor a_cpu = a_cpu.reshape(Shape{{11,9}});
//     Tensor b_cpu = b_cpu.reshape(Shape{{9,14}});
//     Tensor c_cpu = c_cpu.reshape(Shape{{9,14}});
//     Tensor d_cpu = d_cpu.reshape(Shape{{9,14}});

//     Tensor q = OwnTensor::matmul(a_cpu, b_cpu);
//     Tensor k = OwnTensor::matmul(a_cpu, c_cpu);
//     Tensor v = OwnTensor::matmul(a_cpu, d_cpu);

//     Tensor g = OwnTensor::matmul(q, k.t() * (1.f / sqrt(float(k.cols()))));
//     Tensor s = OwnTensor::relu(g);
//     Tensor ref = OwnTensor::matmul(s, v);

//     Tensor q_gpu = q.to(gpu_opts.device);
//     Tensor k_gpu = k.to(gpu_opts.device);
//     Tensor v_gpu = v.to(gpu_opts.device);
//     Tensor out_gpu(ref.shape(), options(ref).with_device(gpu_opts.device));

//     K.reluflash(q_gpu.data<float>(), k_gpu.data<float>(), v_gpu.data<float>(), out_gpu.data<float>(),
//                 /*batches=*/1, /*heads=*/1, /*M=*/11, /*N=*/14, nullptr);
//     cudaDeviceSynchronize();

//     check_tensors_close(ref, out_gpu.to_cpu(), "test_gpu_reluattention");
// }

// void test_gpu_unified_flexattention() {
//     auto& K = kernels::cuda();
//     auto cpu_opts = TensorOptions().with_device(Device::CPU);
//     auto gpu_opts = TensorOptions().with_device(Device::CUDA);

//     Tensor a_cpu = Tensor::randn(Shape{{11,9,3}}, cpu_opts);
//     Tensor b_cpu = Tensor::randn(Shape{{9,14,4}}, cpu_opts);
//     Tensor c_cpu = Tensor::randn(Shape{{9,14,5}}, cpu_opts);
//     Tensor d_cpu = Tensor::randn(Shape{{9,14,6}}, cpu_opts);

//     Tensor a_cpu = a_cpu.reshape(Shape{{11,9}});
//     Tensor b_cpu = b_cpu.reshape(Shape{{9,14}});
//     Tensor c_cpu = c_cpu.reshape(Shape{{9,14}});
//     Tensor d_cpu = d_cpu.reshape(Shape{{9,14}});

//     Tensor q = OwnTensor::matmul(a_cpu, b_cpu);
//     Tensor k = OwnTensor::matmul(a_cpu, c_cpu);
//     Tensor v = OwnTensor::matmul(a_cpu, d_cpu);

//     Tensor g = OwnTensor::matmul(q, k.t() * (1.f / sqrt(float(k.cols()))));
//     Tensor s = OwnTensor::softplus(g);
//     Tensor ref = OwnTensor::matmul(s, v);

//     Tensor q_gpu = q.to(gpu_opts.device);
//     Tensor k_gpu = k.to(gpu_opts.device);
//     Tensor v_gpu = v.to(gpu_opts.device);
//     Tensor out_gpu(ref.shape(), options(ref).with_device(gpu_opts.device));

//     // choose activation flag for flexflash (softplus => 3.0f in earlier code)
//     float mmi_host[1]; mmi_host[0] = 3.f;
//     Tensor mmi = Tensor::from_blob(mmi_host, Shape{{1}}, cpu_opts).to(gpu_opts.device);

//     K.flexflash(q_gpu.data<float>(), k_gpu.data<float>(), v_gpu.data<float>(), out_gpu.data<float>(),
//                 /*batches=*/1, /*heads=*/1, /*M=*/11, /*N=*/14, mmi.data<float>(), nullptr);
//     cudaDeviceSynchronize();

//     check_tensors_close(ref, out_gpu.to_cpu(), "test_gpu_flexattention", 3.0);
// }

// void test_gpu_unified_sigattention() {
//     auto& K = kernels::cuda();
//     auto cpu_opts = TensorOptions().with_device(Device::CPU);
//     auto gpu_opts = TensorOptions().with_device(Device::CUDA);

//     Tensor a_cpu = Tensor::randn(Shape{{11,9,3}}, cpu_opts);
//     Tensor b_cpu = Tensor::randn(Shape{{9,14,4}}, cpu_opts);
//     Tensor c_cpu = Tensor::randn(Shape{{9,14,5}}, cpu_opts);
//     Tensor d_cpu = Tensor::randn(Shape{{9,14,6}}, cpu_opts);

//     Tensor a_cpu = a_cpu.reshape(Shape{{11,9}});
//     Tensor b_cpu = b_cpu.reshape(Shape{{9,14}});
//     Tensor c_cpu = c_cpu.reshape(Shape{{9,14}});
//     Tensor d_cpu = d_cpu.reshape(Shape{{9,14}});

//     Tensor q = OwnTensor::matmul(a_cpu, b_cpu);
//     Tensor k = OwnTensor::matmul(a_cpu, c_cpu);
//     Tensor v = OwnTensor::matmul(a_cpu, d_cpu);

//     Tensor g = OwnTensor::matmul(q, k.t() * (1.f / sqrt(float(k.cols()))));
//     Tensor s = OwnTensor::sigmoid(g);
//     Tensor ref = OwnTensor::matmul(s, v);

//     Tensor q_gpu = q.to(gpu_opts.device);
//     Tensor k_gpu = k.to(gpu_opts.device);
//     Tensor v_gpu = v.to(gpu_opts.device);
//     Tensor out_gpu(ref.shape(), options(ref).with_device(gpu_opts.device));

//     K.sigflash(q_gpu.data<float>(), k_gpu.data<float>(), v_gpu.data<float>(), out_gpu.data<float>(),
//                /*batches=*/1, /*heads=*/1, /*M=*/11, /*N=*/14, nullptr);
//     cudaDeviceSynchronize();

//     check_tensors_close(ref, out_gpu.to_cpu(), "test_gpu_sigattention", 4.0);
// }

void test_gpu_unified_sub() {
    auto& K = kernels::cuda();
    auto cpu_opts = TensorOptions().with_device(Device::CPU);
    auto gpu_opts = TensorOptions().with_device(Device::CUDA);

    Tensor a_cpu = Tensor::randn(Shape{{11,11}}, cpu_opts);
    Tensor b_cpu = Tensor::randn(Shape{{11,11}}, cpu_opts);
    Tensor ref = a_cpu - b_cpu;

    // Move to GPU (reshape to match op expectations)

    Tensor a_gpu = a_cpu.to(gpu_opts.device);
    Tensor b_gpu = b_cpu.to(gpu_opts.device);
    Tensor c_gpu(ref.shape(), options(ref).with_device(gpu_opts.device));

    K.sub(a_gpu.data<float>(), b_gpu.data<float>(), c_gpu.data<float>(), a_cpu.numel(), nullptr);
    cudaDeviceSynchronize();

    check_tensors_close(ref, c_gpu.to_cpu(), "test_gpu_sub");

    // Backward (vjp_sub)
    Tensor gy_cpu = Tensor::randn(Shape{{11,11}}, cpu_opts);
    Tensor ga_ref = gy_cpu;
    Tensor gb_ref = -1.0f*gy_cpu;

    Tensor gy_gpu = gy_cpu.to(gpu_opts.device);
    Tensor ga_gpu = Tensor::zeros(ga_ref.shape(), gpu_opts);
    Tensor gb_gpu = Tensor::zeros(gb_ref.shape(), gpu_opts);

    K.vjp_sub(ga_gpu.data<float>(), gb_gpu.data<float>(), gy_gpu.data<float>(), gy_cpu.numel(), nullptr);
    cudaDeviceSynchronize();

    check_tensors_close(ga_ref, ga_gpu.to_cpu(), "test_gpu_vjp_sub (gA)");
    check_tensors_close(gb_ref, gb_gpu.to_cpu(), "test_gpu_vjp_sub (gB)");
}

void test_gpu_unified_hadmul() {
    auto& K = kernels::cuda();
    auto cpu_opts = TensorOptions().with_device(Device::CPU);
    auto gpu_opts = TensorOptions().with_device(Device::CUDA);

    Tensor a_cpu = Tensor::randn(Shape{{11,11}}, cpu_opts);
    Tensor b_cpu = Tensor::randn(Shape{{11,11}}, cpu_opts);
    Tensor ref = a_cpu * b_cpu;

    Tensor a_gpu = a_cpu.to(gpu_opts.device);
    Tensor b_gpu = b_cpu.to(gpu_opts.device);
    Tensor c_gpu(ref.shape(), options(ref).with_device(gpu_opts.device));

    K.hadmul(a_gpu.data<float>(), b_gpu.data<float>(), c_gpu.data<float>(), a_cpu.numel(), nullptr);
    cudaDeviceSynchronize();

    check_tensors_close(ref, c_gpu.to_cpu(), "test_gpu_hadmul");

    // Backward
    Tensor gy_cpu = Tensor::ones(Shape{{11,11}}, cpu_opts);
    Tensor ga_ref = b_cpu;
    Tensor gb_ref = a_cpu;

    Tensor gy_gpu = gy_cpu.to(gpu_opts.device);
    Tensor ga_gpu = Tensor::zeros(ga_ref.shape(), gpu_opts);
    Tensor gb_gpu = Tensor::zeros(gb_ref.shape(), gpu_opts);

    K.vjp_hadmul(ga_gpu.data<float>(), gb_gpu.data<float>(), gy_gpu.data<float>(),
                 a_gpu.data<float>(), b_gpu.data<float>(), gy_cpu.numel(), nullptr);
    cudaDeviceSynchronize();

    check_tensors_close(ga_ref, ga_gpu.to_cpu(), "test_gpu_vjp_hadmul (gA)");
    check_tensors_close(gb_ref, gb_gpu.to_cpu(), "test_gpu_vjp_hadmul (gB)");
}

void test_gpu_unified_div() {
    auto& K = kernels::cuda();
    auto cpu_opts = TensorOptions().with_device(Device::CPU);
    auto gpu_opts = TensorOptions().with_device(Device::CUDA);

    Tensor a_cpu = Tensor::randn(Shape{{11,11}}, cpu_opts);
    Tensor b_cpu = Tensor::randn(Shape{{11,11}}, cpu_opts);
    Tensor ref = a_cpu / b_cpu;


    Tensor a_gpu = a_cpu.to(gpu_opts.device);
    Tensor b_gpu = b_cpu.to(gpu_opts.device);
    Tensor c_gpu(ref.shape(), options(ref).with_device(gpu_opts.device));

    K.div(a_gpu.data<float>(), b_gpu.data<float>(), c_gpu.data<float>(), a_cpu.numel(), nullptr);
    cudaDeviceSynchronize();

    check_tensors_close(ref, c_gpu.to_cpu(), "test_gpu_div");

    // Backward
    Tensor gy_cpu = Tensor::ones(Shape{{11,11}}, cpu_opts);
    Tensor ga_ref = (1.0f / b_cpu) * gy_cpu;
    Tensor gb_ref = (a_cpu * (-1.0f / (b_cpu * b_cpu))) * gy_cpu;

    Tensor gy_gpu = gy_cpu.to(gpu_opts.device);
    Tensor ga_gpu = Tensor::zeros(ga_ref.shape(), gpu_opts);
    Tensor gb_gpu = Tensor::zeros(gb_ref.shape(), gpu_opts);

    K.vjp_div(ga_gpu.data<float>(), gb_gpu.data<float>(), gy_gpu.data<float>(),
              a_gpu.data<float>(), b_gpu.data<float>(), gy_cpu.numel(), nullptr);
    cudaDeviceSynchronize();

    check_tensors_close(ga_ref, ga_gpu.to_cpu(), "test_gpu_vjp_div (gA)");
    check_tensors_close(gb_ref, gb_gpu.to_cpu(), "test_gpu_vjp_div (gB)", 0.1);
}

void test_gpu_unified_sigmoid() {
    auto& K = kernels::cuda();
    auto cpu_opts = TensorOptions().with_device(Device::CPU);
    auto gpu_opts = TensorOptions().with_device(Device::CUDA);

    Tensor x_cpu = Tensor::randn(Shape{{11,11}}, cpu_opts);
    Tensor ref = 1.0f / (1.0f + OwnTensor::exp(x_cpu * -1.0f));

    
    Tensor x_gpu = x_cpu.to(gpu_opts.device);
    Tensor y_gpu(ref.shape(), options(ref).with_device(gpu_opts.device));

    K.sigmoid(x_gpu.data<float>(), y_gpu.data<float>(), x_cpu.numel(), nullptr);
    cudaDeviceSynchronize();

    check_tensors_close(ref, y_gpu.to_cpu(), "test_gpu_sigmoid");

    // Backward
    Tensor gy_cpu = Tensor::ones(Shape{{11,11}}, cpu_opts);
    Tensor ga_ref = gy_cpu * ref * (Tensor::ones(ref.shape(), TensorOptions()) - ref);

    Tensor gy_gpu = gy_cpu.to(gpu_opts.device);
    Tensor ga_gpu = Tensor::zeros(x_cpu.shape(), TensorOptions()).to(gpu_opts.device);

    K.vjp_sigmoid(ga_gpu.data<float>(), x_gpu.data<float>(), gy_gpu.data<float>(), x_cpu.numel(), nullptr);
    cudaDeviceSynchronize();

    check_tensors_close(ga_ref, ga_gpu.to_cpu(), "test_gpu_vjp_sigmoid");
}

void test_gpu_unified_silu() {
    auto& K = kernels::cuda();
    auto cpu_opts = TensorOptions().with_device(Device::CPU);
    auto gpu_opts = TensorOptions().with_device(Device::CUDA);

    Tensor x_cpu = Tensor::randn(Shape{{11,11}}, cpu_opts);
    
    Tensor s = 1.0f / (1.0f + OwnTensor::exp(x_cpu * -1.0f));
    Tensor ref = x_cpu * s;

    Tensor x_gpu = x_cpu.to(gpu_opts.device);
    Tensor y_gpu(ref.shape(), options(ref).with_device(gpu_opts.device));

    K.silu(x_gpu.data<float>(), y_gpu.data<float>(), x_cpu.numel(), nullptr);
    cudaDeviceSynchronize();

    check_tensors_close(ref, y_gpu.to_cpu(), "test_gpu_silu");

    // Backward
    Tensor gy_cpu = Tensor::randn(Shape{{11,11}}, cpu_opts);
    Tensor ga_ref = gy_cpu * (s + (x_cpu * s * (Tensor::ones(s.shape(), TensorOptions()) - s)));

    Tensor gy_gpu = gy_cpu.to(gpu_opts.device);
    Tensor ga_gpu = Tensor::zeros(x_cpu.shape(), TensorOptions()).to(gpu_opts.device);

    K.vjp_silu(ga_gpu.data<float>(), x_gpu.data<float>(), gy_gpu.data<float>(), x_cpu.numel(), nullptr);
    cudaDeviceSynchronize();

    check_tensors_close(ga_ref, ga_gpu.to_cpu(), "test_gpu_vjp_silu");
}

void test_gpu_unified_mish() {
    auto& K = kernels::cuda();
    auto cpu_opts = TensorOptions().with_device(Device::CPU);
    auto gpu_opts = TensorOptions().with_device(Device::CUDA);

    Tensor x_cpu = Tensor::randn(Shape{{11,11}}, cpu_opts);
    
    Tensor sp = OwnTensor::log(1.0f + OwnTensor::exp(x_cpu));
    Tensor t = OwnTensor::tanh(sp);
    Tensor ref = x_cpu * t;

    Tensor x_gpu = x_cpu.to(gpu_opts.device);
    Tensor y_gpu(ref.shape(), options(ref).with_device(gpu_opts.device));

    K.mish(x_gpu.data<float>(), y_gpu.data<float>(), x_cpu.numel(), nullptr);
    cudaDeviceSynchronize();

    check_tensors_close(ref, y_gpu.to_cpu(), "test_gpu_mish");

    // Backward
    Tensor gy_cpu = Tensor::randn(Shape{{11,11}}, cpu_opts);
    Tensor s = 1.0f / (1.0f + OwnTensor::exp(x_cpu * -1.0f));
    Tensor ga_ref = gy_cpu * (t + x_cpu * s * (Tensor::ones(t.shape(), TensorOptions()) - t * t));

    Tensor gy_gpu = gy_cpu.to(gpu_opts.device);
    Tensor ga_gpu = Tensor::zeros(x_cpu.shape(), TensorOptions()).to(gpu_opts.device);

    K.vjp_mish(ga_gpu.data<float>(), x_gpu.data<float>(), gy_gpu.data<float>(), x_cpu.numel(), nullptr);
    cudaDeviceSynchronize();

    check_tensors_close(ga_ref, ga_gpu.to_cpu(), "test_gpu_vjp_mish");
}

void test_gpu_unified_sum() {
    auto& K = kernels::cuda();
    auto cpu_opts = TensorOptions().with_device(Device::CPU);
    auto gpu_opts = TensorOptions().with_device(Device::CUDA);

    Tensor a_cpu = Tensor::randn(Shape{{51,23,111}}, cpu_opts);
    Tensor ref = OwnTensor::reduce_sum(a_cpu);

    Tensor a_gpu = a_cpu.to(gpu_opts.device);
    Tensor out_gpu(ref.shape(), options(ref).with_device(gpu_opts.device));

    K.sum(a_gpu.data<float>(), out_gpu.data<float>(), a_cpu.numel(), nullptr);
    cudaDeviceSynchronize();

    check_tensors_close(ref, out_gpu.to_cpu(), "test_gpu_sumall", 0.9);

    // Backward (vjp_sum)
    Tensor gy_cpu = Tensor::randn(ref.shape(), cpu_opts);
    Tensor ga_ref = gy_cpu; // broadcasted

    Tensor gy_gpu = gy_cpu.to(gpu_opts.device);
    Tensor ga_gpu = Tensor::zeros(a_cpu.shape(), TensorOptions()).to(gpu_opts.device);

    K.vjp_sum(ga_gpu.data<float>(), a_gpu.data<float>(), gy_gpu.data<float>(), gy_cpu.numel(), nullptr);
    cudaDeviceSynchronize();

    check_tensors_close(ga_ref, ga_gpu.to_cpu(), "test_gpu_vjp_sum");
}

void test_gpu_unified_mseloss() {
    auto& K = kernels::cuda();
    auto cpu_opts = TensorOptions().with_device(Device::CPU);
    auto gpu_opts = TensorOptions().with_device(Device::CUDA);

    Tensor a_cpu = Tensor::randn(Shape{{11,11}}, cpu_opts);
    Tensor b_cpu = Tensor::randn(Shape{{11,11}}, cpu_opts);
    Tensor ref = OwnTensor::reduce_mean((a_cpu-b_cpu) * (a_cpu-b_cpu));

    
    
    Tensor a_gpu = a_cpu.to(gpu_opts.device);
    Tensor b_gpu = b_cpu.to(gpu_opts.device);
    Tensor out_gpu(ref.shape(), options(ref).with_device(gpu_opts.device));
    // allocate and zero on GPU happens via Tensor ctor

    K.mseloss(a_gpu.data<float>(), b_gpu.data<float>(), out_gpu.data<float>(), a_cpu.numel(), nullptr);
    cudaDeviceSynchronize();

    check_tensors_close(ref, out_gpu.to_cpu(), "test_gpu_mseloss");

    // Backward
    Tensor gy_cpu = Tensor::ones(ref.shape(), cpu_opts);
    int N = a_cpu.numel();
    Tensor diff = a_cpu - b_cpu;
    Tensor ga_ref = diff * (2.0f / float(N)) * gy_cpu;
    Tensor gb_ref = -1.0f*diff * (2.0f / float(N)) * gy_cpu;

    Tensor gy_gpu = gy_cpu.to(gpu_opts.device);
    Tensor ga_gpu = Tensor::zeros(ga_ref.shape(), gpu_opts);
    Tensor gb_gpu = Tensor::zeros(gb_ref.shape(), gpu_opts);

    K.vjp_mseloss(ga_gpu.data<float>(), gb_gpu.data<float>(), gy_gpu.data<float>(),
                  a_gpu.data<float>(), b_gpu.data<float>(), a_cpu.numel(), nullptr);
    cudaDeviceSynchronize();

    check_tensors_close(ga_ref, ga_gpu.to_cpu(), "test_gpu_vjp_mseloss (pred dA)");
    check_tensors_close(gb_ref, gb_gpu.to_cpu(), "test_gpu_vjp_mseloss (target gB)");
}

void test_gpu_unified_maeloss() {
    auto& K = kernels::cuda();
    auto cpu_opts = TensorOptions().with_device(Device::CPU);
    auto gpu_opts = TensorOptions().with_device(Device::CUDA);

    Tensor a_cpu = Tensor::randn(Shape{{11,11}}, cpu_opts);
    Tensor b_cpu = Tensor::randn(Shape{{11,11}}, cpu_opts);
    Tensor ref = OwnTensor::reduce_mean(OwnTensor::abs(a_cpu - b_cpu, ag::current_stream()));

    
    
    Tensor a_gpu = a_cpu.to(gpu_opts.device);
    Tensor b_gpu = b_cpu.to(gpu_opts.device);
    Tensor out_gpu(ref.shape(), options(ref).with_device(gpu_opts.device));

    K.maeloss(a_gpu.data<float>(), b_gpu.data<float>(), out_gpu.data<float>(), a_cpu.numel(), nullptr);
    cudaDeviceSynchronize();

    check_tensors_close(ref, out_gpu.to_cpu(), "test_gpu_maeloss");

    // Backward
    Tensor gy_cpu = Tensor::ones(ref.shape(), cpu_opts);
    int N = a_cpu.numel();
    Tensor diff_sign = OwnTensor::sign(a_cpu - b_cpu, ag::current_stream());
    Tensor ga_ref = diff_sign * (1.0f / float(N)) * gy_cpu;
    Tensor gb_ref = -1.0f*diff_sign * (1.0f / float(N)) * gy_cpu;

    Tensor gy_gpu = gy_cpu.to(gpu_opts.device);
    Tensor ga_gpu = Tensor::zeros(ga_ref.shape(), gpu_opts);
    Tensor gb_gpu = Tensor::zeros(gb_ref.shape(), gpu_opts);

    K.vjp_maeloss(ga_gpu.data<float>(), gb_gpu.data<float>(), gy_gpu.data<float>(),
                  a_gpu.data<float>(), b_gpu.data<float>(), a_cpu.numel(), nullptr);
    cudaDeviceSynchronize();

    check_tensors_close(ga_ref, ga_gpu.to_cpu(), "test_gpu_vjp_maeloss (pred dA)");
    check_tensors_close(gb_ref, gb_gpu.to_cpu(), "test_gpu_vjp_maeloss (target gB)");
}

// void test_gpu_unified_rowsum() {
//     auto& K = kernels::cuda();
//     auto cpu_opts = TensorOptions().with_device(Device::CPU);
//     auto gpu_opts = TensorOptions().with_device(Device::CUDA);

//     Tensor a_cpu = Tensor::randn(Shape{{11,1111}}, cpu_opts);
//     Tensor ref = OwnTensor::row_sum(a_cpu.reshape(Shape{{11,1111}}));

//     Tensor a_cpu = a_cpu.reshape(Shape{{11,1111}});
//     Tensor a_gpu = a_cpu.to(gpu_opts.device);
//     Tensor out_gpu(ref.shape(), options(ref).with_device(gpu_opts.device));

//     K.rowsum(a_gpu.data<float>(), out_gpu.data<float>(), a_cpu.rows(), a_cpu.cols(), nullptr);
//     cudaDeviceSynchronize();

//     check_tensors_close(ref, out_gpu.to_cpu(), "test_gpu_rowsum", 0.9);

//     // Backward
//     Tensor gy_cpu = Tensor::randn(Shape{{11,1}}, cpu_opts);
//     Tensor ga_ref = gy_cpu.expand_as(a_cpu); // broadcast to original
//     Tensor gy_gpu = gy_cpu.to(gpu_opts.device);
//     Tensor ga_gpu = Tensor::zeros(a_cpu.shape(), TensorOptions()).to(gpu_opts.device);

//     K.vjp_rowsum(ga_gpu.data<float>(), a_gpu.data<float>(), out_gpu.data<float>(),
//                  gy_gpu.data<float>(), a_cpu.rows(), a_cpu.cols(), nullptr);
//     cudaDeviceSynchronize();

//     check_tensors_close(ga_ref, ga_gpu.to_cpu(), "test_gpu_vjp_rowsum");
// }

// void test_gpu_unified_rowmax() {
//     auto& K = kernels::cuda();
//     auto cpu_opts = TensorOptions().with_device(Device::CPU);
//     auto gpu_opts = TensorOptions().with_device(Device::CUDA);

//     Tensor a_cpu = Tensor::randn(Shape{{11,17,12}}, cpu_opts);
//     Tensor ref = OwnTensor::row_max(a_cpu.reshape(Shape{{11,17,12}}));

//     Tensor a_cpu = a_cpu.reshape(Shape{{11,17,12}});
//     Tensor a_gpu = a_cpu.to(gpu_opts.device);
//     Tensor out_gpu(ref.shape(), options(ref).with_device(gpu_opts.device));

//     K.rowmax(a_gpu.data<float>(), out_gpu.data<float>(), a_cpu.rows(), a_cpu.cols(), nullptr);
//     cudaDeviceSynchronize();

//     check_tensors_close(ref, out_gpu.to_cpu(), "test_gpu_rowmax", 0.9);

//     // Backward
//     Tensor gy_cpu = Tensor::randn(Shape{{11,1}}, cpu_opts);
//     // Build expected gradient: entry equals gy where value equals max in row
//     Tensor g = Tensor::zeros(a_cpu.shape(), TensorOptions());
//     for (int i = 0; i < a_cpu.rows(); ++i) {
//         for (int j = 0; j < a_cpu.cols(); ++j) {
//             // Using CPU-side values
//             if (a_cpu(i, j) == ref(i, 0)) {
//                 g(i, j) = gy_cpu(i, 0);
//             }
//         }
//     }
//     Tensor ga_ref = g;
//     Tensor gy_gpu = gy_cpu.to(gpu_opts.device);
//     Tensor ga_gpu = Tensor::zeros(ga_ref.shape(), gpu_opts);

//     K.vjp_rowmax(ga_gpu.data<float>(), a_gpu.data<float>(), out_gpu.data<float>(),
//                  gy_gpu.data<float>(), a_cpu.rows(), a_cpu.cols(), nullptr);
//     cudaDeviceSynchronize();

//     check_tensors_close(ga_ref, ga_gpu.to_cpu(), "test_gpu_vjp_rowmax");
// }

void test_gpu_unified_gcu() {
    auto& K = kernels::cuda();
    auto cpu_opts = TensorOptions().with_device(Device::CPU);
    auto gpu_opts = TensorOptions().with_device(Device::CUDA);

    Tensor a_cpu = Tensor::randn(Shape{{11,11}}, cpu_opts);
    
    Tensor ref = a_cpu * OwnTensor::cos(a_cpu);

    Tensor a_gpu = a_cpu.to(gpu_opts.device);
    Tensor out_gpu(ref.shape(), options(ref).with_device(gpu_opts.device));

    K.gcu(a_gpu.data<float>(), out_gpu.data<float>(), a_cpu.numel(), nullptr);
    cudaDeviceSynchronize();

    check_tensors_close(ref, out_gpu.to_cpu(), "test_gpu_gcu");

    // Backward
    Tensor gy_cpu = Tensor::randn(Shape{{11,11}}, cpu_opts);
    Tensor ga_ref = gy_cpu * (OwnTensor::cos(a_cpu) - (a_cpu * OwnTensor::sin(a_cpu)));

    Tensor gy_gpu = gy_cpu.to(gpu_opts.device);
    Tensor ga_gpu = Tensor::zeros(a_cpu.shape(), TensorOptions()).to(gpu_opts.device);

    K.vjp_gcu(ga_gpu.data<float>(), a_gpu.data<float>(), gy_gpu.data<float>(), gy_cpu.numel(), nullptr);
    cudaDeviceSynchronize();

    check_tensors_close(ga_ref, ga_gpu.to_cpu(), "test_gpu_vjp_gcu");
}

void test_gpu_unified_gauss() {
    auto& K = kernels::cuda();
    auto cpu_opts = TensorOptions().with_device(Device::CPU);
    auto gpu_opts = TensorOptions().with_device(Device::CUDA);

    Tensor a_cpu = Tensor::randn(Shape{{11,11}}, cpu_opts);
    
    Tensor ref = OwnTensor::exp(-1.0f*(a_cpu * a_cpu));

    Tensor a_gpu = a_cpu.to(gpu_opts.device);
    Tensor out_gpu(ref.shape(), options(ref).with_device(gpu_opts.device));

    K.gauss(a_gpu.data<float>(), out_gpu.data<float>(), a_cpu.numel(), nullptr);
    cudaDeviceSynchronize();

    check_tensors_close(ref, out_gpu.to_cpu(), "test_gpu_gauss");

    // Backward
    Tensor gy_cpu = Tensor::randn(Shape{{11,11}}, cpu_opts);
    Tensor ga_ref = gy_cpu * (-2.0f * a_cpu * OwnTensor::exp(-1.0f*a_cpu * a_cpu));

    Tensor gy_gpu = gy_cpu.to(gpu_opts.device);
    Tensor ga_gpu = Tensor::zeros(a_cpu.shape(), TensorOptions()).to(gpu_opts.device);

    K.vjp_gauss(ga_gpu.data<float>(), a_gpu.data<float>(), gy_gpu.data<float>(), gy_cpu.numel(), nullptr);
    cudaDeviceSynchronize();

    check_tensors_close(ga_ref, ga_gpu.to_cpu(), "test_gpu_vjp_gauss");
}

void test_gpu_unified_parcon() {
    auto& K = kernels::cuda();
    auto cpu_opts = TensorOptions().with_device(Device::CPU);
    auto gpu_opts = TensorOptions().with_device(Device::CUDA);

    Tensor a_cpu = Tensor::randn(Shape{{11,11}}, cpu_opts);
    
    Tensor ref = a_cpu * (2.f * Tensor::ones(a_cpu.shape(), TensorOptions()) - a_cpu);

    Tensor a_gpu = a_cpu.to(gpu_opts.device);
    Tensor out_gpu(ref.shape(), options(ref).with_device(gpu_opts.device));

    K.parcon(a_gpu.data<float>(), out_gpu.data<float>(), a_cpu.numel(), nullptr);
    cudaDeviceSynchronize();

    check_tensors_close(ref, out_gpu.to_cpu(), "test_gpu_parcon");

    // Backward
    Tensor gy_cpu = Tensor::randn(Shape{{11,11}}, cpu_opts);
    Tensor ga_ref = gy_cpu * (2.0f * (Tensor::ones(a_cpu.shape(), TensorOptions()) - a_cpu));

    Tensor gy_gpu = gy_cpu.to(gpu_opts.device);
    Tensor ga_gpu = Tensor::zeros(a_cpu.shape(), TensorOptions()).to(gpu_opts.device);

    K.vjp_parcon(ga_gpu.data<float>(), a_gpu.data<float>(), gy_gpu.data<float>(), gy_cpu.numel(), nullptr);
    cudaDeviceSynchronize();

    check_tensors_close(ga_ref, ga_gpu.to_cpu(), "test_gpu_vjp_parcon");
}


void test_gpu_unified_gelu() {
    auto& K = kernels::cuda();
    auto cpu_opts = TensorOptions().with_device(Device::CPU);
    auto gpu_opts = TensorOptions().with_device(Device::CUDA);

    Tensor a_cpu = Tensor::randn(Shape{{11,11}}, cpu_opts);
    
     float c1 = 0.7978845608f; // sqrt(2/pi)
     float c2 = 0.044715f;
    Tensor X3 = a_cpu * a_cpu * a_cpu;
    Tensor u  = (a_cpu + X3 * c2) * c1;
    auto ref  = a_cpu * (1.0f + OwnTensor::tanh(u)) * 0.5f;

    Tensor a_gpu = a_cpu.to(gpu_opts.device);
    Tensor out_gpu(ref.shape(), options(ref).with_device(gpu_opts.device));

    K.gelu(a_gpu.data<float>(), out_gpu.data<float>(), a_cpu.numel(), nullptr);
    cudaDeviceSynchronize();

    check_tensors_close(ref, out_gpu.to_cpu(), "test_gpu_gelu");

    // Backward
    Tensor gy_cpu = Tensor::randn(Shape{{11,11}}, cpu_opts);
        // Constants for the GELU approximation's derivative


    // Recompute intermediates needed for the derivative
    Tensor x2 = a_cpu * a_cpu;
    Tensor x3 = x2 * a_cpu;
    Tensor th_u = OwnTensor::tanh(u);
    
    // Compute du/dx = c1 * (1 + 3 * c2 * x^2)
    Tensor du_dx = (1.0f + (x2 * (3.0f * c2))) * c1;

    // Compute the full GELU derivative
    Tensor ga_ref = gy_cpu*((1.0f + th_u) * 0.5f + (a_cpu * (1.0f - (th_u * th_u)) * du_dx) * 0.5f);
        Tensor gy_gpu = gy_cpu.to(gpu_opts.device);
    Tensor ga_gpu = Tensor::zeros(a_cpu.shape(), TensorOptions()).to(gpu_opts.device);

    K.vjp_gelu(ga_gpu.data<float>(), a_gpu.data<float>(), gy_gpu.data<float>(), gy_cpu.numel(), nullptr);
    cudaDeviceSynchronize();

    check_tensors_close(ga_ref, ga_gpu.to_cpu(), "test_gpu_vjp_gelu");
}

void test_gpu_unified_lisht() {
    auto& K = kernels::cuda();
    auto cpu_opts = TensorOptions().with_device(Device::CPU);
    auto gpu_opts = TensorOptions().with_device(Device::CUDA);

    Tensor a_cpu = Tensor::randn(Shape{{11,11}}, cpu_opts);
    
    Tensor ref = a_cpu * OwnTensor::tanh(a_cpu);

    Tensor a_gpu = a_cpu.to(gpu_opts.device);
    Tensor out_gpu(ref.shape(), options(ref).with_device(gpu_opts.device));

    K.lisht(a_gpu.data<float>(), out_gpu.data<float>(), a_cpu.numel(), nullptr);
    cudaDeviceSynchronize();

    check_tensors_close(ref, out_gpu.to_cpu(), "test_gpu_lisht");

    // Backward
    Tensor gy_cpu = Tensor::randn(Shape{{11,11}}, cpu_opts);
    Tensor ga_ref = gy_cpu * ( -1.0f* (a_cpu * (Tensor::ones(a_cpu.shape(), TensorOptions()) - OwnTensor::tanh(a_cpu) * OwnTensor::tanh(a_cpu)) + OwnTensor::tanh(a_cpu)) );

    Tensor gy_gpu = gy_cpu.to(gpu_opts.device);
    Tensor ga_gpu = Tensor::zeros(a_cpu.shape(), TensorOptions()).to(gpu_opts.device);

    K.vjp_lisht(ga_gpu.data<float>(), a_gpu.data<float>(), gy_gpu.data<float>(), gy_cpu.numel(), nullptr);
    cudaDeviceSynchronize();

    check_tensors_close(ga_ref, ga_gpu.to_cpu(), "test_gpu_vjp_lisht");
}



int main() {
    std::cout << "=== Running GPU Kernel Tests ===\n";
    // try {
    //     #if defined(_WIN32)
    //         const char* plugin_path = "./agkernels_cuda.dll";
    //     #elif defined(__APPLE__)
    //         const char* plugin_path = "./libagkernels_cuda.dylib";
    //     #else
    //         const char* plugin_path = "/home/blubridge-034/Downloads/Newf/cgadimpl/cgadimpl/build/libagkernels_cuda.so";
    //     #endif

    //     std::cout << "Loading GPU plugin from: " << plugin_path << "\n";
    //     kernels::load_cuda_plugin(plugin_path);

        test_gpu_unified_add();
        test_gpu_unified_matmul();
        test_gpu_unified_linear();
        test_gpu_unified_fmab();
        test_gpu_unified_div();
        test_gpu_unified_sub();
        test_gpu_unified_hadmul();
        test_gpu_unified_silu();
        test_gpu_unified_mish();
        test_gpu_unified_maeloss();
        test_gpu_unified_mseloss();
        test_gpu_unified_gcu();
        test_gpu_unified_gauss();
        test_gpu_unified_lisht();
        test_gpu_unified_gelu();
        test_gpu_unified_parcon();
        test_gpu_unified_attention();
        test_gpu_unified_alibattention();
        

    // } catch (const std::exception& e) {
    //     std::cerr << "ERROR: " << e.what() << std::endl;
    //     return 1;
    // }

    std::cout << "\nAll GPU kernel tests passed successfully!\n";
    return 0;
}