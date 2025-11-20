#include "ad/ag_all.hpp"   // Main umbrella header for the framework
#include <iostream>
#include <vector>
#include <random>
#include <limits>
#include <cmath>

// -----------------------------
// Pure tensor ALiBi attention
// x:  [H, T, D]
// Wq: [D, D]
// Wk: [D, D]
// Wv: [D, D]
// returns y: [H, T, D]
// -----------------------------
Tensor alibi_attention_tensor(
    const Tensor& x,
    const Tensor& Wq,
    const Tensor& Wk,
    const Tensor& Wv
) {
    using namespace ag;
    using namespace OwnTensor;

    const auto& dims = x.shape().dims;
    if (dims.size() != 3) {
        throw std::runtime_error("alibi_attention_tensor: x must be [H,T,D]");
    }

    const int H = static_cast<int>(dims[0]);
    const int T = static_cast<int>(dims[1]);
    const int D = static_cast<int>(dims[2]);

    float scale = 1.f / std::sqrt(static_cast<float>(D));

    // Projections: [H, T, D]
    Tensor q = OwnTensor::matmul(x, Wq);  // [H,T,D]
    Tensor k = OwnTensor::matmul(x, Wk);  // [H,T,D]
    Tensor v = OwnTensor::matmul(x, Wv);  // [H,T,D]

    // Logits: [H, T, T]
    Tensor logits = OwnTensor::matmul(q, k.t()) * scale;

    // Build full ALiBi bias [H, T, T] on CPU, then move to logits.device()
    Tensor bias_cpu(
        logits.shape(),
        OwnTensor::TensorOptions()
            .with_device(OwnTensor::Device::CPU)
            .with_dtype(logits.dtype())
    );

    {
        float slope_start = 1.0f / std::pow(2.0f, 8.0f / H);

        dispatch_by_dtype(bias_cpu.dtype(), [&](auto dummy) {
            using Tval = decltype(dummy);
            Tval* data = bias_cpu.data<Tval>();

            for (int h = 0; h < H; ++h) {
                float slope = std::pow(slope_start, h + 1);

                for (int i = 0; i < T; ++i) {
                    for (int j = 0; j < T; ++j) {
                        float v;
                        if (j > i) {
                            // causal mask
                            v = -std::numeric_limits<float>::infinity();
                        } else {
                            // ALiBi penalty
                            v = -static_cast<float>(T - 1 - j) * slope;
                        }

                        const int idx = h * (T * T) + i * T + j;
                        data[idx] = static_cast<Tval>(v);
                    }
                }
            }
        });
    }

    Tensor bias = bias_cpu.to(logits.device());
    Tensor g    = logits + bias;  // [H,T,T]

    // Softmax over last dim
    Tensor max_g = OwnTensor::reduce_max(g, {-1}, true);   // [H,T,1]
    Tensor exp_g = OwnTensor::exp(g - max_g);              // [H,T,T]
    Tensor sum_g = OwnTensor::reduce_sum(exp_g, {-1}, true); // [H,T,1]
    Tensor s     = exp_g / sum_g;                          // [H,T,T]

    // Output: y = s @ v  -> [H,T,D]
    Tensor y = OwnTensor::matmul(s, v);
    return y;
}

// -----------------------------
// Tensor-only wrapper mirroring
// the tiling logic from your
// Value-based AlibiAttention::operator().
// x_2d: [T, D]
// Wq/Wk/Wv: [D, D]
// n_heads: H
// returns y: [T, D]
// -----------------------------
Tensor alibi_attention_from_2d_tiled(
    const Tensor& x_2d,
    const Tensor& Wq,
    const Tensor& Wk,
    const Tensor& Wv,
    int n_heads
) {
    using namespace ag;
    using namespace OwnTensor;

    const auto& dims = x_2d.shape().dims;
    if (dims.size() != 2) {
        throw std::runtime_error("alibi_attention_from_2d_tiled: x_2d must be [T,D]");
    }

    const int T = static_cast<int>(dims[0]);
    const int D = static_cast<int>(dims[1]);

    // 1. Copy input to CPU for safe pointer-based tiling
    Tensor x_host = x_2d.to(OwnTensor::Device::CPU);

    auto host_opts = TensorOptions()
                         .with_device(OwnTensor::Device::CPU)
                         .with_req_grad(x_2d.requires_grad());

    // 2. Allocate [n_heads, T, D] on CPU and tile along head dimension
    Tensor x_heads_host(Shape{{n_heads, T, D}}, host_opts);

    {
        float*       dst = x_heads_host.data<float>();   // host pointer
        const float* src = x_host.data<float>();         // host pointer
        const int per_seq = T * D;

        for (int h = 0; h < n_heads; ++h) {
            std::memcpy(dst + h * per_seq, src, per_seq * sizeof(float));
        }
    }

    // 3. Move [n_heads, T, D] back to original device of input
    Tensor x_heads = x_heads_host.to(x_2d.device());      // [H, T, D]

    // 4. Call the 3D ALiBi tensor function you already have
    Tensor y_heads = alibi_attention_tensor(x_heads, Wq, Wk, Wv); // [H,T,D]

    // 5. Collapse heads [H, T, D] -> [T, D] by averaging over head dimension
    Tensor y = OwnTensor::reduce_mean(y_heads, {0}, false); // -> [T, D]

    return y;
}

// ----------------------------------------
// Helper: check for NaN / Inf in a Tensor
// ----------------------------------------
bool has_nan_or_inf(const Tensor& t) {
    using namespace ag;
    using namespace OwnTensor;

    Tensor cpu = t.to(OwnTensor::Device::CPU);
    const auto& dims = cpu.shape().dims;

    std::size_t n = 1;
    for (auto d : dims) n *= static_cast<std::size_t>(d);

    const float* data = cpu.data<float>();
    for (std::size_t i = 0; i < n; ++i) {
        float v = data[i];
        if (std::isnan(v) || std::isinf(v)) {
            return true;
        }
    }
    return false;
}

// ----------------------------------------
// One test run for a given device
// ----------------------------------------
void run_test_on_device(Device dev, const char* name) {
    using namespace ag;
    using namespace OwnTensor;

    std::cout << "\n=== Running ALiBi tensor test on " << name << " ===\n";

    const int H = 4;   // heads
    const int T = 8;   // sequence length
    const int D = 16;  // head dim (also d_model here)

    auto opts = TensorOptions().with_device(dev).with_req_grad(false);

    // ---------- 3D direct test ----------
    Tensor x3  = Tensor::randn(Shape{{H, T, D}}, opts);
    Tensor Wq3 = Tensor::randn(Shape{{D, D}},   opts);
    Tensor Wk3 = Tensor::randn(Shape{{D, D}},   opts);
    Tensor Wv3 = Tensor::randn(Shape{{D, D}},   opts);

    Tensor y3;
    try {
        y3 = alibi_attention_tensor(x3, Wq3, Wk3, Wv3);
    } catch (const std::exception& e) {
        std::cerr << "Exception during ALiBi attention (3D) on " << name
                  << ": " << e.what() << "\n";
        return;
    }

    std::cout << "3D output shape: [";
    const auto& out_dims = y3.shape().dims;
    for (std::size_t i = 0; i < out_dims.size(); ++i) {
        std::cout << out_dims[i];
        if (i + 1 < out_dims.size()) std::cout << ", ";
    }
    std::cout << "]\n";

    bool bad3 = has_nan_or_inf(y3);
    if (bad3) {
        std::cout << "⚠️  3D output contains NaN or Inf.\n";
    } else {
        std::cout << "✅ 3D output is finite (no NaN/Inf).\n";
    }

    Tensor y3_cpu = y3.to(OwnTensor::Device::CPU);
    const float* p3 = y3_cpu.data<float>();
    std::cout << "First few values of y3[0,0,:]: ";
    for (int d = 0; d < std::min(D, 8); ++d) {
        std::cout << p3[d] << " ";
    }
    std::cout << "\n";

    // ---------- 2D → 3D tiling wrapper test ----------
    Tensor x2  = Tensor::randn(Shape{{T, D}}, opts);  // [T,D], like input.val()
    Tensor Wq2 = Tensor::randn(Shape{{D, D}}, opts);
    Tensor Wk2 = Tensor::randn(Shape{{D, D}}, opts);
    Tensor Wv2 = Tensor::randn(Shape{{D, D}}, opts);

    Tensor y2;
    try {
        y2 = alibi_attention_from_2d_tiled(x2, Wq2, Wk2, Wv2, /*n_heads=*/H);
    } catch (const std::exception& e) {
        std::cerr << "Exception during ALiBi 2D→3D wrapper on " << name
                  << ": " << e.what() << "\n";
        return;
    }

    std::cout << "2D-wrapper output shape: [";
    const auto& out2_dims = y2.shape().dims;
    for (std::size_t i = 0; i < out2_dims.size(); ++i) {
        std::cout << out2_dims[i];
        if (i + 1 < out2_dims.size()) std::cout << ", ";
    }
    std::cout << "]\n";

    bool bad2 = has_nan_or_inf(y2);
    if (bad2) {
        std::cout << "⚠️  2D-wrapper output contains NaN or Inf.\n";
    } else {
        std::cout << "✅ 2D-wrapper output is finite (no NaN/Inf).\n";
    }

    Tensor y2_cpu = y2.to(OwnTensor::Device::CPU);
    const float* p2 = y2_cpu.data<float>();
    std::cout << "First few values of y2[0,:]: ";
    for (int d = 0; d < std::min(D, 8); ++d) {
        std::cout << p2[d] << " ";
    }
    std::cout << "\n";
}

int main() {
    using namespace ag;

    std::cout << "========================================\n";
    std::cout << "--- ALiBi Tensor Attention Test Only ---\n";
    std::cout << "========================================\n";

    // CPU test
    run_test_on_device(Device::CPU, "CPU");

    // CUDA test (if you built with WITH_CUDA)
#ifdef WITH_CUDA
    run_test_on_device(Device::CUDA, "CUDA");
#else
    std::cout << "\n(Skipping CUDA test: built without WITH_CUDA)\n";
#endif

    std::cout << "\nDone.\n";
    return 0;
}
