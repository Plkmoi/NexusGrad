#include "ad/ag_all.hpp"   // Main umbrella header for the framework
#include <iostream>
#include <vector>
#include <iomanip>
#include <random>
#include <optim.hpp>


#include "ad/ag_all.hpp"
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

//  const Tensor& x = a->value;
//     // AG_ASSERT(dims.size() == 3, "alibiatt_nodeops expects x with rank 3 [H, T, D]");


//     const Tensor& Wq = b->value;  // [D, D]
//     const Tensor& Wk = c->value;  // [D, D]
//     const Tensor& Wv = d->value;  // [D, D]

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

    Tensor x  = Tensor::randn(Shape{{H, T, D}}, opts);
    Tensor Wq = Tensor::randn(Shape{{D, D}},   opts);
    Tensor Wk = Tensor::randn(Shape{{D, D}},   opts);
    Tensor Wv = Tensor::randn(Shape{{D, D}},   opts);

    Tensor y;
    try {
        y = alibi_attention_tensor(x, Wq, Wk, Wv);
    } catch (const std::exception& e) {
        std::cerr << "Exception during ALiBi attention on " << name
                  << ": " << e.what() << "\n";
        return;
    }

    std::cout << "Output shape: [";
    const auto& out_dims = y.shape().dims;
    for (std::size_t i = 0; i < out_dims.size(); ++i) {
        std::cout << out_dims[i];
        if (i + 1 < out_dims.size()) std::cout << ", ";
    }
    std::cout << "]\n";

    bool bad = has_nan_or_inf(y);
    if (bad) {
        std::cout << "⚠️  Output contains NaN or Inf.\n";
    } else {
        std::cout << "✅ Output is finite (no NaN/Inf).\n";
    }

    // print a tiny slice so you can eyeball it
    Tensor y_cpu = y.to(OwnTensor::Device::CPU);
    const float* p = y_cpu.data<float>();
    std::cout << "First few values of y[0,0,:]: ";
    for (int d = 0; d < std::min(D, 8); ++d) {
        std::cout << p[d] << " ";
    }
    std::cout << "\n";
}


bool compare_tensors(const Tensor& A, const Tensor& B, const char* name) {
    Tensor A_cpu = A.to(Device::CPU);
    Tensor B_cpu = B.to(Device::CPU);

    const auto& dims = A_cpu.shape().dims;
    size_t N = 1;
    for (auto d : dims) N *= d;

    const float* pa = A_cpu.data<float>();
    const float* pb = B_cpu.data<float>();

    for (size_t i = 0; i < N; ++i) {
        float a = pa[i], b = pb[i];
        if (std::fabs(a - b) > 1e-3f && !(std::isnan(a) && std::isnan(b))) {
            std::cout << "❌ MISMATCH in " << name << " at index " << i
                      << ": CPU=" << a << ", CUDA=" << b << "\n";
                      A_cpu.display();
                      B_cpu.display();
            return false;
        }
    }

    std::cout << "✅ " << name << " matches.\n";
    return true;
}

int main() {
    const int H = 4, T = 8, D = 16;

    auto cpu_opts  = TensorOptions().with_device(Device::CPU);
    auto cuda_opts = TensorOptions().with_device(Device::CUDA);

    // Random inputs
    Tensor x_cpu  = Tensor::randn(Shape{{H,T,D}}, cpu_opts);
    Tensor Wq_cpu = Tensor::randn(Shape{{D,D}},   cpu_opts);
    Tensor Wk_cpu = Tensor::randn(Shape{{D,D}},   cpu_opts);
    Tensor Wv_cpu = Tensor::randn(Shape{{D,D}},   cpu_opts);

    Tensor x_gpu  = x_cpu.to(Device::CUDA);
    Tensor Wq_gpu = Wq_cpu.to(Device::CUDA);
    Tensor Wk_gpu = Wk_cpu.to(Device::CUDA);
    Tensor Wv_gpu = Wv_cpu.to(Device::CUDA);

    // 1. Projections
    Tensor q_cpu = matmul(x_cpu, Wq_cpu);
    Tensor q_gpu = matmul(x_gpu, Wq_gpu);
    compare_tensors(q_cpu, q_gpu, "q");

    Tensor k_cpu = matmul(x_cpu, Wk_cpu);
    Tensor k_gpu = matmul(x_gpu, Wk_gpu);
    compare_tensors(k_cpu, k_gpu, "k");

    Tensor v_cpu = matmul(x_cpu, Wv_cpu);
    Tensor v_gpu = matmul(x_gpu, Wv_gpu);
    compare_tensors(v_cpu, v_gpu, "v");

    // 2. Logits
    Tensor logits_cpu = matmul(q_cpu, k_cpu.t());
    Tensor logits_gpu = matmul(q_gpu, k_gpu.t());
    compare_tensors(logits_cpu, logits_gpu, "logits");

    // 3. Bias construction (CPU only, then copy)
    Tensor bias_cpu(
        logits_cpu.shape(),
        TensorOptions().with_device(Device::CPU)
    );

    int Hh = H;
    float slope_start = 1.0f / std::pow(2.0f, 8.0f / Hh);
    float* data = bias_cpu.data<float>();

    for (int h = 0; h < Hh; ++h) {
        float slope = std::pow(slope_start, h + 1);
        for (int i = 0; i < T; ++i) {
            for (int j = 0; j < T; ++j) {
                float v = (j > i)
                    ? -INFINITY
                    : -(float)(T - 1 - j) * slope;
                data[h*T*T + i*T + j] = v;
            }
        }
    }

    Tensor bias_gpu = bias_cpu.to(Device::CUDA);

    // 4. g = logits + bias
    Tensor g_cpu = logits_cpu + bias_cpu;
    Tensor g_gpu = logits_gpu + bias_gpu;
    compare_tensors(g_cpu, g_gpu, "g");

    g_cpu.display();
    g_gpu.to_cpu().display();

    // 5. Softmax steps
    Tensor max_cpu = reduce_max(g_cpu, {1}, true);
    Tensor max_gpu = reduce_max(g_gpu, {1}, true);
    compare_tensors(max_cpu, max_gpu, "max_g");

    Tensor exp_cpu = exp(g_cpu - max_cpu);
    Tensor exp_gpu = exp(g_gpu - max_gpu);
    compare_tensors(exp_cpu, exp_gpu, "exp(g-max)");

    Tensor sum_cpu = reduce_sum(exp_cpu, {-1}, true);
    Tensor sum_gpu = reduce_sum(exp_gpu, {-1}, true);
    compare_tensors(sum_cpu, sum_gpu, "sum_g");

    Tensor s_cpu = exp_cpu / sum_cpu;
    Tensor s_gpu = exp_gpu / sum_gpu;
    compare_tensors(s_cpu, s_gpu, "softmax scores s");

    // 6. final y = s @ v
    Tensor y_cpu = matmul(s_cpu, v_cpu);
    Tensor y_gpu = matmul(s_gpu, v_gpu);
    compare_tensors(y_cpu, y_gpu, "y");

    std::cout << "\n=== DONE ===\n";
    return 0;
}
