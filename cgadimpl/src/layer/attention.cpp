#include <cmath>
#include <cassert>
#include "tensor.hpp" 
#include <ad/autodiff.hpp>
#include "layer/attention.hpp"

namespace ag::layer {



Attention::Attention(int batch, int in_features, int out_features, int H, Device dev) {
    float scale = sqrtf(0.02f / in_features);
    auto param_opts = OwnTensor::TensorOptions().with_device(dev).with_req_grad(true);
    Tensor wq = OwnTensor::Tensor::randn(Shape{{batch, in_features, out_features}}, param_opts) * scale;
    Tensor wk = OwnTensor::Tensor::randn(Shape{{batch, in_features, out_features}}, param_opts) * scale;
    Tensor wv = OwnTensor::Tensor::randn(Shape{{batch, in_features, out_features}}, param_opts) * scale;

    Q = make_tensor(wq, "q");
    K = make_tensor(wk, "k");
    V = make_tensor(wv, "v");
    heads = H;

    params_.push_back(Q);
    params_.push_back(K);
    params_.push_back(V);
}

Value Attention::operator()(Value input) {   
    return attention(input, Q, K, V, heads);
}



AlibiAttention::AlibiAttention(int d_model, int n_heads, float m, Device dev)
    : d_model_(d_model),
      n_heads_(n_heads),
      m_(m)
{
    auto param_opts = OwnTensor::TensorOptions().with_device(dev).with_req_grad(true);

    // Simple per-head-shared projection weights [d_model, d_model]
    Tensor wq = OwnTensor::Tensor::randn(Shape{{d_model, d_model}}, param_opts);
    Tensor wk = OwnTensor::Tensor::randn(Shape{{d_model, d_model}}, param_opts);
    Tensor wv = OwnTensor::Tensor::randn(Shape{{d_model, d_model}}, param_opts);
    Wq_ = make_tensor(wq, "Wq_alibi");
    Wk_ = make_tensor(wk, "Wk_alibi");
    Wv_ = make_tensor(wv, "Wv_alibi");

    params_.push_back(Wq_);
    params_.push_back(Wk_);
    params_.push_back(Wv_);

}


Value AlibiAttention::operator()(Value input) {
    using namespace OwnTensor;

    // Expect input: [T, d_model_]
    Tensor x_t = input.val();
    const auto& dims = x_t.shape().dims;
    if (dims.size() != 2 || static_cast<int>(dims[1]) != d_model_) {
        throw std::runtime_error("AlibiAttention::operator(): input must be [T, d_model_]");
    }

    const int T = static_cast<int>(dims[0]);
    const int D = d_model_;
    const int H = n_heads_;

    // 1. Copy input to CPU for safe tiling
    Tensor x_host = x_t.to(Device::CPU);

    auto host_opts = TensorOptions()
                         .with_device(Device::CPU)
                         .with_req_grad(x_t.requires_grad());

    // 2. Allocate [H, T, D] on CPU and tile along head dimension
    Tensor x_heads_host(Shape{{H, T, D}}, host_opts);

    {
        float*       dst = x_heads_host.data<float>();   // host pointer
        const float* src = x_host.data<float>();         // host pointer
        const int per_seq = T * D;

        for (int h = 0; h < H; ++h) {
            std::memcpy(dst + h * per_seq, src, per_seq * sizeof(float));
        }
    }

    // 3. Move [H, T, D] back to the original device of the input
    Tensor x_heads = x_heads_host.to(x_t.device());      // [H, T, D]

    // 4. Wrap as a Value and call the 3D ALiBi Node op
    //
    //    NOTE: This makes X_heads a *leaf* in the graph.
    //    Gradients will flow into Wq_/Wk_/Wv_ (which are also Values),
    //    but not back into the original `input` (same as your earlier design).
    //
    Value X_heads = make_tensor(x_heads, "X_alibi_heads");

    int dummy_m = m_;   // or just 1.0f if m_ isn't used inside

    Value Y_heads = alibiatt(X_heads, Wq_, Wk_, Wv_, dummy_m); // [H, T, D]

    // 5. Collapse heads [H, T, D] → [T, D] by averaging over head dim
    //
    //    This uses Tensor-only reduce_mean, so gradient does not flow back
    //    through the head-averaging itself (but still flows into Wq_/Wk_/Wv_).
    //
    Tensor yh = Y_heads.val();                  // [H, T, D]
    Tensor yh_cpu = yh.to(Device::CPU);
Tensor y_cpu  = OwnTensor::reduce_mean(yh_cpu, {0}, false);
Tensor y      = y_cpu.to(yh.device());    // -> [T, D]
auto m = make_tensor(y, "Y_alibi_out");



    return m;
}

} // namespace ag::nn