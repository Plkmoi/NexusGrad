#include "layer/affine.hpp"
#include <cmath>
#include <cassert>
#include "tensor.hpp" 
#include <ad/autodiff.hpp>
#include "layer/attention.hpp"

namespace ag::layer {



Attention::Attention(int in_features, int out_features, Device dev) {
    float scale = sqrtf(2.0f / in_features);
    auto param_opts = OwnTensor::TensorOptions().with_device(dev).with_req_grad(true);
    Tensor wq = OwnTensor::Tensor::randn(Shape{{in_features, out_features}}, param_opts) * scale;
    Tensor wk = OwnTensor::Tensor::randn(Shape{{in_features, out_features}}, param_opts) * scale;
    Tensor wv = OwnTensor::Tensor::randn(Shape{{in_features, out_features}}, param_opts) * scale;

    Q = make_tensor(wq, "q");
    K = make_tensor(wk, "k");
    V = make_tensor(wv, "v");

    params_.push_back(Q);
    params_.push_back(K);
    params_.push_back(V);
}

Value Attention::operator()(Value input) {   
    return attention(input, Q, K, V);
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
    // We expect 'input' to be [T, d_model] (in your toy: [B, d_model]).
    // We'll tile it over heads to get [n_heads, T, d_model] and feed that into alibiatt_nodeops.

    Tensor x_t = input.val();               // [T, d_model] on CPU or CUDA
    const auto& dims = x_t.shape().dims;
    // AG_ASSERT(dims.size() == 2 && dims[1] == d_model_);
    int T = dims[0];

    // 1. Copy input to CPU for safe pointer-based tiling
    Tensor x_host = x_t.to(OwnTensor::Device::CPU);

    auto host_opts = OwnTensor::TensorOptions()
                         .with_device(OwnTensor::Device::CPU)
                         .with_req_grad(x_t.requires_grad());

    // 2. Allocate [n_heads, T, d_model] on CPU and tile along head dimension
    Tensor x_heads_host(Shape{{n_heads_, T, d_model_}}, host_opts);

    {
        float*       dst = x_heads_host.data<float>();   // host pointer
        const float* src = x_host.data<float>();         // host pointer
        const int per_seq = T * d_model_;

        for (int h = 0; h < n_heads_; ++h) {
            std::memcpy(dst + h * per_seq, src, per_seq * sizeof(float));
        }
    }

    // 3. Move [n_heads, T, d_model] back to the original device (CPU or CUDA)
    Tensor x_heads = x_heads_host.to(x_t.device());      // [n_heads, T, d_model]

    // NOTE: this creates a *leaf* Value. Gradients will flow through alibiatt
    // into Wq_/Wk_/Wv_ but NOT back into 'input'. That’s fine for now if you
    // only care about updating the weights.
    Value X_heads = make_tensor(x_heads, "X_alibi_heads");

    // 4. Call the 3D alibi attention op. Inside, alibiatt_nodeops expects:
    //     a->value : [n_heads, T, d_model]
    //     Wq/Wk/Wv : [d_model, d_model]
    // and will produce logits [n_heads, T, T], apply ALiBi bias, softmax, and
    // project to y: [n_heads, T, d_model].
    Value Y_heads = alibiatt(X_heads, Wq_, Wk_, Wv_, m_);

    // 5. Collapse heads back to [T, d_model] by averaging over head dimension.
    // ---- Option A (if you have a Value-level reduce_mean op) ----
    // Value y = reduce_mean(Y_heads, {0}, /*keepdim=*/false);
    // return y;

    // ---- Option B (Tensor-only mean, breaks graph at this point) ----
    Tensor yh = Y_heads.val();                         // [n_heads, T, d_model]
    Tensor y  = OwnTensor::reduce_mean(yh, {0}, false); // -> [T, d_model]
    return make_tensor(y, "Y_alibi_out");
}

} // namespace ag::nn