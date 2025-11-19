#include "layer/affine.hpp"
#include <cmath>
#include <cassert>
#include "tensor.hpp" 
#include <ad/autodiff.hpp>
#include "layer/attention.hpp"

namespace ag::layer {



Attention::Attention(int in_features, int out_features, Device dev) {
    float scale = sqrtf(0.02f / in_features);
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
    using namespace OwnTensor;

    // Expect input: [T, d_model_]
    Value X_heads = expand(input, n_heads_);   // NEW NODE

    // STEP 2: Attention op (already fine)
    float dummy_m = m_;

    Value Y_heads = alibiatt(X_heads, Wq_, Wk_, Wv_, dummy_m); // [H, T, D]

    // 5. Collapse heads [H, T, D] → [T, D] by averaging over head dim
    //
    //    This uses Tensor-only reduce_mean, so gradient does not flow back
    //    through the head-averaging itself (but still flows into Wq_/Wk_/Wv_).
    //
//     Tensor yh = Y_heads.val();                  // [H, T, D]
//     Tensor yh_cpu = yh.to(Device::CPU);
// Tensor y_cpu  = OwnTensor::reduce_mean(yh_cpu, {0}, false);
// Tensor y      = y_cpu.to(yh.device());    // -> [T, D]

auto y = rowmean(Y_heads, 0);

    return y;
}

} // namespace ag::nn