#include <cmath>
#include <cassert>
#include "tensor.hpp" 
#include <ad/autodiff.hpp>
#include "layer/attention.hpp"

namespace ag::layer {



Attention::Attention(int batch, int in_features, int out_features, int H, Device dev) {
    float scale = sqrtf(20.f / out_features);
    auto param_opts = OwnTensor::TensorOptions().with_device(dev).with_req_grad(true);
    Tensor wq = OwnTensor::Tensor::randn(Shape{{out_features, out_features}}, param_opts) * scale;
    Tensor wk = OwnTensor::Tensor::randn(Shape{{out_features, out_features}}, param_opts) * scale;
    Tensor wv = OwnTensor::Tensor::randn(Shape{{out_features, out_features}}, param_opts) * scale;

    Q = make_tensor(wq, "q");
    K = make_tensor(wk, "k");
    V = make_tensor(wv, "v");
    heads = H;

    params_.push_back(Q);
    params_.push_back(K);
    params_.push_back(V);
}

// Value Attention::operator()(Value input) {   
//     return attention(input, Q, K, V, heads);
// }



AlibiAttention::AlibiAttention(int batch, int in_features, int out_features, int H, Device dev)

{
    float scale = sqrtf(0.002f / out_features);
    
    auto param_opts = OwnTensor::TensorOptions().with_device(dev).with_req_grad(true);

    // Simple per-head-shared projection weights [d_model, d_model]
    Tensor wq = OwnTensor::Tensor::randn(Shape{{out_features, out_features}}, param_opts) * scale;
    Tensor wk = OwnTensor::Tensor::randn(Shape{{out_features, out_features}}, param_opts) * scale;
    Tensor wv = OwnTensor::Tensor::randn(Shape{{out_features, out_features}}, param_opts) * scale;
    Wq_ = make_tensor(wq, "Wq_alibi");
    Wk_ = make_tensor(wk, "Wk_alibi");
    Wv_ = make_tensor(wv, "Wv_alibi");
    heads = H;

    params_.push_back(Wq_);
    params_.push_back(Wk_);
    params_.push_back(Wv_);
    

}


// Value AlibiAttention::operator()(Value input) {
//     return alibiatt(input, Wq_, Wk_, Wv_, heads);
// }

} // namespace ag::nn