#include "nn/nn.hpp"
#include <cmath>
#include <cassert>
#include "tensor.hpp" 

namespace ag::nn {

void Module::to(Device dev) {
    for (Value& p : params_) {
        if (p.node) {
            p.node->value = p.node->value.to(dev);
            p.node->grad = OwnTensor::Tensor::zeros(p.node->value.shape(), ag::options(p.node->value));
        }
    }
}

void Module::zero_grad() {
    for (Value& p : params_) {
        if (p.node && p.node->requires_grad()) {
            p.node->grad = OwnTensor::Tensor::zeros(p.node->value.shape(), ag::options(p.node->value));
        }
    }
}

Linear::Linear(int in_features, int out_features, Device dev) {
    float scale = sqrtf(2.0f / in_features);
    auto param_opts = OwnTensor::TensorOptions().with_device(dev).with_req_grad(true);
    Tensor w_tensor = OwnTensor::Tensor::randn(Shape{{out_features, in_features}}, param_opts) * scale;
    Tensor b_tensor = OwnTensor::Tensor::zeros(Shape{{1, out_features}}, param_opts);
    W = make_tensor(w_tensor, "W");
    b = make_tensor(b_tensor, "b");
    params_.push_back(W);
    params_.push_back(b);
}


SWIGLU::SWIGLU(int in_features, int out_features, Device dev) {
    float scale = sqrtf(2.0f / in_features);
    auto param_opts = OwnTensor::TensorOptions().with_device(dev).with_req_grad(true);
    Tensor w_tensor = OwnTensor::Tensor::randn(Shape{{out_features, in_features}}, param_opts) * scale;
    Tensor b_tensor = OwnTensor::Tensor::zeros(Shape{{1, out_features}}, param_opts);
    Tensor wa_tensor = OwnTensor::Tensor::randn(Shape{{out_features, in_features}}, param_opts) * scale;
    Tensor ba_tensor = OwnTensor::Tensor::zeros(Shape{{1, out_features}}, param_opts);
    W = make_tensor(w_tensor, "W");
    b = make_tensor(b_tensor, "b");
    Wa = make_tensor(wa_tensor, "Wa");
    ba = make_tensor(ba_tensor, "ba");
    params_.push_back(W);
    params_.push_back(b);
    params_.push_back(Wa);
    params_.push_back(ba);
}

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

Value SWIGLU::operator()(Value input) {   
    return swiglu(input, W, b, Wa, ba);
}

Value Linear::operator()(Value input) {   
    return linear(input, W, b);
}


RMSNorm::RMSNorm(Device dev) {

}

Value RMSNorm::operator()(Value input) {   
    return realrms(input, gamma);
}

Softmax::Softmax(Device dev) {

}

Value Softmax::operator()(Value input) {   
    return softmax_row(input);
}

Sequential::Sequential(const std::vector<Module*>& modules)
    : layers_(modules)
{
    for (auto* mod : layers_) {
        for (auto& p : mod->parameters()) {
            params_.push_back(p);
        }
    }
}

Value Sequential::operator()(Value x) {
    for (auto* layer : layers_) {
        x = (*layer)(x);
    }
    return x;
}

Value ReLU::operator()(Value input) {
    return ag::relu(input);
}

} // namespace ag::nn