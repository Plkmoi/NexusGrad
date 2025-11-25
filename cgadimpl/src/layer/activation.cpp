#include <cmath>
#include <cassert>
#include "tensor.hpp" 
#include <ad/autodiff.hpp>
#include "layer/activation.hpp"

namespace ag::layer {



SWIGLU::SWIGLU(int batch, int in_features, int out_features, Device dev) {
    float scale = sqrtf(0.02f / in_features);
    auto param_opts = OwnTensor::TensorOptions().with_device(dev).with_req_grad(true);
    Tensor w_tensor = OwnTensor::Tensor::randn(Shape{{batch, in_features, out_features}}, TensorOptions()) * scale;
    Tensor b_tensor = OwnTensor::Tensor::zeros(Shape{{batch, 1, in_features}}, TensorOptions());
    Tensor wa_tensor = OwnTensor::Tensor::randn(Shape{{batch, in_features, out_features}}, TensorOptions()) * scale;
    Tensor ba_tensor = OwnTensor::Tensor::zeros(Shape{{batch, 1, in_features}}, TensorOptions());
    W = make_tensor(w_tensor, "W");
    b = make_tensor(b_tensor, "b");
    Wa = make_tensor(wa_tensor, "Wa");
    ba = make_tensor(ba_tensor, "ba");
    params_.push_back(W);
    params_.push_back(b);
    params_.push_back(Wa);
    params_.push_back(ba);
}




Value SWIGLU::operator()(Value input) {   
    return swiglu(input, W, b, Wa, ba);
}


LeakyReLU::LeakyReLU(float alpha, Device dev)
{
_alpha = alpha;

}

Value LeakyReLU::operator()(Value input) {
    return ag::leaky_relu(input, _alpha);
}


Softmax::Softmax(Device dev) {

}

Value Softmax::operator()(Value input) {   
    return softmax_row(input);
}


Value ReLU::operator()(Value input) {
    return ag::relu(input);
}

Value Swish::operator()(Value input) {
    return ag::silu(input);
}

Value GELU::operator()(Value input) {
    return ag::gelu(input);
}

Value ParCon::operator()(Value input) {
    return ag::parcon(input);
}

Value Gauss::operator()(Value input) {
    return ag::gaus(input);
}

Value Mish::operator()(Value input) {
    return ag::mish(input);
}

Value Tanh::operator()(Value input) {
    return ag::tanh(input);
}

Value GCU::operator()(Value input) {
    return ag::gcu(input);
}

Value Softplus::operator()(Value input) {
    return ag::softplus(input);
}

Value LiSHT::operator()(Value input) {
    return ag::softplus(input);
}

Value Sigmoid::operator()(Value input) {
    return ag::sigmoid(input);
}

} // namespace ag::nn