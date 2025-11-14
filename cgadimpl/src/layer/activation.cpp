#include "layer/affine.hpp"
#include <cmath>
#include <cassert>
#include "tensor.hpp" 
#include <ad/autodiff.hpp>
#include "layer/activation.hpp"

namespace ag::layer {



SWIGLU::SWIGLU(int in_features, int out_features, Device dev) {
    float scale = sqrtf(0.02f / in_features);
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





Value SWIGLU::operator()(Value input) {   
    return swiglu(input, W, b, Wa, ba);
}


Softmax::Softmax(Device dev) {

}

Value Softmax::operator()(Value input) {   
    return softmax_row(input);
}


Value ReLU::operator()(Value input) {
    return ag::relu(input);
}



} // namespace ag::nn