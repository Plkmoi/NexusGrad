#include "layer/affine.hpp"
#include <cmath>
#include <cassert>
#include "tensor.hpp" 
#include <ad/autodiff.hpp>
#include "layer/optim.hpp"

namespace ag::layer {

void Layer::to(Device dev) {
    for (Value& p : params_) {
        if (p.node) {
            p.node->value = p.node->value.to(dev);
            p.node->grad = OwnTensor::Tensor::zeros(p.node->value.shape(), ag::options(p.node->value));
        }
    }
}

void Layer::zero_grad() {
    for (Value& p : params_) {
        if (p.node && p.node->requires_grad()) {
            p.node->grad = OwnTensor::Tensor::zeros(p.node->value.shape(), ag::options(p.node->value));
        }
    }
}


Linear::Linear(int batch, int in_features, int out_features, Device dev) {
    float scale = sqrtf(0.02f / in_features);
    auto param_opts = OwnTensor::TensorOptions().with_device(dev).with_req_grad(true);
    Tensor w_tensor = OwnTensor::Tensor::randn(Shape{{in_features, out_features}}, param_opts);
    Tensor b_tensor = OwnTensor::Tensor::zeros(Shape{{1, in_features}}, param_opts);
    W = make_tensor(w_tensor, "W");
    b = make_tensor(b_tensor, "b");
    params_.push_back(W);
    params_.push_back(b);
}



// Value Linear::operator()(Value input) {   
//     return linear(input, W, b);
// }





Traverse::Traverse(const std::vector<Layer*>& Layers)
    : layers_(Layers)
{
    for (auto* mod : layers_) {
        for (auto& p : mod->parameters()) {
            params_.push_back(p);
        }
    }
}


Value Traverse::operator()(Value x) {
    for (auto* layer : layers_) {
        x = (*layer)(x);
    }
    return x;
}


    ResidualBlock::ResidualBlock(const std::vector<Layer*>& Layers)
        : Layers_(Layers)
    {
        // Collect parameters from inner Layers
        for (auto* mod : Layers_) {
            for (auto& p : mod->parameters()) {
                params_.push_back(p);
            }
        }
    }

    // x -> Layers... -> y, then return x + y
    Value ResidualBlock::operator()(Value x) {
        Value skip = x;
        for (auto* layer : Layers_) {
            x = (*layer)(x);
        }
        return x + skip;
    }




} // namespace ag::nn