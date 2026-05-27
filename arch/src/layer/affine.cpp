#include <cmath>
#include <cassert>
#include "layer/archlist.hpp"

namespace flow {

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

void Layer::forward(Value inp) {


        if(inp.node->op== Op::Leaf)
    {
        params_[0] = shallow(inp);


    }


for (Value& p : params_) {



        if (p.node && p.node->requires_grad()) {
        if (p.node->op == Op::Leaf) continue;  // already has a value

        auto fn = fwd_lookup(p.node->op);  // you can reuse your op forward registry
        // auto r = n->shared_from_this();
        if (fn) fn(p.node);        }
    }
}

void Layer::backward(Value root) {

    if(root.node->op!= Op::Leaf)
    {
        ag::backward(root);
        return;
    }



for (auto p = params_.rbegin(); p != params_.rend(); ++p) {
        if ((*p).node && (*p).node->requires_grad()) {
        // if ((*p).node->op == Op::Leaf) continue;  // already has a value

                const Tensor& gy = (*p).node->grad;


        auto fn = vjp_lookup((*p).node->op);  // you can reuse your op forward registry
        // auto r = n->shared_from_this();
        if (fn) fn((*p).node.get(), gy);        }
    }
}


void Layer::save(const std::string& path) {
    std::map<std::string, OwnTensor::Tensor> weight_map;
    
    // 1. Collect all valid tensors from your params
    for (size_t i = 0; i < params_.size(); ++i) {
        Value& p = params_[i];
        if (p.node) {
            // Use a unique name for each param
            // If p doesn't have a name string, use the index
            std::string name = "param_" + std::to_string(i); 
            
            // Make sure we bring it to CPU before saving to disk
            weight_map[name] = p.node->value.to(Device::CPU);
        }
    }

    // 2. Call the SafeTensors writer (as discussed)
    flow::save_safetensors(path, weight_map);
    
    std::cout << "[Layer] Saved " << weight_map.size() << " tensors to " << path << "\n";
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



Value Linear::operator()(Value input) {   
    return linear(input, W, b);
}

EmbedLinear::EmbedLinear(int batch, int in_features, int out_features, Tensor Weight, Device dev) {
    float scale = sqrtf(0.02f / in_features);
    auto param_opts = OwnTensor::TensorOptions().with_device(dev).with_req_grad(true);
    Tensor w_tensor = Weight.to(dev);
    Tensor b_tensor = OwnTensor::Tensor::zeros(Shape{{1, in_features}}, param_opts);
    W = Value(std::make_shared<Node>(w_tensor, Op::Leaf, true, "W"));
    b = make_tensor(b_tensor, "b");
    params_.push_back(W);
    params_.push_back(b);
}



Value EmbedLinear::operator()(Value input) {   
    return linear(input, W, b);
}





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
    // inflag = m;
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