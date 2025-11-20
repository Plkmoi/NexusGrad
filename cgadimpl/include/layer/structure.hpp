#pragma once
#include "layer/affine.hpp" 
#include "ad/graph.hpp"
#include "ad/ops.hpp"
#include <vector>

namespace ag::layer {

class Layer {
public:
    virtual ~Layer() = default;
    virtual Value operator()(Value input) = 0; 

    Value operator()(const Tensor& input) {
        Value graph_input = ag::make_tensor(input);
        return this->operator()(graph_input);
    }

    std::vector<Value>& parameters() { return params_; }

    void to(Device dev);
    void zero_grad();

protected:
    std::vector<Value> params_;
};


class Linear : public Layer {
public:
    Linear(int in_features, int out_features, Device dev = Device::CPU);
    Value operator()(Value input) override;
private:
    Value W, b;
};

class ResidualBlock : public Layer {
public:
    ResidualBlock(const std::vector<Layer*>& Layers);
    Value operator()(Value x) override;

private:
    std::vector<Layer*> Layers_;
};


class Traverse : public Layer {
public:
    Traverse(const std::vector<Layer*>& Layers);
    Value operator()(Value x) override;
    const std::vector<Layer*>& get_layers() const { return layers_; }
private:
    std::vector<Layer*> layers_;
};


} // namespace ag::nn