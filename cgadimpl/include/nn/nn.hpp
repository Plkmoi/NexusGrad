#pragma once

#include "ad/graph.hpp"
#include "ad/ops.hpp"
#include <vector>

namespace ag::nn {

class Module {
public:
    virtual ~Module() = default;
    virtual Value operator()(Value input) = 0; // Takes by value
    // virtual Value operator()(Value input, Value inputb) {
    //     (void)inputb;      // silence unused warning
    //     return (*this)(input);
    // }


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

class Linear : public Module {
public:
    Linear(int in_features, int out_features, Device dev = Device::CPU);
    Value operator()(Value input) override;
private:
    Value W, b;
};

class Attention : public Module {
public:
    Attention(int in_features, int out_features, Device dev = Device::CPU);
    Value operator()(Value input) override;
private:
    Value X, Q, K, V;
};

class SWIGLU : public Module {
public:
    SWIGLU(int in_features, int out_features, Device dev = Device::CPU);
    Value operator()(Value input) override;
private:
    Value W, b, Wa, ba;
};

class RMSNorm : public Module {
public:
    RMSNorm(Device dev = Device::CPU);
    Value operator()(Value input) override;
private:
    float gamma = 1.0;
};

class Softmax : public Module {
public:
    Softmax(Device dev = Device::CPU);
    Value operator()(Value input) override;
private:
    float gamma = 1.0;
};

class ResidualBlock : public Module {
public:
    explicit ResidualBlock(const std::vector<Module*>& modules)
        : modules_(modules)
    {
        // Collect parameters from inner modules
        for (auto* mod : modules_) {
            for (auto& p : mod->parameters()) {
                params_.push_back(p);
            }
        }
    }

    // x -> modules... -> y, then return x + y
    Value operator()(Value x) override {
        Value skip = x;
        for (auto* layer : modules_) {
            x = (*layer)(x);
        }
        return x + skip;
    }

private:
    std::vector<Module*> modules_;
};

class Sequential : public Module {
public:
    Sequential(const std::vector<Module*>& modules);
    Value operator()(Value x) override;
    const std::vector<Module*>& get_layers() const { return layers_; }
private:
    std::vector<Module*> layers_;
};

class ReLU : public Module {
public:
    Value operator()(Value input) override;
};

} // namespace ag::nn