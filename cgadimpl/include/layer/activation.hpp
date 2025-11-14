#pragma once
#include "layer/affine.hpp"

#include "ad/graph.hpp"
#include "ad/ops.hpp"
#include <vector>

namespace ag::layer {



class SWIGLU : public Layer {
public:
    SWIGLU(int in_features, int out_features, Device dev = Device::CPU);
    Value operator()(Value input) override;
private:
    Value W, b, Wa, ba;
};



class Softmax : public Layer {
public:
    Softmax(Device dev = Device::CPU);
    Value operator()(Value input) override;
private:
    float gamma = 1.0;
};





class ReLU : public Layer {
public:
    Value operator()(Value input) override;
};






} // namespace ag::nn