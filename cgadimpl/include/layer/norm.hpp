#pragma once
#include "layer/affine.hpp"

#include "ad/graph.hpp"
#include "ad/ops.hpp"
#include <vector>

namespace ag::layer {


class RMSNorm : public Layer {
public:
    RMSNorm(float gamma = 0.1f, Device dev = Device::CPU);
    Value operator()(Value input) override;

private:
    float _gamma = 1.0;
};

class LayerNorm : public Layer {
public:
    LayerNorm(float beta = 0.0f, float gamma = 1.0f, Device dev = Device::CPU);
    Value operator()(Value input) override;

private:
    float _gamma = 1.0;
    float _beta = 0.0;
};

class DynTanh : public Layer {
public:
    DynTanh(float alpha = 0.5f, float beta = 0.0f, float gamma = 1.0f, Device dev = Device::CPU);
    Value operator()(Value input) override;

private:
    float _gamma = 1.0;
    float _beta = 0.0;
    float _alpha = 0.5;
};


} // namespace ag::nn