#pragma once
#include "layer/affine.hpp"

#include "ad/graph.hpp"
#include "ad/ops.hpp"
#include <vector>

namespace ag::layer {



class Attention : public Layer {
public:
    Attention(int in_features, int out_features, Device dev = Device::CPU);
    Value operator()(Value input) override;
private:
    Value X, Q, K, V;
};








class AlibiAttention : public Layer {
public:
    AlibiAttention(int d_model, int n_heads, float m, Device dev = Device::CPU);

    Value operator()(Value input) override;

private:
    int d_model_;
    int n_heads_;
    float m_;
    Value Wq_, Wk_, Wv_;
};


} // namespace ag::nn