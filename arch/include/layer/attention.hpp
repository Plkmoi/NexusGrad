#pragma once
#include "affine.hpp"
#include "layer/archlist.hpp"    // your CGAD class

#include <vector>

namespace flow {



class Attention : public Layer {
public:
    Attention(int batch, int in_features, int H, int out_features, Device dev = Device::CPU);
    Value operator()(Value input) override;
private:
    Value X, Q, K, V;
    int heads;
};








class AlibiAttention : public Layer {
public:
    AlibiAttention(int batch, int in_features, int out_features, int H, Device dev = Device::CPU);

    Value operator()(Value input) override;

private:
    int heads;
    Value Wq_, Wk_, Wv_;
};


} // namespace ag::nn