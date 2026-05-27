#pragma once
#include "affine.hpp"

#include "layer/archlist.hpp"
#include <vector>

namespace flow {



class SWIGLU : public Layer {
public:
    SWIGLU(int batch, int in_features, int out_features, int hidden_features, Device dev = Device::CPU);
    Value operator()(Value input) override;
private:
    Value W, b, Wa, ba, Wc, bc;
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


class LeakyReLU : public Layer {
public:
    LeakyReLU(float alpha, Device dev = Device::CPU);
    Value operator()(Value input) override;

private:
    float _alpha = 1.0f;
};


class Swish : public Layer {
public:
    Value operator()(Value input) override;
};


class ParCon : public Layer {
public:
    Value operator()(Value input) override;
};

class Mish : public Layer {
public:
    Value operator()(Value input) override;
};

class Gauss : public Layer {
public:
    Value operator()(Value input) override;
};

class Softplus : public Layer {
public:
    Value operator()(Value input) override;
};

class GCU : public Layer {
public:
    Value operator()(Value input) override;
};

class LiSHT : public Layer {
public:
    Value operator()(Value input) override;
};

class Tanh : public Layer {
public:
    Value operator()(Value input) override;
};

class Sigmoid : public Layer {
public:
    Value operator()(Value input) override;
};

class GELU : public Layer {
public:
    Value operator()(Value input) override;
};



} // namespace ag::nn