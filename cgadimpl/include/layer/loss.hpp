#pragma once
#include "layer/affine.hpp"

#include "ad/graph.hpp"
#include "ad/ops.hpp"
#include <vector>

namespace ag::layer {


class CEWithLogits : public Layer {
public:
    CEWithLogits();
    Value operator()(Value pred, Value target);

private:
};

class KLDivergence : public Layer {
public:
    KLDivergence();
    Value operator()(Value pred, Value target);

private:

};

class MSELoss : public Layer {
public:
    MSELoss();
    Value operator()(Value pred, Value target);

private:

};

class MAELoss : public Layer {
public:
    MAELoss();
    Value operator()(Value pred, Value target);

private:

};


} // namespace ag::nn