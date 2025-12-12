#pragma once
#include "affine.hpp"

#include "layer/archlist.hpp"    // your CGAD class

#include <vector>

namespace flow {


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