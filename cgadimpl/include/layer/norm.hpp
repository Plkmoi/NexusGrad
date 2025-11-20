#pragma once
#include "layer/affine.hpp" 
#include "ad/graph.hpp"
#include "ad/ops.hpp"
#include <vector>

namespace ag::layer {



class RMSNorm : public Layer {
public:
    RMSNorm(Device dev = Device::CPU);
    Value operator()(Value input) override;
private:
    float gamma = 1.0;
};

} // namespace ag::nn