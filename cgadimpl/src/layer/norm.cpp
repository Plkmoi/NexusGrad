#include "layer/affine.hpp"
#include <cmath>
#include <cassert>
#include "tensor.hpp" 
#include <ad/autodiff.hpp>

namespace ag::layer {



RMSNorm::RMSNorm(Device dev) {

}

Value RMSNorm::operator()(Value input) {   
    return realrms(input, gamma);
}




} // namespace ag::nn