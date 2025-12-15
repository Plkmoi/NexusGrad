#include <cmath>
#include <cassert>
#include "tensor.hpp" 
#include <ad/autodiff.hpp>
#include "layer/norm.hpp"

namespace ag::layer {



RMSNorm::RMSNorm(float gamma, Device dev) {
_gamma = gamma;
}

// Value RMSNorm::operator()(Value input) {   
//     return realrms(input, _gamma);
// }


LayerNorm::LayerNorm(float beta, float gamma, Device dev) {
_gamma = gamma;
_beta = beta;
}

// Value LayerNorm::operator()(Value input) {   
//     return relaynor(input,_beta,  _gamma);
// }

DynTanh::DynTanh(float alpha, float beta, float gamma, Device dev) {
_alpha = alpha;
_gamma = gamma;
_beta = beta;
}

// Value DynTanh::operator()(Value input) {   
//     return dyntanh(input,_alpha, _beta,  _gamma);
// }

} // namespace ag::nn