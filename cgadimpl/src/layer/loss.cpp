#include <cmath>
#include <cassert>
#include "tensor.hpp" 
#include <ad/autodiff.hpp>
#include "layer/loss.hpp"

namespace ag::layer {



CEWithLogits::CEWithLogits() {
}

Value CEWithLogits::operator()(Value pred, Value target) {   
    return cross_entropy_with_logits(pred, target);
}


KLDivergence::KLDivergence() {
}

Value KLDivergence::operator()(Value pred, Value target) {   
    return kldivergence(pred, target);
}

MSELoss::MSELoss() {
}

Value MSELoss::operator()(Value pred, Value target) {   
    return mse_loss(pred, target);
}

MAELoss::MAELoss() {
}

Value MAELoss::operator()(Value pred, Value target) {   
    return mae_loss(pred, target);
}

} // namespace ag::nn