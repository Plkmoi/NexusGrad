#include <cmath>
#include <cassert>
#include "layer/archlist.hpp"

namespace flow {



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