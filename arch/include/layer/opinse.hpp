#pragma once
#include "affine.hpp"

#include "layer/archlist.hpp"    // your CGAD class

#include <vector>

namespace flow {


const char* op_namea(Op o);

VjpFn vjp_lookupa(Op op);

void backwarda(const Value& root, const Tensor* grad_seed=nullptr);




} // namespace ag::nn