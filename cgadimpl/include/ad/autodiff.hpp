// =====================
// file: include/ag/autodiff.hpp (declarations only)
// =====================
#pragma once
#include <unordered_map>
#include "ad/ops.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

namespace ag {


void zero_grad(const Value& root);
void backward (const Value& root, const Tensor* grad_seed=nullptr);
void valsend(const Value& root);
void grasend(const Value& root);
void unisend(const Value& root);
Tensor jvp (const Value& root, const std::unordered_map<Node*, Tensor>& seed);


} // namespace ag