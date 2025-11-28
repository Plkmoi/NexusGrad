// =====================
// file: include/ag/ag_all.hpp (umbrella)
// =====================
#pragma once
#include "ad/autodiff.hpp"
#include "ad/debug.hpp"   
#include "ad/cuda_graphs.hpp"
#include "nn/nn.hpp"
#include "ad/kernels_api.hpp"
#include <cuda_runtime.h>
#include "layer/affine.hpp"
#include "layer/activation.hpp"
#include "layer/attention.hpp"
#include "layer/norm.hpp"
#include "layer/optim.hpp"
#include "layer/loss.hpp"