// =====================
// file: include/ag/detail/cudaops.hpp
// =====================
#pragma once

#include "ad/graph.hpp"
#include "ad/checkpoint.hpp"
#include "ad/kernels_api.hpp"
#include "ad/debug.hpp"
#include <iostream>
#include <math.h>
#include <iterator>
#include <memory>
#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>


namespace ag {
namespace detail {

std::shared_ptr<Node> add_cudaops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b);
std::shared_ptr<Node> sub_cudaops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b);
std::shared_ptr<Node> mul_cudaops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b);
std::shared_ptr<Node> div_cudaops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b);
std::shared_ptr<Node> sigmoidiff_cudaops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> sigmoid_cudaops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> relu_cudaops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> log_cudaops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> tanh_cudaops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> gcu_cudaops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> mish_cudaops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> gaus_cudaops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> parcon_cudaops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> lisht_cudaops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> softplus_cudaops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> silu_cudaops(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> gelu_cudaops(const std::shared_ptr<Node>& x);


}
}