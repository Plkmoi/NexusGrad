// =====================
// file: src/cudaops.cpp
// =====================
#include "ad/cudaops.hpp"
#include "ad/cudarray.hpp"
#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>


namespace ag {
namespace detail {

float* add_supply(float* a, float* b, float misiz)
{



    auto* fn = run_cuda_add;
    if (!fn)
        throw std::runtime_error("No CUDA Add kernel registered");



    fn(a, b, n->d_array, misiz);

    // cudaMemcpy(n->value.data(), n->d_array, M * N * sizeof(float),
    //            cudaMemcpyDeviceToHost);

    // // Debug print
    // std::cout << "[CUDA ADD output preview]: ";
    // for (int i = 0; i < std::min(10, M * N); ++i)
    //     std::cout << n->value.data()[i] << " ";
    // std::cout << "\n";

    n->inputs = {a, b};
    return n;




}









}
}