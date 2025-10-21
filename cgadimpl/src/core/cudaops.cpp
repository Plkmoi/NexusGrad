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
// =====================================================
// ADD
// =====================================================
std::shared_ptr<Node> add_cudaops(const std::shared_ptr<Node>& a,
                                  const std::shared_ptr<Node>& b)
{
    auto A = a->d_array;
    auto B = b->d_array;

    auto [M, K]  = a->value.shape();
    auto [K2, N] = b->value.shape();

    auto* fn = run_cuda_add;
    if (!fn)
        throw std::runtime_error("No CUDA Add kernel registered");

    Tensor C({M, N});
    auto n = std::make_shared<Node>(C, (a->requires_grad || b->requires_grad),
                                    Op::Add, "+", true);

    fn(A, B, n->d_array, M * K);
    n->siz = M*K;

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


// =====================================================
// SUB
// =====================================================
std::shared_ptr<Node> sub_cudaops(const std::shared_ptr<Node>& a,
                                  const std::shared_ptr<Node>& b)
{
    auto A = a->d_array;
    auto B = b->d_array;

    auto [M, K]  = a->value.shape();
    auto [K2, N] = b->value.shape();

    auto* fn = run_cuda_sub;
    if (!fn)
        throw std::runtime_error("No CUDA Sub kernel registered");

    Tensor C({M, N});
    auto n = std::make_shared<Node>(C, (a->requires_grad || b->requires_grad),
                                    Op::Sub, "-", true);

    fn(A, B, n->d_array, M * K);
       n->siz = M*K;


    // cudaMemcpy(n->value.data(), n->d_array, M * N * sizeof(float),
    //            cudaMemcpyDeviceToHost);

    // std::cout << "[CUDA SUB output preview]: ";
    // for (int i = 0; i < std::min(10, M * N); ++i)
    //     std::cout << n->value.data()[i] << " ";
    // std::cout << "\n";

    n->inputs = {a, b};
    return n;
}


// =====================================================
// MUL (Hadamard)
// =====================================================
std::shared_ptr<Node> mul_cudaops(const std::shared_ptr<Node>& a,
                                     const std::shared_ptr<Node>& b)
{
    auto A = a->d_array;
    auto B = b->d_array;

    auto [M, K]  = a->value.shape();
    auto [K2, N] = b->value.shape();

    auto* fn = run_cuda_hadmul;
    if (!fn)
        throw std::runtime_error("No CUDA Hadamard kernel registered");

    Tensor C({M, N});
    auto n = std::make_shared<Node>(C, (a->requires_grad || b->requires_grad),
                                    Op::Mul, "⨀", true);

    fn(A, B, n->d_array, M * K);
       n->siz = M*K;

    cudaMemcpy(n->value.data(), n->d_array, M * N * sizeof(float),
               cudaMemcpyDeviceToHost);

    // std::cout << "[CUDA MUL output preview]: ";
    // for (int i = 0; i < std::min(10, M * N); ++i)
    //     std::cout << n->value.data()[i] << " ";
    // std::cout << "\n";

    n->inputs = {a, b};
    return n;
}


// =====================================================
// DIV
// =====================================================
std::shared_ptr<Node> div_cudaops(const std::shared_ptr<Node>& a,
                                  const std::shared_ptr<Node>& b)
{
    auto A = a->d_array;
    auto B = b->d_array;

    auto [M, K]  = a->value.shape();
    auto [K2, N] = b->value.shape();

    auto* fn = run_cuda_div;
    if (!fn)
        throw std::runtime_error("No CUDA Div kernel registered");

    Tensor C({M, N});
    auto n = std::make_shared<Node>(C, (a->requires_grad || b->requires_grad),
                                    Op::Div, "/", true);

    fn(A, B, n->d_array, M * K);
       n->siz = M*K;

    cudaMemcpy(n->value.data(), n->d_array, M * N * sizeof(float),
               cudaMemcpyDeviceToHost);

    std::cout << "[CUDA DIV output preview]: ";
    for (int i = 0; i < std::min(10, M * N); ++i)
        std::cout << n->value.data()[i] << " ";
    std::cout << "\n";

    n->inputs = {a, b};
    return n;
}






   
    std::shared_ptr<Node> sigmoid_cudaops(const std::shared_ptr<Node>& x){ 
                      const Tensor& xin = x->value;
        auto X = x->d_array;

    auto [M, K]  = x->value.shape();

    auto* fn = run_cuda_sigmoid;
    if (!fn)
        throw std::runtime_error("No CUDA Div kernel registered");

    Tensor C({M, K});
    auto n = std::make_shared<Node>(C, (x->requires_grad),
                                    Op::Sigmoid, "/", true);

    fn(X, n->d_array, M * K);

    cudaMemcpy(n->value.data(), n->d_array, M * K * sizeof(float),
               cudaMemcpyDeviceToHost);
       n->siz = M*K;

    std::cout << "[CUDA Sigmoid output preview]: ";
    for (int i = 0; i < std::min(10, M * K); ++i)
        std::cout << n->value.data()[i] << " ";
    std::cout << "\n";

    n->inputs = {x};
    return n;
    }





















        std::shared_ptr<Node> sigmoidiff_cudaops(const std::shared_ptr<Node>& x){ 
                      const Tensor& xin = x->value;
        auto X = x->d_array;

    auto [M, K]  = x->value.shape();

    auto* fn = run_cuda_sigmoidiff;
    if (!fn)
        throw std::runtime_error("No CUDA Sigmoidiff kernel registered");

    Tensor C({M, K});
    auto n = std::make_shared<Node>(C, (x->requires_grad),
                                    Op::Sigmoidiff, "/", true);

    fn(X, n->d_array, M * K);
       n->siz = M*K;

    cudaMemcpy(n->value.data(), n->d_array, M * K * sizeof(float),
               cudaMemcpyDeviceToHost);

    std::cout << "[CUDA Sigmoidiff output preview]: ";
    for (int i = 0; i < std::min(10, M * K); ++i)
        std::cout << n->value.data()[i] << " ";
    std::cout << "\n";

    n->inputs = {x};
    return n;
    }

    std::shared_ptr<Node> relu_cudaops(const std::shared_ptr<Node>& x){ 
                           const Tensor& xin = x->value;
        auto X = x->d_array;

    auto [M, K]  = x->value.shape();

    auto* fn = ag::kernels::cpu().relu;
    if (!fn)
        throw std::runtime_error("No CUDA RELU kernel registered");

    Tensor C({M, K});
    auto n = std::make_shared<Node>(C, (x->requires_grad),
                                    Op::Relu, "relu", true);

                    

    fn(X, n->d_array, M * K);
     n->siz = M*K;

    // cudaMemcpy(n->value.data(), n->d_array, M * K * sizeof(float),
    //            cudaMemcpyDeviceToHost);

    // std::cout << "[CUDA Sigmoidiff output preview]: ";
    // for (int i = 0; i < std::min(10, M * K); ++i)
    //     std::cout << n->value.data()[i] << " ";
    // std::cout << "\n";


        n->inputs = { x };
        return n;
    }



std::shared_ptr<Node> log_cudaops(const std::shared_ptr<Node>& x){ 
                           const Tensor& xin = x->value;
        auto X = x->d_array;

    auto [M, K]  = x->value.shape();

    auto* fn = ag::kernels::cpu().log;
    if (!fn)
        throw std::runtime_error("No CUDA Log kernel registered");

    Tensor C({M, K});
    auto n = std::make_shared<Node>(C, (x->requires_grad),
                                    Op::Log, "log", true);

                    

    fn(X, n->d_array, M * K);
     n->siz = M*K;

    // cudaMemcpy(n->value.data(), n->d_array, M * K * sizeof(float),
    //            cudaMemcpyDeviceToHost);

    // std::cout << "[CUDA Sigmoidiff output preview]: ";
    // for (int i = 0; i < std::min(10, M * K); ++i)
    //     std::cout << n->value.data()[i] << " ";
    // std::cout << "\n";


        n->inputs = { x };
        return n;
    }

std::shared_ptr<Node> gcu_cudaops(const std::shared_ptr<Node>& x) { 
    const Tensor& xin = x->value;
    auto X = x->d_array;

    auto [M, K] = x->value.shape();
    auto* fn = ag::kernels::cpu().gcu;
    if (!fn)
        throw std::runtime_error("No CUDA GCU kernel registered");

    Tensor C({M, K});
    auto n = std::make_shared<Node>(C, x->requires_grad, Op::GCU, "gcu", true);
    fn(X, n->d_array, M * K);
    n->siz = M * K;
    n->inputs = { x };
    return n;
}

std::shared_ptr<Node> mish_cudaops(const std::shared_ptr<Node>& x) { 
    const Tensor& xin = x->value;
    auto X = x->d_array;

    auto [M, K] = x->value.shape();
    auto* fn = ag::kernels::cpu().mish;
    if (!fn)
        throw std::runtime_error("No CUDA Mish kernel registered");

    Tensor C({M, K});
    auto n = std::make_shared<Node>(C, x->requires_grad, Op::Mish, "mish", true);
    fn(X, n->d_array, M * K);
    n->siz = M * K;
    n->inputs = { x };
    return n;
}

std::shared_ptr<Node> gaus_cudaops(const std::shared_ptr<Node>& x) { 
    const Tensor& xin = x->value;
    auto X = x->d_array;

    auto [M, K] = x->value.shape();
    auto* fn = ag::kernels::cpu().gaus;
    if (!fn)
        throw std::runtime_error("No CUDA Gaussian kernel registered");

    Tensor C({M, K});
    auto n = std::make_shared<Node>(C, x->requires_grad, Op::Gaus, "gaus", true);
    fn(X, n->d_array, M * K);
    n->siz = M * K;
    n->inputs = { x };
    return n;
}

std::shared_ptr<Node> parcon_cudaops(const std::shared_ptr<Node>& x) { 
    const Tensor& xin = x->value;
    auto X = x->d_array;

    auto [M, K] = x->value.shape();
    auto* fn = ag::kernels::cpu().parcon;
    if (!fn)
        throw std::runtime_error("No CUDA Parcon kernel registered");

    Tensor C({M, K});
    auto n = std::make_shared<Node>(C, x->requires_grad, Op::Parcon, "parcon", true);
    fn(X, n->d_array, M * K);
    n->siz = M * K;
    n->inputs = { x };
    return n;
}

std::shared_ptr<Node> lisht_cudaops(const std::shared_ptr<Node>& x) { 
    const Tensor& xin = x->value;
    auto X = x->d_array;

    auto [M, K] = x->value.shape();
    auto* fn = ag::kernels::cpu().lisht;
    if (!fn)
        throw std::runtime_error("No CUDA LiSHT kernel registered");

    Tensor C({M, K});
    auto n = std::make_shared<Node>(C, x->requires_grad, Op::LiSHT, "lisht", true);
    fn(X, n->d_array, M * K);
    n->siz = M * K;
    n->inputs = { x };
    return n;
}

std::shared_ptr<Node> softplus_cudaops(const std::shared_ptr<Node>& x) { 
    const Tensor& xin = x->value;
    auto X = x->d_array;

    auto [M, K] = x->value.shape();
    auto* fn = ag::kernels::cpu().softplus;
    if (!fn)
        throw std::runtime_error("No CUDA Softplus kernel registered");

    Tensor C({M, K});
    auto n = std::make_shared<Node>(C, x->requires_grad, Op::Softplus, "softplus", true);
    fn(X, n->d_array, M * K);
    n->siz = M * K;
    n->inputs = { x };
    return n;
}

std::shared_ptr<Node> silu_cudaops(const std::shared_ptr<Node>& x) { 
    const Tensor& xin = x->value;
    auto X = x->d_array;

    auto [M, K] = x->value.shape();
    auto* fn = ag::kernels::cpu().silu;
    if (!fn)
        throw std::runtime_error("No CUDA SiLU kernel registered");

    Tensor C({M, K});
    auto n = std::make_shared<Node>(C, x->requires_grad, Op::SiLU, "silu", true);
    fn(X, n->d_array, M * K);
    n->siz = M * K;
    n->inputs = { x };
    return n;
}

std::shared_ptr<Node> gelu_cudaops(const std::shared_ptr<Node>& x) { 
    const Tensor& xin = x->value;
    auto X = x->d_array;

    auto [M, K] = x->value.shape();
    auto* fn = ag::kernels::cpu().gelu;
    if (!fn)
        throw std::runtime_error("No CUDA GELU kernel registered");

    Tensor C({M, K});
    auto n = std::make_shared<Node>(C, x->requires_grad, Op::GELU, "gelu", true);
    fn(X, n->d_array, M * K);
    n->siz = M * K;
    n->inputs = { x };
    return n;
}

std::shared_ptr<Node> tanh_cudaops(const std::shared_ptr<Node>& x) { 
    const Tensor& xin = x->value;
    auto X = x->d_array;

    auto [M, K] = x->value.shape();
    auto* fn = ag::kernels::cpu().tanh;
    if (!fn)
        throw std::runtime_error("No CUDA Tanh kernel registered");

    Tensor C({M, K});
    auto n = std::make_shared<Node>(C, x->requires_grad, Op::Tanh, "tanh", true);
    fn(X, n->d_array, M * K);
    n->siz = M * K;
    n->inputs = { x };
    return n;
}







std::shared_ptr<Node> rowsum_cudaops(const std::shared_ptr<Node>& x) { 
    const Tensor& xin = x->value;
    auto X = x->d_array;

    auto [M, K] = x->value.shape();

    auto* fn = ag::kernels::cpu().rowsum;
    if (!fn)
        throw std::runtime_error("No CUDA RowSum kernel registered");

    Tensor C({M, 1});
    auto n = std::make_shared<Node>(C, x->requires_grad, Op::RowSum, "rowsum", true);

    fn(X, n->d_array, M, K);
    n->siz = M;

    n->inputs = { x };
    return n;
}




std::shared_ptr<Node> rowmax_cudaops(const std::shared_ptr<Node>& x) {
    const Tensor& xin = x->value;
    auto X = x->d_array;

    auto [M, K] = xin.shape();

    auto* fn = ag::kernels::cpu().rowmax;
    if (!fn)
        throw std::runtime_error("No CUDA RowMax kernel registered");

    Tensor C({M, 1});
    auto n = std::make_shared<Node>(C, x->requires_grad, Op::RowMax, "rowmax", true);

    fn(X, n->d_array, M, K);
    n->siz = M;

    n->inputs = {x};
    return n;
}



    std::shared_ptr<Node> leaky_relu_cudaops(const std::shared_ptr<Node>& x, float alpha){ 
        Tensor aT(1,1); aT(0,0)=alpha; 

auto aC = constant(aT, "alpha"); 
          auto X = x->d_array;

    auto [M, K]  = x->value.shape();

    auto* fn = ag::kernels::cpu().leakyrelu;
    if (!fn)
        throw std::runtime_error("No CUDA RELU kernel registered");

    Tensor C({M, K});
    auto n = std::make_shared<Node>(C, (x->requires_grad),
                                    Op::Relu, "relu", true);

                    

    fn(X, &alpha, n->d_array, M * K);
     n->siz = M*K;

    // cudaMemcpy(n->value.data(), n->d_array, M * K * sizeof(float),
    //            cudaMemcpyDeviceToHost);

    // std::cout << "[CUDA Sigmoidiff output preview]: ";
    // for (int i = 0; i < std::min(10, M * K); ++i)
    //     std::cout << n->value.data()[i] << " ";
    // std::cout << "\n";


        n->inputs={x, aC.node}; 
        return n;

    }







}






}