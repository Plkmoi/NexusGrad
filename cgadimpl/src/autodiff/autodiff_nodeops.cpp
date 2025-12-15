// =====================
// FILE: cgadimpl/src/autodiff/autodiff_nodeops.cpp
// =====================
#include "ad/nodeops.hpp"
#include "ad/runtime.hpp"
#include "ad/kernels_api.hpp"
#include <cuda_runtime.h>
#include "TensorLib.h" 
#include <unordered_map>
#include <cmath> 
#include "ad/detail/autodiff_ops.hpp"


        
namespace ag {
namespace detail {
// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------
// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------




static inline float scalar_from_1x1( Tensor& t) {
    return t.data<float>()[0];
}

// ------------------------------------------------------------
// Elementwise binary
// ------------------------------------------------------------
void node_Add( std::shared_ptr<Node> n) { n->value = n->inputs[0]->value + n->inputs[1]->value; }
void node_Sub( std::shared_ptr<Node> n) { n->value = n->inputs[0]->value - n->inputs[1]->value; }
void node_Mul( std::shared_ptr<Node> n) { n->value = n->inputs[0]->value * n->inputs[1]->value; }
void node_Div( std::shared_ptr<Node> n) { n->value = n->inputs[0]->value / n->inputs[1]->value; }

// ------------------------------------------------------------
// Unary elementwise
// ------------------------------------------------------------

void node_Tanh( std::shared_ptr<Node> n) {
    n->value = OwnTensor::tanh(n->inputs[0]->value);
}
void node_Exp( std::shared_ptr<Node> n)  { n->value = OwnTensor::exp(n->inputs[0]->value); }
void node_Log( std::shared_ptr<Node> n)  { n->value = OwnTensor::log(n->inputs[0]->value); }
void node_Cosh( std::shared_ptr<Node> n) { n->value = cosh(n->inputs[0]->value); }
void node_Sinh( std::shared_ptr<Node> n) { n->value = sinh(n->inputs[0]->value); }
void node_Cos ( std::shared_ptr<Node> n) { n->value = cos (n->inputs[0]->value); }
void node_Sin ( std::shared_ptr<Node> n) { n->value = sin (n->inputs[0]->value); }
void node_Sign( std::shared_ptr<Node> n) { n->value = OwnTensor::sign(n->inputs[0]->value, ag::current_stream()); }
void node_Sqrt( std::shared_ptr<Node> n) { n->value = OwnTensor::sqrt(n->inputs[0]->value, ag::current_stream()); }

void node_Reciprocal( std::shared_ptr<Node> n) {
    n->value = 1.0f / n->inputs[0]->value;
}

// ------------------------------------------------------------
// Scalar ⊙ Tensor (ants passed as 1×1 leafs in inputs[1])
// ------------------------------------------------------------


// ------------------------------------------------------------
// Linear algebra & views
// ------------------------------------------------------------
void node_MatMul( std::shared_ptr<Node> n) {
    n->value = matmul(n->inputs[0]->value, n->inputs[1]->value);
}
void node_Transpose( std::shared_ptr<Node> n) {
    n->value = n->inputs[0]->value.t();
}


// ------------------------------------------------------------
// Reductions
// ------------------------------------------------------------
void node_Sum( std::shared_ptr<Node> n) {
    n->value = OwnTensor::reduce_sum(n->inputs[0]->value, {}, false);
}
void node_MeanAll( std::shared_ptr<Node> n) {
    n->value = OwnTensor::reduce_mean(n->inputs[0]->value);
}
void node_RowSum( std::shared_ptr<Node> n) {
    n->value = OwnTensor::reduce_sum(n->inputs[0]->value, {1}, true);
}
void node_RowMax( std::shared_ptr<Node> n) {
    n->value = OwnTensor::reduce_max(n->inputs[0]->value, {1}, true);
}


// ------------------------------------------------------------
// Softmax family
// ------------------------------------------------------------


// ------------------------------------------------------------
// Losses
// ------------------------------------------------------------


// ------------------------------------------------------------
// Attention blocks
// ------------------------------------------------------------



void node_Leaf( std::shared_ptr<Node> n) {
}



// ------------------------------------------------------------
// Masks / misc unary with custom kernels
// ------------------------------------------------------------
void node_Relumask( std::shared_ptr<Node> n) {
     Tensor& X = n->inputs[0]->value;
    Tensor y = OwnTensor::Tensor::zeros(X.shape(), ag::options(X));
    if (X.is_cpu()) {
        dispatch_by_dtype(X.dtype(), [&](auto dummy){
            using T = decltype(dummy);
             T* x = X.data<T>(); T* m = y.data<T>();
            for (int64_t i=0;i<X.numel();++i) if (x[i] > T(0)) m[i] = T(1);
        });
    } else {
        throw std::runtime_error("relumask: CUDA impl missing");
    }
    n->value = std::move(y);
}



    // Tensor forward_eval_node_impl( std::shared_ptr<Node> node) {
    //     if (!node) throw std::runtime_error("forward_eval_node: null node");
    //     switch (node->op) {
    //         case Op::Add: return node->inputs[0]->value + node->inputs[1]->value;
    //         case Op::Sub: return node->inputs[0]->value - node->inputs[1]->value;
    //         case Op::Mul: return node->inputs[0]->value * node->inputs[1]->value;
    //         case Op::MatMul: return Tensor::matmul(node->inputs[0]->value, node->inputs[1]->value);
    //         case Op::Relu: return Tensor::relu(node->inputs[0]->value);
    //         case Op::Sigmoid: return Tensor::sigmoid(node->inputs[0]->value);
    //         case Op::Tanh: return Tensor::tanh(node->inputs[0]->value);
    //         case Op::Exp: return Tensor::exp(node->inputs[0]->value);
    //         case Op::Log: return Tensor::log(node->inputs[0]->value);
    //         case Op::AlibiAttention: {
    //              Tensor &a = node->inputs[0]->value;
    //              Tensor &b = node->inputs[1]->value;
    //              Tensor &c = node->inputs[2]->value;
    //              Tensor &d = node->inputs[3]->value;
    //             Tensor q = Tensor::matmul(a, b);
    //             Tensor k = Tensor::matmul(a, c);
    //             Tensor v = Tensor::matmul(a, d);
    //             Tensor logits = Tensor::matmul(q, Tensor::transpose(k) * (1.f / sqrt(float(k.cols()))));
    //             Tensor bias   = Tensor::alibi(logits.rows(), logits.cols(), /*m*/128);
    //             Tensor g      = logits + bias;
    //             Tensor s      = Tensor::softmax_row(g);
    //             return Tensor::matmul(s, v);
    //         }
    //         case Op::Leaf:
    //             return node->value;
    //         default:
    //             if (!node->tape.empty()) {
    //                 return *(node->tape.back());
    //             }
    //             throw std::runtime_error("forward_eval_node: unsupported op for recompute");
    //     }
    // }

    } // namespace detail

    FwdFn fwd_lookup(Op op){
    switch(op){
    #define OP(name, arity, str) case Op::name: return &detail::node_##name;
    #include "ad/detail/ops.def"
    #undef OP
    default: return nullptr;
    }
}
    } // namespace ag