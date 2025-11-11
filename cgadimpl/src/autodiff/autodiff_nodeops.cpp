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
void node_Relu( std::shared_ptr<Node> n) {
     Tensor& X = n->inputs[0]->value;
    n->value = (X + OwnTensor::abs(X, ag::current_stream())) * 0.5f;
}
void node_Sigmoid( std::shared_ptr<Node> n) {
     Tensor& X = n->inputs[0]->value;
    n->value = 1.0f / (1.0f + OwnTensor::exp(X * -1.0f));
}
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
void node_Softplus( std::shared_ptr<Node> n) {
     Tensor& X = n->inputs[0]->value;
    n->value = OwnTensor::log(1.0f + OwnTensor::exp(X));
}
void node_Mish( std::shared_ptr<Node> n) {
     Tensor& X = n->inputs[0]->value;
    Tensor sp = OwnTensor::log(1.0f + OwnTensor::exp(X));
    n->value   = X * OwnTensor::tanh(sp);
}
void node_GELU( std::shared_ptr<Node> n) {
     Tensor& X = n->inputs[0]->value;
     float c1 = 0.7978845608f; // sqrt(2/pi)
     float c2 = 0.044715f;
    Tensor X3 = X * X * X;
    Tensor u  = (X + X3 * c2) * c1;
    n->value  = X * (1.0f + OwnTensor::tanh(u)) * 0.5f;
}
void node_SiLU( std::shared_ptr<Node> n) {
     Tensor& X = n->inputs[0]->value;
    Tensor sig = 1.0f / (1.0f + OwnTensor::exp(X * -1.0f));
    n->value   = X * sig;
}
void node_Gaus( std::shared_ptr<Node> n) {
     Tensor& X = n->inputs[0]->value;
    n->value = OwnTensor::exp((X * X) * -1.0f);
}
void node_Parcon( std::shared_ptr<Node> n) {
     Tensor& X = n->inputs[0]->value;
    n->value = X * (2.0f - X);
}
void node_LiSHT( std::shared_ptr<Node> n) {
     Tensor& X = n->inputs[0]->value;
    n->value = X * OwnTensor::tanh(X);
}
void node_GCU( std::shared_ptr<Node> n) {
     Tensor& X = n->inputs[0]->value;
    n->value = X * OwnTensor::cos(X);
}
void node_Reciprocal( std::shared_ptr<Node> n) {
    n->value = 1.0f / n->inputs[0]->value;
}

// ------------------------------------------------------------
// Scalar ⊙ Tensor (ants passed as 1×1 leafs in inputs[1])
// ------------------------------------------------------------
void node_LeakyRelu( std::shared_ptr<Node> n) {
     Tensor& X = n->inputs[0]->value;
     float alpha = scalar_from_1x1(n->inputs[1]->value);
    Tensor absx = OwnTensor::abs(X, ag::current_stream());
    Tensor pos  = (X + absx) * 0.5f;
    Tensor neg  = (X - absx) * 0.5f;
    n->value = pos + neg * alpha;
}

// ------------------------------------------------------------
// Linear algebra & views
// ------------------------------------------------------------
void node_MatMul( std::shared_ptr<Node> n) {
    n->value = matmul(n->inputs[0]->value, n->inputs[1]->value);
}
void node_Transpose( std::shared_ptr<Node> n) {
    n->value = n->inputs[0]->value.t();
}
void node_Linear( std::shared_ptr<Node> n) {
    // inputs: X, W, b   (W: [out,in])
     Tensor& X = n->inputs[0]->value;
     Tensor& W = n->inputs[1]->value;
     Tensor& b = n->inputs[2]->value;
    n->value = matmul(X, W.t()) + b;
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
void node_RMSNorm( std::shared_ptr<Node> n) {
     Tensor& X = n->inputs[0]->value;
    Tensor var   = OwnTensor::reduce_mean(X * X, {-1}, true);
    Tensor rsqrt = 1.0f / OwnTensor::sqrt(var + 1e-5f, ag::current_stream());
    n->value     = X * rsqrt;
}
void node_RealRMSNorm( std::shared_ptr<Node> n) {
    // inputs: X, G
     Tensor& X = n->inputs[0]->value;
     Tensor& G = n->inputs[1]->value;
     float inv_cols = 1.0f / static_cast<float>(X.shape().dims.back());
    Tensor var   = OwnTensor::reduce_sum(X * X, {-1}, true) * inv_cols;
    Tensor rsqrt = 1.0f / OwnTensor::sqrt(var + 1e-5f, ag::current_stream());
    n->value     = (X * rsqrt) * G;
}
void node_LayerNorm( std::shared_ptr<Node> n) {
     Tensor& X = n->inputs[0]->value;
    Tensor mean = OwnTensor::reduce_mean(X, {-1}, true);
    Tensor xm   = X - mean;
    Tensor var  = OwnTensor::reduce_mean(xm * xm, {-1}, true);
    n->value    = xm / OwnTensor::sqrt(var + 1e-5f, ag::current_stream());
}
void node_RealLayerNorm( std::shared_ptr<Node> n) {
    // inputs: X, G, B
     Tensor& X = n->inputs[0]->value;
     Tensor& G = n->inputs[1]->value;
     Tensor& B = n->inputs[2]->value;
    Tensor mean = OwnTensor::reduce_mean(X, {-1}, true);
    Tensor xm   = X - mean;
    Tensor var  = OwnTensor::reduce_mean(xm * xm, {-1}, true);
    Tensor norm = xm / OwnTensor::sqrt(var + 1e-5f, ag::current_stream());
    n->value    = norm * G + B;
}

// ------------------------------------------------------------
// Softmax family
// ------------------------------------------------------------
void node_SoftmaxRow( std::shared_ptr<Node> n) {
     Tensor& Z = n->inputs[0]->value;
    Tensor m   = OwnTensor::reduce_max(Z, {-1}, true);
    Tensor e   = OwnTensor::exp(Z - m);
    Tensor s   = OwnTensor::reduce_sum(e, {-1}, true);
    n->value   = e / s;
}
void node_LogSumExpRow( std::shared_ptr<Node> n) {
     Tensor& Z = n->inputs[0]->value;
    Tensor m   = OwnTensor::reduce_max(Z, {-1}, true);
    Tensor e   = OwnTensor::exp(Z - m);
    Tensor s   = OwnTensor::reduce_sum(e, {-1}, true);
    n->value   = OwnTensor::log(s) + m;
}

// ------------------------------------------------------------
// Losses
// ------------------------------------------------------------
void node_MSELoss( std::shared_ptr<Node> n) {
     Tensor& P = n->inputs[0]->value;
     Tensor& T = n->inputs[1]->value;
    Tensor diff = P - T;
    n->value    = OwnTensor::reduce_mean(diff * diff);
}
void node_MAELoss( std::shared_ptr<Node> n) {
     Tensor& P = n->inputs[0]->value;
     Tensor& T = n->inputs[1]->value;
    n->value = OwnTensor::reduce_mean(OwnTensor::abs(P - T, ag::current_stream()));
}
void node_CeWithLogits( std::shared_ptr<Node> n) {
     Tensor& Z = n->inputs[0]->value; // logits [B,C]
     Tensor& Y = n->inputs[1]->value; // onehot [B,C]
    Tensor m   = OwnTensor::reduce_max(Z, {-1}, true);
    Tensor zs  = Z - m;
    Tensor lse = OwnTensor::log(OwnTensor::reduce_sum(OwnTensor::exp(zs), {-1}, true));
    Tensor lsm = zs - lse;
    Tensor prod= Y * lsm;
    Tensor s   = OwnTensor::reduce_sum(prod, {-1});
    n->value   = OwnTensor::reduce_mean(s * -1.0f);
}
void node_KLDivergence( std::shared_ptr<Node> n) {
     Tensor& Z = n->inputs[0]->value;
     Tensor& Y = n->inputs[1]->value;
    Tensor logY = OwnTensor::log(Y + 1e-9f);
    Tensor m    = OwnTensor::reduce_max(Z, {-1}, true);
    Tensor zs   = Z - m;
    Tensor lse  = OwnTensor::log(OwnTensor::reduce_sum(OwnTensor::exp(zs), {-1}, true));
    Tensor lsmZ = zs - lse;
    Tensor elt  = Y * (logY - lsmZ);
    Tensor s    = OwnTensor::reduce_sum(elt, {-1});
    n->value    = OwnTensor::reduce_mean(s);
}

// ------------------------------------------------------------
// Attention blocks
// ------------------------------------------------------------
void node_Attention( std::shared_ptr<Node> n) {
     Tensor& A = n->inputs[0]->value;
     Tensor& B = n->inputs[1]->value;
     Tensor& C = n->inputs[2]->value;
     Tensor& D = n->inputs[3]->value;

    Tensor q = matmul(A, B);
    Tensor k = matmul(A, C);
    Tensor v = matmul(A, D);

    float scale = 1.f / sqrtf(static_cast<float>(k.shape().dims.back()));
    Tensor g = matmul(q, k.t()) * scale;

    Tensor m   = OwnTensor::reduce_max(g, {-1}, true);
    Tensor e   = OwnTensor::exp(g - m);
    Tensor s   = OwnTensor::reduce_sum(e, {-1}, true);
    Tensor p   = e / s;

    n->value = matmul(p, v);
}
void node_SigAtt( std::shared_ptr<Node> n) {
     Tensor& A = n->inputs[0]->value;
     Tensor& B = n->inputs[1]->value;
     Tensor& C = n->inputs[2]->value;
     Tensor& D = n->inputs[3]->value;

    Tensor q = OwnTensor::matmul(A, B);
    Tensor k = OwnTensor::matmul(A, C);
    Tensor v = OwnTensor::matmul(A, D);

    float scale = 1.f / sqrtf(static_cast<float>(k.shape().dims.back()));
    Tensor g = OwnTensor::matmul(q, k.t()) * scale;

    Tensor s = 1.0f / (1.0f + OwnTensor::exp(g * -1.0f));
    n->value = OwnTensor::matmul(s, v);
}
void node_RELUAtt( std::shared_ptr<Node> n) {
     Tensor& A = n->inputs[0]->value;
     Tensor& B = n->inputs[1]->value;
     Tensor& C = n->inputs[2]->value;
     Tensor& D = n->inputs[3]->value;

    Tensor q = OwnTensor::matmul(A, B);
    Tensor k = OwnTensor::matmul(A, C);
    Tensor v = OwnTensor::matmul(A, D);

    float scale = 1.f / sqrtf(static_cast<float>(k.shape().dims.back()));
    Tensor g = OwnTensor::matmul(q, k.t()) * scale;

    Tensor relu = (g + OwnTensor::abs(g, ag::current_stream())) * 0.5f;
    n->value = OwnTensor::matmul(relu, v);
}
void node_AlibiAttention( std::shared_ptr<Node> n) {
     Tensor& A = n->inputs[0]->value;
     Tensor& B = n->inputs[1]->value;
     Tensor& C = n->inputs[2]->value;
     Tensor& D = n->inputs[3]->value;

    Tensor q = OwnTensor::matmul(A, B);
    Tensor k = OwnTensor::matmul(A, C);
    Tensor v = OwnTensor::matmul(A, D);

    float scale = 1.f / sqrtf(static_cast<float>(k.shape().dims.back()));
    Tensor logits = OwnTensor::matmul(q, k.t()) * scale;

    Tensor bias_cpu(logits.shape(), OwnTensor::TensorOptions().with_dtype(logits.dtype()));
    {
        int n_heads = logits.shape().dims[0];
        int L       = logits.shape().dims[1];
        float slope_start = 1.0f / powf(2.0f, 8.0f / n_heads);
        dispatch_by_dtype(bias_cpu.dtype(), [&](auto dummy){
            using T = decltype(dummy);
            T* data = bias_cpu.data<T>();
            for (int h=0; h<n_heads; ++h) {
                float slope = powf(slope_start, h + 1);
                for (int i=0;i<L;++i) for (int j=0;j<L;++j) {
                    data[h*L*L + i*L + j] =
                        (j > i) ? -std::numeric_limits<float>::infinity()
                                : static_cast<T>(-(L - 1 - j) * slope);
                }
            }
        });
    }
    Tensor g = logits + bias_cpu.to(logits.device());

    Tensor m   = OwnTensor::reduce_max(g, {-1}, true);
    Tensor e   = OwnTensor::exp(g - m);
    Tensor s   = OwnTensor::reduce_sum(e, {-1}, true);
    Tensor p   = e / s;

    n->value = OwnTensor::matmul(p, v);
}

// ------------------------------------------------------------
// Misc blocks
// ------------------------------------------------------------
void node_FMA( std::shared_ptr<Node> n) {
    n->value = matmul(n->inputs[0]->value, n->inputs[1]->value) + n->inputs[2]->value;
}

void node_Leaf( std::shared_ptr<Node> n) {
}



void node_MOE( std::shared_ptr<Node> n) {
     Tensor& X = n->inputs[0]->value;
     Tensor& W = n->inputs[1]->value;
     Tensor& B = n->inputs[2]->value;
    Tensor logits = OwnTensor::matmul(X, W.t()) + B;
    Tensor m   = OwnTensor::reduce_max(logits, {-1}, true);
    Tensor e   = OwnTensor::exp(logits - m);
    Tensor s   = OwnTensor::reduce_sum(e, {-1}, true);
    n->value   = e / s;
}
void node_SWIGLU( std::shared_ptr<Node> n) {
     Tensor& X = n->inputs[0]->value;
     Tensor& A = n->inputs[1]->value;
     Tensor& B = n->inputs[2]->value;
     Tensor& C = n->inputs[3]->value;
     Tensor& D = n->inputs[4]->value;

    Tensor gate = OwnTensor::matmul(X, A.t()) + B;
    Tensor silu = gate * (1.0f / (1.0f + OwnTensor::exp(gate * -1.0f)));
    Tensor proj = OwnTensor::matmul(X, C.t()) + D;
    n->value = silu * proj;
}
void node_RealLayerNorm( std::shared_ptr<Node> n); // already above
void node_Dyntanh( std::shared_ptr<Node> n) {
     Tensor& X = n->inputs[0]->value;
     Tensor& A = n->inputs[1]->value;
     Tensor& B = n->inputs[2]->value;
     Tensor& G = n->inputs[3]->value;
    Tensor h = X * A;
    n->value = OwnTensor::tanh(h) * G + B;
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