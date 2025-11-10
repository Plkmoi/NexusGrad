
// ====================================================================
// FILE: cgadimpl/src/autodiff/autodiff_nodeops.cpp (GPU-Aware Version)
// ====================================================================

#include "ad/detail/autodiff_ops.hpp"
#include "ad/runtime.hpp"
#include "ad/nodeops.hpp"
#include <cmath>
#include <stdexcept> // Required for std::runtime_error

namespace ag {
namespace detail{



std::shared_ptr<Node> node_Relu     (const std::shared_ptr<Node>& x) { return relu_nodeops(x); }
std::shared_ptr<Node> node_Sigmoid  (const std::shared_ptr<Node>& x) { return sigmoid_nodeops(x); }
std::shared_ptr<Node> node_Tanh     (const std::shared_ptr<Node>& x) { return tanh_nodeops(x); }
std::shared_ptr<Node> node_Exp      (const std::shared_ptr<Node>& x) { return exp_nodeops(x); }
std::shared_ptr<Node> node_Log      (const std::shared_ptr<Node>& x) { return log_nodeops(x); }
std::shared_ptr<Node> node_GELU     (const std::shared_ptr<Node>& x) { return gelu_nodeops(x); }
std::shared_ptr<Node> node_SiLU     (const std::shared_ptr<Node>& x) { return silu_nodeops(x); }
std::shared_ptr<Node> node_Mish     (const std::shared_ptr<Node>& x) { return mish_nodeops(x); }
std::shared_ptr<Node> node_Parcon   (const std::shared_ptr<Node>& x) { return parcon_nodeops(x); }
std::shared_ptr<Node> node_LiSHT    (const std::shared_ptr<Node>& x) { return lisht_nodeops(x); }
std::shared_ptr<Node> node_GCU      (const std::shared_ptr<Node>& x) { return gcu_nodeops(x); }
std::shared_ptr<Node> node_Sqrt     (const std::shared_ptr<Node>& x) { return sqrt_nodeops(x); }
std::shared_ptr<Node> node_Sign     (const std::shared_ptr<Node>& x) { return sign_nodeops(x); }
std::shared_ptr<Node> node_Cos      (const std::shared_ptr<Node>& x) { return cos_nodeops(x); }
std::shared_ptr<Node> node_Sin      (const std::shared_ptr<Node>& x) { return sin_nodeops(x); }
std::shared_ptr<Node> node_Cosh     (const std::shared_ptr<Node>& x) { return cosh_nodeops(x); }
std::shared_ptr<Node> node_Sinh     (const std::shared_ptr<Node>& x) { return sinh_nodeops(x); }
std::shared_ptr<Node> node_Transpose(const std::shared_ptr<Node>& x) { return transpose_nodeops(x); }
std::shared_ptr<Node> node_Softplus (const std::shared_ptr<Node>& x) { return softplus_nodeops(x); }
std::shared_ptr<Node> node_Gaus     (const std::shared_ptr<Node>& x) { return gaus_nodeops(x); }
std::shared_ptr<Node> node_Relumask (const std::shared_ptr<Node>& x) { return relumask_nodeops(x); }
std::shared_ptr<Node> node_RMSNorm  (const std::shared_ptr<Node>& x) { return rms_nodeops(x); }
std::shared_ptr<Node> node_LayerNorm(const std::shared_ptr<Node>& x) { return laynor_nodeops(x); }
std::shared_ptr<Node> node_MeanAll  (const std::shared_ptr<Node>& x) { return mean_all_nodeops(x); }
std::shared_ptr<Node> node_SoftmaxRow(const std::shared_ptr<Node>& x){ return softmax_row_nodeops(x); }
std::shared_ptr<Node> node_LogSumExpRow(const std::shared_ptr<Node>& x){ return logsumexp_row_nodeops(x); }

// Binary
std::shared_ptr<Node> node_Add   (const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){ return add_nodeops(a,b); }
std::shared_ptr<Node> node_Sub   (const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){ return sub_nodeops(a,b); }
std::shared_ptr<Node> node_Mul   (const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){ return mul_nodeops(a,b); }
std::shared_ptr<Node> node_Div   (const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){ return div_nodeops(a,b); }
std::shared_ptr<Node> node_MatMul(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){ return matmul_nodeops(a,b); }
// RowSum/RowMax are unary; expose via NodeFn1 lookup only:
std::shared_ptr<Node> node_RowSum(const std::shared_ptr<Node>& a){ return rowsum_nodeops(a); }
std::shared_ptr<Node> node_RowMax(const std::shared_ptr<Node>& a){ return rowmax_nodeops(a); }

// Ternary
std::shared_ptr<Node> node_FMA   (const std::shared_ptr<Node>& a,
                                  const std::shared_ptr<Node>& b,
                                  const std::shared_ptr<Node>& c){ return fmab_nodeops(a,b,c); }

std::shared_ptr<Node> node_Linear(const std::shared_ptr<Node>& X,
                                  const std::shared_ptr<Node>& W,
                                  const std::shared_ptr<Node>& b){ return linear_nodeops(X,W,b); }

// Quaternary
std::shared_ptr<Node> node_Attention(const std::shared_ptr<Node>& a,
                                     const std::shared_ptr<Node>& b,
                                     const std::shared_ptr<Node>& c,
                                     const std::shared_ptr<Node>& d){ return attention_nodeops(a,b,c,d); }

std::shared_ptr<Node> node_RELUAtt (const std::shared_ptr<Node>& a,
                                    const std::shared_ptr<Node>& b,
                                    const std::shared_ptr<Node>& c,
                                    const std::shared_ptr<Node>& d){ return reluatt_nodeops(a,b,c,d); }

std::shared_ptr<Node> node_SigAtt  (const std::shared_ptr<Node>& a,
                                    const std::shared_ptr<Node>& b,
                                    const std::shared_ptr<Node>& c,
                                    const std::shared_ptr<Node>& d){ return sigatt_nodeops(a,b,c,d); }

std::shared_ptr<Node> node_AlibiAttention(const std::shared_ptr<Node>& a,
                                          const std::shared_ptr<Node>& b,
                                          const std::shared_ptr<Node>& c,
                                          const std::shared_ptr<Node>& d){
    float dummy_m = 0.f; // your current signature takes a float& m; prefer pushing bias via inputs
    return alibiatt_nodeops(a,b,c,d, dummy_m);
}

// Losses
std::shared_ptr<Node> node_MSELoss(const std::shared_ptr<Node>& pred,
                                   const std::shared_ptr<Node>& target){ return mse_loss_nodeops(pred, target); }

std::shared_ptr<Node> node_MAELoss(const std::shared_ptr<Node>& pred,
                                   const std::shared_ptr<Node>& target){ return mae_loss_nodeops(pred, target); }

std::shared_ptr<Node> node_CeWithLogits(const std::shared_ptr<Node>& logits,
                                        const std::shared_ptr<Node>& onehot){ return cross_entropy_with_logits_nodeops(logits, onehot); }

std::shared_ptr<Node> node_KLDivergence(const std::shared_ptr<Node>& logits,
                                        const std::shared_ptr<Node>& onehot){ return kldivergence_nodeops(logits, onehot); }

// MoE / SWIGLU (note: SWIGLU is 5-arg; keep separate helper or pass packed)
std::shared_ptr<Node> node_MOE(const std::shared_ptr<Node>& x,
                               const std::shared_ptr<Node>& W,
                               const std::shared_ptr<Node>& b){ return moewe_nodeops(x,W,b); }

// Expose a convenience adapter that expects 5 inputs via a tiny helper:
std::shared_ptr<Node> node_SWIGLU(const std::shared_ptr<Node>& X,
                                  const std::shared_ptr<Node>& A,
                                  const std::shared_ptr<Node>& B,
                                  const std::shared_ptr<Node>& C,
                                  const std::shared_ptr<Node>& D){
    return swiglu_nodeops(X,A,B,C,D);
}




}


FwdFn fwd_lookup(Op op){
    switch(op){
#define OP(name, arity, str) case Op::name: return &detail::node_##name;
#include "ad/detail/ops.def"
#undef OP
        default: return nullptr;
    }
}


}