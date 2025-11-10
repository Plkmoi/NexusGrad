// ====================================================================
// FILE: cgadimpl/src/autodiff/autodiff_nodeops.cpp (Vector-arity Version)
// ====================================================================

#include "ad/detail/autodiff_ops.hpp"
#include "ad/runtime.hpp"
#include "ad/nodeops.hpp"
#include <cmath>
#include <stdexcept>
#include <sstream>

namespace ag {

// Prefer shared ownership in the registry:
using NodePtr = std::shared_ptr<Node>;
using NodeVec = std::vector<NodePtr>;

namespace detail {

// ---- helpers --------------------------------------------------------

static inline void expect_arity(const NodeVec& v, std::size_t need, Op op, const char* opname) {
    if (v.size() != need) {
        std::ostringstream msg;
        msg << "Op " << opname << " expects " << need << " input(s) but got " << v.size();
        throw std::runtime_error(msg.str());
    }
}

#define GET(i)  (v.at(i))

// Macros to cut boilerplate for common arities
#define DEF_UNARY(name, impl) \
    NodePtr node_##name(const NodeVec& v) { \
        expect_arity(v, 1, Op::name, #name); \
        return impl(GET(0)); \
    }

#define DEF_BINARY(name, impl) \
    NodePtr node_##name(const NodeVec& v) { \
        expect_arity(v, 2, Op::name, #name); \
        return impl(GET(0), GET(1)); \
    }

#define DEF_TERNARY(name, impl) \
    NodePtr node_##name(const NodeVec& v) { \
        expect_arity(v, 3, Op::name, #name); \
        return impl(GET(0), GET(1), GET(2)); \
    }

#define DEF_QUATERNARY(name, impl) \
    NodePtr node_##name(const NodeVec& v) { \
        expect_arity(v, 4, Op::name, #name); \
        return impl(GET(0), GET(1), GET(2), GET(3)); \
    }

    inline NodePtr node_Leaf(const NodeVec& v) {
    // You can either:
    // 1) throw (recommended)

    // or 2) if you really want it to “pass through” a sole input, do:
    if (v.size() == 1) return v[0];
    throw std::runtime_error("Leaf expects 1 input in this adapter");
}

// inline NodePtr node_Sum(const NodeVec& v) {
//     // TODO: connect to sum_nodeops(...) if you have it
//     throw std::runtime_error("node_Sum not implemented; wire to your sum_nodeops");
// }

inline NodePtr node_LeakyRelu(const NodeVec& v) {
    // If LeakyRelu in ops.def is arity 2 (x, slope), you'll either pass slope as a Node
    // or capture a scalar. For now, make it compile:
    throw std::runtime_error("node_LeakyRelu not implemented; wire to leakyrelu_nodeops");
}

inline NodePtr node_Dyntanh(const NodeVec& v) {
    throw std::runtime_error("node_Dyntanh not implemented; wire to dyntanh_nodeops");
}

inline NodePtr node_RealLayerNorm(const NodeVec& v) {
    throw std::runtime_error("node_RealLayerNorm not implemented; wire to real_layernorm_nodeops");
}

inline NodePtr node_RealRMSNorm(const NodeVec& v) {
    throw std::runtime_error("node_RealRMSNorm not implemented; wire to real_rmsnorm_nodeops");
}

// ---- unary ----------------------------------------------------------

DEF_UNARY(Relu,        relu_nodeops)
DEF_UNARY(Sigmoid,     sigmoid_nodeops)
DEF_UNARY(Tanh,        tanh_nodeops)
DEF_UNARY(Exp,         exp_nodeops)
DEF_UNARY(Log,         log_nodeops)
DEF_UNARY(GELU,        gelu_nodeops)
DEF_UNARY(SiLU,        silu_nodeops)
DEF_UNARY(Mish,        mish_nodeops)
DEF_UNARY(Parcon,      parcon_nodeops)
DEF_UNARY(LiSHT,       lisht_nodeops)
DEF_UNARY(GCU,         gcu_nodeops)
DEF_UNARY(Sqrt,        sqrt_nodeops)
DEF_UNARY(Sign,        sign_nodeops)
DEF_UNARY(Cos,         cos_nodeops)
DEF_UNARY(Sin,         sin_nodeops)
DEF_UNARY(Cosh,        cosh_nodeops)
DEF_UNARY(Sinh,        sinh_nodeops)
DEF_UNARY(Transpose,   transpose_nodeops)
DEF_UNARY(Softplus,    softplus_nodeops)
DEF_UNARY(Gaus,        gaus_nodeops)
DEF_UNARY(Sum,        sum_nodeops)
DEF_UNARY(Reciprocal,        reci_nodeops)
DEF_UNARY(Relumask,    relumask_nodeops)
DEF_UNARY(RMSNorm,     rms_nodeops)
DEF_UNARY(LayerNorm,   laynor_nodeops)
DEF_UNARY(MeanAll,     mean_all_nodeops)
DEF_UNARY(SoftmaxRow,  softmax_row_nodeops)
DEF_UNARY(LogSumExpRow,logsumexp_row_nodeops)
DEF_UNARY(RowSum,      rowsum_nodeops)
DEF_UNARY(RowMax,      rowmax_nodeops)

// ---- binary ---------------------------------------------------------

DEF_BINARY(Add,        add_nodeops)
DEF_BINARY(Sub,        sub_nodeops)
DEF_BINARY(Mul,        mul_nodeops)
DEF_BINARY(Div,        div_nodeops)
DEF_BINARY(MatMul,     matmul_nodeops)

// ---- ternary --------------------------------------------------------

DEF_TERNARY(FMA,       fmab_nodeops)
DEF_TERNARY(Linear,    linear_nodeops) // (X, W, b)

// ---- quaternary -----------------------------------------------------

DEF_QUATERNARY(Attention, attention_nodeops)
NodePtr node_ALibiAttention(const NodeVec& v); // forward decl (name differs in enum? keep consistent)
NodePtr node_RELUAtt(const NodeVec& v) {
    expect_arity(v, 4, Op::RELUAtt, "RELUAtt");
    return reluatt_nodeops(GET(0), GET(1), GET(2), GET(3));
}
NodePtr node_SigAtt(const NodeVec& v) {
    expect_arity(v, 4, Op::SigAtt, "SigAtt");
    return sigatt_nodeops(GET(0), GET(1), GET(2), GET(3));
}

// Some kernels take extra scalar/bias by ref; push via captured local until API changes.
NodePtr node_AlibiAttention(const NodeVec& v) {
    expect_arity(v, 4, Op::AlibiAttention, "AlibiAttention");
    float dummy_m = 0.f; // TODO: migrate to a proper bias Node input
    return alibiatt_nodeops(GET(0), GET(1), GET(2), GET(3), dummy_m);
}

// ---- 5-ary ----------------------------------------------------------

NodePtr node_SWIGLU(const NodeVec& v) {
    expect_arity(v, 5, Op::SWIGLU, "SWIGLU");
    return swiglu_nodeops(GET(0), GET(1), GET(2), GET(3), GET(4));
}

// ---- losses ---------------------------------------------------------

DEF_BINARY(MSELoss,        mse_loss_nodeops)          // (pred, target)
DEF_BINARY(MAELoss,        mae_loss_nodeops)          // (pred, target)
DEF_BINARY(CeWithLogits,   cross_entropy_with_logits_nodeops) // (logits, onehot)
DEF_BINARY(KLDivergence,   kldivergence_nodeops)      // (logits, onehot)

// ---- MoE ------------------------------------------------------------

DEF_TERNARY(MOE, moewe_nodeops) // (x, W, b)

#undef DEF_UNARY
#undef DEF_BINARY
#undef DEF_TERNARY
#undef DEF_QUATERNARY
#undef GET

} // namespace detail

// Registry lookup stays the same, but the function type must match the vector signature.
FwdFn fwd_lookup(Op op){
    switch(op){
    #define OP(name, arity, str) case Op::name: return &detail::node_##name;
    #include "ad/detail/ops.def"
    #undef OP
    default: return nullptr;
    }
}
} // namespace ag
