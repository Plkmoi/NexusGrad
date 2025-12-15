// ====================================================================
// FILE: cgadimpl/src/autodiff/autodiff_vjp_ops.cpp (GPU-Aware Version)
// ====================================================================

#include "ad/detail/autodiff_ops.hpp"
#include "ad/runtime.hpp"
#include <cmath>
#include <stdexcept> // Required for std::runtime_error
#include "ad/debug.hpp"
#include "ad/kernels_api.hpp"
namespace ag {
namespace detail{

static Tensor reduce_for_broadcast(const Tensor& grad_in, const Tensor& target_val) {
    // If shapes already match, no reduction is needed.
    if (grad_in.shape().dims == target_val.shape().dims) {
        return grad_in;
    }

    const auto& grad_dims = grad_in.shape().dims;
    const auto& target_dims = target_val.shape().dims;
    
    std::vector<int64_t> axes_to_sum;
    int grad_ndim = grad_dims.size();
    int target_ndim = target_dims.size();
    
    // Find which axes of the incoming gradient need to be summed.
    for (int i = 0; i < grad_ndim; ++i) {
        int target_dim_idx = i + target_ndim - grad_ndim;
        if (target_dim_idx < 0 || (target_dims[target_dim_idx] == 1 && grad_dims[i] > 1)) {
            axes_to_sum.push_back(i);
        }
    }

    Tensor summed_grad = grad_in;
    if (!axes_to_sum.empty()) {
        summed_grad = OwnTensor::reduce_nansum(grad_in, axes_to_sum, true);
    }
    
    // Reshape to exactly match the target shape (e.g., from [1, 5] to [5]).
    return summed_grad.reshape(target_val.shape());
}

// // ----- elementwise binary -----
// // Correct: Accumulates gradient for both parents.
void vjp_Add(Node* n, const Tensor& gy){
    Node* A = n->inputs[0].get(); 
    Node* B = n->inputs[1].get();
    if (A->requires_grad()) A->grad += reduce_for_broadcast(gy, A->value);
    if (B->requires_grad()) B->grad += reduce_for_broadcast(gy, B->value);
}

void vjp_Sub(Node* n, const Tensor& gy){
    Node* A = n->inputs[0].get();
    Node* B = n->inputs[1].get();
    if (A->requires_grad()) A->grad += reduce_for_broadcast(gy, A->value);
    if (B->requires_grad()) B->grad += reduce_for_broadcast(gy * -1.0f, B->value);
}

void vjp_Mul(Node* n, const Tensor& gy){
    Node* A = n->inputs[0].get();
    Node* B = n->inputs[1].get();
    if (A->requires_grad()) A->grad += reduce_for_broadcast(gy * B->value, A->value);
    if (B->requires_grad()) B->grad += reduce_for_broadcast(gy * A->value, B->value);
}
void vjp_Div(Node* n, const Tensor& gy){
    Node* A = n->inputs[0].get();
    Node* B = n->inputs[1].get();

    // VJP for A: dL/dA = gy * (1/B)
    if (A->requires_grad())  A->grad += reduce_for_broadcast(gy / B->value, A->value);
    
    // VJP for B: dL/dB = gy * (-A / (B*B))
    if (B->requires_grad()) {
        Tensor grad_B = gy * -1.0f * A->value / (B->value * B->value);
        B->grad += reduce_for_broadcast(grad_B, B->value);
    }
}


// ===================================================================
// vjp_Exp
// ===================================================================

void vjp_Exp(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad()) return;

    // The VJP for exp(x) is gy * exp(x). The forward pass output is exp(x).
    // This uses the stream-aware OwnTensor operator '*' for both CPU and GPU.
    X->grad += gy * n->value;
}

// ===================================================================
// vjp_Log
// ===================================================================

void vjp_Log(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad()) return;

    // The VJP for log(x) is gy / x.
    // This uses the stream-aware OwnTensor operator '/' for both CPU and GPU.
    X->grad += gy / X->value;
}



// ===================================================================
// vjp_Tanh
// ===================================================================
void vjp_Tanh(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad()) return;

    // VJP is gy * (1 - tanh(x)^2)
    // Here, t = n->value is the result of the forward tanh(x)
    const Tensor& t = n->value;
    X->grad += gy * (1.0f - (t * t));
}


// ===================================================================
// vjp_Transpose
// ===================================================================
void vjp_Transpose(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad()) return;

    // The VJP is just the transpose of the gradient.
    X->grad += gy.t();
}


// ===================================================================
// vjp_MatMul
// ===================================================================
void vjp_MatMul(Node* n, const Tensor& gy){
    Node* A_node = n->inputs[0].get();
    Node* B_node = n->inputs[1].get();
    const Tensor& A = A_node->value;
    const Tensor& B = B_node->value;

    // VJP for A: dL/dA = dL/dY @ B^T
    if (A_node->requires_grad()) {
        A_node->grad += OwnTensor::matmul(gy, B.t());
    }

    // VJP for B: dL/dB = A^T @ dL/dY
    if (B_node->requires_grad()) {
        B_node->grad += OwnTensor::matmul(A.t(), gy);
    }
}

// ----- Reductions -----
// ===================================================================
// vjp_Sum
// ===================================================================
void vjp_Sum(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad()) return;

    // `gy` is a 1x1 scalar tensor. The '+' operator will automatically
    // broadcast it to the shape of X->grad.
    X->grad += gy*Tensor::ones(X->grad.shape(), TensorOptions().with_device(X->grad.device()));
    
}

// ===================================================================
// vjp_RowSum
// ===================================================================
void vjp_RowSum(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad()) return;

    // `gy` has shape [B, 1]. The '+' operator will automatically
    // broadcast it to the shape of X->grad, which is [B, C].
    X->grad += gy;
}

// ===================================================================
// vjp_RowMax
// ===================================================================
void vjp_RowMax(Node* n, const Tensor& gy){
    // To implement this, we need to know the *index* of the max element in each row.
    // This requires an `argmax` function which is not available in OwnTensor.
    // The VJP would look like:
    //   Tensor indices = OwnTensor::argmax(X->value, /*axis=*/-1);
    //   X->grad.scatter_add_(/*axis=*/-1, indices, gy);
    throw std::runtime_error("VJP for RowMax cannot be implemented without argmax or comparison ops in the tensor library.");
}

// ===================================================================
// vjp_MeanAll
// ===================================================================

void vjp_MeanAll(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad()) return;
    
    // The gradient needs to be scaled by 1/N.
    float scale = 1.0f / static_cast<float>(X->value.numel());
    
    // `gy` is a scalar. `gy * scale` is also a scalar.
    // The `+=` operator will broadcast this scalar across the entire gradient tensor.
    X->grad += gy * scale;
}


// ===================================================================
// vjp_Reciprocal
// ===================================================================
void vjp_Reciprocal(Node* n, const Tensor& gy){
    Node* X_node = n->inputs[0].get();
    if (!X_node->requires_grad()) return;
    const Tensor& X = X_node->value;
    // VJP is gy * (-1 / X^2)
    X_node->grad += gy * -1.0f / (X * X);
}


// ===================================================================
// vjp_Cosh
// ===================================================================
void vjp_Cosh(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad()) return;

    // VJP is gy * sinh(x)
    X->grad += gy * OwnTensor::sinh(X->value);
}

// ===================================================================
// vjp_Sinh
// ===================================================================
void vjp_Sinh(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad()) return;

    // VJP is gy * cosh(x)
    X->grad += gy * OwnTensor::cosh(X->value);
}


// ===================================================================
// vjp_Sign
// ===================================================================
void vjp_Sign(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad()) return;

    // The gradient is zero. We add gy * 0 to correctly handle shapes
    // in case of broadcasting.
    X->grad += gy * 0.0f;
}

// ===================================================================
// vjp_Cos
// ===================================================================
void vjp_Cos(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad()) return;

    // VJP is gy * -sin(x)
    X->grad += gy * -1.0f * OwnTensor::sin(X->value);
}

// ===================================================================
// vjp_Sin
// ===================================================================
void vjp_Sin(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad()) return;

    // VJP is gy * cos(x)
    X->grad += gy * OwnTensor::cos(X->value);
}


// =================================================================== 
//  vjp_Sqrt
// ===================================================================
void vjp_Sqrt(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad()) return;

    // VJP is gy * (0.5 / sqrt(x)) = gy * 0.5 / y
    // n->value is the result of the forward pass, which is sqrt(x).
    X->grad += gy * 0.5f / n->value;
}

// ===================================================================
// vjp_Relumask
// ===================================================================
void vjp_Relumask(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad()) return;

    // The gradient is 0. We multiply by gy to ensure correct broadcasting
    // for a zero-like tensor.
    X->grad += gy * 0.0f;
}


void vjp_Leaf(Node*, const Tensor&){ /* no-op */ }

} // namespace detail


// -------- dispatch table --------
VjpFn vjp_lookup(Op op){
    switch(op){
#define OP(name, arity, str) case Op::name: return &detail::vjp_##name;
#include "ad/detail/ops.def"
#undef OP
        default: return nullptr;
    }
}

} // namespace ag