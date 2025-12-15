// ====================================================================
// FILE: cgadimpl/src/autodiff/autodiff_jvp_ops.cpp (GPU-Aware Version)
// ====================================================================
#include "ad/detail/autodiff_ops.hpp"
#include <stdexcept> // Required for std::runtime_error
#include "ad/runtime.hpp"

namespace ag {
namespace detail{

// The 'T' shorthand is still useful, so we keep it.
inline const Tensor& T(const std::function<const Tensor&(Node*)>& f, Node* p){ return f(p); }

// ---- elementwise ----

Tensor jvp_Add(Node* n, const std::function<const Tensor&(Node*)>& t){
    // The '+' operator now handles both CPU and GPU, and is stream-aware.
    return T(t, n->inputs[0].get()) + T(t, n->inputs[1].get());
}

Tensor jvp_Sub(Node* n, const std::function<const Tensor&(Node*)>& t){
    // The '-' operator now handles both CPU and GPU.
    return T(t, n->inputs[0].get()) - T(t, n->inputs[1].get());
}

Tensor jvp_Mul(Node* n, const std::function<const Tensor&(Node*)>& t){ 
    Node* A = n->inputs[0].get();
    Node* B = n->inputs[1].get();

    // The '*' and '+' operators now handle both CPU and GPU.
    // The entire expression works on either device and is fully asynchronous on CUDA.
    return (T(t, A) * B->value) + (A->value * T(t, B));
}


// ===================================================================
// jvp_Exp
// ===================================================================
Tensor jvp_Exp(Node* n, const std::function<const Tensor&(Node*)>& t){
Node* X = n->inputs[0].get();
// The '*' operator is device-agnostic and stream-aware.
return T(t, X) * n->value;
}


// ===================================================================
// jvp_Log
// ===================================================================
Tensor jvp_Log(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* X = n->inputs[0].get();
    // The '/' operator is device-agnostic and stream-aware.
    return T(t, X) / X->value;
}

// ===================================================================
// jvp_Tanh
// ===================================================================
Tensor jvp_Tanh(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* X = n->inputs[0].get();
    const Tensor& th = n->value;
    
    // The operators handle broadcasting and device dispatch automatically.
    return T(t, X) * (1.0f - (th * th));
}

// ---- matmul ----
// ===================================================================
// jvp_MatMul
// ===================================================================
Tensor jvp_MatMul(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* A = n->inputs[0].get();
    Node* B = n->inputs[1].get();
    return OwnTensor::matmul(T(t, A), B->value) + OwnTensor::matmul(A->value, T(t, B));
}


// ===================================================================
// jvp_Div
// ===================================================================
Tensor jvp_Div(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* A_node = n->inputs[0].get();
    Node* B_node = n->inputs[1].get();
    const Tensor& A = A_node->value;
    const Tensor& B = B_node->value;
    
    return (T(t, A_node) / B) - (A * T(t, B_node) / (B * B));
}

// ===================================================================
// jvp_Reciprocal
// ===================================================================
Tensor jvp_Reciprocal(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* X_node = n->inputs[0].get();
    const Tensor& X = X_node->value;
    
    return (T(t, X_node) * -1.0f) / (X * X);
}

// ===================================================================
// jvp_Sign
// ===================================================================
Tensor jvp_Sign(Node* n, const std::function<const Tensor&(Node*)>& t){
    // The JVP is 0. We create a zero-filled tensor with the correct shape and device.
    return OwnTensor::Tensor::zeros(n->value.shape(), ag::options(n->value));
}

// ===================================================================
// jvp_Sqrt
// ===================================================================
Tensor jvp_Sqrt(Node* n, const std::function<const Tensor&(Node*)>& t){
    // JVP is tangent(x) * (0.5 / sqrt(x)) = tangent(x) * 0.5 / y
    return T(t, n->inputs[0].get()) * 0.5f / n->value;
}

// ===================================================================
// jvp_Relumask
// ===================================================================
Tensor jvp_Relumask(Node* n, const std::function<const Tensor&(Node*)>& t){
    // Return a zero-filled tensor with the correct shape and device.
    return OwnTensor::Tensor::zeros(n->value.shape(), ag::options(n->value));
}

// ===================================================================
// jvp_Cosh
// ===================================================================
Tensor jvp_Cosh(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* X = n->inputs[0].get();
    return T(t, X) * OwnTensor::sinh(X->value);
}

// ===================================================================
// jvp_Sinh
// ===================================================================
Tensor jvp_Sinh(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* X = n->inputs[0].get();
    return T(t, X) * OwnTensor::cosh(X->value);
}

// ===================================================================
// jvp_Cos
// ===================================================================
Tensor jvp_Cos(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* X = n->inputs[0].get();
    return T(t, X) * -1.0f * OwnTensor::sin(X->value);
}

// ===================================================================
// In file: cgadimpl/src/autodiff/jvp_ops.cpp (Corrected)
// ===================================================================
Tensor jvp_Sin(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* X = n->inputs[0].get();
    return T(t, X) * OwnTensor::cos(X->value);
}


// ===================================================================
// jvp_Transpose
// ===================================================================
Tensor jvp_Transpose(Node* n, const std::function<const Tensor&(Node*)>& t){
    // Call the .t() member function on the tangent tensor.
    return T(t, n->inputs[0].get()).t();
}

// ---- reductions ----
// ===================================================================
// jvp_Sum
// ===================================================================
Tensor jvp_Sum(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* X = n->inputs[0].get();
    return OwnTensor::reduce_sum(t(X));
}

// ===================================================================
// jvp_RowSum
// ===================================================================
Tensor jvp_RowSum(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* X = n->inputs[0].get();
    return OwnTensor::reduce_sum(t(X), {-1}, true); // Sum over the last dimension and keep it
}

// ===================================================================
// jvp_RowMax
// ===================================================================
Tensor jvp_RowMax(Node* n, const std::function<const Tensor&(Node*)>& t){
    // To implement this, we need to know the *index* of the max element in each row.
    // This requires an `argmax` function or comparison operators, which are not 
    // available in OwnTensor.
    throw std::runtime_error("JVP for RowMax cannot be implemented without argmax or comparison ops in the tensor library.");
}

// ===================================================================
// jvp_MeanAll
// ===================================================================
Tensor jvp_MeanAll(Node* n, const std::function<const Tensor&(Node*)>& t){
    Node* X = n->inputs[0].get();
    return OwnTensor::reduce_mean(t(X));
}


Tensor jvp_Leaf(Node*, const std::function<const Tensor&(Node*)>&){
    return Tensor(Shape{}, TensorOptions{}); // unused
}

} // namespace detail


// -------- dispatch table --------
JvpFn jvp_lookup(Op op){
    switch(op){
#define OP(name, arity, str) case Op::name: return &detail::jvp_##name;
#include "ad/detail/ops.def"
#undef OP
        default: return nullptr;
    }
}

} // namespace ag