// =====================
// file: cgadimpl/src/nodeops.cpp
// =====================
#include "ad/nodeops.hpp"
#include "ad/runtime.hpp"
#include "ad/kernels_api.hpp"
#include <cuda_runtime.h>
#include "TensorLib.h" 
#include <unordered_map>
#include <cmath> 




namespace ag {
namespace detail {

// force it to run at startup


std::shared_ptr<Node> add_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){
    // This correctly uses the stream-aware overloaded operator+
    Tensor Y = a->value + b->value; 
    // FIX: Use the new 3-argument Node constructor
    auto n = std::make_shared<Node>(Y, Op::Add, (a->requires_grad() || b->requires_grad()), "+");
    n->inputs = {a, b};
    ag::debug::on_node_created(n);
    return n;
}
  
std::shared_ptr<Node> sub_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){
    // This correctly uses the stream-aware overloaded operator-
    Tensor Y = a->value - b->value;
    // FIX: Use the new 3-argument Node constructor
    auto n = std::make_shared<Node>(Y, Op::Sub, (a->requires_grad() || b->requires_grad()), "-");
    n->inputs = {a, b};
    ag::debug::on_node_created(n);
    return n;
}

std::shared_ptr<Node> mul_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){ 
    // This correctly uses the stream-aware overloaded operator*
    Tensor y = a->value * b->value; 
    // FIX: Use the new 3-argument Node constructor
    auto n = std::make_shared<Node>(y, Op::Mul, (a->requires_grad() || b->requires_grad()), "*"); 
    n->inputs = {a, b}; 
    ag::debug::on_node_created(n); 
    return n; 
}

std::shared_ptr<Node> div_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){
    const Tensor& C = a->value / b->value;

    auto n = std::make_shared<Node>(C, Op::Div, (a->requires_grad() || b->requires_grad()), "/");
    n->inputs = { a, b };
    ag::debug::on_node_created(n);  
    return n;
}

// ================
// flomul nodeops
// ================
std::shared_ptr<Node> flomul_nodeops(const std::shared_ptr<Node>& a, float b) {
    // 1. A static cache to store nodes we create for scalars.
    //    The key is the float value, the value is the shared_ptr to the Node.
    static std::unordered_map<float, std::shared_ptr<Node>> scalar_cache;

    std::shared_ptr<Node> c; // This will be the node for our scalar 'b'

    // 2. Look for the scalar 'b' in our cache.
    auto it = scalar_cache.find(b);

    if (it != scalar_cache.end()) {
        // --- CACHE HIT ---
        // We've already created a node for this float value. Reuse it.
        c = it->second;
    } else {
        // --- CACHE MISS ---
        // This is the first time we've seen this float. Create a new node for it.
        // The tensor for the scalar only needs to be a 1x1 tensor.
        // The multiplication op will automatically handle broadcasting.
        Tensor scalar_tensor = Tensor::full(Shape({1,1}), TensorOptions().with_req_grad(false), b);

        c = std::make_shared<Node>(scalar_tensor, Op::Leaf, "leaf_scalar");

        // Store the new node in the cache for next time.
        scalar_cache[b] = c;
    }

    // 3. Now, perform the multiplication using node 'a' and the (cached or new) scalar node 'c'.
    // The underlying operator* will handle the stream context correctly.
    Tensor y = a->value * c->value;

    auto n = std::make_shared<Node>(y, Op::Mul, a->requires_grad(), "*");
    n->inputs = {a, c};
    ag::debug::on_node_created(n);
    return n;
}


// =====================================================================================================
// matmul nodeops
// =====================================================================================================
std::shared_ptr<Node> matmul_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b) {
    // --- 1. Call the tensor library's matmul function directly ---
    // This function will automatically handle:
    //  - Device checking (CPU vs GPU)
    //  - Dispatching to the correct backend (CPU generic vs. CUDA kernel)
    //  - Getting the current stream from the context for GPU operations
    //  - All dimension and broadcasting validation
    Tensor C = matmul(a->value, b->value);

    // --- 2. Wrap the result in a new Node ---
    // The new Node constructor correctly infers requires_grad from the output tensor C.
    auto n = std::make_shared<Node>(C, Op::MatMul, (a->requires_grad() || b->requires_grad()), "matmul");
    n->inputs = {a, b};
    ag::debug::on_node_created(n);
    return n;
}


// ===================================================================
// reci_nodeops
// ===================================================================

std::shared_ptr<Node> reci_nodeops(const std::shared_ptr<Node>& a) {
    // This correctly uses the stream-aware overloaded operator for scalar / Tensor.
    Tensor y = 1.0f / a->value;
    
    // Use the new 3-argument Node constructor.
    auto n = std::make_shared<Node>(y, Op::Reciprocal, a->requires_grad(),"reciprocal");
    n->inputs = {a};
    ag::debug::on_node_created(n);
    return n;
}

// ===================================================================
// flodiv_nodeops
// ===================================================================

std::shared_ptr<Node> flodiv_nodeops(float b, const std::shared_ptr<Node>& a) {
    // --- Step 1 (Optional but Recommended): Use the Scalar Cache ---
    // This is the same efficient pattern from flomul_nodeops.
    static std::unordered_map<float, std::shared_ptr<Node>> scalar_cache;
    std::shared_ptr<Node> c;
    auto it = scalar_cache.find(b);
    if (it != scalar_cache.end()) {
        c = it->second;
    } else {
        Tensor scalar_tensor = Tensor::full(Shape{{1, 1}}, TensorOptions().with_req_grad(false), b);
        c = std::make_shared<Node>(scalar_tensor, Op::Leaf, "leaf_scalar");
        scalar_cache[b] = c;
    }

    // --- Step 2: Perform the operation ---
    // This correctly uses the stream-aware overloaded operator for Tensor / Tensor.
    Tensor y = c->value / a->value;
    
    // --- Step 3: Create the Node ---
    auto n = std::make_shared<Node>(y, Op::Div, a->requires_grad(), "/");
    n->inputs = {c, a}; // Note the order: c is the numerator, a is the denominator
    ag::debug::on_node_created(n);
    return n;
}

// ===================================================================
// floadd_nodeops
// ===================================================================

std::shared_ptr<Node> floadd_nodeops(float b, const std::shared_ptr<Node>& a) {
    // --- Step 1: Use the Scalar Cache ---
    static std::unordered_map<float, std::shared_ptr<Node>> scalar_cache;
    std::shared_ptr<Node> c;
    auto it = scalar_cache.find(b);
    if (it != scalar_cache.end()) {
        c = it->second;
    } else {
        Tensor scalar_tensor = Tensor::full(Shape{{1, 1}}, TensorOptions().with_req_grad(false), b);
        c = std::make_shared<Node>(scalar_tensor, Op::Leaf, "leaf_scalar");
        scalar_cache[b] = c;
    }

    // --- Step 2: Perform the operation ---
    // This correctly uses the stream-aware overloaded operator for Tensor + Tensor.
    Tensor y = c->value + a->value;
    
    // --- Step 3: Create the Node ---
    auto n = std::make_shared<Node>(y, Op::Add, a->requires_grad(), "+");
    n->inputs = {c, a}; // Order matches the operation
    ag::debug::on_node_created(n);
    return n;
}
// ===================================================================
// relumask_nodeops
// ===================================================================

std::shared_ptr<Node> relumask_nodeops(const std::shared_ptr<Node>& x) {
    const Tensor& xin = x->value;

    // FIX: Use the new factory with options
    Tensor y = OwnTensor::Tensor::zeros(xin.shape(), ag::options(xin));

    if (xin.is_cpu()) {
        // Your existing CPU implementation is fine, just needs the new API
        // We must dispatch by dtype to get the correct pointer type.
        // dispatch_by_dtype(xin.dtype(), [&](auto dummy){
        //     using T = decltype(dummy);
        //     const T* x_data = xin.data<T>();
        //     T* y_data = y.data<T>();
        //     for (int64_t i = 0; i < xin.numel(); ++i) {
        //         if (x_data[i] > T(0)) {
        //             y_data[i] = T(1);
        //         }
        //     }
        // });
    } else {
        // For now, we will add a placeholder, as you don't have a GPU kernel for this.
        // To make this work on GPU, you would need to add a 'relumask_cuda' to your kernels plugin.
        throw std::runtime_error("relumask_nodeops not implemented for CUDA yet.");
    }
    
    // FIX: Use the new Node constructor
    auto n = std::make_shared<Node>(y, Op::Relumask, x->requires_grad(), "relumask");
    n->inputs = {x};
    ag::debug::on_node_created(n);
    return n;
}


// ===================================================================
// cosh_nodeops
// ===================================================================

    std::shared_ptr<Node> cosh_nodeops(const std::shared_ptr<Node>& x){
        Tensor y = cosh(x->value);
        auto n=std::make_shared<Node>(y, Op::Cosh, x->requires_grad(), "cosh");
        n->inputs={x};
        ag::debug::on_node_created(n);
        return n;
    }

// ===================================================================
// sinh_nodeops
// ===================================================================

     std::shared_ptr<Node> sinh_nodeops(const std::shared_ptr<Node>& x){
        Tensor y = sinh(x->value);
        auto n=std::make_shared<Node>(y, Op::Sinh, x->requires_grad(), "sinh");
        n->inputs={x};
        ag::debug::on_node_created(n);
        return n;
    }

// ===================================================================
// cos_nodeops
// ===================================================================


     std::shared_ptr<Node> cos_nodeops(const std::shared_ptr<Node>& x){
        Tensor y = cos(x->value);
        auto n=std::make_shared<Node>(y, Op::Cos, x->requires_grad(), "cosh");
        n->inputs={x};
        ag::debug::on_node_created(n);
        return n;
    }

// ===================================================================
// sin_nodeops
// ===================================================================

     std::shared_ptr<Node> sin_nodeops(const std::shared_ptr<Node>& x){
        Tensor y = sin(x->value);
        auto n=std::make_shared<Node>(y, Op::Sin, x->requires_grad(), "sinh");
        n->inputs={x};
        ag::debug::on_node_created(n);
        return n;
    }

// ===================================================================
// In file: cgadimpl/src/nodeops.cpp (Corrected sign_nodeops)
// ===================================================================

std::shared_ptr<Node> sign_nodeops(const std::shared_ptr<Node>& x){
    // Call the stream-aware OwnTensor::sign function
    Tensor y = OwnTensor::sign(x->value, ag::current_stream());

    // Use the new 3-argument Node constructor
    auto n = std::make_shared<Node>(y, Op::Sign, x->requires_grad(), "sign");
    n->inputs={x};
    ag::debug::on_node_created(n);
    return n;
}


// ===================================================================
// sqrt_nodeops
// ===================================================================
std::shared_ptr<Node> sqrt_nodeops(const std::shared_ptr<Node>& x) {
    // 1. Call the OwnTensor::sqrt function directly.
    // It will handle device dispatch and stream context automatically.
    Tensor y = OwnTensor::sqrt(x->value, ag::current_stream());

    // 2. Wrap the result in a new Node using the correct constructor.
    auto n = std::make_shared<Node>(y, Op::Sqrt, x->requires_grad(), "sqrt");
    n->inputs = {x};
    ag::debug::on_node_created(n);
    return n;
}


// ============================================================================
// sum_nodeops
// ============================================================================
 
std::shared_ptr<Node> sum_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = OwnTensor::reduce_sum(x->value, {}, false);
    auto n = std::make_shared<Node>(y, Op::Sum, x->requires_grad(), "sum");
    n->inputs = {x};
    ag::debug::on_node_created(n);
    return n;
}

// ============================================================================
// transpose_nodeops
// ============================================================================

std::shared_ptr<Node> transpose_nodeops(const std::shared_ptr<Node>& x){
    // .t() is a zero-copy view operation. It doesn't need a stream
    // as no computation is performed. It just returns a new Tensor
    // with different strides. This is highly efficient.
    Tensor y = x->value.t();
    
    // FIX: Use the correct Op and name, and the correct constructor.
    auto n = std::make_shared<Node>(y, Op::Transpose, x->requires_grad(), "transpose");
    n->inputs = {x};
    ag::debug::on_node_created(n);
    return n;
}


// ============================================================================
// exp_nodeops
// ============================================================================

std::shared_ptr<Node> exp_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = OwnTensor::exp(x->value);
    
    // 3. Use the correct Node constructor.
    auto n = std::make_shared<Node>(y, Op::Exp, x->requires_grad(), "exp");
    n->inputs = {x};
    ag::debug::on_node_created(n);
    return n;
}

// ===================================================================
// log_nodeops
// ===================================================================
std::shared_ptr<Node> log_nodeops(const std::shared_ptr<Node>& x){
    Tensor y = OwnTensor::log(x->value);
    
    auto n = std::make_shared<Node>(y, Op::Log, x->requires_grad(), "log");
    n->inputs = {x};
    ag::debug::on_node_created(n);
    return n;
}

// ===============================================================================
// tanh nodeops
// ===============================================================================
  std::shared_ptr<Node> tanh_nodeops(const std::shared_ptr<Node>& x){
    // 1. Call the OwnTensor::tanh function directly.
    // This single call will automatically:
    //  - Check if the tensor is on the CPU or GPU.
    //  - Call the appropriate backend (CPU or CUDA kernel).
    //  - Get the current stream from the context if it's on the GPU.
    //  - Queue the operation asynchronously on that stream.
    Tensor y = OwnTensor::tanh(x->value);

    // 2. Wrap the result in a new Node using the correct constructor.
    auto n = std::make_shared<Node>(y, Op::Tanh, x->requires_grad(), "tanh");
    n->inputs = {x};
    ag::debug::on_node_created(n);
    return n;
}

// ============================================================================================
// rowsum_nodeops
// ============================================================================================
    std::shared_ptr<Node> rowsum_nodeops(const std::shared_ptr<Node>& x){
    // Reduce over axis 1 (the columns), and keep the dimension so shape goes from [B,C] to [B,1].
    Tensor y = OwnTensor::reduce_sum(x->value, {1}, true);
    auto n = std::make_shared<Node>(y, Op::RowSum, x->requires_grad(), "rowsum");
    n->inputs = {x};
    ag::debug::on_node_created(n);
    return n;
}

// ===================================================================
// rowmax_nodeops
// ===================================================================
std::shared_ptr<Node> rowmax_nodeops(const std::shared_ptr<Node>& x){
    // Reduce over axis 1 (columns) and keep the dimension.
    Tensor y = OwnTensor::reduce_max(x->value, {1}, true);
    auto n = std::make_shared<Node>(y, Op::RowMax, x->requires_grad(), "rowmax");
    n->inputs={x};
    ag::debug::on_node_created(n);
    return n;
}



// ===================================================================
// mean_all_nodeops
// ===================================================================
std::shared_ptr<Node> mean_all_nodeops(const std::shared_ptr<Node>& x){
    // reduce_mean with empty axes reduces over the entire tensor
    Tensor y = OwnTensor::reduce_mean(x->value);
    auto n = std::make_shared<Node>(y, Op::MeanAll, x->requires_grad(), "meanall");
    n->inputs={x};
    ag::debug::on_node_created(n);
    return n;
}





    // Tensor forward_eval_node_impl(const std::shared_ptr<Node>& node) {
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
    //             const Tensor &a = node->inputs[0]->value;
    //             const Tensor &b = node->inputs[1]->value;
    //             const Tensor &c = node->inputs[2]->value;
    //             const Tensor &d = node->inputs[3]->value;
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
    } // namespace ag