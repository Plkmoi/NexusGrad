// =============================================
// cgadimpl/src/core/autodiff.cpp
// Summary of the Changes
//     zero_grad():
//         Tensor::zeros(n->value.shape()) was changed to OwnTensor::Tensor::zeros(n->value.shape(), ag::options(n->value)).
//         Reason: This is the most important pattern. We must use the ag::options() helper to create a new zero-filled tensor that has the same device (CPU/CUDA) and dtype as the value tensor it corresponds to.
//     backward():
//         The grad_seed logic was rewritten to use OwnTensor::Tensor::ones with the correct Shape and TensorOptions, again ensuring the seed tensor is created on the correct device.
//         The check n->value.size() == 0 was correctly changed to n->value.numel() == 0 to match the new API.
//     jvp():
//         The initial return Tensor{} was changed to return Tensor{Shape{}, TensorOptions{}} to correctly default-construct an empty tensor.
//         T.reserve(order.numel()) was a bug. order is a std::vector, so we use order.size().
//         The static fallback tensor Z was changed to be a correctly default-constructed empty tensor.
//         The creation of the tangent tensor t was changed from Tensor::zeros(n->value) to the correct OwnTensor::Tensor::zeros(n->value.shape(), ag::options(n->value)) to ensure it's on the right device.
// You have now updated all the core logic of the autodiff engine to be fully compatible with the new OwnTensor library. This was a critical step.
// =============================================
#include <unordered_map>
#include <stdexcept>
#include "ad/autodiff.hpp"
#include "ad/detail/autodiff_ops.hpp"
#include "ad/debug.hpp"
#include <ad/checkpoint.hpp>
namespace ag {

void zero_grad(const Value& root){
    auto order = topo_from(root.node.get());
    for (Node* n : order) if (n->requires_grad()) n->grad = Tensor::zeros(n->value.shape(), ag::options(n->value));
}

void zero_val(const Value& root){
    auto order = topo_from(root.node.get());
    for (Node* n : order) if (n->requires_grad() && n->op!=Op::Leaf) n->value = Tensor::zeros(n->value.shape(), ag::options(n->value).with_device(n->value.device()));
}

Value shallow(const Value& q) {
    if (!q.node) throw std::runtime_error("shallow(): q.node is null");

    auto n = std::make_shared<Node>();   // ← allocate a fresh Node
    n->value = q.node->value;            // copies Tensor handle (may alias storage)
    n->grad  = q.node->grad;
    n->op    = q.node->op;
    n->inputs    = q.node->inputs;
    n->tape    = q.node->tape;

    return Value{std::move(n)};
}

inline std::string printering(Tensor w){
    std::string s = "[";
    const auto& dims = w.shape().dims;
    for (size_t i = 0; i < dims.size(); ++i) {
        s += std::to_string(dims[i]);
        if (i + 1 < dims.size()) s += ", ";
    }
    s += "]";
    return s;
}

void backward(const Value& root, const Tensor* grad_seed){
    auto order = topo_from(root.node.get());
    // std::cout<<"HERE";

    for (Node* n : order) {
        if (n->requires_grad() && n->grad.numel() == 0) {
            n->grad = Tensor::zeros(n->value.shape(), ag::options(n->value));
        }
            // std::cout<<"HERE2";

    }

     // seed
    if (root.node->requires_grad()) {
            // std::cout<<"HERE3";

        if (grad_seed) {
            root.node->grad = *grad_seed;
        } else {
            // Use the new factories and get options from the value tensor
                // std::cout<<"HERE4";

            auto opts = ag::options(root.node->value);
            if (root.node->value.numel() == 1) {
                root.node->grad.fill(1.0f);
            } else {
                root.node->grad = OwnTensor::Tensor::ones(root.node->value.shape(), opts);
            }
        }
    }


    // reverse topo
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        Node* n = *it;
        // The requires_grad() check is now a function call
        if (!n->requires_grad()) continue;
        const Tensor& gy = n->grad;


        
        // 🔍 DEBUG: print incoming grad shapes
    std::cout << "\n>>> BACKWARD NODE: " << n->debug_name ;
    std::cout << " (Op: " << op_name(n->op) << ")\n";
    std::cout << "  value shape: " << printering(n->value) << "\n";
    std::cout << "  grad  shape: " << printering(gy) << "\n";
    std::cout << "  input0 shape: " << printering(n->inputs[0]->value) << "\n";
    if(n->op==Op::AlibiAttention)
    {
    std::cout << "  input1 shape: " << printering(n->inputs[1]->value) << "\n";
    std::cout << "  input2 shape: " << printering(n->inputs[2]->value) << "\n";
    std::cout << "  input3 shape: " << printering(n->inputs[3]->value) << "\n";


    }

        ag::debug::on_backprop_step(n, gy); // (optional) prints one line per node

        if (n->is_checkpoint && n->value.numel() == 0) {
        if (!ag::checkpoint_impl::recompute_subgraph(n->shared_from_this())) {
            throw std::runtime_error("autodiff: failed to recompute checkpointed node during backward");
        }
        }
        VjpFn fn = vjp_lookup(n->op);
        if (fn) fn(n, gy); // handler accumulates into parents
    }
}

Tensor jvp(const Value& root, const std::unordered_map<Node*, Tensor>& seed){
    if (!root.node) return Tensor{Shape{}, TensorOptions{}}; // Return a valid empty tensor
    
    auto order = topo_from(root.node.get());
    std::unordered_map<Node*, Tensor> T;
    T.reserve(order.size()); // Use vector's .size(), not a tensor's .numel()

    auto tangent_of = [&](Node* p) -> const Tensor& {
        auto it = T.find(p);
        if (it != T.end()) return it->second;
        // A static empty tensor is a good fallback
        static Tensor Z{Shape{}, TensorOptions{}}; 
        return Z;
    };

    for (Node* n : order) {
        // seed tangent for this node (if provided), else zeros of the correct shape/device
        Tensor t = OwnTensor::Tensor::zeros(n->value.shape(), ag::options(n->value));
        if (auto it = seed.find(n); it != seed.end()) {
            t = it->second;
        }

        ag::debug::on_jvp_step(n);

        JvpFn fn = jvp_lookup(n->op);
        if (fn) t = fn(n, tangent_of);

        T[n] = t;
    }
    return T[root.node.get()];
}


void disten(Value Y, Device dev){
                Y.node->value = Y.node->value.to(dev);
            Y.node->grad = OwnTensor::Tensor::zeros(Y.node->value.shape(), ag::options(Y.node->value));


}

void forward(const Value& root) {
    auto order = topon_from(root.node);
    for (std::shared_ptr<Node> n : order) {
        if (n->op == Op::Leaf) continue;  // already has a value

        auto fn = fwd_lookup(n->op);  // you can reuse your op forward registry
        // auto r = n->shared_from_this();
        if (fn) fn(n);
    }
}

Value forwardstor(const Value& root) {
    auto order = topon_from(root.node);
    for (std::shared_ptr<Node> n : order) {
        if (n->op == Op::Leaf) continue;  // already has a value

        auto fn = fwd_lookup(n->op);  // you can reuse your op forward registry
        // auto r = n->shared_from_this();
        if (fn) fn(n);
    }

    return root;
}


// void forwarde(const Value& root) {
//     auto order = topo_from(root.node.get());
//     for (Node* n : order) {
//         if (n->op == Op::Leaf) continue;  // already has a value

//         auto fn = fwd_lookup(n->op);  // you can reuse your op forward registry
//         if (fn) {
//             n = fn(n->inputs).get(); // recompute
//             std::cout<<"HERE  ";
//             ag::debug::print_tensor("Work Y", n->value);
//         }
//     }
// }

} // namespace ag