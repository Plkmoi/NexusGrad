// =============================================
// cgadimpl/src/core/autodiff.cpp
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
    for (Node* n : order) if (n->requires_grad) n->grad = Tensor::zeros_like(n->value);
}

void backward(const Value& root, const Tensor* grad_seed){
    auto order = topo_from(root.node.get());

    // seed
    if (root.node->requires_grad) {
        root.node->grad = grad_seed ? *grad_seed
                                    : (root.node->value.size()==1 ? Tensor::ones(1,1)
                                                                  : Tensor::ones_like(root.node->value));
    }

    // reverse topo
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        Node* n = *it;

        if (!n->requires_grad) continue;
        const Tensor& gy = n->grad;
if(n->value.is_cuda()) n->requires_cuda=true;
        ag::debug::on_backprop_step(n, gy); // (optional) prints one line per node

        if (n->is_checkpoint && n->value.size() == 0) {
        if (!ag::checkpoint_impl::recompute_subgraph(n->shared_from_this())) {
            throw std::runtime_error("autodiff: failed to recompute checkpointed node during backward");
        }
        }
        VjpFn fn = vjp_lookup(n->op);
        if (fn) fn(n, gy); // handler accumulates into parents

    }
}



void valsend(const Value& root) {
    auto order = topo_from(root.node.get());
    cudaDeviceSynchronize(); // Wait for all GPU ops to finish

    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        Node* n = *it;
if(n->value.is_cuda()) n->requires_cuda=true;
        if (!n->requires_grad || !(n->value.is_cuda())) continue;
        std::cout << "(" << n->value.numel() << ")\n";

        if (n->value.numel()> 0) {
            n->value = ag::Tensor::from_gpu(n->value.data(), n->value.rows(), n->value.cols());
            
        }

        std::cout << "[CUDA VALSEND output]: ";
        for (int i = 0; i < 10; ++i)
          {  std::cout << n->value.data()[i] << " ";
        std::cout << "(" << n->debug_name << ")\n";}
                    std::cout<<n->value.is_cuda()<<"           "<<"sfbsb"<<"  ";

    }
}

void grasend(const Value& root) {
    auto order = topo_from(root.node.get());
    cudaDeviceSynchronize(); // Wait for all GPU ops to finish

    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        Node* n = *it;
        if (!n->requires_grad || !(n->grad.is_cuda())) continue;
        std::cout << "(" << n->value.numel() << ")\n";

        if (n->grad.numel()> 0) {
            n->grad = ag::Tensor::from_gpu(n->grad.data(), n->grad.rows(), n->grad.cols());
        }

        std::cout << "[CUDA GRASEND output]: ";
        for (int i = 0; i < 10; ++i)
          {  std::cout << n->grad.data()[i] << " ";
        std::cout << "(" << n->debug_name << ")\n";}
    }
}

void unisend(const Value& root)
{
    auto order = topo_from(root.node.get());
    cudaDeviceSynchronize(); // Wait for all GPU ops to finish

    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        Node* n = *it;
        
        if (!n->requires_grad || !(n->grad.is_cuda()) || !(n->value.is_cuda())) continue;
        std::cout << "(" << n->value.numel() << ")\n";

        if (n->grad.numel()> 0 || n->value.numel()> 0) {
            n->requires_cuda = true;
            n->value = ag::Tensor::from_gpu(n->value.data(), n->value.rows(), n->value.cols());
            n->grad = ag::Tensor::from_gpu(n->grad.data(), n->grad.rows(), n->grad.cols());

        }

        std::cout << "[CUDA UNISEND output]: ";
        for (int i = 0; i < 10; ++i)
          {  std::cout << n->value.data()[i] << " ";
        std::cout << "(" << n->debug_name << ")\n";
        std::cout << n->grad.data()[i] << " ";
        std::cout << "(" << n->debug_name << ")\n";;
}
    }
    
}

Tensor jvp(const Value& root, const std::unordered_map<Node*, Tensor>& seed){
    if (!root.node) return Tensor{};
    auto order = topo_from(root.node.get());
    std::unordered_map<Node*, Tensor> T;
    T.reserve(order.size());

    auto tangent_of = [&](Node* p) -> const Tensor& {
        auto it = T.find(p);
        if (it != T.end()) return it->second;
        static Tensor Z; // fallback; shouldn't be used for leaves without seeds
        return Z;
    };

    for (Node* n : order) {
        // seed tangent for this node (if provided), else zeros
        Tensor t = Tensor::zeros_like(n->value);
        if (auto it = seed.find(n); it != seed.end()) t = it->second;

        ag::debug::on_jvp_step(n); // (optional) prints forward-mode step

        JvpFn fn = jvp_lookup(n->op);
        if (fn) t = fn(n, tangent_of);

        T[n] = t;
    }
    return T[root.node.get()];
}

} // namespace ag
