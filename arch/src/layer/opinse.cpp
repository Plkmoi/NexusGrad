#include <cmath>
#include <cassert>
#include "layer/archlist.hpp"

namespace flow {


// const char* op_namea(Op o) {
//   static constexpr std::array<const char*, OpCount> names = {{
//   #define OP(name, arity, str) str,
// #include "layer/newops.def"
//   #undef OP
//   }};
//   return names[static_cast<std::size_t>(o)];
// }



// VjpFn vjp_lookupa(Op op){
// switch(op){
// #define OP(name, arity, str) case Op::name: return &flow::vjp_##name;
// #include "layer/newops.def"
// #undef OP
//     default: return nullptr;
//     std::cout<<op_namea(op);
// }
// }

// void backwarda(const Value& root, const Tensor* grad_seed){
//     auto order = topo_from(root.node.get());
//     // std::cout<<"HERE";

//     for (Node* n : order) {
//         if (n->requires_grad() && n->grad.numel() == 0) {
//             n->grad = Tensor::zeros(n->value.shape(), ag::options(n->value));
//         }
//             // std::cout<<"HERE2";

//     }

//      // seed
//     if (root.node->requires_grad()) {
//             // std::cout<<"HERE3";

//         if (grad_seed) {
//             root.node->grad = *grad_seed;
//         } else {
//             // Use the new factories and get options from the value tensor
//                 // std::cout<<"HERE4";

//             auto opts = ag::options(root.node->value);
//             if (root.node->value.numel() == 1) {
//                 root.node->grad.fill(1.0f);
//             } else {
//                 root.node->grad = OwnTensor::Tensor::ones(root.node->value.shape(), opts);
//             }
//         }
//     }

//     // reverse topo
//     for (auto it = order.rbegin(); it != order.rend(); ++it) {
//         Node* n = *it;
//         // The requires_grad() check is now a function call
//         if (!n->requires_grad()) continue;
//         const Tensor& gy = n->grad;

//         ag::debug::on_backprop_step(n, gy); // (optional) prints one line per node

//         if (n->is_checkpoint && n->value.numel() == 0) {
//         if (!ag::checkpoint_impl::recompute_subgraph(n->shared_from_this())) {
//             throw std::runtime_error("autodiff: failed to recompute checkpointed node during backward");
//         }
//         }
//         VjpFn fn = vjp_lookupa(n->op);
//         if (fn) fn(n, gy); // handler accumulates into parents
// std::cout<<"HERE3";
//         // check_tensors_nangrad(n);
//     }
// }


} // namespace ag::nn