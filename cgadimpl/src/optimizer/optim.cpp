// =====================
// file: cgadimpl/src/optim.cpp
// =====================
#include "optim.hpp"
#include <math.h>
#include "ad/runtime.hpp"

// #include "ag/nodeops.hpp"

namespace ag {

void SGD(const Value& root, int learning_rate , const Tensor* grad_seed) {
    auto order = topo_from(root.node.get());

    // seed
    if (root.node->requires_grad) {
        root.node->grad = grad_seed ? *grad_seed
                                    : (root.node->value.size()==1 ? Tensor::ones(1,1)
                                                                  : Tensor::ones_like(root.node->value));
    }

    // reverse topo
    for (auto it = order.begin(); it != order.end(); ++it) {
        Node* n = *it;
        if (n->requires_grad ) {
{

        auto fn = ag::kernels::cuda().optim;
     if (fn) {
//         for (int i = 0; i < 3; i++) {
//     printf("Row %d sum: %f\n", i, n->value.data()[i]);
// }
            fn(n->value.data(), n->grad.data(), 2.0f, n->value.numel(), ag::current_stream());
            std::cout<<"YES!!!!!!!!!!!";
//                     for (int i = 0; i < 3; i++) {
//     printf("Row %d sum: %f\n", i, n->value.data()[i]);
// }
        } else {
            throw std::runtime_error("Sum forward on CUDA not implemented or loaded.");
        }
}


}
    }

}

}