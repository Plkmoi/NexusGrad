// ===================================================================
// file: cgadimpl/src/optim.cpp (Corrected for OwnTensor)
// ===================================================================
#include "ops/optim.hpp"
#include <math.h>

// No new includes are needed because tensor.hpp brings in everything.

namespace flow {

void Optimizer::SGD(const Value& root, float learning_rate) { // Changed to float for consistency
    auto order = topo_from(root.node.get());

    // NOTE: The 'backward' function is responsible for seeding the initial gradient.
    // The SGD optimizer's job is just to update the weights.
    // Therefore, the "seed" block is not actually needed here, as backward() must
    // have been called first, which would have already populated all the gradients.
    // We can remove it for a cleaner implementation.

    // The loop now correctly iterates forward to find Leaf nodes that are parameters.
    this->epoch = [this, learning_rate, order]()  {

        
    for (Node* n : order) {
        // We only update nodes that are trainable parameters.
        // In our design, these are Leaf nodes that require a gradient.
        if (n->op == Op::Leaf && n->requires_grad()) {
            
            // --- THE FIX: Use the new, stream-aware operators ---
            
            // 1. (-learning_rate * n->grad)
            // This calls the overloaded operator*(float, Tensor), which correctly
            // gets the stream from the context for GPU operations.
            
            // 2. n->value += ...
            // This calls the overloaded operator+=, which also correctly
            // gets the stream from the context for GPU operations.
            n->value += -learning_rate * n->grad;
        }
    }
};
}



void Optimizer::SGDm(const Value& root, float learning_rate, float momen) { // Changed to float for consistency
    topovel_from(root.node.get());
    this->epoch = [this, learning_rate, momen]()  {
    
        for (auto ww : bagstore) 
    {
        //  std::cout<<"I am here";

        auto n = ww->noda;
        if (n->op == Op::Leaf && n->requires_grad()) {
            
            *(ww->velocia) = momen*(*(ww->velocia)) +learning_rate * n->grad;
            n->value = n->value-*(ww->velocia);
            // std::cout<<"I am here";
            
        }
    }
    };


}
Optimizer opti; 

} // namespace ag