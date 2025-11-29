// =====================
// file: cgadimpl/include/ag/optim.hpp (declarations only)
// =====================
#pragma once
#include "ad/ops.hpp"
#include "ad/debug.hpp"
#include "ad/autodiff.hpp"
#include "ad/detail/autodiff_ops.hpp"
#include "tensor.hpp"
#include "ad/debug.hpp"
#include <math.h>
#include <unordered_set>
#include "ad/runtime.hpp"

namespace ag {

    class Optimizer
    {

        private:
        static constexpr float _lr = 0.1f;
        static constexpr float _m = 0.01f;

        


struct Bag {

std::shared_ptr<ag::Node> noda;
std::shared_ptr<Tensor> velocia;
    Bag(const std::shared_ptr<ag::Node>& n_val) 
        : noda(n_val),        // Initialize the 'noda' member
          velocia(std::make_shared<Tensor>(
    Tensor::zeros(n_val->value.shape(),
                  TensorOptions().with_device(n_val->value.device()))
))
      // Initialize the 'velocia' member
    {
        std::cout<<"I was called\n";
    }



};

// Bag storestuf;
std::vector<std::shared_ptr<Bag>> bagstore;


        void topovel_from(std::shared_ptr<Node> root) {
    std::unordered_set<std::shared_ptr<Node>> vis; vis.reserve(256);
    std::function<void(std::shared_ptr<Node>)> dfs = [&](std::shared_ptr<Node> n) {
        if (!n || vis.count(n)) return;
        vis.insert(n);
        for (auto& p : n->inputs)
        if(n->op==Op::Leaf && n->requires_grad()){
            dfs(p); 
        bagstore.push_back(std::make_shared<Bag>(n));
        // storestuf.veloci.push_back(std::make_shared<Tensor>(Tensor(n->value.shape(), TensorOptions().with_device(n->value.device()))));
        // storestuf.nod.push_back(n);
        }
    };
    dfs(root);

}
        

        public:
                void SGD(const Value& root, float learning_rate=_lr);
        void SGDm(const Value& root, float learning_rate=_lr, float momentum=_m);
        using EpochFn = std::function<void()>;

        EpochFn epoch;
    //     ~Optimizer()
    //     {

    //  for (auto ww : bagstore) 
    // {
    //     auto n = ww->noda;
    //     n->value = n->value.to(Device::CPU);
    //     n->grad = n->grad.to(Device::CPU);
        

    // }

    //     }

// void optiche(Value f) {
//     // auto order = topon_from(root.node);
//     // for (std::shared_ptr<Node> n : order) {
//     //     if (n->op == Op::Leaf) continue;  // already has a value

//     //     auto fn = fwd_lookup(n->op);  // you can reuse your op forward registry
//     //     // auto r = n->shared_from_this();
//     //     if (fn) fn(n);
//     // }
// Value m = shallow(f);
//     epoch();

//         auto order = topon_from(f.node);
//     for (std::shared_ptr<Node> n : order) {
//         if (n->op == Op::Leaf) continue;  // already has a value

//         if(reduce_sum(OwnTensor::abs(f.val()-m.val(), ag::current_stream())).data<float>()[0]==0.0)
//         std::cout<<"ok\n";
//     }
// }




    };


extern Optimizer opti;

}