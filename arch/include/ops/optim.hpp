// =====================
// file: cgadimpl/include/ag/optim.hpp (declarations only)
// =====================
#pragma once
#include "layer/archlist.hpp"
#include <math.h>
#include <unordered_set>

namespace flow {






    class Optimizer
    {

        private:
        static constexpr float _lr = 0.1f;
        static constexpr float _m = 0.9f;

        


struct Bag :  std::enable_shared_from_this<Bag> {

std::shared_ptr<ag::Node> noda;
std::shared_ptr<Tensor> velocia;
    Bag(const std::shared_ptr<ag::Node>& n_val) 
        : noda(n_val) ,       // Initialize the 'noda' member
          velocia(std::make_shared<Tensor>(
    Tensor::zeros(n_val->value.shape(),
                  TensorOptions().with_device(n_val->value.device()))
))
      // Initialize the 'velocia' member
    {
        // std::cout<<"I was called\n";
    }

// void desBag(){
// noda->value = noda->value.to(Device::CPU);
// noda->grad = noda->grad.to(Device::CPU);
// *velocia = velocia->to(Device::CPU);

// }

};

// Bag storestuf;
std::vector<std::shared_ptr<Bag>> bagstore;
std::map<Node*, std::shared_ptr<Tensor>> velsto;


 void topovel_from(Node* root){
    std::vector<Node*> order; order.reserve(256);
    std::unordered_set<Node*> vis; vis.reserve(256);
    std::function<void(Node*)> dfs = [&](Node* n){ 
        
        
        if(!n || vis.count(n)) 
        return; vis.insert(n); 
        for(auto& p : n->inputs) 
        dfs(p.get()); 
        if(n->op==Op::Leaf && n->requires_grad())
        {
        velsto.insert({n, std::make_shared<Tensor>(Tensor::zeros(n->value.shape(),
                   ag::options(n->value)))});
        }
    };
    dfs(root);
}


    

        public:
                void SGD(const Value& root, float learning_rate=_lr);
        void SGDm(const Value& root, float learning_rate=_lr, float momentum=_m);
        using EpochFn = std::function<void()>;

        EpochFn epoch;
        void velcle(){ 
            


            for (auto ww = velsto.begin(); ww
         != velsto.end(); ww++) 
    {

        *(ww->second) = (*(ww->second)).to_cpu();

        //  std::cout<<"I am here";

       
    };
        };

        ~Optimizer(){velcle();};
 
    };



extern Optimizer opti;

}