
#include <iostream>

#include "ad/ag_all.hpp"
#include <optim.hpp>
using namespace ag;


void test_aliatt( int Heads, int B, int S, int d_model, int num_layers)
{
    Tensor X = Tensor::randn(Shape({B, S, d_model}), TensorOptions());
    ag::debug::print_tensor("Input Alibi Attention", X);
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));

    auto dev = Device::CPU;
    
    std::vector<ag::layer::Layer*> layers;
    layers.reserve(num_layers * 2 + 2);

    // Build model layers
    for (int i = 0; i < num_layers; ++i) {
        layers.push_back(new ag::layer::ResidualBlock({
            new ag::layer::RMSNorm(),
            new ag::layer::AlibiAttention(B, S, d_model, Heads)
        }));

        layers.push_back(new ag::layer::ResidualBlock({
            new ag::layer::RMSNorm(),
            new ag::layer::SWIGLU(B, S, d_model)
        }));
    }

    layers.push_back(new ag::layer::RMSNorm());
    layers.push_back(new ag::layer::Linear(B, S, d_model));

    ag::layer::Traverse model(layers);

    std::cout << "Model created with " << model.parameters().size()
              << " parameter tensors.\n\n";



    auto r = model(m);


    
        
    auto w = sum(r);
    ag::debug::print_tensor("Result Value Alibi Attention", r.val());
    backward(w);
    ag::debug::print_tensor("Result Gradient Alibi Attention", m.grad());







}




int main(){

test_aliatt(2, 4, 2, 4, 3);

return 0;

}