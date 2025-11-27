
#include <iostream>

#include "ad/ag_all.hpp"
#include <optim.hpp>
#include <iomanip>
using namespace ag;


void test_aliatt( int Heads, int B, int S, int d_model, int K, int num_layers)
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
            new ag::layer::DynTanh(),
            new ag::layer::AlibiAttention(B, S, d_model, Heads)
        }));

        layers.push_back(new ag::layer::ResidualBlock({
            new ag::layer::DynTanh(),
            new ag::layer::Mish()
        }));
    }

    layers.push_back(new ag::layer::DynTanh());
    layers.push_back(new ag::layer::Linear(B, S, d_model));

    ag::layer::Traverse modela(layers);

    std::cout << "Model created with " << modela.parameters().size()
              << " parameter tensors.\n\n";




    auto r = modela(m);

        ag::Value labels = ag::make_tensor(OwnTensor::Tensor::randn(r.val().shape(), TensorOptions()), "labels");





    
        
    auto w = cross_entropy_with_logits(r, labels);
    ag::debug::print_tensor("Result Value Alibi Attention", r.val());
    backward(w);
    ag::debug::print_tensor("Result Gradient Alibi Attention", m.grad());

        double initial_loss = -1.0;
    double final_loss = -1.0;


    for (int epoch = 0; epoch < 11; ++epoch) {
        // a. Zero out all gradients from the previous iteration.
        zero_grad(w);

        // b. Forward pass: call our raw function.
        forward(w);

        // c. Compute the loss.
        
        // d. Backward pass: compute gradients for all parameters.
        backward(w);

        // e. Optimizer step: update all parameters using their gradients.
        ag::SGD(w);

        double current_loss = w.val().to_cpu().data<float>()[0];
        if (epoch == 0) initial_loss = current_loss;
        final_loss = current_loss;

        std::cout << "Epoch " << epoch << ", Loss: " << std::fixed << std::setprecision(4) << current_loss << std::endl;
    }






}




int main(){

test_aliatt(2, 4, 2, 4, 3, 10);

return 0;

}