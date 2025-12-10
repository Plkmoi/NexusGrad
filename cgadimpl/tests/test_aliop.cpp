
#include <iostream>

#include "ad/ag_all.hpp"
#include <optim.hpp>
#include <iomanip>
using namespace ag;


void test_aliatt( int Heads, int B, int S, int d_model, int K, int num_layers)
{

    auto dev = Device::CUDA;
    Tensor X = Tensor::randn(Shape({B, S, d_model}), TensorOptions().with_device(dev));
    ag::debug::print_tensor("Input Alibi Attention", X);
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));  
    std::vector<ag::layer::Layer*> layers;
    layers.reserve(num_layers * 2 + 2);

    // Build model layers
    for (int i = 0; i < num_layers; ++i) {
        layers.push_back(new ag::layer::ResidualBlock({
            new ag::layer::RMSNorm(),
            new ag::layer::AlibiAttention(B, S, d_model, Heads, dev)
        }));

        layers.push_back(new ag::layer::ResidualBlock({
            new ag::layer::RMSNorm(),
            new ag::layer::SWIGLU(B, S, d_model, K, dev)
            // new ag::layer::Mish()
        }));
    }

    layers.push_back(new ag::layer::RMSNorm());
    layers.push_back(new ag::layer::Linear(B, d_model, d_model, dev));

    ag::layer::Traverse modela(layers);

    std::cout << "Model created with " << modela.parameters().size()
              << " parameter tensors.\n\n";




    auto r = modela(m);

    Tensor labels =
    Tensor::zeros(Shape({B, S}), TensorOptions());

float* lbl = labels.data<float>();

for (int b = 0; b < B; ++b) {
    for (int s = 0; s < S; ++s) {
        lbl[b*S + s] = std::rand() % d_model;   // class index
    }
}
Tensor onehot =
    Tensor::zeros(r.val().shape(),
        TensorOptions());

float* hot = onehot.data<float>();
float* lbl2 = labels.data<float>();

for (int b = 0; b < r.val().shape().dims[0]; ++b) {
    for (int s = 0; s < r.val().shape().dims[1]; ++s) {

        int cls = (int)lbl2[b*S + s];

        // index in flat memory:
        // (b, s, cls) => b*(S*d_model) + s*(d_model) + cls
        hot[b * (r.val().shape().dims[1] * r.val().shape().dims[2]) + s * r.val().shape().dims[2] + cls] = 1.0f;
    }
}


        ag::Value labelsa = ag::make_tensor(onehot.to(dev), "labels");



    auto w = cross_entropy_with_logits(r, labelsa);
    zero_val(w);
    forward(w);

    ag::debug::print_tensor("Result Value Alibi Attention", r.val());
    backward(w);
    ag::debug::print_tensor("Result Gradient Alibi Attention", m.grad());

        double initial_loss = -1.0;
    double final_loss = -1.0;
ag::opti.SGDm(r, 0.003);

zero_val(w);
    for (int epoch = 0; epoch < 10; ++epoch) {
        // a. Zero out all gradients from the previous iteration.
        zero_grad(w);

        // b. Forward pass: call our raw function.
        forward(w);

        // c. Compute the loss.
        
        // d. Backward pass: compute gradients for all parameters.
        backward(w);

        // e. Optimizer step: update all parameters using their gradients.
        // ag::SGD(w);
        opti.epoch();

        double current_loss = w.val().to_cpu().data<float>()[0];
        if (epoch == 0) initial_loss = current_loss;
        final_loss = current_loss;

        std::cout << "Epoch " << epoch << ", Loss: " << std::fixed << std::setprecision(4) << current_loss << std::endl;
    }






}



void test_atte( int Heads, int B, int S, int d_model, int K, int num_layers)
{

    auto dev = Device::CUDA;
    Tensor X = Tensor::randn(Shape({B, S, d_model}), TensorOptions().with_device(dev));
    ag::debug::print_tensor("Input Attention", X);
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));  
    std::vector<ag::layer::Layer*> layers;
    layers.reserve(num_layers * 2 + 2);

    // Build model layers
    for (int i = 0; i < num_layers; ++i) {
        layers.push_back(new ag::layer::ResidualBlock({
            new ag::layer::DynTanh(),
            new ag::layer::Attention(B, S, d_model, Heads, dev)
        }));

        layers.push_back(new ag::layer::ResidualBlock({
            new ag::layer::DynTanh(),
            new ag::layer::SWIGLU(B, S, d_model, K, dev)
            // new ag::layer::Mish()
        }));
    }

    layers.push_back(new ag::layer::DynTanh());
    layers.push_back(new ag::layer::Linear(B, S, d_model, dev));

    ag::layer::Traverse modela(layers);

    std::cout << "Model created with " << modela.parameters().size()
              << " parameter tensors.\n\n";




    auto r = modela(m);

Tensor labels =
    Tensor::zeros(Shape({B, S}), TensorOptions());

float* lbl = labels.data<float>();

for (int b = 0; b < B; ++b) {
    for (int s = 0; s < S; ++s) {
        lbl[b*S + s] = std::rand() % d_model;   // class index
    }
}
Tensor onehot =
    Tensor::zeros(r.val().shape(),
        TensorOptions());

float* hot = onehot.data<float>();
float* lbl2 = labels.data<float>();

for (int b = 0; b < r.val().shape().dims[0]; ++b) {
    for (int s = 0; s < r.val().shape().dims[1]; ++s) {

        int cls = (int)lbl2[b*S + s];

        // index in flat memory:
        // (b, s, cls) => b*(S*d_model) + s*(d_model) + cls
        hot[b * (r.val().shape().dims[1] * r.val().shape().dims[2]) + s * r.val().shape().dims[2] + cls] = 1.0f;
    }
}


        ag::Value labelsa = ag::make_tensor(onehot.to(dev), "labels");





    
        
    auto w = cross_entropy_with_logits(r, labelsa);
    ag::debug::print_tensor("Result Value Attention", r.val());
    backward(w);
    ag::debug::print_tensor("Result Gradient Attention", m.grad());

        double initial_loss = -1.0;
    double final_loss = -1.0;
ag::opti.SGDm(w, 0.00001);

zero_val(w);
    for (int epoch = 0; epoch < 11; ++epoch) {
        // a. Zero out all gradients from the previous iteration.
        zero_grad(w);

        // b. Forward pass: call our raw function.
        forward(w);

        // c. Compute the loss.
        
        // d. Backward pass: compute gradients for all parameters.
        backward(w);

        // e. Optimizer step: update all parameters using their gradients.
        // ag::SGD(w);
        opti.epoch();

        double current_loss = w.val().to_cpu().data<float>()[0];
        if (epoch == 0) initial_loss = current_loss;
        final_loss = current_loss;

        std::cout << "Vanilla Epoch " << epoch << ", Loss: " << std::fixed << std::setprecision(4) << current_loss << std::endl;
    }





}

void test_att( int H, int B, int S, int D)
{
    Tensor X = Tensor::randn(Shape({B, S, D}), TensorOptions().with_device(Device::CUDA));
    ag::debug::print_tensor("Input Attention", X);
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));
    // auto atten = ag::layer::Attention(B, S, D, H);
    // auto r = atten(m);
    auto in_features = S;
    auto batch = B;
    auto out_features = D;

            auto param_opts = OwnTensor::TensorOptions().with_device(Device::CUDA).with_req_grad(true);


        float scale = sqrtf(0.02f / out_features);
    Tensor wq = OwnTensor::Tensor::randn(Shape{{batch, out_features, out_features}}, param_opts) * scale;
    Tensor wk = OwnTensor::Tensor::randn(Shape{{out_features, out_features}}, param_opts) * scale;
    Tensor wv = OwnTensor::Tensor::randn(Shape{{out_features, out_features}}, param_opts) * scale;

    auto Q = make_tensor(wq, "q");
    auto K = make_tensor(wk, "k");
    auto V = make_tensor(wv, "v");
    auto r = attention(m, Q, K, V, H);
    
    auto w = sum(r);
    ag::debug::print_tensor("Result Value Attention", r.val().to_cpu());
    backward(w);
    ag::debug::print_tensor("Result Gradient Attention", m.grad().to_cpu());
    ag::opti.SGD(w, 0.01);

    for(int i=0;i<100;i++){
        forward(w);
        backward(w);
        
        opti.epoch();
    }



}



int main(){

test_aliatt(8, 4, 128, 256, 5, 11);
// test_att(4, 4, 64, 128); // working
// test_atte(2, 2, 8, 8, 2, 1);

return 0;

}