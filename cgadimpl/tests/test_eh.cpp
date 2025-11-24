#include <iostream>

#include "ad/ag_all.hpp"
#include <optim.hpp>
using namespace ag;
void test_expand( int H, int B, int S, int D)
{
    Tensor X = Tensor::ones(Shape({B, S, D}), TensorOptions());
    ag::debug::print_tensor("Input", X);
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));
    auto r = expand(m, H);
    auto w = sum(r);
    ag::debug::print_tensor("Result", r.val());
    backward(w);
    ag::debug::print_tensor("Result", m.grad());
    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
        ag::SGD(w);
    }



}

void test_att( int H, int B, int S, int D)
{
    Tensor X = Tensor::ones(Shape({B, S, D}), TensorOptions());
    ag::debug::print_tensor("Input", X);
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));
    // auto atten = ag::layer::Attention(B, S, D, H);
    // auto r = atten(m);
    auto in_features = S;
    auto batch = B;
    auto out_features = D;

        float scale = sqrtf(0.02f / in_features);
    Tensor wq = OwnTensor::Tensor::randn(Shape{{batch, in_features, out_features}}, TensorOptions()) * scale;
    Tensor wk = OwnTensor::Tensor::randn(Shape{{batch, in_features, out_features}}, TensorOptions()) * scale;
    Tensor wv = OwnTensor::Tensor::randn(Shape{{batch, in_features, out_features}}, TensorOptions()) * scale;

    auto Q = make_tensor(wq, "q");
    auto K = make_tensor(wk, "k");
    auto V = make_tensor(wv, "v");
    auto r = attention(m, Q, K, V, H);
    
    auto w = sum(r);
    ag::debug::print_tensor("Result", r.val());
    backward(w);
    ag::debug::print_tensor("Result", m.grad());
    // for(int i=0;i<10;i++){
    //     forward(w);
    //     backward(w);
    //     ag::SGD(w);
    // }



}
int main(){


// test_expand(2, 4, 8, 16);
test_att(2, 4, 64, 256);
return 0;

}