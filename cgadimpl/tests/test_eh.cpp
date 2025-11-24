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
int main(){


test_expand(2, 4, 8, 16);
return 0;

}