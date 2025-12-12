#include <iostream>


#include "layer/archlist.hpp"
#include <ops/optim.hpp>

using namespace flow;

Device devc = Device::CUDA;
void test_att( int H, int B, int S, int D)
{
    Tensor X = Tensor::randn(Shape({B, S, D}), TensorOptions().with_device(devc));
    ag::debug::print_tensor("Input", X);

    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));

    auto atten = flow::Attention(B, S, D, H, devc);
    auto r = atten(m);

    
    auto w = sum(r);
    ag::debug::print_tensor("Result Value Attention", r.val());
    backward(w);
    ag::debug::print_tensor("Result Gradient Attention", m.grad());
         flow::opti.SGD(w, 0.001);
    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
        opti.epoch();
    }

}





void test_aliatt( int H, int B, int S, int D)
{
    Tensor X = Tensor::randn(Shape({B, S, D}), TensorOptions().with_device(devc));
    ag::debug::print_tensor("Input Alibi Attention", X);
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));
    auto atten = flow::AlibiAttention(B, S, D, H, devc);
    auto r = atten(m);

    
    auto w = sum(r);
    ag::debug::print_tensor("Result Value Alibi Attention", r.val());
    backward(w);
    ag::debug::print_tensor("Result Gradient Alibi Attention", m.grad());
    flow::opti.SGD(w, 0.01);
      

    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
        opti.epoch();
    }




}



void test_parcon( int H, int B, int S, int D)
{
    Tensor X = Tensor::randn(Shape({B, S, D}), TensorOptions().with_device(devc));
    ag::debug::print_tensor("Input Parametric Cone", X);
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));

    auto swag = flow::ParCon();
    auto w = swag(m);
    
    // auto w = sum(r);
    ag::debug::print_tensor("Result Value Parametric Cone", w.val());
    backward(w);
    ag::debug::print_tensor("Result Gradient Parametric Cone", m.grad());
                    flow::opti.SGD(w, 0.001);

    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
        opti.epoch();
    }
}


void test_mish( int H, int B, int S, int D)
{
    Tensor X = Tensor::randn(Shape({B, S, D}), TensorOptions().with_device(devc));
    ag::debug::print_tensor("Input Mish", X);
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));

    auto swag = flow::Mish();
    auto w = swag(m);
    
    // auto w = sum(r);
    ag::debug::print_tensor("Result Value Mish", w.val());
    backward(w);
    ag::debug::print_tensor("Result Gradient Mish", m.grad());
                    flow::opti.SGD(w, 0.001);

    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
        opti.epoch();
    }
}


void test_gaus( int H, int B, int S, int D)
{
    Tensor X = Tensor::randn(Shape({B, S, D}), TensorOptions().with_device(devc));
    ag::debug::print_tensor("Input Gauss", X);
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));

    auto swag = flow::Gauss();
    auto w = swag(m);
    
    // auto w = sum(r);
    ag::debug::print_tensor("Result Value Gauss", w.val());
    backward(w);
    ag::debug::print_tensor("Result Gradient Gauss", m.grad());
                    flow::opti.SGD(w, 0.001);

    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
        opti.epoch();
    }
}


void test_silu( int H, int B, int S, int D)
{
    Tensor X = Tensor::randn(Shape({B, S, D}), TensorOptions().with_device(devc));
    ag::debug::print_tensor("Input Silu", X);
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));

    auto swag = flow::Swish();
    auto w = swag(m);
    
    // auto w = sum(r);
    ag::debug::print_tensor("Result Value Silu", w.val());
    backward(w);
    ag::debug::print_tensor("Result Gradient Silu", m.grad());
                    flow::opti.SGD(w, 0.001);

    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
        opti.epoch();
    }
}


void test_gcu( int H, int B, int S, int D)
{
    Tensor X = Tensor::randn(Shape({B, S, D}), TensorOptions().with_device(devc));
    ag::debug::print_tensor("Input GCU", X);
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));

    auto swag = flow::GCU();
    auto w = swag(m);
    
    // auto w = sum(r);
    ag::debug::print_tensor("Result Value GCU", w.val());
    backward(w);
    ag::debug::print_tensor("Result Gradient GCU", m.grad());
    flow::opti.SGD(w, 0.001);

    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
        opti.epoch();
    }
}

void test_gelu( int H, int B, int S, int D)
{
    Tensor X = Tensor::randn(Shape({B, S, D}), TensorOptions().with_device(devc));
    ag::debug::print_tensor("Input GELU", X);
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));

    auto swag = flow::GELU();
    auto w = swag(m);
    
    // auto w = sum(r);
    ag::debug::print_tensor("Result Value GELU", w.val());
    backward(w);
    ag::debug::print_tensor("Result Gradient GELU", m.grad());
    flow::opti.SGD(w, 0.001);
    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
        opti.epoch();
    }
}

void test_sigmoid( int H, int B, int S, int D)
{
    Tensor X = Tensor::randn(Shape({B, S, D}), TensorOptions().with_device(devc));
    ag::debug::print_tensor("Input Sigmoid", X);
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));

    auto swag = flow::Sigmoid();
    auto w = swag(m);
    
    // auto w = sum(r);
    ag::debug::print_tensor("Result Value Sigmoid", w.val());
    backward(w);
    ag::debug::print_tensor("Result Gradient Sigmoid", m.grad());
    flow::opti.SGD(w, 0.001);
    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
        opti.epoch();
    }
}


void test_lisht( int H, int B, int S, int D)
{
    Tensor X = Tensor::randn(Shape({B, S, D}), TensorOptions().with_device(devc));
    ag::debug::print_tensor("Input Lisht", X);
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));

    auto swag = flow::LiSHT();
    auto w = swag(m);
    
    // auto w = sum(r);
    ag::debug::print_tensor("Result Value Lisht", w.val());
    backward(w);
    ag::debug::print_tensor("Result Gradient Lisht", m.grad());
    flow::opti.SGD(w, 0.001);
    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
        opti.epoch();
    }
}


void test_relu( int H, int B, int S, int D)
{
    Tensor X = Tensor::randn(Shape({B, S, D}), TensorOptions().with_device(devc));
    ag::debug::print_tensor("Input RELU", X);
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));


    auto swag = flow::ReLU();
    auto w = swag(m);
    
    // auto w = sum(r);
    ag::debug::print_tensor("Result Value RELU", w.val());
    backward(w);
    ag::debug::print_tensor("Result Gradient RELU", m.grad());
    flow::opti.SGD(w, 0.001);
    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
        opti.epoch();
    }
}



void test_tanh( int H, int B, int S, int D)
{
    Tensor X = Tensor::randn(Shape({B, S, D}), TensorOptions().with_device(devc));
    ag::debug::print_tensor("Input Tanh", X);
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));

    auto swag = flow::Tanh();
    auto w = swag(m);
    
    // auto w = sum(r);
    ag::debug::print_tensor("Result Value Tanh", w.val());
    backward(w);
    ag::debug::print_tensor("Result Gradient Tanh", m.grad());
    flow::opti.SGD(w, 0.001);
    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
        opti.epoch();
    }
}


void test_softplus( int H, int B, int S, int D)
{
    Tensor X = Tensor::randn(Shape({B, S, D}), TensorOptions().with_device(devc));
    ag::debug::print_tensor("Input Softplus", X);
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));

    auto swag = flow::Softplus();
    auto w = swag(m);
    
    // auto w = sum(r);
    ag::debug::print_tensor("Result Value Softplus", w.val());
    backward(w);
    ag::debug::print_tensor("Result Gradient Softplus", m.grad());
    flow::opti.SGD(w, 0.001);
    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
        opti.epoch();
    }
}


void test_leakyrelu( int H, int B, int S, int D)
{
    Tensor X = Tensor::randn(Shape({B, S, D}), TensorOptions().with_device(devc));
    ag::debug::print_tensor("Input LeakyRelu", X);
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));

    auto swag = flow::LeakyReLU(0.1);
    auto w = swag(m);
    
    // auto w = sum(r);
    ag::debug::print_tensor("Result Value LeakyRelu", w.val());
    backward(w);
    ag::debug::print_tensor("Result Gradient LeakyRelu", m.grad());
    flow::opti.SGD(w, 0.001);
    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
        opti.epoch();
    }
}



void test_swiglu( int H, int B, int S, int D, int K)
{
    Tensor X = Tensor::randn(Shape({B, S, D}), TensorOptions().with_device(devc));
    ag::debug::print_tensor("Input SWIGLU", X);
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));
    auto swag = flow::SWIGLU(B, S, D, K, Device::CUDA);
    auto r = swag(m);

    auto w = sum(r);
    ag::debug::print_tensor("Result Value SWIGLU", r.val());
    backward(w);
    ag::debug::print_tensor("Result Gradient SWIGLU", m.grad());
    flow::opti.SGD(w, 0.001);

    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
        opti.epoch();
    }



}

void test_rmsnorm( int H, int B, int S, int D)
{
    Tensor X = Tensor::randn(Shape({B, S, D}), TensorOptions().with_device(devc));
    ag::debug::print_tensor("Input RMSNorm", X);
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));

    auto swag = flow::RMSNorm();
    auto w = swag(m);
    
    // auto w = sum(r);
    ag::debug::print_tensor("Result Value RMSNorm", w.val());
    backward(w);
    ag::debug::print_tensor("Result Gradient RMSNorm", m.grad());
    flow::opti.SGD(w, 0.001);
    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
        opti.epoch();
    }
}

void test_laynorm( int H, int B, int S, int D)
{
    Tensor X = Tensor::randn(Shape({B, S, D}), TensorOptions().with_device(devc));
    ag::debug::print_tensor("Input Layer Norm", X);
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));

    auto swag = flow::LayerNorm();
    auto w = swag(m);
    
    // auto w = sum(r);
    ag::debug::print_tensor("Result Value Layer Norm", w.val());
    backward(w);
    ag::debug::print_tensor("Result Gradient Layer Norm", m.grad());
    flow::opti.SGD(w, 0.001);
    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
        opti.epoch();
    }
}

void test_dyntanh( int H, int B, int S, int D)
{
    Tensor X = Tensor::randn(Shape({B, S, D}), TensorOptions().with_device(devc));
    ag::debug::print_tensor("Input DynTanh", X);
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));

    auto swag = flow::DynTanh();
    auto w = swag(m);
    
    // auto w = sum(r);
    ag::debug::print_tensor("Result Value DynTanh", w.val());
    backward(w);
    ag::debug::print_tensor("Result Gradient DynTanh", m.grad());
    flow::opti.SGD(w, 0.001);
    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
        opti.epoch();
    }
}




int main(){


// test_expand(2, 2, 4, 16);

// test_aliatt(16, 8, 128, 256);
// test_aliatt(2, 4, 64, 1024);
// test_aliatt(2, 4, 64, 2048);
// test_aliatt(2, 4, 128, 1024);
// test_aliatt(8, 16, 2048, 1024);

test_parcon(2, 4, 2, 4);
test_mish(2, 4, 2, 4);
test_gaus(2, 4, 2, 4);
test_silu(2, 4, 2, 4);

test_gcu(2, 4, 2, 4);
test_gelu(2, 4, 2, 4);
test_sigmoid(2, 4, 2, 4);

test_lisht(2, 4, 2, 4);
test_relu(2, 4, 2, 4);
test_softplus(2, 4, 2, 4);
test_tanh(2, 4, 2, 4);
test_leakyrelu(2, 4, 2, 4);

test_att(2, 4, 2, 4);
test_aliatt(2, 4, 2, 4);
test_swiglu(2, 4, 2, 4, 10);
test_rmsnorm(2, 4, 2, 4);
test_laynorm(2, 4, 2, 4);
test_dyntanh(2, 4, 2, 4);
return 0;

}