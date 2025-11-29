#include <iostream>

#include "ad/ag_all.hpp"
#include <optim.hpp>
using namespace ag;
// void test_expand( int H, int B, int S, int D)
// {
//     Tensor X = Tensor::randn(Shape({B, S, D}), TensorOptions().with_device(Device::CUDA));
//     ag::debug::print_tensor("Input", X);
//     auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));
//     auto r = expand(m, H);
//     auto w = sum(r);
//     ag::debug::print_tensor("Result Value", r.val());
//     backward(w);
//     ag::debug::print_tensor("Result Gradient", m.grad());
//     for(int i=0;i<10;i++){
//         forward(w);
//         backward(w);
//                 ag::opti.SGD(w, 0.001);
//         opti.epoch();
//     }



// }

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
    Tensor wk = OwnTensor::Tensor::randn(Shape{{batch, out_features, out_features}}, param_opts) * scale;
    Tensor wv = OwnTensor::Tensor::randn(Shape{{batch, out_features, out_features}}, param_opts) * scale;

    auto Q = make_tensor(wq, "q");
    auto K = make_tensor(wk, "k");
    auto V = make_tensor(wv, "v");
    auto r = attention(m, Q, K, V, H);
    
    auto w = sum(r);
    ag::debug::print_tensor("Result Value Attention", r.val().to_cpu());
    backward(w);
    ag::debug::print_tensor("Result Gradient Attention", m.grad().to_cpu());
    // for(int i=0;i<10;i++){
    //     forward(w);
    //     // backward(w);
    //     ag::opti.SGD(w, 0.001);
    //     opti.epoch();
    // }



}





void test_aliatt( int H, int B, int S, int D)
{
    Tensor X = Tensor::randn(Shape({B, S, D}), TensorOptions().with_device(Device::CUDA));
    ag::debug::print_tensor("Input Alibi Attention", X);
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));
    // auto atten = ag::layer::Attention(B, S, D, H);
    // auto r = atten(m);
    auto in_features = S;
    auto batch = B;
    auto out_features = D;
        auto param_opts = OwnTensor::TensorOptions().with_device(Device::CUDA).with_device(Device::CPU).with_req_grad(true);


        float scale = sqrtf(0.02f / out_features);
    // Tensor wq = OwnTensor::Tensor::randn(Shape{{batch, out_features, out_features}}, param_opts) * scale;
    // Tensor wk = OwnTensor::Tensor::randn(Shape{{batch, out_features, out_features}}, param_opts) * scale;
    // Tensor wv = OwnTensor::Tensor::randn(Shape{{batch, out_features, out_features}}, param_opts) * scale;

    auto Q = ag::Value(std::make_shared<ag::Node>(Tensor::randn(Shape({B, D, D}), TensorOptions().with_device(Device::CUDA))* scale, ag::Op::Leaf, true, "X"));
    auto K = ag::Value(std::make_shared<ag::Node>(Tensor::randn(Shape({B, D, D}), TensorOptions().with_device(Device::CUDA))* scale, ag::Op::Leaf, true, "X"));
    auto V = ag::Value(std::make_shared<ag::Node>(Tensor::randn(Shape({B, D, D}), TensorOptions().with_device(Device::CUDA))* scale, ag::Op::Leaf, true, "X"));
    auto r = alibiatt(m, Q, K, V, H);
    
    auto w = sum(r);
    ag::debug::print_tensor("Result Value Alibi Attention", r.val());
    backward(w);
    ag::debug::print_tensor("Result Gradient Alibi Attention", m.grad());




}



void test_parcon( int H, int B, int S, int D)
{
    Tensor X = Tensor::randn(Shape({B, S, D}), TensorOptions().with_device(Device::CUDA).with_device(Device::CUDA));
    ag::debug::print_tensor("Input Parametric Cone", X);
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));

    auto in_features = S;
    auto batch = B;
    auto out_features = D;

    auto w = parcon(m);
    
    // auto w = sum(r);
    ag::debug::print_tensor("Result Value Parametric Cone", w.val().to_cpu());
    backward(w);
    ag::debug::print_tensor("Result Gradient Parametric Cone", m.grad().to_cpu());
    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
                ag::opti.SGD(w, 0.001);
        opti.epoch();
    }
}


void test_mish( int H, int B, int S, int D)
{
    Tensor X = Tensor::randn(Shape({B, S, D}), TensorOptions().with_device(Device::CUDA).with_device(Device::CUDA));
    ag::debug::print_tensor("Input Mish", X);
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));

    auto in_features = S;
    auto batch = B;
    auto out_features = D;

    auto w = mish(m);
    
    // auto w = sum(r);
    ag::debug::print_tensor("Result Value Mish", w.val());
    backward(w);
    ag::debug::print_tensor("Result Gradient Mish", m.grad());
    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
                ag::opti.SGDm(w, 0.01, 0.0001);
        opti.epoch();
    }
}


void test_gaus( int H, int B, int S, int D)
{
    Tensor X = Tensor::randn(Shape({B, S, D}), TensorOptions().with_device(Device::CUDA));
    ag::debug::print_tensor("Input Gauss", X);
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));

    auto in_features = S;
    auto batch = B;
    auto out_features = D;

    auto w = gaus(m);
    
    // auto w = sum(r);
    ag::debug::print_tensor("Result Value Gauss", w.val());
    backward(w);
    ag::debug::print_tensor("Result Gradient Gauss", m.grad());
    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
                ag::opti.SGD(w, 0.001);
        opti.epoch();
    }
}


void test_silu( int H, int B, int S, int D)
{
    Tensor X = Tensor::randn(Shape({B, S, D}), TensorOptions().with_device(Device::CUDA));
    ag::debug::print_tensor("Input Silu", X);
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));

    auto in_features = S;
    auto batch = B;
    auto out_features = D;

    auto w = silu(m);
    
    // auto w = sum(r);
    ag::debug::print_tensor("Result Value Silu", w.val());
    backward(w);
    ag::debug::print_tensor("Result Gradient Silu", m.grad());
    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
                ag::opti.SGD(w, 0.001);
        opti.epoch();
    }
}


void test_gcu( int H, int B, int S, int D)
{
    Tensor X = Tensor::randn(Shape({B, S, D}), TensorOptions().with_device(Device::CUDA));
    ag::debug::print_tensor("Input GCU", X);
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));

    auto in_features = S;
    auto batch = B;
    auto out_features = D;

    auto w = gcu(m);
    
    // auto w = sum(r);
    ag::debug::print_tensor("Result Value GCU", w.val());
    backward(w);
    ag::debug::print_tensor("Result Gradient GCU", m.grad());
    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
                ag::opti.SGD(w, 0.001);
        opti.epoch();
    }
}

void test_gelu( int H, int B, int S, int D)
{
    Tensor X = Tensor::randn(Shape({B, S, D}), TensorOptions().with_device(Device::CUDA));
    ag::debug::print_tensor("Input GELU", X);
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));

    auto in_features = S;
    auto batch = B;
    auto out_features = D;

    auto w = gelu(m);
    
    // auto w = sum(r);
    ag::debug::print_tensor("Result Value GELU", w.val());
    backward(w);
    ag::debug::print_tensor("Result Gradient GELU", m.grad());
    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
                ag::opti.SGD(w, 0.001);
        opti.epoch();
    }
}

void test_sigmoid( int H, int B, int S, int D)
{
    Tensor X = Tensor::randn(Shape({B, S, D}), TensorOptions().with_device(Device::CUDA));
    ag::debug::print_tensor("Input Sigmoid", X);
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));

    auto in_features = S;
    auto batch = B;
    auto out_features = D;

    auto w = sigmoid(m);
    
    // auto w = sum(r);
    ag::debug::print_tensor("Result Value Sigmoid", w.val());
    backward(w);
    ag::debug::print_tensor("Result Gradient Sigmoid", m.grad());
    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
                ag::opti.SGD(w, 0.001);
        opti.epoch();
    }
}


void test_lisht( int H, int B, int S, int D)
{
    Tensor X = Tensor::randn(Shape({B, S, D}), TensorOptions().with_device(Device::CUDA));
    ag::debug::print_tensor("Input Lisht", X);
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));

    auto in_features = S;
    auto batch = B;
    auto out_features = D;

    auto w = lisht(m);
    
    // auto w = sum(r);
    ag::debug::print_tensor("Result Value Lisht", w.val());
    backward(w);
    ag::debug::print_tensor("Result Gradient Lisht", m.grad());
    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
                ag::opti.SGD(w, 0.001);
        opti.epoch();
    }
}


void test_relu( int H, int B, int S, int D)
{
    Tensor X = Tensor::randn(Shape({B, S, D}), TensorOptions().with_device(Device::CUDA));
    ag::debug::print_tensor("Input RELU", X);
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));

    auto in_features = S;
    auto batch = B;
    auto out_features = D;

    auto w = relu(m);
    
    // auto w = sum(r);
    ag::debug::print_tensor("Result Value RELU", w.val());
    backward(w);
    ag::debug::print_tensor("Result Gradient RELU", m.grad());
    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
                ag::opti.SGD(w, 0.001);
        opti.epoch();
    }
}



void test_tanh( int H, int B, int S, int D)
{
    Tensor X = Tensor::randn(Shape({B, S, D}), TensorOptions().with_device(Device::CUDA));
    ag::debug::print_tensor("Input Tanh", X);
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));

    auto in_features = S;
    auto batch = B;
    auto out_features = D;

    auto w = tanh(m);
    
    // auto w = sum(r);
    ag::debug::print_tensor("Result Value Tanh", w.val());
    backward(w);
    ag::debug::print_tensor("Result Gradient Tanh", m.grad());
    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
                ag::opti.SGD(w, 0.001);
        opti.epoch();
    }
}


void test_softplus( int H, int B, int S, int D)
{
    Tensor X = Tensor::randn(Shape({B, S, D}), TensorOptions().with_device(Device::CUDA));
    ag::debug::print_tensor("Input Softplus", X);
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));

    auto in_features = S;
    auto batch = B;
    auto out_features = D;

    auto w = softplus(m);
    
    // auto w = sum(r);
    ag::debug::print_tensor("Result Value Softplus", w.val());
    backward(w);
    ag::debug::print_tensor("Result Gradient Softplus", m.grad());
    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
                ag::opti.SGD(w, 0.001);
        opti.epoch();
    }
}


void test_leakyrelu( int H, int B, int S, int D)
{
    Tensor X = Tensor::randn(Shape({B, S, D}), TensorOptions().with_device(Device::CUDA));
    ag::debug::print_tensor("Input LeakyRelu", X);
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));

    auto in_features = S;
    auto batch = B;
    auto out_features = D;
    auto alpha = 0.1f;

    auto w = leaky_relu(m, alpha);
    
    // auto w = sum(r);
    ag::debug::print_tensor("Result Value LeakyRelu", w.val());
    backward(w);
    ag::debug::print_tensor("Result Gradient LeakyRelu", m.grad());
    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
                ag::opti.SGD(w, 0.001);
        opti.epoch();
    }
}



void test_swiglu( int H, int B, int S, int D, int K)
{
    Tensor X = Tensor::randn(Shape({B, S, D}), TensorOptions().with_device(Device::CUDA));
    ag::debug::print_tensor("Input SWIGLU", X);
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));
    // auto atten = ag::layer::Attention(B, S, D, H);
    // auto r = atten(m);
    auto in_features = S;
    auto batch = B;
    auto out_features = D;
    auto hidden_features = K;

        float scale = sqrtf(0.02f / in_features);

            auto param_opts = OwnTensor::TensorOptions().with_device(Device::CUDA).with_device(Device::CPU).with_req_grad(true);


    Tensor w_tensor = OwnTensor::Tensor::randn(Shape{{batch, hidden_features, out_features}}, TensorOptions().with_device(Device::CUDA)) * scale;
    Tensor b_tensor = OwnTensor::Tensor::zeros(Shape{{batch, 1, hidden_features}}, TensorOptions().with_device(Device::CUDA));
    Tensor wa_tensor = OwnTensor::Tensor::randn(Shape{{batch, hidden_features, out_features}}, TensorOptions().with_device(Device::CUDA)) * scale;
    Tensor ba_tensor = OwnTensor::Tensor::zeros(Shape{{batch, 1, hidden_features}}, TensorOptions().with_device(Device::CUDA));
    Tensor wc_tensor = OwnTensor::Tensor::randn(Shape{{batch, out_features, hidden_features}}, TensorOptions().with_device(Device::CUDA)) * scale;
    Tensor bc_tensor = OwnTensor::Tensor::zeros(Shape{{batch, 1, out_features}}, TensorOptions().with_device(Device::CUDA));

    auto W = make_tensor(w_tensor, "W");
    auto b = make_tensor(b_tensor, "b");
    auto Wa = make_tensor(wa_tensor, "Wa");
    auto ba = make_tensor(ba_tensor, "ba");
    auto Wc = make_tensor(wc_tensor, "Wc");
    auto bc = make_tensor(bc_tensor, "bc");

    ag::debug::print_tensor("Weight one", w_tensor);
    ag::debug::print_tensor("Weight onea", wa_tensor);
    


    auto r = swiglu(m, W, b, Wa, ba);
    
    auto w = sum(r);
    ag::debug::print_tensor("Result Value SWIGLU", r.val());
    backward(w);
    ag::debug::print_tensor("Result Gradient SWIGLU", m.grad());
    // for(int i=0;i<10;i++){
    //     forward(w);
    //     backward(w);
    //             ag::opti.SGD(w, 0.001);
        // opti.epoch();
    // }



}


void test_rmsnorm( int H, int B, int S, int D)
{
    Tensor X = Tensor::randn(Shape({B, S, D}), TensorOptions().with_device(Device::CUDA));
    ag::debug::print_tensor("Input RMSNorm", X);
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));

    // auto swag = ag::layer::RMSNorm();
    auto w = realrms(m, 1.0);
    
    // auto w = sum(r);
    ag::debug::print_tensor("Result Value RMSNorm", w.val());
    backward(w);
    ag::debug::print_tensor("Result Gradient RMSNorm", m.grad());
    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
                ag::opti.SGD(w, 0.001);
        opti.epoch();
    }
ag::debug::print_tensor("Result Gradient1 RMSNorm", w.node->inputs[1]->grad.to_cpu().to_cpu());
}


void test_laynorm( int H, int B, int S, int D)
{
    Tensor X = Tensor::randn(Shape({B, S, D}), TensorOptions().with_device(Device::CUDA));
    ag::debug::print_tensor("Input Layer Norm", X);
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));

    // auto swag = ag::layer::RMSNorm();
    auto w = relaynor(m, 0.0, 1.0);
    
    // auto w = sum(r);
    ag::debug::print_tensor("Result Value Layer Norm", w.val());
    backward(w);
    ag::debug::print_tensor("Result Gradient Layer Norm", m.grad());
    // for(int i=0;i<10;i++){
    //     forward(w);
    //     backward(w);
    //             ag::opti.SGD(w, 0.001);
        // opti.epoch();
    // }
ag::debug::print_tensor("Result Gradient Gamma Layer Norm", w.node->inputs[1]->grad.to_cpu());
ag::debug::print_tensor("Result Gradient Beta Layer Norm", w.node->inputs[2]->grad.to_cpu());
}



void test_dyntanh( int H, int B, int S, int D)
{
    Tensor X = Tensor::randn(Shape({B, S, D}), TensorOptions().with_device(Device::CUDA));
    ag::debug::print_tensor("Input DynTanh", X);
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));

    // auto swag = ag::layer::RMSNorm();
    auto w = dyntanh(m, 0.5, 0.0, 1.0);
    
    // auto w = sum(r);
    ag::debug::print_tensor("Result Value DynTanh", w.val());
    backward(w);
    ag::debug::print_tensor("Result Gradient DynTanh", m.grad());
    // for(int i=0;i<10;i++){
    //     forward(w);
    //     backward(w);
    //             ag::opti.SGD(w, 0.001);
        // opti.epoch();
    // }
ag::debug::print_tensor("Result Gradient Alpha DynTanh", w.node->inputs[1]->grad.to_cpu());
ag::debug::print_tensor("Result Gradient Gamma DynTanh", w.node->inputs[3]->grad.to_cpu());
ag::debug::print_tensor("Result Gradient Beta DynTanh", w.node->inputs[2]->grad.to_cpu());
}


int main(){


// test_expand(2, 2, 4, 16);

// test_aliatt(2, 4, 64, 512);
// test_aliatt(2, 4, 64, 1024);
// test_aliatt(2, 4, 64, 2048);
// test_aliatt(2, 4, 128, 1024);
// test_aliatt(8, 16, 2048, 1024);
try {
        #if defined(_WIN32)
            const char* plugin_path = "./agkernels_cuda.dll";
        #elif defined(__APPLE__)
            const char* plugin_path = "./libagkernels_cuda.dylib";
        #else
            const char* plugin_path = "/home/blubridge-034/Downloads/Newf/cgadimpl/cgadimpl/build/libagkernels_cuda.so";
        #endif

        std::cout << "Loading GPU plugin from: " << plugin_path << "\n";
        kernels::load_cuda_plugin(plugin_path);
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

// test_aliatt(2, 4, 32, 128);
test_att(2, 4, 2, 4);
test_swiglu(2, 4, 2, 4, 10);

test_rmsnorm(2, 4, 2, 4);
test_laynorm(2, 4, 2, 4);
test_dyntanh(2, 4, 2, 4);
}
catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
return 0;

}