
#include <iostream>

#include "ad/ag_all.hpp"
#include <optim.hpp>
#include <iomanip>
using namespace ag;


void test_aliatt( int Heads, int B, int S, int d_model, int K, int num_layers)
{

    // auto dev = Device::CUDA;
    // Tensor X = Tensor::randn(Shape({B, S, d_model}), TensorOptions().with_device(dev));
    // ag::debug::print_tensor("Input Alibi Attention", X);
    // auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));  
    // std::vector<ag::layer::Layer*> layers;
    // layers.reserve(num_layers * 2 + 2);

    // // Build model layers
    // for (int i = 0; i < num_layers; ++i) {
    //     layers.push_back(new ag::layer::ResidualBlock({
    //         new ag::layer::RMSNorm(),
    //         new ag::layer::AlibiAttention(B, S, d_model, Heads, dev)
    //     }));

    //     layers.push_back(new ag::layer::ResidualBlock({
    //         new ag::layer::RMSNorm(),
    //         new ag::layer::SWIGLU(B, S, d_model, K, dev)
    //         // new ag::layer::Mish()
    //     }));
    // }

    // layers.push_back(new ag::layer::RMSNorm());
    // layers.push_back(new ag::layer::Linear(B, S, d_model, dev));

    // ag::layer::Traverse modela(layers);

    // std::cout << "Model created with " << modela.parameters().size()
    //           << " parameter tensors.\n\n";




    // auto r = modela(m);

    //     ag::Value labels = ag::make_tensor(OwnTensor::Tensor::randn(r.val().shape(), TensorOptions().with_device(dev)), "labels");


    auto cpu_opts = TensorOptions().with_device(Device::CPU);
    auto gpu_opts = TensorOptions().with_device(Device::CUDA);

    int H = Heads;

    int D = d_model;

    // A: (11,9,3) B: (9,14,4) C: (9,14,5) D: (9,14,6) -> output (11,14)
    Tensor a_cpu = Tensor::randn(Shape{{B,S,D}}, cpu_opts);
Tensor b_cpu = Tensor::randn(Shape{{D, D}}, cpu_opts) * (0.02f / sqrtf(D));
Tensor c_cpu = Tensor::randn(Shape{{D, D}}, cpu_opts) * (0.02f / sqrtf(D));
Tensor d_cpu = Tensor::randn(Shape{{D, D}}, cpu_opts) * (0.02f / sqrtf(D));


    // Build Q,K,V via matmul as original code


       // Build Q,K,V via matmul as original code


    // Tensor q_gpu = b_cpu.to(gpu_opts.device);
    // Tensor k_gpu = c_cpu.to(gpu_opts.device);
    // Tensor v_gpu = d_cpu.to(gpu_opts.device);


    auto zz = a_cpu.to_cuda();

    auto z = ag::Value(std::make_shared<ag::Node>(zz, ag::Op::Leaf, true, "X"));
    auto Q = ag::Value(std::make_shared<ag::Node>(b_cpu.to(gpu_opts.device), ag::Op::Leaf, true, "X"));
    auto Ka = ag::Value(std::make_shared<ag::Node>(c_cpu.to(gpu_opts.device), ag::Op::Leaf, true, "X"));
    auto V = ag::Value(std::make_shared<ag::Node>(d_cpu.to(gpu_opts.device), ag::Op::Leaf, true, "X"));

     
        auto noutam = alibiatt(realrms(z, 1.0), Q, Ka, V, H)+z;
    // Tensor out_gpu(ref.shape(), options(ref).with_device(gpu_opts.device));

    // // flash attention call (standard softmax)
    // K.flashali(q_gpu.data<float>(), k_gpu.data<float>(), v_gpu.data<float>(), out_gpu.data<float>(),
    //         /*batches=*/B, /*heads=*/H, /*M=*/S, /*N=*/D/H, ag::current_stream());
    // cudaDeviceSynchronize();
    // auto outa = out_gpu.to_cpu().transpose(1,2).flatten(2,3).clone();


    ag::debug::print_tensor("Input SWIGLU", noutam.val().to_cpu());
    // auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));
    // auto atten = ag::layer::Attention(B, S, D, H);
    // auto r = atten(m);
    auto in_features = S;
    auto batch = B;
    auto out_features = D;
    auto hidden_features = 5;

        float scale = sqrtf(0.02f / out_features);

            auto param_opts = OwnTensor::TensorOptions().with_device(Device::CUDA).with_device(Device::CPU).with_req_grad(true);


    Tensor w_tensor = OwnTensor::Tensor::randn(Shape{{hidden_features, out_features}}, TensorOptions().with_device(Device::CUDA)) *sqrtf(0.02f / D);
    Tensor b_tensor = OwnTensor::Tensor::zeros(Shape{{1, hidden_features}}, TensorOptions().with_device(Device::CUDA));
    Tensor wa_tensor = OwnTensor::Tensor::randn(Shape{{hidden_features, out_features}}, TensorOptions().with_device(Device::CUDA)) *sqrtf(0.02f  / D);
    Tensor ba_tensor = OwnTensor::Tensor::zeros(Shape{{1, hidden_features}}, TensorOptions().with_device(Device::CUDA));
    Tensor wc_tensor = OwnTensor::Tensor::randn(Shape{{out_features, hidden_features}}, TensorOptions().with_device(Device::CUDA)) *sqrtf(0.02f / D);
    Tensor bc_tensor = OwnTensor::Tensor::zeros(Shape{{1, out_features}}, TensorOptions().with_device(Device::CUDA));

    auto W = make_tensor(w_tensor, "W");
    auto b = make_tensor(b_tensor, "b");
    auto Wa = make_tensor(wa_tensor, "Wa");
    auto ba = make_tensor(ba_tensor, "ba");
    auto Wc = make_tensor(wc_tensor, "Wc");
    auto bc = make_tensor(bc_tensor, "bc");

        Tensor wd_tensor = OwnTensor::Tensor::randn(Shape{{out_features, out_features}}, TensorOptions().with_device(Device::CUDA));
    Tensor bd_tensor = OwnTensor::Tensor::zeros(Shape{{1, out_features}}, TensorOptions().with_device(Device::CUDA));
    auto Wd = make_tensor(wd_tensor, "Wd");
    auto bd = make_tensor(bd_tensor, "bd");

    auto r = linear(realrms(linear(swiglu(realrms(noutam, 1.0), W, b, Wa, ba),Wc, bc)+noutam, 1.0), Wd, bd);

//     Tensor gy_cpu = Tensor::randn(Shape{{11,7,12}}, cpu_opts);
        ag::Value labels = ag::make_tensor(OwnTensor::Tensor::randn(r.val().shape(), TensorOptions().with_device(Device::CUDA)), "labels");

    


    
        
    auto w = sum(r);
    zero_val(w);
    forward(w);

    ag::debug::print_tensor("Result Value Alibi Attention", r.val());
    backward(w);
    ag::debug::print_tensor("Result Gradient Alibi Attention", z.grad());

        double initial_loss = -1.0;
    double final_loss = -1.0;
ag::opti.SGD(w, 0.01);

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
ag::opti.SGD(w, 0.00001);

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

    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
        
        opti.epoch();
    }



}



int main(){

test_aliatt(8, 4, 64, 128, 5, 11);
// test_att(4, 4, 64, 128); // working
// test_atte(2, 2, 8, 8, 2, 1);

return 0;

}