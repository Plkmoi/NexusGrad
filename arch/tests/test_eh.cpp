#include <iostream>

#include "layer/archlist.hpp"
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
// ag::opti.SGD(w, 0.01);
//     for(int i=0;i<10;i++){
//         forward(w);
//         backward(w);
//                 
//         opti.epoch();
//     }


// Helper updated to use types from the correct namespaces
void check_tensors_close(const Tensor& a, const Tensor& b, const std::string& label, float epsilon = 1e-4f) {
    if (a.shape().dims != b.shape().dims) {
        throw std::runtime_error(label + ": Shape mismatch.");
    }
    Tensor a_cpu = a.to_cpu();
    Tensor b_cpu = b.to_cpu();
    const float* a_data = a_cpu.data<float>();
    const float* b_data = b_cpu.data<float>();

    int q = 0;
    float w = 0;

    for (size_t i = 0; i < a.numel(); ++i) {
        if (std::abs(a_data[i] - b_data[i]) > epsilon || std::isnan(a_data[i]) || std::isnan(b_data[i])) {
            std::cout << "FAIL: " << label << " mismatch at index " << i << "\n";

            q=1;
            w=w+std::abs(a_data[i] - b_data[i]);
        }
    }
    auto f = (w/a.numel());
    if(q>0 && (w/a.numel())>0.2)
    {
                    debug::print_tensor("Tensor A (ref)", a);
            debug::print_tensor("Tensor B (out)", b);
                    throw std::runtime_error("Tensor check failed for " + label +" mismatch value " + std::to_string(f));

    }

            
    debug::print_tensor("Tensor A (ref)", a);
            debug::print_tensor("Tensor B (out)", b);

    std::cout << "PASS: " << label << "\n";
}

// }

void test_att( int Ha, int Ba, int Sa, int Da)
{
        auto& K = kernels::cuda();
    auto cpu_opts = TensorOptions().with_device(Device::CPU);
    auto gpu_opts = TensorOptions().with_device(Device::CUDA);

    int H = 2;
    int B = 3;
    int S = 5;
    int D = 6;

    // A: (11,9,3) B: (9,14,4) C: (9,14,5) D: (9,14,6) -> output (11,14)
    Tensor a_cpu = Tensor::randn(Shape{{B,S,D}}, cpu_opts);
    Tensor b_cpu = Tensor::randn(Shape{{D,D}}, cpu_opts)*sqrtf(20.f / D)+0.1*(Tensor::ones(Shape{{D,D}}, cpu_opts));
    Tensor c_cpu = Tensor::randn(Shape{{D,D}}, cpu_opts)*sqrtf(20.f / D)+0.1*(Tensor::ones(Shape{{D,D}}, cpu_opts));
    Tensor d_cpu = Tensor::randn(Shape{{D,D}}, cpu_opts)*sqrtf(20.f / D)+0.1*(Tensor::ones(Shape{{D,D}}, cpu_opts));

    // Build Q,K,V via matmul as original code

\
       // Build Q,K,V via matmul as original code


    Tensor q = matmul(a_cpu, b_cpu.t()).unflatten(2, Shape({H, (b_cpu.shape().dims[1]/H)})).transpose(1,2).clone();
    Tensor k = matmul(a_cpu, c_cpu.t()).unflatten(2, Shape({H, (c_cpu.shape().dims[1]/H)})).transpose(1,2).clone();
    Tensor v = matmul(a_cpu, d_cpu.t()).unflatten(2, Shape({H, (d_cpu.shape().dims[1]/H)})).transpose(1,2).clone();


    float scale = 1.f / sqrtf(static_cast<float>(k.shape().dims.back()));
    Tensor g = matmul(q, k.t()) * scale;



    // Tensor bias = bias_cpu.to(logits.device());
    // Tensor g    = logits +bias_cpu;  // [H,T,T]

    // Re-implement softmax using OwnTensor ops
    Tensor max_val = reduce_max(g, {-1}, true);
    Tensor exp_g = exp(g - max_val);
    Tensor sum_exp_g = reduce_sum(exp_g, {-1}, true);
    Tensor s = exp_g / sum_exp_g;
    //ag::debug::print_tensor("Who is v", v);

    //ag::debug::print_tensor("S middle", s);

    auto ref = matmul(s, v);

    auto refa = ref.transpose(1,2).flatten(2,3).clone();

    // Tensor q_gpu = b_cpu.to(gpu_opts.device);
    // Tensor k_gpu = c_cpu.to(gpu_opts.device);
    // Tensor v_gpu = d_cpu.to(gpu_opts.device);


    auto zz = a_cpu.to_cuda();

    auto z = ag::Value(std::make_shared<ag::Node>(zz, ag::Op::Leaf, true, "X"));
    auto Q = ag::Value(std::make_shared<ag::Node>(b_cpu.to(gpu_opts.device), ag::Op::Leaf, true, "X"));
    auto Ka = ag::Value(std::make_shared<ag::Node>(c_cpu.to(gpu_opts.device), ag::Op::Leaf, true, "X"));
    auto V = ag::Value(std::make_shared<ag::Node>(d_cpu.to(gpu_opts.device), ag::Op::Leaf, true, "X"));

     
        auto outam = attention(z, Q, Ka, V, H);
    // Tensor out_gpu(ref.shape(), options(ref).with_device(gpu_opts.device));

    // // flash attention call (standard softmax)
    // K.flashali(q_gpu.data<float>(), k_gpu.data<float>(), v_gpu.data<float>(), out_gpu.data<float>(),
    //         /*batches=*/B, /*heads=*/H, /*M=*/S, /*N=*/D/H, ag::current_stream());
    // cudaDeviceSynchronize();
    // auto outa = out_gpu.to_cpu().transpose(1,2).flatten(2,3).clone();

    check_tensors_close(refa.to_cpu(), outam.val().to_cpu(), "test_gpu_attention", 0.01);

//     Tensor gy_cpu = Tensor::randn(Shape{{11,7,12}}, cpu_opts);

    
    auto w = sum(outam);
    ag::debug::print_tensor("Result Value Alibi Attention", outam.val().to_cpu());
    backward(w);
    ag::debug::print_tensor("Result Gradient Alibi Attention", z.grad().to_cpu());
    ag::debug::print_tensor("Result Gradient Alibi Attention", z.val().to_cpu());

        ag::debug::print_tensor("Q value", Q.val().to_cpu());
        ag::debug::print_tensor("Q grad", Q.grad().to_cpu());
        ag::debug::print_tensor("K value", Ka.val().to_cpu());
        ag::debug::print_tensor("K grad", Ka.grad().to_cpu());
        ag::debug::print_tensor("V value", V.val().to_cpu());
        ag::debug::print_tensor("V grad", V.grad().to_cpu());



    // auto cpu_opts = TensorOptions().with_device(Device::CPU);
    // auto gpu_opts = TensorOptions().with_device(Device::CUDA);

    // int H = 4;
    // int B = 11;
    // int S = 7;
    // int D = 12;

    // // A: (11,9,3) B: (9,14,4) C: (9,14,5) D: (9,14,6) -> output (11,14)
    // Tensor a_cpu = Tensor::randn(Shape{{B,S,D}}, cpu_opts);
    // Tensor b_cpu = Tensor::randn(Shape{{B,D,D}}, cpu_opts);
    // Tensor c_cpu = Tensor::randn(Shape{{B,D,D}}, cpu_opts);
    // Tensor d_cpu = Tensor::randn(Shape{{B,D,D}}, cpu_opts);

    // // Build Q,K,V via matmul as original code


    // Tensor q = matmul(a_cpu, b_cpu.t()).unflatten(2, Shape({H, (b_cpu.shape().dims[1]/H)})).transpose(1,2).clone();
    // Tensor k = matmul(a_cpu, c_cpu.t()).unflatten(2, Shape({H, (c_cpu.shape().dims[1]/H)})).transpose(1,2).clone();
    // Tensor v = matmul(a_cpu, d_cpu.t()).unflatten(2, Shape({H, (d_cpu.shape().dims[1]/H)})).transpose(1,2).clone();


    

    // float scale = 1.f / sqrtf(static_cast<float>(k.shape().dims.back()));
    // Tensor g = matmul(q, k.t()) * scale;

    // // Re-implement softmax using OwnTensor ops
    // Tensor max_val = reduce_max(g, {-1}, true);
    // Tensor exp_g = exp(g - max_val);
    // Tensor sum_exp_g = reduce_sum(exp_g, {-1}, true);
    // Tensor s = exp_g / sum_exp_g;
    // //ag::debug::print_tensor("Who is v", v);

    // //ag::debug::print_tensor("S middle", s);

    // auto ref = matmul(s, v);


    // Tensor q_gpu = q.to(gpu_opts.device);
    // Tensor k_gpu = k.to(gpu_opts.device);
    // Tensor v_gpu = v.to(gpu_opts.device);
    // Tensor out_gpu(Shape({/*batches=*/a_cpu.shape().dims[0], /*heads=*/H, /*M=*/a_cpu.shape().dims[1], /*N=*/a_cpu.shape().dims[2]/H}), options(ref).with_device(gpu_opts.device));

    // // flash attention call (standard softmax)
    // K.flash(q_gpu.data<float>(), k_gpu.data<float>(), v_gpu.data<float>(), out_gpu.data<float>(),
    //         /*batches=*/B, /*heads=*/H, /*M=*/S, /*N=*/D/H, ag::current_stream());
    // cudaDeviceSynchronize();
    // auto refa = ref.transpose(1,2).flatten(2,3).clone();
    // auto outa = out_gpu.transpose(1,2).flatten(2,3).clone();

    // check_tensors_close(refa, outa, "test_gpu_attention", 0.01);









}



void test_aliatt( int Ha, int Ba, int Sa, int Da)
{
    auto cpu_opts = TensorOptions().with_device(Device::CPU);
    auto gpu_opts = TensorOptions().with_device(Device::CUDA);

    int H = Ha;
    int B = Ba;
    int S = Sa;
    int D = Da;

    // A: (11,9,3) B: (9,14,4) C: (9,14,5) D: (9,14,6) -> output (11,14)
    Tensor a_cpu = Tensor::randn(Shape{{B,S,D}}, cpu_opts);
    Tensor b_cpu = Tensor::randn(Shape{{D,D}}, cpu_opts);
    Tensor c_cpu = Tensor::randn(Shape{{D,D}}, cpu_opts);
    Tensor d_cpu = Tensor::randn(Shape{{D,D}}, cpu_opts);

    // Build Q,K,V via matmul as original code

\
       // Build Q,K,V via matmul as original code


    Tensor q = matmul(a_cpu, b_cpu.t()).unflatten(2, Shape({H, (b_cpu.shape().dims[1]/H)})).transpose(1,2).clone();
    Tensor k = matmul(a_cpu, c_cpu.t()).unflatten(2, Shape({H, (c_cpu.shape().dims[1]/H)})).transpose(1,2).clone();
    Tensor v = matmul(a_cpu, d_cpu.t()).unflatten(2, Shape({H, (d_cpu.shape().dims[1]/H)})).transpose(1,2).clone();


    float scale = 1.f / sqrtf(static_cast<float>(k.shape().dims.back()));
    Tensor logits = matmul(q, k.t()) * scale;

// 3) Build ALiBi bias [B, H, S, S] on CPU, then move to logits.device()
Tensor bias_cpu(
    logits.shape(),
    TensorOptions()
        .with_device(Device::CPU)
        .with_dtype(logits.dtype())
);

{
    float slope_start = 1.0f / std::pow(2.0f, 8.0f / H);

    dispatch_by_dtype(bias_cpu.dtype(), [&](auto dummy) {
        using Tval = decltype(dummy);
        Tval* data = bias_cpu.data<Tval>();

        for (int b = 0; b < B; ++b) {
            for (int h = 0; h < H; ++h) {

                float slope = std::pow(slope_start, h + 1);

                for (int i = 0; i < S; ++i) {
                    for (int j = 0; j < S; ++j) {

                        float v;
                        if (j > i)
                            v = -std::numeric_limits<float>::infinity();  // causal
                        else
                            v = -float(j - i) * slope;                    // ALiBi

                        // index = (((b*H + h)*S + i)*S + j)
                        const int idx = (((b * H + h) * S + i) * S + j);

                        data[idx] = static_cast<Tval>(v);
                    }
                }
            }
        }
    });
}

    

    // Tensor bias = bias_cpu.to(logits.device());
    Tensor g    = logits +bias_cpu;  // [H,T,T]

    // Re-implement softmax using OwnTensor ops
    Tensor max_val = reduce_max(g, {-1}, true);
    Tensor exp_g = exp(g - max_val);
    Tensor sum_exp_g = reduce_sum(exp_g, {-1}, true);
    Tensor s = exp_g / sum_exp_g;
    //ag::debug::print_tensor("Who is v", v);

    //ag::debug::print_tensor("S middle", s);

    auto ref = matmul(s, v);

    auto refa = ref.transpose(1,2).flatten(2,3).clone();

    // Tensor q_gpu = b_cpu.to(gpu_opts.device);
    // Tensor k_gpu = c_cpu.to(gpu_opts.device);
    // Tensor v_gpu = d_cpu.to(gpu_opts.device);

    auto zz = a_cpu.to_cuda();

    auto z = ag::Value(std::make_shared<ag::Node>(zz, ag::Op::Leaf, true, "X"));
    auto Q = ag::Value(std::make_shared<ag::Node>(b_cpu.to(gpu_opts.device), ag::Op::Leaf, true, "X"));
    auto Ka = ag::Value(std::make_shared<ag::Node>(c_cpu.to(gpu_opts.device), ag::Op::Leaf, true, "X"));
    auto V = ag::Value(std::make_shared<ag::Node>(d_cpu.to(gpu_opts.device), ag::Op::Leaf, true, "X"));

        auto outam = alibiatt(z, Q, Ka, V, H);
    // Tensor out_gpu(ref.shape(), options(ref).with_device(gpu_opts.device));

    // // flash attention call (standard softmax)
    // K.flashali(q_gpu.data<float>(), k_gpu.data<float>(), v_gpu.data<float>(), out_gpu.data<float>(),
    //         /*batches=*/B, /*heads=*/H, /*M=*/S, /*N=*/D/H, ag::current_stream());
    // cudaDeviceSynchronize();
    // auto outa = out_gpu.to_cpu().transpose(1,2).flatten(2,3).clone();

    check_tensors_close(refa.to_cpu(), outam.val().to_cpu(), "test_gpu_aliattention", 0.01);

//     Tensor gy_cpu = Tensor::randn(Shape{{11,7,12}}, cpu_opts);
    
    auto w = sum(outam);
    ag::debug::print_tensor("Result Value Alibi Attention", outam.val().to_cpu());
    backward(w);
    ag::debug::print_tensor("Result Gradient Alibi Attention", z.grad().to_cpu());
    ag::debug::print_tensor("Result Gradient Alibi Attention", z.val().to_cpu());

        ag::debug::print_tensor("Q value", Q.val().to_cpu());
        ag::debug::print_tensor("Q grad", Q.grad().to_cpu());
        ag::debug::print_tensor("K value", Ka.val().to_cpu());
        ag::debug::print_tensor("K grad", Ka.grad().to_cpu());
        ag::debug::print_tensor("V value", V.val().to_cpu());
        ag::debug::print_tensor("V grad", V.grad().to_cpu());


//     //     Node* A = n->inputs[0].get();
//     // Node* B = n->inputs[1].get();
//     // Node* C = n->inputs[2].get();
//     // Node* D = n->inputs[3].get();
    
//     // // Tensors from the forward pass, saved on the tape
//     // const Tensor& q = (*n->tape[0]);
//     // const Tensor& k = (*n->tape[1]);
//     // const Tensor& v = (*n->tape[2]);
//     // const Tensor& s = (*n->tape[3]); // The softmax output

//     // ag::debug::print_tensor("Gradient ya", gya);
//     auto gy = gy_cpu.unflatten(2, Shape({q.shape().dims[1], (q.shape().dims[3])})).transpose(1,2).clone();
//     // auto gy = gyo.to(n->value.device());
//     // auto gy = gyo.to(n->value.device());
//     // ag::debug::print_tensor("Gradient y", gy);
//     // ag::debug::print_tensor("Value", v);


//     // Re-implement softmax using OwnTensor ops

//         Tensor dL_ds = OwnTensor::matmul(gy, v.t());
//     Tensor dL_dv = OwnTensor::matmul(s.t(), gy).transpose(1,2).flatten(2,3);

//      //ag::debug::print_tensor("Gradient s", s);
//  //ag::debug::print_tensor("Value", dL_ds);
    
//     // VJP of softmax: s * (dL_ds - row_sum(s * dL_ds))
//     Tensor dot = OwnTensor::reduce_sum(s * dL_ds, {-1}, true);
//     Tensor dL_dg = s * (dL_ds - dot);
    
//     // Propagate gradients back through the Q, K projections
//     Tensor dL_dq = OwnTensor::matmul(dL_dg, k).transpose(1,2).flatten(2,3);
//     Tensor dL_dk = OwnTensor::matmul(dL_dg.t(), q).transpose(1,2).flatten(2,3);

//              //ag::debug::print_tensor("Gradient q", dL_dq);
//  //ag::debug::print_tensor("Value B", B->value);
//  //ag::debug::print_tensor("Value A", A->value);


// //     // Propagate gradients to the weight matrices and the input A
//     Tensor gb_ref = OwnTensor::matmul(dL_dq.t(), a_cpu) * scale;
//     Tensor gc_ref = OwnTensor::matmul(dL_dk.t(), a_cpu) * scale;
//     Tensor gd_ref = OwnTensor::matmul(dL_dv.t(), a_cpu) * scale;

//     Tensor dL_dA_q = OwnTensor::matmul(dL_dq, b_cpu);
//     Tensor dL_dA_k = OwnTensor::matmul(dL_dk, c_cpu);
//     Tensor dL_dA_v = OwnTensor::matmul(dL_dv, d_cpu);
//     Tensor ga_ref = (dL_dA_q * scale) + (dL_dA_k * scale) + dL_dA_v;

//     ag::debug::print_tensor("A Grad", ga_ref);
//     ag::debug::print_tensor("B Grad", gb_ref);
//     ag::debug::print_tensor("C Grad", gc_ref);
//     ag::debug::print_tensor("D Grad", gd_ref);


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
    ag::debug::print_tensor("Result Value Parametric Cone", w.val());
    backward(w);
    ag::debug::print_tensor("Result Gradient Parametric Cone", m.grad());
              ag::opti.SGD(w, 0.01);
      

    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
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
ag::opti.SGD(w, 0.01);
    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
                
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
ag::opti.SGD(w, 0.01);
    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
                
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
                ag::opti.SGDm(w, 0.001, 0.1);
    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
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
ag::opti.SGD(w, 0.01);
    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
                
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
ag::opti.SGD(w, 0.01);
    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
                
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
ag::opti.SGD(w, 0.01);
    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
                
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
ag::opti.SGD(w, 0.01);
    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
                
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
ag::opti.SGD(w, 0.01);
    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
                
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
ag::opti.SGD(w, 0.01);
    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
                
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
ag::opti.SGD(w, 0.01);
    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
                
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
ag::opti.SGD(w, 0.01);
    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
                
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
ag::opti.SGD(w, 0.01);
    for(int i=0;i<10;i++){
        forward(w);
        backward(w);
                
        opti.epoch();
    }



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
ag::debug::print_tensor("Result Gradient1 RMSNorm", w.node->inputs[1]->grad.to_cpu().to_cpu());
ag::opti.SGD(w, 0.01);
    for(int i=0;i<2;i++){
        forward(w);
        backward(w);
                
        opti.epoch();
    }
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
ag::opti.SGD(w, 0.01);
    // for(int i=0;i<10;i++){
    //     forward(w);
    //     backward(w);
    //             
       opti.epoch();
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
ag::opti.SGD(w, 0.01);
    // for(int i=0;i<10;i++){
    //     forward(w);
    //     backward(w);
    //             
      opti.epoch();
    // }
ag::debug::print_tensor("Result Gradient Alpha DynTanh", w.node->inputs[1]->grad.to_cpu());
ag::debug::print_tensor("Result Gradient Gamma DynTanh", w.node->inputs[3]->grad.to_cpu());
ag::debug::print_tensor("Result Gradient Beta DynTanh", w.node->inputs[2]->grad.to_cpu());
}

void test_cewithlogits( int H, int B, int S, int D)
{
    Tensor X = Tensor::randn(Shape({B, S, D}), TensorOptions().with_device(Device::CUDA));
    ag::debug::print_tensor("Input Layer Norm", X);
    auto m = ag::Value(std::make_shared<ag::Node>(X, ag::Op::Leaf, true, "X"));

    Tensor onehot = Tensor::randn(Shape({B, S, D}), TensorOptions().with_device(Device::CUDA));
    auto target = ag::Value(std::make_shared<ag::Node>(onehot, ag::Op::Leaf, true, "X"));

    // auto swag = ag::layer::RMSNorm();
    auto w = cross_entropy_with_logits(m, target);
    
    // auto w = sum(r);
    ag::debug::print_tensor("Input Target Value CE With Logits", target.val());

    ag::debug::print_tensor("Result Value CE With Logits", w.val());
    backward(w);
    ag::debug::print_tensor("Result Input Gradient CE With Logits", m.grad());
    ag::debug::print_tensor("Result Target Gradient CE With Logits", target.grad());
ag::opti.SGD(w, 0.01);
    // for(int i=0;i<10;i++){
    //     forward(w);
    //     backward(w);
    //             
       opti.epoch();
    // }
// ag::debug::print_tensor("Result Gradient Gamma Layer Norm", w.node->inputs[1]->grad.to_cpu());
// ag::debug::print_tensor("Result Gradient Beta Layer Norm", w.node->inputs[2]->grad.to_cpu());
}



int main(){


// test_expand(2, 2, 4, 16);

// test_aliatt(2, 4, 64, 512);
// test_aliatt(2, 4, 64, 1024);
// test_aliatt(2, 4, 64, 2048);
// test_aliatt(2, 4, 128, 1024);
// test_aliatt(8, 16, 2048, 1024);
// try {
//         #if defined(_WIN32)
//             const char* plugin_path = "./agkernels_cuda.dll";
//         #elif defined(__APPLE__)
//             const char* plugin_path = "./libagkernels_cuda.dylib";
//         #else
//             const char* plugin_path = "/home/blubridge-034/Downloads/Newf/cgadimpl/cgadimpl/build/libagkernels_cuda.so";
//         #endif

//         std::cout << "Loading GPU plugin from: " << plugin_path << "\n";
//         kernels::load_cuda_plugin(plugin_path);


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

// test_att(4, 11, 7, 12);
test_swiglu(2, 4, 2, 4, 10);
test_att(2, 4, 2, 4);

test_rmsnorm(2, 4, 2, 4);
test_laynorm(2, 4, 2, 4);
test_dyntanh(2, 4, 2, 4);

test_cewithlogits(2, 4, 2, 4);


// }
// catch (const std::exception& e) {
//         std::cerr << "ERROR: " << e.what() << std::endl;
//         return 1;
//     }
return 0;

}