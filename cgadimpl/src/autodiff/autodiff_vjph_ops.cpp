
#include "ad/detail/autodiff_ops.hpp"
#include "ad/runtime.hpp"
#include <cmath>
#include <stdexcept> // Required for std::runtime_error
#include <iostream>
#include "ad/nodeops.hpp"

namespace ag {
namespace detail{

    void newvjp_Leaf(std::shared_ptr<Node>){ /* no-op */ }

void newvjp_ReduceTo(std::shared_ptr<Node>){ /* no-op */ }


void newvjp_Add(std::shared_ptr<Node> n){
    // Node* A = n->inputs[0].get();
    // Node* B = n->inputs[1].get();
    // if (!A->requires_cuda && !B->requires_cuda) {
    //     if (A->requires_grad) A->grad.add_( rt(gy, A->value) );
    //     if (B->requires_grad) B->grad.add_( rt(-gy, B->value) );
    // } else {
    //     throw std::runtime_error("VJP for Sub on CUDA not implemented yet!");
    // }

    auto A = n->inputs[0]; auto B = n->inputs[1];
    if (!n->requires_cuda) {
        if (A->requires_grad) A->gran->value = n->gran->value + A->gran->value;
        if (B->requires_grad) B->gran->value = n->gran->value + B->gran->value;
                A->grad = A->gran->value;
        B->grad = B->gran->value;
    } else {
        std::cout<<"wedfrthy7";
auto* fn = ag::kernels::cuda().vjp_add;
    if(fn){
           fn(A->gran->value.data(), B->gran->value.data(), n->gran->value.data(),
                                    n->gran->value.numel(), ag::current_stream());
    }
    
        else{
        throw std::runtime_error("VJP for Mul on CUDA not implemented yet!");
        }
    }
}



void newvjp_Sub(std::shared_ptr<Node> n){


    auto A = n->inputs[0]; auto B = n->inputs[1];
    if (!n->requires_cuda) {
        if (A->requires_grad) A->gran = n->gran + A->gran;
        if (B->requires_grad) B->gran = B->gran - n->gran;
        A->grad = A->gran->value;
        B->grad = B->gran->value;
    } else {
        std::cout<<"wedfrthy7";
auto* fn = ag::kernels::cuda().vjp_sub;
    if(fn){
           fn(A->gran->value.data(), B->gran->value.data(), n->gran->value.data(),
                                    n->gran->value.numel(), ag::current_stream());

        // auto n = std::make_shared<Node>(Y, A->gran->requires_grad || n->gran->requires_grad, Op::Sub, "-"); 
        A->gran = std::make_shared<Node>(A->gran->value, A->gran->requires_grad || n->gran->requires_grad, Op::Add, "+");
        B->gran = std::make_shared<Node>(B->gran->value, B->gran->requires_grad || n->gran->requires_grad, Op::Sub, "-");
        
        A->gran->inputs = {A->gran, n->gran}; 
        B->gran->inputs = {B->gran, n->gran}; 
    }
    
        else{
        throw std::runtime_error("VJP for Mul on CUDA not implemented yet!");
        }
    }
}

void newvjp_Mul(std::shared_ptr<Node> n){
    auto A = n->inputs[0]; auto B = n->inputs[1];
    if (!n->requires_cuda) {
        if (A->requires_grad) {
            A->gran = n->gran * B + A->gran;
            A->grad = A->gran->value;
        }
        if (B->requires_grad) {
            B->gran = n->gran * A + B->gran;
            B->grad = B->gran->value;
        }
    } else {
        using namespace ag::kernels;
        auto fn = cuda().vjp_hadmul;
        if(fn) {
            fn(A->gran->value.data(), B->gran->value.data(), n->gran->value.data(),
               A->value.data(), B->value.data(), n->gran->value.numel(), current_stream());
        } else {
            throw std::runtime_error("VJP for Mul on CUDA not implemented yet!");
        }
    }
}

void newvjp_MatMul(std::shared_ptr<Node> n){
    auto A = n->inputs[0]; auto B = n->inputs[1];
    const Tensor& At = A->value;
    const Tensor& Bt = B->value;
    auto [M, K]  = At.shape();
    auto [K2, N] = Bt.shape();
    (void)K2;

    if (A->value.is_cpu()) {
        auto* mm = ag::kernels::cpu().matmul;
        if (A->requires_grad) {
            auto BT = transpose_nodeops(B);
            auto dA = std::make_shared<Node>(Tensor::zeros(M, K), true, Op::MatMul, "matmul");
            if (mm) { 
                mm(n->gran->value.data(), BT->value.data(), dA->value.data(), M, N, K);
            } else {
                dA = matmul_nodeops(n->gran, B);
            }
            A->gran = A->gran + dA;
            A->grad = A->gran->value;
        }
        if (B->requires_grad) {
            auto AT = transpose_nodeops(A);
            auto dB = std::make_shared<Node>(Tensor::zeros(K, N), true, Op::MatMul, "matmul");
            if (mm) {
                mm(AT->value.data(), n->gran->value.data(), dB->value.data(), K, M, N);
            } else {
                dB = matmul_nodeops(AT, n->gran);
            }
            B->gran = B->gran + dB;
            B->grad = B->gran->value;
        }
    } else {
        auto fn = ag::kernels::cuda().vjp_matmul;
        if(fn) {
    //         fn(A->gran->value.data(), B->gran->value.data(), n->gran->value.data(),
    //            At.data(), Bt.data(), M, K, N, ag::current_stream());

    //                auto ghost_dA = std::make_shared<Node>(A->gran->value, A->requires_grad || B->requires_grad, Op::MatMul, "vjp_dA");
    // ghost_dA->inputs = {n->gran, transpose_nodeops(B)};
    
    // A->gran = A->gran + ghost_dA;

    // auto ghost_dB = std::make_shared<Node>(B->gran->value, A->requires_grad || B->requires_grad, Op::MatMul, "vjp_dB");
    // ghost_dB->inputs = {transpose_nodeops(A), n->gran};
    // B->gran = B->gran + ghost_dB;


        } else {
            throw std::runtime_error("VJP for MatMul on CUDA not implemented yet!");
        }
    }
}

void newvjp_Div(std::shared_ptr<Node> n){
    auto A = n->inputs[0]; auto B = n->inputs[1];
    if (!n->requires_cuda) {
        if (A->requires_grad) {
            A->gran = n->gran * reci_nodeops(B) + A->gran;
            A->grad = A->gran->value;
        }
        if (B->requires_grad) {
            B->gran = -n->gran * reci_nodeops(B) * 
                            reci_nodeops(B) * A + B->gran;
            B->grad = B->gran->value;
        }
    } else {
        auto fn = ag::kernels::cuda().vjp_div;
        if(fn) {
            fn(A->gran->value.data(), B->gran->value.data(), n->gran->value.data(),
               A->value.data(), B->value.data(), n->gran->value.numel(), ag::current_stream());
        } else {
            throw std::runtime_error("VJP for Div on CUDA not implemented yet!");
        }
    }
}

void newvjp_Relu(std::shared_ptr<Node> n){
    auto X = n->inputs[0];
    if (!X->requires_grad) return;
    if (!n->requires_cuda) {
        auto mask = relumask_nodeops(X);
        X->gran = n->gran * mask + X->gran;
        X->grad = X->gran->value;
    } else {
        auto fn = ag::kernels::cuda().vjp_relu;
        if(fn) {
            fn(X->gran->value.data(), n->gran->value.data(), X->value.data(),
               X->value.numel(), ag::current_stream());
        } else {
            throw std::runtime_error("VJP for ReLU on CUDA not implemented yet!");
        }
    }
}

void newvjp_Sigmoid(std::shared_ptr<Node> n){
    auto X = n->inputs[0];
    if (!X->requires_grad) return;
    if (!n->requires_cuda) {
        auto ones = make_tensornode(Tensor::ones_like(n->value), "ones", false);

        X->gran = n->gran * n * (ones - n) + X->gran;
        X->grad = X->gran->value;
    } else {
        auto fn = ag::kernels::cuda().vjp_sigmoid;
        if(fn) {
            fn(X->gran->value.data(), n->gran->value.data(), n->value.data(),
               X->value.numel(), ag::current_stream());
        } else {
            throw std::runtime_error("VJP for Sigmoid on CUDA not implemented yet!");
        }
    }
}

void newvjp_Tanh(std::shared_ptr<Node> n){
    auto X = n->inputs[0];
    if (!X->requires_grad) return;
    if (!n->requires_cuda) {
        auto ones = make_tensornode(Tensor::ones_like(n->value), "ones", false);
        X->gran = n->gran * (ones - n * n) + X->gran;
        X->grad = X->gran->value;
    } else {
        auto fn = ag::kernels::cuda().vjp_tanh;
        if(fn) {
            fn(X->gran->value.data(), n->gran->value.data(), X->value.data(),
               X->value.numel(), ag::current_stream());
        } else {
            throw std::runtime_error("VJP for Tanh on CUDA not implemented yet!");
        }
    }
}

void newvjp_Gelu(std::shared_ptr<Node> n){
    auto X = n->inputs[0];
    if (!X->requires_grad) return;
    if (!n->requires_cuda) {
        auto ones = make_tensornode(Tensor::ones_like(n->value), "ones", false);
        auto weuse = make_tensornode(Tensor::vals_like(X->value, 0.044715f), "val", false);
        auto weuse2 = make_tensornode(Tensor::vals_like(X->value, 0.5f), "val2", false);
        auto weuse3 = make_tensornode(Tensor::vals_like(X->value, sqrt(2.0f/M_PI)), "val3", false);
        auto weuse4 = make_tensornode(Tensor::vals_like(X->value, -0.5f), "val4", false);
        auto weuse5 = make_tensornode(Tensor::vals_like(X->value, 1.0f/sqrt(2.0f * M_PI)), "val5", false);
        auto x3 = X * X * X;  // x^3
   
        auto inner = X +weuse * x3;

        auto cdf = weuse2 * (ones + tanh_nodeops(weuse3 * inner));
        auto negHalfX2 = weuse4 * X * X;
        auto pdf = weuse5 * exp_nodeops(negHalfX2);
        X->gran = n->gran * (cdf + X * pdf) + X->gran;
        X->grad = X->gran->value;
    } else {
        auto fn = ag::kernels::cuda().vjp_gelu;
        if(fn) {
            fn(X->gran->value.data(), n->gran->value.data(), X->value.data(),
               X->value.numel(), ag::current_stream());
        } else {
            throw std::runtime_error("VJP for GELU on CUDA not implemented yet!");
        }
    }
}

void newvjp_Silu(std::shared_ptr<Node> n){
    auto X = n->inputs[0];
    if (!X->requires_grad) return;
    if (!n->requires_cuda) {
        auto sig = sigmoid_nodeops(X);
        auto ones = make_tensornode(Tensor::ones_like(sig->value), "ones", false);
        X->gran = n->gran * (sig + X * sig * (ones - sig)) + X->gran;
        X->grad = X->gran->value;
    } else {
        auto fn = ag::kernels::cuda().vjp_silu;
        if(fn) {
            fn(X->gran->value.data(), n->gran->value.data(), X->value.data(),
               X->value.numel(), ag::current_stream());
        } else {
            throw std::runtime_error("VJP for SiLU on CUDA not implemented yet!");
        }
    }
}

void newvjp_Exp(std::shared_ptr<Node> n){
    auto X = n->inputs[0];
    if (!X->requires_grad) return;
    if (!n->requires_cuda) {
        X->gran = n->gran * n + X->gran;
        X->grad = X->gran->value;
    } else {
        auto fn = ag::kernels::cuda().vjp_exp;
        if(fn) {
            fn(X->gran->value.data(), n->gran->value.data(), X->value.data(),
               X->value.numel(), ag::current_stream());
        } else {
            throw std::runtime_error("VJP for Exp on CUDA not implemented yet!");
        }
    }
}

void newvjp_Log(std::shared_ptr<Node> n){
    auto X = n->inputs[0];
    if (!X->requires_grad) return;
    if (!n->requires_cuda) {
        X->gran = n->gran * reci_nodeops(X) + X->gran;
        X->grad = X->gran->value;
    } else {
        auto fn = ag::kernels::cuda().vjp_log;
        if(fn) {
            fn(X->gran->value.data(), n->gran->value.data(), X->value.data(),
               X->value.numel(), ag::current_stream());
        } else {
            throw std::runtime_error("VJP for Log on CUDA not implemented yet!");
        }
    }
}

void newvjp_Sqrt(std::shared_ptr<Node> n){
    auto X = n->inputs[0];
    if (!X->requires_grad) return;
    if (!n->requires_cuda) {
        auto half = make_tensornode(Tensor::vals_like(n->value, 0.5f), "half", true);
        X->gran = half * n->gran * reci_nodeops(n) + X->gran;
        X->grad = X->gran->value;
    } else {
        auto fn = ag::kernels::cuda().vjp_sqrt;
        if(fn) {
            fn(X->gran->value.data(), n->gran->value.data(), n->value.data(),
               X->value.numel(), ag::current_stream());
        } else {
            throw std::runtime_error("VJP for Sqrt on CUDA not implemented yet!");
        }
    }
}

void newvjp_Square(std::shared_ptr<Node> n){
    auto X = n->inputs[0];
    if (!X->requires_grad) return;
    if (!n->requires_cuda) {
        X->gran = 2.0f * n->gran * X + X->gran;
        X->grad = X->gran->value;
    } else {
        auto fn = ag::kernels::cuda().vjp_square;
        if(fn) {
            fn(X->gran->value.data(), n->gran->value.data(), X->value.data(),
               X->value.numel(), ag::current_stream());
        } else {
            throw std::runtime_error("VJP for Square on CUDA not implemented yet!");
        }
    }
}

void newvjp_Sin(std::shared_ptr<Node> n){
    auto X = n->inputs[0];
    if (!X->requires_grad) return;
    if (!n->requires_cuda) {
        X->gran = n->gran * cos_nodeops(X) + X->gran;
        X->grad = X->gran->value;
    } else {
        auto fn = ag::kernels::cuda().vjp_sin;
        if(fn) {
            fn(X->gran->value.data(), n->gran->value.data(), X->value.data(),
               X->value.numel(), ag::current_stream());
        } else {
            throw std::runtime_error("VJP for Sin on CUDA not implemented yet!");
        }
    }
}

void newvjp_Cos(std::shared_ptr<Node> n){
    auto X = n->inputs[0];
    if (!X->requires_grad) return;
    if (!n->requires_cuda) {
        X->gran->value = n->gran->value * (-Tensor::sin(X->value)) + X->gran->value;
        X->grad = X->gran->value;
    } else {
        auto fn = ag::kernels::cuda().vjp_cos;
        if(fn) {
            fn(X->gran->value.data(), n->gran->value.data(), X->value.data(),
               X->value.numel(), ag::current_stream());
        } else {
            throw std::runtime_error("VJP for Cos on CUDA not implemented yet!");
        }
    }
}

void newvjp_Sum(std::shared_ptr<Node> n){
    auto X = n->inputs[0];
    if (!X->requires_grad) return;
    if (!n->requires_cuda) {
        float s = n->gran->value(0,0);
        X->gran->value = X->gran->value + Tensor::ones_like(X->value) * s;
        X->grad = X->gran->value;
    } else {
        auto fn = ag::kernels::cuda().vjp_sum;
        if(fn) {
            fn(X->gran->value.data(), n->gran->value.data(), X->value.data(),
               X->value.numel(), ag::current_stream());
        } else {
            throw std::runtime_error("VJP for Sum on CUDA not implemented yet!");
        }
    }
}

void newvjp_MeanAll(std::shared_ptr<Node> n){
    auto X = n->inputs[0];
    if (!X->requires_grad) return;
    if (!n->requires_cuda) {
        float s = n->gran->value(0,0) / X->value.numel();
        X->gran->value = X->gran->value + Tensor::ones_like(X->value) * s;
        X->grad = X->gran->value;
    } else {
        // VJP for mean is just scaled sum VJP
        auto scale = 1.0f / X->value.numel();
        auto scaled_gy = Tensor::ones_like(n->gran->value) * scale;
        X->gran->value = X->gran->value + scaled_gy;
        X->grad = X->gran->value;
    }
}

void newvjp_RowSum(std::shared_ptr<Node> n){
    auto X = n->inputs[0];
    if (!X->requires_grad) return;
    if (!n->requires_cuda) {
        auto [r, c] = X->value.shape();
        for(int i=0; i<r; i++) {
            float s = n->gran->value(i,0);
            for(int j=0; j<c; j++) {
                X->gran->value(i,j) = X->gran->value(i,j) + s;
            }
        }
        X->grad = X->gran->value;
    } else {
        auto fn = ag::kernels::cuda().vjp_rowsum;
        if(fn) {
            fn(X->gran->value.data(), X->value.data(), nullptr, n->gran->value.data(), 
               X->value.rows(), X->value.cols(), ag::current_stream());
        } else {
            throw std::runtime_error("VJP for RowSum on CUDA not implemented yet!");
        }
    }
}

void newvjp_MSELoss(std::shared_ptr<Node> n){
    auto A = n->inputs[0];
    auto B = n->inputs[1];
    auto scale = 2.0f / A->value.numel();
    if (!n->requires_cuda) {
        if (A->requires_grad) {
            auto diff = A->value - B->value;
            A->gran->value = A->gran->value + n->gran->value * scale * diff;
            A->grad = A->gran->value;
        }
        if (B->requires_grad) {
            auto diff = A->value - B->value;
            B->gran->value = B->gran->value - n->gran->value * scale * diff;
            B->grad = B->gran->value;
        }
    } else {
        auto fn = ag::kernels::cuda().vjp_mseloss;
        if(fn) {
            fn(A->gran->value.data(), B->gran->value.data(), n->gran->value.data(),
               A->value.data(), B->value.data(), n->value.numel(), ag::current_stream());
        } else {
            throw std::runtime_error("VJP for MSELoss on CUDA not implemented yet!");
        }
    }
}

void newvjp_MAELoss(std::shared_ptr<Node> n){
    auto A = n->inputs[0];
    auto B = n->inputs[1];
    auto scale = 1.0f / A->value.numel();
    if (!n->requires_cuda) {
        if (A->requires_grad) {
            auto diff = A->value - B->value;
            auto sign = Tensor::sign(diff);
            A->gran->value = A->gran->value + n->gran->value * scale * sign;
            A->grad = A->gran->value;
        }
        if (B->requires_grad) {
            auto diff = A->value - B->value;
            auto sign = Tensor::sign(diff);
            B->gran->value = B->gran->value - n->gran->value * scale * sign;
            B->grad = B->gran->value;
        }
    } else {
        auto fn = ag::kernels::cuda().vjp_maeloss;
        if(fn) {
            fn(A->gran->value.data(), B->gran->value.data(), n->gran->value.data(),
               A->value.data(), B->value.data(), n->value.numel(), ag::current_stream());
        } else {
            throw std::runtime_error("VJP for MAELoss on CUDA not implemented yet!");
        }
    }
}

void newvjp_CrossEntropyLoss(std::shared_ptr<Node> n) {
    auto probs = n->inputs[0];
    auto target = n->inputs[1];
    
    if (!n->requires_cuda) {
        // Cross entropy derivative: -target_i / probs_i
        if (probs->requires_grad) {
            auto inv_probs = Tensor::reciprocal(probs->value);
            auto grad = -target->value * inv_probs;
            probs->gran->value = probs->gran->value + n->gran->value * grad;
            probs->grad = probs->gran->value;
        }
        // Target is usually a one-hot vector, no grad needed
    } else {
        throw std::runtime_error("VJP for CrossEntropyLoss on CUDA not implemented yet!");
    }
}

void newvjp_Mish(std::shared_ptr<Node> n) {
    auto X = n->inputs[0];
    if (!X->requires_grad) return;
    
    if (!n->requires_cuda) {
        // Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
        // Let sp = softplus(x) = ln(1 + exp(x))
        // Let th = tanh(sp)
        // d/dx[Mish(x)] = tanh(sp) + x * (1 - th^2) * (1 / (1 + exp(-x)))
        auto softplus = Tensor::log(Tensor::ones_like(X->value) + Tensor::exp(X->value));
        auto tanh_sp = Tensor::tanh(softplus);
        auto sigmoid = Tensor::sigmoid(X->value);
        
        auto ones = Tensor::ones_like(tanh_sp);
        auto grad = tanh_sp + X->value * (ones - tanh_sp * tanh_sp) * sigmoid;
        
        X->gran->value = X->gran->value + n->gran->value * grad;
        X->grad = X->gran->value;
    } else {
        auto fn = ag::kernels::cuda().vjp_mish;
        if(fn) {
            fn(X->gran->value.data(), n->gran->value.data(), X->value.data(),
               X->value.numel(), ag::current_stream());
        } else {
            throw std::runtime_error("VJP for Mish on CUDA not implemented yet!");
        }
    }
}

void newvjp_LeakyRelu(std::shared_ptr<Node> n) {
    auto X = n->inputs[0];
    if (!X->requires_grad) return;
    
    const float alpha = 0.01f; // Typical value for leaky slope
    
    if (!n->requires_cuda) {
        auto mask = Tensor::relu_mask(X->value);
        auto slope = Tensor::ones_like(X->value) * alpha;
        auto grad = mask + slope * (Tensor::ones_like(mask) - mask);
        X->gran->value = X->gran->value + n->gran->value * grad;
        X->grad = X->gran->value;
    } else {
        auto fn = ag::kernels::cuda().vjp_leaky_relu;
        if(fn) {
            fn(X->gran->value.data(), X->value.data(), n->gran->value.data(),
               0.01f, n->value.numel(), ag::current_stream());
        } else {
            throw std::runtime_error("VJP for LeakyReLU on CUDA not implemented yet!");
        }
    }
}

void newvjp_HardSigmoid(std::shared_ptr<Node> n) {
    auto X = n->inputs[0];
    if (!X->requires_grad) return;
    
    const float slope = 1.0f/6.0f;
    
    if (!n->requires_cuda) {
        auto negThree = Tensor::vals_like(X->value, -3.0f);
        auto posThree = Tensor::vals_like(X->value, 3.0f);
        auto above_neg = Tensor::relu_mask(X->value + negThree);
        auto below_pos = Tensor::relu_mask(posThree - X->value);
        auto is_valid = above_neg * below_pos;
        auto grad = Tensor::ones_like(X->value) * slope * is_valid;
        X->gran->value = X->gran->value + n->gran->value * grad;
        X->grad = X->gran->value;
    } else {
        auto fn = ag::kernels::cuda().vjp_hard_sigmoid;
        if(fn) {
            fn(X->gran->value.data(), n->gran->value.data(), X->value.data(),
               X->value.numel(), ag::current_stream());
        } else {
            throw std::runtime_error("VJP for HardSigmoid on CUDA not implemented yet!");
        }
    }
}

// Generic fallback: call existing vjp_* implementation (which expects Node* and Tensor gy)
// then mirror parents' Node::grad into their grad-node (`gran`) so both systems stay in sync.
void newvjp_generic(std::shared_ptr<Node> n){
    if (!n) return;
    if (!n->gran) return; // nothing to backprop
    const Tensor& gy = n->gran->value;

    // Lookup the existing vjp function and call it on the raw Node*
    VjpFn vf = vjp_lookup(n->op);
    if (vf) vf(n.get(), gy);

    // Ensure parent grad-nodes (gran) exist and accumulate their values from Node::grad
    // for (auto &p : n->inputs){
    //     if (!p) continue;
    //     // if (!p->gran) {
    //         p->gran = std::make_shared<Node>(p->grad, true, Op::Leaf, "grad");
    //     // }

        // if(n->op==Op::Mul)
        // {
        // A->gran = std::make_shared<Node>(A->gran->value, A->gran->requires_grad || n->gran->requires_grad, Op::Add, "+");
        // B->gran = std::make_shared<Node>(B->gran->value, B->gran->requires_grad || n->gran->requires_grad, Op::Sub, "-");
        
        // A->gran->inputs = {A->gran, n->gran}; 
        // B->gran->inputs = {B->gran, n->gran}; 
    
        // }
    // }
        // accumulate into gran and keep Node::grad in sync
     //   n->gran->value = n->grad;
    //    n->grad = n->gran->value;
    // }
}

} // namespace detail


// -------- dispatch table --------


NewVjpFn newvjp_lookup(Op op){
    switch(op){
#define OP(name, arity, str) case Op::name: return &detail::newvjp_##name;
#include "ad/detail/newops.def"
#undef OP
        default: return &detail::newvjp_generic;
    }
}

} // namespace ag