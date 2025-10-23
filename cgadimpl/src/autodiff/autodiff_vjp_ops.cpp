#include "ad/detail/autodiff_ops.hpp"
#include "ad/cudaops.hpp"
#include <cuda.h>
#include <cmath>
#include "ad/nodeops.hpp"
#include "ad/cudarray.hpp"

namespace ag {
namespace detail{

// helper: reduce a gradient to a parent's shape (broadcast-aware)
inline Tensor rt(const Tensor& g, const Tensor& like){ return Tensor::reduce_to(g, like); }


float* gradacc(float* a, float*b, float misiz)
{


    auto* fn = run_cuda_add;
    if (!fn)
        throw std::runtime_error("No CUDA Add kernel registered");

float* c = nullptr;

    fn(a, b, c, misiz);




    return c;


}

// ----- elementwise binary -----
void vjp_Add(Node* n){

    if(n->cuda_device)
    {

    auto A = n->inputs[0]; auto B = n->inputs[1];
    if (A->requires_grad) A->c_array = gradacc(A->c_array, n->c_array, A->siz);
    if (B->requires_grad) B->c_array = gradacc(B->c_array, n->c_array, B->siz);

    }
    else{


    Node* A = n->inputs[0].get(); Node* B = n->inputs[1].get();
    if (A->requires_grad) A->grad.add_( rt(n->grad, A->value) );
    if (B->requires_grad) B->grad.add_( rt(-n->grad, B->value) );



    }

}
void vjp_Sub(Node* n){
    Node* A = n->inputs[0].get(); Node* B = n->inputs[1].get();
    if (A->requires_grad) A->grad.add_( rt(n->grad, A->value) );
    if (B->requires_grad) B->grad.add_( rt(-n->grad, B->value) );
}
void vjp_Mul(Node* n){
    Node* A = n->inputs[0].get(); Node* B = n->inputs[1].get();
    if (A->requires_grad) A->grad.add_( rt( n->grad * B->value, A->value) );
    if (B->requires_grad) B->grad.add_( rt( n->grad * A->value, B->value) );
}

// ----- elementwise trinary -----
void vjp_FMA(Node* n){
    Node* A = n->inputs[0].get();
    Node* B = n->inputs[1].get();
    Node* C = n->inputs[2].get();

    // External kernel (if plugin loaded), else fallback to Tensor::matmul
    auto* mm = ag::kernels::cpu().matmul;

    // Shapes
    const Tensor& At = A->value;
    const Tensor& Bt = B->value;
    auto [M, K]  = At.shape();
    auto [K2, N] = Bt.shape();
    (void)K2; // assume forward already checked

    if (A->requires_grad){
        Tensor BT = Tensor::transpose(Bt); // (N x K)
        Tensor dA(M, K);                   // temp buffer

        if (mm) {
            // dA = n->grad (MxN) * BT (NxK)
            mm(n->grad.data(), BT.data(), dA.data(), M, N, K);
        } else {
            dA = Tensor::matmul(n->grad, BT);
        }
        A->grad.add_(dA);
    }

    if (B->requires_grad){
        Tensor AT = Tensor::transpose(At); // (K x M)
        Tensor dB(K, N);                   // temp buffer

        if (mm) {
            // dB = AT (KxM) * n->grad (MxN)
            mm(AT.data(), n->grad.data(), dB.data(), K, M, N);
        } else {
            dB = Tensor::matmul(AT, n->grad);
        }
        B->grad.add_(dB);
    }
    if (C->requires_grad) C->grad.add_( rt(n->grad, C->value) );
}

void vjp_LayerNorm(Node* n){

    Node* x = n->inputs[0].get();
     int N = x->value.cols(); // normalize over last dim (row-wise)

   //  debug::print_tensor("n->grad",n->grad);
     
    
    // stddev [2x1] - float
    Tensor std = Tensor::sqrt(*(n->tape[0]) +0.01);
  //  debug::print_tensor("std",std);

    // x - mean [2x1]
    Tensor xmu = x->value - *(n->tape[1]);
 //   debug::print_tensor("xmu",xmu);

    // sum of grad_out along row
    Tensor grad_sum = Tensor::row_sum(n->grad);
  //  debug::print_tensor("grad_sum",grad_sum);

    // dot(grad_out, x - mean) along row
    Tensor grad_dot_xmu = Tensor::row_sum(n->grad * xmu);
   // debug::print_tensor("grad_dot_xmu",grad_dot_xmu);

    // term: N * grad_out
    Tensor term1 = n->grad * float(N);
 //   debug::print_tensor("term1",term1);

    // term: subtract sum of grad_out
    Tensor term2 = term1 - grad_sum;
   // debug::print_tensor("term2",term2);

    // term: subtract (x - mean) * (grad_dot_xmu / (var + eps))
    Tensor term3 = term2 - (xmu * (grad_dot_xmu / (*(n->tape[0]) + 0.01)));
  //  debug::print_tensor("term3",term3);

    // scale: divide by (N * std)
    Tensor dx = term3 / (std * float(N));
  //  debug::print_tensor("dx",dx);

    if (x->requires_grad) x->grad.add_( dx );

}


void vjp_RealLayerNorm(Node* n){

    Node* x = n->inputs[0].get();
    Node* b = n->inputs[1].get();
    Node* g = n->inputs[2].get();
     int N = x->value.cols(); // normalize over last dim (row-wise)

   //  debug::print_tensor("n->grad",n->grad);
     
    
    // stddev [2x1] - float
    Tensor std = Tensor::sqrt(*(n->tape[0]) +0.01);
  //  debug::print_tensor("std",std);

    // x - mean [2x1]
    Tensor xmu = x->value - *(n->tape[1]);
 //   debug::print_tensor("xmu",xmu);

    // sum of grad_out along row
    Tensor grad_sum = Tensor::row_sum(n->grad);
  //  debug::print_tensor("grad_sum",grad_sum);

    // dot(grad_out, x - mean) along row
    Tensor grad_dot_xmu = Tensor::row_sum(n->grad * xmu);
   // debug::print_tensor("grad_dot_xmu",grad_dot_xmu);

    // term: N * grad_out
    Tensor term1 = n->grad * float(N);
 //   debug::print_tensor("term1",term1);

    // term: subtract sum of grad_out
    Tensor term2 = term1 - grad_sum;
   // debug::print_tensor("term2",term2);

    // term: subtract (x - mean) * (grad_dot_xmu / (var + eps))
    Tensor term3 = term2 - (xmu * (grad_dot_xmu / (*(n->tape[0]) + 0.01)));
  //  debug::print_tensor("term3",term3);

    // scale: divide by (N * std)
    Tensor dx = term3 / (std * float(N));
 //debug::print_tensor("dx",term3);
// debug::print_tensor("g",g->value);

    if (x->requires_grad) x->grad.add_( dx);
if (b->requires_grad) b->grad.add_( Tensor::row_sum(n->grad) );   // db = sum over batch
if (g->requires_grad) g->grad.add_( Tensor::row_sum(n->grad * (*(n->tape[2]))) );

}


void vjp_RMSNorm(Node* n){

    Node* x = n->inputs[0].get();
    Tensor rms = *n->tape[0];
    Tensor y   = *n->tape[1];   // normalized x

    // upstream dot
    Tensor dot = Tensor::row_sum(n->grad * y);  // [batch x 1]

    Tensor grad_x = (n->grad / rms) - (y * dot / (rms*x->value.cols()));

    if (x->requires_grad) x->grad.add_(grad_x);


}


void vjp_RealRMSNorm(Node* n){

    Node* x = n->inputs[0].get();
    Node* g = n->inputs[1].get();
    Tensor rms = *n->tape[0];
    Tensor y   = *n->tape[1];   // normalized x


    // upstream dot
    Tensor dot = Tensor::row_sum(n->grad * y);  // [batch x 1]

    Tensor grad_x = g->value*((n->grad / rms) - (y * dot / (rms*x->value.cols())));


    if (x->requires_grad) x->grad.add_(grad_x);
    if (g->requires_grad) g->grad.add_( n->grad * (x->value / rms));


}


// ----- elementwise quarternary -----
void vjp_Attention(Node* n){
    Node* A = n->inputs[0].get();
    Node* B = n->inputs[1].get();
    Node* C = n->inputs[2].get();
    Node* D = n->inputs[3].get();
    
    Tensor q = n->tape[0] ? *n->tape[0] : Tensor();
    Tensor k = n->tape[1] ? *n->tape[1] : Tensor();
    Tensor v = n->tape[2] ? *n->tape[2] : Tensor();
    float scale = 1.0f / std::sqrt(float(k.cols()));
    Tensor s = n->tape[3] ? *n->tape[3] : Tensor();

    // ---- Backprop chain ----

    // y = s v
    Tensor dL_ds = Tensor::matmul(n->grad, Tensor::transpose(v));   // [B x B]
    Tensor dL_dv = Tensor::matmul(Tensor::transpose(s), n->grad);   // [A x D]

    // s = softmax(g)
    Tensor dL_dg; 
    {
        Tensor dot = Tensor::row_sum(s * dL_ds);
        dL_dg = s * (dL_ds - dot);
    }

    // g = q k^T
    Tensor dL_dq = Tensor::matmul(dL_dg, k);
    Tensor dL_dk = Tensor::matmul(Tensor::transpose(dL_dg), q);

    // q = A B
    Tensor dL_dA_q = Tensor::matmul(dL_dq, Tensor::transpose(B->value));
    Tensor dL_dB   = Tensor::matmul(Tensor::transpose(A->value), dL_dq)* scale;;

    // k = A C
    Tensor dL_dA_k = Tensor::matmul(dL_dk, Tensor::transpose(C->value));
    Tensor dL_dC   = Tensor::matmul(Tensor::transpose(A->value), dL_dk)* scale;

    // v = A D
    Tensor dL_dA_v = Tensor::matmul(dL_dv, Tensor::transpose(D->value));
    Tensor dL_dD   = Tensor::matmul(Tensor::transpose(A->value), dL_dv);

    // combine A contributions
    Tensor dL_dA = dL_dA_q + dL_dA_k + dL_dA_v;

    // ---- Accumulate ----
    if (A->requires_grad) A->grad.add_(dL_dA);
    if (B->requires_grad) B->grad.add_(dL_dB);
    if (C->requires_grad) C->grad.add_(dL_dC);
    if (D->requires_grad) D->grad.add_(dL_dD);

}


void vjp_AlibiAttention(Node* n){
    Node* A = n->inputs[0].get();
    Node* B = n->inputs[1].get();
    Node* C = n->inputs[2].get();
    Node* D = n->inputs[3].get();
    
    Tensor q = n->tape[0] ? *n->tape[0] : Tensor();
    Tensor k = n->tape[1] ? *n->tape[1] : Tensor();
    Tensor v = n->tape[2] ? *n->tape[2] : Tensor();
    float scale = 1.0f / std::sqrt(float(k.cols()));
    Tensor s = n->tape[3] ? *n->tape[3] : Tensor();

    // ---- Backprop chain ----

    // y = s v
    Tensor dL_ds = Tensor::matmul(n->grad, Tensor::transpose(v));   // [B x B]
    Tensor dL_dv = Tensor::matmul(Tensor::transpose(s), n->grad);   // [A x D]

    // s = softmax(g)
    Tensor dL_dg; 
    {
        Tensor dot = Tensor::row_sum(s * dL_ds);
        dL_dg = s * (dL_ds - dot);
    }

    // g = q k^T
    Tensor dL_dq = Tensor::matmul(dL_dg, k);
    Tensor dL_dk = Tensor::matmul(Tensor::transpose(dL_dg), q);

    // q = A B
    Tensor dL_dA_q = Tensor::matmul(dL_dq, Tensor::transpose(B->value));
    Tensor dL_dB   = Tensor::matmul(Tensor::transpose(A->value), dL_dq)* scale;;

    // k = A C
    Tensor dL_dA_k = Tensor::matmul(dL_dk, Tensor::transpose(C->value));
    Tensor dL_dC   = Tensor::matmul(Tensor::transpose(A->value), dL_dk)* scale;

    // v = A D
    Tensor dL_dA_v = Tensor::matmul(dL_dv, Tensor::transpose(D->value));
    Tensor dL_dD   = Tensor::matmul(Tensor::transpose(A->value), dL_dv);

    // combine A contributions
    Tensor dL_dA = dL_dA_q + dL_dA_k + dL_dA_v;

    // ---- Accumulate ----
    if (A->requires_grad) A->grad.add_(dL_dA);
    if (B->requires_grad) B->grad.add_(dL_dB);
    if (C->requires_grad) C->grad.add_(dL_dC);
    if (D->requires_grad) D->grad.add_(dL_dD);

}


void vjp_SWIGLU(Node* n){
    Node* X = n->inputs[0].get();
    Node* A = n->inputs[1].get();
    Node* B = n->inputs[2].get();
    Node* C = n->inputs[3].get();
    Node* D = n->inputs[4].get();

    Tensor y = Tensor::matmul(X->value, Tensor::transpose(A->value)) + B->value;
    Tensor q = y * Tensor::sigmoid(y);
    Tensor h = Tensor::matmul(X->value, Tensor::transpose(C->value)) + D->value;
    Tensor w = q * h;

    // derivatives
    Tensor Swishdif = Tensor::sigmoid(y) + y * (Tensor::sigmoid(y) * (Tensor::ones_like(y) - Tensor::sigmoid(y)));

    Tensor dL_dB = Swishdif * h * n->grad;
    Tensor dL_dA = Tensor::matmul(Tensor::transpose(Swishdif * h * n->grad), X->value);

    Tensor dL_dD = q*n->grad;
    Tensor dL_dC = Tensor::matmul(Tensor::transpose(q * n->grad), X->value);

    Tensor dL_dX = Tensor::matmul(Swishdif * h * n->grad, A->value)
                 + Tensor::matmul(q * n->grad, C->value);

    // accumulate grads
    if (X->requires_grad) X->grad.add_(dL_dX);
    if (A->requires_grad) A->grad.add_(dL_dA);
    if (B->requires_grad) B->grad.add_(dL_dB);
    if (C->requires_grad) C->grad.add_(dL_dC);
    if (D->requires_grad) D->grad.add_(dL_dD);

}


// ----- unary activations -----
void vjp_Relu(Node* n){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad) return;

        auto* mm = ag::kernels::cpu().relumask;
            auto [K2, N] = (X->value).shape();

                Tensor dA(K2, N);                   // temp buffer

        if(mm)
            mm((X->value).data(), dA.data(), dA.numel());

        else

        {
            std::cout<<"CUDA is unused";
    dA = Tensor::relu_mask(X->value);

        }


    X->grad.add_( rt( n->grad * dA, X->value) );

}

void vjp_Exp(Node* n){
    Node* X = n->inputs[0].get();
    // if (!X->requires_grad) return;

    //     auto* mm = ag::kernels::cpu().exp;
    //         auto [K2, N] = (X->value).shape();

    //             Tensor dA(K2, N);                   // temp buffer

    //     if(mm)
    //         mm((X->value).data(), dA.data(), dA.numel());

    //     else

    //     {
    //         std::cout<<"CUDA is unused";
    // dA = Tensor::relu_mask(X->value);

    //     }


    X->grad.add_( rt( n->grad * Tensor::exp(X->value), X->value) );
}
void vjp_Log(Node* n){
    Node* X = n->inputs[0].get();
    if (X->requires_grad) X->grad.add_( rt( n->grad / X->value, X->value) );
}

void vjp_GCU(Node* n){
    Node* X = n->inputs[0].get();
    if (X->requires_grad) X->grad.add_( rt( n->grad * (Tensor::cos(X->value)-(X->value*Tensor::sin(X->value))), X->value) );
}

void vjp_Mish(Node* n){
    Node* X = n->inputs[0].get();
    if (X->requires_grad) X->grad.add_( rt( n->grad * (Tensor::tanh( Tensor::softplus(X->value) )-(  (X->value*Tensor::sigmoid(X->value))  / (Tensor::cosh( Tensor::softplus(X->value)*Tensor::cosh( Tensor::softplus(X->value) ))    )            )), X->value) );
}


void vjp_Tanh(Node* n){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad) return;
    Tensor th = n->value, one = Tensor::ones_like(th);
    
    X->grad.add_( rt( n->grad * (one - th*th), X->value) );
}
void vjp_Sigmoid(Node* n){





    auto X = n->inputs[0]; if (!X->requires_grad) return;
    X->grad.add_( rt( n->grad * (Tensor::sigmoid(X->value)*(Tensor::ones_like(X->value)-Tensor::sigmoid(X->value))), X->value) );


}

void vjp_Sigmoidiff(Node* n){





    auto X = n->inputs[0]; if (!X->requires_grad) return;
    X->grad.add_( rt( n->grad * (Tensor::sigmoid(X->value)*(Tensor::ones_like(X->value)-Tensor::sigmoid(X->value))*(Tensor::ones_like(X->value)-(2.0*Tensor::sigmoid(X->value)))), X->value) );


}


void vjp_Softplus(Node* n){
    Node* X = n->inputs[0].get(); if (!X->requires_grad) return;
    X->grad.add_( rt( n->grad * Tensor::sigmoid(X->value), X->value) );
}

void vjp_Gaus(Node* n){
    Node* X = n->inputs[0].get(); if (!X->requires_grad) return;
    X->grad.add_( rt( n->grad * -2*X->value*Tensor::exp(-1*X->value*X->value), X->value) );
}

void vjp_Transpose(Node* n){
    Node* X = n->inputs[0].get(); if (!X->requires_grad) return;

    X->grad.add_( rt( Tensor::transpose(n->grad) , X->value) );
}



void vjp_SiLU(Node* n){
    Node* X = n->inputs[0].get(); if (!X->requires_grad) return;
    Tensor s = Tensor::sigmoid(X->value);
    X->grad.add_( rt( n->grad * ( s + X->value * ( s * (Tensor::ones_like(s)-s) ) ), X->value) );
}

void vjp_Parcon(Node* n){
    Node* X = n->inputs[0].get(); if (!X->requires_grad) return;
    X->grad.add_( rt( n->grad * ( 2 *Tensor::ones_like(X->value)- 2*X->value  ), X->value) );
}

void vjp_LiSHT(Node* n){
    Node* X = n->inputs[0].get(); if (!X->requires_grad) return;
    X->grad.add_( rt( n->grad * ( Tensor::tanh(X->value)+ (Tensor::sech(X->value)*Tensor::sech(X->value)*X->value ) ), X->value) );
}




void vjp_GELU(Node* n){
    Node* X = n->inputs[0].get(); if (!X->requires_grad) return;
    constexpr float c = 0.79788456080286535588f; // sqrt(2/pi)
    int R=X->value.rows(), C=X->value.cols();
    Tensor x=X->value,u(R,C),dudx(R,C);
    for(int i=0;i<R;++i) for(int j=0;j<C;++j){
        float z=x(i,j);
        u(i,j)=c*(z+0.044715f*z*z*z);
        dudx(i,j)=c*(1.f+0.134145f*z*z);
    }
    Tensor th=Tensor::tanh(u), one=Tensor::ones_like(th);
    Tensor dgelu=(one+th)*0.5f + (x * ((one - th*th) * dudx))*0.5f;
    X->grad.add_( rt( n->grad * dgelu, X->value) );
}
void vjp_LeakyRelu(Node* n){
    Node* X = n->inputs[0].get(); Node* A = n->inputs[1].get();
    if (!X->requires_grad) return;
    float a = A->value(0,0);
    int R=X->value.rows(), C=X->value.cols();
    Tensor g(R,C);
    for(int i=0;i<R;++i) for(int j=0;j<C;++j){
        float z=X->value(i,j);
        g(i,j)= n->grad(i,j) * (z>0.f ? 1.f : a);
    }
    X->grad.add_( g );
}

// ----- matmul -----
void vjp_MatMul(Node* n){
   Node* A = n->inputs[0].get();
    Node* B = n->inputs[1].get();

    // External kernel (if plugin loaded), else fallback to Tensor::matmul
    auto* mm = ag::kernels::cpu().matmul;

    // Shapes
    const Tensor& At = A->value;
    const Tensor& Bt = B->value;
    auto [M, K]  = At.shape();
    auto [K2, N] = Bt.shape();
    (void)K2; // assume forward already checked

    if (A->requires_grad){
        Tensor BT = Tensor::transpose(Bt); // (N x K)
        Tensor dA(M, K);                   // temp buffer

        if (mm) {
            // dA = n->grad (MxN) * BT (NxK)
            mm(n->grad.data(), BT.data(), dA.data(), M, N, K);
        } else {
            dA = Tensor::matmul(n->grad, BT);
        }
        A->grad.add_(dA);
    }

    if (B->requires_grad){
        Tensor AT = Tensor::transpose(At); // (K x M)
        Tensor dB(K, N);                   // temp buffer

        if (mm) {
            // dB = AT (KxM) * n->grad (MxN)
            mm(AT.data(), n->grad.data(), dB.data(), K, M, N);
        } else {
            dB = Tensor::matmul(AT, n->grad);
        }
        B->grad.add_(dB);
    }
}

void vjp_Dyntanh(Node* n){
    Node* X = n->inputs[0].get(); 
    Node* A = n->inputs[1].get(); 
    Node* B = n->inputs[2].get(); 
    Node* G = n->inputs[3].get();



        if (X->requires_grad) X->grad.add_(n->grad*Tensor::sech(X->value*A->value)*Tensor::sech(X->value*A->value)*A->value*G->value); 
    if (A->requires_grad) A->grad.add_(n->grad*Tensor::sech(X->value*A->value)*Tensor::sech(X->value*A->value)*X->value*G->value);
    if (B->requires_grad) B->grad.add_(n->grad);
    if (G->requires_grad) G->grad.add_(n->grad*Tensor::tanh(*(n->tape.back()))   );
}

// ----- reductions -----
void vjp_Sum(Node* n){
    Node* X=n->inputs[0].get(); if(!X->requires_grad) return;
    float s = n->grad(0,0);
    X->grad.add_( Tensor::ones_like(X->value) * s );
}
void vjp_RowSum(Node* n){
    Node* X=n->inputs[0].get(); if(!X->requires_grad) return;
    // n->grad [B,1] broadcast across columns
    Tensor g = n->grad * Tensor::ones_like(X->value);
    X->grad.add_( g );
}
void vjp_RowMax(Node* n){
    Node* X=n->inputs[0].get(); if(!X->requires_grad) return;
    int R=X->value.rows(), C=X->value.cols();
    Tensor m = Tensor::row_max(X->value);
    Tensor g(R,C);
    for(int i=0;i<R;++i) for(int j=0;j<C;++j)
        g(i,j) = (X->value(i,j)==m(i,0)) ? n->grad(i,0) : 0.f;
    X->grad.add_( g );
}
void vjp_MeanAll(Node* n){
    Node* X=n->inputs[0].get(); if(!X->requires_grad) return;
    float scale = n->grad(0,0) / float(X->value.rows()*X->value.cols());
    X->grad.add_( Tensor::ones_like(X->value) * scale );
}

// ----- softmax / losses -----
void vjp_SoftmaxRow(Node* n){
    Node* Z = n->inputs[0].get(); if(!Z->requires_grad) return;
    Tensor y = n->value; // softmax(Z)
    Tensor dot = Tensor::row_sum( y * n->grad ); // [B,1]
    Tensor g = y * (n->grad - dot);
    Z->grad.add_( g );
}
void vjp_LogSumExpRow(Node* n){
    Node* Z = n->inputs[0].get(); if(!Z->requires_grad) return;
    Tensor y = Tensor::softmax_row(Z->value);
    Z->grad.add_( y * n->grad ); // n->grad [B,1] broadcast
}
void vjp_CeWithLogits(Node* n /*unused: scalar n->grad*/){
    Node* Z = n->inputs[0].get();
    Node* Y = n->inputs[1].get();
    int B = Z->value.rows();
    Tensor sm = Tensor::softmax_row(Z->value);
    Tensor gZ = (sm - Y->value) * (1.0f / float(B));
    if (Z->requires_grad) Z->grad.add_( gZ );
    if (Y->requires_grad) {
        Tensor lse = Tensor::logsumexp_row(Z->value);
        Tensor lsm = Z->value - lse;
        Tensor gY  = lsm * (-1.0f / float(B));
        Y->grad.add_( gY );
    }
}



void vjp_KLDivergence(Node* n /*unused: scalar n->grad*/){
    Node* Z = n->inputs[0].get();
    Node* Y = n->inputs[1].get();
    int B = Z->value.rows();
    Tensor sm = Tensor::softmax_row(Z->value);
    Tensor gZ = (sm - Y->value) * (1.0f / float(B));
    if (Z->requires_grad) Z->grad.add_( gZ );
    if (Y->requires_grad) {
        Tensor lse = Tensor::logsumexp_row(Z->value);
        Tensor lsm = Z->value - lse;
        Tensor gY = (Tensor::log(Y->value) + Tensor::ones_like(Y->value) - lsm) * (1.0f / float(B));
        Y->grad.add_( gY );
    }
}

void vjp_Div(Node* n){
    Node* A = n->inputs[0].get(); Node* B = n->inputs[1].get();
    if (A->requires_grad) A->grad.add_( rt( n->grad * (Tensor::reciprocal(B->value)), A->value) );
    if (B->requires_grad) B->grad.add_( rt( -n->grad * (Tensor::reciprocal(B->value)) * (Tensor::reciprocal(B->value)) * A->value, B->value) );
}
void vjp_Reciprocal(Node* n){
    Node* X = n->inputs[0].get(); if (!X->requires_grad) return;
    X->grad.add_( rt( -n->grad * (Tensor::reciprocal(X->value)) * (Tensor::reciprocal(X->value)), X->value) );
}





void vjp_Linear(Node* n){
    Node* A = n->inputs[0].get();
    Node* B = n->inputs[1].get();
    Node* C = n->inputs[2].get();

    // External kernel (if plugin loaded), else fallback to Tensor::matmul
    auto* mm = ag::kernels::cpu().matmul;

    // Shapes
    const Tensor& At = A->value;
    const Tensor& Bt = B->value;
    auto [M, K]  = At.shape();
    auto [K2, N] = Bt.shape();
    (void)K2; // assume forward already checked

    if (A->requires_grad){
        Tensor BT = Bt; // (N x K)
        Tensor dA(M, K);                   // temp buffer

        if (mm) {
            // dA = n->grad (MxN) * BT (NxK)
            mm(n->grad.data(), BT.data(), dA.data(), M, N, K);
        } else {
            dA = Tensor::matmul(n->grad, BT);
        }
        A->grad.add_(dA);
    }

    if (B->requires_grad){
        Tensor AT = Tensor::transpose(At); // (K x M)
        Tensor dB(K, N);                   // temp buffer

        if (mm) {
            // dB = AT (KxM) * n->grad (MxN)
            mm(AT.data(), n->grad.data(), dB.data(), K, M, N);
        } else {
            dB = Tensor::matmul(AT, n->grad);
        }
        B->grad.add_(dB);
    }
    if (C->requires_grad) C->grad.add_( rt(n->grad, C->value) );
}






void vjp_Cosh(Node* n){
    Node* X = n->inputs[0].get();
    if (X->requires_grad) X->grad.add_( rt( n->grad * (Tensor::sinh(X->value)), X->value) );
}


void vjp_Sinh(Node* n){
    Node* X = n->inputs[0].get();
    if (X->requires_grad) X->grad.add_( rt( n->grad * (Tensor::cosh(X->value)), X->value) );
}


void vjp_Sign(Node* n){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad) return;
    Tensor th = n->value, one = Tensor::ones_like(th);
    
    X->grad.add_( rt( n->grad * 0.0f, X->value) );
}

void vjp_Cos(Node* n){
    Node* X = n->inputs[0].get();
    if (X->requires_grad) X->grad.add_( rt( -1.0 * n->grad* (Tensor::sin(X->value)), X->value) );
}


void vjp_Sin(Node* n){
    Node* X = n->inputs[0].get();
    if (X->requires_grad) X->grad.add_( rt( n->grad * (Tensor::cos(X->value)), X->value) );
}


void vjp_Sqrt(Node* n){
    Node* X = n->inputs[0].get();
    if (X->requires_grad) X->grad.add_( rt(0.5f * n->grad * Tensor::sqrt(  Tensor::reciprocal(X->value)), X->value) );
}

void vjp_Relumask(Node* n){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad) return;
    Tensor th = n->value, one = Tensor::ones_like(th);

    X->grad.add_( rt( n->grad * 0.0f, X->value) );
}















void vjp_RELUAtt(Node* n){
    Node* A = n->inputs[0].get();
    Node* B = n->inputs[1].get();
    Node* C = n->inputs[2].get();
    Node* D = n->inputs[3].get();

    Tensor q = n->tape[0] ? *n->tape[0] : Tensor();
    Tensor k = n->tape[1] ? *n->tape[1] : Tensor();
    Tensor v = n->tape[2] ? *n->tape[2] : Tensor();
    float scale = 1.0f / std::sqrt(float(k.cols()));
    Tensor s = n->tape[3] ? *n->tape[3] : Tensor();

    // ---- Backprop chain ----

    // y = s v
    Tensor dL_ds = Tensor::matmul(n->grad, Tensor::transpose(v));   // [B x B]
    Tensor dL_dv = Tensor::matmul(Tensor::transpose(s), n->grad);   // [A x D]

    // s = softmax(g)
    Tensor dL_dg; 
    {
        Tensor dot = Tensor::relu_mask(s )* dL_ds;
        dL_dg = dot;
    }

    // g = q k^T
    Tensor dL_dq = Tensor::matmul(dL_dg, k);
    Tensor dL_dk = Tensor::matmul(Tensor::transpose(dL_dg), q);

// q = A B^T
Tensor dL_dA_q = Tensor::matmul(dL_dq, B->value) * scale;
Tensor dL_dB   = Tensor::matmul(Tensor::transpose(dL_dq), A->value) * scale;

// k = A C^T
Tensor dL_dA_k = Tensor::matmul(dL_dk, C->value) * scale;
Tensor dL_dC   = Tensor::matmul(Tensor::transpose(dL_dk), A->value) * scale;

// v = A D^T
Tensor dL_dA_v = Tensor::matmul(dL_dv, D->value);
Tensor dL_dD   = Tensor::matmul(Tensor::transpose(dL_dv), A->value);

    // combine A contributions
    Tensor dL_dA = dL_dA_q + dL_dA_k + dL_dA_v;

    // ---- Accumulate ----
    if (A->requires_grad) A->grad.add_(dL_dA);
    if (B->requires_grad) B->grad.add_(dL_dB);
    if (C->requires_grad) C->grad.add_(dL_dC);
    if (D->requires_grad) D->grad.add_(dL_dD);


}






void vjp_MOE(Node* n){
    Node* X = n->inputs[0].get();
    Node* W = n->inputs[1].get();
    Node* B = n->inputs[2].get();

    Tensor y = Tensor::matmul(X->value, Tensor::transpose(W->value)) + B->value; 

    Tensor dL_dB = n->grad;
    Tensor dL_dW = Tensor::matmul(Tensor::transpose(n->grad), X->value);
    Tensor dL_dX = Tensor::matmul(n->grad, W->value);

    if (X->requires_grad) X->grad.add_(dL_dX);
    if (W->requires_grad) W->grad.add_(dL_dW);
    if (B->requires_grad) B->grad.add_(dL_dB);

}



void vjp_SigAtt(Node* n){
    Node* A = n->inputs[0].get();
    Node* B = n->inputs[1].get();
    Node* C = n->inputs[2].get();
    Node* D = n->inputs[3].get();

    Tensor q = n->tape[0] ? *n->tape[0] : Tensor();
    Tensor k = n->tape[1] ? *n->tape[1] : Tensor();
    Tensor v = n->tape[2] ? *n->tape[2] : Tensor();
    float scale = 1.0f / std::sqrt(float(k.cols()));
    Tensor s = n->tape[3] ? *n->tape[3] : Tensor();

    // ---- Backprop chain ----

    // y = s v
    Tensor dL_ds = Tensor::matmul(n->grad, Tensor::transpose(v));   // [B x B]
    Tensor dL_dv = Tensor::matmul(Tensor::transpose(s), n->grad);   // [A x D]

    // s = softmax(g)
    Tensor dL_dg; 
    {
        Tensor dot = ( s * (Tensor::ones_like(s)-s))* dL_ds;

        dL_dg = dot;
    }

    // g = q k^T
    Tensor dL_dq = Tensor::matmul(dL_dg, k);
    Tensor dL_dk = Tensor::matmul(Tensor::transpose(dL_dg), q);

// q = A B^T
Tensor dL_dA_q = Tensor::matmul(dL_dq, B->value) * scale;
Tensor dL_dB   = Tensor::matmul(Tensor::transpose(dL_dq), A->value) * scale;

// k = A C^T
Tensor dL_dA_k = Tensor::matmul(dL_dk, C->value) * scale;
Tensor dL_dC   = Tensor::matmul(Tensor::transpose(dL_dk), A->value) * scale;

// v = A D^T
Tensor dL_dA_v = Tensor::matmul(dL_dv, D->value);
Tensor dL_dD   = Tensor::matmul(Tensor::transpose(dL_dv), A->value);

    // combine A contributions
    Tensor dL_dA = dL_dA_q + dL_dA_k + dL_dA_v;

    // ---- Accumulate ----
    if (A->requires_grad) A->grad.add_(dL_dA);
    if (B->requires_grad) B->grad.add_(dL_dB);
    if (C->requires_grad) C->grad.add_(dL_dC);
    if (D->requires_grad) D->grad.add_(dL_dD);


}


void vjp_MSELoss(Node* n /*unused: scalar n->grad*/){
    Node* Z = n->inputs[0].get();
    Node* Y = n->inputs[1].get();
    int B = Z->value.rows(), C = Z->value.cols();
    Tensor diff = Z->value - Y->value;
    Tensor gZ = diff * (2.0f / float(B * C));
    Tensor gY = -diff * (2.0f / float(B * C));
    if (Z->requires_grad) Z->grad.add_(gZ);
    if (Y->requires_grad) Y->grad.add_(gY);
}

void vjp_MAELoss(Node* n /*unused: scalar n->grad*/){
    Node* Z = n->inputs[0].get();
    Node* Y = n->inputs[1].get();
    int B = Z->value.rows(), C = Z->value.cols();
    Tensor diff = Tensor::sign(Z->value - Y->value);
    Tensor gZ = diff * (1.0f / float(B * C));
    Tensor gY = -diff * (1.0f / float(B * C));
    if (Z->requires_grad) Z->grad.add_(gZ);
    if (Y->requires_grad) Y->grad.add_(gY);
}


void vjp_Leaf(Node*){ /* no-op */ }


} // anon


// -------- dispatch table --------
VjpFn vjp_lookup(Op op){
    switch(op){
#define OP(name, arity, str) case Op::name: return &detail::vjp_##name;
#include "ad/detail/ops.def"
#undef OP
        default: return nullptr;
    }
}

} // namespace ag
