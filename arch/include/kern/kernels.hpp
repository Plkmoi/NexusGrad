// =========================================================
// FILE: cgadimpl/include/kernels/kernels.hpp
// =========================================================
#pragma once
#include <cstdint>

#if defined(_WIN32)
  #define FLOW_EXPORT __declspec(dllexport)
  #define FLOW_IMPORT __declspec(dllimport)
#else
  #define FLOW_EXPORT __attribute__((visibility("default")))
  #define FLOW_IMPORT
#endif

// ---------- C ABI shared with plugins ----------
extern "C" {

// Keep existing:
static const uint32_t FLOW_KERNELS_ABI_V1 = 1;

// Plain C function-pointer types (no Tensor types here)
typedef void (*flow_relu_fn)(const float* x, float* y, int64_t n);
typedef void (*flow_matmul_fn)(const float* A, const float* B, float* C,
                             int M, int K, int N);                             
typedef void (*flow_gelu_fn)(const float* x, float* y, int64_t n);
typedef void (*flow_leakyrelu_fn)(const float* x, float* y, int64_t n, float alpha);
typedef void (*flow_sigmoid_fn)(const float* x, float* y, int64_t n);
typedef void (*flow_tanh_fn)(const float* x, float* y, int64_t n);
typedef void (*flow_softplus_fn)(const float* x, float* y, int64_t n);
typedef void (*flow_exp_fn)(const float* x, float* y, int64_t n);
typedef void (*flow_log_fn)(const float* x, float* y, int64_t n);
typedef void (*flow_sqrt_fn) (const float* x, float* y, int64_t n);
typedef void (*flow_pow_fn) (const float* x, float* y, int64_t n, float exponent);
typedef void (*flow_linear_fn)(const float* X,const float* W,const float* b,float* Y,int B,int In,int Out);
// CPU function table (can be partially filled; nulls mean "not provided")
typedef void (*elem_bwd_fn)(const float*, const float*, float*, int64_t);
typedef void (*elem_bwd_alpha_fn)(const float*, const float*, float*, int64_t, float);
typedef void (*flow_linear_dW_fn)(const float* X, const float* dY, float* dW, int B, int In, int Out);
typedef void (*flow_linear_dX_fn)(const float* dY, const float* W, float* dX, int B, int In, int Out);
typedef void (*flow_linear_db_fn)(const float* dY, float* db, int B, int Out);


void leakyrelu_bwd_impl_optimized(const float* x, const float* dY, float* dX, int64_t n, float alpha);
void sigmoid_bwd_impl_optimized_from_s(const float* s, const float* dY, float* dX, int64_t n);
void tanh_bwd_impl_optimized_from_t(const float* t, const float* dY, float* dX, int64_t n);
void gelu_bwd_impl_optimized(const float* x, const float* dY, float* dX, int64_t n);
void softplus_bwd_impl_optimized(const float* x, const float* dY, float* dX, int64_t n);
void exp_bwd_impl_optimized_from_y(const float* y, const float* dY, float* dX, int64_t n);
void log_bwd_impl_optimized(const float* x, const float* dY, float* dX, int64_t n);
void sqrt_bwd_impl_optimized_from_y(const float* y, const float* dY, float* dX, int64_t n);
void matmul_bwd_dA_impl_optimized(const float* dC, const float* B, float* dA, int M, int K, int N);
void matmul_bwd_dB_impl_optimized(const float* A, const float* dC, float* dB, int M, int K, int N);
void linear_dW_impl_optimized(const float* X, const float* dY, float* dW, int B, int In, int Out);
void linear_dX_impl_optimized(const float* dY, const float* W, float* dX, int B, int In, int Out);
void linear_db_impl_optimized(const float* dY, float* db, int B, int Out);
void relu_bwd_impl_optimized(const float* x, const float* dY, float* dX, int64_t n);


// CPU function table (can be partially filled; nulls mean "not provided")
struct flow_cpu_v1 {
  uint32_t abi_version;   // must be FLOW_KERNELS_ABI_V1
  flow_relu_fn   relu; //done
  flow_matmul_fn matmul; //done
  flow_gelu_fn gelu;   //done
  flow_leakyrelu_fn leakyrelu;   //done
  flow_sigmoid_fn sigmoid; //done
  flow_tanh_fn tanh; //done
  flow_softplus_fn softplus; //done
  flow_exp_fn exp;  //done
  flow_log_fn log; //done
  flow_sqrt_fn sqrt; //done
  flow_pow_fn pow;
  flow_linear_fn linear;
  //backwards
  elem_bwd_fn relu_bwd;  //done
  elem_bwd_alpha_fn leakyrelu_bwd; // takes alpha  //done
  elem_bwd_fn sigmoid_bwd_from_s;  // if forward stored s //done
  elem_bwd_fn tanh_bwd_from_t; //done
  elem_bwd_fn gelu_bwd; //done
  elem_bwd_fn softplus_bwd; //done
  elem_bwd_fn exp_bwd_from_y; //done
  elem_bwd_fn log_bwd;         //done
  elem_bwd_fn sqrt_bwd_from_y;  //done
  // matmul backward wrappers
 void (*matmul_bwd_dA)(const float*, const float*, float*, int M, int K, int N);   
 void (*matmul_bwd_dB)(const float*, const float*, float*, int M, int K, int N);  
  flow_linear_dW_fn linear_dW;
  flow_linear_dX_fn linear_dX;  
  flow_linear_db_fn linear_db;

};


FLOW_EXPORT int flow_get_cpu_kernels_v1(struct flow_cpu_v1* out);

// ---- NEW: CUDA function pointer types (accept a stream) ----
// Avoid pulling in CUDA headers here: just forward-declare the opaque type.
typedef struct CUstream_st* ag_cuda_stream_t;


// Arithmetic
typedef void (*flow_add_cuda_fn)   (const float* a, const float* b, float* c, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_sub_cuda_fn)   (const float* a, const float* b, float* c, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_hadmul_cuda_fn)   (const float* a, const float* b, float* c, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_div_cuda_fn)   (const float* a, const float* b, float* c, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_pow_cuda_fn)   (const float* a, const float* b, float* c, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_square_cuda_fn)(const float* x, float* y, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_neg_cuda_fn)   (const float* x, float* y, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_clip_cuda_fn)  (const float* x, float* y, float minv, float maxv, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_mseloss_cuda_fn)   (const float* a, const float* b, float* c, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_maeloss_cuda_fn)   (const float* a, const float* b, float* c, int64_t n, ag_cuda_stream_t s);

// Activations
typedef void (*flow_relu_cuda_fn)        (const float* x, float* y, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_leaky_relu_cuda_fn)  (const float* x, float* y, float alpha, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_gelu_cuda_fn)        (const float* x, float* y, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_silu_cuda_fn)        (const float* x, float* y, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_mish_cuda_fn)        (const float* x, float* y, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_tanh_cuda_fn)        (const float* x, float* y, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_sigmoid_cuda_fn)     (const float* x, float* y, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_hard_sigmoid_cuda_fn)(const float* x, float* y, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_hard_swish_cuda_fn)  (const float* x, float* y, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_exp_cuda_fn)         (const float* x, float* y, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_log_cuda_fn)         (const float* x, float* y, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_softplus_cuda_fn)         (const float* x, float* y, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_relumask_cuda_fn)    (const float* x, float* y, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_sum_cuda_fn)         (const float* x, float* y, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_rowsum_cuda_fn)         (const float* x, float* y, int64_t n, int64_t w, ag_cuda_stream_t s);
typedef void (*flow_rowmax_cuda_fn)         (const float* x, float* y, int64_t n, int64_t w, ag_cuda_stream_t s);
typedef void (*flow_sin_cuda_fn)         (const float* x, float* y, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_cos_cuda_fn)         (const float* x, float* y, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_sinh_cuda_fn)         (const float* x, float* y, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_cosh_cuda_fn)         (const float* x, float* y, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_sign_cuda_fn)         (const float* x, float* y, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_softmax_cuda_fn)         (const float* x, float* y, int64_t n, int64_t w, ag_cuda_stream_t s);
typedef void (*flow_logsumexp_cuda_fn)         (const float* x, float* y, int64_t n, int64_t w, ag_cuda_stream_t s);
typedef void (*flow_cewithlogits_cuda_fn)         (const float* x, const float* z, float* y, int64_t n, int64_t w, ag_cuda_stream_t s);
typedef void (*flow_kldivergence_cuda_fn)         (const float* x, const float* z, float* y, int64_t n, int64_t w, ag_cuda_stream_t s);

// Additional custom elementwise ops
typedef void (*flow_gcu_cuda_fn)          (const float* x, float* y, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_gauss_cuda_fn)        (const float* x, float* y, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_parcon_cuda_fn)       (const float* x, float* y, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_lisht_cuda_fn)        (const float* x, float* y, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_optim_cuda_fn)           (float* vX, const float* gy, const float X, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_reci_cuda_fn)         (const float* x, float* y, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_sqrt_cuda_fn)         (const float* x, float* y, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_dyntanh_cuda_fn)         (const float* x, float a, float b, float g, float* y, int64_t n, ag_cuda_stream_t s);

// Core
typedef void (*flow_zero_cuda_fn)  (float* x, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_matmul_cuda_fn)(const float* A, const float* B, float* C, int M, int K, int N, ag_cuda_stream_t s);
typedef void (*flow_gemm_cuda_fn)(const float* A, const float* B, float* C, float* E, int M, int K, int N, ag_cuda_stream_t s);
typedef void (*flow_linear_cuda_fn)(const float* A, const float* B, float* C, float* E, int M, int K, int N, ag_cuda_stream_t s);
typedef void (*flow_flash_attention)(const float* Q, const float* K, const float* V, float* O,
                       int B, int nh, int N, int d, ag_cuda_stream_t);
typedef void (*flow_flash_alibattention)(const float* Q, const float* K, const float* V, float* O,
                       int B, int nh, int N, int d, ag_cuda_stream_t);
typedef void (*flow_reluflash_attention)(const float* Q, const float* K, const float* V, float* O,
                       int B, int nh, int N, int d, ag_cuda_stream_t);
typedef void (*flow_sigflash_attention)(const float* Q, const float* K, const float* V, float* O,
                       int B, int nh, int N, int d, ag_cuda_stream_t);
typedef void (*flow_flexflash_attention)(const float* Q, const float* K, const float* V, float* O,
                       int B, int nh, int N, int d, const float * ww, ag_cuda_stream_t);
typedef void (*flow_swiglu_cuda_fn)(const float* A, const float* B, const float* D, const float* F, const float* G, float* C, float* E, int M, int K, int N, // int W, 
  ag_cuda_stream_t s);




// NEW: VJP (backward) function types for CUDA
// Basic element-wise VJPs
typedef void (*flow_vjp_add_cuda_fn)(float* gA, float* gB, const float* gy,
                                   int64_t n, ag_cuda_stream_t s);
typedef void (*flow_vjp_sub_cuda_fn)(float* gA, float* gB, const float* gy,
                                   int64_t n, ag_cuda_stream_t s);
typedef void (*flow_vjp_hadmul_cuda_fn)(float* gA, float* gB, const float* gy,
                                   const float* A, const float* B,
                                   int64_t n, ag_cuda_stream_t s);
typedef void (*flow_vjp_div_cuda_fn)(float* gA, float* gB, const float* gy,
                                   const float* A, const float* B,
                                   int64_t n, ag_cuda_stream_t s);
typedef void (*flow_vjp_matmul_cuda_fn)(float* gA, float* gB, const float* gy,
                                      const float* A, const float* B,
                                      int M, int K, int N, ag_cuda_stream_t s);
typedef void (*flow_vjp_div_cuda_fn)(float* gA, float* gB, const float* gy,
                                   const float* A, const float* B,
                                   int64_t n, ag_cuda_stream_t s);
typedef void (*flow_vjp_mseloss_cuda_fn)(float* gA, float* gB, const float* gy,
                                   const float* A, const float* B,
                                   int64_t n, ag_cuda_stream_t s);
typedef void (*flow_vjp_maeloss_cuda_fn)(float* gA, float* gB, const float* gy,
                                   const float* A, const float* B,
                                   int64_t n, ag_cuda_stream_t s);

// Additional VJP function types for arithmetic ops
typedef void (*flow_vjp_pow_cuda_fn)(float* gA, float* gB, const float* gy,
                                  const float* A, const float* B, int64_t n,
                                  ag_cuda_stream_t s);
typedef void (*flow_vjp_square_cuda_fn)(float* gX, const float* gy, const float* X,
                                     int64_t n, ag_cuda_stream_t s);
typedef void (*flow_vjp_neg_cuda_fn)(float* gX, const float* gy,
                                  int64_t n, ag_cuda_stream_t s);
typedef void (*flow_vjp_clip_cuda_fn)(float* gX, const float* gy, float minv, float maxv,
                                   const float* X, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_vjp_sqrt_cuda_fn)(float* gX, const float* gy, const float* X,
                                     int64_t n, ag_cuda_stream_t s);

// VJP function types for activation functions
typedef void (*flow_vjp_leaky_relu_cuda_fn)(float* gX, const float* gy, const float* X,
                                         float alpha, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_vjp_sigmoid_cuda_fn)(float* gX, const float* gy, const float* X,
                                      int64_t n, ag_cuda_stream_t s);
typedef void (*flow_vjp_silu_cuda_fn)(float* gX, const float* gy, const float* X,
                                   int64_t n, ag_cuda_stream_t s);
typedef void (*flow_vjp_gelu_cuda_fn)(float* gX, const float* gy, const float* X,
                                   int64_t n, ag_cuda_stream_t s);
typedef void (*flow_vjp_mish_cuda_fn)(float* gX, const float* gy, const float* X,
                                   int64_t n, ag_cuda_stream_t s);
typedef void (*flow_vjp_exp_cuda_fn)(float* gX, const float* gy, const float* X,
                                  int64_t n, ag_cuda_stream_t s);
typedef void (*flow_vjp_log_cuda_fn)(float* gX, const float* gy, const float* X,
                                  int64_t n, ag_cuda_stream_t s);
typedef void (*flow_vjp_hard_sigmoid_cuda_fn)(float* gX, const float* gy, const float* X,
                                          int64_t n, ag_cuda_stream_t s);
typedef void (*flow_vjp_softplusback_cuda_fn )(float* gX, const float* gy, const float* X,
                                          int64_t n, ag_cuda_stream_t s);
typedef void (*flow_vjp_hard_swish_cuda_fn)(float* gX, const float* gy, const float* X,
                                         int64_t n, ag_cuda_stream_t s);
typedef void (*flow_vjp_relu_cuda_fn)(float* gX, const float* gy, const float* X, int64_t n, ag_cuda_stream_t s); 
typedef void (*flow_vjp_tanh_cuda_fn)(float* gX, const float* gy, const float* X, int64_t n, ag_cuda_stream_t s); 
typedef void (*flow_vjp_gemm_cuda_fn)(float* gA, float* gB, float* gC, const float* gy,
                                      const float* A, const float* B, const float* C,
                                      int M, int K, int N, ag_cuda_stream_t s);
typedef void (*flow_vjp_linear_cuda_fn)(float* gA, float* gB, float* gC, const float* gy,
                                      const float* A, const float* B, const float* C,
                                      int M, int K, int N, ag_cuda_stream_t s);
typedef void (*flow_vjp_sum_cuda_fn)(float* gX, const float* gy, const float* X,
                                   int64_t n, ag_cuda_stream_t s);

typedef void (*flow_vjp_softmax_cuda_fn)(float* gX, const float* gy, const float* Y,
                                   int64_t n, int64_t w, ag_cuda_stream_t s); 
typedef void (*flow_vjp_rowmax_cuda_fn)(float* gX, const float* gy, const float* X, const float* Y,
                                   int64_t n, int64_t w, ag_cuda_stream_t s);  
typedef void (*flow_vjp_rowsum_cuda_fn)(float* gX, const float* gy, const float* X, const float* Y,
                                   int64_t n, int64_t w, ag_cuda_stream_t s);  
typedef void (*flow_vjp_sin_cuda_fn)(float* gX, const float* gy, const float* X,
                                  int64_t n, ag_cuda_stream_t s);    
typedef void (*flow_vjp_cos_cuda_fn)(float* gX, const float* gy, const float* X,
                                  int64_t n, ag_cuda_stream_t s);      
typedef void (*flow_vjp_sinh_cuda_fn)(float* gX, const float* gy, const float* X,
                                  int64_t n, ag_cuda_stream_t s);    
typedef void (*flow_vjp_cosh_cuda_fn)(float* gX, const float* gy, const float* X,
                                  int64_t n, ag_cuda_stream_t s); 
// VJP types for the custom ops (gX, X, gy) order chosen to match plugin impl
typedef void (*flow_vjp_gcu_cuda_fn)(float* gX, const float* X, const float* gy, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_vjp_gauss_cuda_fn)(float* gX, const float* X, const float* gy, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_vjp_parcon_cuda_fn)(float* gX, const float* X, const float* gy, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_vjp_lisht_cuda_fn)(float* gX, const float* X, const float* gy, int64_t n, ag_cuda_stream_t s);
typedef void (*flow_vjp_reci_cuda_fn)(float* gX, const float* X, const float* gy, int64_t n, ag_cuda_stream_t s);

// CUDA function table
struct flow_cuda_v1 {
  uint32_t abi_version;
  // ========================================================
  // Forward ops
  // ========================================================

  // Arithmetic
  flow_add_cuda_fn     add;
  flow_sub_cuda_fn     sub;
  flow_hadmul_cuda_fn     hadmul;
  flow_div_cuda_fn     div;
  flow_pow_cuda_fn     pow;
  flow_square_cuda_fn  square;
  flow_neg_cuda_fn     neg;
  flow_clip_cuda_fn    clip;

  // Activations
  flow_relu_cuda_fn         relu;
  flow_leaky_relu_cuda_fn   leaky_relu;
  flow_gelu_cuda_fn         gelu;
  flow_silu_cuda_fn         silu;
  flow_mish_cuda_fn         mish;
  flow_tanh_cuda_fn         tanh;
  flow_sigmoid_cuda_fn      sigmoid;
  flow_hard_sigmoid_cuda_fn hard_sigmoid;
  flow_hard_swish_cuda_fn   hard_swish;
  flow_exp_cuda_fn          exp;
  flow_log_cuda_fn          log;
  flow_softplus_cuda_fn          softplus;
  flow_relumask_cuda_fn     relumask;
  flow_optim_cuda_fn    optim;
  flow_sqrt_cuda_fn    sqrt;

  // Core
  flow_zero_cuda_fn   zero;
  flow_matmul_cuda_fn matmul;
  flow_gemm_cuda_fn gemm;
  flow_linear_cuda_fn       linear;
  flow_flash_attention flash;
  flow_reluflash_attention reluflash;
  flow_sigflash_attention sigflash;
  flow_flexflash_attention flexflash;
  flow_sum_cuda_fn sum;
  flow_mseloss_cuda_fn        mseloss;
  flow_maeloss_cuda_fn        maeloss;
  flow_rowsum_cuda_fn        rowsum;
  flow_rowmax_cuda_fn        rowmax;
  // Custom elementwise ops
  flow_gcu_cuda_fn gcu ;
  flow_gauss_cuda_fn gauss;
  flow_parcon_cuda_fn parcon ;
  flow_lisht_cuda_fn lisht ;
  flow_reci_cuda_fn reci ;
  flow_dyntanh_cuda_fn dyntanh ;
  flow_flash_alibattention flashali;

  flow_sin_cuda_fn        sin; 
  flow_cos_cuda_fn        cos;
  flow_sinh_cuda_fn        sinh; 
  flow_cosh_cuda_fn        cosh;
  flow_sign_cuda_fn        sign; 
  flow_softmax_cuda_fn        softmax; 
  flow_swiglu_cuda_fn        swiglu;
  flow_logsumexp_cuda_fn        logsumexp;
  flow_cewithlogits_cuda_fn        cewithlogits;
  flow_kldivergence_cuda_fn        kldivergence;

  // ========================================================
  // Backward (VJP) ops
  // ========================================================
  // Basic ops VJPs
  flow_vjp_add_cuda_fn    vjp_add;
  flow_vjp_sub_cuda_fn    vjp_sub;
  flow_vjp_hadmul_cuda_fn vjp_hadmul;    // Hadamard multiplication VJP
  flow_vjp_div_cuda_fn    vjp_div;    // Element-wise division VJP
  flow_vjp_matmul_cuda_fn vjp_matmul;
  flow_vjp_relu_cuda_fn   vjp_relu;  
  flow_vjp_tanh_cuda_fn   vjp_tanh; 
  flow_vjp_gemm_cuda_fn   vjp_gemm;
  flow_vjp_linear_cuda_fn vjp_linear;

  // Arithmetic ops VJPs  
  flow_vjp_pow_cuda_fn    vjp_pow;
  flow_vjp_square_cuda_fn vjp_square;
  flow_vjp_neg_cuda_fn    vjp_neg;
  flow_vjp_clip_cuda_fn   vjp_clip;

  // Activation functions VJPs
  flow_vjp_leaky_relu_cuda_fn   vjp_leaky_relu;
  flow_vjp_sigmoid_cuda_fn      vjp_sigmoid;
  flow_vjp_silu_cuda_fn         vjp_silu;
  flow_vjp_gelu_cuda_fn         vjp_gelu;
  flow_vjp_mish_cuda_fn         vjp_mish;
  flow_vjp_exp_cuda_fn          vjp_exp;
  flow_vjp_hard_sigmoid_cuda_fn vjp_hard_sigmoid;
  flow_vjp_hard_swish_cuda_fn   vjp_hard_swish;
  flow_vjp_softplusback_cuda_fn    vjp_sofba;
  flow_vjp_log_cuda_fn        vjp_log       ;
  flow_vjp_sqrt_cuda_fn    vjp_sqrt;

  flow_vjp_mseloss_cuda_fn        vjp_mseloss;
  flow_vjp_sum_cuda_fn        vjp_sum;
  flow_vjp_maeloss_cuda_fn        vjp_maeloss;
  flow_vjp_rowmax_cuda_fn        vjp_rowmax;
  flow_vjp_rowsum_cuda_fn        vjp_rowsum;
  flow_vjp_sin_cuda_fn        vjp_sin; 
  flow_vjp_cos_cuda_fn        vjp_cos;
  flow_vjp_sinh_cuda_fn        vjp_sinh; 
  flow_vjp_cosh_cuda_fn        vjp_cosh;
  flow_vjp_softmax_cuda_fn        vjp_softmax; 

    flow_vjp_gcu_cuda_fn vjp_gcu ;
  flow_vjp_gauss_cuda_fn vjp_gauss ;
  flow_vjp_parcon_cuda_fn vjp_parcon ;
  flow_vjp_lisht_cuda_fn vjp_lisht ;
  flow_vjp_reci_cuda_fn vjp_reci ;

};

// Every CUDA plugin must export this symbol.
FLOW_EXPORT int flow_get_cuda_kernels_v1(struct flow_cuda_v1* out);

} // extern "C"

// ---------- C++ runtime registries & loaders ----------
namespace flow::kernel {

// CPU registry (yours – unchanged)
struct Cpu {
  flow_relu_fn   relu   = nullptr;
  flow_matmul_fn matmul = nullptr;

  
  flow_gelu_fn gelu = nullptr;
  flow_leakyrelu_fn leakyrelu = nullptr;
  flow_sigmoid_fn sigmoid = nullptr;
  flow_tanh_fn tanh = nullptr;
  flow_softplus_fn softplus = nullptr;
  flow_exp_fn exp = nullptr;
  flow_log_fn log = nullptr;
  flow_sqrt_fn sqrt = nullptr;
  flow_pow_fn pow = nullptr;
  flow_linear_fn linear = nullptr;
  //backwards
  elem_bwd_fn relu_bwd = nullptr;
  elem_bwd_alpha_fn leakyrelu_bwd = nullptr;
  elem_bwd_fn sigmoid_bwd_from_s = nullptr;
  elem_bwd_fn tanh_bwd_from_t = nullptr;
  elem_bwd_fn gelu_bwd = nullptr;
  elem_bwd_fn softplus_bwd = nullptr;
  elem_bwd_fn exp_bwd_from_y = nullptr;
  elem_bwd_fn log_bwd = nullptr;
  elem_bwd_fn sqrt_bwd_from_y = nullptr;
  // linear backward wrappers
 void (*matmul_bwd_dA)(const float*, const float*, float*, int M, int K, int N);
 void (*matmul_bwd_dB)(const float*, const float*, float*, int M, int K, int N);
  flow_linear_dW_fn linear_dW = nullptr;
  flow_linear_dX_fn linear_dX = nullptr;
  flow_linear_db_fn linear_db = nullptr;
};

// Global registry accessor
Cpu& cpu();

// Load a plugin and populate the registry
void load_cpu_plugin(const char* path);

// ---- NEW: CUDA registry ----
struct Cuda {
  // Forward
  // Arithmetic
  flow_add_cuda_fn     add    = nullptr;
  flow_sub_cuda_fn     sub    = nullptr;
  flow_hadmul_cuda_fn     hadmul    = nullptr;
  flow_div_cuda_fn     div    = nullptr;
  flow_pow_cuda_fn     pow    = nullptr;
  flow_square_cuda_fn  square = nullptr;
  flow_neg_cuda_fn     neg    = nullptr;
  flow_clip_cuda_fn    clip   = nullptr;
  flow_log_cuda_fn          log = nullptr;
  flow_relumask_cuda_fn     relumask = nullptr;
  flow_mseloss_cuda_fn        mseloss       = nullptr;
  flow_maeloss_cuda_fn        maeloss       = nullptr;

  // Activations
  flow_relu_cuda_fn         relu         = nullptr;
  flow_leaky_relu_cuda_fn   leaky_relu   = nullptr;
  flow_gelu_cuda_fn         gelu         = nullptr;
  flow_silu_cuda_fn         silu         = nullptr;
  flow_mish_cuda_fn         mish         = nullptr;
  flow_tanh_cuda_fn         tanh         = nullptr;
  flow_sigmoid_cuda_fn      sigmoid      = nullptr;
  flow_hard_sigmoid_cuda_fn hard_sigmoid = nullptr;
  flow_hard_swish_cuda_fn   hard_swish   = nullptr;
  flow_optim_cuda_fn    optim = nullptr;
  flow_exp_cuda_fn          exp          = nullptr;
  flow_gemm_cuda_fn         gemm         = nullptr;
  flow_linear_cuda_fn       linear         = nullptr;
  flow_sqrt_cuda_fn    sqrt   = nullptr;
  flow_dyntanh_cuda_fn dyntanh = nullptr;
  flow_flash_alibattention flashali = nullptr;

  flow_flash_attention flash = nullptr;
  flow_softplus_cuda_fn         softplus = nullptr;
  flow_reluflash_attention reluflash = nullptr;
  flow_sigflash_attention sigflash = nullptr;
  flow_flexflash_attention flexflash = nullptr;
  flow_sum_cuda_fn sum = nullptr;
  flow_rowsum_cuda_fn        rowsum       = nullptr;
  flow_rowmax_cuda_fn        rowmax       = nullptr;
  flow_sin_cuda_fn        sin = nullptr; 
  flow_cos_cuda_fn        cos = nullptr;
  flow_sinh_cuda_fn        sinh = nullptr; 
  flow_cosh_cuda_fn        cosh = nullptr;
  flow_sign_cuda_fn        sign = nullptr; 
  flow_softmax_cuda_fn        softmax = nullptr; 
  flow_swiglu_cuda_fn        swiglu = nullptr;
  flow_logsumexp_cuda_fn        logsumexp = nullptr;
  flow_cewithlogits_cuda_fn        cewithlogits = nullptr;
  flow_kldivergence_cuda_fn        kldivergence = nullptr;

  // Custom elementwise ops
  flow_gcu_cuda_fn gcu = nullptr;
  flow_gauss_cuda_fn gauss = nullptr;
  flow_parcon_cuda_fn parcon = nullptr;
  flow_lisht_cuda_fn lisht = nullptr;
  flow_reci_cuda_fn reci = nullptr;

  // Core
  flow_zero_cuda_fn   zero   = nullptr;
  flow_matmul_cuda_fn matmul = nullptr;
  // NEW: Backward
  // Basic ops VJPs
  flow_vjp_add_cuda_fn    vjp_add = nullptr;
  flow_vjp_sub_cuda_fn    vjp_sub = nullptr;
  flow_vjp_hadmul_cuda_fn    vjp_hadmul = nullptr;    // Hadamard multiplication VJP
  flow_vjp_div_cuda_fn    vjp_div = nullptr;    // Element-wise division VJP
  flow_vjp_matmul_cuda_fn vjp_matmul = nullptr;
  flow_vjp_relu_cuda_fn   vjp_relu   = nullptr;
  flow_vjp_tanh_cuda_fn   vjp_tanh   = nullptr;
  flow_vjp_gemm_cuda_fn   vjp_gemm = nullptr;
  flow_vjp_linear_cuda_fn vjp_linear = nullptr;

  // Arithmetic ops VJPs
  flow_vjp_pow_cuda_fn    vjp_pow = nullptr;
  flow_vjp_square_cuda_fn vjp_square = nullptr;
  flow_vjp_neg_cuda_fn    vjp_neg = nullptr;
  flow_vjp_clip_cuda_fn   vjp_clip = nullptr;

  // Activation functions VJPs
  flow_vjp_leaky_relu_cuda_fn   vjp_leaky_relu = nullptr;
  flow_vjp_sigmoid_cuda_fn      vjp_sigmoid = nullptr;
  flow_vjp_silu_cuda_fn         vjp_silu = nullptr;
  flow_vjp_gelu_cuda_fn         vjp_gelu = nullptr;
  flow_vjp_mish_cuda_fn         vjp_mish = nullptr;
  flow_vjp_exp_cuda_fn          vjp_exp = nullptr;
  flow_vjp_sqrt_cuda_fn    vjp_sqrt   = nullptr;

  flow_vjp_hard_sigmoid_cuda_fn vjp_hard_sigmoid = nullptr;
  flow_vjp_hard_swish_cuda_fn   vjp_hard_swish = nullptr;
  flow_vjp_softplusback_cuda_fn    vjp_sofba       = nullptr;
  flow_vjp_log_cuda_fn        vjp_log       = nullptr;
  flow_vjp_mseloss_cuda_fn        vjp_mseloss       = nullptr;
  flow_vjp_sum_cuda_fn        vjp_sum       = nullptr;
  flow_vjp_maeloss_cuda_fn        vjp_maeloss       = nullptr;
  flow_vjp_rowmax_cuda_fn        vjp_rowmax = nullptr; 
  flow_vjp_rowsum_cuda_fn        vjp_rowsum = nullptr;
  flow_vjp_sin_cuda_fn        vjp_sin = nullptr; 
  flow_vjp_cos_cuda_fn        vjp_cos = nullptr;
  flow_vjp_sinh_cuda_fn        vjp_sinh = nullptr; 
  flow_vjp_cosh_cuda_fn        vjp_cosh = nullptr;
  flow_vjp_softmax_cuda_fn        vjp_softmax = nullptr; 
  // Custom VJPs
  flow_vjp_gcu_cuda_fn vjp_gcu = nullptr;
  flow_vjp_gauss_cuda_fn vjp_gauss = nullptr;
  flow_vjp_parcon_cuda_fn vjp_parcon = nullptr;
  flow_vjp_lisht_cuda_fn vjp_lisht = nullptr;
  flow_vjp_reci_cuda_fn vjp_reci = nullptr;
};
Cuda& cuda();
void load_cuda_plugin(const char* path);

} // namespace flow::kernels