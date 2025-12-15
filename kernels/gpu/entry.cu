// ==================================
// FILE: kernels/gpu/entry.cu
// ==================================
#include "kernels/kernels.hpp"
#include <cstdint>


// ============================================================
// Forward declarations (external linkage)
// ============================================================

// Arithmetic
extern void run_flashflex_forwardz(const float* , const float* , const float* , float* ,
                                     int , int , int , int , const float * , cudaStream_t );
extern void add_cuda   (const float*, const float*, float*, int64_t, ag_cuda_stream_t);
extern void sub_cuda   (const float*, const float*, float*, int64_t, ag_cuda_stream_t);
extern void hadmul_cuda   (const float*, const float*, float*, int64_t, ag_cuda_stream_t);
extern void div_cuda   (const float*, const float*, float*, int64_t, ag_cuda_stream_t);
extern void pow_cuda   (const float*, const float*, float*, int64_t, ag_cuda_stream_t);
extern void square_cuda(const float*, float*, int64_t, ag_cuda_stream_t);
extern void neg_cuda   (const float*, float*, int64_t, ag_cuda_stream_t);
extern void clip_cuda  (const float*, float*, float, float, int64_t, ag_cuda_stream_t);
extern void mseloss_cuda   (const float*, const float*, float*, int64_t, ag_cuda_stream_t);
extern void maeloss_cuda   (const float*, const float*, float*, int64_t, ag_cuda_stream_t);
extern void dyntanh_cuda(const float* , float , float , float , float* , int64_t , ag_cuda_stream_t );

// Activations
extern void relu_cuda        (const float*, float*, int64_t, ag_cuda_stream_t);
extern void leaky_relu_cuda  (const float*, float*, float, int64_t, ag_cuda_stream_t);
extern void gelu_cuda        (const float*, float*, int64_t, ag_cuda_stream_t);
extern void silu_cuda        (const float*, float*, int64_t, ag_cuda_stream_t);
extern void mish_cuda        (const float*, float*, int64_t, ag_cuda_stream_t);
extern void tanh_cuda        (const float*, float*, int64_t, ag_cuda_stream_t);
extern void sigmoid_cuda     (const float*, float*, int64_t, ag_cuda_stream_t);
extern void hard_sigmoid_cuda(const float*, float*, int64_t, ag_cuda_stream_t);
extern void hard_swish_cuda  (const float*, float*, int64_t, ag_cuda_stream_t);
extern void exp_cuda         (const float*, float*, int64_t, ag_cuda_stream_t);
extern void sqrt_cuda         (const float*, float*, int64_t, ag_cuda_stream_t);

extern void log_cuda         (const float*, float*, int64_t, ag_cuda_stream_t);
extern void softplus_cuda    (const float*, float*, int64_t, ag_cuda_stream_t);
extern void relumask_cuda    (const float*, float*, int64_t, ag_cuda_stream_t);
extern void sin_cuda         (const float*, float*, int64_t, ag_cuda_stream_t);
extern void cos_cuda         (const float*, float*, int64_t, ag_cuda_stream_t);
extern void sinh_cuda         (const float*, float*, int64_t, ag_cuda_stream_t);
extern void cosh_cuda         (const float*, float*, int64_t, ag_cuda_stream_t);
extern void sign_cuda         (const float*, float*, int64_t, ag_cuda_stream_t);
extern void gcu_cuda         (const float*, float*, int64_t, ag_cuda_stream_t);
extern void gauss_cuda       (const float*, float*, int64_t, ag_cuda_stream_t);
extern void parcon_cuda      (const float*, float*, int64_t, ag_cuda_stream_t);
extern void lisht_cuda       (const float*, float*, int64_t, ag_cuda_stream_t);
extern void reci_cuda        (const float*, float*, int64_t, ag_cuda_stream_t);

// Core ops
extern void zero_cuda (float*, int64_t, ag_cuda_stream_t);
extern void mm_cuda   (const float*, const float*, float*, int, int, int, ag_cuda_stream_t);
extern void gemm_cuda   (const float*, const float*, float*, float*, int, int, int, ag_cuda_stream_t);
extern void linear_cuda   (const float*, const float*, float*, float*, int, int, int, ag_cuda_stream_t);
extern void run_flash_forward(const float* Q, const float* K, const float* V, float* O,
                       int B, int nh, int N, int d, ag_cuda_stream_t);
extern void run_flash_aliforward(const float* Q, const float* K, const float* V, float* O,
                       int B, int nh, int N, int d, ag_cuda_stream_t);
extern void run_flashrelu_forward(const float* Q, const float* K, const float* V, float* O,
                       int B, int nh, int N, int d, ag_cuda_stream_t);
extern void run_flashsig_forward(const float* Q, const float* K, const float* V, float* O,
                       int B, int nh, int N, int d, ag_cuda_stream_t);
extern void sumall_cuda(const float* a, float* c, int64_t n, ag_cuda_stream_t s);
extern void rowsum_cuda(const float* a, float* c, int64_t n, int64_t w, ag_cuda_stream_t s);
extern void rowmax_cuda(const float* a, float* c, int64_t n, int64_t w, ag_cuda_stream_t s);
extern void softmax_cuda(const float* a, float* c, int64_t n, int64_t w, ag_cuda_stream_t s);
extern void logsumexp_cuda(const float* a, float* c, int64_t n, int64_t w, ag_cuda_stream_t s);
extern void cewithlogits_cuda(const float* a, const float* r, float* c, int64_t n, int64_t w, ag_cuda_stream_t s);
extern void kldivergence_cuda(const float* a, const float* r, float* c, int64_t n, int64_t w, ag_cuda_stream_t s);

// ============================================================
// Backward / VJP declarations
// ============================================================
extern void vjp_add_cuda    (float*, float*, const float*, int64_t, ag_cuda_stream_t);
extern void vjp_hadmul_cuda    (float*, float*, const float*, const float*, const float*, int64_t, ag_cuda_stream_t);
extern void vjp_div_cuda    (float*, float*, const float*, const float*, const float*, int64_t, ag_cuda_stream_t);
extern void vjp_matmul_cuda (float*, float*, const float*, const float*, const float*, int, int, int, ag_cuda_stream_t);
extern void vjp_relu_cuda   (float*, const float*, const float*, int64_t, ag_cuda_stream_t);
extern void vjp_tanh_cuda   (float*, const float*, const float*, int64_t, ag_cuda_stream_t);
extern void vjp_gemm_cuda (float*, float*, float*, const float*, const float*, const float*, const float*, int, int, int, ag_cuda_stream_t);
extern void vjp_linear_cuda (float*, float*, float*, const float*, const float*, const float*, const float*, int, int, int, ag_cuda_stream_t);
extern void vjp_sub_cuda    (float*, float*, const float*, int64_t, ag_cuda_stream_t);
extern void vjp_mseloss_cuda    (float*, float*, const float*, const float*, const float*, int64_t, ag_cuda_stream_t);
extern void vjp_maeloss_cuda    (float*, float*, const float*, const float*, const float*, int64_t, ag_cuda_stream_t);

// =========================================================
// All CUDA VJP (Backward) Kernel Declarations
// =========================================================

extern void vjp_pow_cuda           (float*, float*, const float*, const float*, const float*, int64_t, ag_cuda_stream_t);
extern void vjp_square_cuda        (float*, const float*, const float*, int64_t, ag_cuda_stream_t);
extern void vjp_neg_cuda           (float*, const float*, int64_t, ag_cuda_stream_t);
extern void vjp_clip_cuda          (float*, const float*, float, float, const float*, int64_t, ag_cuda_stream_t);

extern void vjp_leaky_relu_cuda    (float*, const float*, const float*, float, int64_t, ag_cuda_stream_t);
extern void vjp_sigmoid_cuda       (float*, const float*, const float*, int64_t, ag_cuda_stream_t);
extern void vjp_silu_cuda          (float*, const float*, const float*, int64_t, ag_cuda_stream_t);
extern void vjp_gelu_cuda          (float*, const float*, const float*, int64_t, ag_cuda_stream_t);
extern void vjp_mish_cuda          (float*, const float*, const float*, int64_t, ag_cuda_stream_t);
extern void vjp_exp_cuda           (float*, const float*, const float*, int64_t, ag_cuda_stream_t);
extern void vjp_hard_sigmoid_cuda  (float*, const float*, const float*, int64_t, ag_cuda_stream_t);
extern void vjp_hard_swish_cuda    (float*, const float*, const float*, int64_t, ag_cuda_stream_t);
extern void vjp_sofba_cuda       (float*, const float*, const float*, int64_t, ag_cuda_stream_t);
extern void vjp_sqrt_cuda           (float*, const float*, const float*, int64_t, ag_cuda_stream_t);

extern void vjp_log_cuda       (float*, const float*, const float*, int64_t, ag_cuda_stream_t);
extern void vjp_sum_cuda           (float*, const float*, const float*, int64_t, ag_cuda_stream_t);
extern void vjp_sin_cuda           (float*, const float*, const float*, int64_t, ag_cuda_stream_t);
extern void vjp_cos_cuda           (float*, const float*, const float*, int64_t, ag_cuda_stream_t);
extern void vjp_sinh_cuda           (float*, const float*, const float*, int64_t, ag_cuda_stream_t);
extern void vjp_cosh_cuda           (float*, const float*, const float*, int64_t, ag_cuda_stream_t);
extern void vjp_gcu_cuda           (float*, const float*, const float*, int64_t, ag_cuda_stream_t);
extern void vjp_gauss_cuda         (float*, const float*, const float*, int64_t, ag_cuda_stream_t);
extern void vjp_parcon_cuda        (float*, const float*, const float*, int64_t, ag_cuda_stream_t);
extern void vjp_lisht_cuda         (float*, const float*, const float*, int64_t, ag_cuda_stream_t);
extern void vjp_reci_cuda          (float*, const float*, const float*, int64_t, ag_cuda_stream_t);

extern void vjp_rowmax_cuda(float* , const float* , const float* , const float* , int64_t , int64_t , ag_cuda_stream_t ); 
extern void vjp_rowsum_cuda(float* , const float* , const float* , const float* , int64_t , int64_t , ag_cuda_stream_t ); 
extern void vjp_softmax_cuda(float* , const float* , const float* , int64_t , int64_t , ag_cuda_stream_t ); 
extern void swiglu_cuda(const float* , const float* , const float* , const float* , const float* , float* , float* ,
                 int , int , int , //int , // W is used for, 
                 ag_cuda_stream_t );
extern void optim_cuda(float* , const float* , const float , int64_t , ag_cuda_stream_t );

// ============================================================
// Registration
// ============================================================
extern "C" FLOW_EXPORT int flow_get_cuda_kernels_v1(flow_cuda_v1* out) {
  if (!out) return -1;

  out->abi_version = FLOW_KERNELS_ABI_V1;

  // ========================================================
  // Forward
  // ========================================================
  out->add          = &add_cuda;
  out->sub          = &sub_cuda;
  out->hadmul          = &hadmul_cuda;
  out->div          = &div_cuda;
  out->pow          = &pow_cuda;
  out->square       = &square_cuda;
  out->neg          = &neg_cuda;
  out->clip         = &clip_cuda;

  out->relu         = &relu_cuda;
  out->leaky_relu   = &leaky_relu_cuda;
  out->gelu         = &gelu_cuda;
  out->silu         = &silu_cuda;
  out->mish         = &mish_cuda;
  out->tanh         = &tanh_cuda;
  out->sigmoid      = &sigmoid_cuda;
  out->hard_sigmoid = &hard_sigmoid_cuda;
  out->hard_swish   = &hard_swish_cuda;
  out->exp          = &exp_cuda;
  out->log          = &log_cuda;
  out->softplus          = &softplus_cuda;

  out->zero         = &zero_cuda;
  out->matmul       = &mm_cuda;
  out->gemm       = &gemm_cuda;
  out->linear       = &linear_cuda;
  out->relumask      = &relumask_cuda;

  out->flash         = &run_flash_forward;
  out->reluflash = &run_flashrelu_forward;
  out->sigflash        = &run_flashsig_forward;
  out->flexflash        = &run_flashflex_forwardz;
  out->flashali         = &run_flash_aliforward;

  out->sum = &sumall_cuda;
  out->mseloss       = &mseloss_cuda;
  out->maeloss       = &maeloss_cuda;
  out->rowsum = &rowsum_cuda;
  out->rowmax = &rowmax_cuda;
  out->sin          = &sin_cuda;
  out->cos          = &cos_cuda;
  out->sinh          = &sinh_cuda;
  out->cosh          = &cosh_cuda;
  out->optim = &optim_cuda;
  out->sqrt = &sqrt_cuda;
  out->dyntanh = &dyntanh_cuda;

  out->gcu          = &gcu_cuda;
  out->gauss        = &gauss_cuda;
  out->parcon       = &parcon_cuda;
  out->lisht        = &lisht_cuda;
  out->reci         = &reci_cuda;
  out->sign          = &sign_cuda;
  out->softmax          = &softmax_cuda;
  out->logsumexp          = &logsumexp_cuda;
  out->swiglu = &swiglu_cuda;
  out->cewithlogits = &cewithlogits_cuda;
  out->kldivergence = &kldivergence_cuda;

  // ========================================================
  // Backward (VJP)
  // ========================================================
  out->vjp_add      = &vjp_add_cuda;
  out->vjp_sub      = &vjp_sub_cuda;
  out->vjp_hadmul      = &vjp_hadmul_cuda;    // Hadamard multiplication VJP
  out->vjp_div      = &vjp_div_cuda;    // Element-wise division VJP
  out->vjp_matmul   = &vjp_matmul_cuda;
  out->vjp_relu     = &vjp_relu_cuda;
  out->vjp_tanh     = &vjp_tanh_cuda;
  out->vjp_gemm     = &vjp_gemm_cuda;
  out->vjp_linear   = &vjp_linear_cuda;
  
  // Link additional VJP functions for arithmetic ops
  out->vjp_pow      = &vjp_pow_cuda;
  out->vjp_square   = &vjp_square_cuda;
  out->vjp_neg      = &vjp_neg_cuda;
  out->vjp_clip     = &vjp_clip_cuda;
  out->vjp_mseloss     = &vjp_mseloss_cuda;
  out->vjp_maeloss     = &vjp_maeloss_cuda;

  // Link VJP functions for activation functions
  out->vjp_leaky_relu = &vjp_leaky_relu_cuda;
  out->vjp_sigmoid    = &vjp_sigmoid_cuda;
  out->vjp_silu       = &vjp_silu_cuda;
  out->vjp_gelu       = &vjp_gelu_cuda;
  out->vjp_mish       = &vjp_mish_cuda;
  out->vjp_exp        = &vjp_exp_cuda;
  out->vjp_hard_sigmoid = &vjp_hard_sigmoid_cuda;
  out->vjp_hard_swish  = &vjp_hard_swish_cuda;
  out->vjp_sofba       = &vjp_sofba_cuda;
  out->vjp_log        = &vjp_log_cuda;

  out->vjp_sum = &vjp_sum_cuda;
  out->vjp_rowmax = &vjp_rowmax_cuda;
  out->vjp_rowsum = &vjp_rowsum_cuda;
  out->vjp_sin        = &vjp_sin_cuda;
  out->vjp_cos        = &vjp_cos_cuda;
  out->vjp_sinh        = &vjp_sinh_cuda;
  out->vjp_cosh        = &vjp_cosh_cuda;  
  out->vjp_softmax        = &vjp_softmax_cuda;  
  // Custom elementwise VJPs
  out->vjp_gcu = &vjp_gcu_cuda;
  out->vjp_gauss = &vjp_gauss_cuda;
  out->vjp_parcon = &vjp_parcon_cuda;
  out->vjp_lisht = &vjp_lisht_cuda;
  out->vjp_reci = &vjp_reci_cuda;
  out->vjp_sqrt = &vjp_sqrt_cuda;

  return 0;
}