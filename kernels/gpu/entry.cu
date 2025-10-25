// ==================================
// FILE: kernels/gpu/entry.cu
// ==================================
#include "ad/kernels_api.hpp"
#include <cstdint>

// ============================================================
// Forward declarations (external linkage)
// ============================================================

// Arithmetic
extern void add_cuda   (const float*, const float*, float*, int64_t, ag_cuda_stream_t);
extern void sub_cuda   (const float*, const float*, float*, int64_t, ag_cuda_stream_t);
extern void mul_cuda   (const float*, const float*, float*, int64_t, ag_cuda_stream_t);
extern void div_cuda   (const float*, const float*, float*, int64_t, ag_cuda_stream_t);
extern void pow_cuda   (const float*, const float*, float*, int64_t, ag_cuda_stream_t);
extern void square_cuda(const float*, float*, int64_t, ag_cuda_stream_t);
extern void neg_cuda   (const float*, float*, int64_t, ag_cuda_stream_t);
extern void clip_cuda  (const float*, float*, float, float, int64_t, ag_cuda_stream_t);

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

// Core ops
extern void zero_cuda (float*, int64_t, ag_cuda_stream_t);
extern void mm_cuda   (const float*, const float*, float*, int, int, int, ag_cuda_stream_t);
extern void gemm_cuda   (const float*, const float*, float*, float*, int, int, int, ag_cuda_stream_t);
extern void linear_cuda   (const float*, const float*, float*, float*, int, int, int, ag_cuda_stream_t);


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



// ============================================================
// Registration
// ============================================================
extern "C" AG_EXPORT int ag_get_cuda_kernels_v1(ag_cuda_v1* out) {
  if (!out) return -1;

  out->abi_version = AG_KERNELS_ABI_V1;

  // ========================================================
  // Forward
  // ========================================================
  out->add          = &add_cuda;
  out->sub          = &sub_cuda;
  out->mul          = &mul_cuda;
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

  out->zero         = &zero_cuda;
  out->matmul       = &mm_cuda;
  out->gemm       = &gemm_cuda;
  out->linear       = &linear_cuda;

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

  // Link VJP functions for activation functions
  out->vjp_leaky_relu = &vjp_leaky_relu_cuda;
  out->vjp_sigmoid    = &vjp_sigmoid_cuda;
  out->vjp_silu       = &vjp_silu_cuda;
  out->vjp_gelu       = &vjp_gelu_cuda;
  out->vjp_mish       = &vjp_mish_cuda;
  out->vjp_exp        = &vjp_exp_cuda;
  out->vjp_hard_sigmoid = &vjp_hard_sigmoid_cuda;
  out->vjp_hard_swish  = &vjp_hard_swish_cuda;

  return 0;
}
