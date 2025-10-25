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
extern void vjp_matmul_cuda (float*, float*, const float*, const float*, const float*, int, int, int, ag_cuda_stream_t);
extern void vjp_relu_cuda   (float*, const float*, const float*, int64_t, ag_cuda_stream_t);
extern void vjp_tanh_cuda   (float*, const float*, const float*, int64_t, ag_cuda_stream_t);
extern void vjp_gemm_cuda (float*, float*, float*, const float*, const float*, const float*, const float*, int, int, int, ag_cuda_stream_t);
extern void vjp_linear_cuda (float*, float*, float*, const float*, const float*, const float*, const float*, int, int, int, ag_cuda_stream_t);


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
  out->vjp_matmul   = &vjp_matmul_cuda;
  out->vjp_relu     = &vjp_relu_cuda;
  out->vjp_tanh     = &vjp_tanh_cuda;
  out->vjp_gemm     = &vjp_gemm_cuda;
  out->vjp_linear     = &vjp_linear_cuda;

  return 0;
}
