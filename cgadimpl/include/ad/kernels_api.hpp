// =========================================================
// FILE: cgadimpl/include/ad/kernels_api.hpp
// =========================================================
#pragma once
#include <cstdint>

#if defined(_WIN32)
  #define AG_EXPORT __declspec(dllexport)
  #define AG_IMPORT __declspec(dllimport)
#else
  #define AG_EXPORT __attribute__((visibility("default")))
  #define AG_IMPORT
#endif

// ---------- C ABI shared with plugins ----------
extern "C" {

// Keep existing:
static const uint32_t AG_KERNELS_ABI_V1 = 1;

// Plain C function-pointer types (no Tensor types here)
typedef void (*ag_relu_fn)(const float* x, float* y, int64_t n);
typedef void (*ag_matmul_fn)(const float* A, const float* B, float* C,
                             int M, int K, int N);                             
typedef void (*ag_gelu_fn)(const float* x, float* y, int64_t n);
typedef void (*ag_leakyrelu_fn)(const float* x, float* y, int64_t n, float alpha);
typedef void (*ag_sigmoid_fn)(const float* x, float* y, int64_t n);
typedef void (*ag_tanh_fn)(const float* x, float* y, int64_t n);
typedef void (*ag_softplus_fn)(const float* x, float* y, int64_t n);
typedef void (*ag_exp_fn)(const float* x, float* y, int64_t n);
typedef void (*ag_log_fn)(const float* x, float* y, int64_t n);
typedef void (*ag_sqrt_fn) (const float* x, float* y, int64_t n);
typedef void (*ag_pow_fn) (const float* x, float* y, int64_t n, float exponent);
typedef void (*ag_linear_fn)(const float* X,const float* W,const float* b,float* Y,int B,int In,int Out);
// CPU function table (can be partially filled; nulls mean "not provided")
typedef void (*elem_bwd_fn)(const float*, const float*, float*, int64_t);
typedef void (*elem_bwd_alpha_fn)(const float*, const float*, float*, int64_t, float);
typedef void (*ag_linear_dW_fn)(const float* X, const float* dY, float* dW, int B, int In, int Out);
typedef void (*ag_linear_dX_fn)(const float* dY, const float* W, float* dX, int B, int In, int Out);
typedef void (*ag_linear_db_fn)(const float* dY, float* db, int B, int Out);


void leakyrelu_bwd_impl_optimized(const float* x, const float* dY, float* dX, int64_t n, float alpha);
void sigmoid_bwd_impl_optimized_from_s(const float* s, const float* dY, float* dX, int64_t n);
void tanh_bwd_impl_optimized_from_t(const float* t, const float* dY, float* dX, int64_t n);
void gelu_bwd_impl_optimized(const float* x, const float* dY, float* dX, int64_t n);
void softplus_bwd_impl_optimized(const float* x, const float* dY, float* dX, int64_t n);
void exp_bwd_impl_optimized_from_y(const float* y, const float* dY, float* dX, int64_t n);
void log_bwd_impl_optimized(const float* x, const float* dY, float* dX, int64_t n);
void sqrt_bwd_impl_optimized_from_y(const float* y, const float* dY, float* dX, int64_t n);
// void matmul_bwd_dA_impl_optimized(const float* dC, const float* B, float* dA, int M, int K, int N);
// void matmul_bwd_dB_impl_optimized(const float* A, const float* dC, float* dB, int M, int K, int N);
void linear_dW_impl_optimized(const float* X, const float* dY, float* dW, int B, int In, int Out);
void linear_dX_impl_optimized(const float* dY, const float* W, float* dX, int B, int In, int Out);
void linear_db_impl_optimized(const float* dY, float* db, int B, int Out);
void relu_bwd_impl_optimized(const float* x, const float* dY, float* dX, int64_t n);


// CPU function table (can be partially filled; nulls mean "not provided")
struct ag_cpu_v1 {
  uint32_t abi_version;   // must be AG_KERNELS_ABI_V1
  ag_relu_fn   relu; //done
  ag_matmul_fn matmul; //done
  ag_gelu_fn gelu;   //done
  ag_leakyrelu_fn leakyrelu;   //done
  ag_sigmoid_fn sigmoid; //done
  ag_tanh_fn tanh; //done
  ag_softplus_fn softplus; //done
  ag_exp_fn exp;  //done
  ag_log_fn log; //done
  ag_sqrt_fn sqrt; //done
  ag_pow_fn pow;
  ag_linear_fn linear;
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
  // void (*matmul_bwd_dA)(const float*, const float*, float*, int M, int K, int N);   ***
  // void (*matmul_bwd_dB)(const float*, const float*, float*, int M, int K, int N);   ***
  ag_linear_dW_fn linear_dW;
  ag_linear_dX_fn linear_dX;  
  ag_linear_db_fn linear_db;

};


AG_EXPORT int ag_get_cpu_kernels_v1(struct ag_cpu_v1* out);

// ---- NEW: CUDA function pointer types (accept a stream) ----
// Avoid pulling in CUDA headers here: just forward-declare the opaque type.
typedef struct CUstream_st* ag_cuda_stream_t;


// Arithmetic
typedef void (*ag_add_cuda_fn)   (const float* a, const float* b, float* c, int64_t n, ag_cuda_stream_t s);
typedef void (*ag_sub_cuda_fn)   (const float* a, const float* b, float* c, int64_t n, ag_cuda_stream_t s);
typedef void (*ag_mul_cuda_fn)   (const float* a, const float* b, float* c, int64_t n, ag_cuda_stream_t s);
typedef void (*ag_div_cuda_fn)   (const float* a, const float* b, float* c, int64_t n, ag_cuda_stream_t s);
typedef void (*ag_pow_cuda_fn)   (const float* a, const float* b, float* c, int64_t n, ag_cuda_stream_t s);
typedef void (*ag_square_cuda_fn)(const float* x, float* y, int64_t n, ag_cuda_stream_t s);
typedef void (*ag_neg_cuda_fn)   (const float* x, float* y, int64_t n, ag_cuda_stream_t s);
typedef void (*ag_clip_cuda_fn)  (const float* x, float* y, float minv, float maxv, int64_t n, ag_cuda_stream_t s);

// Activations
typedef void (*ag_relu_cuda_fn)        (const float* x, float* y, int64_t n, ag_cuda_stream_t s);
typedef void (*ag_leaky_relu_cuda_fn)  (const float* x, float* y, float alpha, int64_t n, ag_cuda_stream_t s);
typedef void (*ag_gelu_cuda_fn)        (const float* x, float* y, int64_t n, ag_cuda_stream_t s);
typedef void (*ag_silu_cuda_fn)        (const float* x, float* y, int64_t n, ag_cuda_stream_t s);
typedef void (*ag_mish_cuda_fn)        (const float* x, float* y, int64_t n, ag_cuda_stream_t s);
typedef void (*ag_tanh_cuda_fn)        (const float* x, float* y, int64_t n, ag_cuda_stream_t s);
typedef void (*ag_sigmoid_cuda_fn)     (const float* x, float* y, int64_t n, ag_cuda_stream_t s);
typedef void (*ag_hard_sigmoid_cuda_fn)(const float* x, float* y, int64_t n, ag_cuda_stream_t s);
typedef void (*ag_hard_swish_cuda_fn)  (const float* x, float* y, int64_t n, ag_cuda_stream_t s);
typedef void (*ag_exp_cuda_fn)         (const float* x, float* y, int64_t n, ag_cuda_stream_t s);

// Core
typedef void (*ag_zero_cuda_fn)  (float* x, int64_t n, ag_cuda_stream_t s);
typedef void (*ag_matmul_cuda_fn)(const float* A, const float* B, float* C, int M, int K, int N, ag_cuda_stream_t s);
typedef void (*ag_gemm_cuda_fn)(const float* A, const float* B, float* C, float* E, int M, int K, int N, ag_cuda_stream_t s);
typedef void (*ag_linear_cuda_fn)(const float* A, const float* B, float* C, float* E, int M, int K, int N, ag_cuda_stream_t s);

// NEW: VJP (backward) function types for CUDA
typedef void (*ag_vjp_add_cuda_fn)(float* gA, float* gB, const float* gy,
                                   int64_t n, ag_cuda_stream_t s);
typedef void (*ag_vjp_matmul_cuda_fn)(float* gA, float* gB, const float* gy,
                                      const float* A, const float* B,
                                      int M, int K, int N, ag_cuda_stream_t s);
typedef void (*ag_vjp_relu_cuda_fn)(float* gX, const float* gy, const float* X, int64_t n, ag_cuda_stream_t s); 
typedef void (*ag_vjp_tanh_cuda_fn)(float* gX, const float* gy, const float* X, int64_t n, ag_cuda_stream_t s); 
typedef void (*ag_vjp_gemm_cuda_fn)(float* gA, float* gB, float* gC, const float* gy,
                                      const float* A, const float* B, const float* C,
                                      int M, int K, int N, ag_cuda_stream_t s);
typedef void (*ag_vjp_linear_cuda_fn)(float* gA, float* gB, float* gC, const float* gy,
                                      const float* A, const float* B, const float* C,
                                      int M, int K, int N, ag_cuda_stream_t s);
// CUDA function table
struct ag_cuda_v1 {
  uint32_t abi_version;
  // ========================================================
  // Forward ops
  // ========================================================

  // Arithmetic
  ag_add_cuda_fn     add;
  ag_sub_cuda_fn     sub;
  ag_mul_cuda_fn     mul;
  ag_div_cuda_fn     div;
  ag_pow_cuda_fn     pow;
  ag_square_cuda_fn  square;
  ag_neg_cuda_fn     neg;
  ag_clip_cuda_fn    clip;

  // Activations
  ag_relu_cuda_fn         relu;
  ag_leaky_relu_cuda_fn   leaky_relu;
  ag_gelu_cuda_fn         gelu;
  ag_silu_cuda_fn         silu;
  ag_mish_cuda_fn         mish;
  ag_tanh_cuda_fn         tanh;
  ag_sigmoid_cuda_fn      sigmoid;
  ag_hard_sigmoid_cuda_fn hard_sigmoid;
  ag_hard_swish_cuda_fn   hard_swish;
  ag_exp_cuda_fn          exp;

  // Core
  ag_zero_cuda_fn   zero;
  ag_matmul_cuda_fn matmul;
  ag_gemm_cuda_fn gemm;
  ag_linear_cuda_fn       linear;

  // ========================================================
  // Backward (VJP) ops
  // ========================================================
  ag_vjp_add_cuda_fn    vjp_add;
  ag_vjp_matmul_cuda_fn vjp_matmul;
  ag_vjp_relu_cuda_fn   vjp_relu;  
  ag_vjp_tanh_cuda_fn   vjp_tanh; 
  ag_vjp_gemm_cuda_fn vjp_gemm;
  ag_vjp_linear_cuda_fn vjp_linear;
};

// Every CUDA plugin must export this symbol.
AG_EXPORT int ag_get_cuda_kernels_v1(struct ag_cuda_v1* out);

} // extern "C"

// ---------- C++ runtime registries & loaders ----------
namespace ag::kernels {

// CPU registry (yours – unchanged)
struct Cpu {
  ag_relu_fn   relu   = nullptr;
  ag_matmul_fn matmul = nullptr;

  
  ag_gelu_fn gelu = nullptr;
  ag_leakyrelu_fn leakyrelu = nullptr;
  ag_sigmoid_fn sigmoid = nullptr;
  ag_tanh_fn tanh = nullptr;
  ag_softplus_fn softplus = nullptr;
  ag_exp_fn exp = nullptr;
  ag_log_fn log = nullptr;
  ag_sqrt_fn sqrt = nullptr;
  ag_pow_fn pow = nullptr;
  ag_linear_fn linear = nullptr;
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
  // void (*matmul_bwd_dA)(const float*, const float*, float*, int M, int K, int N);
  // void (*matmul_bwd_dB)(const float*, const float*, float*, int M, int K, int N);
  ag_linear_dW_fn linear_dW = nullptr;
  ag_linear_dX_fn linear_dX = nullptr;
  ag_linear_db_fn linear_db = nullptr;
};

// Global registry accessor
Cpu& cpu();

// Load a plugin and populate the registry
void load_cpu_plugin(const char* path);

// ---- NEW: CUDA registry ----
struct Cuda {
  // Forward
  // Arithmetic
  ag_add_cuda_fn     add    = nullptr;
  ag_sub_cuda_fn     sub    = nullptr;
  ag_mul_cuda_fn     mul    = nullptr;
  ag_div_cuda_fn     div    = nullptr;
  ag_pow_cuda_fn     pow    = nullptr;
  ag_square_cuda_fn  square = nullptr;
  ag_neg_cuda_fn     neg    = nullptr;
  ag_clip_cuda_fn    clip   = nullptr;

  // Activations
  ag_relu_cuda_fn         relu         = nullptr;
  ag_leaky_relu_cuda_fn   leaky_relu   = nullptr;
  ag_gelu_cuda_fn         gelu         = nullptr;
  ag_silu_cuda_fn         silu         = nullptr;
  ag_mish_cuda_fn         mish         = nullptr;
  ag_tanh_cuda_fn         tanh         = nullptr;
  ag_sigmoid_cuda_fn      sigmoid      = nullptr;
  ag_hard_sigmoid_cuda_fn hard_sigmoid = nullptr;
  ag_hard_swish_cuda_fn   hard_swish   = nullptr;
  ag_exp_cuda_fn          exp          = nullptr;
  ag_gemm_cuda_fn         gemm         = nullptr;
  ag_linear_cuda_fn       linear         = nullptr;

  // Core
  ag_zero_cuda_fn   zero   = nullptr;
  ag_matmul_cuda_fn matmul = nullptr;
  // NEW: Backward
  ag_vjp_add_cuda_fn    vjp_add = nullptr;
  ag_vjp_matmul_cuda_fn vjp_matmul = nullptr;
  ag_vjp_relu_cuda_fn   vjp_relu   = nullptr;
  ag_vjp_tanh_cuda_fn   vjp_tanh   = nullptr;
  ag_vjp_gemm_cuda_fn vjp_gemm = nullptr;
  ag_vjp_linear_cuda_fn vjp_linear = nullptr;
};
Cuda& cuda();
void load_cuda_plugin(const char* path);

} // namespace ag::kernels