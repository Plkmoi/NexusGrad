// cgadimpl/include/ag/kernels_api.hpp
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

// Bump when struct layout changes.
static const uint32_t AG_KERNELS_ABI_V1 = 1;

// Plain C function-pointer types (no Tensor types here)
typedef void (*ag_relu_fn)(const float* x, float* y, int64_t n);
typedef void (*ag_matmul_fn)(const float* A, const float* B, float* C,
                             int M, int K, int N);
typedef void (*ag_gemm_fn)(const float* A, const float* B, const float* C, float* E,
                             int M, int K, int N);
typedef void (*ag_relumask_fn)(const float* x, float* y, int64_t n);
typedef void (*ag_exp_fn)(const float* x, float* y, int64_t n);
typedef void (*ag_sig_fn)(const float* x, float* y, int64_t n);
typedef void (*ag_sigd_fn)(const float* x, float* y, int64_t n);
typedef void (*ag_add_fn)(const float* x, const float* z, float* y, int64_t n);
typedef void (*ag_sub_fn)(const float* x, const float* z, float* y, int64_t n);
typedef void (*ag_hadmul_fn)(const float* x, const float* z, float* y, int64_t n);
typedef void (*ag_flashatt_fn)(const float* Q, const float* K, const float* V,
    float* O, int B, int nh, int N, int d);
typedef void (*ag_div_fn)(const float* x, const float* z, float* y, int64_t n);
typedef void (*ag_gcu_fn)(const float* x, float* y, int64_t n);
typedef void (*ag_mish_fn)(const float* x, float* y, int64_t n);
typedef void (*ag_gaus_fn)(const float* x, float* y, int64_t n);
typedef void (*ag_parcon_fn)(const float* x, float* y, int64_t n);
typedef void (*ag_lisht_fn)(const float* x, float* y, int64_t n);
typedef void (*ag_softplus_fn)(const float* x, float* y, int64_t n);
typedef void (*ag_silu_fn)(const float* x, float* y, int64_t n);
typedef void (*ag_gelu_fn)(const float* x, float* y, int64_t n);
typedef void (*ag_log_fn)(const float* x, float* y, int64_t n);
typedef void (*ag_tanh_fn)(const float* x, float* y, int64_t n);
typedef void (*ag_rowsum_fn)(const float* X, float* Y, int rows, int cols);
typedef void (*ag_rowmax_fn)(const float* X, float* Y, int rows, int cols);
typedef void (*ag_leakyrelu_fn)(const float* x, float* h, float* y, int64_t n);
typedef void (*ag_flashatte_fn)(const float* Q, const float* K, const float* V,
    float* O, int B, int nh, int N, int d);


typedef void (*ag_vjp_add_cuda_fn)(float* gA, float* gB, const float* gy,
                                   int64_t n );
typedef void (*ag_vjp_matmul_cuda_fn)(float* gA, float* gB, const float* gy,
                                      const float* A, const float* B,
                                      int M, int K, int N );
typedef void (*ag_vjp_relu_cuda_fn)(float* gX, const float* gy, const float* X, int64_t n ); 
typedef void (*ag_vjp_tanh_cuda_fn)(float* gX, const float* gy, const float* X, int64_t n ); 
typedef void (*ag_vjp_gemm_cuda_fn)(float* gA, float* gB, float* gC, const float* gy,
                                      const float* A, const float* B, const float* C,
                                      int M, int K, int N );
typedef void (*ag_vjp_linear_cuda_fn)(float* gA, float* gB, float* gC, const float* gy,
                                      const float* A, const float* B, const float* C,
                                      int M, int K, int N );


// CPU function table (can be partially filled; nulls mean "not provided")
struct ag_cpu_v1 {
  uint32_t abi_version;   // must be AG_KERNELS_ABI_V1
  ag_relu_fn   relu;
  ag_matmul_fn matmul;
  ag_gemm_fn fmab;
  ag_relumask_fn   relumask;
  ag_exp_fn exp;
  ag_sig_fn sigmoid;
  ag_sigd_fn sigmoidiff;
    ag_add_fn add;
  ag_sub_fn sub;
  ag_hadmul_fn hadmul;
    ag_flashatt_fn flasha;
      ag_div_fn div;
  ag_gcu_fn gcu;
  ag_mish_fn mish;
  ag_gaus_fn gaus;
  ag_parcon_fn parcon;
  ag_lisht_fn lisht;
  ag_softplus_fn softplus;
  ag_silu_fn silu;
  ag_gelu_fn gelu;
  ag_log_fn log;
  ag_tanh_fn tanh;
  ag_rowsum_fn rowsum;
  ag_rowmax_fn rowmax;
  ag_leakyrelu_fn leakyrelu;
      ag_flashatte_fn flashae;
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

// Every CPU plugin must export this symbol.
AG_EXPORT int ag_get_cpu_kernels_v1(struct ag_cpu_v1* out);

} // extern "C"

// ---------- C++ runtime registry & loader ----------
namespace ag::kernels {

struct Cpu {
  ag_relu_fn   relu   = nullptr;
  ag_matmul_fn matmul = nullptr;
    ag_gemm_fn fmab = nullptr;
    ag_relumask_fn relumask = nullptr;
    ag_exp_fn exp = nullptr;
ag_sig_fn sigmoid = nullptr;
  ag_sigd_fn sigmoidiff = nullptr;
   ag_add_fn add = nullptr;
  ag_sub_fn sub = nullptr;
  ag_hadmul_fn hadmul = nullptr;
    ag_flashatt_fn flasha = nullptr;
    ag_div_fn div=nullptr;
      ag_gcu_fn gcu = nullptr;
  ag_mish_fn mish = nullptr;
  ag_gaus_fn gaus = nullptr;
  ag_parcon_fn parcon = nullptr;
  ag_lisht_fn lisht = nullptr;
  ag_softplus_fn softplus = nullptr;
  ag_silu_fn silu = nullptr;
  ag_gelu_fn gelu = nullptr;
  ag_log_fn log = nullptr;
  ag_tanh_fn tanh = nullptr;
  ag_rowsum_fn rowsum = nullptr;
  ag_rowmax_fn rowmax = nullptr;
  ag_leakyrelu_fn leakyrelu = nullptr;
        ag_flashatte_fn flashae = nullptr;
  ag_vjp_add_cuda_fn    vjp_add = nullptr;
  ag_vjp_matmul_cuda_fn vjp_matmul = nullptr;
  ag_vjp_relu_cuda_fn   vjp_relu   = nullptr;
  ag_vjp_tanh_cuda_fn   vjp_tanh   = nullptr;
  ag_vjp_gemm_cuda_fn vjp_gemm = nullptr;
  ag_vjp_linear_cuda_fn vjp_linear = nullptr;
};

// Global registry accessor
Cpu& cpu();

// Load a plugin and populate the registry
void load_cpu_plugin(const char* path);

} // namespace ag::kernels
