/*
  This file contains docstrings for use in the Python bindings.
  Do not edit! They were automatically extracted by pybind11_mkdoc.
 */

#define __EXPAND(x)                                      x
#define __COUNT(_1, _2, _3, _4, _5, _6, _7, COUNT, ...)  COUNT
#define __VA_SIZE(...)                                   __EXPAND(__COUNT(__VA_ARGS__, 7, 6, 5, 4, 3, 2, 1))
#define __CAT1(a, b)                                     a ## b
#define __CAT2(a, b)                                     __CAT1(a, b)
#define __DOC1(n1)                                       __doc_##n1
#define __DOC2(n1, n2)                                   __doc_##n1##_##n2
#define __DOC3(n1, n2, n3)                               __doc_##n1##_##n2##_##n3
#define __DOC4(n1, n2, n3, n4)                           __doc_##n1##_##n2##_##n3##_##n4
#define __DOC5(n1, n2, n3, n4, n5)                       __doc_##n1##_##n2##_##n3##_##n4##_##n5
#define __DOC6(n1, n2, n3, n4, n5, n6)                   __doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6
#define __DOC7(n1, n2, n3, n4, n5, n6, n7)               __doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6##_##n7
#define DOC(...)                                         __EXPAND(__EXPAND(__CAT2(__DOC, __VA_SIZE(__VA_ARGS__)))(__VA_ARGS__))

#if defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif


static const char *__doc_ag_kernels_Cpu = R"doc()doc";

static const char *__doc_ag_kernels_Cpu_exp = R"doc()doc";

static const char *__doc_ag_kernels_Cpu_exp_bwd_from_y = R"doc()doc";

static const char *__doc_ag_kernels_Cpu_gelu = R"doc()doc";

static const char *__doc_ag_kernels_Cpu_gelu_bwd = R"doc()doc";

static const char *__doc_ag_kernels_Cpu_leakyrelu = R"doc()doc";

static const char *__doc_ag_kernels_Cpu_leakyrelu_bwd = R"doc()doc";

static const char *__doc_ag_kernels_Cpu_linear = R"doc()doc";

static const char *__doc_ag_kernels_Cpu_linear_dW = R"doc()doc";

static const char *__doc_ag_kernels_Cpu_linear_dX = R"doc()doc";

static const char *__doc_ag_kernels_Cpu_linear_db = R"doc()doc";

static const char *__doc_ag_kernels_Cpu_log = R"doc()doc";

static const char *__doc_ag_kernels_Cpu_log_bwd = R"doc()doc";

static const char *__doc_ag_kernels_Cpu_matmul = R"doc()doc";

static const char *__doc_ag_kernels_Cpu_matmul_bwd_dA = R"doc()doc";

static const char *__doc_ag_kernels_Cpu_matmul_bwd_dB = R"doc()doc";

static const char *__doc_ag_kernels_Cpu_pow = R"doc()doc";

static const char *__doc_ag_kernels_Cpu_relu = R"doc()doc";

static const char *__doc_ag_kernels_Cpu_relu_bwd = R"doc()doc";

static const char *__doc_ag_kernels_Cpu_sigmoid = R"doc()doc";

static const char *__doc_ag_kernels_Cpu_sigmoid_bwd_from_s = R"doc()doc";

static const char *__doc_ag_kernels_Cpu_softplus = R"doc()doc";

static const char *__doc_ag_kernels_Cpu_softplus_bwd = R"doc()doc";

static const char *__doc_ag_kernels_Cpu_sqrt = R"doc()doc";

static const char *__doc_ag_kernels_Cpu_sqrt_bwd_from_y = R"doc()doc";

static const char *__doc_ag_kernels_Cpu_tanh = R"doc()doc";

static const char *__doc_ag_kernels_Cpu_tanh_bwd_from_t = R"doc()doc";

static const char *__doc_ag_kernels_Cuda = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_add = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_clip = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_cos = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_cosh = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_div = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_dyntanh = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_exp = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_flash = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_flexflash = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_gauss = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_gcu = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_gelu = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_gemm = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_hadmul = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_hard_sigmoid = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_hard_swish = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_leaky_relu = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_linear = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_lisht = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_log = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_maeloss = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_matmul = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_mish = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_mseloss = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_neg = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_optim = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_parcon = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_pow = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_reci = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_relu = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_reluflash = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_relumask = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_rowmax = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_rowsum = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_sigflash = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_sigmoid = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_sign = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_silu = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_sin = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_sinh = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_softmax = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_softplus = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_sqrt = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_square = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_sub = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_sum = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_swiglu = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_tanh = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_add = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_clip = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_cos = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_cosh = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_div = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_exp = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_gauss = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_gcu = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_gelu = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_gemm = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_hadmul = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_hard_sigmoid = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_hard_swish = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_leaky_relu = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_linear = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_lisht = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_log = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_maeloss = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_matmul = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_mish = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_mseloss = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_neg = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_parcon = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_pow = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_reci = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_relu = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_rowmax = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_rowsum = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_sigmoid = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_silu = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_sin = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_sinh = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_sofba = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_softmax = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_sqrt = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_square = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_sub = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_sum = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_tanh = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_zero = R"doc()doc";

static const char *__doc_ag_kernels_cpu = R"doc()doc";

static const char *__doc_ag_kernels_cuda = R"doc()doc";

static const char *__doc_ag_kernels_load_cpu_plugin = R"doc()doc";

static const char *__doc_ag_kernels_load_cuda_plugin = R"doc()doc";

#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif

