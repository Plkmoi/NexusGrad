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

static const char *__doc_ag_kernels_Cpu_add = R"doc()doc";

static const char *__doc_ag_kernels_Cpu_div = R"doc()doc";

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

static const char *__doc_ag_kernels_Cpu_mul = R"doc()doc";

static const char *__doc_ag_kernels_Cpu_pow = R"doc()doc";

static const char *__doc_ag_kernels_Cpu_relu = R"doc()doc";

static const char *__doc_ag_kernels_Cpu_relu_bwd = R"doc()doc";

static const char *__doc_ag_kernels_Cpu_sigmoid = R"doc()doc";

static const char *__doc_ag_kernels_Cpu_sigmoid_bwd_from_s = R"doc()doc";

static const char *__doc_ag_kernels_Cpu_softplus = R"doc()doc";

static const char *__doc_ag_kernels_Cpu_softplus_bwd = R"doc()doc";

static const char *__doc_ag_kernels_Cpu_sqrt = R"doc()doc";

static const char *__doc_ag_kernels_Cpu_sqrt_bwd_from_y = R"doc()doc";

static const char *__doc_ag_kernels_Cpu_sub = R"doc()doc";

static const char *__doc_ag_kernels_Cpu_tanh = R"doc()doc";

static const char *__doc_ag_kernels_Cpu_tanh_bwd_from_t = R"doc()doc";

static const char *__doc_ag_kernels_Cuda = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_add = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_div = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_exp = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_matmul = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_mul = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_relu = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_sub = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_add = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_matmul = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_vjp_relu = R"doc()doc";

static const char *__doc_ag_kernels_Cuda_zero = R"doc()doc";

static const char *__doc_ag_kernels_cpu = R"doc()doc";

static const char *__doc_ag_kernels_cuda = R"doc()doc";

static const char *__doc_ag_kernels_load_cpu_plugin = R"doc()doc";

static const char *__doc_ag_kernels_load_cuda_plugin = R"doc()doc";

#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif

