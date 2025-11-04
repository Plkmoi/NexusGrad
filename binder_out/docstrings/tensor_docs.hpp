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


static const char *__doc_ag_Device = R"doc()doc";

static const char *__doc_ag_Device_CPU = R"doc()doc";

static const char *__doc_ag_Device_CUDA = R"doc()doc";

static const char *__doc_ag_Tensor = R"doc()doc";

static const char *__doc_ag_Tensor_Tensor = R"doc()doc";

static const char *__doc_ag_Tensor_Tensor_2 = R"doc()doc";

static const char *__doc_ag_Tensor_abs = R"doc()doc";

static const char *__doc_ag_Tensor_add = R"doc()doc";

static const char *__doc_ag_Tensor_alibi = R"doc()doc";

static const char *__doc_ag_Tensor_c = R"doc()doc";

static const char *__doc_ag_Tensor_cols = R"doc()doc";

static const char *__doc_ag_Tensor_cos = R"doc()doc";

static const char *__doc_ag_Tensor_cosh = R"doc()doc";

static const char *__doc_ag_Tensor_data = R"doc()doc";

static const char *__doc_ag_Tensor_data_2 = R"doc()doc";

static const char *__doc_ag_Tensor_data_ptr = R"doc()doc";

static const char *__doc_ag_Tensor_dev = R"doc()doc";

static const char *__doc_ag_Tensor_device = R"doc()doc";

static const char *__doc_ag_Tensor_exp = R"doc()doc";

static const char *__doc_ag_Tensor_floten = R"doc()doc";

static const char *__doc_ag_Tensor_gelu_tanh = R"doc()doc";

static const char *__doc_ag_Tensor_is_cpu = R"doc()doc";

static const char *__doc_ag_Tensor_is_cuda = R"doc()doc";

static const char *__doc_ag_Tensor_leaky_relu = R"doc()doc";

static const char *__doc_ag_Tensor_log = R"doc()doc";

static const char *__doc_ag_Tensor_logsumexp_row = R"doc()doc";

static const char *__doc_ag_Tensor_mae_loss = R"doc()doc";

static const char *__doc_ag_Tensor_matmul = R"doc()doc";

static const char *__doc_ag_Tensor_mean_all = R"doc()doc";

static const char *__doc_ag_Tensor_mse_loss = R"doc()doc";

static const char *__doc_ag_Tensor_numel = R"doc()doc";

static const char *__doc_ag_Tensor_ones = R"doc()doc";

static const char *__doc_ag_Tensor_ones_like = R"doc()doc";

static const char *__doc_ag_Tensor_operator_call = R"doc()doc";

static const char *__doc_ag_Tensor_operator_call_2 = R"doc()doc";

static const char *__doc_ag_Tensor_r = R"doc()doc";

static const char *__doc_ag_Tensor_randn = R"doc()doc";

static const char *__doc_ag_Tensor_reciprocal = R"doc()doc";

static const char *__doc_ag_Tensor_reduce_to = R"doc()doc";

static const char *__doc_ag_Tensor_relu = R"doc()doc";

static const char *__doc_ag_Tensor_relu_mask = R"doc()doc";

static const char *__doc_ag_Tensor_row_max = R"doc()doc";

static const char *__doc_ag_Tensor_row_sum = R"doc()doc";

static const char *__doc_ag_Tensor_rows = R"doc()doc";

static const char *__doc_ag_Tensor_rt = R"doc()doc";

static const char *__doc_ag_Tensor_sech = R"doc()doc";

static const char *__doc_ag_Tensor_shape = R"doc()doc";

static const char *__doc_ag_Tensor_sigmoid = R"doc()doc";

static const char *__doc_ag_Tensor_sign = R"doc()doc";

static const char *__doc_ag_Tensor_silu = R"doc()doc";

static const char *__doc_ag_Tensor_sin = R"doc()doc";

static const char *__doc_ag_Tensor_sinh = R"doc()doc";

static const char *__doc_ag_Tensor_size = R"doc()doc";

static const char *__doc_ag_Tensor_softmax_row = R"doc()doc";

static const char *__doc_ag_Tensor_softplus = R"doc()doc";

static const char *__doc_ag_Tensor_sqrt = R"doc()doc";

static const char *__doc_ag_Tensor_sum_all = R"doc()doc";

static const char *__doc_ag_Tensor_sum_scalar = R"doc()doc";

static const char *__doc_ag_Tensor_tanh = R"doc()doc";

static const char *__doc_ag_Tensor_to = R"doc()doc";

static const char *__doc_ag_Tensor_transpose = R"doc()doc";

static const char *__doc_ag_Tensor_vales = R"doc()doc";

static const char *__doc_ag_Tensor_vals_like = R"doc()doc";

static const char *__doc_ag_Tensor_zeros = R"doc()doc";

static const char *__doc_ag_Tensor_zeros_like = R"doc()doc";

#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif

