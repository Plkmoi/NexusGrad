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


static const char *__doc_ag_CheckpointOptions = R"doc()doc";

static const char *__doc_ag_CheckpointOptions_2 = R"doc()doc";

static const char *__doc_ag_add = R"doc()doc";

static const char *__doc_ag_add_2 = R"doc()doc";

static const char *__doc_ag_alibiatt = R"doc()doc";

static const char *__doc_ag_alibiatt_2 = R"doc()doc";

static const char *__doc_ag_attention = R"doc()doc";

static const char *__doc_ag_attention_2 = R"doc()doc";

static const char *__doc_ag_checkpoint = R"doc()doc";

static const char *__doc_ag_checkpoint_2 = R"doc()doc";

static const char *__doc_ag_cross_entropy_with_logits = R"doc()doc";

static const char *__doc_ag_cross_entropy_with_logits_2 = R"doc()doc";

static const char *__doc_ag_div = R"doc()doc";

static const char *__doc_ag_div_2 = R"doc()doc";

static const char *__doc_ag_dyntanh = R"doc()doc";

static const char *__doc_ag_dyntanh_2 = R"doc()doc";

static const char *__doc_ag_exp = R"doc()doc";

static const char *__doc_ag_exp_2 = R"doc()doc";

static const char *__doc_ag_floadd = R"doc()doc";

static const char *__doc_ag_floadd_2 = R"doc()doc";

static const char *__doc_ag_flodiv = R"doc()doc";

static const char *__doc_ag_flodiv_2 = R"doc()doc";

static const char *__doc_ag_flomul = R"doc()doc";

static const char *__doc_ag_flomul_2 = R"doc()doc";

static const char *__doc_ag_fmab = R"doc()doc";

static const char *__doc_ag_fmab_2 = R"doc()doc";

static const char *__doc_ag_forward_eval_node = R"doc()doc";

static const char *__doc_ag_forward_eval_node_2 = R"doc()doc";

static const char *__doc_ag_gaus = R"doc()doc";

static const char *__doc_ag_gaus_2 = R"doc()doc";

static const char *__doc_ag_gcu = R"doc()doc";

static const char *__doc_ag_gcu_2 = R"doc()doc";

static const char *__doc_ag_gelu = R"doc()doc";

static const char *__doc_ag_gelu_2 = R"doc()doc";

static const char *__doc_ag_inplace_checkpoint = R"doc()doc";

static const char *__doc_ag_inplace_checkpoint_2 = R"doc()doc";

static const char *__doc_ag_kldivergence = R"doc()doc";

static const char *__doc_ag_kldivergence_2 = R"doc()doc";

static const char *__doc_ag_laynor = R"doc()doc";

static const char *__doc_ag_laynor_2 = R"doc()doc";

static const char *__doc_ag_leaky_relu = R"doc()doc";

static const char *__doc_ag_leaky_relu_2 = R"doc()doc";

static const char *__doc_ag_linear = R"doc()doc";

static const char *__doc_ag_linear_2 = R"doc()doc";

static const char *__doc_ag_lisht = R"doc()doc";

static const char *__doc_ag_lisht_2 = R"doc()doc";

static const char *__doc_ag_log = R"doc()doc";

static const char *__doc_ag_log_2 = R"doc()doc";

static const char *__doc_ag_logsumexp_row = R"doc()doc";

static const char *__doc_ag_logsumexp_row_2 = R"doc()doc";

static const char *__doc_ag_mae_loss = R"doc()doc";

static const char *__doc_ag_mae_loss_2 = R"doc()doc";

static const char *__doc_ag_mambassm = R"doc()doc";

static const char *__doc_ag_mambassm_2 = R"doc()doc";

static const char *__doc_ag_matmul = R"doc()doc";

static const char *__doc_ag_matmul_2 = R"doc()doc";

static const char *__doc_ag_mean_all = R"doc()doc";

static const char *__doc_ag_mean_all_2 = R"doc()doc";

static const char *__doc_ag_mish = R"doc()doc";

static const char *__doc_ag_mish_2 = R"doc()doc";

static const char *__doc_ag_moewe = R"doc()doc";

static const char *__doc_ag_moewe_2 = R"doc()doc";

static const char *__doc_ag_mse_loss = R"doc()doc";

static const char *__doc_ag_mse_loss_2 = R"doc()doc";

static const char *__doc_ag_mul = R"doc()doc";

static const char *__doc_ag_mul_2 = R"doc()doc";

static const char *__doc_ag_operator_add = R"doc()doc";

static const char *__doc_ag_operator_add_2 = R"doc()doc";

static const char *__doc_ag_operator_add_3 = R"doc()doc";

static const char *__doc_ag_operator_add_4 = R"doc()doc";

static const char *__doc_ag_operator_add_5 = R"doc()doc";

static const char *__doc_ag_operator_add_6 = R"doc()doc";

static const char *__doc_ag_operator_div = R"doc()doc";

static const char *__doc_ag_operator_div_2 = R"doc()doc";

static const char *__doc_ag_operator_div_3 = R"doc()doc";

static const char *__doc_ag_operator_div_4 = R"doc()doc";

static const char *__doc_ag_operator_mul = R"doc()doc";

static const char *__doc_ag_operator_mul_2 = R"doc()doc";

static const char *__doc_ag_operator_mul_3 = R"doc()doc";

static const char *__doc_ag_operator_mul_4 = R"doc()doc";

static const char *__doc_ag_operator_mul_5 = R"doc()doc";

static const char *__doc_ag_operator_mul_6 = R"doc()doc";

static const char *__doc_ag_operator_sub = R"doc()doc";

static const char *__doc_ag_operator_sub_2 = R"doc()doc";

static const char *__doc_ag_parcon = R"doc()doc";

static const char *__doc_ag_parcon_2 = R"doc()doc";

static const char *__doc_ag_realrms = R"doc()doc";

static const char *__doc_ag_realrms_2 = R"doc()doc";

static const char *__doc_ag_relaynor = R"doc()doc";

static const char *__doc_ag_relaynor_2 = R"doc()doc";

static const char *__doc_ag_relu = R"doc()doc";

static const char *__doc_ag_relu_2 = R"doc()doc";

static const char *__doc_ag_reluatt = R"doc()doc";

static const char *__doc_ag_reluatt_2 = R"doc()doc";

static const char *__doc_ag_rms = R"doc()doc";

static const char *__doc_ag_rms_2 = R"doc()doc";

static const char *__doc_ag_rowmax = R"doc()doc";

static const char *__doc_ag_rowmax_2 = R"doc()doc";

static const char *__doc_ag_rowsum = R"doc()doc";

static const char *__doc_ag_rowsum_2 = R"doc()doc";

static const char *__doc_ag_sigatt = R"doc()doc";

static const char *__doc_ag_sigatt_2 = R"doc()doc";

static const char *__doc_ag_sigmoid = R"doc()doc";

static const char *__doc_ag_sigmoid_2 = R"doc()doc";

static const char *__doc_ag_sign = R"doc()doc";

static const char *__doc_ag_sign_2 = R"doc()doc";

static const char *__doc_ag_silu = R"doc()doc";

static const char *__doc_ag_silu_2 = R"doc()doc";

static const char *__doc_ag_softmax_row = R"doc()doc";

static const char *__doc_ag_softmax_row_2 = R"doc()doc";

static const char *__doc_ag_softplus = R"doc()doc";

static const char *__doc_ag_softplus_2 = R"doc()doc";

static const char *__doc_ag_sub = R"doc()doc";

static const char *__doc_ag_sub_2 = R"doc()doc";

static const char *__doc_ag_sum = R"doc()doc";

static const char *__doc_ag_sum_2 = R"doc()doc";

static const char *__doc_ag_swiglu = R"doc()doc";

static const char *__doc_ag_swiglu_2 = R"doc()doc";

static const char *__doc_ag_tanh = R"doc()doc";

static const char *__doc_ag_tanh_2 = R"doc()doc";

static const char *__doc_ag_transpose = R"doc()doc";

static const char *__doc_ag_transpose_2 = R"doc()doc";

#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif

