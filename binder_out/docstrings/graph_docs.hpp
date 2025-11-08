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


static const char *__doc_ag_Node = R"doc()doc";

static const char *__doc_ag_Node_2 = R"doc()doc";

static const char *__doc_ag_Node_Node = R"doc()doc";

static const char *__doc_ag_Node_Node_2 = R"doc()doc";

static const char *__doc_ag_Node_debug_name = R"doc()doc";

static const char *__doc_ag_Node_grad = R"doc()doc";

static const char *__doc_ag_Node_has_saved_rng = R"doc()doc";

static const char *__doc_ag_Node_inputs = R"doc()doc";

static const char *__doc_ag_Node_is_checkpoint = R"doc()doc";

static const char *__doc_ag_Node_op = R"doc()doc";

static const char *__doc_ag_Node_requires_grad = R"doc()doc";

static const char *__doc_ag_Node_requires_grad_flag = R"doc()doc";

static const char *__doc_ag_Node_saved_inputs = R"doc()doc";

static const char *__doc_ag_Node_saved_rng_blob = R"doc()doc";

static const char *__doc_ag_Node_shape = R"doc()doc";

static const char *__doc_ag_Node_tape = R"doc()doc";

static const char *__doc_ag_Node_value = R"doc()doc";

static const char *__doc_ag_Value = R"doc()doc";

static const char *__doc_ag_Value_Value = R"doc()doc";

static const char *__doc_ag_Value_Value_2 = R"doc()doc";

static const char *__doc_ag_Value_grad = R"doc()doc";

static const char *__doc_ag_Value_grad_2 = R"doc()doc";

static const char *__doc_ag_Value_node = R"doc()doc";

static const char *__doc_ag_Value_shape = R"doc()doc";

static const char *__doc_ag_Value_shape_2d = R"doc()doc";

static const char *__doc_ag_Value_val = R"doc()doc";

static const char *__doc_ag_Value_val_2 = R"doc()doc";

static const char *__doc_ag_jit_CompileOptions = R"doc()doc";

static const char *__doc_ag_jit_CompileOptions_use_cuda_graph = R"doc()doc";

static const char *__doc_ag_jit_Compiled = R"doc()doc";

static const char *__doc_ag_jit_Compiled_Impl = R"doc()doc";

static const char *__doc_ag_jit_Compiled_p = R"doc()doc";

static const char *__doc_ag_jit_Compiled_run = R"doc()doc";

static const char *__doc_ag_jit_compile = R"doc()doc";

static const char *__doc_ag_make_tensor = R"doc()doc";

static const char *__doc_ag_topo_from = R"doc()doc";

#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif

