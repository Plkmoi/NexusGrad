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

static const char *__doc_ag_CheckpointOptions_detach_inputs = R"doc()doc";

static const char *__doc_ag_CheckpointOptions_force = R"doc()doc";

static const char *__doc_ag_CheckpointOptions_max_recompute_depth = R"doc()doc";

static const char *__doc_ag_CheckpointOptions_save_inputs = R"doc()doc";

static const char *__doc_ag_CheckpointOptions_save_rng = R"doc()doc";

static const char *__doc_ag_CheckpointOptions_verbose = R"doc()doc";

static const char *__doc_ag_auto_checkpoint_by_depth = R"doc()doc";

static const char *__doc_ag_auto_checkpoint_by_depth_2 = R"doc()doc";

static const char *__doc_ag_auto_checkpoint_every_n = R"doc()doc";

static const char *__doc_ag_auto_checkpoint_every_n_2 = R"doc()doc";

static const char *__doc_ag_checkpoint_impl_is_checkpointed = R"doc()doc";

static const char *__doc_ag_checkpoint_impl_is_checkpointed_2 = R"doc()doc";

static const char *__doc_ag_checkpoint_impl_mark_node_checkpoint = R"doc()doc";

static const char *__doc_ag_checkpoint_impl_mark_node_checkpoint_2 = R"doc()doc";

static const char *__doc_ag_checkpoint_impl_recompute_subgraph = R"doc()doc";

static const char *__doc_ag_checkpoint_impl_recompute_subgraph_2 = R"doc()doc";

static const char *__doc_ag_unnamed_struct_at_cgadimpl_include_ad_checkpoint_hpp_34_8 = R"doc()doc";

#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif

