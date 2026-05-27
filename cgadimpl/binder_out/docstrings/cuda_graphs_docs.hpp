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


static const char *__doc_ag_CudaGraphRunner = R"doc()doc";

static const char *__doc_ag_CudaGraphRunner_CudaGraphRunner = R"doc()doc";

static const char *__doc_ag_CudaGraphRunner_CudaGraphRunner_2 = R"doc()doc";

static const char *__doc_ag_CudaGraphRunner_begin_capture =
R"doc(Puts the framework into capture mode on a private stream. All
subsequent CUDA operations will be recorded into the graph.)doc";

static const char *__doc_ag_CudaGraphRunner_end_capture =
R"doc(Stops capturing and instantiates the graph for execution. Resets the
framework's current stream back to the default.)doc";

static const char *__doc_ag_CudaGraphRunner_graph = R"doc()doc";

static const char *__doc_ag_CudaGraphRunner_instance = R"doc()doc";

static const char *__doc_ag_CudaGraphRunner_is_capturing = R"doc()doc";

static const char *__doc_ag_CudaGraphRunner_operator_assign = R"doc()doc";

static const char *__doc_ag_CudaGraphRunner_replay =
R"doc(Launches the entire captured graph with a single call. This is the
high-performance replay function.

Returns:
    True if replay was successful, false otherwise.)doc";

static const char *__doc_ag_CudaGraphRunner_stream = R"doc()doc";

#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif

